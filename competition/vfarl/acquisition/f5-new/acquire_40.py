"""F5-new acquisition: save original rows, resolve commits, acquire sources.

Mirrors s600-common-root/collect.py prepare() but for F5 40.
Outputs to acquisition/f5-new/ and replay/artifacts/f5-new/.
Uses replay/.venv (prepare_sfr.acquire).
"""
import hashlib, json, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REPLAY = ROOT / "competition/vfarl/replay"
ART = REPLAY / "artifacts/f5-new"
REV = "77cab9055d42ab4a5c25c89a8f937096db13558e"
sys.path.insert(0, str(REPLAY))
from prepare_sfr import acquire

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def label(p): return str(Path(p).relative_to(ROOT))
def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")

def canonical_of(row):
    # swesmith/org__repo.sha -> org/repo
    repo = row["repo"]
    assert repo.startswith("swesmith/")
    suffix = repo.split("/")[-1]  # org__repo.sha
    base = suffix.rsplit(".", 1)[0]  # org__repo
    assert "__" in base, base
    return base.replace("__", "/")

def snapshot_of(iid):
    # org__repo.sha.mutation -> sha
    parts = iid.split(".")
    assert len(parts) >= 3, iid
    return parts[1]

def load_rows():
    rows = {}
    for p in (ROOT / "competition/vfarl/acquisition").glob("f5-new/pages/train-*.json"):
        if "receipt" in str(p):
            continue
        d = json.loads(p.read_text())
        for e in d.get("rows", []):
            rows[e["row"]["instance_id"]] = (e.get("row_idx"), e["row"], str(p.relative_to(ROOT)))
    for pat in ["expand-resource/mido__mido-train-*.json",
                "expand-resource/pudo__dataset-train-*.json",
                "expand-resource/pexpect__ptyprocess.1067dbda-train.json",
                "expand-resource/aio-libs__async-timeout.d0baa9f1-train.json",
                "s600-specialist-a/artifacts/pages/click-*.json"]:
        for p in (ROOT / "competition/vfarl/acquisition").glob(pat):
            if "receipt" in str(p):
                continue
            d = json.loads(p.read_text())
            for e in d.get("rows", []):
                rows[e["row"]["instance_id"]] = (e.get("row_idx"), e["row"], str(p.relative_to(ROOT)))
    # fallback from .task.json
    for f in (ROOT / "competition/vfarl/acquisition").rglob("*.task.json"):
        try:
            d = json.loads(f.read_text())
            r = d.get("row")
            if r and r["instance_id"] not in rows:
                rows[r["instance_id"]] = (d.get("row_idx", -1), r, str(f.relative_to(ROOT)))
        except Exception:
            pass
    return rows

def acquire_one(item):
    key, iid = item
    rows = load_rows()
    assert iid in rows, iid
    idx, row, src = rows[iid]
    # validate train
    assert row.get("problem_statement", "").strip(), iid
    assert row.get("FAIL_TO_PASS") and row.get("PASS_TO_PASS"), iid
    # save original
    task_path = HERE / (iid + ".task.json")
    if not task_path.exists():
        save(task_path, {"source_dataset": "SWE-bench/SWE-smith-py", "revision": REV,
                         "split": "train", "row_idx": idx, "row": row, "source_page": src})
    # resolve execution commit
    refs_path = HERE / "artifacts/refs" / (row["repo"].replace("/", "__") + ".txt")
    # use per-repo refs cache to avoid 40 ls-remote storms; fetch once per repo
    return (key, iid, row, idx, src)

def resolve_repo_commits(pairs):
    # pairs: list of (key,iid,row)
    by_repo = {}
    for key, iid, row, idx, src in pairs:
        by_repo.setdefault(row["repo"], []).append((key, iid, row, idx, src))
    out = []
    for repo, items in by_repo.items():
        refs_path = HERE / "artifacts/refs" / (repo.replace("/", "__") + ".txt")
        if not refs_path.exists():
            print(f"ls-remote {repo}", flush=True)
            r = subprocess.run(["git", "ls-remote", f"https://github.com/{repo}.git"],
                               capture_output=True, text=True, check=True)
            refs_path.parent.mkdir(parents=True, exist_ok=True)
            refs_path.write_text(r.stdout)
        refs = dict((f[1], f[0]) for f in (l.split() for l in refs_path.read_text().splitlines()) if len(f) == 2)
        for key, iid, row, idx, src in items:
            ref = f"refs/heads/{iid}"
            assert ref in refs, f"{iid} missing ref in {repo}"
            commit = refs[ref]
            assert len(commit) == 40, commit
            out.append((key, iid, row, idx, src, commit, str(refs_path.relative_to(ROOT))))
    return out

def main():
    sel = json.loads((HERE / "selected-40.json").read_text())["candidates"]
    # assign keys f5-new-<slug>
    import re
    pairs = []
    for c in sel:
        iid = c["instance_id"]
        # slug: repo-short + mutation suffix
        slug = re.sub(r"[^a-z0-9]+", "-", iid.lower())[:60].strip("-")
        key = f"f5-new-{slug}"
        pairs.append((key, iid))
    # load rows
    rows = load_rows()
    full = []
    for key, iid in pairs:
        assert iid in rows, iid
        idx, row, src = rows[iid]
        full.append((key, iid, row, idx, src))
    resolved = resolve_repo_commits(full)
    # acquire upstreams first, one per canonical repo, single-threaded to avoid
    # concurrent writes to the same shared upstream directory.
    uniq = {}
    for key, iid, row, idx, src, commit, refs_rel in resolved:
        canon = canonical_of(row)
        snap = snapshot_of(iid)
        uniq.setdefault((canon, snap), (canon, snap))
    ups = {}
    for (canon, snap) in sorted(uniq):
        upstream_root = ART / (canon.replace("/", "__") + "-upstream")
        print(f"acquire upstream {canon} {snap}", flush=True)
        ups[(canon, snap)] = acquire(canon, snap, upstream_root)
    # acquire buggy sources (parallel 4, distinct dirs so safe)
    def do_one(entry):
        key, iid, row, idx, src, commit, refs_rel = entry
        canon = canonical_of(row)
        snap = snapshot_of(iid)
        source = ART / (key + "-source")
        buggy = acquire(row["repo"], commit, source)
        up = ups[(canon, snap)]
        upstream_root = ART / (canon.replace("/", "__") + "-upstream")
        return {"key": key, "instance_id": iid, "row_idx": idx,
                "source_page": src, "execution_commit": commit,
                "refs": refs_rel, "canonical_repo": canon, "snapshot": snap,
                "buggy": buggy, "upstream": up,
                "upstream_dir": str(upstream_root.relative_to(ROOT)),
                "workspace": str(source.relative_to(ROOT)),
                "F2P": len(row["FAIL_TO_PASS"]), "P2P": len(row["PASS_TO_PASS"])}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(do_one, resolved))
    save(HERE / "acquired-40.json", {"date": "2026-10-25", "revision": REV, "tasks": results})
    print(f"acquired={len(results)}")
    for r in results:
        print(r["key"], r["instance_id"], r["execution_commit"][:8], r["canonical_repo"], r["snapshot"])

if __name__ == "__main__":
    main()
