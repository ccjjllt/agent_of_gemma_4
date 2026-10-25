"""Acquire f5-fresh single task: resolve execution commit, acquire sources."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
REPLAY = ROOT / "competition/vfarl/replay"
ART = REPLAY / "artifacts/f5-fresh"
REV = "77cab9055d42ab4a5c25c89a8f937096db13558e"
sys.path.insert(0, str(REPLAY))
from prepare_sfr import acquire


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def label(p):
    return str(Path(p).relative_to(ROOT))


def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")


def canonical_of(row):
    repo = row["repo"]
    assert repo.startswith("swesmith/")
    suffix = repo.split("/")[-1].rsplit(".", 1)[0]
    assert "__" in suffix, suffix
    return suffix.replace("__", "/")


def snapshot_of(iid):
    return iid.split(".")[1]


def main():
    contracts = [c for c in json.loads((HERE / "selected-contracts.json").read_text())["tasks"]
                 if c.get("pool", "smith") == "smith"]
    # load rows from f5-fresh task.json files
    results = []
    # group by repo for refs (single repo here, but generic)
    by_repo = {}
    for c in contracts:
        iid = c["instance_id"]
        task_path = HERE / (iid + ".task.json")
        doc = json.loads(task_path.read_text())
        row = doc["row"]
        idx = doc.get("row_idx", -1)
        src = doc.get("source_page", str(task_path.relative_to(ROOT)))
        by_repo.setdefault(row["repo"], []).append((c, iid, row, idx, src))
    for repo, items in by_repo.items():
        refs_path = HERE / "artifacts-refs" / (repo.replace("/", "__") + ".txt")
        if not refs_path.exists():
            print(f"ls-remote {repo}", flush=True)
            r = subprocess.run(["git", "ls-remote", f"https://github.com/{repo}.git"],
                               capture_output=True, text=True, check=True)
            refs_path.parent.mkdir(parents=True, exist_ok=True)
            refs_path.write_text(r.stdout)
        refs = dict((f[1], f[0]) for f in (l.split() for l in refs_path.read_text().splitlines()) if len(f) == 2)
        for c, iid, row, idx, src in items:
            ref = f"refs/heads/{iid}"
            assert ref in refs, f"{iid} missing ref in {repo}"
            commit = refs[ref]
            assert len(commit) == 40, commit
            key = c["key"]
            canon = canonical_of(row)
            snap = snapshot_of(iid)
            # upstream first (shared per canon/snap)
            upstream_root = ART / (canon.replace("/", "__") + "-upstream")
            print(f"acquire upstream {canon} {c['snapshot']}", flush=True)
            up = acquire(canon, c["snapshot"], upstream_root)
            source = ART / (key + "-source")
            print(f"acquire buggy {repo} {commit[:8]}", flush=True)
            buggy = acquire(row["repo"], commit, source)
            results.append({
                "key": key, "instance_id": iid, "row_idx": idx, "source_page": src,
                "execution_commit": commit,
                "refs": str(refs_path.relative_to(ROOT)),
                "canonical_repo": canon, "snapshot": snap,
                "buggy": buggy, "upstream": up,
                "upstream_dir": str(upstream_root.relative_to(ROOT)),
                "workspace": str(source.relative_to(ROOT)),
                "F2P": len(row["FAIL_TO_PASS"]), "P2P": len(row["PASS_TO_PASS"]),
            })
    save(HERE / "acquired-fresh.json", {"date": "2026-10-25", "revision": REV,
        # Preserve non-Smith entries (e.g. Gym/V2 recorded by their own flow).
        "tasks": [t for t in (json.loads((HERE / "acquired-fresh.json").read_text())["tasks"]
                              if (HERE / "acquired-fresh.json").exists() else [])
                  if t.get("key") not in {r["key"] for r in results}] + results})
    print(f"acquired={len(results)}")
    for r in results:
        print(r["key"], r["instance_id"], r["execution_commit"][:8])


if __name__ == "__main__":
    main()
