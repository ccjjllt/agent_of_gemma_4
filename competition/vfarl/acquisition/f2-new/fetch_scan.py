"""F2-new batch: fetch pinned train pages and heuristic-scan for F2 copy/ownership candidates.

Reads-only HF datasets-server at pinned Smith-py revision. Saves pages + receipts.
Scans patches for copy/ownership/identity signals. Incidental .copy() without
ownership semantics is down-scored for manual demotion to S/R.
"""
import hashlib, json, re, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PAGES = HERE / "pages"
REV = "77cab9055d42ab4a5c25c89a8f937096db13558e"
BASE = ("https://datasets-server.huggingface.co/rows?dataset=SWE-bench%2FSWE-smith-py"
        "&config=default&split=train&revision=" + REV)

# repo_key -> (start_row, end_row) from smith-task-index.jsonl; all ALLOWED snapshots
# (none of these 49 forbidden common repo_keys; verified 2026-10-25)
TARGETS = {
    "mido__mido.a0158ff9": (29508, 29868),
    "pallets__click.fde47b4b": (32065, 32709),
    "mozilla__bleach.73871d76": (47255, 47737),
    "davidhalter__parso.338a5760": (30522, 31132),
    "pytest-dev__iniconfig.16793ead": (940, 1002),
    "rustedpy__result.0b855e1e": (23102, 23165),
    "r1chardj0n3s__parse.30da9e4f": (40837, 40926),
    "pallets__markupsafe.620c06c9": (30484, 30521),
    "cloudpipe__cloudpickle.6220b0ce": (41756, 41870),
    "agronholm__typeguard.b6a7e438": (1615, 1947),
    "pydantic__pydantic.acb0f10f": (31133, 31808),
    "seperman__deepdiff.ed252022": (18162, 19125),
    "encode__starlette.db5063c2": (27459, 28179),
    "pyupio__safety.7654596b": (8220, 8978),
    "python-trio__trio.cfbbe2c1": (44716, 45493),
    "paramiko__paramiko.23f92003": (9725, 10644),
    "aio-libs__async-timeout.d0baa9f1": (16834, 16852),
    "borntyping__python-colorlog.dfa10f59": (34680, 34707),
    "john-kurkowski__tldextract.3d1bf184": (36730, 36829),
    "marshmallow-code__webargs.dbde72fe": (9154, 9474),
    "martinblech__xmltodict.0952f382": (29869, 29912),
    "pylint-dev__astroid.b114f6b5": (24298, 26067),
    "scanny__python-pptx.278b47b1": (38448, 40836),
    "spulec__freezegun.5f171db0": (49339, 49446),
    "un33k__python-slugify.872b3750": (36573, 36590),
    "cantools__cantools.0c6a7871": (6780, 8219),
    "pygments__pygments.27649ebb": (3228, 4657),
    "oauthlib__oauthlib.1fd52536": (0, 939),
}

F2_STRONG = [
    r"__copy__", r"__deepcopy__", r"copy\.copy", r"copy\.deepcopy",
    r"deepcopy\(", r"model_copy", r"__getstate__", r"__setstate__",
    r"__getnewargs", r"__reduce", r"def copy\(",
]

F2_OWN = [
    r"\bshared\b", r"\bownership\b", r"\bidentity\b", r"\balias\b",
    r"sentinel", r"shallow", r"deep.?copy", r"\bis\b.*\bis not\b|\bis not\b.*\bis\b",
]

# incidental list/dict .copy() without ownership context -> demote, not reject outright
F2_INCIDENTAL = [
    r"decorator_list\.copy\(\)", r"\.copy\(\)\s*$",
]


def fetch_page(offset, length=100):
    import time
    path = PAGES / f"train-{offset}.json"
    url = BASE + f"&offset={offset}&length={length}"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                body = r.read()
            rev = r.headers.get("X-Revision", "")
            blob = json.loads(body)
            path.write_bytes(json.dumps(blob, ensure_ascii=False).encode())
            (PAGES / f"train-{offset}.receipt.json").write_text(json.dumps(
                {"url": url, "offset": offset, "length": length, "split": "train",
                 "revision": REV, "response_X_Revision": rev,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "rows": len(blob.get("rows", []))}, indent=2) + "\n")
            time.sleep(3)
            return blob
        except Exception as e:
            wait = 6 * (attempt + 1)
            print(f"fetch offset={offset} attempt={attempt} error={e!r} sleep={wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"fetch failed offset={offset}")


def score(row):
    patch = row.get("patch", "") or ""
    iid = row["instance_id"]
    f2p = row.get("FAIL_TO_PASS") or []
    p2p = row.get("PASS_TO_PASS") or []
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    if not f2p or not p2p or set(f2p) & set(p2p):
        return None
    if not (row.get("problem_statement") or "").strip():
        return None
    strong_hits = [pat for pat in F2_STRONG if re.search(pat, patch)]
    own_hits = [pat for pat in F2_OWN if re.search(pat, patch, re.I)]
    ps = (row.get("problem_statement") or "").lower()
    ps_hits = [k for k in ["copy", "identical", "identity", "shared", "alias", "ownership", "sentinel", "shallow", "deepcopy", " is "] if k in ps]
    incidental = [pat for pat in F2_INCIDENTAL if re.search(pat, patch)]
    # hard F2 needs copy mechanism + ownership/identity context; incidental .copy() alone demoted
    if not strong_hits and not (own_hits and ("copy" in patch.lower() or "copy" in ps)):
        return {"score": -50, "strong": [], "own": own_hits,
                "f2p": len(f2p), "p2p": len(p2p),
                "reject": "NO_COPY_OWNERSHIP_SIGNAL_DEMOTE_S_OR_R",
                "patch_head": patch[:600], "ps_hits": ps_hits, "incidental": incidental}
    # incidental decorator_list.copy etc with no ownership -> demote S
    if not strong_hits and incidental and not own_hits:
        return {"score": -40, "strong": [], "own": [],
                "f2p": len(f2p), "p2p": len(p2p),
                "reject": "INCIDENTAL_COPY_DEMOTE_S",
                "patch_head": patch[:600], "ps_hits": ps_hits, "incidental": incidental}
    seq_bonus = 2 if re.search(r"\bis\b.*\bis not\b|\bis not\b.*\bis\b", patch) else 0
    return {"score": len(strong_hits) * 4 + len(own_hits) * 2 + len(ps_hits) + seq_bonus,
            "strong": strong_hits, "own": own_hits[:8],
            "f2p": len(f2p), "p2p": len(p2p),
            "reject": None,
            "patch_head": patch[:600], "ps_hits": ps_hits, "incidental": incidental}


def main():
    only = sys.argv[1:] or list(TARGETS)
    PAGES.mkdir(parents=True, exist_ok=True)
    index = {}
    for l in (ROOT / "competition/vfarl/acquisition/general/smith-task-index.jsonl").read_text().splitlines():
        r = json.loads(l)
        index[r["instance_id"]] = r
    man = json.loads((ROOT / "competition/vfarl/s600/course-manifest.json").read_text())
    used = set(man["courses"]["common"]["instance_ids"]) | set(man["courses"]["specialist"]["instance_ids"])
    # also exclude any supplemental/pilot identities (conservative: avoid all 635)
    try:
        sup = json.loads((ROOT / "competition/vfarl/acquisition/supplemental/catalog.json").read_text())
        used |= {t["instance_id"] for t in sup["tasks"]}
    except Exception:
        pass
    try:
        p = json.loads((ROOT / "competition/vfarl/pilot96/catalog.json").read_text())
        used |= {t["instance_id"] for t in p["tasks"]}
    except Exception:
        pass
    excluded = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}
    out = []
    for repo_key in only:
        if repo_key not in TARGETS:
            print(f"unknown repo {repo_key}", flush=True)
            continue
        start, end = TARGETS[repo_key]
        off = start - (start % 100)
        while off <= end:
            blob = fetch_page(off)
            for entry in blob.get("rows", []):
                if not (start <= entry["row_idx"] <= end):
                    continue
                row = entry["row"]
                if row["instance_id"] in used:
                    continue
                try:
                    suffix = row["repo"].split("/")[-1].split(".")[0].replace("__", "/")
                    if suffix.lower() in excluded:
                        continue
                except Exception:
                    pass
                s = score(row)
                if s is None:
                    continue
                out.append({"instance_id": row["instance_id"], "row_idx": entry["row_idx"],
                            "repo": row["repo"], **s})
            off += 100
    out.sort(key=lambda d: -d["score"])
    (HERE / "scan-hits.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"scanned_hits={len(out)}")
    pos = [d for d in out if (d.get("reject") is None)]
    rej = [d for d in out if (d.get("reject") is not None)]
    print(f"f2_signal={len(pos)} demoted_S_or_R={len(rej)}")
    for d in out[:120]:
        print(d["score"], d["instance_id"], f'F2P={d["f2p"]} P2P={d["p2p"]} reject={d.get("reject")} strong={d.get("strong")} own={d.get("own")} ps={d.get("ps_hits")}')
    (HERE / "scan-used.json").write_text(json.dumps({"used_excluded": len(used), "excluded_repos": sorted(excluded), "targets": only}, indent=2) + "\n")


if __name__ == "__main__":
    main()
