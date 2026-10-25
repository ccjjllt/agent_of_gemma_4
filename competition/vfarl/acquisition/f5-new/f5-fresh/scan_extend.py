"""F5-fresh extension scan: identical policy to scan_fresh.py, new pages only.

Reads acquisition/f5-new/f5-fresh/pages-fresh/train-*.json (+ receipts),
applies the exact fresh-strict filters (F2P+P2P<50, single-file, single-hunk,
nonempty prompt, 75 banned + 5 used + f5-fresh used + S600 + 4 repos + used
instances incl. new keeps). Output: fresh-extended-scan.json. Never rewrites
fresh-strict-scan.json. Old dirs untouched.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path("/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4")
VFARL = ROOT / "competition/vfarl"
HERE = VFARL / "acquisition/f5-new/f5-fresh"
sys.path.insert(0, str(HERE))
from scan_fresh import F5_POS, EXCLUDED, load_banned, load_s600_used, load_used_instances, canon_of_smith, short_of_smith, score_patch


def main():
    banned, banned_shorts, used5 = load_banned()
    s600_used = load_s600_used()
    used_inst = load_used_instances()
    # extend used instances with f5-fresh keeps + reviewed (no quota fillers)
    try:
        contracts = json.loads((HERE / "selected-contracts.json").read_text())["tasks"]
        for c in contracts:
            used_inst.add(c["instance_id"])
    except Exception:
        pass
    try:
        rev = json.loads((HERE / "reviewed-candidates.json").read_text())["tasks"]
        for t in rev:
            used_inst.add(t["instance_id"])
    except Exception:
        pass
    # f5-fresh used snapshots: scrapy 35212ec5 + acquired-fresh snapshots
    fresh_used_shorts = {("scrapy/scrapy", "35212ec5")}
    try:
        acq = json.loads((HERE / "acquired-fresh.json").read_text())["tasks"]
        for t in acq:
            fresh_used_shorts.add((t["canonical_repo"].lower(), t["snapshot"].lower()))
    except Exception:
        pass
    print(f"banned={len(banned)} s600={len(s600_used)} used_inst={len(used_inst)}", flush=True)

    rows = {}
    for p in sorted((HERE / "pages-fresh").glob("train-*.json")):
        if "receipt" in str(p):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        entries = d.get("rows", []) if isinstance(d, dict) else []
        for e in entries:
            if not isinstance(e, dict):
                continue
            r = e.get("row", e if "instance_id" in e else None)
            if r and "instance_id" in r and r["instance_id"] not in rows:
                rows[r["instance_id"]] = (e.get("row_idx", -1), r, str(p.relative_to(ROOT)))
    print(f"extension cached rows={len(rows)}", flush=True)

    # also exclude instances already in the base 99 scan
    base = json.loads((HERE / "fresh-strict-scan.json").read_text())
    for h in base["smith_hits"]:
        used_inst.add(h["instance_id"])

    hits = []
    for iid, (idx, row, src) in rows.items():
        if iid in s600_used or iid in used_inst:
            continue
        if not row.get("repo", "").startswith("swesmith/"):
            continue
        try:
            canon = canon_of_smith(row)
        except Exception:
            continue
        if canon.lower() in EXCLUDED:
            continue
        short = short_of_smith(iid)
        if (canon.lower(), short.lower()) in banned_shorts:
            continue
        if (canon.lower(), short.lower()) in {(c.lower(), s.lower()) for c, s in used5}:
            continue
        if (canon.lower(), short.lower()) in fresh_used_shorts:
            continue
        patch = row.get("patch", "")
        f2p = row.get("FAIL_TO_PASS") or []
        p2p = row.get("PASS_TO_PASS") or []
        if isinstance(f2p, str):
            f2p = json.loads(f2p)
        if isinstance(p2p, str):
            p2p = json.loads(p2p)
        if not f2p or not p2p or set(f2p) & set(p2p):
            continue
        if not (row.get("problem_statement") or "").strip():
            continue
        total = len(f2p) + len(p2p)
        if total >= 50:
            continue
        files = re.findall(r"diff --git a/(\S+)", patch)
        if len(files) != 1 or any(".." in f for f in files):
            continue
        hunks = len([l for l in patch.splitlines() if l.startswith("@@ ")])
        if hunks != 1:
            continue
        hits2, rej = score_patch(patch)
        if hits2 is None:
            continue
        hits.append({
            "pool": "smith-py", "instance_id": iid, "row_idx": idx, "repo": row["repo"],
            "canon": canon, "short": short, "hits": hits2[:6], "file": files[0],
            "patch_len": len(patch), "hunks": hunks, "f2p": len(f2p), "p2p": len(p2p),
            "total": total, "F2P_sample": f2p[:2], "source_page": src,
        })
    hits.sort(key=lambda d: (d["total"], d["f2p"]))
    print(f"extension smith small-full<50 single-hunk timing hits={len(hits)}", flush=True)
    for h in hits[:60]:
        print(f'{h["total"]:3d} F2P={h["f2p"]} P2P={h["p2p"]} {h["instance_id"]} {h["canon"]}::{h["short"]} {h["file"]} {h["hits"][:3]}', flush=True)
    out = {"date": "2026-10-25", "revision_smithpy": "77cab9055d42ab4a5c25c89a8f937096db13558e",
           "policy": "IDENTICAL to fresh-strict-scan.json policy; new pages-fresh pages only; f5-fresh used snapshots + new keeps excluded",
           "counts": {"smith": len(hits)}, "smith_hits": hits}
    (HERE / "fresh-extended-scan.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("wrote fresh-extended-scan.json", out["counts"], flush=True)


if __name__ == "__main__":
    main()
