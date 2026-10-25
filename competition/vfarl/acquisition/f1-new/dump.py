"""Dump full row (patch, F2P/P2P, statement) for manual F1 review."""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGES = HERE / "pages"

rows = {}
for p in PAGES.glob("train-*.json"):
    if p.name.endswith(".receipt.json"):
        continue
    blob = json.loads(p.read_text())
    for e in blob.get("rows", []):
        rows[e["row"]["instance_id"]] = (e["row_idx"], e["row"])

for iid in sys.argv[1:]:
    if iid not in rows:
        print("MISSING", iid)
        continue
    idx, r = rows[iid]
    f2p = r["FAIL_TO_PASS"]
    p2p = r["PASS_TO_PASS"]
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    print("=" * 100)
    print("IID:", iid, "row_idx:", idx, "repo:", r["repo"])
    print("F2P:", len(f2p), "P2P:", len(p2p))
    print("F2P_NODES:", json.dumps(f2p, ensure_ascii=False))
    print("STMT:", (r.get("problem_statement") or "")[:600])
    print("PATCH:")
    print(r.get("patch", "")[:6000])
