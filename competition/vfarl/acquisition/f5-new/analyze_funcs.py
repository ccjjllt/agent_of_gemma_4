"""Analyze F5 40 buggy vs upstream to find single production-function deltas."""
import ast, difflib, hashlib, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).resolve().parent
REPLAY = ROOT / "competition/vfarl/replay"
ART = REPLAY / "artifacts/f5-new"

acq = json.loads((HERE / "acquired-40.json").read_text())["tasks"]
# load rows
rows = {}
for p in (ROOT / "competition/vfarl/acquisition").glob("f5-new/pages/train-*.json"):
    if "receipt" in str(p):
        continue
    d = json.loads(p.read_text())
    for e in d.get("rows", []):
        rows[e["row"]["instance_id"]] = e["row"]
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
            rows[e["row"]["instance_id"]] = e["row"]
for f in (ROOT / "competition/vfarl/acquisition").rglob("*.task.json"):
    try:
        d = json.loads(f.read_text())
        r = d.get("row")
        if r and r["instance_id"] not in rows:
            rows[r["instance_id"]] = r
    except Exception:
        pass

def func_at(path, lineno):
    text = path.read_text()
    tree = ast.parse(text)
    # find deepest function/class containing lineno
    best = None
    def visit(node, stack):
        nonlocal best
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            if node.lineno <= lineno <= node.end_lineno:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    best = (list(stack) + [node], node.lineno, node.end_lineno)
                for child in ast.iter_child_nodes(node):
                    visit(child, (list(stack) + [node]) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else stack)
    visit(tree, [])
    return best

out = []
for t in acq:
    key, iid = t["key"], t["instance_id"]
    row = rows[iid]
    patch = row["patch"]
    # parse changed files from patch
    import re
    files = re.findall(r"diff --git a/(\S+)", patch)
    # compare buggy vs upstream for those files
    src = ROOT / t["workspace"]
    ups = ROOT / t["upstream_dir"]
    diffs = []
    for f in files:
        b = (src / f)
        u = (ups / f)
        if not b.exists() or not u.exists():
            diffs.append({"file": f, "missing": True})
            continue
        bt, ut = b.read_text(), u.read_text()
        if bt == ut:
            diffs.append({"file": f, "identical": True})
            continue
        # unified diff
        sm = difflib.unified_diff(bt.splitlines(), ut.splitlines(), lineterm="")
        # find first changed line in buggy
        blines = bt.splitlines()
        ulines = ut.splitlines()
        import difflib as dl
        m = list(dl.SequenceMatcher(None, blines, ulines).get_opcodes())
        # find first non-equal
        first = next((x for x in m if x[0] != "equal"), None)
        diffs.append({"file": f, "opcodes": m[:5], "first_change": first,
                      "buggy_lines": len(blines), "upstream_lines": len(ulines)})
    # try to find enclosing function for first change in buggy
    sym = None
    rel = files[0] if files else None
    if rel and (src / rel).exists():
        # get first changed lineno from opcodes
        for d in diffs:
            if "first_change" in d and d["first_change"]:
                tag, i1, i2, j1, j2 = d["first_change"]
                lineno = i1 + 1
                try:
                    best = func_at(src / rel, lineno)
                    if best:
                        stack, s, e = best
                        # qualified name
                        q = ".".join(n.name for n in stack)
                        sym = {"qualified": q, "start": s, "end": e, "lineno": lineno}
                except Exception as ex:
                    sym = {"error": repr(ex)}
                break
    out.append({"key": key, "instance_id": iid, "files": files,
                "patch_len": len(patch), "hunks": patch.count("@@"),
                "diffs": [{"file": d.get("file"), "first_change": d.get("first_change"),
                           "buggy_lines": d.get("buggy_lines"), "upstream_lines": d.get("upstream_lines")} for d in diffs],
                "symbol_guess": sym,
                "F2P": t["F2P"], "P2P": t["P2P"]})

(HERE / "function-analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(f"analyzed={len(out)}")
for o in out:
    print(o["key"], o["files"], o["symbol_guess"])
