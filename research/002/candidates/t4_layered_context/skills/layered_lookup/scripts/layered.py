#!/usr/bin/env python3
"""Layered code-context lookup: outline -> locate -> inspect. Stdlib only.

Usage (run via run_skill_script in the workspace sandbox):
  python3 layered.py outline --path <rel-dir, default .> [--max-files 8] [--max-defs 24]
  python3 layered.py locate --query <q1> --source <provenance1> [--query <q2> --source <provenance2> ...] [--max-hits 8]
  python3 layered.py inspect --path <rel.py> --symbol <Name> [--line N] [--max-chars 4000]

Common: [--root /workspace] [--allow-non-workspace-root]

Confined to --root (default /workspace). Never follows symlinks. Skips hidden
dirs, venv/.venv, node_modules, __pycache__, .git/.hg, files > 512 KiB.
Single stdout JSON object per call, hard cap 4000 chars (valid JSON preserved;
text fields are trimmed first and `truncated`/`is_truncated` is set, never a
mid-string slice of the JSON blob). No network, no reference patches, no
neural/vector model; text + AST only.

Discipline (enforced by SKILL.md, repeated here because it matters):
- outline signatures and locate snippets (<=400 chars, trimmed to <=120 under
  budget pressure) are NOT complete functions. NEVER treat the first 400
  chars as a full body.
- inspect returns `complete:true` ONLY when the full AST node is included.
  Truncated inspect output (`complete:false`) MUST NOT be used as precise-edit
  input. Re-inspect with a narrower target instead.
"""
import argparse
import ast
import json
import os
from pathlib import Path

SKIP_DIRS = {"__pycache__", "venv", ".venv", "node_modules", ".git", ".hg"}
MAX_FILE_BYTES = 512 * 1024
MAX_SNIPPET_CHARS = 400
TRIMMED_SNIPPET_CHARS = 120
OUTPUT_CAP = 4000
MAX_HITS_HARD = 8
MAX_QUERIES_HARD = 4
MAX_FUNCTIONS_HARD = 2


def err(msg):
    return {"status": "error", "error": msg}


def safe_root(root: str, allow_non_ws: bool):
    p = Path(root).resolve()
    ws = Path("/workspace").resolve()
    if p.is_symlink():
        raise ValueError("symlink root rejected")
    if p != ws and not allow_non_ws:
        raise ValueError("non-workspace root rejected (use --allow-non-workspace-root for local diagnosis)")
    if not p.is_dir():
        raise ValueError("root missing")
    return p


def resolve_under(root: Path, rel: str):
    if rel.startswith("/") or rel.startswith("\\") or ".." in Path(rel).parts:
        raise ValueError("path must be repo-relative without ..")
    fp = (root / rel)
    try:
        rp = fp.resolve()
    except OSError:
        raise ValueError("resolve failed")
    try:
        rp.relative_to(root)
    except ValueError:
        raise ValueError("path escapes root")
    return rp


def iter_py_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d not in SKIP_DIRS]
        dirnames[:] = [d for d in dirnames
                       if not Path(dirpath, d).is_symlink()]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            fp = Path(dirpath, fn)
            if fp.is_symlink():
                continue
            try:
                if fp.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield fp


class DefVisitor(ast.NodeVisitor):
    def __init__(self):
        self.stack = []
        self.defs = []

    def _record(self, node, kind):
        name = node.name
        qual = ".".join(self.stack + [name]) if self.stack else name
        end = getattr(node, "end_lineno", node.lineno)
        self.defs.append({"name": name, "qualname": qual, "kind": kind,
                          "line": node.lineno, "end_line": end})

    def visit_ClassDef(self, node):
        self._record(node, "class")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self._record(node, "def")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node):
        self._record(node, "async def")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def parse_defs(text):
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    v = DefVisitor()
    v.visit(tree)
    return v.defs


def signature_of(lines, lineno, width=160):
    if 1 <= lineno <= len(lines):
        sig = lines[lineno - 1].strip()
        if sig:
            return sig[:width]
    return ""


def snippet_around(lines, lineno, width=3):
    lo = max(1, lineno - width)
    hi = lineno + width
    return "".join(lines[lo - 1:hi])[:MAX_SNIPPET_CHARS]


def fit_json(out, trim_fn, cap=OUTPUT_CAP):
    blob = json.dumps(out, ensure_ascii=False)
    if len(blob) <= cap:
        return blob
    out["truncated"] = True
    if "is_truncated" in out:
        out["is_truncated"] = True
    trim_fn(out)
    blob = json.dumps(out, ensure_ascii=False)
    while len(blob) > cap:
        if not trim_fn(out, aggressive=True):
            break
        blob = json.dumps(out, ensure_ascii=False)
    if len(blob) > cap:
        # last resort: keep metadata, drop the largest text payload
        for key in ("definitions", "candidates", "queries", "functions"):
            if key in out and isinstance(out[key], list) and out[key]:
                out[key] = out[key][:1]
                blob = json.dumps(out, ensure_ascii=False)
                if len(blob) <= cap:
                    break
    blob = json.dumps(out, ensure_ascii=False)
    if len(blob) > cap:
        # metadata-only fallback (always valid JSON, always <= cap for our schemas)
        keep = {"status": out.get("status", "ok"), "truncated": True,
                "error": "output budget exceeded; narrow scope and retry"}
        for k in ("command", "path", "symbol"):
            if k in out:
                keep[k] = out[k]
        blob = json.dumps(keep, ensure_ascii=False)[:cap]
    return blob


def cmd_outline(args, root):
    rel = args.path or "."
    try:
        rp = resolve_under(root, rel)
    except ValueError as e:
        return json.dumps(err(str(e)), ensure_ascii=False)
    if rp.is_symlink() or not rp.exists():
        return json.dumps(err("missing or symlink"), ensure_ascii=False)
    if not rp.is_dir():
        return json.dumps(err("outline target must be a directory; use inspect for functions"), ensure_ascii=False)
    max_files = min(max(1, args.max_files), MAX_HITS_HARD)
    max_defs = min(max(1, args.max_defs), 32)

    try:
        subdirs = sorted([p.name for p in rp.iterdir()
                          if p.is_dir() and not p.is_symlink()
                          and not p.name.startswith(".") and p.name not in SKIP_DIRS])[:16]
    except OSError:
        return json.dumps(err("list failed"), ensure_ascii=False)

    py_files = []
    for fp in iter_py_files(rp):
        try:
            relp = fp.relative_to(root).as_posix()
        except ValueError:
            continue
        py_files.append(relp)
        if len(py_files) >= 256:
            break
    py_files.sort(key=lambda s: (len(s), s))
    picked = py_files[:max_files]

    definitions = []
    file_info = []
    for relp in picked:
        fp = root / relp
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            file_info.append({"path": relp, "def_count": 0, "read": "failed"})
            continue
        lines = text.splitlines()
        defs = parse_defs(text)
        file_info.append({"path": relp, "def_count": len(defs)})
        for d in defs:
            sig = signature_of(lines, d["line"]) or d["qualname"]
            definitions.append({"path": relp, "kind": d["kind"], "name": d["name"],
                                "qualname": d["qualname"], "line": d["line"],
                                "end_line": d["end_line"], "signature": sig})

    def rank(d):
        return (len(d["path"]), d["line"])
    definitions.sort(key=rank)
    total_defs = len(definitions)
    definitions = definitions[:max_defs]

    out = {"status": "ok", "command": "outline",
           "path": rp.relative_to(root).as_posix() if rp != root else ".",
           "dirs": subdirs,
           "py_files": file_info,
           "py_files_total_scanned": len(py_files),
           "py_files_shown": len(picked),
           "definitions": definitions,
           "definitions_total": total_defs,
           "definitions_shown": len(definitions),
           "truncated": len(py_files) > len(picked) or total_defs > len(definitions),
           "note": "signatures are NOT function bodies; call inspect for a complete function before reasoning about semantics or editing"}

    def trim(o, aggressive=False):
        changed = False
        lim = 80 if aggressive else TRIMMED_SNIPPET_CHARS
        for d in o.get("definitions", []):
            if len(d.get("signature", "")) > lim:
                d["signature"] = d["signature"][:lim]
                changed = True
        if aggressive and len(o.get("definitions", [])) > 1:
            o["definitions"] = o["definitions"][:max(1, len(o["definitions"]) // 2)]
            o["definitions_shown"] = len(o["definitions"])
            changed = True
        return changed

    return fit_json(out, trim)


def find_hits_for_query(root, query):
    per_file = []
    for fp in iter_py_files(root):
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines(keepends=True)
        try:
            rel = fp.relative_to(root).as_posix()
        except ValueError:
            continue
        cands = []
        for d in parse_defs(text):
            if d["name"] == query:
                kind = d["kind"]
                cands.append({"path": rel, "kind": kind, "symbol": query,
                              "line": d["line"], "end_line": d["end_line"],
                              "snippet": snippet_around(lines, d["line"]),
                              "source": "ast"})
        file_text_hits = 0
        for i, ln in enumerate(lines, 1):
            if query in ln and file_text_hits < 4:
                if any(abs(c["line"] - i) <= 1 and c["path"] == rel for c in cands):
                    continue
                cands.append({"path": rel, "kind": "text", "symbol": query,
                              "line": i, "end_line": i,
                              "snippet": ln.strip()[:MAX_SNIPPET_CHARS],
                              "source": "text"})
                file_text_hits += 1
        per_file.extend(cands)
        if len(per_file) >= 64:
            break
    return per_file


def cmd_locate(args, root):
    queries = args.query or []
    sources = args.source or []
    if not queries or len(queries) > MAX_QUERIES_HARD:
        return json.dumps(err("provide 1-4 --query values (online use: 2-4 with --source each)"), ensure_ascii=False)
    if len(sources) != len(queries):
        return json.dumps(err("each --query needs exactly one matching --source provenance string"), ensure_ascii=False)
    for q in queries:
        q = (q or "").strip()
        if not q or len(q) > 256 or any(c in q for c in "\n\r\x00"):
            return json.dumps(err("bad query"), ensure_ascii=False)
    for s in sources:
        s = (s or "").strip()
        if not s or len(s) > 200:
            return json.dumps(err("bad source: each query needs a non-empty provenance (<=200 chars), e.g. issue/error/outline"), ensure_ascii=False)
    max_hits = min(max(1, args.max_hits), MAX_HITS_HARD)

    all_cands = []
    for qi, (q, s) in enumerate(zip(queries, sources)):
        for c in find_hits_for_query(root, q.strip()):
            c = dict(c)
            c["query_index"] = qi
            c["query"] = q.strip()
            c["provenance"] = s.strip()
            all_cands.append(c)

    def rank(c):
        return (c["query_index"], 0 if c["source"] == "ast" else 1,
                len(c["path"]), c["line"])
    all_cands.sort(key=rank)
    total_raw = len(all_cands)
    picked = all_cands[:max_hits]

    per_query = []
    for qi, (q, s) in enumerate(zip(queries, sources)):
        per_query.append({"query": q.strip(), "source": s.strip(),
                          "hits": [c for c in picked if c["query_index"] == qi]})

    files = {c["path"] for c in picked}
    out = {"status": "ok", "command": "locate",
           "queries": per_query, "candidates": picked,
           "total_raw": total_raw, "truncated": total_raw > len(picked),
           "static_approx": True, "ambiguous": len(files) > 1,
           "note": "same-name hits are static approximations, NOT proven call edges; snippets (<=400 chars) are NOT complete functions"}

    def trim(o, aggressive=False):
        changed = False
        for c in o.get("candidates", []):
            if len(c.get("snippet", "")) > TRIMMED_SNIPPET_CHARS:
                c["snippet"] = c["snippet"][:TRIMMED_SNIPPET_CHARS]
                changed = True
        for q in o.get("queries", []):
            for c in q.get("hits", []):
                if len(c.get("snippet", "")) > TRIMMED_SNIPPET_CHARS:
                    c["snippet"] = c["snippet"][:TRIMMED_SNIPPET_CHARS]
                    changed = True
        if aggressive and len(o.get("candidates", [])) > 1:
            o["candidates"] = o["candidates"][:max(1, len(o["candidates"]) // 2)]
            for q in o.get("queries", []):
                q["hits"] = [c for c in o["candidates"] if c["query_index"] == o["queries"].index(q)]
            changed = True
        return changed

    return fit_json(out, trim)


def cmd_inspect(args, root):
    try:
        rp = resolve_under(root, args.path)
    except ValueError as e:
        return json.dumps(err(str(e)), ensure_ascii=False)
    if rp.is_symlink() or not rp.is_file():
        return json.dumps(err("missing or symlink"), ensure_ascii=False)
    if rp.suffix != ".py":
        return json.dumps(err("inspect target must be a .py file"), ensure_ascii=False)
    symbol = (args.symbol or "").strip()
    if not symbol or len(symbol) > 256 or any(c in symbol for c in "\n\r\x00"):
        return json.dumps(err("bad symbol"), ensure_ascii=False)
    max_chars = min(max(args.max_chars, 256), OUTPUT_CAP)
    max_functions = min(max(1, args.max_functions), MAX_FUNCTIONS_HARD)

    try:
        text = rp.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return json.dumps(err("read failed"), ensure_ascii=False)
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)
    defs = parse_defs(text)
    matches = [d for d in defs if d["name"] == symbol]
    rel = rp.relative_to(root).as_posix()

    if not matches:
        known = [{"kind": d["kind"], "name": d["name"], "qualname": d["qualname"],
                  "line": d["line"], "end_line": d["end_line"]} for d in defs[:8]]
        return json.dumps({"status": "error", "error": "symbol not found in file",
                           "path": rel, "known_definitions": known,
                           "hint": "use outline to list definitions, then retry with an exact name"}, ensure_ascii=False)
    if args.line is not None:
        matches = [d for d in matches if d["line"] <= args.line <= d["end_line"]]
        if not matches:
            return json.dumps({"status": "error", "error": "no definition of that name spans --line; retry with outline line numbers",
                               "path": rel, "symbol": symbol}, ensure_ascii=False)
    if len(matches) > max_functions:
        cands = [{"kind": d["kind"], "name": d["name"], "qualname": d["qualname"],
                  "line": d["line"], "end_line": d["end_line"],
                  "signature": signature_of(text.splitlines(), d["line"]) or d["qualname"]}
                 for d in matches]
        return json.dumps({"status": "ambiguous", "error": "multiple definitions; retry with --line",
                           "path": rel, "symbol": symbol, "candidates": cands,
                           "bodies": [], "note": "bodies withheld to protect the output budget"}, ensure_ascii=False)

    functions = []
    for d in matches[:max_functions]:
        s, e = d["line"], d["end_line"]
        full = "".join(lines[s - 1:e])
        if len(full) <= max_chars:
            functions.append({"path": rel, "kind": d["kind"], "name": d["name"],
                              "qualname": d["qualname"], "line": s, "end_line": e,
                              "complete": True, "is_truncated": False,
                              "next_start": None, "chars": len(full),
                              "total_chars": len(full), "text": full})
        else:
            buf, used, last = [], 0, s - 1
            for i in range(s, e + 1):
                ln = lines[i - 1] if 1 <= i <= total_lines else ""
                if used + len(ln) > max_chars and buf:
                    break
                buf.append(ln)
                used += len(ln)
                last = i
            functions.append({"path": rel, "kind": d["kind"], "name": d["name"],
                              "qualname": d["qualname"], "line": s, "end_line": e,
                              "complete": False, "is_truncated": True,
                              "next_start": last + 1 if last < e else None,
                              "remaining_end": e,
                              "chars": len("".join(buf)), "total_chars": len(full),
                              "text": "".join(buf),
                              "warning": "TRUNCATED: not the complete function; MUST NOT use as precise-edit input"})

    out = {"status": "ok", "command": "inspect", "path": rel, "symbol": symbol,
           "functions": functions, "functions_shown": len(functions),
           "truncated": any(f["is_truncated"] for f in functions),
           "note": "only functions with complete:true are the full AST node and may support exact reasoning; truncated output must never seed old_string"}

    def trim(o, aggressive=False):
        changed = False
        # shrink truncated bodies first; never silently mark truncated as complete
        for f in o.get("functions", []):
            if not f.get("complete") and len(f.get("text", "")) > 800:
                # keep head so boundaries stay truthful, stay incomplete
                f["text"] = f["text"][:800]
                f["chars"] = len(f["text"])
                changed = True
        if aggressive and len(o.get("functions", [])) > 1:
            o["functions"] = o["functions"][:1]
            o["functions_shown"] = 1
            changed = True
        return changed

    return fit_json(out, trim)


def main():
    ap = argparse.ArgumentParser(
        description="Layered code-context lookup (outline/locate/inspect). Stdlib only, 4000-char output cap.")
    sub = ap.add_subparsers(dest="command", required=True)

    p_o = sub.add_parser("outline", help="directory + definition signatures (no bodies)")
    p_o.add_argument("--path", default=".")
    p_o.add_argument("--max-files", type=int, default=8)
    p_o.add_argument("--max-defs", type=int, default=24)
    p_o.add_argument("--root", default="/workspace")
    p_o.add_argument("--allow-non-workspace-root", action="store_true",
                     help="Allow non-/workspace --root for local diagnosis only.")

    p_l = sub.add_parser("locate", help="2-4 sourced text/AST queries, <=8 total candidates")
    p_l.add_argument("--query", action="append", required=True)
    p_l.add_argument("--source", action="append", required=True,
                     help="Provenance per query, e.g. issue:<substring> / error:<message> / outline:<symbol>. Must match --query count.")
    p_l.add_argument("--max-hits", type=int, default=8)
    p_l.add_argument("--root", default="/workspace")
    p_l.add_argument("--allow-non-workspace-root", action="store_true",
                     help="Allow non-/workspace --root for local diagnosis only.")

    p_i = sub.add_parser("inspect", help="expand 1-2 complete functions by AST boundaries")
    p_i.add_argument("--path", required=True)
    p_i.add_argument("--symbol", required=True)
    p_i.add_argument("--line", type=int, default=None)
    p_i.add_argument("--max-chars", type=int, default=4000)
    p_i.add_argument("--max-functions", type=int, default=2)
    p_i.add_argument("--root", default="/workspace")
    p_i.add_argument("--allow-non-workspace-root", action="store_true",
                     help="Allow non-/workspace --root for local diagnosis only.")

    args = ap.parse_args()
    try:
        root = safe_root(args.root, args.allow_non_workspace_root)
    except ValueError as e:
        print(json.dumps(err(str(e)), ensure_ascii=False))
        return
    if args.command == "outline":
        print(cmd_outline(args, root))
    elif args.command == "locate":
        print(cmd_locate(args, root))
    elif args.command == "inspect":
        print(cmd_inspect(args, root))


if __name__ == "__main__":
    main()
