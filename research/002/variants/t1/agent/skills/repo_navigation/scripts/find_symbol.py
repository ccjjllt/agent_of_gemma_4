#!/usr/bin/env python3
"""Find Python symbols via text + AST. Stdlib only. Bounded output.

Usage:
  python3 find_symbol.py --symbol NAME [--root /workspace] [--max-hits 8] [--expand] [--allow-non-workspace-root]

Confined to --root (default /workspace). Never follows symlinks.
Skips hidden dirs, venv, node_modules, __pycache__, files > 512 KiB.
Matches def / async def / class / plain text hits for NAME.
--expand adds static-approx reference candidates (same-name text hits),
labelled static_approx:true, never claimed as proven call edges.
Non-/workspace roots require --allow-non-workspace-root (local diagnosis only).
"""
import argparse
import ast
import json
import os
from pathlib import Path

SKIP_DIRS = {"__pycache__", "venv", ".venv", "node_modules", ".git", ".hg"}
MAX_FILE_BYTES = 512 * 1024
MAX_SNIPPET_CHARS = 400


def safe_root(root: str, allow_non_ws: bool = False) -> Path:
    p = Path(root).resolve()
    ws = Path("/workspace").resolve()
    # Enforce workspace confinement: ws variable gates non-/workspace roots.
    # Kaggle run must use /workspace; local diagnosis needs explicit flag.
    if p.is_symlink():
        raise SystemExit(json.dumps({"status": "error", "error": "symlink root rejected"}))
    if p != ws and not allow_non_ws:
        raise SystemExit(json.dumps({"status": "error", "error": "non-workspace root rejected (use --allow-non-workspace-root for local diagnosis)"}))
    return p


def iter_py_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # prune in place
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".") and d not in SKIP_DIRS]
        # never descend through symlinked dirs
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


def snippet_of(lines, lineno, width=3):
    lo = max(1, lineno - width)
    hi = lineno + width
    chunk = "".join(lines[lo - 1:hi])[:MAX_SNIPPET_CHARS]
    return chunk


def find_in_file(fp: Path, root: Path, symbol: str):
    cands = []
    try:
        text = fp.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return cands
    lines = text.splitlines(keepends=True)
    rel = fp.relative_to(root).as_posix()
    # AST pass for def/async def/class (exact name match)
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and node.name == symbol:
                kind = ("async def" if isinstance(node, ast.AsyncFunctionDef)
                        else "def" if isinstance(node, ast.FunctionDef) else "class")
                end = getattr(node, "end_lineno", node.lineno)
                cands.append({
                    "path": rel, "kind": kind, "symbol": symbol,
                    "line": node.lineno, "end_line": end,
                    "snippet": snippet_of(lines, node.lineno),
                    "source": "ast",
                })
    except SyntaxError:
        pass
    # Text fallback: exact substring hits (bounded per file)
    for i, ln in enumerate(lines, 1):
        if symbol in ln and len([c for c in cands if c["path"] == rel]) < 4:
            # skip lines already covered by AST hits nearby
            if any(abs(c["line"] - i) <= 1 and c["path"] == rel for c in cands):
                continue
            cands.append({
                "path": rel, "kind": "text", "symbol": symbol,
                "line": i, "end_line": i,
                "snippet": ln.strip()[:MAX_SNIPPET_CHARS],
                "source": "text",
            })
            if len(cands) >= 12:
                break
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--root", default="/workspace")
    ap.add_argument("--max-hits", type=int, default=8)
    ap.add_argument("--expand", action="store_true")
    ap.add_argument("--allow-non-workspace-root", action="store_true",
                    help="Allow non-/workspace --root for local diagnosis only.")
    args = ap.parse_args()
    symbol = args.symbol.strip()
    out = {"status": "ok", "symbol": symbol, "candidates": [],
           "truncated": False, "static_approx": False, "ambiguous": False}
    if not symbol or len(symbol) > 256 or any(c in symbol for c in "\n\r\x00"):
        out.update({"status": "error", "error": "bad symbol"})
        print(json.dumps(out, ensure_ascii=False))
        return
    try:
        root = safe_root(args.root, args.allow_non_workspace_root)
    except SystemExit as e:
        print(str(e))
        return
    if not root.is_dir():
        out.update({"status": "error", "error": "root missing"})
        print(json.dumps(out, ensure_ascii=False))
        return
    max_hits = min(max(1, args.max_hits), 8)
    all_cands = []
    for fp in iter_py_files(root):
        all_cands.extend(find_in_file(fp, root, symbol))
        if len(all_cands) >= 64:
            break
    # Rank: ast hits first, then shorter paths, then line number
    def rank(c):
        return (0 if c["source"] == "ast" else 1, len(c["path"]), c["line"])
    all_cands.sort(key=rank)
    picked = all_cands[:max_hits]
    out["candidates"] = picked
    out["truncated"] = len(all_cands) > len(picked)
    out["total_raw"] = len(all_cands)
    # Ambiguity: same symbol in >1 file
    files = {c["path"] for c in picked}
    out["ambiguous"] = len(files) > 1
    if args.expand and picked:
        refs = []
        for fp in iter_py_files(root):
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = fp.relative_to(root).as_posix()
            for i, ln in enumerate(text.splitlines(), 1):
                if symbol in ln and not any(c["path"] == rel and c["line"] == i for c in picked):
                    refs.append({"path": rel, "line": i,
                                 "snippet": ln.strip()[:MAX_SNIPPET_CHARS]})
                    if len(refs) >= max_hits:
                        break
            if len(refs) >= max_hits:
                break
        out["ref_hits"] = refs
        out["static_approx"] = True
    blob = json.dumps(out, ensure_ascii=False)
    if len(blob) > 4000:
        # hard cap: keep candidates count but trim snippets
        for c in out["candidates"]:
            c["snippet"] = c["snippet"][:120]
        for r in out.get("ref_hits", []):
            r["snippet"] = r["snippet"][:120]
        out["truncated"] = True
        blob = json.dumps(out, ensure_ascii=False)[:4000]
        print(blob)
    else:
        print(blob)


if __name__ == "__main__":
    main()
