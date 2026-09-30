#!/usr/bin/env python3
"""Bounded exact read with truthful boundaries. Stdlib only.

Usage:
  python3 read_range.py --path REL --start N --end M [--root /workspace] [--max-chars 4000] [--allow-non-workspace-root]

Returns start_line, end_line_shown (last line actually included),
is_truncated, next_start, text. Never inflates end_line beyond shown text.
Rejects absolute paths, .. escapes, symlinks, files outside root.
Non-/workspace roots require --allow-non-workspace-root (local diagnosis only).
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--root", default="/workspace")
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--allow-non-workspace-root", action="store_true",
                    help="Allow non-/workspace --root for local diagnosis only.")
    args = ap.parse_args()
    out = {"status": "ok"}
    root = Path(args.root).resolve()
    ws = Path("/workspace").resolve()
    if root.is_symlink():
        print(json.dumps({"status": "error", "error": "symlink root rejected"}))
        return
    if root != ws and not args.allow_non_workspace_root:
        print(json.dumps({"status": "error", "error": "non-workspace root rejected (use --allow-non-workspace-root for local diagnosis)"}))
        return
    rel = args.path
    if rel.startswith("/") or ".." in Path(rel).parts:
        print(json.dumps({"status": "error", "error": "path must be repo-relative without .."}))
        return
    fp = (root / rel)
    try:
        rp = fp.resolve()
    except OSError:
        print(json.dumps({"status": "error", "error": "resolve failed"}))
        return
    try:
        rp.relative_to(root)
    except ValueError:
        print(json.dumps({"status": "error", "error": "path escapes root"}))
        return
    if rp.is_symlink() or not rp.is_file():
        print(json.dumps({"status": "error", "error": "missing or symlink"}))
        return
    if args.start < 1 or args.end < args.start:
        print(json.dumps({"status": "error", "error": "bad range"}))
        return
    try:
        lines = rp.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except OSError:
        print(json.dumps({"status": "error", "error": "read failed"}))
        return
    total = len(lines)
    s = min(args.start, total) if total else args.start
    e = min(args.end, total)
    # accumulate until max-chars (stable cap 4000; use next_start to continue)
    max_chars = min(max(args.max_chars, 256), 4000)
    buf = []
    used = 0
    last = s - 1
    for i in range(s, e + 1):
        ln = lines[i - 1] if 1 <= i <= total else ""
        if used + len(ln) > max_chars and buf:
            break
        buf.append(ln)
        used += len(ln)
        last = i
    text = "".join(buf)
    truncated = (last < e) or (e < args.end)
    out.update({
        "path": rel,
        "start_line": s if total else args.start,
        "end_line_shown": last if buf else s - 1,
        "end_line_requested": args.end,
        "total_lines": total,
        "is_truncated": truncated,
        "next_start": (last + 1) if truncated and last < total else None,
        "chars": len(text),
        "text": text[:max_chars],
    })
    blob = json.dumps(out, ensure_ascii=False)
    print(blob[:4000])


if __name__ == "__main__":
    main()
