#!/usr/bin/env python3
"""Pre-submit hygiene report. Read-only (except git plumbing reads). Stdlib only.

Usage: python3 check_patch.py [--root /workspace] [--allow-non-workspace-root]
Reports files_changed, production/test/tmp splits, per-file syntax via
ast.parse, git diff --check when available, diff size, empty-source flag.
Never deletes files, never submits. Scratch files belong in /tmp, not repo.
Non-/workspace roots require --allow-non-workspace-root (local diagnosis only).
"""
import argparse
import ast
import json
import subprocess
from pathlib import Path

TEST_MARKERS = ("test_", "_test.py", "/tests/", "tests/")
TMP_SUFFIX = (".log", ".tmp", ".bak", ".orig", ".rej", ".pyc")
TMP_NAMES = ("repro_", "reproduce", "scratch", "debug_")


def sh(cmd, cwd, timeout=20):
    try:
        r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "")[:6000], (r.stderr or "")[:2000]
    except Exception as e:
        return -1, "", f"exec-failed: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace")
    ap.add_argument("--allow-non-workspace-root", action="store_true",
                    help="Allow non-/workspace --root for local diagnosis only.")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    ws = Path("/workspace").resolve()
    out = {"status": "ok", "root": str(root)}
    if root.is_symlink() or not root.is_dir():
        print(json.dumps({"status": "error", "error": "bad root"}))
        return
    if root != ws and not args.allow_non_workspace_root:
        print(json.dumps({"status": "error", "error": "non-workspace root rejected (use --allow-non-workspace-root for local diagnosis)"}))
        return
    code, diff, err = sh(["git", "diff", "--binary", "--name-only", "--", "."],
                         cwd=root)
    if code != 0:
        code2, diff2, err2 = sh(["git", "diff", "--name-only", "--", "."], cwd=root)
        code, diff, err = code2, diff2, err2
    files = [l.strip() for l in diff.splitlines() if l.strip()] if code == 0 else []
    out["git_available"] = (code == 0)
    out["git_stderr"] = err[:500] if code != 0 else ""
    out["files_changed"] = files
    prod, tests, tmp = [], [], []
    for f in files:
        low = f.lower()
        is_test = low.startswith("tests/") or "/tests/" in low or \
            Path(f).name.startswith("test_") or low.endswith("_test.py")
        is_tmp = low.endswith(TMP_SUFFIX) or any(k in Path(f).name for k in TMP_NAMES)
        if is_tmp:
            tmp.append(f)
        elif is_test:
            tests.append(f)
        else:
            prod.append(f)
    out["production_files"] = prod
    out["test_files_touched"] = tests
    out["tmp_files"] = tmp
    # syntax check on changed .py files
    syntax = {}
    for f in files:
        if not f.endswith(".py"):
            continue
        fp = root / f
        try:
            rp = fp.resolve()
            rp.relative_to(root)
            if rp.is_symlink() or not rp.is_file():
                syntax[f] = "missing-or-symlink"
                continue
            src = rp.read_text(encoding="utf-8", errors="replace")
            ast.parse(src)  # supports async def
            syntax[f] = "ok"
        except SyntaxError as e:
            syntax[f] = f"syntax-error: {e}"
        except Exception as e:
            syntax[f] = f"read-failed: {e}"
    out["syntax"] = syntax
    out["syntax_ok"] = all(v == "ok" for v in syntax.values())
    c2, check_out, check_err = sh(["git", "diff", "--check", "--", "."], cwd=root)
    out["diff_check_exit"] = c2
    out["diff_check"] = (check_out + check_err)[:2000]
    c3, numstat, _ = sh(["git", "diff", "--numstat", "--", "."], cwd=root)
    out["numstat"] = numstat[:2000]
    c4, stat, _ = sh(["git", "diff", "--stat", "--", "."], cwd=root)
    out["stat"] = stat[:2000]
    out["empty_source_diff"] = len(prod) == 0
    blob = json.dumps(out, ensure_ascii=False)
    print(blob[:4000] if len(blob) > 4000 else blob)


if __name__ == "__main__":
    main()
