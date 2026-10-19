#!/usr/bin/env python3
"""P005 I1 diff snapshots: keep at most 4 same-task patch states, restore one.

Stdlib only. No network. Never reads reference patches, heldout tasks, host
model/data directories, or other attempts' outputs.

Why this file exists (handoff doc I1 + frontier report D):
  T2 `apply` returns one diff; a later wrong edit can silently destroy an
  already-good patch (Coherence Collapse: near-correct states get corrupted).
  This tool cheaply snapshots the current worktree state so an overturned
  hypothesis can return to a *named, evidenced* earlier state. Snapshot
  existence never means correctness: NOTHING here picks a "best" patch, and
  restore is always an explicit model action.

Entries (single stable skill entry; one flag string per call, no nested
quoting layers -- I1 escaping rule):
  snapshot --note "..." [--parent p_xxx|none]
      Capture tracked worktree changes vs HEAD: full bytes of each changed
      production file (cap 8 files / 200 KiB each / 512 KiB total) + per-file
      sha256 + HEAD. Records parent_patch_id explicitly (default "none",
      must be "none" or a known id). At most MAX_SNAPSHOTS=4 per task root;
      beyond that the call FAILS with the existing ids -- no silent eviction,
      no auto-ranking.
  list [--patch-id p_xxx]
      Show manifests (ids, parent, HEAD, files+hashes, note). Never contents.
  restore --patch-id p_xxx
      Write snapshot bytes back (atomic tmp+replace; delete tombstoned paths),
      then report per-file results + current diff stat. Warns on HEAD change
      but restores worktree files regardless. Never restores "the best" --
      only the id asked for.

State lives in the sandbox system temp dir bound to the task root
(<tmp>/p005_snapshots/root_<hash>/), outside the repo so snapshots never enter
`git diff`; the model cannot set the state root. Same binding idea as the T2
span_store, except snapshots are keyed by repo toplevel only (not HEAD):
working-tree edits do not change HEAD, and snapshots must survive the edits
they record. Each snapshot still records its HEAD for provenance.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from workspace_paths import resolve_task_root, resolve_issue_input

STORE_DIRNAME = "p005_snapshots"
STORE_FILENAME = "store.json"
MAX_SNAPSHOTS = 4
MAX_FILES = 8
MAX_FILE_BYTES = 200 * 1024
MAX_TOTAL_BYTES = 512 * 1024
OUTPUT_CAP = 4000


def emit(obj):
    blob = json.dumps(obj, ensure_ascii=False)
    if len(blob) > OUTPUT_CAP:
        minimal = {"status": "error", "error": "output-too-large",
                   "hint": "use list --patch-id for one snapshot at a time"}
        blob = json.dumps(minimal, ensure_ascii=False)
    sys.stdout.write(blob)


def fail(error, **extra):
    emit({"status": "error", "error": error, **extra})


def system_temp_base():
    for key in ("TMPDIR", "TEMP", "TMP"):
        v = os.environ.get(key)
        if v:
            try:
                p = Path(v)
                if p.is_dir():
                    return p
            except OSError:
                continue
    try:
        return Path(tempfile.gettempdir())
    except Exception:
        return Path("/tmp")


def validate_task_root(root_arg, allow_non_ws):
    return resolve_task_root(root_arg, allow_non_ws)


def _git_out(args, cwd, timeout=15):
    try:
        r = subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True,
                           timeout=timeout)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def repo_toplevel(root):
    out = _git_out(["rev-parse", "--show-toplevel"], cwd=root)
    if out:
        try:
            return str(Path(out.decode("utf-8").strip()).resolve())
        except (OSError, UnicodeDecodeError):
            pass
    return str(root)


def repo_head(root):
    out = _git_out(["rev-parse", "HEAD"], cwd=root)
    if out:
        try:
            return out.decode("utf-8").strip() or "no-git"
        except UnicodeDecodeError:
            return "no-git"
    return "no-git"


def state_dir_for(toplevel):
    base = system_temp_base() / STORE_DIRNAME
    h = hashlib.sha256(toplevel.encode("utf-8")).hexdigest()[:16]
    d = base / ("root_" + h)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_store(spath, toplevel):
    if spath.is_file():
        try:
            data = json.loads(spath.read_bytes().decode("utf-8"))
            meta = data.get("meta", {})
            if meta.get("toplevel") == toplevel and \
               isinstance(data.get("snapshots"), dict):
                return data
        except Exception:
            pass
    import secrets
    data = {"meta": {"nonce": secrets.token_hex(8), "toplevel": toplevel},
            "snapshots": {}, "seq": 0}
    save_store(spath, data)
    return data


def save_store(spath, data):
    spath.parent.mkdir(parents=True, exist_ok=True)
    tmp = spath.parent / (STORE_FILENAME + ".%d.tmp" % os.getpid())
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, spath)


def check_rel(rel):
    if not rel or rel.startswith("/") or rel.startswith("\\"):
        return False
    parts = rel.replace("\\", "/").split("/")
    return not any(p in ("", ".", "..") for p in parts)


def changed_paths(root):
    """Tracked worktree paths differing from HEAD (NUL-separated, no shell)."""
    out = _git_out(["diff", "--no-renames", "HEAD", "--name-only", "-z"], cwd=root)
    if out is None:
        return None
    untracked = _git_out(["ls-files", "--others", "--exclude-standard", "-z"], cwd=root)
    if untracked is None:
        return None
    return sorted({p for p in (out + untracked).decode("utf-8").split("\x00") if p})


def diff_stat(root):
    out = _git_out(["diff", "HEAD", "--stat", "--"], cwd=root)
    if out is None:
        return ""
    try:
        return out.decode("utf-8", errors="replace")[-1200:]
    except Exception:
        return ""


def cmd_snapshot(args, root):
    toplevel = repo_toplevel(root)
    head = repo_head(root)
    d = state_dir_for(toplevel)
    data = load_store(d / STORE_FILENAME, toplevel)
    snaps = data["snapshots"]
    if len(snaps) >= MAX_SNAPSHOTS:
        fail("snapshot-budget-exceeded",
             hint="at most %d snapshots per task; explicitly restore one and "
                  "continue -- nothing is auto-evicted or auto-ranked" %
                  MAX_SNAPSHOTS,
             kept=sorted(snaps.keys()))
        return
    parent = args.parent or "none"
    if parent != "none" and parent not in snaps:
        fail("unknown parent_patch_id '%s'" % parent,
             known=sorted(snaps.keys()))
        return
    paths = changed_paths(root)
    if paths is None:
        fail("cannot read worktree diff (need git checkout at root)")
        return
    if len(paths) > MAX_FILES:
        fail("snapshot-too-large",
             hint="%d changed files > %d cap; narrow the worktree diff first" %
                  (len(paths), MAX_FILES),
             files=paths[:MAX_FILES])
        return
    files = []
    total = 0
    for rel in paths:
        if not check_rel(rel):
            fail("snapshot refused: odd path in diff", path=rel[:100])
            return
        fp = root / rel
        try:
            rp = fp.resolve()
        except OSError:
            fail("cannot resolve changed path", path=rel[:100])
            return
        try:
            rp.relative_to(root)
        except ValueError:
            fail("changed path escapes root", path=rel[:100])
            return
        if fp.is_symlink():
            fail("snapshot refused: symlink in diff", path=rel[:100])
            return
        if not rp.exists():
            files.append({"path": rel, "status": "deleted",
                          "sha256": "deleted"})
            continue
        if not rp.is_file():
            fail("snapshot refused: non-file in diff", path=rel[:100])
            return
        try:
            raw = rp.read_bytes()
        except OSError:
            fail("cannot read changed file", path=rel[:100])
            return
        if len(raw) > MAX_FILE_BYTES:
            fail("snapshot-too-large",
                 hint="file %s over %d bytes; snapshots keep small source "
                      "states only" % (rel, MAX_FILE_BYTES))
            return
        total += len(raw)
        if total > MAX_TOTAL_BYTES:
            fail("snapshot-too-large",
                 hint="total over %d bytes" % MAX_TOTAL_BYTES)
            return
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            fail("snapshot refused: non-UTF-8 file", path=rel[:100])
            return
        files.append({"path": rel, "status": "modified",
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "bytes": len(raw), "text": text, "mode": rp.stat().st_mode & 0o777})
    import secrets
    for _ in range(32):
        pid = "p_" + secrets.token_hex(8)
        if pid not in snaps:
            break
    else:
        fail("patch id space exhausted")
        return
    data["seq"] = data.get("seq", 0) + 1
    snaps[pid] = {"patch_id": pid, "parent_patch_id": parent,
                  "seq": data["seq"], "head": head,
                  "note": args.note or "",
                  "files": files, "diff_stat": diff_stat(root)}
    save_store(d / STORE_FILENAME, data)
    emit({"status": "ok", "command": "snapshot", "patch_id": pid,
          "parent_patch_id": parent, "head": head,
          "files": [{"path": f["path"], "status": f["status"],
                     "sha256": f["sha256"]} for f in files],
          "snapshots_kept": sorted(snaps.keys()),
          "note": "stored only; nothing here judges which patch is correct"})


def cmd_list(args, root):
    toplevel = repo_toplevel(root)
    d = state_dir_for(toplevel)
    data = load_store(d / STORE_FILENAME, toplevel)
    snaps = data["snapshots"]
    if args.patch_id:
        rec = snaps.get(args.patch_id)
        if rec is None:
            fail("unknown patch_id '%s'" % args.patch_id,
                 known=sorted(snaps.keys()))
            return
        picked = [rec]
    else:
        picked = sorted(snaps.values(), key=lambda r: r.get("seq", 0))
    manifest = [{"patch_id": r["patch_id"],
                 "parent_patch_id": r["parent_patch_id"], "seq": r["seq"],
                 "head": r["head"], "note": r.get("note", ""),
                 "files": [{"path": f["path"], "status": f["status"],
                            "sha256": f["sha256"]} for f in r["files"]],
                 "diff_stat": r.get("diff_stat", "")} for r in picked]
    emit({"status": "ok", "command": "list", "snapshots": manifest,
          "count": len(manifest), "budget": MAX_SNAPSHOTS})


def cmd_restore(args, root):
    pid = (args.patch_id or "").strip()
    toplevel = repo_toplevel(root)
    data = load_store(state_dir_for(toplevel) / STORE_FILENAME, toplevel)
    rec = data["snapshots"].get(pid)
    if rec is None:
        fail("unknown patch_id", known=sorted(data["snapshots"]))
        return
    head = repo_head(root)
    if head != rec["head"]:
        fail("HEAD changed; cannot restore a different task/base", head_now=head)
        return
    current = changed_paths(root)
    if current is None:
        fail("cannot inspect current worktree")
        return
    recorded = {f["path"]: f for f in rec["files"]}
    paths = sorted(set(current) | set(recorded))
    if len(paths) > MAX_FILES:
        fail("restore file budget exceeded")
        return
    plan = []
    try:
        for rel in paths:
            if not check_rel(rel):
                raise ValueError("invalid restore path")
            fp = root / rel
            if fp.is_symlink() or root not in fp.resolve().parents:
                raise ValueError("symlink or escaped restore path")
            if fp.exists() and not fp.is_file():
                raise ValueError("non-file restore path")
            before = fp.read_bytes() if fp.exists() else None
            mode_before = fp.stat().st_mode & 0o777 if fp.exists() else 0o644
            if before is not None and len(before) > MAX_FILE_BYTES:
                raise ValueError("current file too large for reversible restore")
            if rel in recorded:
                entry = recorded[rel]
                after = None if entry["status"] == "deleted" else entry["text"].encode("utf-8")
                mode_after = entry.get("mode", mode_before)
                if after is not None and hashlib.sha256(after).hexdigest() != entry["sha256"]:
                    raise ValueError("snapshot content hash mismatch")
            else:
                # This path changed after the checkpoint. Restore HEAD, or remove
                # an untracked/new path absent from HEAD and from the snapshot.
                after = _git_out(["show", head + ":" + rel], cwd=root)
                tree = _git_out(["ls-tree", head, "--", rel], cwd=root)
                mode_after = int(tree.split()[0], 8) & 0o777 if tree else 0o644
            if after is not None and len(after) > MAX_FILE_BYTES:
                raise ValueError("baseline file too large")
            plan.append((fp, before, mode_before, after, mode_after))
    except (ValueError, OSError) as error:
        fail(str(error))
        return

    def write(fp, raw, mode):
        if raw is None:
            if fp.exists():
                fp.unlink()
            return
        fp.parent.mkdir(parents=True, exist_ok=True)
        temp = fp.parent / (fp.name + ".p005restore.%d.tmp" % os.getpid())
        temp.write_bytes(raw)
        os.chmod(temp, mode)
        os.replace(temp, fp)

    completed = []
    try:
        for entry in plan:
            fp, before, mode_before, after, mode_after = entry
            completed.append(entry)
            write(fp, after, mode_after)
    except OSError as error:
        rollback_errors = []
        for fp, before, mode_before, _, _ in reversed(completed):
            try:
                write(fp, before, mode_before)
            except OSError as rollback_error:
                rollback_errors.append(str(rollback_error))
        fail("restore failed", detail=str(error), rollback_errors=rollback_errors)
        return
    emit({"status": "ok", "command": "restore", "restored_patch_id": pid,
          "head_now": head, "files": [str(p.relative_to(root)) for p, *_ in plan],
          "current_diff_stat": diff_stat(root),
          "note": "Requested worktree snapshot restored; correctness is not inferred."})


def build_parser():
    ap = argparse.ArgumentParser(
        prog="patch_store.py",
        description="P005 I1 diff snapshots (max 4, explicit restore). "
                    "Stdlib only, complete JSON output.")
    ap.add_argument("--root", default="/workspace")
    ap.add_argument("--allow-non-workspace-root", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_s = sub.add_parser("snapshot", help="capture current worktree state")
    p_s.add_argument("--note", default=None)
    p_s.add_argument("--parent", default="none")
    p_s.set_defaults(func=cmd_snapshot)
    p_l = sub.add_parser("list", help="show snapshot manifests")
    p_l.add_argument("--patch-id", default=None)
    p_l.set_defaults(func=cmd_list)
    p_r = sub.add_parser("restore", help="restore one explicit patch id")
    p_r.add_argument("--patch-id", required=True)
    p_r.set_defaults(func=cmd_restore)
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    try:
        root = validate_task_root(args.root, args.allow_non_workspace_root)
    except ValueError as e:
        print(json.dumps({"status": "error", "error": str(e)},
                         ensure_ascii=False))
        return
    args.func(args, root)


if __name__ == "__main__":
    main()
