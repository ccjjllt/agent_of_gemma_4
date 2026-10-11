#!/usr/bin/env python3
"""T2 precise span edit: inspect -> apply_span on one script entry. Stdlib only.

Two stable actions the model sees (via run_skill_script, no new host tool):

  python3 source_edit.py inspect --path REL --start N --end M
      [--root /workspace] [--max-chars 4000] [--allow-non-workspace-root]

  python3 source_edit.py apply --span-id SP_ID --replacement TEXT
      [--root /workspace] [--allow-non-workspace-root]
  (multiline local replay: --replacement-file TMPFILE instead of --replacement;
   with the default confinement the file must live under --root; with
   --allow-non-workspace-root an absolute /tmp path is accepted for manual
   diagnosis only. The model path uses --replacement directly.)

inspect uses the same 1-based inclusive range/order accumulation as the frozen
read_range.py so the first experiment isolates the edit-interface increment.
It returns complete_text plus a span_id ONLY when the requested range is fully
shown within the output cap; any clipped/truncated display yields an empty
span_id and must never be used for apply (细则1/5).

apply_span(span_id, replacement) splices exactly one line interval of one file,
atomically (tmp + os.replace), after re-hashing the file: hash mismatch ->
stale_source with a reread suggestion, no fuzzy match, no unescape guessing, no
scope widening (细则3). Replacement is used byte-exact (UTF-8); unedited bytes
and the newline style of untouched lines are preserved because only the slice
is replaced. Failure leaves no half modification (细则5).

Success returns a concise unified diff plus post_source_sha256; source_sha256 is
always the pre-edit hash. updated_span_id is issued only when the new slice is
fully displayable inside the 4000-char total JSON cap, else "" with a suggested
reread range. Output is always complete legal JSON sized by serialized length,
never by slicing the JSON string (细则5).

Scope: apply refuses test paths (test_*.py, *_test.py, tests/) and scratch
names; necessary new production files keep using official write_file (细则6).
No reference-patch/heldout reads, no per-repo answers, no replacement
generation, no fuzzy auto-fixer. UTF-8 decode failure is explicit.
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import span_store as store

OUTPUT_CAP = 4000
MAX_REPLACEMENT_CHARS = 16000
SPAN_RE = re.compile(r"^sp_[0-9a-f]{16}$")

TEST_SUFFIX = ("_test.py",)
TEST_PREFIX = ("test_",)
TEST_DIR_PARTS = ("tests", "test")
TMP_SUFFIX = (".log", ".tmp", ".bak", ".orig", ".rej", ".pyc")
TMP_NAMES = ("repro_", "reproduce", "scratch", "debug_")


def emit(obj: dict) -> None:
    blob = json.dumps(obj, ensure_ascii=False)
    # Never slice the JSON string; shrink optional fields instead.
    # Bounded passes guarantee termination even on adversarial sizes.
    for _ in range(12):
        if len(blob) <= OUTPUT_CAP:
            break
        shrunk = False
        if isinstance(obj.get("compact_diff"), str) and obj["compact_diff"]:
            obj["compact_diff"] = obj["compact_diff"][: len(obj["compact_diff"]) // 2]
            obj["compact_diff_truncated"] = True
            shrunk = True
        elif isinstance(obj.get("complete_text"), str) and obj.get("span_id", "") == "":
            # Clipped inspect display only: halve display text (both aliases)
            # each pass so the loop converges; never slice the JSON string.
            shrunk = False
            for k in ("complete_text", "text"):
                v = obj.get(k)
                if isinstance(v, str) and v:
                    obj[k] = v[:max(256, len(v) // 2)]
                    shrunk = True
            obj["output_fit_trim"] = True
            if not shrunk:
                break
        elif "updated_span_text_preview" in obj:
            obj.pop("updated_span_text_preview", None)
            shrunk = True
        if not shrunk:
            break
        blob = json.dumps(obj, ensure_ascii=False)
    if len(blob) > OUTPUT_CAP:
        # Last resort: minimal legal error, still complete JSON.
        minimal = {"status": "error", "error": "output-too-large",
                   "hint": "re-inspect a smaller range"}
        blob = json.dumps(minimal, ensure_ascii=False)
    sys.stdout.write(blob)


def fail(error: str, **extra) -> None:
    emit({"status": "error", "error": error, **extra})


def check_rel_path(rel: str) -> Optional[str]:
    if not rel or rel.startswith("/") or rel.startswith("\\"):
        return "path must be repo-relative without .."
    parts = rel.replace("\\", "/").split("/")
    if any(p in ("", ".", "..") for p in parts):
        return "path must be repo-relative without .."
    return None


def resolve_under(root, rel: str):
    from pathlib import Path
    err = check_rel_path(rel)
    if err:
        return None, err
    # lstat each component (no following): reject symlink file or symlink dirs.
    parts = rel.replace("\\", "/").split("/")
    cur = root
    for p in parts:
        cur = cur / p
        try:
            if cur.is_symlink():
                return None, "missing or symlink"
        except OSError:
            return None, "resolve failed"
    fp = (root / rel)
    try:
        rp = fp.resolve()
    except OSError:
        return None, "resolve failed"
    try:
        rp.relative_to(root)
    except ValueError:
        return None, "path escapes root"
    if rp.is_symlink() or not rp.is_file():
        return None, "missing or symlink"
    return rp, None


def is_test_or_tmp(rel: str) -> Optional[str]:
    low = rel.lower().replace("\\", "/")
    parts = low.split("/")
    name = parts[-1] if parts else ""
    if name.startswith(TEST_PREFIX) or low.endswith(TEST_SUFFIX):
        return "test file"
    if any(p in TEST_DIR_PARTS for p in parts):
        return "test dir"
    if low.endswith(TMP_SUFFIX):
        return "scratch suffix"
    if any(k in name for k in TMP_NAMES):
        return "scratch name"
    return None


# ---------------- inspect ----------------

def cmd_inspect(args) -> None:
    try:
        root = store.validate_task_root(args.root, args.allow_non_workspace_root)
    except ValueError as e:
        fail(str(e))
        return
    if args.start < 1 or args.end < args.start:
        fail("bad range")
        return
    rp, err = resolve_under(root, args.path)
    if err:
        fail(err, path=args.path)
        return
    try:
        raw = rp.read_bytes()
    except OSError:
        fail("read failed", path=args.path)
        return
    pre_hash = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")  # strict: failure is explicit (细则3)
    except UnicodeDecodeError:
        fail("utf8-decode-failed",
             path=args.path,
             hint="file is not valid UTF-8; use read_file/run_command to inspect, do not edit via span")
        return
    # Preserve newline style: keepends split; untouched lines round-trip intact.
    lines = text.splitlines(keepends=True)
    total = len(lines)
    s = min(args.start, total) if total else args.start
    e = min(args.end, total)
    # Same accumulation order/cap as frozen read_range.py.
    max_chars = min(max(args.max_chars, 256), 4000)
    buf, used, last = [], 0, s - 1
    for i in range(s, e + 1):
        ln = lines[i - 1] if 1 <= i <= total else ""
        if used + len(ln) > max_chars and buf:
            break
        buf.append(ln)
        used += len(ln)
        last = i
    shown_text = "".join(buf)
    truncated = (last < e) or (e < args.end)
    next_start = (last + 1) if truncated and last < total else None
    out = {
        "status": "ok",
        "path": args.path,
        # T1-compatible aliases (same range/order):
        "start_line": s if total else args.start,
        "end_line_shown": last if buf else s - 1,
        "end_line_requested": args.end,
        "total_lines": total,
        "is_truncated": truncated,
        "next_start": next_start,
        "chars": len(shown_text),
        "text": shown_text,
        # §6 spec names:
        "actual_range": {"start": s if total else args.start,
                         "end_shown": last if buf else s - 1,
                         "end_requested": args.end, "total_lines": total},
        "complete_text": shown_text,
        "truncation": {"is_truncated": truncated, "next_start": next_start},
        "source_sha256": pre_hash,
        "span_id": "",
    }
    if truncated or not buf:
        out["hint"] = ("clipped display carries no editable handle; "
                       "re-inspect a smaller window with overlap, then edit")
        emit(out)
        return
    # Complete display: bind handle (nonce+repo+path+hash+byte interval+bytes).
    prefix = "".join(lines[: s - 1])
    slice_text = "".join(lines[s - 1: last])
    byte_start = len(prefix.encode("utf-8"))
    byte_end = byte_start + len(slice_text.encode("utf-8"))
    data, spath = store.load_or_init(root)
    meta = data["meta"]
    span_id = store.new_span_id(data["spans"])
    data["spans"][span_id] = {
        "span_id": span_id,
        "nonce": meta["nonce"],
        "repo_toplevel": meta["toplevel"],
        "repo_head": meta["head"],
        "path": args.path,
        "source_sha256": pre_hash,
        "start_line": s,
        "end_line_shown": last,
        "end_line_requested": args.end,
        "total_lines": total,
        "byte_start": byte_start,
        "byte_end": byte_end,
        "original_text": slice_text,
        "created_seq": max([v.get("created_seq", 0)
                            for v in data["spans"].values()] + [0]) + 1,
    }
    store.save_store(spath, data)
    out["span_id"] = span_id
    out["byte_range"] = {"start": byte_start, "end": byte_end}
    emit(out)


# ---------------- apply ----------------

def _load_replacement(args) -> Optional[str]:
    if args.replacement_file:
        rp = args.replacement_file
        base = None
        try:
            root = store.validate_task_root(args.root,
                                            args.allow_non_workspace_root)
            base = root
        except ValueError:
            return None
        # Confined: repo-relative unless local-diagnosis flag allows /tmp abs.
        from pathlib import Path
        if os.path.isabs(rp):
            if not args.allow_non_workspace_root:
                return None
            cand = Path(rp)
            try:
                cand.resolve().relative_to(Path("/tmp").resolve())
            except Exception:
                try:
                    cand.resolve().relative_to(Path("/var/folders").resolve())
                except Exception:
                    return None
            try:
                if cand.is_symlink() or not cand.is_file():
                    return None
                return cand.read_text(encoding="utf-8")  # strict
            except (OSError, UnicodeDecodeError):
                return None
        else:
            if check_rel_path(rp):
                return None
            full = (base / rp)
            try:
                rr = full.resolve()
                rr.relative_to(base)
                if rr.is_symlink() or not rr.is_file():
                    return None
                return rr.read_text(encoding="utf-8")  # strict
            except (OSError, UnicodeDecodeError, ValueError):
                return None
    return args.replacement


def cmd_apply(args) -> None:
    sid = (args.span_id or "").strip()
    if not SPAN_RE.match(sid):
        emit({"status": "invalid_span", "error": "malformed span_id",
              "hint": "re-inspect the exact range for a fresh handle"})
        return
    try:
        root = store.validate_task_root(args.root, args.allow_non_workspace_root)
    except ValueError as e:
        emit({"status": "invalid_span", "error": str(e)})
        return
    data, spath = store.load_or_init(root)
    meta = data["meta"]
    rec = data["spans"].get(sid)
    if rec is None:
        emit({"status": "invalid_span", "error": "unknown span_id",
              "hint": "re-inspect the exact range for a fresh handle"})
        return
    if rec.get("nonce") != meta["nonce"] or \
       rec.get("repo_toplevel") != meta["toplevel"] or \
       rec.get("repo_head") != meta["head"]:
        emit({"status": "invalid_span", "error": "span belongs to another task/root",
              "hint": "re-inspect under the current task root"})
        return
    rel = rec["path"]
    # Scope gate (细则6): production source only via handle.
    gate = is_test_or_tmp(rel)
    if gate:
        emit({"status": "invalid_replacement",
              "error": f"span path refused ({gate}); handles edit production source only",
              "hint": "never edit tests/scratch via span; leave them untouched"})
        return
    rp, err = resolve_under(root, rel)
    if err:
        emit({"status": "invalid_span", "error": err,
              "suggested_reread": {"path": rel, "start": rec["start_line"],
                                   "end": rec["end_line_shown"]}})
        return
    try:
        cur_raw = rp.read_bytes()
    except OSError:
        emit({"status": "invalid_span", "error": "read failed",
              "suggested_reread": {"path": rel, "start": rec["start_line"],
                                   "end": rec["end_line_shown"]}})
        return
    cur_hash = hashlib.sha256(cur_raw).hexdigest()
    if cur_hash != rec["source_sha256"]:
        emit({"status": "stale_source",
              "error": "source changed since inspect; re-read required",
              "path": rel,
              "suggested_reread": {"path": rel, "start": rec["start_line"],
                                   "end": rec["end_line_shown"]},
              "hint": "re-inspect, do not fuzzy-match or widen scope"})
        return
    try:
        cur_text = cur_raw.decode("utf-8")
    except UnicodeDecodeError:
        emit({"status": "invalid_span", "error": "utf8-decode-failed"})
        return
    replacement = _load_replacement(args)
    if replacement is None or not isinstance(replacement, str):
        emit({"status": "invalid_replacement", "error": "missing replacement text"})
        return
    if len(replacement) > MAX_REPLACEMENT_CHARS:
        emit({"status": "invalid_replacement",
              "error": f"replacement too large ({len(replacement)} > {MAX_REPLACEMENT_CHARS})",
              "hint": "narrow the span and retry with a smaller replacement"})
        return
    # Single-interval, single-file splice; replacement used byte-exact.
    lines = cur_text.splitlines(keepends=True)
    s, e = rec["start_line"], rec["end_line_shown"]
    if s < 1 or e < s or (lines and (s > len(lines) or e > len(lines))):
        emit({"status": "stale_source",
              "error": "recorded range no longer maps to file lines",
              "suggested_reread": {"path": rel, "start": s, "end": e}})
        return
    old_slice = "".join(lines[s - 1: e]) if lines else ""
    # Sanity: hash-bound record already pins content; no fuzzy fallback here.
    new_text = ("".join(lines[: s - 1]) + replacement + "".join(lines[e:])) \
        if lines else replacement
    try:
        new_raw = new_text.encode("utf-8")
    except UnicodeEncodeError:
        emit({"status": "invalid_replacement", "error": "replacement not encodable"})
        return
    # Atomic single-file write (tmp + replace): no half modification.
    try:
        tmp = rp.parent / (rp.name + f".t2span.{os.getpid()}.tmp")
        tmp.write_bytes(new_raw)
        os.replace(tmp, rp)
    except OSError as ex:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        emit({"status": "invalid_span", "error": f"atomic write failed: {ex}"})
        return
    post_hash = hashlib.sha256(new_raw).hexdigest()
    new_lines = new_text.splitlines()
    repl_nlines = len(replacement.splitlines())
    new_end = s + repl_nlines - 1 if repl_nlines else s - 1
    diff = "".join(difflib.unified_diff(
        old_slice.splitlines(), replacement.splitlines(),
        fromfile=f"a/{rel}@{s}-{e}", tofile=f"b/{rel}@{s}-{new_end}",
        lineterm="", n=3))
    if len(diff) > 1500:
        diff = diff[:1500]
    out = {
        "status": "applied",
        "path": rel,
        "changed_lines": {"start_line": s, "old_end_line": e,
                          "new_end_line": new_end},
        "compact_diff": diff,
        "source_sha256": cur_hash,       # pre-edit hash, explicit
        "post_source_sha256": post_hash,  # post-edit hash
        "updated_span_id": "",
    }
    # updated_span_id only when the new slice is fully displayable in-budget.
    preview_probe = {"status": "applied", "compact_diff": diff,
                     "replacement_len": len(replacement)}
    probe_len = len(json.dumps(preview_probe, ensure_ascii=False))
    if repl_nlines > 0 and len(replacement) + probe_len + 600 <= OUTPUT_CAP:
        data2, spath2 = store.load_or_init(root)  # same nonce (HEAD unchanged)
        meta2 = data2["meta"]
        # Re-derive byte interval of the new slice from the written file.
        try:
            verify_raw = rp.read_bytes()
            verify_text = verify_raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            verify_text = new_text
        vlines = verify_text.splitlines(keepends=True)
        new_slice = "".join(vlines[s - 1: new_end]) if repl_nlines else ""
        bstart = len("".join(vlines[: s - 1]).encode("utf-8"))
        bend = bstart + len(new_slice.encode("utf-8"))
        nid = store.new_span_id(data2["spans"])
        data2["spans"][nid] = {
            "span_id": nid, "nonce": meta2["nonce"],
            "repo_toplevel": meta2["toplevel"], "repo_head": meta2["head"],
            "path": rel, "source_sha256": post_hash,
            "start_line": s, "end_line_shown": new_end,
            "end_line_requested": new_end,
            "total_lines": len(vlines),
            "byte_start": bstart, "byte_end": bend,
            "original_text": new_slice,
            "created_seq": max([v.get("created_seq", 0)
                                for v in data2["spans"].values()] + [0]) + 1,
        }
        store.save_store(spath2, data2)
        out["updated_span_id"] = nid
        out["updated_range"] = {"start": s, "end": new_end}
    else:
        out["suggested_reread"] = {"path": rel, "start": s, "end": new_end}
        out["hint"] = ("new slice not fully displayable in-budget; "
                       "re-inspect the suggested range for a fresh handle")
    emit(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="source_edit.py",
        description="T2 span inspect/apply. Stdlib only, JSON to stdout, "
                    "total cap 4000 chars, never truncated mid-JSON.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect", help="read a range; get span_id if complete")
    pi.add_argument("--path", required=True, help="repo-relative file path")
    pi.add_argument("--start", type=int, required=True)
    pi.add_argument("--end", type=int, required=True)
    pi.add_argument("--root", default="/workspace")
    pi.add_argument("--max-chars", type=int, default=4000)
    pi.add_argument("--allow-non-workspace-root", action="store_true")
    pi.set_defaults(func=cmd_inspect)
    pa = sub.add_parser("apply", help="replace one inspected span by id")
    pa.add_argument("--span-id", required=True)
    pa.add_argument("--replacement", default=None,
                    help="full new text for the span (exact bytes)")
    pa.add_argument("--replacement-file", default=None,
                    help="local-replay alternative: file holding replacement")
    pa.add_argument("--root", default="/workspace")
    pa.add_argument("--allow-non-workspace-root", action="store_true")
    pa.set_defaults(func=cmd_apply)
    return ap


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
