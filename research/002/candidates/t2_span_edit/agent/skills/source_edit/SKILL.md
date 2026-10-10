---
name: source_edit
description: Handle-bound precise single-span edit. inspect returns exact text plus a span_id; apply_span replaces only that span after hash check. No fuzzy match, no code generation, stdlib only.
---

# source_edit skill

`inspect` keeps the exact bytes; the model only supplies the new text.
It does not generate fixes and never reads reference patches or heldout tasks.

Workspace root is `/workspace`. All paths are confined under it. Symlinks are
never followed. Scripts use only the Python standard library. Final stdout is
complete legal JSON capped at 4000 chars of serialized length (fields shrink,
the JSON string itself is never sliced). Local diagnosis must pass explicit
`--allow-non-workspace-root`.

## Scripts (run via `run_skill_script` in the workspace sandbox)

### `inspect` — read a range, get an editable handle only when complete

```sh
python3 source_edit.py inspect --path <repo/relative.py> --start <1-based> --end <incl> [--max-chars 4000] [--root /workspace] [--allow-non-workspace-root]
```

- Same 1-based inclusive range/order accumulation as frozen `read_range.py`
  (first experiment isolates the edit-interface increment).
- Returns `status`, `span_id`, `source_sha256` (**pre-edit** hash),
  `actual_range`, `complete_text`, `truncation {is_truncated,next_start}`,
  `byte_range`, plus `read_range`-compatible aliases
  (`start_line/end_line_shown/text/is_truncated/next_start`).
- Handle binds the task nonce + repo toplevel + HEAD + relative path + file
  byte hash + line↔byte interval + original bytes. Only an actually fully
  displayed slice is issued a handle; a clipped/truncated/skeleton display
  returns `span_id: ""` and must never be used for `apply`.
- UTF-8 decode failure returns `utf8-decode-failed` explicitly; unedited bytes
  and the newline style of untouched lines are preserved by byte-exact splice.
- State lives in code-constructed sandbox temp
  (`<tmp>/p002_t2_spans/root_<hash>/store.json`), outside the repo so it never
  enters `git diff`; the model cannot set the state root. First access creates
  a nonce; later calls reuse it. Root/HEAD mismatch (next task) resets the
  store: handles never cross tasks.

### `apply_span` — replace exactly one inspected span, atomically

```sh
python3 source_edit.py apply --span-id <sp_hex> --replacement <exact new text> [--root /workspace] [--allow-non-workspace-root]
```

- Single interval, single file, atomic (`tmp + os.replace`); failure leaves no
  half modification. Multiple spans: re-`inspect` each one separately; no
  multi-span transaction scheduler in this version.
- Pre-write re-hashes the file: match → write; mismatch → `stale_source` with
  `suggested_reread`, no fuzzy match, no unescape guessing, no widened scope.
- Statuses: `applied` | `stale_source` | `invalid_span` | `invalid_replacement`.
  Success returns `changed_lines`, concise `compact_diff`, `source_sha256`
  (pre) + `post_source_sha256` (post). `updated_span_id` is issued only when
  the new slice is fully displayable inside the output cap, else `""` plus a
  suggested reread range.
- `replacement` is used byte-exact (UTF-8); the model provides the complete new
  text. The script is not an auto-generator or fuzzy fixer.
- Scope: production source only. Test paths (`test_*.py`, `*_test.py`,
  `tests/`) and scratch names are refused; genuinely new production files keep
  using official `write_file` instead of a handle.

## Rules the agent must follow

1. `inspect` → `apply_span(span_id, replacement)`; never retype the old
   snippet, never apply from a clipped display.
2. On `stale_source`/`invalid_span`: re-`inspect` the suggested range, narrow
   if needed, retry once. Never widen scope to force a match.
3. Keep the total JSON in mind: if no `updated_span_id` is returned, re-inspect
   before the next edit. Quote `compact_diff`/`post_source_sha256` as evidence.
