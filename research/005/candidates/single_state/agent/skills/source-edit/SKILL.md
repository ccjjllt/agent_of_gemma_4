---
name: source-edit
description: Handle-bound precise single-span edit plus explicit diff snapshots. inspect returns exact text plus a span_id only when complete; apply_span replaces only that span after hash check; patch_store keeps at most 4 named states with explicit restore. No fuzzy match, no code generation, no auto-best, stdlib only.
---

# source-edit skill (P005 vendored copy + snapshot extension)

Vendored from the T2 prototype
`research/002/candidates/t2_span_edit/agent/skills/source_edit/`
(`source_edit.py`, `span_store.py` byte-identical at pack time; frozen
originals untouched). Adds `patch_store.py` (I1 diff snapshots) in the same
scripts directory. All output uses complete-JSON budgeting (fields shrink
first, JSON never sliced) -- the frozen T1 `read_range` slicing behavior is
not reused here.

Workspace root is `/workspace`. All paths confined under it. Symlinks never
followed. Stdlib only. Local diagnosis must pass explicit
`--allow-non-workspace-root`.

## Scripts (run via `run_skill_script` in the workspace sandbox)

### `inspect` / `apply_span` — unchanged T2 semantics

```sh
python3 source_edit.py inspect --path <repo/relative.py> --start <1-based> --end <incl> [--max-chars 4000] [--root /workspace] [--allow-non-workspace-root]
python3 source_edit.py apply --span-id <sp_hex> --replacement <exact new text> [--root /workspace] [--allow-non-workspace-root]
```

- Same 1-based inclusive range/order accumulation as frozen `read_range.py`.
- Complete display only yields `span_id`; clipped display yields `span_id: ""`
  and must never be used for `apply`. Model supplies only `replacement`.
- `stale_source` → re-`inspect` suggested range, retry once narrowed.
- Scope: production source only (tests/scratch refused).

### `patch_store.py` — at most 4 named worktree states, explicit restore

```sh
python3 patch_store.py snapshot --note "..." [--parent p_xxx|none] [--root /workspace] [--allow-non-workspace-root]
python3 patch_store.py list [--patch-id p_xxx] [--root /workspace] [--allow-non-workspace-root]
python3 patch_store.py restore --patch-id <p_xxx> [--root /workspace] [--allow-non-workspace-root]
```

- Snapshot records `parent_patch_id` + per-file sha256 + HEAD + full bytes of
  tracked changes vs HEAD (cap 8 files / 200 KiB each / 512 KiB total).
- Beyond 4 snapshots the call FAILS listing kept ids: no silent eviction,
  no auto-ranking, no "best" pick. Restore is always an explicit `--patch-id`.
- State in sandbox system temp bound to the task root (T2 binding idea),
  outside the repo, never enters `git diff`. Single stable entry, one flag
  string per call (no nested quoting layers).

## Rules the agent must follow

1. `inspect` → `apply_span(span_id, replacement)`; never from clipped display.
2. On `stale_source`/`invalid_span`: re-`inspect`, retry once narrowed.
3. Snapshot before risky rewrites; after overturned assumptions `restore` the
   named state and record why -- never stack patches on a refuted premise.
4. Quote `compact_diff`/`post_source_sha256`/`patch_id` as evidence; model
   "done" text is not success evidence.
