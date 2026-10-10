---
name: repo_navigation
description: Deterministic text/AST navigation for generic Python repos. Finds symbols (incl. async def), returns bounded source slices, and reports pre-submit patch hygiene. No network, no answers, stdlib only.
---

# repo_navigation skill

Deterministic helper. It does not fix code and does not submit. It only
returns bounded evidence the agent must verify by reading the source.

Workspace root is `/workspace`. All paths are confined under it. Symlinks are
never followed. Scripts use only the Python standard library. Default rejects
non-`/workspace` `--root`; local diagnosis must pass explicit
`--allow-non-workspace-root`.

## Scripts (run via `run_skill_script` in the workspace sandbox)

All scripts print a JSON object to stdout. Stable cap is 4000 chars per
call on final stdout; caller sees `truncated: true`/`is_truncated: true` when
capped. When truncated, continue with `next_start` sharding (read_range
returns `next_start`; re-issue with smaller window and overlap).

### 1. `find_symbol.py`

Search identifiers from the issue text (file/function/class/error strings).

```sh
python3 find_symbol.py --symbol <name> [--root /workspace] [--max-hits 8] [--allow-non-workspace-root]
python3 find_symbol.py --symbol <name> --expand [--max-hits 8] [--allow-non-workspace-root]
```

- Scans `*.py` under root, skipping hidden dirs, `venv/`, `node_modules/`,
  `__pycache__/`, files > 512 KiB.
- Matches: `def <name>`, `async def <name>`, `class <name>`, exact text hits.
- Default output: up to 8 candidates with `path`, `kind`
  (`def`/`async def`/`class`/`text`), `line`, `end_line`, short `snippet`.
- `--expand`: for each confirmed candidate, add static近似 reference/call
  candidates (`ref_hits`, max 8 total) found by bounded text search for the
  symbol name. Results are labelled `static_approx: true`; same-name hits are
  NOT proven call edges. Ambiguity is reported via `ambiguous: true`.
- Never reads reference patches, other attempts, or host files.

### 2. `read_range.py`

Exact bounded read with truthful boundaries.

```sh
python3 read_range.py --path <repo/relative.py> --start <1-based> --end <incl> [--max-chars 4000] [--root /workspace] [--allow-non-workspace-root]
```

- Returns `start_line`, `end_line_shown`, `is_truncated`, `next_start`, and
  `text`. `end_line_shown` is the last line actually included, never a
  truncation-inflated value.
- Rejects absolute paths, `..` escapes, missing files, symlinks, non-`.py`
  outside the repo tree (only files under `/workspace`).

### 3. `check_patch.py`

Pre-submit hygiene report. Read-only except `git diff` plumbing; it never
deletes files and never calls submit.

```sh
python3 check_patch.py [--root /workspace] [--allow-non-workspace-root]
```

Reports: `files_changed`, `production_files` (non-test source), `test_files_touched`,
`tmp_files` (repro/logs that would pollute the diff), `syntax_ok` per changed
`.py` (via `ast.parse`, incl. `async def` support), `diff_check` (runs
`git diff --check` when git is available), `diff_bytes`, `empty_source_diff`.
A non-empty report is not a pass/fail verdict; the agent decides.

## Rules the agent must follow

1. Copy `old_string` byte-for-byte from `read_range.py`/`read_file` output.
   On `old_string not found`, re-read before retrying; never blindly repeat.
2. Candidate count and output caps are initial values for measurement, not
   proven optima. Record when the skill returns empty/ambiguous.
3. Temporary analysis stays in `/tmp`; never write scratch files into the repo
   tree that would enter the final diff.
