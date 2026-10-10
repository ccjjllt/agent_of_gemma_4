---
name: layered_lookup
description: Layered code evidence (outline signatures, 2-4 sourced locate queries, 1-2 complete-function inspects). Bounded candidates and shared output budget. No answers, no vectors, stdlib only.
---

# layered_lookup skill

Deterministic helper for tasks that still lack useful function/branch
evidence. It does not fix code and does not submit. It only returns bounded
evidence the agent must verify by reading the source.

Workspace root is `/workspace`. All paths are confined under it. Symlinks are
never followed. Scripts use only the Python standard library. Default rejects
non-`/workspace` `--root`; local diagnosis must pass explicit
`--allow-non-workspace-root`.

## Scripts (run via `run_skill_script` in the workspace sandbox)

One script, three subcommands. Every call prints a single JSON object to
stdout. Stable cap is 4000 chars per call on final stdout with valid JSON
preserved (text fields are trimmed first; caller sees `truncated: true` /
`is_truncated: true` when capped). Shared budget: at most 8 candidates per
`locate` call, at most 2 function bodies per `inspect` call, at most 8 files /
32 definitions per `outline` call.

### 1. `outline` — directory plus definition signatures, no bodies

```sh
python3 layered.py outline --path <rel-dir, default .> [--max-files 8] [--max-defs 24] [--root /workspace] [--allow-non-workspace-root]
```

- Lists immediate subdirectories, up to `--max-files` Python files (sorted,
  with per-file `def_count`), and up to `--max-defs` definitions with `path`,
  `kind` (`def`/`async def`/`class`), `name`, `qualname`, `line`, `end_line`
  and a one-line `signature` (<=160 chars).
- Scans `*.py` under the target dir, skipping hidden dirs, `venv/`,
  `node_modules/`, `__pycache__/`, `.git`/`.hg`, files > 512 KiB.
- `signature` is the declaration line only. It is NOT the function body.
- Never reads reference patches, other attempts, or host files.

### 2. `locate` — 2-4 sourced queries, at most 8 total candidates

```sh
python3 layered.py locate --query <q1> --source <provenance1> [--query <q2> --source <provenance2> ...] [--max-hits 8] [--root /workspace] [--allow-non-workspace-root]
```

- Online use: pass 2-4 `--query` values, each with exactly one matching
  `--source` provenance string (<=200 chars), e.g.
  `issue:<substring from the issue>`, `error:<observed message>`,
  `outline:<symbol seen in outline>`. A missing or empty `--source` is an
  error. (Single-query calls are for local diagnosis only.)
- Per query: exact `def`/`async def`/`class` name match via AST first, then
  bounded same-file text hits. Global ranking keeps at most `--max-hits`
  (hard max 8) across all queries.
- Results are labelled `static_approx: true`; same-name hits are NOT proven
  call edges. Ambiguity is reported via `ambiguous: true`.
- Snippets (<=400 chars, trimmed to <=120 under budget pressure) are context
  windows, NOT complete functions.

### 3. `inspect` — expand one or two complete functions by AST boundaries

```sh
python3 layered.py inspect --path <repo/relative.py> --symbol <Name> [--line N] [--max-chars 4000] [--max-functions 2] [--root /workspace] [--allow-non-workspace-root]
```

- Finds exact-name `def`/`async def`/`class` nodes in the file and returns
  the full node text (`line`..`end_line`). At most 2 bodies per call.
- Each body carries `complete` (full AST node included), `is_truncated`,
  `next_start`/`remaining_end`, `chars`/`total_chars`, and truthful `line` /
  `end_line` boundaries. Multiple same-name definitions without `--line`
  return an `ambiguous` candidate list with NO bodies; retry with `--line`.
- Rejects absolute paths, `..` escapes, missing files, symlinks, non-`.py`
  targets, files outside the root.

## Rules the agent must follow

1. NEVER treat an `outline` signature, a `locate` snippet, or the first 400
   characters of any output as the complete function. Before reasoning about
   semantics or editing, call `inspect` and require `complete: true`.
2. NEVER use truncated output (`truncated`/`is_truncated: true`,
   `complete: false`) as precise-edit input (`old_string`). Re-inspect with a
   narrower target (exact `--symbol` plus `--line`) until `complete: true`,
   or fall back to `read_file`/`read_range` exact reads.
3. Every `locate` query must cite its source. Do not run cryptic one-word
   sweeps; prefer issue text, observed error strings, and symbols already
   seen in `outline`.
4. Candidate counts and output caps are initial values for measurement, not
   proven optima. Record when the skill returns empty/ambiguous or when
   `outline` shows no relevant file.
5. Temporary analysis stays in `/tmp`; never write scratch files into the
   repo tree that would enter the final diff.
6. No neural or vector retrieval is involved. Do not add embeddings, rerank
   models, or network calls around this skill.
