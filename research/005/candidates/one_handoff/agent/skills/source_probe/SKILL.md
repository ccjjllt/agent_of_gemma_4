---
name: source_probe
description: Decision-oriented reading plus on-demand path probe. locate (1-4 queries + question), inspect (1-3 targets, exact|outline), probe (one command + question + timeout, optional repo-filtered setprofile trace). Truncated output never editable; exit 0 proves nothing alone. Stdlib only.
---

# source_probe skill

Implements handoff-doc I3, reusing the T4 `layered_lookup` ideas (sourced
queries, AST-boundary complete reads, shared output budget) with complete-JSON
budgeting (never sliced mid-string).

Workspace root is `/workspace`. Local diagnosis must pass explicit
`--allow-non-workspace-root`.

## Scripts (run via `run_skill_script` in the workspace sandbox)

```sh
python3 source_probe.py locate --query <q1> [--query <q2> ...] --question 'what decision answers?' [--source <prov1> ...] [--scope <subdir>] [--max-hits 8]
python3 source_probe.py inspect --target 'path:symbol[:line]' [--target ...] --question '...' --mode exact|outline [--max-chars 4000]
python3 source_probe.py probe --command '<single command string>' --question '...' [--timeout 60] [--assert 'name:substring' ...] [--trace]
```

- Every call needs one `--question`: the fact to discriminate, not a long
  reasoning dump. `locate` takes 1-4 queries, `inspect` 1-3 targets.
- `locate`/`inspect` relations are `textual_reference` only. `exact` returns
  full AST nodes (`complete: true`) or a marked TRUNCATED head with NO handle
  -- this tool issues no `span_id` ever; only `source_edit inspect` does, and
  only for complete displays. `outline` returns signatures, never bodies.
- `probe` runs only to confirm a trigger/path: single command string (no
  nested quoting layers), named `--assert` substring checks are the only
  recorded observations. `exit_kind` is `ok|assertion|exception|timeout|
  setup_error`; exit 0 NEVER implies reproduced or fixed.
- `--trace` (plain `python3 <script.py> [args]` under root only) installs
  `sys.setprofile`/`threading.setprofile` filtered to the task repo and
  reports ordered functions + counts + a short repo-marked stack. No host
  files collected, no full call graph built. Anything else runs untraced with
  a note. Probe artifacts live in the sandbox temp dir, never in the repo.

## Rules the agent must follow

1. NEVER treat outline signatures, locate snippets, or truncated heads as
   complete functions or edit input.
2. One discriminating read/probe per ambiguity; with sufficient evidence edit
   directly via `source_edit`.
3. If the probe only shows missing fixtures/imports, record `setup_error` and
   keep reasoning from source -- do not burn budget fixing the environment.
