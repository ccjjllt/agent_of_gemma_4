---
name: working-state
description: Issue contract plus at most two sourced hypotheses and one next action. record/show store only what the model was told and what tools returned; record-issue/write-handoff/read-handoff compose the executor handoff. Never infers root causes, stdlib only.
---

# working-state skill

Implements handoff-doc I2 (behavioral constraint + two-hypothesis state) and
the I4 handoff file round-trip. The tool stores; the model judges. State lives
in the sandbox system temp dir bound to the task root, outside the repo.

Workspace root is `/workspace`. Local diagnosis must pass explicit
`--allow-non-workspace-root`.

## Scripts (run via `run_skill_script` in the workspace sandbox)

```sh
python3 working_state.py record [--trigger T --expected E --preserve P] \
  [--fact 'id|source_ref|claim'] [--hypothesis 'id|layer|claim|support|contradiction'] \
  [--drop-fact id] [--drop-hypothesis id] [--chosen id] [--unknown U] \
  [--next-kind inspect|probe|edit --next-target T --next-purpose P] \
  [--current-patch-id p_xxx] [--root /workspace] [--allow-non-workspace-root]
python3 working_state.py show [--root /workspace] [--allow-non-workspace-root]
```

- `layer` is one of `producer|transformer|consumer|boundary`; at most 2
  hypotheses (a third FAILS until one is dropped/overturned). Drops apply
  before adds, so overturn-then-replace fits one call.
- `support`/`contradiction` are comma-separated fact/observation ids.
  Claims must not contain `|` (use `--facts-json`/`--hyps-json` for exact text).
- `show` prints the state plus `changed_fields` and current `next_action`.

```sh
python3 working_state.py record-issue --issue-file <path> [--budget-note ...] [--source ...]
python3 working_state.py write-handoff --code 'relpath:start-end' [--code ...]
python3 working_state.py read-handoff --handle <path>
```

- Handoff use only: `record-issue` stores the model-passed visible problem
  text verbatim (repo-relative path, or absolute under system temp with the
  diagnosis flag). The skill never reads host task files.
- `write-handoff` re-reads registered code ranges verbatim, collects the live
  worktree diff, and composes `handoff.json` (~2-4K tokens advisory; over
  budget it trims prose first, keeps code and full issue). Requires a stored
  issue, at least one hypothesis, and one next action -- otherwise it is a
  status card, not a handoff, and the call fails.
- `read-handoff` (executor's first action) reads the file back by handle;
  handles must live under the sandbox temp dir or the task root.

## Rules the agent must follow

1. Record the contract before editing; leave `unknown` honest instead of
   inventing expectations.
2. Update the state when preparing an edit or when new output overturns an
   assumption; the next action must answer the discrepancy.
3. Local checks describe one behavior only; `success` prints never mark the
   contract satisfied by themselves.
