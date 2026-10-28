# Exact selection-bound teacher prompt configuration

`competition.vfarl.replay.run_official` now accepts optional `--prompt-selection PATH`. It reads the explicit fixed100 train selection, matches the actual task instance and task-file SHA, and loads budget/harness from that entry's hash-bound `actual_replay_result`. JSON numeric types are preserved. The unmodified official `build_agent_prompt` must then produce the selection's exact prompt SHA before an output directory, sandbox or tool is created. This profile supplies configuration only; old report outcomes are not reassigned to the current task.

Example (existing official host Python/PYTHONPATH and actual workspace/recipe remain required):

```sh
python3.12 -m competition.vfarl.replay.run_official \
  --task <actual-current-task.json> --workspace <source-workspace> \
  --recipe <reviewed-recipe.json> --output <new-run-directory> \
  --prompt-selection competition/vfarl/l4x4/round100/question-revision-r3/selection.json \
  --source-layout . --verify --verify-baseline
```

In this mode omit `--time-minutes`, `--tool-calls`, `--command-timeout`, and `--verification-timeout`: any explicitly supplied one is rejected, even if numerically equal to the selection value. Other existing dependency/source/verification flags remain available. Successful opt-in `run-summary.json` adds `prompt_selection_profile` with selection path/SHA, actual task SHA, source report path/SHA, exact budget/harness, and expected/actual prompt SHA. Default runs add no such summary field and retain the prior CLI numeric behavior.

The small API is `selection_prompt_profile(selection_path, task_path) -> (budget, harness, profile)` followed by `bind_selection_prompt(profile, official_prompt)`; the latter checks the actual official builder output. The production replay calls both before sandbox construction.

Manual CPU evidence: the actual official builder reproduced all100 current r3 prompt hashes, with72 integer60 budgets and28 float30.0 budgets; indices4/40 retain int60 and82 retains float30.0. No source prompt was normalized or substituted. Actual CLI parsing retained default int60 and legacy explicit `--time-minutes 60` float60.0. Two actual replay rejection paths (abbreviated explicit `--time=60` conflict and a bound wrong expected prompt hash) stopped before output/sandbox creation; no tools were executed.

The default-path replay AST, projected through the new disabled opt-in branches and excluding the absent summary unpack, is exactly the prior replay AST. Existing replay/config creation, tools, verification and legacy defaults were not otherwise changed. `main` changes are the new optional argument and value-preserving argparse actions that record the four explicit settings, including abbreviations and `--flag=value`.

No oracle, model, GPU, new teacher or frozen artifact was run/modified. Observations are in `manual-observations.json`; this is prompt production compatibility evidence, not model/repair gain.
