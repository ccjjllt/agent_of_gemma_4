# Source and change notice — P006 E2

Reference author: Wilmer E. Henao (`verracodeguacas`).

Exact upstream: https://www.kaggle.com/code/verracodeguacas/gemma-4-agent-budget-fit-single-agent?scriptVersionId=355310852 (Budget-Fit Single Agent, V5).

License: **Apache License 2.0**. The exact V5 page License section was read through Chrome accessibility on 2026-10-13 and links https://www.apache.org/licenses/LICENSE-2.0 . This is observed page evidence, not an assumed default. Full license text accompanies this file as `LICENSE-Budget-Fit.txt` and is copied into the run's frozen bundle with the source/version/change record.

The upstream four-file dictionary is checked against the complete archived V5 Notebook source with safe literal parsing. This study changes only `max_time_minutes: 8` to `5`, and the two matching `8 minutes` prompt phrases to `5 minutes`. Tool declarations, command timeout 240, 28-call limit, 80 turns, sampling and thinking remain unchanged. The modified reference is called `budget_fit_5m` / matched-five-minute; it does **not** inherit the upstream 0.18 result.

The other arm, `exec_closure`, comes unchanged from the locally recovered completed 2026-10-11 experiment. Source SHA and every Agent file SHA are recorded in `experiment-manifest.json` and `source-manifest.json`. The prior P005 carrier implementation is reused as a source dependency without changing historical files.

Neither this package nor its generator submits a competition entry. Default `pair.py` execution is a bounded GPU experiment; `prepare.py` is CPU-only setup and official Agent construction. Keep the source notices and license with redistributed research artifacts.
