# Task 3 T0 base vs adapter pairing — SKIPPED

Date: 2026-10-10. Worktree `p005-sampling-sft`. **No GPU. No scored pairing. No competition submit.**

## No resolved-delta table

Pairing did not run. There is no per-issue base/adapter resolved table, no new-fix / regression counts, and no duration comparison. Do not invent a delta from NLL, from the diagnostic DDP adapter (`ccjjllt/p005-training-ddp-probe`), or from earlier T0 edits.

Machine-readable skip: [`skipped.json`](skipped.json).

## Why

Binding gate: Task 3 pairing runs **only if** Task 2 produced a **real-data SFT** adapter.

| Upstream | Status | Implication |
|---|---|---|
| Task 2 kernel `ccjjllt/p005-capacity-sft-v2` v1 | 8K complete-tx OOM (actual 6032, 21.47 GiB/rank); 16K not retried | **gpu_sft=false**. No adapter exported. |
| Diagnostic adapter `ccjjllt/p005-training-ddp-probe` | compatibility / DDP probe only | **Not** a repair-gain arm. Unused here. |

Quota pointer (unchanged this task): [`quota-after.json`](../capacity-sft-v2-20261010/quota-after.json) — used 12002.806s, Task 2 Δ +1247s (0.346h). This skip launched no GPU.

Train issues (cannot be pairing evidence): `rich_2725`, `rich_3518`, `rich_3942`, `rich_4075`. Select: `fastapi_11194`, `fastapi_14873`, `fastapi_5077`, `fastapi_9425`.

## Reserved eight IDs (proposed, not run)

Still excluded from v2 train/select (and same repo+commit):

`fastapi_13713`, `requests_6592`, `requests_6644`, `requests_7309`, `requests_7505`, `rich_3130`, `rich_3296`, `rich_3469`

These were **not** evaluated. `gpu_launched=false`. `gpu_sft=false`.

## Carrier

Did **not** create `notebooks/005-base-adapter-pair-single-cell.py` as a fake completed carrier. After a real SFT adapter exists, derive that notebook from `notebooks/002-joint-explore-v30-single-cell.py` (frozen T0, six tools, same budgets, `max_output_tokens 16384`). Do not overwrite P002/P003 freeze files.

## Next step

After a **memory-feasible complete-tx SFT** exports a real adapter (`gpu_sft=true`), run T0-fixed pairing on these eight IDs: same T0 prompt, official six tools, budgets 5min/50/80/60s, `max_output_tokens 16384`, one vLLM load, base (`n_adapters=0`) then adapter. Conclusions only from resolved differences. Eight IDs decide whether to expand; they do not extrapolate a medal.
