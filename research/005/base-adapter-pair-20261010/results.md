# Task 5 T0 base vs adapter pairing — SKIPPED

Date: 2026-10-10. Worktree `p005-next-round-sft`. **No GPU. No scored pairing. No competition submit.**

## No resolved-delta table

Pairing did not run. There is no per-issue base/adapter resolved table, no new-fix / regression counts, and no duration comparison. Do not invent a delta from NLL, from the diagnostic DDP adapter, or from earlier T0 edits.

Machine-readable skip: [`skipped.json`](skipped.json).

## Why

Binding gate: Task 5 pairing runs **only if** Task 3 proved CT can load an adapter **and** Task 4 produced a **real-data SFT** adapter.

| Upstream | Status | Implication |
|---|---|---|
| Task 3 | `tool_call_ok_no_repair_claim` | Compatibility only. Diagnostic adapter is **not** a repair-gain arm. |
| Task 4 | `trainable_ok=false`, `gpu_launched=false` | No repair adapter. 0/15 train fit 4096; 3/15 fit 8192. |

Train issues (cannot be pairing evidence): `rich_2725`, `rich_3518`, `rich_3777`, `rich_3942`, `rich_4075`.

## Candidate eval issues (proposed, not run)

Pool: `length_report.json` `dev_panel_not_in_train` (19 IDs). Proposed 12, disjoint from train:

`fastapi_11194`, `fastapi_13713`, `fastapi_14873`, `fastapi_5077`, `fastapi_9425`, `requests_6592`, `requests_6644`, `requests_7309`, `requests_7505`, `rich_3130`, `rich_3296`, `rich_3469`

These were **not** evaluated. `gpu_launched=false`.

## Carrier

Did **not** create `notebooks/005-base-adapter-pair-single-cell.py` as a fake completed carrier. After a real SFT adapter exists, derive that notebook from `notebooks/002-joint-explore-v30-single-cell.py` (frozen T0, six tools, same budgets, `max_output_tokens 16384`). Do not overwrite P002/P003 freeze files.

## Next step

When Task 4 (or a successor) exports a real-data SFT adapter with `trainable_ok=true`, run T0-fixed pairing on these disjoint issues: same T0 prompt, six tools, budgets (5min/50/80/60s), one load, base then adapter or interleaved. Conclusions only from resolved differences.
