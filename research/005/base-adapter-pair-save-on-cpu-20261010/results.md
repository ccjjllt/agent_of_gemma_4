# Task 3 SFT and T0 base vs adapter pairing — SKIPPED

Date: 2026-10-11. Worktree `p005-save-on-cpu`. **No GPU. No SFT. No scored pairing. No competition submit.**

## No resolved-delta table

SFT and pairing did not run. There is no adapter fingerprint, no per-issue base/adapter resolved table, no new-fix / regression counts, and no duration comparison. Do not invent a delta from NLL, from the diagnostic DDP adapter (`ccjjllt/p005-training-ddp-probe`), or from earlier T0 edits.

Machine-readable skip: [`skipped.json`](skipped.json).

## Review correction (2026-10-11)

The Task 1 worker omitted `model.train()`, so its decoder checkpointing was inactive. The historical failure observations remain valid, but the inferred normal-training memory mechanism does not. See [review](../save-on-cpu-6032-20261010/review-20261011.json). The SFT/pairing skip remains valid because no real adapter exists.

## Why

Binding gate: Task 3 SFT/pairing runs **only if** Task 2 proved at least one complete train length can backprop **and** there are independent train samples disjoint from the reserved eight.

| Upstream | Status | Implication |
|---|---|---|
| Task 1 kernel `ccjjllt/p005-save-on-cpu-6032` v1 | both arms failed first Adam step; baseline CUDA OOM in **forward** decoder layer 1/60 MLP `gate_proj` NF4 dequant (248 MiB, 21.548 GiB); `save_on_cpu` host SIGKILL during forward (~20.75 GiB GPU) | **gpu_sft=false**. No adapter. Capacity updates discarded. |
| Task 2 capacity 9841/12832 | **BLOCKED**; not launched | No backprop at a real train length. |
| Task 2 CPU scan | 214 complete prefixes; 24 ≤8192; 0 new independent issues | Same 9 eligible issues. Reserved eight stay out. |

Quota pointer (unchanged this task): [`quota-after.json`](../save-on-cpu-6032-20261010/quota-after.json) — used 19451.299s, Task 1 Δ +1063.808s (0.296h). This skip launched no GPU.

Train issues (cannot be pairing evidence): `rich_2725`, `rich_3518`, `rich_3942`, `rich_4075`. Select: `fastapi_11194`, `fastapi_14873`, `fastapi_5077`, `fastapi_9425`.

## Reserved eight IDs (proposed, not run)

Still excluded from v2 train/select (and same repo+commit):

`fastapi_13713`, `requests_6592`, `requests_6644`, `requests_7309`, `requests_7505`, `rich_3130`, `rich_3296`, `rich_3469`

These were **not** evaluated. `gpu_launched=false`. `gpu_sft=false`.

## Carrier

Did **not** create `notebooks/005-base-adapter-pair-single-cell.py` as a fake completed carrier. After a real SFT adapter exists, derive that notebook from `notebooks/002-joint-explore-v30-single-cell.py` (frozen T0, six tools, same budgets, `max_output_tokens 16384`). Do not overwrite P002/P003 freeze files.

## Next cut

Correct `model.train()` and repeat one matched 6032-token comparison with decoder checkpointing active in both arms. Record host/GPU memory before choosing an operator-specific optimization. Live MLP/dequant attribution and host pinned-memory OOM remain unproven under the intended recipe. **Not** a 16K retry; FSDP-QLoRA stays deferred.

After a **memory-feasible complete-tx SFT** exports a real adapter (`gpu_sft=true`) from a clean init, run T0-fixed pairing on these eight IDs: same T0 prompt, official six tools, budgets 5min/50/80/60s, `max_output_tokens 16384`, one vLLM load, base (`n_adapters=0`) then adapter. Conclusions only from resolved differences. Eight IDs decide whether to expand; they do not extrapolate a medal.
