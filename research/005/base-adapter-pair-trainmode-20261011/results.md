# Task 4 SFT and T0 base vs adapter pairing — SKIPPED

Date: 2026-10-11. Worktree `p005-trainmode-save-on-cpu`. **No GPU. No SFT. No scored pairing. No competition submit.**

## No resolved-delta table

SFT and pairing did not run. There is no adapter fingerprint, no per-issue base/adapter resolved table, no new-fix / regression counts, and no duration comparison. Do not invent a delta from NLL, from the diagnostic DDP adapter (`ccjjllt/p005-training-ddp-probe`), or from earlier T0 edits.

Machine-readable skip: [`skipped.json`](skipped.json). SFT block: [`../sft-candidate-20261011/blocked.json`](../sft-candidate-20261011/blocked.json).

## Why

Binding gate: SFT/pairing runs **only if** (a) at least one complete **train** length backprop succeeded **and** (b) there are independent non-reserved trainable samples. Both false.

| Upstream | Status | Implication |
|---|---|---|
| Train-mode kernel `ccjjllt/p005-save-on-cpu-6032-train-mode` v1 (metadata id `ccjjllt/p005-save-on-cpu-6032-tm`) | Valid checkpointed-training comparison; both arms failed first Adam. Baseline CUDA OOM in **forward** at layer 47 SDPA (378 MiB, 21.69/22.03 GiB, calls 48/60). save_on_cpu forward completed then CUDA OOM in **backward** at layer 53 (124 MiB, 21.91 GiB, calls 60/60). Wrapper `success=true` only means both arms recorded oom/completed. | **gpu_sft=false**. No adapter. |
| 9841/12832 | **BLOCKED**; not launched (`did_not_push_9841`, `did_not_push_12832`, `did_not_retry_16k`, `did_not_rewrite_fsdp`) | No backprop at a real train length. |
| Task 2 CPU scan | 0 new independent issues, 0 new recovery events; corpus cannot grow without new rollouts; v2 census unchanged | Same 9 eligible issues. Reserved eight stay out. |

Quota pointer (unchanged this skip): [`quota-after.json`](../save-on-cpu-6032-trainmode-20261011/quota-after.json) — used 27969.592 s; 6032 run Δ +612.944 s (0.170262 h). This skip launched no GPU.

Train issues (cannot be pairing evidence): `rich_2725`, `rich_3518`, `rich_3942`, `rich_4075`. Select: `fastapi_11194`, `fastapi_14873`, `fastapi_5077`, `fastapi_9425`.

Historical v1 `ccjjllt/p005-save-on-cpu-6032` result status remains `invalid_checkpointed_training_comparison`.

## Reserved eight IDs (proposed, not run)

Still excluded from v2 train/select (and same repo+commit):

`fastapi_13713`, `requests_6592`, `requests_6644`, `requests_7309`, `requests_7505`, `rich_3130`, `rich_3296`, `rich_3469`

These were **not** evaluated. `gpu_launched=false`. `gpu_sft=false`.

## Carrier

Did **not** create `notebooks/005-base-adapter-pair-single-cell.py` as a fake completed carrier. After a real SFT adapter exists, derive that notebook from `notebooks/002-joint-explore-v30-single-cell.py` (frozen T0, six tools, same budgets, `max_output_tokens 16384`). Do not overwrite P002/P003 freeze files.

## Next cut

From the actual failure stage: baseline live operator is SDPA at layer 47 under active checkpointing; offload failed in backward at layer 53 (CUDA, not host). Host pin_memory SIGKILL is not established (`oom_kill=0`, MemAvailable still huge). **Not** a 16K retry; FSDP-QLoRA stays deferred. Do not revive withdrawn v1 claims (“Live MLP/dequant is the established root cause”, “host pin_memory caused SIGKILL”).

Offload completed forward where baseline did not; still no Adam. That is not a promised GiB savings for 9841/12832 or for training.

After a **memory-feasible complete-tx SFT** exports a real adapter (`gpu_sft=true`) from a clean init **and** independent non-reserved samples exist, run T0-fixed pairing on these eight IDs. Eight IDs decide whether to expand; they do not extrapolate a medal.
