# Train-mode 6032 save_on_cpu comparison — 2026-10-11

Private 4×L4 kernel **COMPLETE** v1. **Live URL slug:** `ccjjllt/p005-save-on-cpu-6032-train-mode`. Metadata id remains `ccjjllt/p005-save-on-cpu-6032-tm`. Cite both. Poll future status on the live slug.

Same complete select transaction `fastapi_11194-3f51fb716edd-tx01` (6,032 tokens; 662 supervised after shift). NF4 / LoRA rank8 α16 / `selected_chunked` / SDPA / `norms_fp32` fixed; only backbone `save_on_cpu` varies. Fresh DDP worker per arm.

Machine-readable: [`result.json`](result.json). Rank-000 reports: [`baseline`](kaggle-output/save-on-cpu-6032-tm/baseline/rank-000-report.json), [`save_on_cpu`](kaggle-output/save-on-cpu-6032-tm/save_on_cpu/rank-000-report.json).

**This comparison is valid for checkpointed training.** Both arms recorded `outer_training=true`, `base_training=true`, and all 60 decoder layers `decoder_checkpointing_active=true`. Baseline checkpoint **calls** sum 48/60 (died mid-forward). save_on_cpu calls 60/60 after a completed forward.

Wrapper `success=true` means both arms recorded oom/completed, **not** that training succeeded. Neither arm completed the first Adam step (`optimizer_reached=false`, no `adam_state`).

Historical 20261010 [`result.json`](../save-on-cpu-6032-20261010/result.json) status remains `invalid_checkpointed_training_comparison`. This directory does not rewrite v1 kaggle-output or old cpu-grad-check numbers.

## baseline (`save_on_cpu=false`)

CUDA OOM in **forward** at `language_model.layers.47`. Traceback includes `sdpa_attention_forward` / `scaled_dot_product_attention`. Tried 378 MiB; 21.69 GiB in use of 22.03. `optimizer_reached=false`. Host cgroup `oom_kill=0`.

| Field | Rank-000 |
|---|---|
| load | 181.354 s; allocated 17.209 GiB |
| DDP wrap peak allocated | 17.407 GiB |
| forward | started; **not** completed |
| last decoder | 47 `language_model.layers.47` forward |
| checkpoint calls | 48/60 |
| CUDA error | Tried 378.00 MiB; 21.69 GiB in use of 22.03; 349.06 MiB free; PyTorch allocated 21.29 GiB |
| failure_memory peak allocated | 21.469 GiB (bytes 23052150784) |
| RSS / MemAvailable | 2.080 GiB / 180.917 GiB |
| cgroup oom_kill | 0 |
| Adam | not reached; `adam_state` absent |

Load seconds are the cold first arm, not an offload effect.

This traceback with **active** checkpointing is SDPA, not v1's inactive-checkpoint `gate_proj` / NF4 dequant at layer 1. Do not copy “gate_proj is the root cause” from v1 onto this run.

## save_on_cpu (`save_on_cpu=true`, installed)

Forward **completed** (15.38 s, loss 0.0603, last layer 59, peak allocated 18.988 GiB, calls 60/60). Then CUDA OOM in **backward** at layer 53. Tried 124 MiB; 21.91 GiB in use. `optimizer_reached=false`, no adam_state. Host RSS ~6.29 GiB, MemAvailable ~166 GiB, cgroup `oom_kill=0`. Not SIGKILL.

| Field | Rank-000 |
|---|---|
| load | 11.667 s (second arm; cache warm); allocated 17.209 GiB |
| save_on_cpu_installed | true |
| forward | **completed** 15.380 s; loss 0.06029659882187843; last layer 59; peak allocated 18.988 GiB; peak reserved 19.107 GiB; calls 60/60 |
| backward | started; OOM at `language_model.layers.53` |
| traceback | `torch.autograd.backward` / `_engine_run_backward` |
| CUDA error | Tried 124.00 MiB; 21.91 GiB in use of 22.03; 107.06 MiB free; PyTorch allocated 21.39 GiB |
| failure_memory peak allocated | 21.447 GiB (bytes 23028763136) |
| RSS at OOM | 6.292 GiB (6756032512 bytes) |
| MemAvailable at OOM | 166.200 GiB (178456420352 bytes) |
| cgroup oom_kill | 0 |
| Adam | not reached; `adam_state` absent |

Host RSS rose during forward (about 1.55 → 6.14 GiB at forward_end) but MemAvailable stayed huge and `oom_kill=0`. Host pin_memory SIGKILL is **not** established.

## What this does and does not show

Measured difference: offload completed the 6032-token forward that baseline did not; still no Adam.

Do **not** convert that into a promised GiB savings for 9841/12832 or for training. The 3.62 GiB activation estimate is not a claim.

Withdrawn v1 claims remain withdrawn: not “Live MLP/dequant is the established root cause”; not “host pin_memory caused SIGKILL”.

## Quota

| | |
|---|---|
| before | 27356.648 s |
| after | 27969.592 s |
| delta | 612.944 s = 0.170262 h (under 0.6 h cap) |
| wrapper wall | 300.776 s |

No 9841/12832 GPU, no SFT, no pairing, no competition submit in this write-up. Long-seq block: [`blocked-longseq.json`](blocked-longseq.json).

## Next cut

From the **actual** failure stage:

- baseline live operator is SDPA at layer 47 under active checkpointing
- offload failed in backward at layer 53 (CUDA, not host)

Attention path → actual kernel/mask/KV. Do not jump to FSDP-QLoRA or a 16K retry from this record.
