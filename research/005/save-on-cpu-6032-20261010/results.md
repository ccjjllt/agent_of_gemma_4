# 6032-token save_on_cpu comparison (2026-10-10)

Private kernel `ccjjllt/p005-save-on-cpu-6032` v1. Same complete tx `fastapi_11194-3f51fb716edd-tx01` (6032 tokens). Two fresh DDP workers. No 16K retry. Capacity adapter discarded.

**CPU (tiny Gemma, torch 2.14.1):** loss 3.7964298725128174 both arms; max LoRA grad |Δ| 8.94e-8 (20 tensors). Default `save_on_cpu` remains off.

**Baseline:** CUDA OOM in **forward**, last decoder **layer 1/60** MLP `gate_proj` NF4 dequant (`F.linear` 248 MiB). Load 17.210 GiB; failure allocated 21.548 GiB. Optimizer not reached.

**save_on_cpu (new process, flag on):** forward ran ~49 s; nvidia-smi 20.75 GiB; then host **SIGKILL**. No Python OOM report. Optimizer not reached.

**Not claimed:** 3.62 GiB saved-hidden estimate. Live layer-1 MLP workspace is the measured limiter. First Adam step did not complete → Task 2 BLOCKED.

Quota Δ 1063.808 s (0.296 h). Details: `result.json`, `kaggle-output/`.
