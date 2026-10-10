# Train-mode SFT candidate — BLOCKED

Date: 2026-10-11. No GPU. `gpu_launched=false`. `gpu_sft=false`.

SFT is gated on (a) a complete **train** length backprop with first Adam and (b) independent non-reserved samples. Both false.

- Train-mode 6032 kernel `ccjjllt/p005-save-on-cpu-6032-train-mode` (metadata id `ccjjllt/p005-save-on-cpu-6032-tm`) v1: valid checkpointed comparison, no first Adam.
- 9841/12832 not launched.
- Task 2: 0 new independent issues, 0 new recovery events; corpus cannot grow without new rollouts.
- v2 census `trainable_ok=false` (0/11 train transactions fit 8192).

Machine-readable: [`blocked.json`](blocked.json). Do not truncate. Do not treat `ccjjllt/p005-training-ddp-probe` as a repair-gain arm. Do not change `training/configs/sft_qat_nf4_ddp.yaml` `max_sequence_length` (still 1024).
