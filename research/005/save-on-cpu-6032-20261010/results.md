# 6032-token save_on_cpu comparison — reviewed 2026-10-11

**The v1 comparison is invalid for the intended checkpointed training recipe.** The worker omitted `model.train()`: the loaded backbone/decoders remained in eval mode and decoder checkpointing did not execute. CPU reproduction shows zero decoder checkpoint calls before the fix and two after it on a two-layer Gemma. See [review](review-20261011.json).

Private kernel `ccjjllt/p005-save-on-cpu-6032` v1 used the same complete 6032-token select transaction in two fresh DDP workers. The original Kaggle outputs remain unchanged; pre-review source and summaries are recoverable at `3cde33a`.

**Observed baseline:** CUDA OOM in forward at decoder index 1/60 MLP `gate_proj`, in the NF4 fallback's `F.linear`, requesting 248 MiB. Load 17.210 GiB; failure allocated 21.548 GiB. These are observations with inactive decoder checkpointing, not proof of a live-workspace bottleneck in normal training.

**Observed save_on_cpu:** forward ran about 49 seconds; sampled GPU usage reached 20.75 GiB, then rank 2 received SIGKILL. No Python OOM report. Host OOM or pinned-memory exhaustion is unverified without RSS/cgroup evidence.

Neither arm completed the first Adam step; no long-sequence run, SFT adapter or repair-gain evaluation followed. Quota snapshot delta was 1063.808 seconds (0.296 hours).

The worker now enables training explicitly and records/validates decoder checkpoint prerequisites. CPU loss/gradient comparison covers checkpointing off and on, with actual checkpoint calls observed. The corrected carrier has **not** run on Kaggle. The next experiment is the corrected 6032-token comparison, not a larger-window retry.
