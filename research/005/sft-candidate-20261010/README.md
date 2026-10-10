# Task 4 SFT candidate — BLOCKED

Date: 2026-10-10. No GPU. No 31B load. `gpu_launched=false`.

Task 2 `trainable_ok=false` (`chosen_max_seq_len=8192`). Train window counts: **0/15 fit 4096**, **3/15 fit 8192**, majority exceed 8192. Census: [`research/005/length-census-20261010/length_report.json`](../length-census-20261010/length_report.json). Machine-readable block: [`blocked.json`](blocked.json).

This directory records the block so later work does not “just train anyway”:

- Do not truncate history or tool JSON to fake an 8192-fit corpus.
- Do not train on the 3 short prefixes as if they represented the repair distribution.
- Do not treat diagnostic synthetic 2K DDP success as a window choice.
- Do not change `training/configs/sft_qat_nf4_ddp.yaml` `max_sequence_length` (still **1024**) as if 8192 SFT ran.

Unblock only when either:

1. New complete-context samples fit the chosen window (8192) without truncation, with `trainable_ok=true`; or
2. A measured 4K/16K training-capacity study exists (16384 is a census bucket, not a selected training window).

Loss on a diagnostic adapter ≠ repair gain. Task 5 pairing is skipped without a real-data SFT adapter.
