# CPU data expansion — naturally shorter complete txs (2026-10-11)

No GPU. v2 census files not overwritten. Reserved eight IDs and same repo+commit stay out. Recompute: `/tmp/p005-training-check/bin/python scan.py --results <main-checkout>/results`.

**Independent issues cannot grow without new rollouts.** 92 exposed traces cover 27 issue IDs; prepare still yields the same 39 traces / 214 complete prefixes / **9 eligible issues**. Extra traces are the reserved pairing eight, ineligible panel/dev_future tasks, or missing `result.json`. None of those are invented into train.

**All 214 complete prefixes** (full causal history, no truncation/JSON clip):

| window | txs | issues | recovery events |
|---|---|---|---|
| 2048 | 0 | 0 | 0 |
| 4096 | 0 | 0 | 0 |
| 8192 | 24 | 7 | 0 |
| 16384 | 98 | 9 | 8 (of 10 in the full pool) |

Pool quantiles: p0=5235, p50=17878, p100=50702. Behaviors: 78 local_edit / 7 tool_recovery / 11 verify_close / 118 other.

**Naturally shorter complete txs exist, but they are earlier prefixes of the same 9 issues**, not a new corpus. 21/24 ≤8192 rows are `tx01`. Labeled ≤8192: **6 local_edit, 3 issues, 0 recoveries**. Five of those six were not in the v2 capped sample (v2 only kept `fastapi_11194-3f51fb716edd-tx01` at 6032). Do not treat the 18 unlabeled short prefixes, or these early edits, as the repair distribution. v2 train still has **0** txs ≤8192.

Lists: `complete_tx_lengths.json`, `short_labeled.json`, `window_report.json`.
