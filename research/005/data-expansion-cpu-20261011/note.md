# CPU independent-issue expansion (2026-10-11)

No GPU. No new rollouts. v2 census files not overwritten. Reserved eight IDs and same repo+commit stay out. Protocol `p005-harness-tools-v1`.

**Independent issues cannot grow without new rollouts.** Walked 92 `trace_*.json` files under `results/` (20261010 inventory: 92 traces, 9 eligible issues). New files by sha256: **0**. New independent non-reserved issues versus frozen v2: **none**. New recovery events measured: **0**.

Did not re-prepare the old 214 complete prefixes; no new eligible protocol-compatible traces appeared.

Eligible non-reserved issue ids in splits that still have **no** unused protocol-compatible trace:

- `fastapi_11194`
- `fastapi_14873`
- `fastapi_5077`
- `fastapi_9425`
- `rich_2725`
- `rich_3518`
- `rich_3777`
- `rich_3942`
- `rich_4075`

84 other non-reserved, non-heldout split issues have zero protocol-compatible traces (not P002-eligible; listed in `window_report.json`). Do not invent them into train.

Independent issues cannot grow without new rollouts. No new eligible non-reserved protocol-compatible issue id appeared beyond the historical v2 issue set. Repeated shorter prefixes of the same 9 issues are not independent issues. Reserved issues and snapshots remain excluded.
