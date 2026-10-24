# R5 replacement screening, 2026-10-24

Three fixed public train candidates were inspected against complete SWE-smith-py task rows. All three are excluded as clean R5 replacements. Native train source rows were SHA-256/byte checked against `smith-next-1000-acquisition.json`; only their initial user prompts were read. Public task `/rows` responses verified the pinned SHA, row index and instance ID. No source commands or replay ran.

| Candidate | F2P / P2P | Visible original required parameters | Decision |
| --- | ---: | --- | --- |
| Moto PR 5177 | 7 / 0 | No parameter suffixes | Broad grants/signing/MAC deletion and wrong imports; native Grant ImportError claim does not match retained Grant class. Official task is grants feature request. |
| Marshmallow PR 2768 | 1188 / 0 | 635 parameterized IDs | Deletes pprint implementation; not a real lazy/optional/import-order defect. |
| Result PR 116 | 49 / 0 | No parameter suffixes; 4 type-checking YAML IDs | Deletes async decorator implementation and exports; not a reexport failure with implemented lower symbol. |

Full task paths, mirror/image identities, exact branch SHAs and causal notes are in `r5-three-candidate-review.json`; acquisition receipts are in `r5-three-candidate-receipts.json`. Every original required node ID remains visible in its full task row. No selectors were dropped or rewritten and no optional skip evidence was asserted.

A follow-on source search was explicitly authorized across two cached acquisition manifests. It admitted only rows with Smith trajectories revision `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`, physical split `tool` and current purpose `train`. It read 2103 permitted initial user prompts after byte/hash validation, skipped 3 pilot96 ID duplicates, and found 55 broad import-related hits. `artifacts/r5-native-prompt-search.json` records source identities and initial prompts; neither trajectory execution nor final patch fields were read.

Best follow-on source leads, awaiting complete task diff review:

- `spulec__freezegun.5f171db0.func_pm_op_swap__9ad3btja`: modules imported during freeze_time keep FakeDatetime after context exit. A genuine import-timing and runtime module restoration trigger is stated.
- `pylint-dev__astroid.b114f6b5.pr_2410`: preimporting six.moves changes StringIO inference, with sys.modules/namespace/failed-import-hook interaction described. Exact native cached bug commit `61d3f02fb40f20013aae3c8168e536fb2df32c04`.

These are leads, not admitted training tasks or claims of validated benefit.
