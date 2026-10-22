# P015 on-policy action RFT curation

This is action RFT from actual 31B rollout requests, not full-task SFT. Raw v1
captures remain immutable. The final assistant has separate reasoning and visible
text; the current v3 encoder does not support that terminal form, so no terminal
text is rewritten or synthesized.

`curate_on_policy_actions.py` accepts explicit `--attempt` paths (repeat for each
train draw), optional `--capture-audit`, and `--output`. It checks frozen train
identity and original base before opening trace bodies. For each issue it chooses
the resolved, judge-valid draw with the minimum recorded `tool_calls_used`, breaking
ties by path. This choice precedes action-length curation; another draw is not
substituted because its windows fit better. Actual raw function-call counts remain
in the source captures (the v1 boltons budget counter is21, raw calls22).

The underlying converter checks each current response against raw model content,
function arguments/IDs and actual ADK execution observations. Each example retains
that actual request's complete history plus its current complete action and real
observations. An auxiliary response, changed unsupported schema, incomplete
response, or transition where the next history was compacted excludes only that
window, with its reason recorded. No preceding uncompressed history is spliced
into a later request. Original episode capture flags and any explicit audit remain
separate provenance, never silently rewritten.

First-round curation uses thought-mask,8192 tokens, at most6 targets per issue.
Priority is successful edit of an original production file retained in the final
resolved patch, actual original upstream test invocation, nonempty submit, relevant
source read, then status. Original snapshot membership and final patch paths bound
production/source classifications. Shell-only edits and ambiguous validation
commands are conservatively left unselected rather than guessed. Original upstream
pytest/unittest assertion exits0 or1 may count as real validation; interruption,
usage errors and no-test collection do not. Nonzero test exits are not uniformly
classified as bad actions.

No write_file action is selected. New/repro script writes and commands invoking or
handling them remain context only. Each admitted issue must have an in-cap successful
production edit. Over-cap windows are rejected whole. There is no context clipping,
old-assistant loss, tool-result loss or terminal-format approximation.

Initial real example: mahmoud__boltons-298 chooses call indices0,1,8,9,11 (source
read) and13 (production edit),246 supervised tokens. The edit's full window is8165
tokens. The first post-edit command is8286 tokens and runs the new repro script;
submit is11560 tokens. Neither is forced into the8K dataset. All original22 actions
remain available in the immutable captures and complete-window diagnostic material.

Run with the existing tokenizer environment (no installation or GPU):

```sh
/private/tmp/p009-replay-env/bin/python research/015/data-preparation/curate_on_policy_actions.py \
  --attempt research/015/inference/output-v1/run-015-31b-base-and-train-sampling-v1-20261022T050436Z-7bae0a25/attempts/03-mahmoud__boltons-298-base \
  --capture-audit research/015/inference/evidence-v1/boltons298-consumer-close.json \
  --output research/015/data/on-policy-boltons298-v1-curated-final
```

For new8 draw results, provide their train attempt paths using repeated `--attempt`;
no dev path is authorized for this entry. The output `dataset-manifest.json` contains
`encoded_data_sha256`, all per-action decisions/hashes, chosen draw records and the
actual tools file. Training must use that tools file, not the older p005 schema.
This curation does not initiate training or claim further repair gains.
