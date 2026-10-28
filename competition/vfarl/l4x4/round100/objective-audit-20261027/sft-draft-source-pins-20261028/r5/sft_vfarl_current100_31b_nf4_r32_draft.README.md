# Current100 31B SFT draft — not run

The adjacent flat YAML uses the existing single/FSDP SFT entry points. It selects official31B QAT-derived NF4, r32/alpha64/dropout0, seven text projection types, seed0 and explicit adapter_init_seed0. It does not name a model directory: a future authorized run must supply --model with a complete existing official cache. The current local metadata-only cache is insufficient. No source guard is relaxed.

The data/protocol paths explicitly reference question-revision-r3/supervision-r5/trainer-input-r5. Encoded SHA is a5dea1aecd85bf8478d29cd1ebf2588df498a3baa8c0ee4b7d20eaa43f20d704: 805 rows, 78423 shift targets, 2057832 input tokens, maximum 4649; 19 rows from 7 tasks exceed 4K. The child manifest SHA is fc1c4260ec53277b25b51e96fdecd318f70a36a805197e5fca3a4a28f55c91c3; the parent manifest SHA is 277fbf08d90d41544f33fb5b18c002d497d15097d11db81dc81f78b4662ee812. R5 contains 51 fresh teacher sources / 413 decisions and 49 historical sources / 392 decisions. tokenizer, CT and deployment-tools are the exact exported configuration; do not substitute old harness_tools or the VFARL collector schema. The 8192 limit is data admission, not measured 31B/r32 backward capacity. max_steps=1, LR=1e-6 and warmup=0 remain an unrun diagnostic proposal. With micro_batch_size=1 and gradient_accumulation_steps=1, that one step would consume one row; it would not complete training on 100 tasks or establish an optimal learning rate or capacity.

Optional adapter_init_seed is absent from DEFAULTS and omitted old configs. Existing seed behavior and old config identities therefore remain unchanged. When explicitly set, it must be an integer>=0; the single/DDP/FSDP SFT entry points pass it to the existing loader's CPU PEFT initialization branch. Resume/init-run still loads the saved adapter and does not recreate A/B. The new key is recorded by existing full-config metadata/signatures. VFARL read_config explicitly rejects model.adapter_init_seed and directs callers to existing model.seed, so this field cannot be silently ignored by RL. DDP receives the optional field too; this plumbing does not recommend31B DDP capacity. This controls fresh-phase initialization; it is not a VFARL saved-actor warm-start or an Unsloth SFT execution bridge.

Manual CPU initialization evidence is competition/vfarl/l4x4/round100/objective-audit-20261027/sft-adapter-init-seed-20261028.json. Actual tiny Gemma4ForConditionalGeneration plus PEFT was saved/reloaded through the existing production loader: seed0 gave identical adapter hashes with0 versus257 prior RNG draws; old unseeded controls differed. Loading a saved tiny adapter while requesting another seed preserved its hash. All old SFT config validation outputs were identical, invalid optional seeds were rejected, the single/FSDP call sites were inspected, and that earlier single CLI --dry-run consumed the then-current R4 data. The R5 routing-only CLI result is competition/vfarl/l4x4/round100/objective-audit-20261027/sft-draft-r5-dry-run-20261028.json: 805 rows / 78423 targets, weights_loaded=false and dependencies_imported=false. This dry-run reads encoded data and checks protocol_id agreement; it does not repeat the R5 producer's tokenizer/template/tools fingerprint reconstruction. No model, optimizer, 31B, NF4, GPU or oracle ran for the R5 routing update.

CPU-only plan inspection already exercised:

```sh
python -m training.train_sft \
  --config training/configs/sft_vfarl_current100_31b_nf4_r32_draft.yaml \
  --data competition/vfarl/l4x4/round100/question-revision-r3/supervision-r5/trainer-input-r5/encoded.jsonl \
  --dry-run
```

Actual training remains unexecuted and requires separate authorization plus the complete cache, an explicit fresh output path and measured capacity.
