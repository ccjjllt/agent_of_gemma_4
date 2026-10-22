# P015 FSDP-QLoRA engineering, 2026-10-22

Implementation only; no 31B GPU result or CT adapter compatibility claim.
GPU scheduling belongs to the root agent. Existing 2048-token synthetic updates
and the failed 6032-token select diagnostics remain the historical model results.

`training.train_fsdp` loads each rank's existing validated official-QAT-derived
NF4 cache, then shards. Initial loading still replicates ~17 GiB per GPU; this
implementation does not promise CPU-RAM-efficient/meta initialization.

`pack_floating_nf4_storage` reinterprets contiguous uint8 bytes as BF16, restoring
packed `[elements,1]` layout. It never casts their numerical values, dequantizes,
or requantizes them. Module and Params4bit storage metadata change together;
original QuantState objects and scale/code tensors remain unchanged.

PEFT's transformer/trainable-leaf wrapping policy is combined with separate
frozen FP32 leaf units for norms. `use_orig_params=False`; BF16 frozen base,
FP32 adapters and FP32 norms consequently do not share mixed-dtype flat parameters.
FSDP parameter/buffer casting is disabled to protect packed bytes. Computation
uses BF16 autocast, while adapter gradients reduce in FP32. Every microbatch
synchronizes gradients: FSDP `no_sync` would retain unsharded gradients.

Loss and deployment/source validation reuse the existing trainer. Real updates
now require explicit `meta.group.split=train`; dev/select/reserved/unknown are
rejected even if mislabeled `training_eligible`. The epoch objective normalizes
over global selected next-token labels, including incomplete final batches;
empty ranks join collectives with zero-weight duplicated context.

Export gathers each trainable LoRA FSDP unit across all ranks in identical order,
retaining complete CPU tensors only on rank0. It never assembles the frozen base.
PEFT 0.21.2's save_pretrained structural filtering was observed to silently save
an empty adapter for a nested FSDP model despite a full clean supplied state.
The implementation therefore writes standard bias-free default LoRA keys
explicitly with safetensors and saves the independent PEFT config. Key count,
original full shapes, and finite values must match. Checkpoints are adapter-only;
optimizer resume is explicitly unsupported rather than pretending a rank0
optimizer snapshot is complete.

Manual CPU evidence (temporary commands only, no test files):

- Python 3.11, torch 2.8.0 CPU, transformers 5.13.1, peft 0.21.2,
  bitsandbytes 0.50.2.
- 2048-byte initialized NF4 Params4bit: BF16 view retained exact storage pointer;
  byte roundtrip was identical; original QuantState object identity remained.
- Actual bitsandbytes CPU NF4 dequantization before/after repacking returned
  bit-identical BF16 `[64,64]` tensors.
- Two independent local Gloo ranks, tiny one-layer Llama with BF16 frozen base,
  FP32 frozen norms and FP32 LoRA: one optimizer step completed. Rank losses
  3.44497179985 / 3.47000479698, shared grad norm 0.237117603421.
  Four full 2D adapter tensors saved, reloaded with PeftModel.from_pretrained,
  and compared exactly equal to saved tensors. This is a serialization/wrap
  mechanism check, not evidence of Gemma4 31B capacity or service loading.
- Local macOS fork-based execution segfaulted; independent interpreter processes
  completed. Initial sandbox loopback resolution failed; local-loopback execution
  outside sandbox succeeded. No GPU was started.

The plain-assistant ending mask changed to `p015-full-transaction-v3`. It labels
only the `<turn|>` already emitted by the complete deployed template for selected
plain assistant messages. Tool actions retain their existing `<tool_call|>`
supervision and acquire no forced EOS. Official 31B CT v2 tokenizer manually
rendered `Done.<turn|>` as intended; a masked failed action remained unlabelled.
Data producer re-encoded its replayed transactions offline. Those are edit/action
capacity samples, not complete autonomous task trajectories or resolved claims.

Sources inspected for implementation:
https://huggingface.co/docs/bitsandbytes/main/fsdp_qlora
https://huggingface.co/docs/peft/main/accelerate/fsdp
https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/0.50.2/bitsandbytes/nn/modules.py
Locally installed peft 0.21.2 `tuners_utils.py` / `save_and_load.py` / `peft_model.py`.


Frozen GPU carrier (not launched by this agent):
`fsdp-kernel/run.py`, 107659 bytes, SHA256
`64b00848208fd9f630e19addffb162d5aeeffcf0355805eccaf127123c78ffca`.
The builder embeds nine explicit-train local edit/action transactions from the
v3 dataset, including the real 2686-token edit recovery. It plans two four-issue
updates containing the 4203-token issue, then one complete 6846-token tail issue
update. Zero-weight duplicated rows let empty ranks join FSDP collectives without
adding repeated supervision. Checkpoints export after every optimizer step;
if the third update fails, the first two adapters remain. The 6846 setting is a
maximum input limit, not a claim of executed capacity. Grouped attention is off.
Dependencies, extracted package sources and input JSONL live in `/tmp`; only
small metadata, metrics and adapters live under `/kaggle/working/p015-fsdp`.
Platform bound 2400 seconds, worker 2100 seconds; maximum platform estimate at 2x
is 1.3333 quota hours. Root owns launch and records actual time/failure/result.


## Actual GPU result received, 2026-10-22

The frozen four-L4 carrier completed normally (platform COMPLETE). Its real
training manifest reports three optimizer updates over nine train-only local
edit/action transactions, 32,324 effective input tokens and 8,649 selected tokens.
Step1 included4203 tokens; step3 completed the whole6846-token transaction.
This is capacity/action evidence, not autonomous task SFT or repair gain.
Grouped attention was disabled. The step3 tail had rank counts2315/0/0/0;
rank0 loss was multiplied by4 before FSDP rank averaging, so only one sample
entered the objective. All ranks physically computed6846 tokens for collectives.

Across ranks, the actual post-initialization loop peaks were
16.380761GiB allocated and
21.175781GiB reserved. These replace the earlier
13.1GB pre-backward snapshot as peak evidence. The loop plus adapter exports
took149.15s; model loading/setup and platform quota are separate. Different
batches produced losses0.25999/0.32189/0.22169, which do not establish a
same-example fitting trend.

The GPU inspector recorded600 F32 LoRA tensors,53,043,200 parameters,300 A/B
pairs over all60 text layers. Every reported shape matches both the original
unwrapped adapter shape map and the independent official text architecture.
Final safetensors size212,261,592bytes; recorded SHA256
`f1d7cfe7479a26b91d2f29b1bba45191368c4575f7b22be42f3b268963173563`.
The frozen producer checked every gathered tensor for finite values before save;
those checks passed on the successful GPU run. The root subsequently completed a local payload rehash and full F32 value scan: all 600 tensors / 53,043,200 parameters are finite and all 600 tensors are nonzero. The SHA256 exactly matches the GPU manifest; see adapter-local-inspection.json. The actual training base remains the official
QAT-derived NF4 cache, correctly preserved as adapter base_model_name_or_path.
Official31B CTv2 template/tokenizer were checked, but this does not establish
CT service load/request routing or any repair benefit. See fsdp-result-summary.json.


## Subsequent official CT v2 service smoke, 2026-10-22

The next base-and-train-sampling job mounted the exact final two adapter files
(SHA256s identical to the GPU producer and local scan) using official discovery
and VllmServer. The command explicitly served official31B CT v2 with TP4,
LoRA enabled, one adapter and max LoRA rank8. Its mount was the independent
/tmp diagnostic path recorded in the immutable runtime artifact. One hello
request for model p015_capacity_diag returned HTTP200 with response.model
p015_capacity_diag in4.109654 seconds,8 completion tokens. Its finish_reason
was length because the deliberate smoke limit was8; it is not a training target.
This establishes adapter loading and routed completion in that service. It does
not establish tool-call behavior, autonomous repair quality or LoRA gain.

Base attempts still used the original CT model path with no adapter selection.
All23 wire responses in the successful train boltons298 attempt name that base
path. The actual service nevertheless had LoRA enabled and the diagnostic
adapter mounted: future weight A/B must match those service conditions.
Original training-producer manifest runtime_verified=False is preserved as
historical training-time metadata; fsdp-result-summary.json now includes the
subsequent independent CT runtime evidence and actual serving conditions.
