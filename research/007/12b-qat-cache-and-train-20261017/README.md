# P007 12B QAT NF4 Cache, Real-Sample SFT, and Official 12B CT v2 Probe

## 1. Free CPU NF4 Cache (`ccjjllt/p007-12b-cpu-nf4-cache`) — `COMPLETE`
- **Source**: `google/gemma-4-12B-it-qat-q4_0-unquantized` @ `b6ed86275a6a5735884e208bfed95b445a684ca2` (`config.json` SHA256 `a323d02f...`, `model.safetensors` 23,919,549,408 bytes, SHA256 `26f2cee4...`).
- **Deployment CT v2 Assets**: `google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2` (`chat_template.jinja` SHA256 `36e3a42e...`, `tokenizer.json` SHA256 `cc8d3a0c...`).
- **Output**: `/kaggle/working/qat-12b-nf4/` (7,710,759,119 bytes, 331 `Params4bit` quantized tensors across 4 `<2GB` shards + co-located CT v2 tokenizer/template + `cache-manifest.json`), plus 6 compatible Python 3.12 training wheels in `/kaggle/working/training-wheels/`.
- **Wall Time**: `721.02s` on CPU (`0s` GPU quota).

## 2. 12B Real-Sample Training & Official 12B CT v2 Action/Distribution Probe (`ccjjllt/p007-12b-train-and-ct-probe`)
- **Generator**: `build_12b_gpu_probe.py`
- **CPU Preparation (`ccjjllt/p007-12b-train-and-ct-probe-cpu-prepare`)**: Verifies bundle unpacking, wheelhouse v31 + PEFT/Accelerate installation, `train_sft.py --dry-run`, `validate_pinned_model_source`, `validate_nf4_cache`, and `validate_tokenizer_protocol` without GPU allocation.
- **GPU Execution (`probe_12b.py`, two sequential isolated child processes)**:
  1. **Phase 1 (NF4 Training + Independent PEFT Export + Zero-Delta Control)**:
     - Runs `training/train_sft.py` on the frozen `click` train sample (`encoded-12b-train-capacity.jsonl`, 3,368 input tokens, 519 supervised tokens).
     - Exports `click_r8_step1` via `training/export_peft.py` and builds a matched zero-delta control adapter `zero_r8_ctrl` (identical config, keys, shapes, and `lora_A`, with `lora_B = 0`).
     - Evaluates `torch.no_grad()` NF4 selected-token loss under `zero_r8_ctrl` vs `click_r8_step1` on both `click` (train) and `pydicom` (select diagnostic, never updated).
  2. **Phase 2 (Official 12B CT v2 Serving + Base / Zero-Ctrl / Real Adapter Comparison + Tool Execution)**:
     - Launches official `VllmServer` on `google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2` with `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` and both `click_r8_step1` and `zero_r8_ctrl` registered.
     - Compares exact teacher-forced log-probabilities (`prompt_logprobs=1`) across all 519 supervised tokens of `click` and all 226 supervised tokens of `pydicom` under Base vs `zero_r8_ctrl` vs `click_r8_step1`.
     - Compares live `/v1/chat/completions` (`temperature=0.0`, `top_logprobs=5`) and executes the generated tool calls against the real buggy `click` and `pydicom` source files.
