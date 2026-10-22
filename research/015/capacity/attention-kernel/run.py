"""P015 GPU operator experiment: real Gemma masks/strides, no model weights.

Run as a Kaggle script with the existing p005-training-cpu-stage output attached.
Only GPU 0 is used. No task text, reference fix, tokenizer or model weights read.
The parent process bounds the one experiment to 600 seconds after installation.
"""
from pathlib import Path
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

OUT = Path('/kaggle/working/p015-attention')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker():
    global torch, F
    import importlib.metadata as md
    import torch
    import torch.nn.functional as F
    from transformers import Gemma4TextConfig
    from transformers.masking_utils import create_causal_mask
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    # Replaced at packaging time from the existing official-QAT cache config.
    config = Gemma4TextConfig.from_dict(CONFIG)
    config._attn_implementation = 'sdpa'
    assert config.num_attention_heads == 32 and config.num_global_key_value_heads == 4
    assert config.global_head_dim == 512 and config.attention_dropout == 0
    assert config.num_kv_shared_layers == 0
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The experiment requires BF16 CUDA')
    dtype, device, length = torch.bfloat16, 'cuda:0', 6032
    report = {'purpose': 'operator-capacity-mechanism; not training or repair gain',
              'config_sha256': CONFIG_SHA256, 'seed': 15022, 'length': length,
              'versions': {x: md.version(x) for x in ['torch', 'transformers']},
              'device': torch.cuda.get_device_name(0), 'arms': []}
    # Exactly the all-valid padded-row path used by Gemma4TextModel.forward.
    embedding_shape = torch.empty((1, length, config.hidden_size), dtype=dtype, device=device)
    padding = torch.ones((1, length), dtype=torch.long, device=device)
    positions = torch.arange(length, device=device).unsqueeze(0)
    mask = create_causal_mask(config=config, inputs_embeds=embedding_shape,
                             attention_mask=padding, past_key_values=None,
                             position_ids=positions)
    report['mask'] = None if mask is None else {'shape': list(mask.shape),
        'dtype': str(mask.dtype), 'stride': list(mask.stride())}
    del embedding_shape, padding, positions
    module = torch.nn.Module()
    module.config = config
    module.is_causal, module.is_sliding = True, False
    module.head_dim, module.num_key_value_groups, module.scaling = 512, 8, 1.0
    module.train()
    # BLHD contiguous buffers followed by H/L transpose reproduce HF strides.
    rng = torch.Generator(device='cpu').manual_seed(report['seed'])
    originals = [torch.randn((1, length, h, 512), generator=rng, dtype=dtype)
                 for h in (32, 4, 4)]
    upstream = torch.randn((1, length, 32, 512), generator=rng, dtype=dtype)
    reference = None
    for arm in ['hf_repeat_kv', 'grouped_kv']:
        tensors = output = None
        gc.collect()
        torch.cuda.empty_cache()
        item = {'arm': arm, 'phase': 'inputs'}
        report['arms'].append(item)
        write_json(OUT / 'result.json', report)
        try:
            leaves = [x.to(device).requires_grad_() for x in originals]
            tensors = [x.transpose(1, 2) for x in leaves]
            grad = upstream.to(device)
            item['input_strides'] = [list(x.stride()) for x in tensors]
            torch.cuda.synchronize()
            item['input_allocated_bytes'] = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            item['phase'] = 'forward'
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                started = time.perf_counter()
                if arm == 'hf_repeat_kv':
                    output = sdpa_attention_forward(module, *tensors, attention_mask=mask,
                                                    dropout=0.0, scaling=1.0)[0]
                else:
                    output = grouped_attention(module, *tensors, attention_mask=mask)[0]
                torch.cuda.synchronize()
                item['forward_seconds'] = time.perf_counter() - started
                item['forward_allocated_bytes'] = torch.cuda.memory_allocated()
                item['forward_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                item['forward_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                torch.cuda.reset_peak_memory_stats()
                item['phase'] = 'backward'
                started = time.perf_counter()
                output.backward(grad)
                torch.cuda.synchronize()
                item['backward_seconds'] = time.perf_counter() - started
                item['backward_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                item['backward_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            item['operators'] = sorted({e.key for e in prof.key_averages()
                if any(term in e.key.lower() for term in ['attention', 'bmm', 'softmax', 'clone', 'contiguous'])})
            values = [output.detach().cpu()] + [x.grad.detach().cpu() for x in leaves]
            item['finite'] = all(bool(x.isfinite().all()) for x in values)
            if not item['finite']:
                raise RuntimeError('nonfinite output or gradient')
            if reference is None:
                reference = values
            else:
                item['differences'] = []
                for name, actual, expected in zip(['output', 'q_grad', 'k_grad', 'v_grad'], values, reference):
                    diff = (actual.float() - expected.float()).abs()
                    item['differences'].append({'name': name, 'max_abs': diff.max().item(),
                        'mean_abs': diff.mean().item(), 'reference_abs_max': expected.abs().max().item(),
                        'relative_l2': ((actual.float()-expected.float()).norm() /
                                        expected.float().norm().clamp_min(1e-12)).item()})
                del values
            item['phase'], item['status'] = 'complete', 'completed'
        except Exception as exc:
            item['status'], item['error'] = 'failed', repr(exc)
            item['traceback'] = traceback.format_exc()
        finally:
            # Each arm releases every CUDA graph before the next memory measurement.
            output = tensors = leaves = grad = prof = None
            gc.collect()
            torch.cuda.empty_cache()
            write_json(OUT / 'result.json', report)
            print(json.dumps(item, allow_nan=False), flush=True)
    report['complete'] = all(x.get('status') == 'completed' for x in report['arms'])
    write_json(OUT / 'result.json', report)
    return 0 if report['complete'] else 2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reports = list(Path('/kaggle/input').rglob('cpu-stage-report.json'))
    if len(reports) != 1:
        raise RuntimeError('Attach exactly one existing p005-training-cpu-stage output')
    stage = json.loads(reports[0].read_text())
    if stage.get('success') is not True:
        raise RuntimeError('The staged dependency run failed')
    wheels = []
    # Do not replace the CUDA torch stack; use only staged non-torch packages.
    for item in stage['wheels']:
        name = item['name']
        if name.lower().startswith(('torch-', 'triton-', 'nvidia_')):
            raise RuntimeError('Unexpected CUDA-stack wheel')
        # CPU stage used Python 3.13. Its only non-abi3 CPython wheel is
        # PyYAML, which the GPU image already provides; do not install it
        # into a different Python minor version.
        if name.lower().startswith('pyyaml-'):
            continue
        path = reports[0].parent / 'training-wheels' / name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise RuntimeError('Dependency identity differs: ' + name)
        wheels.append(str(path))
    deps = OUT / 'deps'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--target', str(deps), *wheels], check=True, timeout=100)
    env = dict(os.environ, PYTHONPATH=str(deps), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
    proc = subprocess.run([sys.executable, __file__, '--worker'], env=env, timeout=600)
    return proc.returncode


def grouped_global_sdpa(query, key, value, *, attention_mask=None,
                        is_causal=False, scaling=1.0, dropout=0.0):
    """Compute four independent head groups; return [B, 32, L, 512].

    Supported: dense Gemma4 global attention, [B,32,L,512] query and
    [B,4,S,512] key/value, identical floating dtype/device, zero dropout.
    Masks are boolean (True means allowed) or same-dtype additive masks with
    shape [L,S] or [B|1,32|1,L|1,S|1]. Full-head masks are sliced with the query
    heads. Non-square causal alignment follows PyTorch's upper-left convention.

    Dropout is rejected: splitting calls changes random-number consumption and
    cannot promise the baseline mask for a given seed. Packed/ragged input,
    multimodal masks, attention scores and custom tensor subclasses are outside
    this small candidate's intended scope. Gemma4 uses scaling=1.0, not 1/sqrt(D).
    """
    if dropout != 0.0:
        raise ValueError("grouped global SDPA requires zero attention dropout")
    if not isinstance(is_causal, bool):
        raise ValueError("is_causal must be an explicit bool")
    if any(not isinstance(x, torch.Tensor) or x.ndim != 4 for x in (query, key, value)):
        raise ValueError("query/key/value must be dense four-dimensional tensors")
    batch, heads, length, dim = query.shape
    if heads != 32 or dim != 512 or batch < 1 or length < 1:
        raise ValueError("expected Gemma4 global query shape [B,32,L,512]")
    if key.shape != value.shape or key.shape[:2] != (batch, 4) or key.shape[-1] != 512 or key.shape[-2] < 1:
        raise ValueError("expected matching key/value shape [B,4,S,512]")
    if not query.is_floating_point() or any(x.dtype != query.dtype or x.device != query.device for x in (key, value)):
        raise ValueError("query/key/value must share floating dtype and device")
    if attention_mask is not None:
        if is_causal:
            raise ValueError("pass the complete mask or is_causal, not both")
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.device != query.device:
            raise ValueError("attention mask must be a tensor on the query device")
        if attention_mask.dtype not in (torch.bool, query.dtype):
            raise ValueError("attention mask must be boolean or match the query dtype")
        source_length = key.shape[-2]
        shape = attention_mask.shape
        if attention_mask.ndim == 2:
            valid = shape[0] in (1, length) and shape[1] in (1, source_length)
        elif attention_mask.ndim == 4:
            valid = (shape[0] in (1, batch) and shape[1] in (1, 32)
                     and shape[2] in (1, length) and shape[3] in (1, source_length))
        else:
            valid = False
        if not valid:
            raise ValueError("unsupported mask layout; expected 2D or broadcastable 4D mask")

    outputs = []
    for group in range(4):
        start, stop = group * 8, (group + 1) * 8
        mask = attention_mask
        if mask is not None and mask.ndim == 4 and mask.shape[1] == 32:
            mask = mask[:, start:stop]
        # The head stride is zero. Do not add contiguous()/repeat_interleave()
        # here: materializing these views would defeat the intended mechanism.
        group_key = key[:, group:group + 1].expand(-1, 8, -1, -1)
        group_value = value[:, group:group + 1].expand(-1, 8, -1, -1)
        outputs.append(F.scaled_dot_product_attention(
            query[:, start:stop], group_key, group_value,
            attn_mask=mask, dropout_p=0.0, is_causal=is_causal,
            scale=scaling, enable_gqa=False))
    # This concatenation temporarily coexists with the four output tensors.
    # Grouping does not remove output storage or guarantee a lower peak.
    return torch.cat(outputs, dim=1)

def gemma4_grouped_global_attention_forward(module, query, key, value,
                                          attention_mask, dropout=0.0,
                                          scaling=None, is_causal=None, **kwargs):
    """HF 5.13.1 attention-interface adapter, global layers only."""
    if getattr(module, "is_sliding", True) or getattr(module, "head_dim", None) != 512:
        raise ValueError("grouped interface is only for Gemma4 global head_dim=512")
    if kwargs.get("output_attentions"):
        raise ValueError("grouped interface returns no attention score matrix")
    if kwargs.get("sliding_window") is not None:
        raise ValueError("global grouped interface does not implement sliding windows")
    for name in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
        if kwargs.get(name) is not None:
            raise ValueError("packed attention is unsupported")
    causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
    # Match HF sdpa_attention_forward's mask/causal dispatch exactly.
    causal = bool(query.shape[2] > 1 and attention_mask is None and causal)
    scale = module.scaling if scaling is None else scaling
    output = grouped_global_sdpa(
        query, key, value, attention_mask=attention_mask, is_causal=causal,
        scaling=scale, dropout=dropout)
    return output.transpose(1, 2).contiguous(), None

grouped_attention = gemma4_grouped_global_attention_forward
CONFIG = {'attention_bias': False, 'attention_dropout': 0.0, 'attention_k_eq_v': True, 'bos_token_id': 2, 'dtype': 'bfloat16', 'enable_moe_block': False, 'eos_token_id': 1, 'final_logit_softcapping': 30.0, 'global_head_dim': 512, 'head_dim': 256, 'hidden_activation': 'gelu_pytorch_tanh', 'hidden_size': 5376, 'hidden_size_per_layer_input': 0, 'initializer_range': 0.02, 'intermediate_size': 21504, 'layer_types': ['sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention'], 'max_position_embeddings': 262144, 'model_type': 'gemma4_text', 'moe_intermediate_size': None, 'num_attention_heads': 32, 'num_experts': None, 'num_global_key_value_heads': 4, 'num_hidden_layers': 60, 'num_key_value_heads': 16, 'num_kv_shared_layers': 0, 'pad_token_id': 0, 'rms_norm_eps': 1e-06, 'rope_parameters': {'full_attention': {'partial_rotary_factor': 0.25, 'rope_theta': 1000000.0, 'rope_type': 'proportional'}, 'sliding_attention': {'rope_theta': 10000.0, 'rope_type': 'default'}}, 'sliding_window': 1024, 'tie_word_embeddings': True, 'top_k_experts': None, 'use_bidirectional_attention': 'vision', 'use_cache': True, 'use_double_wide_mlp': False, 'vocab_size': 262144, 'vocab_size_per_layer_input': 262144}
CONFIG_SHA256 = 'f234355033b785cea51349d7ee9dc13a5fd3ffd3201bcb9a963b9fc871722075'
if __name__ == '__main__':
    raise SystemExit(worker() if '--worker' in sys.argv else main())
