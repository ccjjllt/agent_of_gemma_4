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


# CONFIG, CONFIG_SHA256 and grouped_attention are embedded in the runnable carrier.
if __name__ == '__main__':
    raise SystemExit(worker() if '--worker' in sys.argv else main())
