"""CPU-only 12B QAT NF4 cache preparation; zero GPU usage.

Downloads the exact pinned HuggingFace google/gemma-4-12B-it-qat-q4_0-unquantized
revision b6ed86275a6a5735884e208bfed95b445a684ca2 into /tmp, verifies its
config.json and 23.9 GB model.safetensors SHA256, verifies the mounted official
12B CT v2 deployment tokenizer/template, verifies a tiny Gemma4Unified CPU NF4
roundtrip, and saves the quantized 12B NF4 cache + provenance manifest under
/kaggle/working/qat-12b-nf4 alongside compatible training wheels.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

OUT = Path('/kaggle/working')
started = time.monotonic()
report = {'purpose': 'CPU-only 12B QAT-derived NF4 cache; not a trained model', 'events': []}


def event(stage, **kw):
    report.update(kw)
    report['events'].append({'stage': stage, 'elapsed_seconds': round(time.monotonic() - started, 3)})
    (OUT / 'cache-12b-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'stage': stage, **kw}), flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


CHILD = r'''
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import urllib.request

import bitsandbytes as bnb
import psutil
import torch
import transformers as tr

torch.set_num_threads(min(4, os.cpu_count() or 2))
out = Path('/kaggle/working')

PINNED = {
    'base_family': 'gemma-4-12b-it-qat-q4_0-unquantized',
    'repo_id': 'google/gemma-4-12B-it-qat-q4_0-unquantized',
    'revision': 'b6ed86275a6a5735884e208bfed95b445a684ca2',
    'model_type': 'gemma4_unified',
    'architecture': 'Gemma4UnifiedForConditionalGeneration',
    'config_bytes': 4310,
    'config_sha256': 'a323d02f68420f6fa3a3548130a0d36356075a4047a622e57148558f8eee7077',
    'weights_name': 'model.safetensors',
    'weights_bytes': 23919549408,
    'weights_sha256': '26f2cee4292298a3f9f92209643c37c80e34e011381e22434088870d9439a0a0',
    'deployment_ct_handle': 'google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2',
    'deployment_ct_config_sha256': 'ba14713d084391532b285f7c252b954b7c6c0db97f427a9d8fb788edc2949168',
    'deployment_chat_template_sha256': '36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0',
    'deployment_tokenizer_sha256': {
        'tokenizer.json': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
        'tokenizer_config.json': 'ee66825f1a587ca13bcb90d7e60f59bc99e84f3b8bc0cefdc3ea595cd5abf773',
    },
}


def emit(stage, **data):
    payload = {'stage': stage, **data}
    (out / 'cache-12b-child-report.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload), flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download_verified(url, dest, expected_bytes, expected_sha256):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    total = 0
    t0 = time.monotonic()
    req = urllib.request.Request(url, headers={'User-Agent': 'kaggle-p007-12b-cpu-cache/1.0'})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open('wb') as f:
        while True:
            chunk = resp.read(16 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            total += len(chunk)
    digest = h.hexdigest()
    if expected_bytes is not None and total != expected_bytes:
        raise RuntimeError(f'Size mismatch for {dest.name}: expected {expected_bytes}, got {total}')
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f'SHA256 mismatch for {dest.name}: expected {expected_sha256}, got {digest}')
    return {'bytes': total, 'sha256': digest, 'seconds': round(time.monotonic() - t0, 3)}


assert not torch.cuda.is_available(), 'CPU stage must not consume GPU'
emit('cpu_runtime', torch=torch.__version__, bnb=bnb.__version__, transformers=tr.__version__,
     ram=psutil.virtual_memory()._asdict(), disk_tmp=shutil.disk_usage('/tmp')._asdict())

# 1. Locate and verify mounted official 12B CT v2 deployment assets.
ct_dirs = []
for p in Path('/kaggle/input').rglob('tokenizer.json'):
    if '12b-it-qat-w4a16-ct' in str(p):
        ct_dirs.append(p.parent)
assert len(ct_dirs) == 1, f'Expected one mounted 12B CT v2 directory, found {ct_dirs}'
ct_dir = ct_dirs[0]
assert sha256_file(ct_dir / 'config.json') == PINNED['deployment_ct_config_sha256']
assert sha256_file(ct_dir / 'chat_template.jinja') == PINNED['deployment_chat_template_sha256']
for tname, tsha in PINNED['deployment_tokenizer_sha256'].items():
    assert sha256_file(ct_dir / tname) == tsha, f'CT tokenizer mismatch: {tname}'
emit('ct_deployment_assets_verified', ct_dir=str(ct_dir))

# 2. Download pinned 12B QAT config.json first and test tiny Gemma4Unified NF4 roundtrip on CPU.
raw_dir = Path('/tmp/p007-12b-qat-raw')
if raw_dir.exists():
    shutil.rmtree(raw_dir)
raw_dir.mkdir(parents=True)
base_url = f"https://huggingface.co/{PINNED['repo_id']}/resolve/{PINNED['revision']}"
cfg_dl = download_verified(
    f"{base_url}/config.json",
    raw_dir / 'config.json',
    PINNED['config_bytes'],
    PINNED['config_sha256'],
)
emit('config_downloaded', **cfg_dl)

full_cfg = json.loads((raw_dir / 'config.json').read_text())
assert full_cfg.get('model_type') == 'gemma4_unified'
assert not full_cfg.get('quantization_config')

# Construct tiny Gemma4Unified from the real config structure to verify CPU NF4 roundtrip.
tiny_dict = json.loads(json.dumps(full_cfg))
tc = tiny_dict['text_config']
tc.update({
    'vocab_size': 64, 'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 2,
    'num_attention_heads': 2, 'num_key_value_heads': 2, 'head_dim': 8, 'global_head_dim': 8,
    'num_global_key_value_heads': 2, 'layer_types': ['sliding_attention', 'full_attention'],
    'sliding_window': 16, 'max_position_embeddings': 64, 'hidden_size_per_layer_input': 0,
    'vocab_size_per_layer_input': 64, 'num_kv_shared_layers': 0,
})
vc = tiny_dict['vision_config']
vc.update({
    'hidden_size': 16, 'intermediate_size': 32, 'num_hidden_layers': 1,
    'num_attention_heads': 2, 'num_key_value_heads': 2, 'head_dim': 8, 'position_embedding_size': 4,
})
if tiny_dict.get('audio_config'):
    ac = tiny_dict['audio_config']
    ac.update({
        'hidden_size': 16, 'num_hidden_layers': 1, 'num_attention_heads': 2,
        'conv_kernel_size': 3, 'subsampling_conv_channels': [8, 8],
    })
for tok_key in ('image_token_id', 'boi_token_id', 'eoi_token_id', 'audio_token_id', 'boa_token_id', 'eoa_token_id', 'eos_token_id', 'bos_token_id', 'pad_token_id'):
    if tok_key in tiny_dict and isinstance(tiny_dict[tok_key], int) and tiny_dict[tok_key] >= 64:
        tiny_dict[tok_key] = tiny_dict[tok_key] % 32 + 16
for tok_key in ('eos_token_id', 'bos_token_id', 'pad_token_id'):
    if tok_key in tc and isinstance(tc[tok_key], int) and tc[tok_key] >= 64:
        tc[tok_key] = tc[tok_key] % 16

tiny_base = Path('/tmp/tiny-12b-unified-base')
tiny_nf4 = Path('/tmp/tiny-12b-unified-nf4')
if tiny_base.exists():
    shutil.rmtree(tiny_base)
if tiny_nf4.exists():
    shutil.rmtree(tiny_nf4)
tiny_base.mkdir(parents=True)
(tiny_base / 'config.json').write_text(json.dumps(tiny_dict, indent=2))
tiny_cfg = tr.AutoConfig.from_pretrained(str(tiny_base), local_files_only=True)
model_cls = getattr(tr, PINNED['architecture'])
torch.manual_seed(0)
model = model_cls(tiny_cfg).to(torch.bfloat16)
model.save_pretrained(tiny_base, safe_serialization=True)
del model
gc.collect()

quant = tr.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = model_cls.from_pretrained(
    str(tiny_base),
    local_files_only=True,
    dtype=torch.bfloat16,
    quantization_config=quant,
    device_map={'': 'cpu'},
    attn_implementation='sdpa',
)
quantized = [(n, p) for n, p in model.named_parameters() if p.__class__.__name__ == 'Params4bit']
assert quantized and all(p.bnb_quantized for _, p in quantized)
fingerprint = {n: hashlib.sha256(p.detach().view(torch.uint8).numpy().tobytes()).hexdigest() for n, p in quantized}
model.save_pretrained(tiny_nf4, safe_serialization=True)
del model, quantized
gc.collect()

model = model_cls.from_pretrained(
    str(tiny_nf4),
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map={'': 'cpu'},
    attn_implementation='sdpa',
)
reloaded = {n: hashlib.sha256(p.detach().view(torch.uint8).numpy().tobytes()).hexdigest()
            for n, p in model.named_parameters() if p.__class__.__name__ == 'Params4bit'}
assert fingerprint == reloaded, 'Unified NF4 packed bytes changed on save/reload'
emit('tiny_unified_cpu_nf4_roundtrip_ok', quantized_tensors=len(fingerprint))
del model
gc.collect()
shutil.rmtree(tiny_base, ignore_errors=True)
shutil.rmtree(tiny_nf4, ignore_errors=True)

# 3. Download and verify the full 23.9 GB unquantized 12B QAT model.safetensors into /tmp.
assert shutil.disk_usage('/tmp').free > 26 * 1024**3, 'Insufficient /tmp disk space for 23.9 GB 12B QAT weights'
assert psutil.virtual_memory().available > 18 * 1024**3, 'Insufficient CPU RAM for 12B NF4 quantization'
emit('full_12b_download_begin', url=f"{base_url}/{PINNED['weights_name']}")
weight_dl = download_verified(
    f"{base_url}/{PINNED['weights_name']}",
    raw_dir / PINNED['weights_name'],
    PINNED['weights_bytes'],
    PINNED['weights_sha256'],
)
emit('full_12b_download_verified', **weight_dl)

# 4. Quantize full 12B QAT to NF4 on CPU and save under /kaggle/working/qat-12b-nf4.
t_load = time.monotonic()
model = model_cls.from_pretrained(
    str(raw_dir),
    local_files_only=True,
    dtype=torch.bfloat16,
    quantization_config=quant,
    device_map={'': 'cpu'},
    attn_implementation='sdpa',
)
load_seconds = round(time.monotonic() - t_load, 3)
footprint_bytes = int(model.get_memory_footprint())
quantized_count = sum(1 for p in model.parameters() if p.__class__.__name__ == 'Params4bit')
emit('full_12b_cpu_nf4_loaded', load_seconds=load_seconds, footprint_bytes=footprint_bytes,
     quantized_tensors=quantized_count)

# Remove raw 23.9 GB file from /tmp before saving shards.
(raw_dir / PINNED['weights_name']).unlink()

dest = out / 'qat-12b-nf4'
if dest.exists():
    shutil.rmtree(dest)
dest.mkdir(parents=True)
t_save = time.monotonic()
model.save_pretrained(dest, safe_serialization=True, max_shard_size='2GB')
save_seconds = round(time.monotonic() - t_save, 3)

# Copy verified deployment CT v2 tokenizer and chat template into dest so tokenizer + template are co-located.
for fname in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
    shutil.copy2(ct_dir / fname, dest / fname)

files_manifest = []
for p in sorted(dest.iterdir()):
    if not p.is_file():
        continue
    files_manifest.append({
        'name': p.name,
        'bytes': p.stat().st_size,
        'sha256': sha256_file(p),
    })

manifest = {
    'manifest_version': 1,
    'base_family': PINNED['base_family'],
    'source': f"{PINNED['repo_id']}@{PINNED['revision']}",
    'source_repo_id': PINNED['repo_id'],
    'source_revision': PINNED['revision'],
    'model_type': PINNED['model_type'],
    'architecture': PINNED['architecture'],
    'source_config_sha256': PINNED['config_sha256'],
    'source_weights_name': PINNED['weights_name'],
    'source_weights_bytes': PINNED['weights_bytes'],
    'source_weights_sha256': PINNED['weights_sha256'],
    'deployment_ct_handle': PINNED['deployment_ct_handle'],
    'deployment_ct_config_sha256': PINNED['deployment_ct_config_sha256'],
    'deployment_chat_template_sha256': PINNED['deployment_chat_template_sha256'],
    'deployment_tokenizer_sha256': PINNED['deployment_tokenizer_sha256'],
    'purpose': 'Untrained 12B QAT-derived NF4 cache for training only; not competition inference base',
    'versions': {
        'torch': torch.__version__,
        'transformers': tr.__version__,
        'bitsandbytes': bnb.__version__,
    },
    'download_seconds': weight_dl['seconds'],
    'load_seconds': load_seconds,
    'save_seconds': save_seconds,
    'footprint_bytes': footprint_bytes,
    'quantized_tensors': quantized_count,
    'files': files_manifest,
    'total_bytes': sum(x['bytes'] for x in files_manifest),
}
(dest / 'cache-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
emit('complete', success=True, total_bytes=manifest['total_bytes'],
     quantized_tensors=quantized_count, files_count=len(files_manifest))
'''

proc = None
try:
    stages = list(Path('/kaggle/input').rglob('cpu-stage-report.json'))
    assert len(stages) == 1, f'Expected one cpu-stage-report.json, found {stages}'
    stage = json.loads(stages[0].read_text())
    assert stage['success']
    wheels_out = OUT / 'training-wheels'
    wheels_out.mkdir(exist_ok=True)
    wheels = []
    copied_wheels = []
    for w in stage['wheels']:
        if 'cp313' in w['name']:
            continue  # Base Python 3.12 image already has PyYAML; skip cp313 wheel.
        p = stages[0].parent / 'training-wheels' / w['name']
        assert sha256_file(p) == w['sha256'], f"Wheel SHA mismatch: {w['name']}"
        dst = wheels_out / w['name']
        shutil.copy2(p, dst)
        wheels.append(str(dst))
        copied_wheels.append(w)
    (OUT / 'training-wheels-manifest.json').write_text(json.dumps({'wheels': copied_wheels}, indent=2) + '\n')
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', '--target', '/tmp/p007-12b-deps', *wheels],
        check=True,
        timeout=120,
    )
    child_path = Path('/tmp/p007-12b-cache-child.py')
    child_path.write_text(CHILD)
    env = dict(
        os.environ,
        PYTHONPATH='/tmp/p007-12b-deps',
        PYTHONUNBUFFERED='1',
        TOKENIZERS_PARALLELISM='false',
    )
    event('child_start', wheels_installed=len(wheels))
    with (OUT / 'cache-12b-child.log').open('w') as log:
        proc = subprocess.Popen(
            [sys.executable, str(child_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

        def relay():
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if line.startswith('{'):
                    print(line.rstrip(), flush=True)

        thread = threading.Thread(target=relay, daemon=True)
        thread.start()
        try:
            code = proc.wait(timeout=3300)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = 124
        thread.join(timeout=10)
    event('finished', success=(code == 0), exit_code=code)
    if code:
        print((OUT / 'cache-12b-child.log').read_text()[-12000:], flush=True)
except Exception as exc:
    event('failed', success=False, error=repr(exc), traceback=traceback.format_exc())
finally:
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
if not report.get('success'):
    raise SystemExit(2)
