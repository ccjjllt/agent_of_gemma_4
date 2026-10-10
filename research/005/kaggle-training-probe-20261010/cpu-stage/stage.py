"""CPU-only preparation: resolve wheels without replacing torch; inspect mounted model."""
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import urllib.request
import urllib.parse

OUT = Path('/kaggle/working')
WHEELS = OUT / 'training-wheels'
WHEELS.mkdir(exist_ok=True)
START = time.monotonic()
report = {'purpose': 'CPU preparation only; no GPU and no 31B weight loading', 'python': sys.version,
          'platform': platform.platform(), 'phases': []}

def save(phase, **values):
    report.update(values)
    report['phases'].append({'phase': phase, 'elapsed_seconds': time.monotonic() - START})
    (OUT / 'cpu-stage-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'phase': phase, **values}), flush=True)

def run(cmd, timeout=420):
    subprocess.run(cmd, check=True, timeout=timeout)

try:
    import torch
    save('runtime', torch=torch.__version__, cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        raise RuntimeError('CPU stage unexpectedly has a GPU')
    models = []
    for cfgpath in Path('/kaggle/input').rglob('config.json'):
        try:
            cfg = json.loads(cfgpath.read_text())
        except Exception:
            continue
        idx = cfgpath.parent / 'model.safetensors.index.json'
        if cfg.get('model_type') == 'gemma4' and idx.exists() and not cfg.get('quantization_config'):
            shards = sorted(set(json.loads(idx.read_text())['weight_map'].values()))
            missing = [name for name in shards if not (cfgpath.parent / name).is_file()]
            models.append({'path': str(cfgpath.parent), 'shards': len(shards), 'missing': missing,
                           'bytes': sum((cfgpath.parent / name).stat().st_size for name in shards if (cfgpath.parent / name).is_file()),
                           'config_sha256': hashlib.sha256(cfgpath.read_bytes()).hexdigest()})
    save('model_inventory', models=models)
    if len(models) != 1 or models[0]['missing']:
        raise RuntimeError('Expected one complete unquantized Gemma4 input')
    requested = ['transformers==5.13.1', 'peft==0.21.2', 'accelerate==1.15.0',
                 'safetensors==0.8.0', 'tokenizers==0.22.2', 'PyYAML==6.0.3', 'bitsandbytes']
    resolution = OUT / 'pip-resolution.json'
    run([sys.executable, '-m', 'pip', 'install', '--dry-run', '--report', str(resolution), *requested])
    resolved = json.loads(resolution.read_text())['install']
    proposals = {x['metadata']['name'].lower().replace('_', '-'): x for x in resolved}
    forbidden = [n for n in proposals if n == 'torch' or n == 'triton' or n.startswith('nvidia-')]
    if forbidden:
        raise RuntimeError('Resolver would replace stock CUDA stack: ' + repr(forbidden))
    # Include requested packages even if already present. Resolver adds their missing dependencies.
    pins = {p.split('==')[0].lower().replace('_', '-'): p for p in requested}
    pins.update({n: n + '==' + p['metadata']['version'] for n, p in proposals.items()})
    for name, pin in list(pins.items()):
        if '==' not in pin:
            pins[name] = name + '==' + md.version(name)
    run([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:', '--no-deps', '-d', str(WHEELS), *pins.values()])
    wheels = sorted(WHEELS.glob('*.whl'))
    save('wheels_downloaded', pins=list(pins.values()), wheels=[{'name': p.name, 'bytes': p.stat().st_size,
         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in wheels])
    target = '/tmp/p005-training-deps'
    run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index', '--target', target, *map(str, wheels)])
    code = '''
import json, sys, importlib.metadata as md
import torch, transformers, peft, accelerate, bitsandbytes
from transformers import AutoConfig, AutoTokenizer, Gemma4Config, Gemma4TextConfig, Gemma4ForConditionalGeneration
path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
cfg = AutoConfig.from_pretrained(path, local_files_only=True)
assert cfg.model_type == 'gemma4'
assert 'Gemma4' in type(cfg).__name__
assert tok.get_chat_template()
print(json.dumps({'imports_ok': True, 'model_type': cfg.model_type, 'tokenizer_size': len(tok),
 'template_length': len(tok.get_chat_template()), 'versions': {n: md.version(n) for n in
 ['torch','transformers','peft','accelerate','bitsandbytes','safetensors','tokenizers']}}), flush=True)
'''
    env = dict(os.environ, PYTHONPATH=target, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    check = subprocess.run([sys.executable, '-c', code, models[0]['path']], env=env, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (OUT / 'cpu-import-check.log').write_text(check.stdout)
    print(check.stdout, flush=True)
    if check.returncode:
        raise RuntimeError('Training imports or official tokenizer failed')
    save('complete', success=True)
except Exception as exc:
    save('failed', success=False, error=repr(exc), traceback=traceback.format_exc())
    raise
