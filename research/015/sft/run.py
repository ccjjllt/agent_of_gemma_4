"""One bounded P015 verified on-policy action RFT run. Building does not run it."""
from pathlib import Path
import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import zlib

OUT = Path('/kaggle/working/p015-task-rft')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(zlib.decompress(base64.b64decode(BUNDLE)))
    root = Path('/tmp/p015-task-rft-src')
    for relative, content in bundle['sources'].items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    data = Path('/tmp/p015-on-policy-action-train.jsonl')
    data.write_text(bundle['data'])
    assert sha(data) == bundle['manifest']['selected_data_sha256']
    (OUT/'carrier-manifest.json').write_text(json.dumps(bundle['manifest'], indent=2))
    reports = list(Path('/kaggle/input').rglob('cpu-stage-report.json'))
    if len(reports) != 1:
        raise RuntimeError('Attach exactly one existing p005-training-cpu-stage output')
    stage = json.loads(reports[0].read_text())
    if stage.get('success') is not True:
        raise RuntimeError('Dependency CPU stage was unsuccessful')
    wheels = []
    for item in stage['wheels']:
        name = item['name']
        if name.lower().startswith(('torch-', 'triton-', 'nvidia_')):
            raise RuntimeError('Unexpected CUDA-stack replacement wheel')
        if name.lower().startswith('pyyaml-'):
            continue
        path = reports[0].parent/'training-wheels'/name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise RuntimeError('Staged dependency identity changed: '+name)
        wheels.append(str(path))
    deps = Path('/tmp/p015-task-rft-deps')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--target', str(deps), *wheels], check=True, timeout=100)
    caches = []
    for path in Path('/kaggle/input').rglob('cache-manifest.json'):
        value = json.loads(path.read_text())
        if value.get('source') == 'google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2':
            caches.append(path.parent)
    if len(caches) != 1:
        raise RuntimeError('Attach exactly one official-source p005-cpu-nf4-cache output')
    tokenizers = []
    for path in Path('/kaggle/input').rglob('source-manifest.json'):
        value = json.loads(path.read_text())
        if value.get('source') == 'google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2':
            tokenizers.append((path.parent, value))
    if len(tokenizers) != 1:
        raise RuntimeError('Attach exactly one official P015 31B CT v2 tokenizer CPU output')
    tokenizer, tokenizer_manifest = tokenizers[0]
    for name, item in tokenizer_manifest['files'].items():
        path = tokenizer/name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise RuntimeError('Official tokenizer asset identity changed: '+name)
    cfg = bundle['config']
    cfg.update(model_name_or_path=str(caches[0]), tokenizer_name_or_path=str(tokenizer),
               tool_schema_path=str(root/'training/configs/harness_tools.json'),
               chat_template_path=str(tokenizer/'chat_template.jinja'))
    config = OUT/'training-config.json'
    config.write_text(json.dumps(cfg, indent=2))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(deps), str(root)]),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
        NCCL_DEBUG='WARN', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
        FSDP_TRANSFORMER_CLS_TO_WRAP='Gemma4TextDecoderLayer')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
        '--max-restarts=0', '-m', 'training.train_fsdp', '--config', str(config),
        '--data', str(data), '--out', str(OUT/'run')]
    print(json.dumps({'purpose': 'verified on-policy action RFT; full-task SFT and repair gain remain unproven',
        'command': command, 'setup_seconds': time.monotonic()-started,
        'worker_timeout_seconds': bundle['manifest']['worker_timeout_seconds'],
        'grouped_attention': False}), flush=True)
    # Kill the complete process group on timeout; torchrun's ranks must not be
    # left burning quota after their controller has exited.
    proc = subprocess.Popen(command, env=env, cwd=root, start_new_session=True)
    try:
        result = proc.wait(timeout=bundle['manifest']['worker_timeout_seconds'])
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)
        raise RuntimeError('FSDP RFT worker exceeded the frozen timeout; process group terminated')
    print(json.dumps({'worker_exit_code': result, 'elapsed_seconds': time.monotonic()-started}), flush=True)
    return result


# BUNDLE is embedded at packaging time.
if __name__ == '__main__':
    raise SystemExit(main())
