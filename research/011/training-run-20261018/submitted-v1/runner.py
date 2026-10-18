"""Offline environment setup followed by the complete P011 training stage."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
BUNDLE = Path(os.environ['P011_BUNDLE_ROOT']).resolve()
OUT = Path(os.environ['P011_RUN_ROOT']).resolve()
DEADLINE = float(os.environ['P011_DEADLINE_MONOTONIC'])
PINS = json.loads((HERE / 'runtime-pins.json').read_text())
RUN = json.loads((HERE / 'run-manifest.json').read_text())
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE))
from training_stage import load_frozen_data, run_logged, sha256, write_json


def phase(name, **extra):
    write_json(OUT / 'phase.json', {'phase': name, 'time': time.time(), **extra})


def resolve_input(candidates, marker, substring=None):
    for candidate in candidates:
        if candidate.is_dir() and (candidate / marker).is_file():
            return candidate
    found = sorted({path.parent for path in Path('/kaggle/input').rglob(marker)
                    if substring is None or substring in str(path)})
    if len(found) != 1:
        raise RuntimeError(f'Need one attached {marker} input; found {found[:5]}')
    return found[0]


def main():
    if sys.version_info[:2] != (3, 12) or platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('Use the pinned original Linux x86_64 CPython3.12 image')
    phase('frozen_training_data')
    manifest_path = BUNDLE / RUN['dataset_manifest_path']
    data_path, rows, data_summary = load_frozen_data(BUNDLE, manifest_path)
    if sha256(manifest_path) != RUN['dataset_manifest_sha256']:
        raise RuntimeError('Packaged dataset manifest differs from build-time freeze')
    if data_summary['planned_optimizer_steps'] != RUN['planned_optimizer_steps']:
        raise RuntimeError('Build/runtime training step plans differ')
    write_json(OUT / 'input-data-summary.json', data_summary)
    phase('offline_environment')
    for loaded in ('torch', 'transformers', 'vllm'):
        if loaded in sys.modules:
            raise RuntimeError('Use a fresh interpreter; package already loaded: ' + loaded)
    cache = Path(os.environ['P011_NF4_CACHE']) if os.environ.get('P011_NF4_CACHE') else resolve_input(
        [], 'cache-manifest.json', 'qat-12b-nf4')
    if sha256(cache / 'cache-manifest.json') != PINS['nf4_cache_manifest_sha256']:
        raise RuntimeError('P007 immutable NF4 cache manifest changed')
    model = Path(os.environ['P011_CT_TOKENIZER']) if os.environ.get('P011_CT_TOKENIZER') else resolve_input(
        [Path('/kaggle/input/models/google/gemma-4/other/gemma-4-12b-it-qat-w4a16-ct/2'),
         Path('/kaggle/input/gemma-4-12b-it-qat-w4a16-ct/2')], 'tokenizer.json', 'gemma-4-12b-it-qat-w4a16-ct/2')
    for name, key in [('tokenizer.json', 'tokenizer_sha256'), ('tokenizer_config.json', 'tokenizer_config_sha256'),
                      ('config.json', 'config_sha256')]:
        if sha256(model / name) != PINS['model_pin'][key]:
            raise RuntimeError('Deployment CT v2 file differs: ' + name)
    wheelhouse = resolve_input(
        [Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse'),
         Path('/kaggle/input/gemma-4-developer-agent-wheelhouse')], 'swegemma-0.2.11-py3-none-any.whl')
    source_wheels = [p for p in sorted(wheelhouse.glob('*.whl')) if 'cutlass' not in p.name.lower()]
    inventory = {p.name: p.stat().st_size for p in source_wheels}
    if inventory != PINS['wheelhouse_v31_files']:
        raise RuntimeError('Official wheelhouse inventory differs from the recorded P009 v31 runtime')
    try:
        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename
    except ImportError:
        from pip._vendor.packaging.tags import sys_tags
        from pip._vendor.packaging.utils import parse_wheel_filename
    tags = set(sys_tags())
    staging = OUT / 'wheel-links'
    staging.mkdir()
    wheel_manifest = []
    for path in source_wheels:
        name = path.name.replace('cu128', '+cu128') if 'cu128' in path.name and '+' not in path.name else path.name
        if not parse_wheel_filename(name)[3] & tags:
            raise RuntimeError('Incompatible wheel before installation: ' + name)
        distribution = path.name.split('-')[0].replace('_', '-').lower()
        fingerprint = sha256(path)
        expected = PINS['official_packages'].get(distribution)
        if expected and fingerprint != expected.split(' sha256 ')[1]:
            raise RuntimeError('Official package changed: ' + distribution)
        wheel_manifest.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': fingerprint})
        (staging / name).symlink_to(path)
    write_json(OUT / 'wheel-manifest.json', wheel_manifest)
    os.environ.update(TRANSFORMERS_NO_TF='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='1')
    run_logged([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps',
                '--force-reinstall', *[str(p) for p in sorted(staging.glob('*.whl'))]],
               OUT / 'official-dependencies.log', BUNDLE, min(420, DEADLINE - time.monotonic() - 180))
    importlib.invalidate_caches()
    phase('continuous_sft', rows=len(rows), planned_steps=RUN['planned_optimizer_steps'])
    command = [sys.executable, '-u', str(HERE / 'training_stage.py'), '--bundle', str(BUNDLE),
               '--dataset-manifest', str(manifest_path), '--out', str(OUT / 'training'),
               '--cache', str(cache), '--ct', str(model),
               '--timeout', str(max(1, min(RUN['training_stage_max_seconds'],
                                          int(DEADLINE - time.monotonic() - 30))))]
    if os.environ.get('P011_RESUME_CHECKPOINT'):
        command.extend(['--resume-checkpoint', os.environ['P011_RESUME_CHECKPOINT']])
    run_logged(command, OUT / 'training-stage.log', BUNDLE, DEADLINE - time.monotonic() - 10)
    summary = json.loads((OUT / 'training/stage-summary.json').read_text())
    if summary['status'] != 'completed':
        raise RuntimeError('Training stage did not report complete epochs')
    phase('completed', optimizer_steps=summary['optimizer_steps'], adapter_sha256=summary['adapter_sha256'])
    write_json(OUT / 'run-summary.json', {
        'status': 'completed', 'training': summary,
        'data_manifest_sha256': RUN['dataset_manifest_sha256'],
        'diagnostic_runs': 0, 'submission_created': False})


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        raise
