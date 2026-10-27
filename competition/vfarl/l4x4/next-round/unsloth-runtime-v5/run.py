import sys
sys.argv = ['prepare_unsloth_runtime.py', '--base-runtime-sha256', 'c80c38ed9799f05e79c5938956f939b099625a1a356a3b6405e6d7dcb04fafa9']
"""Extend the accepted Linux runtime with Unsloth dependencies on a CPU worker.

This prepares an offline GPU runtime. It never imports Unsloth, loads a model,
or claims that CUDA kernels or the VFARL integration have been exercised.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.parse
import urllib.request


WORK = Path('/kaggle/working')
STAGE = WORK / 'vfarl-cpu-stage'
EVIDENCE = WORK / 'vfarl-unsloth-preparation'
BASE_RUNTIME_SHA256 = '9475d9a2617135c276d3b4ad87884b28425f871e806c07b62bfc62d4af0cce7e'
EXPECTED = {'transformers': '5.13.1', 'peft': '0.21.2',
            'bitsandbytes': '0.50.2', 'accelerate': '1.15.0',
            'tokenizers': '0.22.2'}
UNSLOTH_PINS = {'unsloth': '2026.10.9', 'unsloth-zoo': '2026.10.9'}
# Official 2026.10.9 metadata conflicts with stock torchao, datasets and certifi.
# Only these helpers may be shadowed; CUDA and the model stack stay put.
OVERLAY_REPLACEMENTS = {'torchao', 'datasets', 'certifi'}
STARTED = time.monotonic()
REPORT = {'purpose': 'CPU-only dependency resolution and offline runtime assembly',
          'gpu_model_loaded': False, 'unsloth_import_attempted': False,
          'base_runtime_sha256': BASE_RUNTIME_SHA256, 'phases': []}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def record(phase, **fields):
    REPORT.update(fields)
    REPORT['phases'].append({'phase': phase, 'seconds': time.monotonic() - STARTED})
    write(EVIDENCE / 'preparation.json', REPORT)
    print(json.dumps({'phase': phase, **fields}), flush=True)


def unpack(path, root):
    # Keep the same symlink and path semantics as the accepted GPU carrier.
    root = root.resolve()
    def safe_filter(member, destination):
        if member.issym() and os.path.isabs(member.linkname):
            target = Path(member.linkname).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError('Archive symlink escapes extraction root')
            member = copy.copy(member)
            member.linkname = os.path.relpath(target, (root / member.name).parent)
        return tarfile.data_filter(member, destination)
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if not (root / member.name).resolve().is_relative_to(root):
                raise RuntimeError('Archive path escape')
        archive.extractall(root, filter=safe_filter)


def installed_versions():
    versions = {}
    for distribution in metadata.distributions():
        name = canonical(distribution.metadata['Name'])
        versions.setdefault(name, distribution.version)
    return versions


def command(arguments, env, log):
    result = subprocess.run(arguments, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    print(result.stdout[-12000:], flush=True)
    if result.returncode:
        raise RuntimeError('Dependency preparation failed (%d); see %s' %
                           (result.returncode, log.name))


def main(base_runtime_sha256=None):
    expected_base = base_runtime_sha256 or BASE_RUNTIME_SHA256
    if not re.fullmatch(r'[0-9a-f]{64}', expected_base):
        raise ValueError('Expected an explicit lowercase SHA256 for the base runtime')
    REPORT['base_runtime_sha256'] = expected_base
    EVIDENCE.mkdir(parents=True, exist_ok=False)
    import torch
    if torch.cuda.is_available() or torch.cuda.device_count():
        raise RuntimeError('CPU preparation refuses any attached CUDA device')
    if sys.version_info[:2] != (3, 12) or torch.__version__ != '2.10.0+cu128':
        raise RuntimeError('Pinned Python3.12 / torch2.10.0+cu128 image required')
    if STAGE.exists():
        raise RuntimeError('Fresh CPU stage path required')
    record('cpu_environment', python=sys.version, platform=platform.platform(),
           stock_torch=torch.__version__, stock_triton=metadata.version('triton'))
    archives = list(Path('/kaggle/input').rglob('vfarl-cpu-stage.bin'))
    if len(archives) != 1 or digest(archives[0]) != expected_base:
        raise RuntimeError('Missing or changed accepted base runtime archive')
    unpack(archives[0], WORK)
    runtime = STAGE / 'runtime'
    sys.path.insert(0, str(runtime))
    importlib.invalidate_caches()
    environment = dict(os.environ, PYTHONPATH=str(runtime),
                       PIP_DISABLE_PIP_VERSION_CHECK='1')
    base_preparation = STAGE / 'preparation.json'
    base_preparation_sha = digest(base_preparation)
    initial = installed_versions()
    if initial.get('torch') != REPORT['stock_torch'] or initial.get('triton') != REPORT['stock_triton']:
        raise RuntimeError('Accepted overlay must reuse the pinned stock torch/triton')
    for name, expected in EXPECTED.items():
        if initial.get(name) != expected:
            raise RuntimeError('Accepted runtime version differs: %s=%r, expected %s' %
                               (name, initial.get(name), expected))
    # Preserve the accepted runtime while repairing the three declared helper
    # conflicts in a new overlay. Task venvs are never changed.
    constraints = EVIDENCE / 'existing-constraints.txt'
    preserved = {name: version for name, version in initial.items()
                 if name not in OVERLAY_REPLACEMENTS}
    constraints.write_text(''.join('%s==%s\n' % item for item in sorted(preserved.items())))
    record('base_runtime_ready', base_archive_bytes=archives[0].stat().st_size,
           base_preparation_sha256=base_preparation_sha,
           base_packages=initial, preserved_packages=preserved,
           allowed_overlay_replacements={name: initial.get(name) for name in sorted(OVERLAY_REPLACEMENTS)},
           unsloth_pins=UNSLOTH_PINS, constraints_sha256=digest(constraints))
    tokenizer = STAGE / 'tokenizer'
    original_tokenizer = {p.name: digest(p) for p in tokenizer.iterdir() if p.is_file()}
    roots = [p.parent for p in Path('/kaggle/input').rglob('tokenizer.json')
             if 'gemma-4-31b-it-qat-w4a16-ct' in str(p)]
    if len(roots) != 1:
        raise RuntimeError('Expected exactly one official CTv2 processor mount')
    added = []
    for path in sorted(roots[0].glob('*processor*.json')):
        target = tokenizer / path.name
        if not target.exists():
            shutil.copy2(path, target)
            added.append({'name': path.name, 'bytes': target.stat().st_size,
                          'sha256': digest(target)})
    if not any('processor' in p.name for p in tokenizer.glob('*.json')):
        raise RuntimeError('Official processor/preprocessor assets are absent')
    if any(digest(tokenizer / name) != value for name, value in original_tokenizer.items()):
        raise RuntimeError('Existing tokenizer/template/config changed')
    record('processor_assets', official_source='google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2',
           existing_tokenizer_files=original_tokenizer, added_processor_files=added)
    resolver_report = EVIDENCE / 'pip-resolver-report.json'
    command([sys.executable, '-m', 'pip', 'install', '--dry-run', '--report',
             str(resolver_report), '--constraint', str(constraints),
             *[name + '==' + version for name, version in UNSLOTH_PINS.items()]],
            environment, EVIDENCE / 'pip-resolver.log')
    resolved = json.loads(resolver_report.read_text())
    proposals = resolved.get('install', [])
    forbidden = []
    for proposal in proposals:
        name = canonical(proposal['metadata']['name'])
        if ((name in initial and name not in OVERLAY_REPLACEMENTS)
                or name in ('torch', 'triton', 'pytorch-triton') or name.startswith('nvidia-')):
            forbidden.append({'name': name, 'version': proposal['metadata']['version']})
    if forbidden:
        record('dependency_conflict', forbidden_replacements=forbidden)
        raise RuntimeError('Resolver proposed replacing the frozen runtime or CUDA stack')
    selected = {canonical(p['metadata']['name']): p['metadata']['version'] for p in proposals}
    if not all(name in selected or name in initial for name in ('unsloth', 'unsloth-zoo')):
        raise RuntimeError('Resolver did not supply both Unsloth packages')
    replacements = {name: {'previous_version': initial[name], 'overlay_version': version}
                    for name, version in selected.items() if name in initial}
    record('dependencies_resolved', newly_selected=selected, overlay_replacements=replacements,
           resolver_report_sha256=digest(resolver_report))
    wheelhouse = WORK / 'vfarl-unsloth-wheels'
    wheelhouse.mkdir(exist_ok=False)
    wheels = []
    for proposal in proposals:
        info = proposal['download_info']
        url = info['url']
        name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
        if not name.endswith('.whl'):
            raise RuntimeError('Only resolved wheels may be installed: ' + name)
        path = wheelhouse / name
        urllib.request.urlretrieve(url, path)
        actual = digest(path)
        expected = info.get('archive_info', {}).get('hashes', {}).get('sha256')
        if not expected or actual != expected:
            raise RuntimeError('Resolved wheel hash mismatch: ' + name)
        wheels.append({'package': canonical(proposal['metadata']['name']),
                       'version': proposal['metadata']['version'], 'wheel': name,
                       'bytes': path.stat().st_size, 'sha256': actual, 'url': url})
        upgrade = ['--upgrade'] if canonical(proposal['metadata']['name']) in replacements else []
        command([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-compile',
                 *upgrade, '--target', str(runtime), str(path)], environment,
                EVIDENCE / ('install-%s.log' % canonical(proposal['metadata']['name'])))
    subprocess_code = (
        'import importlib.metadata as m,json,torch,transformers,peft,bitsandbytes,accelerate,trl,torchao,xformers,datasets; '
        'names=("torch","triton","transformers","peft","bitsandbytes","accelerate",'
        '"tokenizers","trl","unsloth","unsloth_zoo","torchao","xformers","datasets","certifi"); '
        'print(json.dumps({"versions":{n:m.version(n) for n in names},'
        '"cuda_available":torch.cuda.is_available(),'
        '"unsloth_import_attempted":False}))')
    command([sys.executable, '-c', subprocess_code], environment,
            EVIDENCE / 'cpu-module-imports.log')
    # Read metadata only; importing Unsloth on CPU would incorrectly require GPU.
    expected_final = dict(EXPECTED, torch=initial['torch'], triton=initial['triton'], **UNSLOTH_PINS)
    expected_final.update({name: value['overlay_version'] for name, value in replacements.items()})
    final_output = subprocess.run([sys.executable, '-c',
        'import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in '+repr(list(expected_final))+ '}))'],
        env=environment, text=True, capture_output=True, check=True)
    final_versions = json.loads(final_output.stdout)
    for name, expected in expected_final.items():
        if final_versions.get(name) != expected:
            raise RuntimeError('Runtime package differs from resolved assembly: ' + name)
    extension = {'base_runtime_sha256': expected_base,
                 'base_preparation_sha256': base_preparation_sha,
                 'new_packages': wheels, 'original_tokenizer_files': original_tokenizer,
                 'overlay_replacements': replacements, 'final_versions': final_versions,
                 'added_processor_files': added, 'gpu_verified': False,
                 'unsloth_import_attempted': False,
                 'limitation': 'Only CPU dependency assembly and ordinary module imports ran. CUDA, model load, kernels, FSDP and VFARL remain unmeasured.'}
    old = json.loads(base_preparation.read_text())
    old['unsloth_extension'] = extension
    write(base_preparation, old)
    write(EVIDENCE / 'unsloth-extension.json', extension)
    record('runtime_assembled', new_packages=wheels, task_environments_reused=True,
           gpu_verified=False, ordinary_cpu_imports_completed=True)
    shutil.rmtree(wheelhouse)
    target = WORK / 'vfarl-cpu-stage.bin'
    with tarfile.open(target, 'w:gz', compresslevel=1) as archive:
        archive.add(STAGE, arcname=STAGE.name)
    archive_record = {'path': target.name, 'bytes': target.stat().st_size,
                      'sha256': digest(target), 'base_runtime_sha256': expected_base,
                      'gpu_verified': False, 'new_packages': selected,
                      'overlay_replacements': replacements,
                      'wall_seconds': time.monotonic() - STARTED}
    write(WORK / 'runtime-archive.json', archive_record)
    record('archive_completed', **archive_record, success=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-runtime-sha256', default=BASE_RUNTIME_SHA256,
                        help='Actual SHA256 of the mounted accepted CPU archive; historical default retained')
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{64}', args.base_runtime_sha256):
        parser.error('--base-runtime-sha256 must be a lowercase SHA256')
    try:
        main(args.base_runtime_sha256)
    except BaseException:
        EVIDENCE.mkdir(exist_ok=True)
        (EVIDENCE / 'exception.txt').write_text(traceback.format_exc())
        record('failed', success=False, error=traceback.format_exc())
        raise
    finally:
        if STAGE.exists():
            shutil.rmtree(STAGE)
        wheels = WORK / 'vfarl-unsloth-wheels'
        if wheels.exists():
            shutil.rmtree(wheels)
