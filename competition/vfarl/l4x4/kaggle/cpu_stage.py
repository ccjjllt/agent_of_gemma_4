"""Prepare reusable Linux runtime assets on a CPU Kaggle worker; no model load."""
from pathlib import Path
import hashlib
import importlib.metadata as md
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

OUT = Path('/kaggle/working/vfarl-cpu-stage')
OUT.mkdir(parents=True, exist_ok=True)
START = time.monotonic()
REPORT = {'purpose': 'Linux dependency and isolation preparation; zero GPU/model execution',
          'python': sys.version, 'platform': platform.platform(), 'phases': [],
          'task_environments': []}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(phase, **values):
    REPORT.update(values)
    REPORT['phases'].append({'phase': phase, 'elapsed_seconds': time.monotonic() - START})
    (OUT / 'preparation.json').write_text(json.dumps(REPORT, indent=2) + '\n')
    print(json.dumps({'phase': phase, **values}), flush=True)


def run(command, *, env=None, log=None, check=True):
    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if log:
        Path(log).write_text(result.stdout)
    print(result.stdout[-6000:], flush=True)
    if check and result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {command!r}')
    return result


def main():
    import torch
    if torch.cuda.is_available() or torch.cuda.device_count():
        raise RuntimeError('CPU preparation refuses an attached GPU')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use the pinned Linux Python 3.12 Kaggle image')
    record('runtime', torch=torch.__version__, cuda_available=False)
    # CPU-only capability evidence. Full state isolation is never inferred from OS name.
    bwrap = shutil.which('bwrap')
    if not bwrap and shutil.which('apt-get'):
        run(['apt-get', 'update'], check=False, log=OUT / 'apt-update.log')
        run(['apt-get', 'install', '-y', 'bubblewrap'], check=False, log=OUT / 'apt-bwrap.log')
        bwrap = shutil.which('bwrap')
    command = ([bwrap, '--unshare-user', '--unshare-pid', '--unshare-net', '--die-with-parent',
                '--ro-bind', '/', '/', '--proc', '/proc', '--dev', '/dev',
                '--tmpfs', '/tmp', '--', '/bin/sh', '-c', 'printf vfarl-isolated'] if bwrap else [])
    isolation = run(command, check=False, log=OUT / 'bwrap.log') if command else None
    if bwrap:
        (OUT / 'bin').mkdir(exist_ok=True)
        shutil.copy2(bwrap, OUT / 'bin/bwrap')
    record('isolation', isolation={'command': command,
           'returncode': isolation.returncode if isolation else None,
           'available': bool(isolation and isolation.returncode == 0),
           'state_claim': 'namespace_isolation_available' if isolation and isolation.returncode == 0
                          else 'observable_replay_only'})

    caches = []
    for manifest in Path('/kaggle/input').rglob('cache-manifest.json'):
        value = json.loads(manifest.read_text())
        if value.get('source') == 'google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2':
            caches.append({'path': str(manifest.parent), 'manifest_sha256': digest(manifest),
                           'source': value['source']})
    if len(caches) != 1:
        raise RuntimeError(f'Expected one existing official NF4 cache: {caches}')
    record('cache_inventory', nf4_cache=caches[0], cache_copied=False, weights_loaded=False)
    roots = [p.parent for p in Path('/kaggle/input').rglob('tokenizer.json')
             if 'gemma-4-31b-it-qat-w4a16-ct' in str(p)]
    if len(roots) != 1:
        raise RuntimeError(f'Expected one official CT v2 tokenizer mount: {roots}')
    tokenizer = OUT / 'tokenizer'
    tokenizer.mkdir(exist_ok=True)
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                 'config.json', 'generation_config.json'):
        shutil.copy2(roots[0] / name, tokenizer / name)
    record('tokenizer', tokenizer={'source': 'google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2',
           'files': {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
                     for p in tokenizer.iterdir()}})

    wheelhouse = [p for p in Path('/kaggle/input').rglob('adk_submission-*.whl')]
    if len(wheelhouse) != 1:
        raise RuntimeError(f'Expected one official wheelhouse: {wheelhouse}')
    wheelhouse = wheelhouse[0].parent
    core_wheels = []
    for pattern in ('adk_submission-*.whl', 'adk_eval_core-*.whl', 'swegemma-*.whl'):
        found = list(wheelhouse.glob(pattern))
        if len(found) != 1:
            raise RuntimeError(f'Missing/ambiguous official wheel {pattern}')
        core_wheels.append(found[0])
    requested = ['transformers==5.13.1', 'peft==0.21.2', 'accelerate==1.15.0',
                 'safetensors==0.8.0', 'tokenizers==0.22.2', 'PyYAML==6.0.3',
                 'bitsandbytes==0.50.2', 'uv', *map(str, core_wheels)]
    resolution = OUT / 'runtime-resolution.json'
    constraint = OUT / 'stock-torch.txt'
    constraint.write_text('torch==' + torch.__version__ + '\n')
    run([sys.executable, '-m', 'pip', 'install', '--dry-run', '--report', str(resolution),
         '--find-links', str(wheelhouse), '-c', str(constraint), *requested],
        log=OUT / 'runtime-resolution.log')
    proposals = json.loads(resolution.read_text())['install']
    forbidden = [x['metadata']['name'] for x in proposals
                 if x['metadata']['name'].lower().replace('_', '-').startswith(('torch', 'nvidia-', 'triton'))]
    if forbidden:
        raise RuntimeError('Refuse replacing image CUDA packages: ' + repr(forbidden))
    wheels = OUT / 'runtime-wheels'
    wheels.mkdir(exist_ok=True)
    pins = {x['metadata']['name']: x['metadata']['name'] + '==' + x['metadata']['version']
            for x in proposals}
    pins.update({p.split('==')[0]: p for p in requested if '==' in p})
    pins.pop('swegemma', None)
    pins.pop('adk-eval-core', None)
    pins.pop('adk-submission', None)
    run([sys.executable, '-m', 'pip', 'download', '--no-deps', '--only-binary=:all:',
         '--find-links', str(wheelhouse), '-d', str(wheels), *pins.values()],
        log=OUT / 'runtime-download.log')
    for p in core_wheels:
        shutil.copy2(p, wheels / p.name)
    deps = OUT / 'runtime'
    run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--target', str(deps),
         *map(str, sorted(wheels.glob('*.whl')))], log=OUT / 'runtime-install.log')
    env = dict(os.environ, PYTHONPATH=str(deps))
    run([sys.executable, '-c', 'import torch,transformers,peft,bitsandbytes,swegemma,adk_submission,adk_eval_core; '
         'from transformers import AutoTokenizer; import sys; '
         't=AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True); '
         'print({"imports_ok":True,"tokenizer_size":len(t),"torch":torch.__version__})', str(tokenizer)],
        env=env, log=OUT / 'runtime-import.log')
    record('runtime_ready', runtime_wheels=[{'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)}
                                           for p in sorted(wheels.glob('*.whl'))])
    uv = deps / 'bin/uv'
    if not uv.is_file():
        raise RuntimeError('uv executable missing from staged runtime')
    python_root = OUT / 'python'
    env = dict(env, UV_PYTHON_INSTALL_DIR=str(python_root))
    profiles = ENVIRONMENT_PROFILES['profiles']
    for index, profile in enumerate(profiles):
        key = f'profile-{index:02d}'
        dest = OUT / 'task-envs' / key
        dest.parent.mkdir(exist_ok=True)
        match = re.search(r'/python(3\.\d+)/', profile['source_profile'])
        version = profile.get('python_version_pin') or (match.group(1) if match else '3.12')
        result = {'id': key, 'source_profile': profile['source_profile'],
                  'tasks': profile['tasks'], 'python_requested': version,
                  'python_abi': '.'.join(version.split('.')[:2]), 'success': False,
                  'excluded_host_packages': {'pipdeptree': 'Unpublished local development version; no task dependency.'} if 'pipdeptree' in (profile.get('packages') or {}) else {}}
        try:
            run([str(uv), 'python', 'install', version], env=env, log=OUT / f'{key}-python.log')
            run([str(uv), 'venv', '--python', version, str(dest)], env=env,
                log=OUT / f'{key}-venv.log')
            packages = profile.get('packages') or {}
            pins = [f'{name}=={version}' for name, version in packages.items()
                    if name.lower().replace('_', '-') not in ('pip', 'pipdeptree', 'swegemma', 'adk-submission', 'adk-eval-core')]
            if not pins:
                pins = ['pytest==8.3.3']
            run([str(uv), 'pip', 'install', '--python', str(dest / 'bin/python'),
                 '--find-links', str(wheelhouse), *pins], env=env, log=OUT / f'{key}-install.log')
            inventory = run([str(uv), 'pip', 'freeze', '--python', str(dest / 'bin/python')],
                            env=env, log=OUT / f'{key}-freeze.txt')
            result.update(success=True, environment_path=str(dest), actual_packages=inventory.stdout.splitlines())
        except Exception as exc:
            result['error'] = str(exc)
        REPORT['task_environments'].append(result)
        record('task_environment', last_task_environment=result)
    record('complete', success=all(p['success'] for p in REPORT['task_environments']))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        (OUT / 'exception.txt').write_text(traceback.format_exc())
        record('failed', success=False, error=traceback.format_exc())
        raise
    finally:
        # Opaque extension avoids Kaggle recursively unpacking venv/source archives.
        record('archive_start')
        with tarfile.open('/kaggle/working/vfarl-cpu-stage.bin', 'w:gz', compresslevel=1) as archive:
            archive.add(OUT, arcname=OUT.name)
        print(json.dumps({'phase': 'archive_completed', 'bytes': Path('/kaggle/working/vfarl-cpu-stage.bin').stat().st_size}), flush=True)
        shutil.copy2(OUT / 'preparation.json', '/kaggle/working/vfarl-preparation.json')
        shutil.rmtree(OUT)
