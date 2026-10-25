"""Kaggle lifecycle: reuse CPU runtime, prevalidate or collect then update once.

This file is packaged by build_runner.py. Importing/building does not submit a job.
"""
from pathlib import Path
import base64
import copy
import ctypes
import ctypes.util
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zlib


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unique(pattern):
    paths = list(Path('/kaggle/input').rglob(pattern))
    if len(paths) != 1:
        raise RuntimeError('Missing/ambiguous mounted input ' + pattern + ': ' + repr(paths))
    return paths[0]


def wait_workers(workers):
    positions = [0] * len(workers)
    pending = [''] * len(workers)
    while True:
        running = False
        fatal = False
        for index, (proc, log) in enumerate(workers):
            code = proc.poll()
            running = running or code is None
            fatal = fatal or code not in (None, 0)
            with Path(log.name).open() as reader:
                reader.seek(positions[index])
                pending[index] += reader.read()
                positions[index] = reader.tell()
            lines = pending[index].split('\n')
            pending[index] = lines.pop()
            for line in lines:
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(event, dict) and event.get('phase') in ('prevalidate', 'collection'):
                    print(json.dumps({'worker': index, **event}), flush=True)
        if fatal:
            # A real model/capacity error ends this experiment instead of
            # consuming the remaining 100-task schedule on the other GPUs.
            for proc, _ in workers:
                if proc.poll() is None:
                    proc.terminate()
            break
        if not running:
            break
        time.sleep(2)
    failures = []
    for index, (proc, log) in enumerate(workers):
        code = proc.wait()
        log.close()
        if code:
            failures.append((index, code))
    return failures


def unpack(path, root, include=None):
    root = root.resolve()
    def safe_filter(member, destination):
        if include is not None and not include(member.name):
            return None
        if member.issym() and os.path.isabs(member.linkname):
            target = Path(member.linkname).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError('Archive symlink escapes extraction root')
            member = copy.copy(member)
            member.linkname = os.path.relpath(target, (root / member.name).parent)
        return tarfile.data_filter(member, destination)
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError('Archive path escape')
        tar.extractall(root, filter=safe_filter)


def record_capabilities(out):
    probes = {'uid': os.getuid(), 'euid': os.geteuid(),
              'process_status': Path('/proc/self/status').read_text(),
              'libseccomp': ctypes.util.find_library('seccomp')}
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    probes['landlock'] = {'abi': abi, 'errno': ctypes.get_errno() if abi < 0 else 0}
    commands = [
        ['unshare', '--mount', '/bin/true'],
        ['unshare', '--pid', '--fork', '/bin/true'],
        ['unshare', '--net', '/bin/true'],
        [sys.executable, '-c', 'import tempfile,os; d=tempfile.mkdtemp(prefix="vfarl-chroot-"); '
         'os.chdir(d); os.chroot(d); print("chroot succeeded",os.getcwd())'],
        [sys.executable, '-c', 'import ctypes; l=ctypes.CDLL(None,use_errno=True); '
         'r=l.prctl(38,1,0,0,0); print({"no_new_privs":r,"errno":ctypes.get_errno()})'],
    ]
    probes['commands'] = []
    for command in commands:
        try:
            result = subprocess.run(command, text=True, capture_output=True)
            probes['commands'].append({'command': command, 'returncode': result.returncode,
                                       'stdout': result.stdout, 'stderr': result.stderr})
        except Exception as exc:
            probes['commands'].append({'command': command, 'error': str(exc)})
    (out / 'isolation-capabilities.json').write_text(json.dumps(probes, indent=2) + '\n')
    print(json.dumps({'phase': 'isolation_capabilities', **probes}), flush=True)


def prepare_cpu_repairs(stage, source, out, prep):
    if not ONLY_TASKS:
        record_capabilities(out)
    profiles = json.loads((source / 'competition/vfarl/l4x4/kaggle/environment-profiles.json').read_text())['profiles']
    corrections = []
    uv_path = str(stage / 'runtime/bin/uv')
    managed_env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(stage / 'python'))
    for result in prep['task_environments']:
        original = next((p for p in profiles if p['source_profile'] == result['source_profile']), None)
        pin = original.get('python_version_pin') if original else None
        if (not pin or result['python_requested'] == pin
                or (ONLY_TASKS and not set(result['tasks']) & set(ONLY_TASKS))):
            continue
        target = stage / 'task-envs' / (result['id'] + '-py' + pin.replace('.', ''))
        old_request = result['python_requested']
        result.update(success=False, python_requested=pin, python_abi='.'.join(pin.split('.')[:2]),
                      environment_path=str(target), python_patch_repair_failed=True)
        commands = [[uv_path, 'python', 'install', pin],
                    [uv_path, 'venv', '--python', pin, str(target)],
                    [uv_path, 'pip', 'install', '--python', str(target / 'bin/python'),
                     *[name + '==' + version for name,version in original['packages'].items()
                       if name.lower().replace('_','-') not in ('pip', 'pipdeptree', 'swegemma', 'adk-submission', 'adk-eval-core')]]]
        try:
            with (out / (result['id'] + '-python-patch-repair.log')).open('w') as log:
                for command in commands:
                    subprocess.run(command, env=managed_env, stdout=log, stderr=subprocess.STDOUT, check=True)
            freeze = subprocess.run([uv_path, 'pip', 'freeze', '--python', str(target / 'bin/python')],
                                    env=managed_env, text=True, capture_output=True, check=True)
            result.update(success=True, actual_packages=freeze.stdout.splitlines(), python_patch_repair_failed=False,
                          previous_python_request=old_request, python_pin_reason=original['python_version_pin_reason'])
        except Exception as exc:
            result['error'] = str(exc)
        print(json.dumps({'phase': 'python_patch_repair', 'id': result['id'],
                          'python': pin, 'success': result['success'], 'error': result.get('error')}), flush=True)
    for result in prep['task_environments']:
        if result['success'] or result.get('python_patch_repair_failed'):
            continue
        original = next((p for p in profiles if p['source_profile'] == result['source_profile']), None)
        if original is None:
            continue  # An inherited stock profile is recreated below from its own pins.
        excluded = {'pipdeptree': 'Local development version is unpublished; unrelated host dependency-inspection package.'}
        pins = [name + '==' + version for name,version in original['packages'].items()
                if name.lower().replace('_', '-') not in ('pip', 'pipdeptree', 'swegemma', 'adk-submission', 'adk-eval-core')]
        target = stage / 'task-envs' / result['id']
        command = [str(stage / 'runtime/bin/uv'), 'pip', 'install', '--python', str(target / 'bin/python'),
                   '--find-links', str(stage / 'runtime-wheels'), *pins]
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out / (result['id'] + '-repair.log')).write_text(proc.stdout)
        correction = {'profile': result['id'], 'excluded_packages': excluded,
                      'returncode': proc.returncode, 'command': command}
        corrections.append(correction)
        print(json.dumps({'phase': 'environment_repair', **correction}), flush=True)
        if proc.returncode == 0:
            freeze = subprocess.run([str(stage / 'runtime/bin/uv'), 'pip', 'freeze', '--python',
                                     str(target / 'bin/python')], text=True, capture_output=True, check=True)
            result.update(success=True, environment_path=str(target), actual_packages=freeze.stdout.splitlines(),
                          excluded_packages=excluded, previous_error=result.pop('error', None))
    prep['success'] = all(p['success'] for p in prep['task_environments'])
    prep['linux_repairs'] = corrections
    stock = json.loads((source / 'competition/vfarl/l4x4/kaggle/stock-environment-pins.json').read_text())
    stock_groups = stock['profiles']
    if ONLY_TASKS:
        stock_groups = [group for group in stock_groups if set(group['tasks']) & set(ONLY_TASKS)]
    assigned = {task for group in stock_groups for task in group['tasks']}
    for previous in prep['task_environments']:
        previous['tasks'] = [task for task in previous['tasks'] if task not in assigned]
    uv = str(stage / 'runtime/bin/uv')
    managed = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(stage / 'python'))
    for group in stock_groups:
        target = stage / 'task-envs' / group['id']
        result = {'id': group['id'], 'source_profile': 'explicit dependencies from original parent runtime',
                  'tasks': group['tasks'], 'python_requested': group['python'], 'success': False}
        # For a concrete failure reproduction, reuse an exactly matching
        # completed venv rather than reinstalling the same four pytest pins.
        reuse = None
        for old in prep['task_environments']:
            actual = dict(line.split('==', 1) for line in old.get('actual_packages', []) if '==' in line)
            norm = lambda values: {name.lower().replace('_', '-'): version for name,version in values.items()}
            if (old.get('success') and old.get('python_requested') == group['python']
                    and norm(actual) == norm(group['packages']) and Path(old['environment_path']).is_dir()):
                reuse = old
                break
        if reuse is not None:
            result.update(success=True, environment_path=reuse['environment_path'],
                          actual_packages=reuse['actual_packages'], reused_profile_id=reuse['id'])
            prep['task_environments'].append(result)
            print(json.dumps({'phase': 'inherited_environment', **result}), flush=True)
            continue
        commands = [[uv, 'python', 'install', group['python']],
                    [uv, 'venv', '--python', group['python'], str(target)],
                    [uv, 'pip', 'install', '--python', str(target / 'bin/python'),
                     *[name + '==' + version for name,version in group['packages'].items()]]]
        try:
            with (out / (group['id'] + '-prepare.log')).open('w') as log:
                for command in commands:
                    subprocess.run(command, env=managed, stdout=log, stderr=subprocess.STDOUT, check=True)
            freeze = subprocess.run([uv, 'pip', 'freeze', '--python', str(target / 'bin/python')],
                                    env=managed, text=True, capture_output=True, check=True)
            result.update(success=True, environment_path=str(target), actual_packages=freeze.stdout.splitlines())
        except Exception as exc:
            result['error'] = str(exc)
        prep['task_environments'].append(result)
        print(json.dumps({'phase': 'inherited_environment', **result}), flush=True)
    prep['success'] = all(p['success'] for p in prep['task_environments'])
    (stage / 'preparation.json').write_text(json.dumps(prep, indent=2) + '\n')
    (out / 'runtime-preparation-repair.json').write_text(json.dumps(corrections, indent=2) + '\n')


def main():
    started = time.monotonic()
    out = Path('/kaggle/working/vfarl-' + MODE)
    out.mkdir(exist_ok=False)
    if MODE == 'capabilities':
        record_capabilities(out)
        return
    material_archive = unique('vfarl-train100.bin')
    if sha(material_archive) != MATERIAL_SHA256:
        raise RuntimeError('Frozen train100 archive identity changed')
    material_filter = None
    if ONLY_TASKS:
        prefixes = tuple('vfarl-input/assets/' + task + '/' for task in ONLY_TASKS)
        material_filter = lambda name: (name.endswith(('.json', '.jsonl')) or '/src/' in name
                                         or name.startswith(prefixes))
    unpack(material_archive, Path('/kaggle/working'), include=material_filter)
    material = Path('/kaggle/working/vfarl-input')
    runtime_archive = unique('vfarl-cpu-stage.bin')
    actual_runtime_sha = sha(runtime_archive)
    if RUNTIME_SHA256 and actual_runtime_sha != RUNTIME_SHA256:
        raise RuntimeError('Completed runtime archive identity changed')
    runtime_filter = None
    if ONLY_TASKS == ['Knio__dominate.9082227e.pr_166']:
        runtime_filter = lambda name: any(part in name for part in (
            '/runtime/', '/python/cpython-3.12', '/task-envs/profile-00/', '/preparation.json'))
    unpack(runtime_archive, Path('/kaggle/working'), include=runtime_filter)
    stage = Path('/kaggle/working/vfarl-cpu-stage')
    prep = json.loads((stage / 'preparation.json').read_text())
    if not prep.get('runtime_wheels'):
        raise RuntimeError('CPU stage did not complete runtime imports')
    # Replace the packaged source with this carrier's independently frozen source.
    source = material / 'src'
    bundle = json.loads(zlib.decompress(base64.b64decode(SOURCE_BUNDLE)))
    for relative, content in bundle.items():
        target = source / relative
        if not target.resolve().is_relative_to(source.resolve()):
            raise RuntimeError('Source bundle path escape')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    if MODE == 'prevalidate':
        prepare_cpu_repairs(stage, source, out, prep)
    by_task = {task: p for p in prep['task_environments'] for task in p['tasks']}
    for asset in (material / 'assets').iterdir():
        task = json.loads((asset / 'task.json').read_text())
        actual = by_task.get(task['instance_id'])
        config = json.loads((asset / 'runtime.json').read_text())
        historical = json.loads((asset / 'run-summary.json').read_text())
        config['site_packages_mode'] = historical.get('task_site_packages_mode') or 'pth'
        config['site_packages_mode_source'] = 'original run-summary; pth only where no explicit copy mode was recorded'
        config['source_console_scripts'] = {}
        if task['instance_id'].startswith('mahmoud__glom.'):
            config['source_console_scripts']['glom'] = 'glom.cli:console_main'
        if task['instance_id'].startswith('theskumar__python-dotenv.'):
            config['source_console_scripts']['dotenv'] = 'dotenv.__main__:cli'
        if task['instance_id'].startswith('Instagram__MonkeyType.'):
            config['site_packages_mode'] = 'copy'
        config['linux_environment_ready'] = bool(actual and actual['success'])
        config['linux_preparation'] = actual
        if actual and actual.get('environment_path'):
            config['python'] = actual['environment_path'] + '/bin/python'
            abi = actual.get('python_abi') or '.'.join(actual['python_requested'].split('.')[:2])
            config['site_packages'] = actual['environment_path'] + '/lib/python' + abi + '/site-packages'
            config['environment_profile_id'] = actual['id']
        if actual and actual['success']:
            config['current_packages'] = dict(line.split('==', 1) for line in actual['actual_packages'] if '==' in line)
            config['current_packages_are_source_expectations'] = False
        (asset / 'runtime.json').write_text(json.dumps(config, indent=2) + '\n')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(stage / 'runtime'), str(source)]),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               PATH=str(stage / 'bin') + os.pathsep + os.environ.get('PATH', ''))
    # Kaggle denied all namespace probes. CPU task acceptance and GPU rollout
    # must use the same production chroot executor and its live capability probe.
    isolation = 'chroot'
    common = [sys.executable, '-m', 'competition.vfarl.l4x4.run',
              '--selection', str(material / 'selection.json'),
              '--questions', str(material / 'questions.jsonl'),
              '--assets', str(material / 'assets'),
              '--tools', str(source / 'competition/vfarl/six-tools.json'),
              '--isolation', isolation]
    for task in ONLY_TASKS:
        common.extend(['--only-task', task])
    evidence = {'mode': MODE, 'material_sha256': MATERIAL_SHA256, 'runtime_sha256': actual_runtime_sha,
                'source_sha256': {p: hashlib.sha256(s.encode()).hexdigest() for p,s in bundle.items()},
                'isolation': isolation, 'started_at': time.time()}
    (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
    if MODE == 'prevalidate':
        worker_count = min(4, len(ONLY_TASKS)) if ONLY_TASKS else 4
        workers = []
        try:
            for rank in range(worker_count):
                log = (out / f'prevalidate-{rank}.log').open('w')
                command = common + ['--prevalidate', '--worker-index', str(rank), '--worker-count', str(worker_count),
                                    '--out', str(out / f'tasks-{rank}')]
                workers.append((subprocess.Popen(command, env=env, cwd=source, stdout=log,
                                                  stderr=subprocess.STDOUT), log))
            failures = wait_workers(workers)
            if failures:
                raise RuntimeError('CPU verification worker failed: ' + str(failures))
        finally:
            for proc, log in workers:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
                if not log.closed:
                    log.close()
        with (out / 'prevalidation.jsonl').open('w') as merged:
            for rank in range(worker_count):
                merged.write((out / f'tasks-{rank}/prevalidation.jsonl').read_text())
        if ONLY_TASKS:
            shutil.rmtree(stage)
            shutil.rmtree(material)
            return
        archive_path = Path('/kaggle/working/vfarl-cpu-stage.bin')
        with tarfile.open(archive_path, 'w:gz', compresslevel=1) as archive:
            archive.add(stage, arcname=stage.name)
        (out / 'runtime-archive.json').write_text(json.dumps({'sha256': sha(archive_path),
                'bytes': archive_path.stat().st_size, 'source_runtime_sha256': actual_runtime_sha}, indent=2) + '\n')
        # The immutable archive contains all runtime files; publishing the raw
        # venv forest duplicates gigabytes and tens of thousands of small files.
        shutil.rmtree(stage)
        shutil.rmtree(material)
        return
    # Hardware and runtime are measured again on the actual GPU worker.
    probe = subprocess.run([sys.executable, '-c',
        'import torch,json; print(json.dumps({"count":torch.cuda.device_count(),'
        '"gpus":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))'],
        env=env, text=True, capture_output=True, check=True)
    hardware = json.loads(probe.stdout.strip().splitlines()[-1])
    if hardware['count'] != 4 or any('L4' not in name for name in hardware['gpus']):
        raise RuntimeError('This carrier requires the measured four-L4 Kaggle accelerator: ' + str(hardware))
    (out / 'hardware.json').write_text(json.dumps(hardware, indent=2) + '\n')
    merged_reports = [p for p in Path('/kaggle/input').rglob('prevalidation.jsonl')
                      if p.parent.name == 'vfarl-prevalidate']
    if len(merged_reports) != 1:
        raise RuntimeError('Expected one merged 100-task Linux prevalidation report')
    prevalidation = merged_reports[0]
    valid = [json.loads(line) for line in prevalidation.read_text().splitlines() if line.strip()]
    if len(valid) != 100 or len({r['instance_id'] for r in valid}) != 100:
        raise RuntimeError('Require actual Linux prevalidation records for the frozen 100-task pool')
    usable = {r['instance_id'] for r in valid if r.get('usable')}
    if ONLY_TASKS:
        usable &= set(ONLY_TASKS)
    if not usable:
        raise RuntimeError('No requested task has a real usable baseline/repaired environment')
    evidence.update(planned_task_pool=100, usable_task_ids=sorted(usable), usable_task_count=len(usable),
                    unavailable_task_count=100 - len({r['instance_id'] for r in valid if r.get('usable')}),
                    sampling_scope='Only individually CPU-accepted tasks; other pool members are not claimed as sampled.')
    (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
    caches = []
    for p in Path('/kaggle/input').rglob('cache-manifest.json'):
        record = json.loads(p.read_text())
        if record.get('source') == 'google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2':
            caches.append(p.parent)
    if len(caches) != 1:
        raise RuntimeError('Missing/ambiguous immutable official NF4 cache')
    # Use the stage's dependency path in this coordinator only for config loading.
    sys.path.insert(0, str(stage / 'runtime'))
    import yaml
    config = yaml.safe_load((source / 'training/configs/vfarl_l4x4.yaml').read_text())
    config['model'].update(model_name_or_path=str(caches[0]),
        tokenizer_name_or_path=str(stage / 'tokenizer'),
        chat_template_path=str(stage / 'tokenizer/chat_template.jinja'),
        tool_schema_path=str(source / 'competition/vfarl/six-tools.json'), device='cuda:0',
        max_sequence_length=CONTEXT_LENGTH)
    config_path = out / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    active = []
    try:
        for rank, states in enumerate((2, 1, 1, 1)):
            command = common + ['--config', str(config_path), '--prevalidation', str(prevalidation),
                '--context', str(CONTEXT_LENGTH), '--draws', '1', '--branches', '3', '--max-states', str(states),
                '--worker-index', str(rank), '--worker-count', '4', '--out', str(out / f'worker-{rank}')]
            log = (out / f'worker-{rank}.log').open('w')
            proc = subprocess.Popen(command, env=dict(env, CUDA_VISIBLE_DEVICES=str(rank)),
                                    cwd=source, stdout=log, stderr=subprocess.STDOUT)
            active.append((proc, log))
        failed = wait_workers(active)
        if failed:
            raise RuntimeError('Collection process failed; retain partial evidence: ' + str(failed))
    finally:
        for proc, log in active:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            if not log.closed:
                log.close()
    focal = out / 'focal.jsonl'
    with focal.open('w') as merged:
        for rank in range(4):
            path = out / f'worker-{rank}/focal.jsonl'
            if path.is_file():
                merged.write(path.read_text())
    # All rollout model processes have exited before allocating the FSDP model.
    train_env = dict(env, FSDP_TRANSFORMER_CLS_TO_WRAP='Gemma4TextDecoderLayer',
                     NCCL_DEBUG='WARN', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
               '--max-restarts=0', '-m', 'training.train_vfarl', '--config', str(config_path),
               '--data', str(focal), '--out', str(out / 'update')]
    with (out / 'update.log').open('w') as log:
        subprocess.run(command, env=train_env, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
    manifest_path = out / 'update/training_manifest.json'
    updated = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    evidence['one_update_completed'] = updated.get('optimizer_steps') == 1
    if evidence['one_update_completed']:
        post = []
        try:
            for rank in range(4):
                command = common + ['--config', str(config_path), '--prevalidation', str(prevalidation),
                    '--adapter-run', str(out / 'update'), '--context', str(CONTEXT_LENGTH), '--draws', '1',
                    '--max-states', '0', '--worker-index', str(rank), '--worker-count', '100',
                    '--out', str(out / f'post-worker-{rank}')]
                log = (out / f'post-worker-{rank}.log').open('w')
                proc = subprocess.Popen(command, env=dict(env, CUDA_VISIBLE_DEVICES=str(rank)),
                                        cwd=source, stdout=log, stderr=subprocess.STDOUT)
                post.append((proc, log))
            failures = wait_workers(post)
            evidence['post_update_process_failures'] = failures
            if failures:
                raise RuntimeError('Post-update collection failed: ' + str(failures))
        finally:
            for proc, log in post:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
                if not log.closed:
                    log.close()
    else:
        evidence['learning_signal_note'] = 'No optimizer-step manifest; inspect real focal outcomes and update.log.'
    evidence['wall_seconds'] = time.monotonic() - started
    (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        Path('/kaggle/working/vfarl-exception.txt').write_text(traceback.format_exc())
        raise
