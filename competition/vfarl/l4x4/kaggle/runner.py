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
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import zlib

_SPAWNED = []
_INPUT_TREES = []
_OUTPUT = None


def spawn(command, **options):
    options['start_new_session'] = True
    process = subprocess.Popen(command, **options)
    _SPAWNED.append(process)
    return process


def stop_worker(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)
    except ProcessLookupError:
        process.wait()


def own_input_tree(path):
    if path.exists():
        raise RuntimeError('Preserve a previous unpacked input instead of overwriting it: ' + str(path))
    _INPUT_TREES.append(path)


def cleanup_experiment_inputs():
    if MODE not in ('train', 'prevalidate') or _OUTPUT is None:
        return
    record = {'removed_input_trees': [], 'all_model_workers_stopped': False}
    try:
        for process in _SPAWNED:
            stop_worker(process)
        record['all_model_workers_stopped'] = all(p.poll() is not None for p in _SPAWNED)
        if not record['all_model_workers_stopped']:
            raise RuntimeError('A model worker is still alive; preserve its runtime inputs')
        # Each lease belongs to a task UID allocated by this fresh kernel's
        # production jail. Clean escaped/background descendants before inputs.
        leases = Path('/tmp/vfarl-private-uid-leases')
        allocated = {int(p.name) for p in leases.iterdir() if p.name.isdigit()} if leases.is_dir() else set()
        active_uids = set()
        for status in Path('/proc').glob('[0-9]*/status'):
            try:
                fields = dict(line.split(':', 1) for line in status.read_text().splitlines() if ':' in line)
                if fields.get('State', '').strip()[:1] not in ('Z', 'X'):
                    active_uids.update(int(uid) for uid in fields.get('Uid', '').split() if int(uid) in allocated)
            except (FileNotFoundError, ProcessLookupError):
                pass
        if active_uids:
            sys.path.insert(0, '/kaggle/working/vfarl-cpu-stage/runtime')
            sys.path.insert(0, '/kaggle/working/vfarl-input/src')
            from competition.vfarl.l4x4.jail import quiesce
            for uid in active_uids:
                quiesce({'uid': uid})
        record['remaining_task_uids_cleaned'] = sorted(active_uids)
        record['removed_task_sandboxes'] = []
        record['preserved_interrupted_workspaces'] = []
        # SIGTERM can interrupt TaskEnvironment.close(). These are only the
        # live per-draw sandbox mirrors, including readonly runtime files;
        # durable state snapshots and all action/tool/verifier records live
        # beside this directory and remain intact.
        for path in _OUTPUT.glob('*worker-*/*/draw-*/sandboxes'):
            if path.is_dir() and not path.is_symlink() and path.resolve().is_relative_to(_OUTPUT.resolve()):
                roots = list(path.glob('swegemma_sandbox_*'))
                for jail in path.glob('vfarl-task-jail-*'):
                    roots.extend((jail / path.relative_to('/')).glob('swegemma_sandbox_*'))
                seen = set()
                for root in roots:
                    workspace = (root / 'workspace').resolve()
                    if (not workspace.is_dir() or workspace in seen
                            or not workspace.is_relative_to(path.resolve())):
                        continue
                    seen.add(workspace)
                    archive_path = path.parent / ('interrupted-workspace-' + root.name + '.tgz')
                    with tarfile.open(archive_path, 'w:gz', compresslevel=1, dereference=False) as archive:
                        archive.add(workspace, arcname='workspace')
                    record['preserved_interrupted_workspaces'].append({
                        'archive': str(archive_path), 'sha256': sha(archive_path),
                        'bytes': archive_path.stat().st_size, 'includes_git': True,
                        'scope': 'Actual task workspace only; excludes venv and readonly runtime mirrors'})
                shutil.rmtree(path)
                record['removed_task_sandboxes'].append(str(path))
        for path in _INPUT_TREES:
            if path.is_dir():
                shutil.rmtree(path)
                record['removed_input_trees'].append(str(path))
    except Exception as error:
        record['error'] = repr(error)
    (_OUTPUT / 'input-cleanup.json').write_text(json.dumps(record, indent=2) + '\n')


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


def local_nf4_cache(source, out):
    """Read the immutable mount sequentially once before four model readers."""
    manifest = json.loads((source / 'cache-manifest.json').read_text())
    files = manifest['files']
    total = sum(row['bytes'] for row in files)
    available = shutil.disk_usage('/tmp').free
    record = {'source': str(source), 'bytes': total, 'free_bytes_before': available,
              'reserve_bytes': 8 * 1024**3, 'copied': False,
              'source_manifest_sha256': sha(source / 'cache-manifest.json')}
    if available < total + record['reserve_bytes']:
        record['reason'] = 'Insufficient temporary disk to copy weights while reserving task/checkpoint space'
        print(json.dumps({'phase': 'local_nf4_cache', **record}), flush=True)
        (out / 'local-nf4-cache.json').write_text(json.dumps(record, indent=2) + '\n')
        return source
    target = Path('/tmp') / ('vfarl-local-nf4-cache-' + str(os.getpid()))
    own_input_tree(target)
    target.mkdir()
    started = time.monotonic()
    for row in files:
        name = row['name']
        if Path(name).name != name:
            raise RuntimeError('NF4 cache manifest filename escapes cache')
        copied, checksum = 0, hashlib.sha256()
        with (source / name).open('rb') as reader, (target / name).open('wb') as writer:
            for block in iter(lambda: reader.read(8 * 1024**2), b''):
                writer.write(block)
                checksum.update(block)
                copied += len(block)
        if copied != row['bytes'] or checksum.hexdigest() != row['sha256']:
            raise RuntimeError('Copied NF4 weights differ from immutable producer manifest: ' + name)
        (target / name).chmod(0o444)
        print(json.dumps({'phase': 'local_nf4_cache_file', 'name': name, 'bytes': copied,
                          'elapsed_seconds': time.monotonic() - started}), flush=True)
    shutil.copy2(source / 'cache-manifest.json', target / 'cache-manifest.json')
    record.update(copied=True, destination=str(target), elapsed_seconds=time.monotonic() - started,
                  verification='Every copied file matches the original producer size and SHA256')
    print(json.dumps({'phase': 'local_nf4_cache', **record}), flush=True)
    (out / 'local-nf4-cache.json').write_text(json.dumps(record, indent=2) + '\n')
    return target


def wait_workers(workers):
    positions = [0] * len(workers)
    pending = [''] * len(workers)
    last_gpu_observation = 0.0
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
                if isinstance(event, dict) and event.get('phase') in (
                        'prevalidate', 'collection', 'collection_error', 'model_loading', 'model_ready',
                        'policy_request', 'policy_action', 'episode_result', 'focal_group',
                        'empty_partition', 'jail_system_public_ca', 'actor_focal_scoring',
                        'actor_focal_score', 'probability_calibration', 'calibration_sample',
                        'calibration_complete', 'actor_calibration_sample',
                        'training_model_loading', 'training_model_ready',
                        'focal_backward_completed', 'optimizer_step_committed',
                        'training_update_completed'):
                    print(json.dumps({'worker': index, **event}), flush=True)
        if MODE == 'train' and time.monotonic() - last_gpu_observation >= 60:
            last_gpu_observation = time.monotonic()
            observation = {'phase': 'gpu_observation', 'time': time.time(),
                           'workers': [{'pid': proc.pid, 'returncode': proc.poll()}
                                       for proc, _ in workers]}
            for worker, (proc, log) in zip(observation['workers'], workers):
                try:
                    fields = dict(line.split(':', 1) for line in Path(f'/proc/{proc.pid}/status').read_text().splitlines() if ':' in line)
                    worker['host_process'] = {name: fields[name].strip()
                                              for name in ('State', 'VmRSS', 'VmSwap', 'Threads') if name in fields}
                    with Path(log.name).open('rb') as stream:
                        stream.seek(max(0, Path(log.name).stat().st_size - 800))
                        worker['raw_log_tail'] = stream.read().decode(errors='replace')
                except FileNotFoundError:
                    worker['host_process'] = {'State': 'process exited'}
            try:
                result = subprocess.run(['nvidia-smi',
                    '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=10)
                observation.update(returncode=result.returncode, gpu_rows=result.stdout.splitlines(),
                                   error=result.stderr.strip())
            except Exception as error:
                observation['error'] = repr(error)
            print(json.dumps(observation), flush=True)
            if _OUTPUT is not None:
                with (_OUTPUT / 'gpu-observations.jsonl').open('a') as stream:
                    stream.write(json.dumps(observation) + '\n')
        if fatal:
            # A real model/capacity error ends this experiment instead of
            # consuming the remaining 100-task schedule on the other GPUs.
            for proc, _ in workers:
                if proc.poll() is None:
                    stop_worker(proc)
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
            print(json.dumps({'phase': 'worker_failure', 'worker': index, 'exit_code': code,
                              'raw_log_tail': Path(log.name).read_text(errors='replace')[-10000:]}), flush=True)
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
    # Task selection may change before any policy sampling. Rebind new members
    # only to an already prepared, explicitly matching dependency profile.
    selected_profiles = {profile['source_profile']: profile for profile in profiles}
    for result in prep['task_environments']:
        if result['source_profile'] in selected_profiles:
            result['tasks'] = list(selected_profiles[result['source_profile']]['tasks'])
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
        norm = lambda values: {re.sub(r'[-_.]+', '-', name).lower(): version for name,version in values.items()}
        excluded_plugins = group.get('runtime_excluded_distributions', {})
        effective_packages = {name: version for name, version in group['packages'].items()
                              if re.sub(r'[-_.]+', '-', name).lower() not in excluded_plugins}
        wanted = norm(effective_packages)
        pin_hash = hashlib.sha256(json.dumps(wanted, sort_keys=True).encode()).hexdigest()[:10]
        target = stage / 'task-envs' / (group['id'] + '-' + pin_hash)
        result = {'id': group['id'], 'source_profile': 'explicit dependencies from original parent runtime',
                  'tasks': group['tasks'], 'python_requested': group['python'], 'success': False}
        # Keep the exact prior group when its requested pins match; a resolved
        # transitive extra is not a reason to overwrite a valid environment.
        reuse = None
        for old in sorted(prep['task_environments'], key=lambda row: row['id'] != group['id']):
            actual = dict(line.split('==', 1) for line in old.get('actual_packages', []) if '==' in line)
            actual_normalized = norm(actual)
            pins_match = all(actual_normalized.get(name) == version for name,version in wanted.items())
            same_or_exact = old['id'] == group['id'] or actual_normalized == wanted
            if (old.get('success') and old.get('python_requested') == group['python']
                    and pins_match and same_or_exact and Path(old['environment_path']).is_dir()):
                reuse = old
                break
        if reuse is not None:
            excluded = {re.sub(r'[-_.]+', '-', name).lower()
                        for name in group.get('excluded', {}).get('repair_distributions', [])}
            excluded.update(excluded_plugins)
            prohibited = {name:version for name,version in norm(dict(line.split('==',1)
                          for line in reuse['actual_packages'] if '==' in line)).items() if name in excluded}
            if prohibited:
                with (out / (group['id'] + '-remove-target-distribution.log')).open('w') as log:
                    subprocess.run([uv, 'pip', 'uninstall', '--python', reuse['environment_path'] + '/bin/python',
                                    *prohibited], env=managed, stdout=log, stderr=subprocess.STDOUT, check=True)
                freeze = subprocess.run([uv, 'pip', 'freeze', '--python', reuse['environment_path'] + '/bin/python'],
                                        env=managed, text=True, capture_output=True, check=True)
                reuse['actual_packages'] = freeze.stdout.splitlines()
                result['removed_distributions'] = prohibited
                result['runtime_exclusion_reasons'] = excluded_plugins
            result.update(success=True, environment_path=reuse['environment_path'],
                          actual_packages=reuse['actual_packages'], reused_profile_id=reuse['id'])
            prep['task_environments'].append(result)
            print(json.dumps({'phase': 'inherited_environment', **result}), flush=True)
            continue
        commands = [[uv, 'python', 'install', group['python']],
                    [uv, 'venv', '--clear', '--python', group['python'], str(target)],
                    [uv, 'pip', 'install', '--no-deps', '--python', str(target / 'bin/python'),
                     *[name + '==' + version for name,version in effective_packages.items()]]]
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
    for profile in profiles:
        additions = profile.get('linux_additional_packages', {})
        if not additions or (ONLY_TASKS and not set(profile['tasks']) & set(ONLY_TASKS)):
            continue
        for prepared in prep['task_environments']:
            if prepared['source_profile'] != profile['source_profile'] or not prepared['success']:
                continue
            actual = dict(line.split('==', 1) for line in prepared['actual_packages'] if '==' in line)
            missing = {name: version for name, version in additions.items() if actual.get(name) != version}
            if missing:
                command = [uv, 'pip', 'install', '--no-deps', '--python',
                           prepared['environment_path'] + '/bin/python',
                           *[name + '==' + version for name, version in missing.items()]]
                with (out / (prepared['id'] + '-runtime-resources.log')).open('w') as log:
                    result = subprocess.run(command, env=managed, stdout=log, stderr=subprocess.STDOUT)
                correction = {'profile': prepared['id'], 'added_packages': missing,
                              'source': profile.get('linux_additional_packages_source'),
                              'returncode': result.returncode}
                corrections.append(correction)
                print(json.dumps({'phase': 'runtime_resources', **correction}), flush=True)
                if result.returncode:
                    prepared.update(success=False, error='Required runtime resource installation failed')
                    continue
                freeze = subprocess.run([uv, 'pip', 'freeze', '--python', prepared['environment_path'] + '/bin/python'],
                                        env=managed, text=True, capture_output=True, check=True)
                prepared['actual_packages'] = freeze.stdout.splitlines()
            prepared['linux_additional_packages'] = additions
            prepared['linux_additional_packages_source'] = profile.get('linux_additional_packages_source')
    for profile in profiles:
        declaration = profile.get('linux_source_metadata')
        if not declaration or (ONLY_TASKS and declaration['instance_id'] not in ONLY_TASKS):
            continue
        prepared = next(row for row in prep['task_environments']
                        if row['source_profile'] == profile['source_profile'])
        if not prepared['success']:
            continue
        asset = source.parent / 'assets' / declaration['instance_id']
        result = dict(declaration, package_code_installed=False,
                      source_snapshot_sha256=sha(asset / 'source-snapshot.tgz'))
        try:
            with tempfile.TemporaryDirectory(prefix='vfarl-source-metadata-') as temporary:
                temporary = Path(temporary)
                workspace, backend, metadata = [temporary / name for name in ('workspace', 'backend', 'metadata')]
                workspace.mkdir()
                metadata.mkdir()
                with tarfile.open(asset / 'source-snapshot.tgz') as archive:
                    archive.extractall(workspace, filter='data')
                if sha(workspace / 'pyproject.toml') != declaration['pyproject_sha256']:
                    raise RuntimeError('Source distribution metadata pyproject identity changed')
                with (out / (prepared['id'] + '-source-metadata.log')).open('w') as log:
                    subprocess.run([uv, 'pip', 'install', '--no-deps', '--target', str(backend),
                                    declaration['backend_requirement']], env=managed,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                    command = [sys.executable, '-c',
                        'import json,sys,importlib.metadata as m; from flit_core.buildapi import prepare_metadata_for_build_wheel; '
                        'name=prepare_metadata_for_build_wheel(sys.argv[1]); '
                        'print(json.dumps({"directory":name,"backend_version":m.version("flit_core")}))', str(metadata)]
                    built = subprocess.run(command, cwd=workspace, env=dict(managed, PYTHONPATH=str(backend)),
                                           text=True, capture_output=True)
                    log.write(built.stdout + built.stderr)
                    built.check_returncode()
                details = json.loads(built.stdout.strip().splitlines()[-1])
                expected = declaration['name'].replace('-', '_') + '-' + declaration['version'] + '.dist-info'
                if details['directory'] != expected:
                    raise RuntimeError('Source backend produced an unexpected distribution metadata identity')
                abi = prepared.get('python_abi') or '.'.join(prepared['python_requested'].split('.')[:2])
                destination = Path(prepared['environment_path']) / 'lib' / ('python' + abi) / 'site-packages' / expected
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(metadata / expected, destination)
                result.update(backend_version=details['backend_version'],
                              files={str(p.relative_to(destination)): sha(p) for p in destination.rglob('*') if p.is_file()})
            observed = subprocess.run([prepared['environment_path'] + '/bin/python', '-I', '-c',
                'import importlib.metadata,sys; print(importlib.metadata.version(sys.argv[1]))', declaration['name']],
                text=True, capture_output=True, check=True)
            if observed.stdout.strip() != declaration['version']:
                raise RuntimeError('Installed source distribution metadata version differs from pyproject')
            freeze = subprocess.run([uv, 'pip', 'freeze', '--python', prepared['environment_path'] + '/bin/python'],
                                    env=managed, text=True, capture_output=True, check=True)
            prepared['actual_packages'] = freeze.stdout.splitlines()
            prepared['source_distribution_metadata'] = result
            result['success'] = True
        except Exception as error:
            prepared.update(success=False, error='Source metadata preparation failed: ' + repr(error))
            result.update(success=False, error=repr(error))
        corrections.append(result)
        print(json.dumps({'phase': 'source_distribution_metadata', **result}), flush=True)
    prep['success'] = all(p['success'] for p in prep['task_environments'])
    (stage / 'preparation.json').write_text(json.dumps(prep, indent=2) + '\n')
    (out / 'runtime-preparation-repair.json').write_text(json.dumps(corrections, indent=2) + '\n')


def run_probability_calibration(source, config_path, out, env, evidence):
    """Score real recorded actions on actor/FSDP paths without an optimizer."""
    data = source / CALIBRATION_DATA
    if not data.is_file():
        raise RuntimeError('The frozen real-action calibration manifest is missing')
    record = {'source_manifest': CALIBRATION_DATA, 'source_manifest_sha256': sha(data),
              'optimizer_requested': False, 'stages': []}
    evidence['probability_calibration'] = record
    (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
    for stage in ('actor', 'fsdp'):
        destination = out / ('calibration-' + stage)
        arguments = ['-m', 'training.train_vfarl', '--config', str(config_path),
                     '--data', str(data), '--calibrate', stage, '--out', str(destination)]
        if stage == 'actor':
            command = [sys.executable, *arguments]
            stage_env = dict(env, CUDA_VISIBLE_DEVICES='0,1')
        else:
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nproc-per-node=4', '--max-restarts=0', *arguments]
            stage_env = dict(env, CUDA_VISIBLE_DEVICES='0,1,2,3',
                FSDP_TRANSFORMER_CLS_TO_WRAP='Gemma4TextDecoderLayer',
                NCCL_DEBUG='WARN', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        started = time.monotonic()
        print(json.dumps({'phase': 'probability_calibration_started', 'stage': stage,
                          'source_manifest_sha256': record['source_manifest_sha256'],
                          'optimizer_requested': False}), flush=True)
        with (out / ('calibration-' + stage + '.log')).open('w') as log:
            process = spawn(command, env=stage_env, cwd=source, stdout=log, stderr=subprocess.STDOUT)
            failures = wait_workers([(process, log)])
        stage_record = {'stage': stage, 'exit_code': process.returncode,
                        'seconds': time.monotonic() - started,
                        'output': str(destination), 'optimizer_requested': False}
        record['stages'].append(stage_record)
        (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps({'phase': 'probability_calibration_finished', **stage_record}), flush=True)
        if failures:
            raise RuntimeError('The actual zero-update probability study failed in ' + stage)
    record['completed'] = True
    (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')


def task_scopes(selection):
    """Keep independent research comparison tasks outside reward collection."""
    task_ids = [row['instance_id'] for row in selection['tasks']]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise RuntimeError('The frozen task selection must be nonempty and unique')
    split = selection.get('research_split')
    if split is None:
        return task_ids, task_ids, []
    collection = split['collection_task_ids']
    validation = split['validation_task_ids']
    if (not collection or len(collection) != len(set(collection))
            or len(validation) != len(set(validation))
            or set(collection) & set(validation)
            or set(collection) | set(validation) != set(task_ids)):
        raise RuntimeError('Research collection/validation identities must partition the frozen tasks')
    return task_ids, collection, validation


def named_tasks(common, task_ids):
    arguments, index = [], 0
    while index < len(common):
        if common[index] == '--only-task':
            index += 2
        else:
            arguments.append(common[index])
            index += 1
    for task in task_ids:
        arguments += ['--only-task', task]
    return arguments


def fixed_rollouts(source, config_path, prevalidation, out, env, common, task_ids,
                   prefix, *, adapter_run=None, focal=None, task_limit=None):
    """Use the same frozen task order, policy budget and seeds before/after."""
    workers = []
    try:
        for rank in range(2):
            command = named_tasks(common, task_ids) + [
                '--config', str(config_path), '--prevalidation', str(prevalidation),
                '--context', str(CONTEXT_LENGTH), '--draws', '1', '--max-states', '0',
                '--policy-gpus', '2', '--policy-backend', POLICY_BACKEND,
                '--worker-index', str(rank), '--worker-count', '2',
                '--out', str(out / f'{prefix}-{rank}')]
            if adapter_run is not None:
                command += ['--adapter-run', str(adapter_run)]
            if focal is not None:
                command += ['--score-focal-data', str(focal)]
            if task_limit is not None:
                command += ['--task-limit', str(task_limit)]
            log = (out / f'{prefix}-{rank}.log').open('w')
            process = spawn(command, env=dict(env, CUDA_VISIBLE_DEVICES=('0,1' if rank == 0 else '2,3')),
                            cwd=source, stdout=log, stderr=subprocess.STDOUT)
            workers.append((process, log))
        failures = wait_workers(workers)
        if failures:
            raise RuntimeError('Fixed policy comparison failed in ' + prefix + ': ' + str(failures))
    finally:
        for process, log in workers:
            if process.poll() is None:
                stop_worker(process)
            if not log.closed:
                log.close()
    results = []
    for rank in range(2):
        path = out / f'{prefix}-{rank}/collection.jsonl'
        if path.is_file():
            results.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    by_id = {row['instance_id']: row for row in results}
    expected = task_ids[:task_limit] if task_limit is not None else task_ids
    if set(by_id) != set(expected) or len(results) != len(expected):
        raise RuntimeError('Fixed policy comparison did not return exactly the requested task identities')
    return [by_id[task] for task in expected]


def reuse_collection(out, evidence):
    """Copy one reviewed, untrained root focal batch without new sampling.

    The immutable source cannot mark itself consumed. The new update records
    this exact batch SHA, and the research coordinator must not submit it again.
    """
    carriers = [path for path in Path('/kaggle/input').rglob('carrier.json')
                if path.parent.name == 'vfarl-train'
                and sha(path) == COLLECTION_CARRIER_SHA256]
    if len(carriers) != 1:
        raise RuntimeError('Require one mounted completed collection with the reviewed carrier SHA256')
    carrier = carriers[0]
    root = carrier.parent
    original = json.loads(carrier.read_text())
    if (original.get('mode') != 'train' or original.get('collect_only') is not True
            or original.get('one_update_completed') is not False
            or original.get('optimizer_requested') is not False
            or 'wall_seconds' not in original
            or any((root / 'update' / name).exists() for name in
                   ('optimizer-step.json', 'training_manifest.json'))):
        raise RuntimeError('Only a completed collection-only batch with no prior update may be continued')
    for key in ('material_sha256', 'runtime_sha256',
                'collection_task_ids', 'independent_validation_task_ids', 'training_config'):
        if original.get(key) != evidence.get(key):
            raise RuntimeError('Completed collection differs from this frozen continuation: ' + key)
    # A learner implementation repair must not force another expensive actor
    # collection. Preserve both revisions while binding all actor/tool sources.
    learner_source = 'training/train_vfarl.py'
    learner_sources = {learner_source, 'training/train_vfarl_unsloth.py'}
    original_sources = original.get('source_sha256', {})
    current_sources = evidence.get('source_sha256', {})
    if ({key: value for key, value in original_sources.items() if key not in learner_sources}
            != {key: value for key, value in current_sources.items() if key not in learner_sources}):
        raise RuntimeError('Completed collection actor, tools or preparation sources differ')
    if original.get('actor_policy_backend') != POLICY_BACKEND:
        raise RuntimeError('Completed collection actor backend differs from this continuation')
    focal = root / 'focal.jsonl'
    if not focal.is_file() or sha(focal) != COLLECTION_FOCAL_SHA256:
        raise RuntimeError('Completed root focal batch differs from the reviewed SHA256')
    allowed = set(evidence['collection_task_ids'])
    for line in focal.read_text().splitlines():
        if line.strip() and json.loads(line).get('instance_id') not in allowed:
            raise RuntimeError('Reused focal record is outside the frozen collection task scope')
    shutil.copyfile(focal, out / 'focal.jsonl')
    before = root / 'actor-before.jsonl'
    if before.is_file():
        shutil.copyfile(before, out / 'actor-before.jsonl')
    for rank in range(2):
        previous = root / f'worker-{rank}' / 'collection.jsonl'
        if previous.is_file():
            target = out / f'worker-{rank}' / 'collection.jsonl'
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(previous, target)
    record = dict(kernel=COLLECTION_KERNEL, carrier_sha256=COLLECTION_CARRIER_SHA256,
                  focal_sha256=COLLECTION_FOCAL_SHA256, focal_bytes=focal.stat().st_size,
                  new_collection_performed=False, original_wall_seconds=original['wall_seconds'],
                  collection_learner_source_sha256=original_sources.get(learner_source),
                  current_learner_source_sha256=current_sources.get(learner_source),
                  collection_learner_sources={key: original_sources.get(key) for key in sorted(learner_sources)},
                  current_learner_sources={key: current_sources.get(key) for key in sorted(learner_sources)},
                  original_dynamic_stopping=original.get('dynamic_stopping'),
                  original_maximum_focal_states=original.get('maximum_focal_states'),
                  consumption_rule='One reward update for this exact behavior batch. Preserve this SHA in research records and never resubmit it after a committed optimizer step.')
    evidence['reused_collection'] = record
    evidence['gpu_topology'].update(collector_processes=0, focal_state_quotas=[0, 0])
    print(json.dumps({'phase': 'completed_collection_reused', **record}), flush=True)
    (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
    return out / 'focal.jsonl'


def main():
    global _OUTPUT
    started = time.monotonic()
    out = Path('/kaggle/working/vfarl-' + MODE)
    out.mkdir(exist_ok=False)
    _OUTPUT = out
    if MODE == 'capabilities':
        record_capabilities(out)
        return
    material_archive = unique(MATERIAL_ARCHIVE_NAME)
    if sha(material_archive) != MATERIAL_SHA256:
        raise RuntimeError('Frozen task material archive identity changed')
    own_input_tree(Path('/kaggle/working/vfarl-input'))
    material_filter = None
    if ONLY_TASKS:
        with tarfile.open(material_archive) as archive:
            frozen_selection = json.load(archive.extractfile('vfarl-input/selection.json'))
        _, _, comparison_ids = task_scopes(frozen_selection)
        included_ids = set(ONLY_TASKS) | (set(comparison_ids) if MODE == 'train' and not COLLECT_ONLY else set())
        prefixes = tuple('vfarl-input/assets/' + task + '/' for task in included_ids)
        material_filter = lambda name: (name.endswith(('.json', '.jsonl')) or '/src/' in name
                                         or name.startswith(prefixes))
    unpack(material_archive, Path('/kaggle/working'), include=material_filter)
    material = Path('/kaggle/working/vfarl-input')
    selection = json.loads((material / 'selection.json').read_text())
    task_ids, collection_ids, validation_ids = task_scopes(selection)
    if not set(ONLY_TASKS) <= set(task_ids):
        raise RuntimeError('Requested task is outside the frozen material selection')
    if MODE == 'train' and ONLY_TASKS:
        if not set(ONLY_TASKS) <= set(collection_ids):
            raise RuntimeError('Reward collection cannot include independent validation tasks')
        collection_ids = [task for task in collection_ids if task in ONLY_TASKS]
    runtime_archive = unique('vfarl-cpu-stage.bin')
    actual_runtime_sha = sha(runtime_archive)
    if RUNTIME_SHA256 and actual_runtime_sha != RUNTIME_SHA256:
        raise RuntimeError('Completed runtime archive identity changed')
    own_input_tree(Path('/kaggle/working/vfarl-cpu-stage'))
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
    else:
        # A later pre-policy selection can reuse an immutable dependency
        # profile. This binds membership only; task usability still comes from
        # that task's actual CPU acceptance record below.
        profile_inventory = json.loads((source / 'competition/vfarl/l4x4/kaggle/environment-profiles.json').read_text())['profiles']
        selected = {profile['source_profile']: profile for profile in profile_inventory}
        for prepared in prep['task_environments']:
            if prepared['source_profile'] != 'stock' and prepared['source_profile'] in selected:
                prepared['tasks'] = list(selected[prepared['source_profile']]['tasks'])
    by_task = {task: p for p in prep['task_environments'] for task in p['tasks']}
    by_profile = {p['id']: p for p in prep['task_environments']}
    selected_entries = {row['instance_id']: row for row in selection['tasks']}
    for asset in (material / 'assets').iterdir():
        task = json.loads((asset / 'task.json').read_text())
        declared_profile = selected_entries[task['instance_id']].get('linux_environment_profile_id')
        actual = by_profile.get(declared_profile) if declared_profile else by_task.get(task['instance_id'])
        config = json.loads((asset / 'runtime.json').read_text())
        historical = json.loads((asset / 'run-summary.json').read_text())
        config['site_packages_mode'] = historical.get('task_site_packages_mode') or 'pth'
        config['site_packages_mode_source'] = 'original run-summary; pth only where no explicit copy mode was recorded'
        config['source_console_scripts'] = {}
        if task['instance_id'].startswith('mahmoud__glom.'):
            config['source_console_scripts']['glom'] = 'glom.cli:console_main'
        if task['instance_id'].startswith('theskumar__python-dotenv.'):
            config['source_console_scripts']['dotenv'] = 'dotenv.__main__:cli'
        if task['instance_id'].startswith('iterative__dvc.'):
            config['source_console_scripts']['dvc'] = 'dvc.cli:main'
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
    if MODE == 'train':
        env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
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
                'isolation': isolation, 'started_at': time.time(),
                'frozen_task_count': len(task_ids),
                'collection_task_ids': collection_ids, 'independent_validation_task_ids': validation_ids}
    if MODE == 'train':
        evidence['torch_allocator_configuration'] = env['PYTORCH_ALLOC_CONF']
    (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
    if MODE == 'prevalidate':
        worker_count = min(4, len(ONLY_TASKS)) if ONLY_TASKS else 4
        workers = []
        try:
            for rank in range(worker_count):
                log = (out / f'prevalidate-{rank}.log').open('w')
                command = common + ['--prevalidate', '--worker-index', str(rank), '--worker-count', str(worker_count),
                                    '--out', str(out / f'tasks-{rank}')]
                workers.append((spawn(command, env=env, cwd=source, stdout=log,
                                                  stderr=subprocess.STDOUT), log))
            failures = wait_workers(workers)
            if failures:
                raise RuntimeError('CPU verification worker failed: ' + str(failures))
        finally:
            for proc, log in workers:
                if proc.poll() is None:
                    stop_worker(proc)
                if not log.closed:
                    log.close()
        with (out / 'prevalidation.jsonl').open('w') as merged:
            for rank in range(worker_count):
                merged.write((out / f'tasks-{rank}/prevalidation.jsonl').read_text())
        if ONLY_TASKS and not EXPORT_RUNTIME:
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
    merged_reports = ([source / name for name in PREVALIDATION_REPORTS] if PREVALIDATION_REPORTS else
                      [p for p in Path('/kaggle/input').rglob('prevalidation.jsonl')
                       if p.parent.name == 'vfarl-prevalidate'])
    if not merged_reports or (not PREVALIDATION_REPORTS and len(merged_reports) != 1):
        raise RuntimeError('Require explicitly frozen CPU reports or one mounted merged report')
    selected_ids = set(task_ids)
    by_id, source_records = {}, []
    for report in merged_reports:
        rows = [json.loads(line) for line in report.read_text().splitlines() if line.strip()]
        source_record = {'path': str(report), 'sha256': sha(report), 'rows': len(rows),
                         'reused_task_ids': [], 'superseded_task_ids': []}
        for row in rows:
            key = row['instance_id']
            if key not in selected_ids:
                continue
            if key in by_id:
                source_record['superseded_task_ids'].append(key)
            by_id[key] = row
            source_record['reused_task_ids'].append(key)
        source_records.append(source_record)
    valid = [by_id[key] for key in sorted(by_id)]
    prevalidation = out / 'reused-prevalidation.jsonl'
    prevalidation.write_text(''.join(json.dumps(row) + '\n' for row in valid))
    (out / 'prevalidation-reuse-sources.json').write_text(json.dumps(source_records, indent=2) + '\n')
    if set(by_id) != selected_ids:
        raise RuntimeError('Missing actual Linux prevalidation records for frozen tasks: ' +
                           ', '.join(sorted(selected_ids - set(by_id))))
    all_usable = {r['instance_id'] for r in valid if r.get('usable')}
    usable = all_usable & set(collection_ids)
    if not usable:
        raise RuntimeError('No requested task has a real usable baseline/repaired environment')
    if not COLLECT_ONLY and not CALIBRATION_ONLY and not set(validation_ids) <= all_usable:
        raise RuntimeError('Independent comparison requires usable CPU environments for every frozen validation task')
    common = named_tasks(common, [task for task in collection_ids if task in usable])
    evidence.update(planned_task_pool=len(task_ids), usable_task_ids=sorted(usable), usable_task_count=len(usable),
                    unavailable_task_count=len(task_ids) - len(all_usable),
                    sampling_scope='Only individually CPU-accepted tasks; other pool members are not claimed as sampled.')
    focal_quotas = ((FOCAL_STATES, FOCAL_STATES) if DYNAMIC_QUEUE else
                    ((FOCAL_STATES + 1) // 2, FOCAL_STATES // 2))
    evidence.update(collect_only=COLLECT_ONLY, maximum_focal_states=FOCAL_STATES,
                    calibration_only=CALIBRATION_ONLY,
                    calibration_manifest=CALIBRATION_DATA,
                    actor_policy_backend=POLICY_BACKEND,
                    learner_backend=LEARNER_BACKEND,
                    dynamic_queue=DYNAMIC_QUEUE,
                    dynamic_stopping=dict(stop_after_mixed_groups=STOP_AFTER_MIXED_GROUPS,
                        stop_after_tasks=STOP_AFTER_TASKS,
                        stop_after_seconds=STOP_AFTER_SECONDS,
                        rule='Stop new task claims and new state reservations only; complete active state groups and keep all results',
                        mixed_eligibility='The same prepare_rows complete/distinct-action criterion as the learner',
                        limitation='Stopping on observed mixed groups is an adaptive focal curriculum, not an unbiased task-population estimate'))
    evidence['gpu_topology'] = {'collector_processes': 0 if CALIBRATION_ONLY or FOCAL_DATA or COLLECTION_KERNEL else 2, 'gpus_per_policy': 2,
                               'visible_device_groups': ['0,1', '2,3'],
                               'focal_state_quotas': list(focal_quotas),
                               'focal_quota_semantics': 'shared global reservations' if DYNAMIC_QUEUE else 'fixed per-worker quotas',
                               'branch_parallelism': 'Branches within one reserved task/state remain serial',
                               'training_ranks': 0 if COLLECT_ONLY or CALIBRATION_ONLY else 1 if LEARNER_BACKEND == 'unsloth' else 4,
                               'training_visible_gpus': 0 if COLLECT_ONLY or CALIBRATION_ONLY else 2 if LEARNER_BACKEND == 'unsloth' else 4,
                               'calibration_actor_gpus': 2 if CALIBRATION_DATA else 0,
                               'calibration_fsdp_ranks': 4 if CALIBRATION_DATA else 0,
                               'post_update_task_limit': 0 if COLLECT_ONLY or CALIBRATION_ONLY or UPDATE_ONLY else len(validation_ids) or 4,
                               'post_update_collectors': 0 if COLLECT_ONLY or CALIBRATION_ONLY or UPDATE_ONLY else 2}
    (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
    caches = []
    for p in Path('/kaggle/input').rglob('cache-manifest.json'):
        record = json.loads(p.read_text())
        if record.get('source') == 'google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2':
            caches.append(p.parent)
    if len(caches) != 1:
        raise RuntimeError('Missing/ambiguous immutable official NF4 cache')
    model_cache = local_nf4_cache(caches[0], out)
    # Use the stage's dependency path in this coordinator only for config loading.
    sys.path.insert(0, str(stage / 'runtime'))
    sys.path.insert(0, str(source))
    import yaml
    training_config_path = source / TRAINING_CONFIG
    config = yaml.safe_load(training_config_path.read_text())
    evidence['training_config'] = {'path': TRAINING_CONFIG,
        'sha256': sha(training_config_path),
        'declared_context': config['model']['max_sequence_length'],
        'effective_context': CONTEXT_LENGTH}
    config['model'].update(model_name_or_path=str(model_cache),
        tokenizer_name_or_path=str(stage / 'tokenizer'),
        chat_template_path=str(stage / 'tokenizer/chat_template.jinja'),
        tool_schema_path=str(source / 'competition/vfarl/six-tools.json'), device='cuda:0',
        max_sequence_length=CONTEXT_LENGTH)
    config_path = out / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
    if CALIBRATION_ONLY:
        run_probability_calibration(source, config_path, out, env, evidence)
        evidence.update(one_update_completed=False, optimizer_requested=False,
                        wall_seconds=time.monotonic() - started)
        (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
        return
    if FOCAL_DATA:
        from training.train_vfarl import prepare_rows
        recovered = source / FOCAL_DATA
        recovery = json.loads((source / FOCAL_PROVENANCE).read_text())
        if sha(recovered) != recovery['focal_sha256']:
            raise RuntimeError('Recovered focal data differs from its original byte-bound record')
        if any(json.loads(line).get('instance_id') not in set(collection_ids)
               for line in recovered.read_text().splitlines() if line.strip()):
            raise RuntimeError('Recovered data contains a task outside the declared training scope')
        reward_rows, recovered_groups = prepare_rows(recovered, CONTEXT_LENGTH)
        if not reward_rows:
            raise RuntimeError('Recovered data has no complete verified mixed state group')
        focal = out / 'focal.jsonl'
        shutil.copyfile(recovered, focal)
        evidence['recovered_focal'] = recovery
        evidence['focal_groups'] = recovered_groups
        evidence['gpu_topology'].update(collector_processes=0, focal_state_quotas=[0, 0])
        print(json.dumps(dict(phase='recovered_focal_loaded', focal_sha256=sha(focal),
            mixed_groups=recovered_groups['mixed_group_count'], sampled_actions=len(reward_rows),
            new_collection_performed=False)), flush=True)
        (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
    elif COLLECTION_KERNEL:
        focal = reuse_collection(out, evidence)
    else:
        task_queue = out / 'task-queue.json'
        if DYNAMIC_QUEUE:
            from competition.vfarl.l4x4.run import TaskQueue
            selected_entries = [row for row in selection['tasks'] if row['instance_id'] in usable]
            TaskQueue(task_queue, [row['instance_id'] for row in selected_entries],
                selection_sha256=sha(material / 'selection.json'), maximum_states=FOCAL_STATES,
                stop_after_mixed_groups=STOP_AFTER_MIXED_GROUPS, stop_after_tasks=STOP_AFTER_TASKS,
                stop_after_seconds=STOP_AFTER_SECONDS,
                execution_contract=dict(config_sha256=sha(config_path), policy_backend=POLICY_BACKEND,
                    context=CONTEXT_LENGTH, draws=1, branches=3, turns=32, tool_calls=32,
                    turn_tokens=1536, episode_tokens=8192))
            evidence['task_queue'] = dict(path=str(task_queue),
                frozen_task_order=[row['instance_id'] for row in selected_entries],
                unit='Whole task with its declared draws; branches remain serial within each reserved state')
            (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
        active = []
        try:
            for rank, states in enumerate(focal_quotas):
                command = common + ['--config', str(config_path), '--prevalidation', str(prevalidation),
                    '--context', str(CONTEXT_LENGTH), '--draws', '1', '--branches', '3', '--max-states', str(states),
                    '--policy-gpus', '2', '--worker-index', str(rank), '--worker-count', '2',
                    '--policy-backend', POLICY_BACKEND,
                    '--out', str(out / f'worker-{rank}')]
                if DYNAMIC_QUEUE:
                    command += ['--task-queue', str(task_queue),
                        '--stop-after-mixed-groups', str(STOP_AFTER_MIXED_GROUPS),
                        '--stop-after-tasks', str(STOP_AFTER_TASKS),
                        '--stop-after-seconds', str(STOP_AFTER_SECONDS)]
                log = (out / f'worker-{rank}.log').open('w')
                proc = spawn(command, env=dict(env, CUDA_VISIBLE_DEVICES=('0,1' if rank == 0 else '2,3')),
                                        cwd=source, stdout=log, stderr=subprocess.STDOUT)
                active.append((proc, log))
            failed = wait_workers(active)
            if failed:
                raise RuntimeError('Collection process failed; retain partial evidence: ' + str(failed))
        finally:
            for proc, log in active:
                if proc.poll() is None:
                    stop_worker(proc)
                if not log.closed:
                    log.close()
        focal = out / 'focal.jsonl'
        with focal.open('w') as merged:
            for rank in range(2):
                path = out / f'worker-{rank}/focal.jsonl'
                if path.is_file():
                    merged.write(path.read_text())
        actor_before = out / 'actor-before.jsonl'
        with actor_before.open('w') as merged:
            for rank in range(2):
                path = out / f'worker-{rank}/actor-before.jsonl'
                if path.is_file():
                    merged.write(path.read_text())
    if CALIBRATION_DATA:
        run_probability_calibration(source, config_path, out, env, evidence)
    if COLLECT_ONLY:
        evidence.update(one_update_completed=False, optimizer_requested=False,
                        wall_seconds=time.monotonic() - started)
        (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
        print(json.dumps({'phase': 'collection_only_complete', 'optimizer_requested': False,
                          'maximum_focal_states': FOCAL_STATES}), flush=True)
        return
    evidence['training_behavior_batch_sha256'] = sha(focal)
    validation_before = []
    validation_after_optimizer = bool(validation_ids and
                                      (COLLECTION_KERNEL or LEARNER_BACKEND == 'unsloth'))
    measurement_order = []
    if validation_ids and not UPDATE_ONLY:
        # Spend comparison rollouts only after this collection produced actual
        # usable reward differences. These tasks never supplied focal rewards.
        from training.train_vfarl import prepare_rows
        reward_rows, group_summary = prepare_rows(focal, CONTEXT_LENGTH)
        evidence['focal_groups'] = group_summary
        if reward_rows and not validation_after_optimizer:
            print(json.dumps({'phase': 'independent_validation_started', 'stage': 'before',
                              'task_ids': validation_ids}), flush=True)
            validation_before = fixed_rollouts(source, config_path, prevalidation, out, env,
                common, validation_ids, 'research-before')
            evidence['independent_validation_before'] = validation_before
            measurement_order.append('unupdated_policy_baseline')
        elif reward_rows:
            # The completed collection already fixes the data for this one
            # update. Measure the independent control afterwards by loading
            # the original base and fresh seed0 adapter, never the update.
            evidence['independent_validation_note'] = (
                'The unupdated-policy baseline is measured after the optimizer step; '
                'research-before denotes policy identity, not wall-clock order. '
                'It uses the same frozen base, fresh config-seed zero-B adapter, '
                'task order, sampling seeds and budgets.')
        else:
            evidence['independent_validation_note'] = 'No mixed sampled-action group; no update comparison was requested.'
        evidence['measurement_order'] = measurement_order
        evidence['actual_update_before_baseline'] = False
        (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
    # All rollout model processes have exited before allocating the FSDP model.
    if LEARNER_BACKEND == 'unsloth':
        train_env = dict(env, CUDA_VISIBLE_DEVICES='0,1')
        for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE'):
            train_env.pop(key, None)
        command = [sys.executable, '-m', 'training.train_vfarl_unsloth',
                   '--config', str(config_path), '--data', str(focal), '--out', str(out / 'update'),
                   '--actor-backend', POLICY_BACKEND, '--learner-backend', 'unsloth', '--policy-gpus', '2']
        learner_ranks, learner_gpus = 1, 2
    else:
        train_env = dict(env, FSDP_TRANSFORMER_CLS_TO_WRAP='Gemma4TextDecoderLayer',
                         NCCL_DEBUG='WARN', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
                   '--max-restarts=0', '-m', 'training.train_vfarl', '--config', str(config_path),
                   '--data', str(focal), '--out', str(out / 'update'),
                   '--actor-backend', POLICY_BACKEND, '--learner-backend', LEARNER_BACKEND]
        learner_ranks, learner_gpus = 4, 4
    evidence['training_execution'] = dict(backend=LEARNER_BACKEND, processes=learner_ranks,
                                          visible_gpus=learner_gpus)
    with (out / 'update.log').open('w') as log:
        print(json.dumps({'phase': 'training_update_started', 'ranks': learner_ranks,
                          'visible_gpus': learner_gpus, 'backend': LEARNER_BACKEND,
                          'focal_data': str(focal)}), flush=True)
        train_process = spawn(command, env=train_env, cwd=source, stdout=log, stderr=subprocess.STDOUT)
        wait_workers([(train_process, log)])
    manifest_path = out / 'update/training_manifest.json'
    updated = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    evidence['one_update_completed'] = updated.get('optimizer_steps') == 1
    if validation_ids and evidence['one_update_completed']:
        measurement_order.append('optimizer_update')
    evidence['training_process_exit_code'] = train_process.returncode
    evidence['post_update_diagnostics_completed'] = updated.get('post_update_diagnostics_completed', False)
    print(json.dumps({'phase': 'training_update_finished', 'exit_code': train_process.returncode,
                      'optimizer_steps': updated.get('optimizer_steps', 0),
                      'post_update_diagnostics_completed': evidence['post_update_diagnostics_completed']}), flush=True)
    (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
    if train_process.returncode and not evidence['one_update_completed']:
        raise RuntimeError('Training failed before a real committed optimizer step; inspect retained update.log')
    if train_process.returncode:
        evidence['post_update_diagnostic_failure'] = (
            'The optimizer step and adapter were committed before the failed diagnostic phase. '
            'This batch is not trained again; only independent post-update rollout follows.')
    if UPDATE_ONLY:
        evidence['independent_validation_note'] = 'Deferred in this short real optimizer/timing run; no independent improvement is claimed.'
        evidence['wall_seconds'] = time.monotonic() - started
        (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
        return
    if evidence['one_update_completed']:
        post_ids = [task for task in collection_ids if task in usable][:4]
        if validation_ids:
            if validation_after_optimizer:
                evidence['actual_update_before_baseline'] = True
                (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
                print(json.dumps({'phase': 'independent_validation_started', 'stage': 'before',
                                  'wall_clock_stage': 'after_optimizer',
                                  'policy': 'original-base-fresh-config-seed-zero-B-adapter',
                                  'task_ids': validation_ids}), flush=True)
                validation_before = fixed_rollouts(source, config_path, prevalidation, out, env,
                    common, validation_ids, 'research-before')
                evidence['independent_validation_before'] = validation_before
                measurement_order.append('unupdated_policy_baseline')
            if len(validation_before) != len(validation_ids):
                raise RuntimeError('Preserve the committed update; independent pre-update comparison is incomplete')
            post_ids = validation_ids
        elif DYNAMIC_QUEUE:
            # Pair post-update rollouts only with originals that actually
            # completed before adaptive stopping, preserving frozen order.
            completed_before = set()
            for rank in range(2):
                path = out / f'worker-{rank}/collection.jsonl'
                if path.is_file():
                    completed_before.update(row['instance_id'] for row in
                        (json.loads(line) for line in path.read_text().splitlines() if line.strip())
                        if row.get('verified') is True and row.get('reward') in (0, 1))
            post_ids = [row['instance_id'] for row in selection['tasks']
                if row['instance_id'] in completed_before][:4]
        if not post_ids:
            raise RuntimeError('A committed focal update has no task for a paired post-update run')
        evidence['post_update_task_ids'] = post_ids
        prefix = 'research-after' if validation_ids else 'post-worker'
        (out / 'carrier.json').write_text(json.dumps(evidence, indent=2) + '\n')
        validation_after = fixed_rollouts(source, config_path, prevalidation, out, env,
            common, post_ids, prefix, adapter_run=out / 'update', focal=focal)
        if validation_ids:
            measurement_order.append('updated_policy_rollouts')
            pairs = []
            for before, after in zip(validation_before, validation_after):
                pairs.append(dict(instance_id=before['instance_id'],
                    before_reward=before['reward'], after_reward=after['reward'],
                    before_verified=before['verified'], after_verified=after['verified'],
                    before_verification=before['verification'], after_verification=after['verification'],
                    before_policy_id=before['policy_id'], after_policy_id=after['policy_id']))
            comparison = dict(task_ids=validation_ids, pairs=pairs,
                selection_sha256=sha(material / 'selection.json'),
                context=CONTEXT_LENGTH, draws=1, focal_states=0,
                measurement_order=measurement_order,
                actual_update_before_baseline=validation_after_optimizer,
                baseline_policy='Original frozen base and fresh config-seed zero-B adapter; research-before describes policy identity.',
                limitation='Small independent research comparison within allowed train sources; no official heldout score.')
            (out / 'research-validation.json').write_text(json.dumps(comparison, indent=2) + '\n')
            evidence['independent_validation_after'] = validation_after
            evidence['independent_validation_comparison'] = str(out / 'research-validation.json')
        with (out / 'actor-after.jsonl').open('w') as merged:
            for rank in range(2):
                path = out / f'{prefix}-{rank}/actor-after.jsonl'
                if path.is_file():
                    merged.write(path.read_text())
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
    finally:
        cleanup_experiment_inputs()
