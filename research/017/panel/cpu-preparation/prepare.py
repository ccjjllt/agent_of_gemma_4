"""Actual Linux CPU preparation of frozen C sources; no model calls or new tests.

Reuses the P015 independent-venv, upstream criterion and buggy/reference design.
FREEZE is injected by build_carrier.py from real acquired source files.
"""
import hashlib
import configparser
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import venv
import zipfile

DEADLINE = None


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def execute(command, log, cwd=None, env=None, timeout=180):
    if DEADLINE is not None:
        remaining = DEADLINE - time.monotonic() - 120
        if remaining <= 0:
            raise RuntimeError('CPU experiment deadline reached; reserve final artifacts')
        timeout = min(timeout, remaining)
    start = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
    save(log, output)
    return {'command': command, 'returncode': 124 if timed_out else process.returncode,
            'timed_out': timed_out, 'seconds': time.monotonic() - start, 'tail': output[-3000:]}


def require_ok(result, description):
    if result['returncode']:
        raise RuntimeError(description + ': exit ' + str(result['returncode']))


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def main():
    global DEADLINE
    DEADLINE = time.monotonic() + FREEZE['wall_seconds']
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('Run this CPU carrier in the pinned Linux/Python3.12 image')
    out = Path('/kaggle/working/p017-confirmation-preparation')
    out.mkdir(parents=True, exist_ok=False)
    scratch = Path('/tmp/p017-confirmation-preparation')
    scratch.mkdir(parents=True, exist_ok=False)
    archives = list(Path('/kaggle/input').rglob(FREEZE['archive_name']))
    if len(archives) != 1:
        raise RuntimeError('Expected one frozen opaque source archive')
    source = archives[0]
    if source.stat().st_size != FREEZE['archive_bytes'] or digest(source) != FREEZE['archive_sha256']:
        raise RuntimeError('CPU source archive differs from frozen bytes')
    material = scratch / 'source'
    material.mkdir()
    with tarfile.open(source, 'r:*') as archive:
        archive.extractall(material, filter='data')
    for name, pin in FREEZE['files'].items():
        path = material / name
        if path.stat().st_size != pin['bytes'] or digest(path) != pin['sha256']:
            raise RuntimeError('Source member differs: ' + name)
    tasks = {task['instance_id']: task for task in json.loads((material / 'agent-tasks.json').read_text())}
    plans = json.loads((material / 'environment-plans.json').read_text())
    if set(tasks) != set(FREEZE['task_ids']) or set(plans) != set(tasks):
        raise RuntimeError('Source identities disagree')
    prepared = scratch / 'prepared'
    prepared.mkdir()
    save(prepared / 'agent-tasks.json', [tasks[iid] for iid in FREEZE['task_ids']])
    shared_wheels = prepared / 'environments/shared-wheels'
    shared_wheels.mkdir(parents=True)
    for cached in sorted((material / 'cached-wheels').glob('*.whl')):
        shutil.copyfile(cached, shared_wheels / cached.name)
    criteria, records = [], []
    for iid in FREEZE['task_ids']:
        if DEADLINE - time.monotonic() < 180:
            break
        row = json.loads((material / 'host-source' / (iid + '.json')).read_text())
        plan = plans[iid]
        logs = out / 'logs' / iid
        run = scratch / 'runs' / iid
        run.mkdir(parents=True)
        env_dir = prepared / 'environments' / iid
        env_dir.mkdir()
        req = env_dir / 'requirements.txt'
        save(req, '\n'.join(plan['requirements']) + '\n')
        save(env_dir / 'source-adaptation.json', plan)
        wheelhouse = env_dir / 'wheelhouse'
        wheelhouse.mkdir()
        criterion_path = prepared / 'criterion-patches' / (iid + '.patch')
        if row.get('test_patch'):
            save(criterion_path, row['test_patch'])
        reference_path = run / 'reference.patch'
        save(reference_path, row['patch'])
        snapshot = material / 'snapshots' / (iid + '.tar.gz')
        (prepared / 'snapshots').mkdir(exist_ok=True)
        shutil.copyfile(snapshot, prepared / 'snapshots' / snapshot.name)
        item = {'iid': iid, 'instance_id': iid, 'split': 'dev', 'module': plan['module'],
                'f2p_nodes': row['FAIL_TO_PASS'], 'p2p_nodes': row['PASS_TO_PASS'],
                'timeout_seconds': 180,
                'environment': {'requirements': str(req.relative_to(prepared)),
                    'wheelhouse': str(wheelhouse.relative_to(prepared)),
                    'install_editable': plan['install_editable'], 'pytest_plugins': plan['pytest_plugins'],
                    'pytest_args': plan['pytest_args']},
                'source_sha256': FREEZE['files']['host-source/' + iid + '.json']['sha256'],
                'source_python': plan['source_python'], 'p2p_coverage': 'source_nodes' if row['PASS_TO_PASS'] else 'not_provided'}
        if row.get('test_patch'):
            item['criterion_patch_path'] = str(criterion_path.relative_to(prepared))
        criteria.append(item)
        record = {'iid': iid, 'split': 'dev', 'python': sys.version, 'started': time.time(),
                  'source_python': plan['source_python'], 'f2p_nodes': len(row['FAIL_TO_PASS']),
                  'p2p_nodes': len(row['PASS_TO_PASS']), 'p2p_coverage': item['p2p_coverage'],
                  'valid_prepost': False, 'reference_content_visible_to_agent': False}
        try:
            # Preparation may use network; all later task execution uses exactly
            # the resulting wheel bytes with --no-index and no dependency resolver.
            offline = ['--no-index'] if FREEZE.get('offline_wheel_cache') else []
            wheel = execute([sys.executable, '-m', 'pip', 'wheel', *offline, '--no-deps', '--wheel-dir', str(shared_wheels),
                             '--find-links', str(shared_wheels), '-r', str(req)], logs / 'wheel-build.log', timeout=600)
            record['wheel_build'] = wheel
            require_ok(wheel, 'Source-pinned wheel preparation failed')
            available = {}
            for path in shared_wheels.glob('*.whl'):
                parts = path.name.split('-')
                available[(canonical(parts[0]), parts[1])] = path
            actual_plugins = []
            for requirement in plan['requirements']:
                name, version = requirement.split('==')
                path = available.get((canonical(name), version))
                if path is None:
                    raise RuntimeError('Exact source wheel absent: ' + requirement)
                (wheelhouse / path.name).symlink_to(os.path.relpath(path, wheelhouse))
                if canonical(name) in plan['pytest_plugin_packages']:
                    with zipfile.ZipFile(path) as wheel_zip:
                        entrypoints = [n for n in wheel_zip.namelist() if n.endswith('.dist-info/entry_points.txt')]
                        if len(entrypoints) != 1:
                            raise RuntimeError('Expected actual pytest entry points: ' + name)
                        parsed = configparser.ConfigParser()
                        parsed.read_string(wheel_zip.read(entrypoints[0]).decode())
                        actual_plugins.extend(parsed['pytest11'].values())
            plan['pytest_plugins'] = sorted(set(actual_plugins))
            item['environment']['pytest_plugins'] = plan['pytest_plugins']
            record['actual_pytest_plugins'] = plan['pytest_plugins']
            save(env_dir / 'source-adaptation.json', plan)
            env_root = run / 'venv'
            venv.EnvBuilder(with_pip=False, system_site_packages=True, symlinks=True).create(env_root)
            python = env_root / 'bin/python'
            env = dict(os.environ, PATH=str(env_root / 'bin') + os.pathsep + os.environ.get('PATH', ''),
                       PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTEST_PLUGINS=','.join(plan['pytest_plugins']),
                       PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
                       LANG='C.UTF-8', LC_ALL='C.UTF-8')
            installed = execute([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps', '--force-reinstall',
                                 '--find-links', str(wheelhouse), '-r', str(req)], logs / 'offline-install.log', env=env)
            record['offline_install'] = installed
            require_ok(installed, 'Offline dependency installation failed')
            for state in ('buggy', 'reference'):
                checkout = run / state
                checkout.mkdir()
                with tarfile.open(snapshot, 'r:*') as archive:
                    archive.extractall(checkout, filter='data')
                repo = checkout / 'repo'
                head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
                if head != tasks[iid]['base_commit']:
                    raise RuntimeError('Snapshot HEAD differs from frozen base commit')
                record[state + '_head'] = head
                env['PYTHONPATH'] = str(repo / 'src') + os.pathsep + str(repo)
                if state == 'reference':
                    result = execute(['git', 'apply', str(reference_path)], logs / 'reference-apply.log', cwd=repo)
                    require_ok(result, 'Reference patch failed to apply')
                if row.get('test_patch'):
                    result = execute(['git', 'apply', str(criterion_path)], logs / (state + '-criterion-apply.log'), cwd=repo)
                    require_ok(result, 'Existing source criterion patch failed to apply')
                if plan['install_editable']:
                    result = execute([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-build-isolation',
                                      '--find-links', str(wheelhouse), '-e', str(repo)],
                                     logs / (state + '-editable-install.log'), cwd=repo, env=env)
                    record[state + '_editable'] = result
                    require_ok(result, 'Offline source editable installation failed')
                origin_code = ('import importlib.util,json,pathlib; s=importlib.util.find_spec(' + repr(plan['module'])
                    + '); p=pathlib.Path(s.origin).resolve() if s and s.origin else None; '
                    'print(json.dumps({"origin":str(p),"valid":bool(p and p.is_relative_to(pathlib.Path.cwd().resolve()))}))')
                origin = execute([str(python), '-c', origin_code], logs / (state + '-module-origin.log'), cwd=repo, env=env)
                require_ok(origin, 'Module source resolution failed')
                report = json.loads((logs / (state + '-module-origin.log')).read_text().strip().splitlines()[-1])
                record[state + '_module_source'] = report
                if not report['valid']:
                    raise RuntimeError('Imported module is outside the current task source')
                truncated = plan.get('resolve_truncated_p2p', [])
                if truncated:
                    roots = sorted({node.split('[', 1)[0] for node in truncated})
                    plugins = [arg for plugin in plan['pytest_plugins'] for arg in ('-p', plugin)]
                    collected = execute([str(python), '-m', 'pytest', '--collect-only', '-q',
                                         '-p', 'no:cacheprovider', '-o', 'addopts=',
                                         *plugins, *plan['pytest_args'], *roots],
                                        logs / (state + '-source-node-collection.log'), cwd=repo, env=env)
                    record[state + '_source_node_collection'] = collected
                    require_ok(collected, 'Existing parametrized source nodes failed collection')
                    lines = (logs / (state + '-source-node-collection.log')).read_text().splitlines()
                    mapping = {}
                    for original in truncated:
                        matches = sorted({line.strip() for line in lines
                                          if line.strip().startswith(original) and line.strip().endswith(']')})
                        if len(matches) != 1:
                            raise RuntimeError('Source node prefix is not uniquely recoverable: ' + original)
                        mapping[original] = matches[0]
                    if record.get('p2p_node_mapping', mapping) != mapping:
                        raise RuntimeError('Buggy/reference source node mappings differ')
                    record['p2p_node_mapping'] = mapping
                    item['source_p2p_nodes'] = row['PASS_TO_PASS']
                    item['p2p_node_mapping'] = mapping
                    item['p2p_nodes'] = [mapping.get(node, node) for node in row['PASS_TO_PASS']]
                for group in ('f2p', 'p2p'):
                    nodes = item[group + '_nodes']
                    if not nodes:
                        record[state + '_' + group] = None
                        continue  # Never turn an absent P2P list into bare pytest.
                    plugins = [arg for plugin in plan['pytest_plugins'] for arg in ('-p', plugin)]
                    command = [str(python), '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-o', 'addopts=',
                               *plugins, *plan['pytest_args'], *nodes]
                    executed = execute(command, logs / (state + '-' + group + '.log'), cwd=repo, env=env, timeout=180)
                    record[state + '_' + group] = executed
            has_p2p = bool(item['p2p_nodes'])
            record['valid_prepost'] = (record['buggy_f2p']['returncode'] == 1
                and record['reference_f2p']['returncode'] == 0
                and (not has_p2p or record['buggy_p2p']['returncode'] == record['reference_p2p']['returncode'] == 0))
            record['p2p_prepost_pass'] = (record['buggy_p2p']['returncode'] == record['reference_p2p']['returncode'] == 0) if has_p2p else None
        except Exception as error:
            record['execution_error'] = str(error)
            save(logs / 'exception.txt', traceback.format_exc())
        finally:
            shutil.rmtree(run, ignore_errors=True)
        record['finished'] = time.time()
        records.append(record)
        save(out / 'prevalidation-results.json', records)
        print('CPU_RESULT ' + json.dumps(record), flush=True)
    save(prepared / 'host-criteria.json', {'tasks': criteria})
    # Host sources and reference patches are excluded from the GPU material.
    # Invalid cases remain present with explicit CPU results; this is not an
    # automatic final confirmation selection or a silent replacement policy.
    save(prepared / 'cpu-prevalidation-results.json', records)
    save(prepared / 'source-provenance.json', {'freeze': FREEZE, 'teacher_messages_acquired': False,
                                             'reference_patches_packaged': False})
    final_archive = out / 'prepared-confirmation-material.bin'
    with tarfile.open(final_archive, 'w:gz') as archive:
        for path in sorted(prepared.iterdir()):
            archive.add(path, arcname=path.name)
    valid = [r['iid'] for r in records if r['valid_prepost']]
    summary = {'planned_tasks': len(FREEZE['task_ids']), 'executed_tasks': len(records),
               'not_started_task_ids': [iid for iid in FREEZE['task_ids'] if iid not in {r['iid'] for r in records}],
               'valid_prepost': len(valid), 'valid_task_ids': valid,
               'p2p_covered_tasks': [r['iid'] for r in records if r['p2p_nodes']],
               'f2p_only_tasks': [r['iid'] for r in records if not r['p2p_nodes']],
               'execution_errors': sum('execution_error' in r for r in records), 'model_calls': 0,
               'material_archive_name': final_archive.name, 'material_bytes': final_archive.stat().st_size,
               'material_sha256': digest(final_archive), 'input_source_sha256': FREEZE['archive_sha256'],
               'scope': 'Actual CPU observations; root must freeze final selection before any model results.'}
    if FREEZE.get('repair_plan_sha256'):
        summary['repair_plan_sha256'] = FREEZE['repair_plan_sha256']
        summary['prior_valid_tasks_not_rerun'] = FREEZE['prior_valid_tasks_not_rerun']
        summary['scope'] = 'Bounded CPU repair rerun only; prior valid tasks were not re-executed. Combine evidence explicitly, never count them as new runs.'
    save(out / 'summary.json', summary)
    print('FINAL ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
