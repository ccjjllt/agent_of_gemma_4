"""Freeze the acquired P017 C sources into a CPU-only preparation carrier."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent
ROOT = HERE.parents[3]
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'
MODULES = {'networkx/networkx': 'networkx', 'hgrecco/pint': 'pint',
           'omni-us/jsonargparse': 'jsonargparse', 'rdflib/rdflib': 'rdflib',
           'facelessuser/soupsieve': 'soupsieve', 'hynek/structlog': 'structlog',
           'fatal1ty/mashumaro': 'mashumaro', 'reata/sqllineage': 'sqllineage'}
PLUGINS = {'pytest-mock': 'pytest_mock', 'pytest-asyncio': 'pytest_asyncio.plugin',
           'pytest-subtests': 'pytest_subtests', 'pytest-mpl': 'pytest_mpl.plugin',
           'pytest-benchmark': 'pytest_benchmark.plugin', 'pytest-cov': 'pytest_cov.plugin'}
# Match a runnable Python 3.12 pytest/build toolchain. Library pins stay at their
# source values: incompatible historical libraries must fail visibly, not drift.
TOOLS = {'pytest': '8.3.5', 'pluggy': '1.5.0', 'iniconfig': '2.1.0',
         'packaging': '24.2', 'exceptiongroup': '1.2.2', 'tomli': '2.2.1',
         'setuptools': '75.8.0', 'wheel': '0.45.1'}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def environment_plan(row):
    module = MODULES[row['repo'].lower()]
    conda = {canonical(name): version for name, version in re.findall(
        r'^  - ([A-Za-z0-9_.-]+)=([^=\s]+)(?:=[^\s]+)?$', row['environment'], re.M)}
    pins, transformations = {}, []
    for line in row['requirements'].splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('-e git+'):
            egg = re.search(r'#egg=([A-Za-z0-9_.-]+)$', line)
            if not egg or canonical(egg[1]) != canonical(module) or row['base_commit'] not in line:
                raise ValueError('Unexpected editable source requirement: ' + row['instance_id'])
            transformations.append({'source': line, 'action': 'Use exact snapshot editable source; no VCS install'})
            continue
        pinned = re.fullmatch(r'([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)', line)
        local = re.fullmatch(r'([A-Za-z0-9_.-]+) @ file://[^\s]+', line)
        if pinned:
            name, version = canonical(pinned[1]), pinned[2]
        elif local and canonical(local[1]) in conda:
            name = canonical(local[1])
            version = conda[name]
            transformations.append({'source': line, 'replacement': name + '==' + version,
                                    'reason': 'Version recovered from this row\'s conda environment'})
        else:
            raise ValueError('Unresolved source requirement for ' + row['instance_id'] + ': ' + line)
        if name == canonical(module):
            raise ValueError('Do not install a released copy of the evaluated source module')
        pins[name] = version
    for name, version in TOOLS.items():
        if pins.get(name) != version:
            transformations.append({'package': name, 'from': pins.get(name), 'to': version,
                                    'reason': 'Explicit Python 3.12 CPU/GPU preparation toolchain; not yet validated'})
            pins[name] = version
    return {'module': module, 'requirements': [name + '==' + pins[name] for name in sorted(pins)],
            'pytest_plugins': [plugin for package, plugin in PLUGINS.items() if package in pins],
            'pytest_plugin_packages': [package for package in PLUGINS if package in pins],
            'pytest_args': ['--no-header', '-rA', '--tb=line', '--color=no', '-W', 'ignore::DeprecationWarning'],
            'install_editable': True, 'source_python': row['install_config'].get('python'),
            'target_python': '3.12', 'source_install_config': row['install_config'],
            'source_commands_executed': False, 'transformations': transformations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-id', required=True)
    parser.add_argument('--kernel-id', required=True)
    parser.add_argument('--title', default='P017 Confirmation CPU Preparation')
    parser.add_argument('--wall-seconds', type=int, default=7200)
    parser.add_argument('--repair-plan', type=Path,
                        help='Explicit bounded task/environment repair plan; keeps the original identity freeze')
    parser.add_argument('--panel-dir', type=Path, default=PANEL,
                        help='Explicit locally assembled execution inputs; default is the original panel')
    args = parser.parse_args()
    if args.wall_seconds < 900:
        raise ValueError('CPU preparation wall budget must be at least 900 seconds')
    for value in (args.dataset_id, args.kernel_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', value):
            raise ValueError('Expected explicit owner/slug')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use a new output directory; do not mutate a frozen carrier')
    panel = args.panel_dir.resolve()
    freeze = json.loads((panel / 'confirmation-identity-freeze.json').read_text())
    source = json.loads((panel / 'source-acquisition.json').read_text())
    if source['freeze_sha256'] != digest(panel / 'confirmation-identity-freeze.json'):
        raise ValueError('Source acquisition is not bound to this identity freeze')
    source_files = {item['instance_id']: item for item in source['files']}
    snapshots = {item['instance_id']: item for item in json.loads((panel / 'snapshot-acquisition.json').read_text())}
    ids = [item['instance_id'] for item in freeze['primary']]
    if set(ids) != set(source_files) or not set(ids).issubset(snapshots):
        raise ValueError('All frozen primary sources and snapshots must be available')
    repair = json.loads(args.repair_plan.read_text()) if args.repair_plan else None
    if repair:
        if repair.get('input_freeze_sha256', repair.get('original_freeze_sha256')) != source['freeze_sha256']:
            raise ValueError('Repair plan does not reference this original freeze')
        chosen = set(repair['tasks'])
        if not chosen or not chosen.issubset(ids):
            raise ValueError('Repair tasks must be frozen primary identities')
        ids = [iid for iid in ids if iid in chosen]
    plans, files = {}, {}
    for item in freeze['primary']:
        iid = item['instance_id']
        if iid not in ids:
            continue
        for relative, pin in [('host-source/' + iid + '.json', source_files[iid]),
                              ('snapshots/' + iid + '.tar.gz', snapshots[iid])]:
            path = panel / relative
            if path.stat().st_size != pin['bytes'] or digest(path) != pin['sha256']:
                raise ValueError('Acquired input differs: ' + relative)
            files[relative] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        row = json.loads((panel / 'host-source' / (iid + '.json')).read_text())
        if row['instance_id'] != iid or row['base_commit'] != item['base_commit'] or not row['FAIL_TO_PASS']:
            raise ValueError('Source identity or F2P absent: ' + iid)
        for key in ('FAIL_TO_PASS', 'PASS_TO_PASS'):
            if not isinstance(row[key], list) or not all(isinstance(n, str) and n and not n.startswith('-') for n in row[key]):
                raise ValueError('Expected explicit upstream nodes: ' + iid)
        plans[iid] = environment_plan(row)
        if repair:
            change = repair['tasks'][iid]
            if 'environment_plan' in change:
                plans[iid] = json.loads(json.dumps(change['environment_plan']))
            plan = plans[iid]
            pins = dict(req.split('==') for req in plan['requirements'])
            if plan['module'] != MODULES[row['repo'].lower()] or any(pins.get(name) != version for name, version in TOOLS.items()):
                raise ValueError('Explicit environment plan changed module or frozen Python3.12 toolchain')
            if change.get('source_import_only'):
                plan['install_editable'] = False
                plan['transformations'].append({'action': 'Import exact source via PYTHONPATH without editable packaging',
                                               'reason': change['reason']})
            existing = dict(pin.split('==') for pin in plan['requirements'])
            for name, version in change.get('build_requirements', {}).items():
                if name in existing and existing[name] != version:
                    raise ValueError('Bounded backend repair must not change existing library/tool pin: ' + name)
                existing[name] = version
            plan['requirements'] = [name + '==' + existing[name] for name in sorted(existing)]
            plan['resolve_truncated_p2p'] = change.get('resolve_truncated_p2p', [])
            if not set(plan['resolve_truncated_p2p']).issubset(row['PASS_TO_PASS']):
                raise ValueError('Unknown source P2P selector in repair plan')
            plan['repair_reason'] = change['reason']
            plan['build_backend_sources'] = change.get('build_backend_sources', [])
    for relative in ('agent-tasks.json', 'confirmation-identity-freeze.json', 'source-acquisition.json', 'snapshot-acquisition.json'):
        path = panel / relative
        files[relative] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    staging = args.output_dir / 'staging'
    staging.mkdir(parents=True)
    for relative in files:
        dest = staging / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(panel / relative, dest)
    if repair:
        selected_tasks = [x for x in json.loads((panel / 'agent-tasks.json').read_text()) if x['instance_id'] in ids]
        save(staging / 'agent-tasks.json', selected_tasks)
        save(staging / 'repair-plan.json', repair)
        for name in ('agent-tasks.json', 'repair-plan.json'):
            files[name] = {'bytes': (staging / name).stat().st_size, 'sha256': digest(staging / name)}
        cache = repair['prior_material']
        archive = ROOT / cache['path']
        if archive.stat().st_size != cache['bytes'] or digest(archive) != cache['sha256']:
            raise ValueError('Prior CPU wheel cache archive differs')
        cache_dir = staging / 'cached-wheels'
        cache_dir.mkdir()
        with tarfile.open(archive, 'r:*') as old:
            for member in old.getmembers():
                if member.isfile() and member.name.startswith('environments/shared-wheels/') and member.name.endswith('.whl'):
                    target = cache_dir / Path(member.name).name
                    with old.extractfile(member) as stream, target.open('wb') as out:
                        shutil.copyfileobj(stream, out)
        for pin in repair.get('additional_wheels', repair.get('new_backend_wheels', [])):
            wheel = ROOT / pin['path']
            if digest(wheel) != pin['sha256']:
                raise ValueError('New backend wheel differs: ' + pin['path'])
            target = cache_dir / wheel.name
            if target.exists() and digest(target) != pin['sha256']:
                raise ValueError('Cache wheel collision: ' + wheel.name)
            shutil.copyfile(wheel, target)
        available = {(canonical(w.name.split('-')[0]), w.name.split('-')[1]) for w in cache_dir.glob('*.whl')}
        for iid, plan in plans.items():
            missing = [req for req in plan['requirements'] if tuple(req.split('==')) not in available]
            if missing:
                raise ValueError('Offline repair cache incomplete for ' + iid + ': ' + str(missing))
        for wheel in cache_dir.glob('*.whl'):
            files[str(wheel.relative_to(staging))] = {'bytes': wheel.stat().st_size, 'sha256': digest(wheel)}
    save(staging / 'environment-plans.json', plans)
    files['environment-plans.json'] = {'bytes': (staging / 'environment-plans.json').stat().st_size,
                                     'sha256': digest(staging / 'environment-plans.json')}
    dataset = args.output_dir / 'dataset'
    dataset.mkdir()
    archive_path = dataset / 'p017-confirmation-source.bin'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for relative in sorted(files):
            archive.add(staging / relative, arcname=relative, recursive=False)
    pin = {'archive_name': archive_path.name, 'archive_bytes': archive_path.stat().st_size,
           'archive_sha256': digest(archive_path), 'files': files, 'task_ids': ids,
           'freeze_sha256': source['freeze_sha256'], 'source_dataset': args.dataset_id,
           'wall_seconds': args.wall_seconds, 'platform_seconds': args.wall_seconds + 300}
    if repair:
        pin['repair_plan_sha256'] = digest(args.repair_plan)
        pin['offline_wheel_cache'] = True
        pin['prior_valid_tasks_not_rerun'] = repair['prior_valid_tasks_not_rerun']
    save(args.output_dir / 'input-manifest.json', pin)
    save(dataset / 'dataset-metadata.json', {'id': args.dataset_id, 'title': args.title + ' Sources',
                                           'licenses': [{'name': 'other'}]})
    kernel = args.output_dir / 'kernel'
    kernel.mkdir()
    runner = (HERE / 'prepare.py').read_text()
    code = 'FREEZE = ' + repr(pin) + '\n' + runner
    ast.parse(code)
    (kernel / 'run.py').write_text(code)
    save(kernel / 'kernel-metadata.json', {'id': args.kernel_id, 'title': args.title,
        'code_file': 'run.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
        'enable_gpu': False, 'enable_tpu': False, 'enable_internet': not bool(repair),
        'dataset_sources': [args.dataset_id], 'competition_sources': [], 'kernel_sources': [], 'model_sources': [],
        'docker_image': IMAGE, 'docker_image_pinning_type': 'original'})
    save(args.output_dir / 'carrier-manifest.json', {'input': pin, 'runner_sha256': digest(HERE / 'prepare.py'),
        'source_sha256': digest(kernel / 'run.py'), 'model_calls': 0, 'gpu_enabled': False,
        'scope': 'CPU dependency materialization and existing source criteria; no model or teacher rollouts'})
    shutil.rmtree(staging)
    print(json.dumps({'output': str(args.output_dir), 'tasks': len(ids), 'archive_bytes': pin['archive_bytes'],
                      'archive_sha256': pin['archive_sha256'], 'gpu_enabled': False}, indent=2))


if __name__ == '__main__':
    main()
