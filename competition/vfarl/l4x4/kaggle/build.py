"""Package a frozen S600 train-only task inventory and current VFARL source."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_id(value):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', value)


def build(out, selection_path, *, dataset_id='qingyu1232/vfarl-l4x4-train100',
          dataset_title='VFARL L4x4 Train100'):
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', dataset_id):
        raise ValueError('Dataset identity must be an explicit owner/dataset')
    selection_path = selection_path.resolve()
    selection = json.loads(selection_path.read_text())
    tasks = selection['tasks']
    if not tasks or len({task['instance_id'] for task in tasks}) != len(tasks):
        raise ValueError('Expected a nonempty frozen inventory with unique task identities')
    profiles = json.loads((HERE / 'environment-profiles.json').read_text())['profiles']
    by_task = {task: (i, profile) for i, profile in enumerate(profiles) for task in profile['tasks']}
    material = out / 'material'
    assets = material / 'assets'
    assets.mkdir(parents=True, exist_ok=False)
    rows = []
    for task in tasks:
        instance = task['instance_id']
        if task.get('physical_split') != 'train':
            raise ValueError('Only the declared S600 train inventory may be packaged: ' + instance)
        canonical = ROOT / task['actual_replay_result']
        if sha(canonical) != task['actual_replay_result_sha256']:
            raise ValueError('Changed canonical result: ' + instance)
        report = json.loads(canonical.read_text())
        parent = (ROOT / report['reports']['repaired']['path']).parent
        dest = assets / safe_id(instance)
        dest.mkdir()
        adapted = ROOT / task['task_input']
        if sha(adapted) != task['task_input_sha256']:
            raise ValueError('Changed adapted task: ' + instance)
        shutil.copy2(adapted, dest / 'task.json')
        for name in ('source-snapshot.tgz', 'submitted.patch', 'run-summary.json'):
            shutil.copy2(parent / name, dest / name)
        if sha(dest / 'source-snapshot.tgz') != report['snapshot_sha256']:
            raise ValueError('Source snapshot identity changed: ' + instance)
        declared_profile = task.get('linux_environment_profile_id')
        if declared_profile is None:
            index, profile = by_task[instance]
        else:
            match_profile = re.fullmatch(r'profile-([0-9]{2})', declared_profile)
            if not match_profile or int(match_profile.group(1)) >= len(profiles):
                raise ValueError('Unknown frozen Linux environment profile: ' + str(declared_profile))
            index = int(match_profile.group(1))
            profile = profiles[index]
            if (report.get('task_site_packages_profile') or 'stock') != profile['source_profile']:
                raise ValueError('Selected Linux profile differs from task source environment: ' + instance)
        historical = json.loads((parent / 'run-summary.json').read_text())
        match = re.search(r'/python(3\.\d+)/', profile['source_profile'])
        version = match.group(1) if match else '3.12'
        env_root = '/kaggle/working/vfarl-cpu-stage/task-envs/profile-%02d' % index
        runtime = {
            'python': env_root + '/bin/python',
            'site_packages': env_root + '/lib/python' + version + '/site-packages',
            'source_layout': report.get('sandbox_source_layout_path') or '.',
            'current_packages': profile.get('packages') or {},
            'current_packages_are_source_expectations': True,
            'environment_profile_id': 'profile-%02d' % index,
            'source_python_profile': report.get('task_python_profile'),
            'source_site_packages_profile': profile['source_profile'],
            'source_preload_module': report.get('sandbox_source_preload_module'),
            'source_import_alias_root': report.get('sandbox_source_import_alias_root'),
            'source_dependency_bindings': report.get('task_dependency_bindings'),
            'linux_prevalidation_required': True,
            'site_packages_mode': historical.get('task_site_packages_mode') or 'pth',
            'source_console_scripts': ({'glom': 'glom.cli:console_main'} if instance.startswith('mahmoud__glom.')
                                       else {'dotenv': 'dotenv.__main__:cli'} if instance.startswith('theskumar__python-dotenv.')
                                       else {'dvc': 'dvc.cli:main'} if instance.startswith('iterative__dvc.')
                                       else {}),
        }
        if instance.startswith('Instagram__MonkeyType.'):
            runtime['site_packages_mode'] = 'copy'
        (dest / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
        rows.append({'instance_id': instance, 'asset_directory': safe_id(instance),
                     'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                               for p in sorted(dest.iterdir())}})
    shutil.copy2(selection_path, material / 'selection.json')
    question_path = selection_path.parent / selection.get('questions', {}).get('path', 'questions.jsonl')
    if selection.get('questions', {}).get('sha256') and sha(question_path) != selection['questions']['sha256']:
        raise ValueError('Selected question file content changed')
    shutil.copy2(question_path, material / 'questions.jsonl')
    substitutions = selection_path.parent / 'substitutions.json'
    if substitutions.is_file():
        shutil.copy2(substitutions, material / 'substitutions.json')
    source = material / 'src'
    files = list((ROOT / 'training').glob('*.py')) + list((ROOT / 'training/encoding').glob('*.py'))
    files += list((ROOT / 'competition/vfarl/l4x4').glob('*.py'))
    files += [ROOT / 'competition/vfarl/replay/run_official.py',
              ROOT / 'competition/vfarl/six-tools.json',
              ROOT / 'training/configs/vfarl_l4x4.yaml',
              ROOT / 'training/configs/harness_tools.json']
    for path in files:
        dest = source / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    manifest = {'selection_sha256': sha(selection_path), 'tasks': rows,
                'source_files': {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()},
                'purpose': 'Judge-side task assets, not policy inputs; source licenses retained inside snapshots.'}
    (material / 'material-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    dataset = out / 'dataset'
    dataset.mkdir()
    archive = dataset / 'vfarl-train100.bin'
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(material, arcname='vfarl-input')
    metadata = {'id': dataset_id, 'title': dataset_title,
                'licenses': [{'name': 'other'}],
                'description': 'Private frozen S600 train-only tasks, original source snapshots and judged demonstration patches for Linux prevalidation. No heldout data. Source licenses remain in original snapshots. Reference patches stay outside agent workspaces.'}
    (dataset / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    result = {'tasks': len(rows), 'dataset_id': dataset_id,
              'archive': str(archive), 'bytes': archive.stat().st_size,
              'sha256': sha(archive), 'selection_sha256': sha(selection_path),
              'source_files': manifest['source_files']}
    (out / 'material-build.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--selection-path', type=Path, default=ROOT / 'competition/vfarl/l4x4/selection.json')
    parser.add_argument('--dataset-id', default='qingyu1232/vfarl-l4x4-train100')
    parser.add_argument('--dataset-title', default='VFARL L4x4 Train100')
    args = parser.parse_args()
    build(args.out.resolve(), args.selection_path, dataset_id=args.dataset_id,
          dataset_title=args.dataset_title)
