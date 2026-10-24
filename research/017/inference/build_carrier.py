"""Build a frozen P017 Base/Step-68 carrier from real final artifacts. No launch."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
P015 = ROOT / 'research/015/inference'
FINAL_SUMMARY = ROOT / 'research/016/training/main-short-training/final-result-summary.json'
DEFAULT_SELECTION = ROOT / 'research/015/evaluation/final-selection.json'
NAME = 'p017_sft_step68'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--adapter-dataset', help='Private mounted dataset, owner/slug')
    source.add_argument('--adapter-kernel', help='Mounted output kernel, owner/slug')
    parser.add_argument('--adapter-source-version', type=int, required=True,
                        help='Actual uploaded dataset/kernel version; content SHA is checked at runtime')
    parser.add_argument('--adapter-layout', choices=('flat', 'nested'), default='nested',
                        help='PEFT files beside manifest (flat) or in its adapter/ child (nested)')
    parser.add_argument('--training-manifest', type=Path, required=True)
    parser.add_argument('--phase', choices=('D', 'C'), default='D')
    parser.add_argument('--selection', type=Path, default=DEFAULT_SELECTION)
    material_source = parser.add_mutually_exclusive_group()
    material_source.add_argument('--material-kernel', help='CPU-prepared output kernel, owner/slug')
    material_source.add_argument('--material-dataset', help='Private CPU-prepared dataset, owner/slug')
    parser.add_argument('--material-source-version', type=int,
                        help='Actual material source version, when supplied by the uploader')
    parser.add_argument('--task-start', type=int, default=0, help='Zero-based contiguous pair shard start')
    parser.add_argument('--task-count', type=int, help='Number of whole task pairs in this shard')
    parser.add_argument('--workers', type=int, choices=(1, 4),
                        help='Default D=4, C=1; serving max-num-seqs remains frozen at 4')
    parser.add_argument('--wall-seconds', type=int, default=4800,
                        help='Experiment deadline including server startup; platform allowance adds 600 seconds')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'kernel-d')
    parser.add_argument('--manifest-path', type=Path, default=HERE / 'carrier-manifest-d.json')
    parser.add_argument('--kernel-id', required=True, help='Explicit launch account and kernel slug')
    parser.add_argument('--title', default='P017 Step68 Weight AB')
    args = parser.parse_args()
    source_kind = 'dataset' if args.adapter_dataset else 'kernel'
    source_slug = args.adapter_dataset or args.adapter_kernel
    material_kind = 'dataset' if args.material_dataset else 'kernel'
    material_slug = args.material_dataset or args.material_kernel or 'zhuating/p015-natural-pr-panel-cpu'
    for value in (source_slug, material_slug, args.kernel_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', value):
            raise ValueError('Expected explicit owner/slug: ' + value)
    if args.adapter_source_version < 1 or args.wall_seconds < 1800:
        raise ValueError('Use a real source version and at least 1800 experiment seconds')
    if args.material_source_version is not None and args.material_source_version < 1:
        raise ValueError('Material source version must be positive')
    if args.task_start < 0 or args.task_count is not None and args.task_count < 1:
        raise ValueError('Task shard must be nonempty and start at a nonnegative index')
    workers = args.workers if args.workers is not None else (4 if args.phase == 'D' else 1)

    # The P016 completion summary pins identity; it is never substituted for the
    # actual trainer manifest, nor is a calibration manifest accepted here.
    final = json.loads(FINAL_SUMMARY.read_text())
    manifest_pin = final['files_read_and_hashed']['training_manifest.json']
    if (args.training_manifest.stat().st_size != manifest_pin['bytes']
            or sha(args.training_manifest) != manifest_pin['sha256']):
        raise ValueError('Supply the authentic P016 Step-68 final training_manifest.json')
    training = json.loads(args.training_manifest.read_text())
    if (training.get('status') != 'training_completed'
            or training.get('signature') != final['signature']
            or training.get('metrics', {}).get('optimizer_steps') != 68
            or training.get('config', {}).get('base_family') != 'gemma-4-31b-it-qat-q4_0-unquantized'):
        raise ValueError('Manifest does not describe the completed P016 Step-68 run')
    adapter = training['adapter']
    cfg = adapter['adapter_config']
    if (adapter['rank'] != 8 or cfg['r'] != 8 or cfg['lora_alpha'] != 16
            or cfg['peft_type'] != 'LORA' or cfg['bias'] != 'none'
            or len(adapter['tensors']) != 600 or adapter['tensor_parameters'] != 53043200):
        raise ValueError('Expected the independent rank8 text q/o/MLP adapter')
    files = adapter['files']
    if set(files) != {'adapter_config.json', 'adapter_model.safetensors'}:
        raise ValueError('Expected exactly two final PEFT files')
    for name, pin in files.items():
        if pin != final['files_read_and_hashed']['adapter/' + name]:
            raise ValueError('Final adapter identity differs: ' + name)

    selection = json.loads(args.selection.read_text())
    all_ids = selection['dev_task_ids']
    if (not isinstance(all_ids, list) or not all_ids
            or not all(isinstance(iid, str) and re.fullmatch(r'[A-Za-z0-9_.-]+', iid) for iid in all_ids)
            or len(set(all_ids)) != len(all_ids)
            or set(all_ids) & set(selection['train_task_ids'])):
        raise ValueError('Selection must contain unique independent dev task identities')
    if args.phase == 'D':
        frozen = json.loads(DEFAULT_SELECTION.read_text())
        if any(selection[key] != frozen[key] for key in
               ('dev_task_ids', 'material_sha256', 'material_bytes')):
            raise ValueError('D uses the existing frozen P015 panel; use phase C for a new panel')
    elif args.selection.resolve() == DEFAULT_SELECTION.resolve() or selection.get('model_results_seen') is not False:
        raise ValueError('C requires a new identity selection frozen before candidate results')
    if (not re.fullmatch(r'[0-9a-f]{64}', selection['material_sha256'])
            or type(selection['material_bytes']) is not int or selection['material_bytes'] <= 0):
        raise ValueError('Freeze the prepared material SHA256 and byte count')
    archive_name = selection.get('material_archive_name', 'prepared-panel-material.tar.gz')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', archive_name):
        raise ValueError('Invalid material archive name')
    end = len(all_ids) if args.task_count is None else args.task_start + args.task_count
    if args.task_start >= len(all_ids) or end > len(all_ids):
        raise ValueError('Task shard exceeds the frozen selection')
    ids = all_ids[args.task_start:end]
    protocol = json.loads((ROOT / 'research/015/protocol.json').read_text())
    base_dir = P015 / 'base-agent'
    base = {str(p.relative_to(base_dir)): p.read_text() for p in sorted(base_dir.rglob('*')) if p.is_file()}
    if {k: hashlib.sha256(v.encode()).hexdigest() for k, v in base.items()} != protocol['baseline_agent_members']:
        raise ValueError('Base files differ from the frozen runtime-v2 Agent')
    if re.search(r'^adapter\s*:', base['agent.yaml'], re.M):
        raise ValueError('Base Agent unexpectedly selects an adapter')
    model_line = 'model: ' + protocol['base_model'] + '\n'
    if base['agent.yaml'].count(model_line) != 1:
        raise ValueError('Expected exactly one base model alias in Agent YAML')
    adapted = dict(base)
    adapted['agent.yaml'] = base['agent.yaml'].replace(model_line, model_line + 'adapter: ' + NAME + '\n')
    bundle = {'base': base, 'adapter': adapted}
    settings = {
        'study': 'p017-step68-weight-ab', 'phase': args.phase,
        'task_ids': ids, 'task_count': len(ids), 'selection_task_count': len(all_ids),
        'task_start': args.task_start, 'selection_sha256': sha(args.selection),
        'model_alias': protocol['base_model'], 'model_path_pattern': 'gemma-4-31b-it-qat-w4a16-ct/2',
        'budgets': protocol['budgets'], 'thinking_budget': protocol['sampling']['thinking_budget'],
        'workers': workers, 'server_max_num_seqs': 4, 'wall_seconds': args.wall_seconds,
        'platform_seconds': args.wall_seconds + 600, 'cleanup_seconds': 180,
        'attempt_seconds': 720, 'attempt_setup_reserve_seconds': 180,
        'material_archive_name': archive_name, 'material_sha256': selection['material_sha256'],
        'material_bytes': selection['material_bytes'],
        'material_source': {'kind': material_kind, 'slug': material_slug, 'version': args.material_source_version},
        'purpose': 'contemporaneous Base/Step-68 pairs; phases and concurrency reported separately',
        'adapter': {'name': NAME, 'source_kind': source_kind, 'source_slug': source_slug,
                    'source_version': args.adapter_source_version, 'layout': args.adapter_layout,
                    'manifest_sha256': sha(args.training_manifest),
                    'signature': training['signature'], 'files': files, 'rank': 8,
                    'parameters': adapter['tensor_parameters'], 'checkpoint_step': 68},
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    }
    spec = importlib.util.spec_from_file_location('p015_base_builder', P015 / 'build_carrier.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = (P015 / 'attempt_worker.py').read_text()
    coordinator = (HERE / 'coordinator.py').read_text()
    source_text = ('SETTINGS = ' + repr(settings) + '\n' + module.BOOTSTRAP + '\nBUNDLE = ' + repr(bundle)
                   + '\nWORKER_SOURCE = ' + repr(worker) + '\n' + coordinator)
    ast.parse(source_text)
    for path in (args.output_dir, args.manifest_path):
        resolved = path.resolve()
        if resolved == P015.resolve() or P015.resolve() in resolved.parents:
            raise ValueError('Do not write into frozen P015 inference outputs')
    if args.manifest_path.exists() or args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Use new carrier output and manifest paths')
    datasets = ['metric/gemma-4-developer-agent-wheelhouse']
    kernels = []
    (datasets if material_kind == 'dataset' else kernels).append(material_slug)
    (datasets if source_kind == 'dataset' else kernels).append(source_slug)
    metadata = {'id': args.kernel_id, 'title': args.title, 'code_file': 'run.py', 'language': 'python',
                'kernel_type': 'script', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
                'enable_internet': False, 'dataset_sources': datasets, 'kernel_sources': kernels,
                'competition_sources': ['gemma-4-developer-agent'],
                'model_sources': [protocol['source_model_version']],
                'docker_image': module.IMAGE, 'docker_image_pinning_type': 'original'}
    result = {'settings': settings, 'source_sha256': hashlib.sha256(source_text.encode()).hexdigest(),
              'worker_sha256': hashlib.sha256(worker.encode()).hexdigest(),
              'coordinator_sha256': hashlib.sha256(coordinator.encode()).hexdigest(),
              'builder_sha256': sha(Path(__file__)), 'source_bytes': len(source_text.encode()),
              'platform_seconds': settings['platform_seconds'], 'planned_attempts': 2 * len(ids),
              'independent_dev_issues': len(ids), 'optional_thinking_arm': False,
              'agent_members': protocol['baseline_agent_members'],
              'only_AB_agent_difference': 'official adapter YAML field',
              'historical5of12_used_as_control': False,
              'source_version_note': 'Version records uploader provenance; exact mounted content is enforced by SHA256.'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'run.py').write_text(source_text)
    (args.output_dir / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
