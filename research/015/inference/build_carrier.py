"""Freeze the actual P015 base-policy repair and train-sampling experiment."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461'

BOOTSTRAP = r'''
from pathlib import Path
import importlib
import hashlib
import json
import os
import subprocess
import sys
import tarfile

os.environ.update({'LITELLM_LOCAL_MODEL_COST_MAP':'True', 'TRANSFORMERS_NO_TF':'1',
                   'VLLM_WORKER_MULTIPROC_METHOD':'spawn', 'VLLM_NO_USAGE_STATS':'1',
                   'VLLM_ENGINE_READY_TIMEOUT_S':'1200', 'OTEL_SDK_DISABLED':'true',
                   'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS':'1',
                   'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
inputs = Path('/kaggle/input')
markers = list(inputs.rglob('swegemma-0.2.11-py3-none-any.whl'))
if len(markers) != 1:
    raise RuntimeError('Official wheelhouse mount missing/ambiguous: '+str(markers))
wheelhouse = markers[0].parent
staging = Path('/tmp/p015-official-wheels')
staging.mkdir(exist_ok=True)
for path in wheelhouse.glob('*.whl'):
    if 'cutlass' in path.name.lower():
        continue
    name = path.name.replace('cu128', '+cu128') if 'cu128' in path.name and '+' not in path.name else path.name
    dest = staging/name
    if not dest.exists():
        dest.symlink_to(path)
print('PHASE install_official_runtime', flush=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','--force-reinstall',*[str(p) for p in sorted(staging.glob('*.whl'))]],check=True)
archives = list(inputs.rglob(SETTINGS['material_archive_name']))
if len(archives) != 1:
    raise RuntimeError('CPU-prepared input archive missing/ambiguous: '+str(archives))
if (archives[0].stat().st_size != SETTINGS['material_bytes']
        or hashlib.sha256(archives[0].read_bytes()).hexdigest() != SETTINGS['material_sha256']):
    raise RuntimeError('Prepared material differs from CPU-validated frozen bytes')
MATERIAL_ROOT = '/tmp/p015-agent-material'
material = Path(MATERIAL_ROOT)
material.mkdir(exist_ok=False)
with tarfile.open(archives[0], mode='r:*') as archive:
    archive.extractall(material, filter='data')
importlib.invalidate_caches()
print('PHASE inputs_ready', flush=True)
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True,
                        help='Frozen task_ids selected from actual CPU-valid cases')
    parser.add_argument('--material-kernel', default='zhuating/p015-natural-pr-panel-cpu')
    parser.add_argument('--capacity-kernel', default='zhuating/p015-fsdp-capacity')
    parser.add_argument('--output-dir', type=Path, default=HERE/'kernel')
    parser.add_argument('--manifest-path', type=Path, default=HERE/'carrier-manifest.json')
    parser.add_argument('--kernel-id', default='zhuating/p015-base-and-train-sampling')
    parser.add_argument('--title', default='P015 Base and Train Sampling')
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    task_ids = selection['task_ids']
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise RuntimeError('Task selection is empty or duplicated')
    repeats = selection.get('repeats', 1)
    if type(repeats) is not int or repeats not in (1, 2):
        raise RuntimeError('repeats must be 1 or 2')
    identity_fields = ('material_sha256', 'material_bytes')
    has_identity = all(key in selection for key in identity_fields)
    if any(key in selection for key in identity_fields) and not has_identity:
        raise RuntimeError('Freeze archive SHA256 and size together')
    archive_name = selection.get('material_archive_name', 'prepared-panel-material.tar.gz')
    if archive_name != 'prepared-panel-material.tar.gz' and not has_identity:
        raise RuntimeError('Custom archive name requires its frozen SHA256 and size')
    material = {
        'material_archive_name': archive_name,
        'material_sha256': selection['material_sha256'] if has_identity else 'eeebb65c1a03d71d42ed29d6d081b27af59a78a9c1c32ab7593d69b31f6ff8cc',
        'material_bytes': selection['material_bytes'] if has_identity else 81303711}
    if (not isinstance(material['material_archive_name'], str)
            or not re.fullmatch(r'[A-Za-z0-9_.-]+', material['material_archive_name'])
            or not isinstance(material['material_sha256'], str)
            or not re.fullmatch(r'[0-9a-f]{64}', material['material_sha256'])
            or type(material['material_bytes']) is not int or material['material_bytes'] <= 0):
        raise RuntimeError('Invalid frozen material identity')
    study = selection.get('study', '31b-base-and-train-sampling-v1')
    purpose = selection.get('purpose', 'new dev base repairs plus separate train-only rollouts; no LoRA gain claim')
    if not isinstance(study, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', study) or not isinstance(purpose, str) or not purpose:
        raise RuntimeError('Invalid study or purpose')
    base = HERE / 'base-agent'
    bundle = {'base': {str(p.relative_to(base)): p.read_text()
                       for p in sorted(base.rglob('*')) if p.is_file()}}
    protocol = json.loads((ROOT/'research/015/protocol.json').read_text())
    actual = {name: hashlib.sha256(content.encode()).hexdigest()
              for name, content in bundle['base'].items()}
    if actual != protocol['baseline_agent_members']:
        raise RuntimeError('Base Agent differs from the official 0.15 frozen source')
    capacity_manifest = ROOT/'research/015/capacity/output-fsdp-metadata/p015-fsdp/run/training_manifest.json'
    capacity = json.loads(capacity_manifest.read_text())
    if (capacity['status'] != 'training_completed' or capacity['mode'] != 'fsdp_qlora'
            or capacity['adapter']['rank'] != 8 or len(capacity['adapter']['tensors']) != 600):
        raise RuntimeError('Expected the completed rank8 capacity adapter manifest')
    settings = {'study': study, 'repeats': repeats, **material,
                'model_alias': protocol['base_model'],
                'model_path_pattern': 'gemma-4-31b-it-qat-w4a16-ct/2',
                'task_ids': task_ids, 'task_count': len(task_ids),
                'budgets': protocol['budgets'], 'concurrency': 4,
                'wall_seconds': 3600, 'attempt_seconds': 720,
                'code_commit': subprocess.check_output(
                    ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'quota_estimate_upper_hours': 4200*2/3600,
                'purpose': purpose,
                'selection_sha256': digest(args.selection),
                'capacity_adapter_diagnostic': {
                    'source_kernel': args.capacity_kernel, 'name': 'p015_capacity_diag',
                    'training_manifest_sha256': digest(capacity_manifest),
                    'training_signature': capacity['signature'],
                    'files': capacity['adapter']['files'],
                    'purpose': 'one hello request only; excluded from repair statistics; not task-SFT evaluation'}}
    worker = (HERE/'attempt_worker.py').read_text()
    coordinator = (HERE/'coordinator.py').read_text()
    source = ('SETTINGS = '+repr(settings)+'\n'+BOOTSTRAP+'\nBUNDLE = '+repr(bundle)
              +'\nWORKER_SOURCE = '+repr(worker)+'\n'+coordinator)
    ast.parse(source)
    kernel = args.output_dir; kernel.mkdir(parents=True, exist_ok=True)
    (kernel/'run.py').write_text(source)
    metadata = {'id':args.kernel_id,
                'title':args.title, 'code_file':'run.py',
                'language':'python', 'kernel_type':'script', 'is_private':True,
                'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
                'dataset_sources':['metric/gemma-4-developer-agent-wheelhouse'],
                'competition_sources':['gemma-4-developer-agent'],
                'kernel_sources':[args.material_kernel, args.capacity_kernel],
                'model_sources':[protocol['source_model_version']],
                'docker_image':IMAGE,'docker_image_pinning_type':'original'}
    (kernel/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    manifest = {'settings':settings,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                'source_bytes':len(source.encode()),'worker_sha256':hashlib.sha256(worker.encode()).hexdigest(),
                'coordinator_sha256':hashlib.sha256(coordinator.encode()).hexdigest(),
                'platform_seconds':4200,'material_kernel':args.material_kernel,
                'agent_members':actual}
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
