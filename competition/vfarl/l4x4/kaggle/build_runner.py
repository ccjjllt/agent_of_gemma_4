"""Freeze a Kaggle carrier; no submission or GPU allocation occurs here."""
from pathlib import Path
import argparse
import ast
import base64
import hashlib
import json
import math
import re
import zlib

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--material-build', type=Path, required=True)
parser.add_argument('--mode', choices=('capabilities', 'jail-probe', 'prevalidate', 'train'), required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--runtime-kernel', help='Explicit completed owner/kernel/version to mount.')
parser.add_argument('--runtime-sha256', help='Expected archive SHA256, mandatory for a GPU carrier.')
parser.add_argument('--context', type=int, choices=(4096, 8192, 16384), default=4096)
parser.add_argument('--training-config', type=Path,
                    default=ROOT / 'training/configs/vfarl_l4x4.yaml',
                    help='Explicit model/RL configuration to freeze; runtime context follows --context.')
parser.add_argument('--only-task', action='append', default=[])
parser.add_argument('--kernel-id')
parser.add_argument('--kernel-owner', default='qingyu1232',
                    help='Owner of the new private carrier and default train100 dataset.')
parser.add_argument('--dataset-source',
                    help='Explicit owner/dataset/version of the private train100 material; overrides owner/version defaults.')
parser.add_argument('--runtime-dataset',
                    help='Explicit owner/dataset/version containing vfarl-cpu-stage.bin, instead of a runtime kernel.')
parser.add_argument('--nf4-kernel', default='ccjjllt/p005-cpu-nf4-cache',
                    help='Completed original NF4 producer; ignored if --nf4-dataset is supplied.')
parser.add_argument('--nf4-dataset',
                    help='Explicit owner/dataset/version carrying the unchanged, producer-hashed NF4 cache.')
parser.add_argument('--extra-dataset-source', action='append', default=[],
                    help='Additional explicitly versioned private input, e.g. exported adapters or runtime support.')
parser.add_argument('--collection-kernel',
                    help='Completed owner/kernel/version containing an untrained collection-only focal batch; reuse it without collecting again.')
parser.add_argument('--collection-carrier-sha256',
                    help='SHA256 of the completed collection root vfarl-train/carrier.json.')
parser.add_argument('--collection-focal-sha256',
                    help='SHA256 of its root vfarl-train/focal.jsonl; never a worker-local file.')
parser.add_argument('--focal-data', type=Path,
                    help='Local exact-token focal data recovered from a cancelled private collection; freeze it without resampling.')
parser.add_argument('--focal-provenance', type=Path,
                    help='Recovery record binding focal_sha256 to the original source files and collection version.')
parser.add_argument('--title')
parser.add_argument('--dataset-version', type=int, default=1)
parser.add_argument('--export-runtime', action='store_true', help='Publish repaired runtime even for a CPU task subset.')
parser.add_argument('--collect-only', action='store_true', help='Collect requested real rollouts/branches without optimizer or post-update runs.')
parser.add_argument('--update-only', action='store_true',
                    help='End a short real research run after the optimizer attempt; independent policy rollouts are deferred.')
parser.add_argument('--focal-states', type=int, default=5, help='Maximum focal states across both collectors; zero means rollout coverage only.')
parser.add_argument('--policy-backend', choices=('native', 'unsloth'), default='native',
                    help='Actor execution backend, selected independently of the learner.')
parser.add_argument('--learner-backend', choices=('native', 'unsloth'), default='native',
                    help='Native four-rank FSDP or experimental Unsloth two-GPU model-parallel PPO learner.')
parser.add_argument('--resident-rl', action='store_true',
                    help='Keep two Unsloth replicas resident; collect small batches, update, synchronize, then continue.')
parser.add_argument('--resident-batch-tasks', type=int, default=4)
parser.add_argument('--resident-mixed-groups', type=int, default=2)
parser.add_argument('--resident-max-states', type=int, default=4)
parser.add_argument('--resident-seconds', type=float, default=0,
                    help='Wall-time budget for resident new claims; complete in-flight work and its last update.')
parser.add_argument('--resident-thinking-budget', type=int, default=256)
parser.add_argument('--resident-turn-tokens', type=int, default=768)
parser.add_argument('--resident-episode-tokens', type=int, default=8192)
parser.add_argument('--initial-adapter-kernel', help='Completed owner/kernel/version containing the initial plain LoRA adapter.')
parser.add_argument('--initial-adapter-sha256', help='SHA256 of that adapter_model.safetensors.')
parser.add_argument('--initial-adapter-config-sha256', help='SHA256 of its adapter_config.json.')
parser.add_argument('--dynamic-queue', action='store_true',
                    help='Two double-GPU actors claim whole tasks from a shared frozen-order queue.')
parser.add_argument('--stop-after-mixed-groups', type=int, default=0,
                    help='Dynamic queue: stop new task/state claims after N valid mixed groups; keep in-flight groups complete.')
parser.add_argument('--stop-after-tasks', type=int, default=0,
                    help='Dynamic queue: cap total globally claimed tasks; zero uses the complete declared task list.')
parser.add_argument('--stop-after-seconds', type=float, default=0,
                    help='Stop new dynamic task/state claims after this collection time, retaining in-flight results.')
parser.add_argument('--calibration-data', type=Path,
                    help='Manifest of real, source-bound actions for an explicit zero-update actor/FSDP probability study.')
parser.add_argument('--calibration-action-file', type=Path, action='append', default=[],
                    help='Real actions.jsonl referenced by the calibration manifest; preserved verbatim in the source bundle.')
parser.add_argument('--calibration-only', action='store_true',
                    help='Run actor then FSDP probability calibration only, without task collection or any optimizer.')
parser.add_argument('--prevalidation-report', type=Path, action='append', default=[],
                    help='Actual local CPU report to freeze; later reports supersede earlier task records.')
parser.add_argument('--jail-source', type=Path, help='Explicit CPU candidate or already CPU-accepted historical GPU jail source to freeze.')
args = parser.parse_args()
if args.resident_rl:
    if (args.mode != 'train' or args.policy_backend != 'unsloth' or args.learner_backend != 'unsloth'
            or args.collect_only or args.update_only or args.focal_data or args.collection_kernel
            or args.calibration_data or args.calibration_only):
        parser.error('Resident RL is an Unsloth actor/learner training run without an existing behavior batch')
    if (min(args.resident_batch_tasks, args.resident_mixed_groups, args.resident_max_states,
            args.resident_thinking_budget, args.resident_turn_tokens, args.resident_episode_tokens) < 1
            or not math.isfinite(args.resident_seconds) or args.resident_seconds <= 0):
        parser.error('Resident run requires positive batch/token budgets and a finite positive wall budget')
if args.initial_adapter_kernel:
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[0-9]+', args.initial_adapter_kernel):
        parser.error('Initial adapter source requires an explicit owner/kernel/version')
    if any(not re.fullmatch(r'[0-9a-f]{64}', value or '') for value in
           (args.initial_adapter_sha256, args.initial_adapter_config_sha256)):
        parser.error('Initial adapter source requires weight and config SHA256')
elif args.initial_adapter_sha256 or args.initial_adapter_config_sha256:
    parser.error('Initial adapter hashes require its completed kernel source')
if args.update_only and (args.mode != 'train' or args.collect_only or args.calibration_data or args.calibration_only):
    parser.error('--update-only is for a real training run without collection-only or calibration mode')
if args.focal_data or args.focal_provenance:
    if not (args.focal_data and args.focal_provenance):
        parser.error('Recovered focal data and provenance must be supplied together')
    if args.mode != 'train' or args.collect_only or args.collection_kernel or args.calibration_data or args.calibration_only:
        parser.error('Recovered focal data is for a direct update, without new collection or calibration')
if args.mode == 'train' and (not (args.runtime_kernel or args.runtime_dataset) or not args.runtime_sha256):
    parser.error('GPU carrier requires a completed, versioned runtime input and archive SHA256')
if args.runtime_kernel and args.runtime_dataset:
    parser.error('Specify one completed runtime kernel or runtime dataset, not both')
if args.runtime_kernel and len(args.runtime_kernel.split('/')) != 3:
    parser.error('--runtime-kernel must include an explicit completed version')
if args.collection_kernel:
    if not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[0-9]+', args.collection_kernel):
        parser.error('--collection-kernel must name an explicit completed owner/kernel/version')
    if args.mode != 'train' or args.collect_only or args.calibration_data or args.calibration_only:
        parser.error('Completed collection reuse is a training-only continuation, without new collection or calibration')
    if any(not re.fullmatch(r'[0-9a-f]{64}', value or '') for value in
           (args.collection_carrier_sha256, args.collection_focal_sha256)):
        parser.error('Collection reuse requires the reviewed completed carrier and focal SHA256 values')
elif args.collection_carrier_sha256 or args.collection_focal_sha256:
    parser.error('Collection output hashes require --collection-kernel')
if args.focal_states < 0:
    parser.error('--focal-states must be nonnegative')
if not re.fullmatch(r'[A-Za-z0-9_-]+', args.kernel_owner):
    parser.error('Invalid --kernel-owner')
if args.kernel_id and not args.kernel_id.startswith(args.kernel_owner+'/'):
    parser.error('--kernel-id owner must match --kernel-owner; private input defaults follow that owner')
for dataset in [args.dataset_source, args.runtime_dataset, args.nf4_dataset, *args.extra_dataset_source]:
    if dataset and not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[0-9]+', dataset):
        parser.error('Dataset inputs must use an explicit owner/dataset/version')
if (min(args.stop_after_mixed_groups, args.stop_after_tasks, args.stop_after_seconds) < 0
        or not math.isfinite(args.stop_after_seconds)):
    parser.error('Dynamic stopping budgets must be nonnegative')
if (args.stop_after_mixed_groups or args.stop_after_tasks or args.stop_after_seconds) and not args.dynamic_queue:
    parser.error('Dynamic stopping rules require --dynamic-queue')
if args.dynamic_queue and (args.mode != 'train' or args.calibration_only):
    parser.error('--dynamic-queue is only for actual training collection')
if args.policy_backend != 'native' and (args.mode != 'train' or args.calibration_data):
    parser.error('An alternate actor backend is for an explicit collection arm, not the old native calibration path')
if args.learner_backend == 'unsloth' and (args.mode != 'train' or args.policy_backend != 'unsloth'
                                         or args.calibration_data or args.calibration_only):
    parser.error('The Unsloth learner is an explicit Unsloth-actor training arm, without native FSDP calibration')
if args.calibration_only and (args.mode != 'train' or not args.calibration_data):
    parser.error('--calibration-only requires --mode train and --calibration-data')
if args.calibration_data and args.mode != 'train':
    parser.error('Probability calibration is restricted to the explicit GPU experiment mode')
if args.calibration_action_file and not args.calibration_data:
    parser.error('--calibration-action-file requires --calibration-data')
training_config = args.training_config.resolve(strict=True)
if not training_config.is_relative_to(ROOT) or not training_config.is_file():
    parser.error('--training-config must be an existing file inside this research repository')
training_config_path = str(training_config.relative_to(ROOT))
jail_override = None
if args.jail_source:
    jail_hash = hashlib.sha256(args.jail_source.read_bytes()).hexdigest()
    accepted_gpu_jails = {
        'd1cacb126907061481d8f2275e746cd08a5fb22944eafff5c641b88dc5af072a':
            'Actual CPU tail10/1 and GPU two-card rollout environment',
        '71336c47450c4c8e1baf9359aca3a3108641c09c1dc2e221068e9446d88d98e9':
            'Actual DVC CPU/3 complete baseline 0 / reviewed repair 1',
    }
    if args.mode == 'train' and jail_hash not in accepted_gpu_jails:
        parser.error('GPU jail override must be an already CPU-accepted frozen source; unvalidated candidates are CPU-only')
    jail_override = {'path': str(args.jail_source), 'sha256': jail_hash,
                     'accepted_evidence': accepted_gpu_jails.get(jail_hash, 'CPU candidate: acceptance not yet claimed')}
material = json.loads(args.material_build.read_text())
material_archive_name = Path(material['archive']).name
paths = list((ROOT / 'training').glob('*.py')) + list((ROOT / 'training/encoding').glob('*.py'))
paths += list((ROOT / 'competition/vfarl/l4x4').glob('*.py'))
paths += [ROOT / 'competition/vfarl/replay/run_official.py', ROOT / 'competition/vfarl/six-tools.json',
          ROOT / 'training/configs/vfarl_l4x4.yaml', ROOT / 'training/configs/harness_tools.json',
          ROOT / 'competition/vfarl/l4x4/kaggle/environment-profiles.json']
stock = ROOT / 'competition/vfarl/l4x4/kaggle/stock-environment-pins.json'
if stock.is_file():
    paths.append(stock)
paths.append(training_config)
calibration_path = None
if args.calibration_data:
    for path in [args.calibration_data, *args.calibration_action_file]:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(ROOT) or not resolved.is_file():
            parser.error('Calibration source must be an existing file inside this research repository')
        paths.append(resolved)
    calibration_path = str(args.calibration_data.resolve().relative_to(ROOT))
bundle = {str(p.relative_to(ROOT)): p.read_text() for p in paths}
focal_data_path, focal_provenance_path = None, None
if args.focal_data:
    for path in (args.focal_data, args.focal_provenance):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(ROOT) or not resolved.is_file():
            parser.error('Recovered inputs must be existing files inside this research repository')
        bundle[str(resolved.relative_to(ROOT))] = resolved.read_text()
    focal_data_path = str(args.focal_data.resolve().relative_to(ROOT))
    focal_provenance_path = str(args.focal_provenance.resolve().relative_to(ROOT))
    recovery = json.loads(args.focal_provenance.read_text())
    if recovery.get('focal_sha256') != hashlib.sha256(args.focal_data.read_bytes()).hexdigest():
        parser.error('Recovered focal SHA differs from its provenance record')
if args.jail_source:
    bundle['competition/vfarl/l4x4/jail.py'] = args.jail_source.read_text()
prevalidation_paths = []
for index, path in enumerate(args.prevalidation_report):
    name = f'competition/vfarl/l4x4/kaggle/frozen-prevalidation-{index}.jsonl'
    bundle[name] = path.read_text()
    prevalidation_paths.append(name)
packed = base64.b64encode(zlib.compress(json.dumps(bundle).encode())).decode()
source = f'MODE = {args.mode!r}\nMATERIAL_SHA256 = {material["sha256"]!r}\nMATERIAL_ARCHIVE_NAME = {material_archive_name!r}\nRUNTIME_SHA256 = {args.runtime_sha256!r}\nCONTEXT_LENGTH = {args.context!r}\nTRAINING_CONFIG = {training_config_path!r}\nONLY_TASKS = {args.only_task!r}\nEXPORT_RUNTIME = {args.export_runtime!r}\nCOLLECT_ONLY = {args.collect_only!r}\nFOCAL_STATES = {args.focal_states!r}\nPOLICY_BACKEND = {args.policy_backend!r}\nLEARNER_BACKEND = {args.learner_backend!r}\nDYNAMIC_QUEUE = {args.dynamic_queue!r}\nSTOP_AFTER_MIXED_GROUPS = {args.stop_after_mixed_groups!r}\nSTOP_AFTER_TASKS = {args.stop_after_tasks!r}\nCALIBRATION_DATA = {calibration_path!r}\nCALIBRATION_ONLY = {args.calibration_only!r}\nPREVALIDATION_REPORTS = {prevalidation_paths!r}\nSOURCE_BUNDLE = {packed!r}\n'
source += (HERE / 'runner.py').read_text()
resident_settings = dict(batch_task_limit=args.resident_batch_tasks,
    mixed_groups=args.resident_mixed_groups, maximum_states=args.resident_max_states,
    stop_after_seconds=args.resident_seconds, thinking_budget=args.resident_thinking_budget,
    turn_tokens=args.resident_turn_tokens, episode_tokens=args.resident_episode_tokens)
source = (f'RESIDENT_RL = {args.resident_rl!r}\n'
          f'RESIDENT_SETTINGS = {resident_settings!r}\n'
          f'INITIAL_ADAPTER_KERNEL = {args.initial_adapter_kernel!r}\n'
          f'INITIAL_ADAPTER_SHA256 = {args.initial_adapter_sha256!r}\n'
          f'INITIAL_ADAPTER_CONFIG_SHA256 = {args.initial_adapter_config_sha256!r}\n'
          f'FOCAL_DATA = {focal_data_path!r}\n'
          f'FOCAL_PROVENANCE = {focal_provenance_path!r}\n'
          f'UPDATE_ONLY = {args.update_only!r}\n'
          f'STOP_AFTER_SECONDS = {args.stop_after_seconds!r}\n'
          f'COLLECTION_KERNEL = {args.collection_kernel!r}\n'
          f'COLLECTION_CARRIER_SHA256 = {args.collection_carrier_sha256!r}\n'
          f'COLLECTION_FOCAL_SHA256 = {args.collection_focal_sha256!r}\n') + source
if args.mode == 'capabilities':
    # This CPU preparation step uploads only generic stdlib capability reads.
    # It contains no repository/training source, task material or data manifest.
    generic = (HERE / 'runner.py').read_text()
    function = next(node for node in ast.parse(generic).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'record_capabilities')
    source = ('from pathlib import Path\nimport os, sys, json, subprocess, ctypes, ctypes.util\n'
              + ast.get_source_segment(generic, function)
              + '\nout=Path("/kaggle/working/capabilities"); out.mkdir(); record_capabilities(out)\n')
elif args.mode == 'jail-probe':
    jail = bundle['competition/vfarl/l4x4/jail.py']
    source = '''from pathlib import Path
import importlib.util,json,subprocess,sys,types
from types import SimpleNamespace
root=Path('/tmp/vfarl-jail-probe-source'); root.mkdir()
module=root/'jail.py'
module.write_text(JAIL_SOURCE)
# This small capability run uses a stdlib subprocess adapter with real pipes.
# It validates production jail execution, not the full official harness integration.
def execute_subprocess_command(**arguments):
    arguments['args']=arguments.pop('command')
    result=subprocess.run(**arguments,text=True,capture_output=True)
    return SimpleNamespace(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
package=types.ModuleType('adk_eval_core'); sandbox=types.ModuleType('adk_eval_core.sandbox')
sandbox.execute_subprocess_command=execute_subprocess_command
sys.modules['adk_eval_core']=package;sys.modules['adk_eval_core.sandbox']=sandbox
spec=importlib.util.spec_from_file_location('vfarl_jail',module)
jail=importlib.util.module_from_spec(spec);spec.loader.exec_module(jail)
result=jail.configure([])
result['executor_scope']='stdlib subprocess adapter; official harness integration still separate'
print(json.dumps(result),flush=True)
Path('/kaggle/working/jail-capabilities.json').write_text(json.dumps(result,indent=2)+'\\n')
'''
    source = 'JAIL_SOURCE = ' + repr(jail) + '\n' + source
args.out.mkdir(parents=True, exist_ok=False)
(args.out / 'run.py').write_text(source)
gpu = args.mode == 'train'
dataset_sources = [args.dataset_source or f'{args.kernel_owner}/vfarl-l4x4-train100/{args.dataset_version}']
dataset_sources += [source for source in [args.runtime_dataset, args.nf4_dataset, *args.extra_dataset_source] if source]
dataset_sources = list(dict.fromkeys(dataset_sources))
kernel_sources = ([args.nf4_kernel] if gpu and not args.nf4_dataset else [])
kernel_sources += ([args.runtime_kernel] if args.runtime_kernel else [] if args.runtime_dataset else
                   [f'{args.kernel_owner}/vfarl-l4x4-cpu-stage/2'])
if args.collection_kernel:
    kernel_sources.append(args.collection_kernel)
if args.initial_adapter_kernel:
    kernel_sources.append(args.initial_adapter_kernel)
kernel_sources = list(dict.fromkeys(kernel_sources))
metadata = {'id': args.kernel_owner + '/vfarl-l4x4-' + args.mode, 'title': 'VFARL L4x4 ' + args.mode.title(),
            'code_file': 'run.py', 'language': 'python', 'kernel_type': 'script',
            'is_private': True, 'enable_gpu': gpu, 'enable_tpu': False, 'enable_internet': not gpu,
            'dataset_sources': dataset_sources,
            'competition_sources': ['gemma-4-developer-agent'],
            'kernel_sources': kernel_sources,
            'model_sources': [],
            'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
            'docker_image_pinning_type': 'original'}
if args.kernel_id:
    metadata['id'] = args.kernel_id
if args.title:
    metadata['title'] = args.title
if args.mode in ('capabilities', 'jail-probe'):
    metadata.update(dataset_sources=[], kernel_sources=[], competition_sources=[], enable_internet=False)
(args.out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
manifest = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'source_files': {name: hashlib.sha256(content.encode()).hexdigest() for name,content in bundle.items()},
            'material_sha256': material['sha256'], 'mode': args.mode,
            'material_archive_name': material_archive_name,
            'runtime_kernel': args.runtime_kernel,
            'runtime_dataset': args.runtime_dataset,
            'runtime_sha256': args.runtime_sha256,
            'collection_kernel': args.collection_kernel,
            'collection_carrier_sha256': args.collection_carrier_sha256,
            'collection_focal_sha256': args.collection_focal_sha256,
            'context_length': args.context,
            'training_config': {'path': training_config_path,
                                'sha256': hashlib.sha256(training_config.read_bytes()).hexdigest()},
            'only_tasks': args.only_task,
            'dataset_version': args.dataset_version,
            'export_runtime': args.export_runtime,
            'collect_only': args.collect_only,
            'maximum_focal_states': args.focal_states,
            'policy_backend': args.policy_backend,
            'learner_backend': args.learner_backend,
            'update_only': args.update_only,
            'resident_rl': args.resident_rl,
            'resident_settings': resident_settings if args.resident_rl else None,
            'initial_adapter_kernel': args.initial_adapter_kernel,
            'initial_adapter_sha256': args.initial_adapter_sha256,
            'initial_adapter_config_sha256': args.initial_adapter_config_sha256,
            'focal_data': focal_data_path,
            'focal_provenance': focal_provenance_path,
            'dynamic_queue': args.dynamic_queue,
            'stop_after_mixed_groups': args.stop_after_mixed_groups,
            'stop_after_tasks': args.stop_after_tasks,
            'stop_after_seconds': args.stop_after_seconds,
            'kernel_owner': metadata['id'].split('/')[0],
            'dataset_sources': metadata['dataset_sources'],
            'kernel_sources': metadata['kernel_sources'],
            'calibration_only': args.calibration_only,
            'calibration_data': ({'path': calibration_path, 'sha256': hashlib.sha256(args.calibration_data.read_bytes()).hexdigest()}
                                 if args.calibration_data else None),
            'calibration_action_files': [{'path': str(path.resolve().relative_to(ROOT)),
                                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                                         for path in args.calibration_action_file],
            'collector_focal_quotas': ([args.focal_states, args.focal_states] if args.dynamic_queue else
                                      [(args.focal_states + 1) // 2, args.focal_states // 2]),
            'focal_quota_semantics': 'Shared global reservations' if args.dynamic_queue else 'Fixed per-worker quotas',
            'jail_source_override': jail_override,
            'prevalidation_reports': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                                     for path in args.prevalidation_report],
            'command': ['kaggle', 'kernels', 'push', '-p', str(args.out)] + (['--accelerator', 'NvidiaL4'] if gpu else []),
            'accelerator_note': 'NvidiaL4 is the prior measured four physical L4 competition shape; runner measures exact topology again.',
            'time_limit': 'No additional task wall-time cap; platform/account limits still apply.'}
(args.out / 'carrier-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
