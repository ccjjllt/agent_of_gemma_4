"""Freeze a Kaggle carrier; no submission or GPU allocation occurs here."""
from pathlib import Path
import argparse
import ast
import base64
import hashlib
import json
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
parser.add_argument('--title')
parser.add_argument('--dataset-version', type=int, default=1)
parser.add_argument('--export-runtime', action='store_true', help='Publish repaired runtime even for a CPU task subset.')
parser.add_argument('--collect-only', action='store_true', help='Collect requested real rollouts/branches without optimizer or post-update runs.')
parser.add_argument('--focal-states', type=int, default=5, help='Maximum focal states across both collectors; zero means rollout coverage only.')
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
if args.mode == 'train' and (not args.runtime_kernel or not args.runtime_sha256):
    parser.error('GPU carrier requires a completed, versioned runtime kernel and archive SHA256')
if args.runtime_kernel and len(args.runtime_kernel.split('/')) != 3:
    parser.error('--runtime-kernel must include an explicit completed version')
if args.focal_states < 0:
    parser.error('--focal-states must be nonnegative')
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
if args.jail_source:
    bundle['competition/vfarl/l4x4/jail.py'] = args.jail_source.read_text()
prevalidation_paths = []
for index, path in enumerate(args.prevalidation_report):
    name = f'competition/vfarl/l4x4/kaggle/frozen-prevalidation-{index}.jsonl'
    bundle[name] = path.read_text()
    prevalidation_paths.append(name)
packed = base64.b64encode(zlib.compress(json.dumps(bundle).encode())).decode()
source = f'MODE = {args.mode!r}\nMATERIAL_SHA256 = {material["sha256"]!r}\nRUNTIME_SHA256 = {args.runtime_sha256!r}\nCONTEXT_LENGTH = {args.context!r}\nTRAINING_CONFIG = {training_config_path!r}\nONLY_TASKS = {args.only_task!r}\nEXPORT_RUNTIME = {args.export_runtime!r}\nCOLLECT_ONLY = {args.collect_only!r}\nFOCAL_STATES = {args.focal_states!r}\nCALIBRATION_DATA = {calibration_path!r}\nCALIBRATION_ONLY = {args.calibration_only!r}\nPREVALIDATION_REPORTS = {prevalidation_paths!r}\nSOURCE_BUNDLE = {packed!r}\n'
source += (HERE / 'runner.py').read_text()
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
metadata = {'id': 'qingyu1232/vfarl-l4x4-' + args.mode, 'title': 'VFARL L4x4 ' + args.mode.title(),
            'code_file': 'run.py', 'language': 'python', 'kernel_type': 'script',
            'is_private': True, 'enable_gpu': gpu, 'enable_tpu': False, 'enable_internet': not gpu,
            'dataset_sources': ['qingyu1232/vfarl-l4x4-train100/' + str(args.dataset_version)],
            'competition_sources': ['gemma-4-developer-agent'],
            'kernel_sources': (['ccjjllt/p005-cpu-nf4-cache'] if gpu else []) +
                              [args.runtime_kernel or 'qingyu1232/vfarl-l4x4-cpu-stage/2'],
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
            'runtime_kernel': args.runtime_kernel,
            'runtime_sha256': args.runtime_sha256,
            'context_length': args.context,
            'training_config': {'path': training_config_path,
                                'sha256': hashlib.sha256(training_config.read_bytes()).hexdigest()},
            'only_tasks': args.only_task,
            'dataset_version': args.dataset_version,
            'export_runtime': args.export_runtime,
            'collect_only': args.collect_only,
            'maximum_focal_states': args.focal_states,
            'calibration_only': args.calibration_only,
            'calibration_data': ({'path': calibration_path, 'sha256': hashlib.sha256(args.calibration_data.read_bytes()).hexdigest()}
                                 if args.calibration_data else None),
            'calibration_action_files': [{'path': str(path.resolve().relative_to(ROOT)),
                                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                                         for path in args.calibration_action_file],
            'collector_focal_quotas': [(args.focal_states + 1) // 2, args.focal_states // 2],
            'jail_source_override': jail_override,
            'prevalidation_reports': [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                                     for path in args.prevalidation_report],
            'command': ['kaggle', 'kernels', 'push', '-p', str(args.out)] + (['--accelerator', 'NvidiaL4'] if gpu else []),
            'accelerator_note': 'NvidiaL4 is the prior measured four physical L4 competition shape; runner measures exact topology again.',
            'time_limit': 'No additional task wall-time cap; platform/account limits still apply.'}
(args.out / 'carrier-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
