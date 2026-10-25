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
parser.add_argument('--context', type=int, choices=(4096, 8192), default=4096)
parser.add_argument('--only-task', action='append', default=[])
parser.add_argument('--kernel-id')
parser.add_argument('--title')
parser.add_argument('--dataset-version', type=int, default=1)
args = parser.parse_args()
if args.mode == 'train' and (not args.runtime_kernel or not args.runtime_sha256):
    parser.error('GPU carrier requires a completed, versioned runtime kernel and archive SHA256')
if args.runtime_kernel and len(args.runtime_kernel.split('/')) != 3:
    parser.error('--runtime-kernel must include an explicit completed version')
material = json.loads(args.material_build.read_text())
paths = list((ROOT / 'training').glob('*.py')) + list((ROOT / 'training/encoding').glob('*.py'))
paths += list((ROOT / 'competition/vfarl/l4x4').glob('*.py'))
paths += [ROOT / 'competition/vfarl/replay/run_official.py', ROOT / 'competition/vfarl/six-tools.json',
          ROOT / 'training/configs/vfarl_l4x4.yaml', ROOT / 'training/configs/harness_tools.json',
          ROOT / 'competition/vfarl/l4x4/kaggle/environment-profiles.json']
stock = ROOT / 'competition/vfarl/l4x4/kaggle/stock-environment-pins.json'
if stock.is_file():
    paths.append(stock)
bundle = {str(p.relative_to(ROOT)): p.read_text() for p in paths}
packed = base64.b64encode(zlib.compress(json.dumps(bundle).encode())).decode()
source = f'MODE = {args.mode!r}\nMATERIAL_SHA256 = {material["sha256"]!r}\nRUNTIME_SHA256 = {args.runtime_sha256!r}\nCONTEXT_LENGTH = {args.context!r}\nONLY_TASKS = {args.only_task!r}\nSOURCE_BUNDLE = {packed!r}\n'
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
    jail = (ROOT / 'competition/vfarl/l4x4/jail.py').read_text()
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
            'kernel_sources': ['ccjjllt/p005-cpu-nf4-cache',
                               args.runtime_kernel or 'qingyu1232/vfarl-l4x4-cpu-stage/2'],
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
            'material_sha256': material['sha256'], 'mode': args.mode,
            'runtime_kernel': args.runtime_kernel,
            'runtime_sha256': args.runtime_sha256,
            'context_length': args.context,
            'only_tasks': args.only_task,
            'dataset_version': args.dataset_version,
            'command': ['kaggle', 'kernels', 'push', '-p', str(args.out)] + (['--accelerator', 'NvidiaL4'] if gpu else []),
            'accelerator_note': 'NvidiaL4 is the prior measured four physical L4 competition shape; runner measures exact topology again.',
            'time_limit': 'No additional task wall-time cap; platform/account limits still apply.'}
(args.out / 'carrier-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
