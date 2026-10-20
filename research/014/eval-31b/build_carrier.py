"""Freeze the actual P014 experiment into Kaggle's single uploaded code file."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

BOOTSTRAP = r'''
from pathlib import Path
import importlib
import json
import os
import subprocess
import sys
import time

os.environ.update({'LITELLM_LOCAL_MODEL_COST_MAP':'True', 'TRANSFORMERS_NO_TF':'1',
                   'VLLM_WORKER_MULTIPROC_METHOD':'spawn', 'VLLM_NO_USAGE_STATS':'1',
                   'VLLM_ENGINE_READY_TIMEOUT_S':'1200', 'OTEL_SDK_DISABLED':'true',
                   'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
inputs = Path('/kaggle/input')
markers = list(inputs.rglob('swegemma-0.2.11-py3-none-any.whl'))
if len(markers) != 1:
    raise RuntimeError('Official wheelhouse mount missing/ambiguous: '+str(markers))
wheelhouse = markers[0].parent
staging = Path('/tmp/p014-official-wheels')
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
panel_markers = list(inputs.rglob('agent-tasks.json'))
if len(panel_markers) != 1:
    raise RuntimeError('CPU panel mount missing/ambiguous: '+str(panel_markers))
panel = panel_markers[0].parent
extra_wheels = list((panel/'wheelhouse').glob('*.whl')) + list((panel/'wheels').glob('*.whl'))
if extra_wheels:
    subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps',*[str(p) for p in extra_wheels]],check=True)
importlib.invalidate_caches()
print('PHASE inputs_ready', flush=True)
'''


def main():
    panel_path = ROOT/'research/014/panel/material/agent-tasks.json'
    doc = json.loads(panel_path.read_text())
    tasks = doc if isinstance(doc,list) else doc['tasks']
    baseline = ROOT/'research/007/candidates/public022_v10/agent'
    candidate = ROOT/'research/014/candidate/agent'
    bundles = {name: {str(p.relative_to(folder)):p.read_text() for p in sorted(folder.rglob('*')) if p.is_file()}
               for name,folder in [('public022_v10',baseline),('p014_single_v1',candidate)]}
    settings = {'study':'31b-pair-v1', 'model_alias':'gemma-4-31b-it-qat-w4a16-ct',
                'model_path_pattern':'gemma-4-31b-it-qat-w4a16-ct/2',
                'task_ids':[t['instance_id'] for t in tasks][:8], 'task_count':8,
                'budgets':{'timeout_seconds':240,'max_tool_calls':28,'max_time_minutes':8,'max_turns':80},
                'concurrency':4,'wall_seconds':5100,'attempt_seconds':1020,
                'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'quota_estimate_upper_hours':3.2,'purpose':'actual repair wins/losses on frozen new developer tasks'}
    worker = (HERE/'attempt_worker.py').read_text()
    coordinator = (HERE/'coordinator.py').read_text()
    source = '# Frozen P014 research run; no competition submission.\n'+BOOTSTRAP+'\nSETTINGS = '+repr(settings)+'\nBUNDLE = '+repr(bundles)+'\nWORKER_SOURCE = '+repr(worker)+'\n'+coordinator
    ast.parse(source)
    kernel=HERE/'kernel'; kernel.mkdir(exist_ok=True)
    (kernel/'run.py').write_text(source)
    metadata=json.loads((ROOT/'research/013/eval-31b-diag/kernel-v3/kernel-metadata.json').read_text())
    metadata.update(id='zhuating/p014-31b-pair-v1',title='P014 31B Pair V1',code_file='run.py',
                    kernel_sources=['zhuating/p014-frozen-eight-material-cpu'])
    (kernel/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    manifest={'settings':settings,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
              'worker_sha256':hashlib.sha256(worker.encode()).hexdigest(),
              'coordinator_sha256':hashlib.sha256(coordinator.encode()).hexdigest(),
              'agent_task_sha256':hashlib.sha256(panel_path.read_bytes()).hexdigest(),
              'members':{arm:{rel:hashlib.sha256(text.encode()).hexdigest() for rel,text in files.items()} for arm,files in bundles.items()},
              'source_bytes':len(source.encode()),'research_result':None}
    (HERE/'run-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'kernel':str(kernel),'bytes':manifest['source_bytes'],'sha256':manifest['source_sha256'],'task_ids':settings['task_ids']},ensure_ascii=False))


if __name__=='__main__':
    main()
