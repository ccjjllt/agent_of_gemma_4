"""Build a private, bounded GPU probe. Building never launches Kaggle work."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
worker = (ROOT / 'gpu-worker.py').read_text()
wrapper = r'''
import hashlib, json, os, platform, signal, subprocess, sys, threading, time, traceback
from pathlib import Path
OUT = Path('/kaggle/working')
START = time.monotonic()
report = {'python': sys.version, 'platform': platform.platform(), 'worker_timeout_seconds': 900,
          'server_timeout_seconds': 1200, 'phases': []}
def save(phase, **values):
    report.update(values)
    report['phases'].append({'phase': phase, 'elapsed_seconds': time.monotonic() - START})
    (OUT / 'gpu-wrapper-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'phase': phase, **values}), flush=True)
stop = threading.Event()
def monitor():
    with (OUT / 'gpu-telemetry.jsonl').open('w', buffering=1) as fh:
        while not stop.is_set():
            try:
                p = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw',
                                    '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=8)
                fh.write(json.dumps({'elapsed_seconds': time.monotonic()-START, 'csv': p.stdout.strip(), 'code': p.returncode})+'\n')
            except Exception as exc:
                fh.write(json.dumps({'error': repr(exc)})+'\n')
            stop.wait(10)
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
proc = None
try:
    stages = list(Path('/kaggle/input').rglob('cpu-stage-report.json'))
    if len(stages) != 1:
        raise RuntimeError('Expected one CPU stage output')
    stage = json.loads(stages[0].read_text())
    if not stage.get('success'):
        raise RuntimeError('CPU preparation did not complete')
    if stage['python'].split()[0].split('.')[:2] != platform.python_version().split('.')[:2]:
        raise RuntimeError('CPU/GPU Python ABI differs; prepare compatible wheels on CPU first')
    wheel_dir = stages[0].parent / 'training-wheels'
    wheels = []
    for entry in stage['wheels']:
        path = wheel_dir / entry['name']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError('Staged wheel hash mismatch: '+entry['name'])
        if path.name.lower().startswith(('torch-', 'triton-', 'nvidia_')):
            raise RuntimeError('GPU torch replacement is prohibited')
        wheels.append(str(path))
    save('cpu_stage_verified', cpu_stage=str(stages[0]), wheel_count=len(wheels))
    target = '/tmp/p005-training-deps'
    installed = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                                '--target', target, *wheels], timeout=120)
    if installed.returncode:
        raise RuntimeError('Offline dependency installation failed')
    save('offline_dependencies_installed')
    worker_path = Path('/tmp/p005-gpu-worker.py')
    worker_path.write_text(WORKER_SOURCE)
    env = dict(os.environ, PYTHONPATH=target, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', PYTORCH_ALLOC_CONF='expandable_segments:True')
    with (OUT / 'gpu-worker.log').open('w') as log:
        proc = subprocess.Popen([sys.executable, str(worker_path)], env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        def relay():
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if line.startswith('{'):
                    print(line.rstrip(), flush=True)
        reader = threading.Thread(target=relay, daemon=True)
        reader.start()
        save('worker_started', pid=proc.pid)
        try:
            code = proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            save('worker_timeout', success=False)
        else:
            save('worker_finished', success=(code == 0), worker_exit_code=code)
        reader.join(timeout=10)
    print((OUT / 'gpu-worker.log').read_text()[-18000:], flush=True)
except Exception as exc:
    save('failed', success=False, error=repr(exc), traceback=traceback.format_exc())
    raise
finally:
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
    stop.set()
    thread.join(timeout=10)
    save('session_end')
if not report.get('success', False):
    raise SystemExit(report.get('worker_exit_code') or 1)
'''
dest = ROOT / 'gpu-probe'
dest.mkdir(exist_ok=True)
(dest / 'probe.py').write_text('WORKER_SOURCE = ' + repr(worker) + '\n' + wrapper)
metadata = {
    'id': 'ccjjllt/p005-training-gpu-probe', 'title': 'P005 Training GPU Probe',
    'code_file': 'probe.py', 'language': 'python', 'kernel_type': 'script', 'is_private': True,
    'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
    'dataset_sources': [], 'competition_sources': ['gemma-4-developer-agent'],
    'kernel_sources': ['ccjjllt/p005-training-cpu-stage'],
    'model_sources': ['google/gemma-4/transformers/gemma-4-31b-it-qat-q4_0-unquantized/2'],
}
(dest / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
print(dest)
