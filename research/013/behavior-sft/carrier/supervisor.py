"""Supervise one P013 behavior SFT job; preserve complete checkpoints on timeout."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid


HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / 'run-manifest.json').read_text())
LIMIT = int(PLAN['worker_wall_seconds'])
RUN_ID = 'run-013-behavior-sft-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
OUT = Path('/kaggle/working/results/013') / RUN_ID
OUT.mkdir(parents=True)
started = time.monotonic()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def descendants(root):
    parents = {}
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            value = next(x for x in path.read_text().splitlines() if x.startswith('PPid:'))
            parents[int(path.parent.name)] = int(value.split()[1])
        except (OSError, ValueError, StopIteration):
            pass
    found, frontier = [], [root]
    while frontier:
        frontier = [pid for pid, parent in parents.items() if parent in frontier and pid not in found]
        found.extend(frontier)
    return found


def stop(process):
    pids = list(reversed(descendants(process.pid))) + [process.pid]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(3)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass


def telemetry():
    row = {'time': time.time(), 'elapsed_seconds': time.monotonic() - started}
    try:
        row['phase'] = json.loads((OUT / 'phase.json').read_text())
    except (OSError, ValueError):
        pass
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        row['gpu_csv'] = result.stdout.strip()
    except Exception as exc:
        row['gpu_sampling_error'] = repr(exc)
    with (OUT / 'gpu-telemetry.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')


write_json(OUT / 'run-plan.json', PLAN)
write_json(OUT / 'carrier-source.json', json.loads((HERE / 'carrier-manifest.json').read_text()))
env = dict(os.environ, P013_RUN_ROOT=str(OUT), P013_RUN_ID=RUN_ID,
           P013_DEADLINE_MONOTONIC=str(started + LIMIT - 15), PYTHONUNBUFFERED='1',
           PYTHONDONTWRITEBYTECODE='1')
process = None
timed_out = False
error = None
try:
    with (OUT / 'runner.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', str(HERE / 'runner.py')],
                                   stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        offset, sample_at = 0, 0.0
        while process.poll() is None:
            if time.monotonic() - started >= LIMIT:
                timed_out = True
                break
            if time.monotonic() >= sample_at:
                telemetry()
                sample_at = time.monotonic() + 30
                with (OUT / 'runner.log').open() as stream:
                    stream.seek(offset)
                    text = stream.read()
                    offset = stream.tell()
                if text:
                    print(text[-12000:], end='', flush=True)
                metrics = OUT / 'training/train-run/metrics.jsonl'
                if metrics.is_file():
                    lines = metrics.read_text().splitlines()
                    if lines:
                        print('latest_training_metric ' + lines[-1], flush=True)
            time.sleep(1)
except BaseException:
    error = traceback.format_exc()
    (OUT / 'supervisor-exception.txt').write_text(error)
finally:
    if process is not None:
        stop(process)
    completed = OUT / 'run-summary.json'
    checkpoints = []
    for marker in sorted((OUT / 'training/train-run').glob('checkpoint-*/progress.json')):
        required = ['training_state.pt', 'adapter/adapter_model.safetensors'] + [f'rank-{r:03d}-rng.pt' for r in range(4)]
        if all((marker.parent / name).is_file() for name in required):
            checkpoints.append({'path': str(marker.parent), 'progress': json.loads(marker.read_text())})
    summary = {'run_id': RUN_ID, 'status': 'completed' if completed.is_file() and not timed_out and not error else 'incomplete',
               'timed_out': timed_out, 'exit_code': process.returncode if process else None,
               'wall_seconds': time.monotonic() - started, 'worker_wall_seconds': LIMIT,
               'complete_checkpoints': checkpoints, 'error': error,
               'platform_wall_seconds': PLAN['platform_wall_seconds']}
    write_json(OUT / 'supervisor-report.json', summary)
    print(json.dumps(summary), flush=True)
if summary['status'] != 'completed':
    raise RuntimeError('P013 behavior SFT incomplete; logs and complete checkpoints retained at ' + str(OUT))
