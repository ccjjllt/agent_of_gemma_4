# Execution starts here. This cell supervises a fresh interpreter containing the
# entire installation/server/evaluation lifecycle, including a hard wall bound.
from pathlib import Path
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
import uuid
import zipfile

WALL_SECONDS = min(1680, int(os.environ.get('P010_WALL_SECONDS', '1680')))  # Includes pip installation, model loading and compilation.
PREPARE_ONLY = os.environ.get('P010_PREPARE_ONLY') == '1' or '--prepare-only' in sys.argv
RUN_ID = 'run-010-budget-pair-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
RUN_ROOT = Path('/kaggle/working/results/010') / RUN_ID
RUN_ROOT.mkdir(parents=True, exist_ok=False)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sample():
    row = {'time': time.time(), 'elapsed_seconds': time.monotonic() - started}
    phase_path = RUN_ROOT / 'phase.json'
    try:
        row.update(json.loads(phase_path.read_text()))
        row['phase_at'] = row.pop('time')
        row['time'] = time.time()
    except (OSError, ValueError):
        row['phase'] = 'setup'
    try:
        response = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw',
                                   '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        row['gpu_csv'] = response.stdout.strip()
        row['nvidia_smi_exit_code'] = response.returncode
        row['gpus'] = []
        for line in response.stdout.splitlines():
            index, name, util, used, total, power = [part.strip() for part in line.split(',')]
            row['gpus'].append({'index': int(index), 'name': name, 'utilization_percent': float(util),
                                'memory_used_mib': float(used), 'memory_total_mib': float(total),
                                'power_w': None if power == '[N/A]' else float(power)})
    except Exception as exc:
        row['gpu_sampling_error'] = str(exc)
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/metrics', timeout=2) as response:
            metrics = response.read().decode()
        prefixes = ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:gpu_cache_usage_perc',
                    'vllm:kv_cache_usage_perc', 'vllm:generation_tokens_total', 'vllm:prompt_tokens_total')
        selected = [line for line in metrics.splitlines() if line.startswith(prefixes)]
        row['vllm_metrics'] = selected
        row['vllm_requests_running'] = sum(float(line.rsplit(' ', 1)[1]) for line in selected
                                          if line.startswith('vllm:num_requests_running'))
        row['vllm_requests_waiting'] = sum(float(line.rsplit(' ', 1)[1]) for line in selected
                                          if line.startswith('vllm:num_requests_waiting'))
    except Exception as exc:
        row['vllm_metrics_unavailable'] = str(exc)
    with (RUN_ROOT / 'gpu-telemetry.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
        stream.flush()
    return row



def descendant_pids(root_pid):
    parents = {}
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            row = next(line for line in path.read_text().splitlines() if line.startswith('PPid:'))
            parents[int(path.parent.name)] = int(row.split()[1])
        except (OSError, ValueError, StopIteration):
            pass
    found, frontier = [], [root_pid]
    while frontier:
        frontier = [pid for pid, parent in parents.items() if parent in frontier and pid not in found]
        found.extend(frontier)
    return found


def stop_process_tree(process, grace=3, own_group=False):
    children = descendant_pids(process.pid)
    pids = list(reversed(children)) + [process.pid]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if own_group:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(grace)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass

if sys.version_info[:2] != (3, 12) or platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
    raise RuntimeError('Use the pinned Starter Linux CPython 3.12 image; no packages or model loaded')
runner = Path(__file__).parent / 'coordinator.py'
SOURCE_MANIFEST = json.loads((Path(__file__).parent / 'carrier-manifest.json').read_text())
write_json(RUN_ROOT / 'carrier-source.json', SOURCE_MANIFEST)
started = time.monotonic()
env = dict(os.environ, P010_RUN_ID=RUN_ID, P010_RUN_ROOT=str(RUN_ROOT), PYTHONUNBUFFERED='1',
           P010_PREPARE_ONLY='1' if PREPARE_ONLY else '0',
           P010_EVALUATION_DEADLINE=str(started + min(1620, WALL_SECONDS - 60)))
process = None
timed_out = False
error = None
rows = []
try:
    with (RUN_ROOT / 'coordinator.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', str(runner)], env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        offset = 0
        next_sample = 0.0
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= WALL_SECONDS:
                timed_out = True
                break
            if elapsed >= next_sample:
                rows.append({'phase': 'cpu_prepare', 'time': time.time(), 'elapsed_seconds': elapsed} if PREPARE_ONLY else sample())
                next_sample = time.monotonic() - started + 10
                with (RUN_ROOT / 'coordinator.log').open() as follow:
                    follow.seek(offset)
                    text = follow.read()
                    offset = follow.tell()
                if text:
                    print(text[-16000:], end='', flush=True)
            time.sleep(min(1, max(0, WALL_SECONDS - (time.monotonic() - started))))
except BaseException:
    error = traceback.format_exc()
    (RUN_ROOT / 'supervisor_exception.txt').write_text(error)
finally:
    if process is not None:
        # Includes server processes, both Evaluators, and separately sessioned
        # sandbox descendants. No retry or second model startup.
        stop_process_tree(process, own_group=True)
    result_path = RUN_ROOT / 'attempt_status.json'
    states = json.loads(result_path.read_text()) if result_path.exists() else {}
    for attempt_id, state in states.items():
        local = Path(state['results_dir']) / 'attempt_status.json'
        if local.is_file():
            states[attempt_id] = json.loads(local.read_text())
        if states[attempt_id].get('status') in ('running', 'initializing', 'planned'):
            states[attempt_id]['status'] = 'outer_wall_timeout' if timed_out else 'interrupted_or_not_run'
    write_json(RUN_ROOT / 'attempt_status.json', states)
    expected = 'prepared' if PREPARE_ONLY else 'completed'
    complete = len(states) == (2 if PREPARE_ONLY else 6) and all(item.get('status') == expected for item in states.values())
    ok = process is not None and process.returncode == 0 and complete and not timed_out and error is None
    report = {'run_id': RUN_ID, 'status': ('prepared' if PREPARE_ONLY else 'completed') if ok else 'partial_or_failed',
              'mode': 'cpu_prepare' if PREPARE_ONLY else 'evaluate',
              'model_requests': 0 if PREPARE_ONLY else None, 'judge_runs': 0 if PREPARE_ONLY else None,
              'exit_code': None if process is None else process.returncode, 'timed_out': timed_out,
              'wall_seconds': time.monotonic() - started, 'hard_worker_limit_seconds': WALL_SECONDS,
              'attempt_states': states, 'completed_attempts': sum(s.get('status') == 'completed' for s in states.values()),
              'prepared_agents': sum(s.get('status') == 'prepared' for s in states.values()),
              'internal_lifecycle_target_seconds': 1740, 'platform_session_limit_seconds': 1800,
              'configured_max_concurrent_attempts': 4, 'image_pin': SOURCE_MANIFEST['image_pin'],
              'quota_note': 'Kaggle quota accounting is external; actual wall and GPU samples are retained.'}
    # Averages are sampled utilization, never throughput or complete GPU busy time.
    report['sampled_phase_summary'] = {}
    for phase in sorted({row['phase'] for row in rows}):
        selected = [row for row in rows if row['phase'] == phase]
        by_gpu = {}
        for row in selected:
            for gpu in row.get('gpus', []):
                by_gpu.setdefault(str(gpu['index']), []).append(gpu['utilization_percent'])
        report['sampled_phase_summary'][phase] = {
            'samples': len(selected), 'gpu_mean_utilization_percent': {k: sum(v) / len(v) for k, v in by_gpu.items()},
            'vllm_samples_available': sum('vllm_requests_running' in row for row in selected),
            'maximum_sampled_requests_running': max((row['vllm_requests_running'] for row in selected
                                                     if 'vllm_requests_running' in row), default=None)}
    write_json(RUN_ROOT / 'supervisor-report.json', report)
    # A hard-killed official server cannot run stop(); retain its original log.
    locator = RUN_ROOT / 'server-log-path.json'
    output_deadline = started + min(1740, WALL_SECONDS + 60)
    if locator.is_file():
        source = Path(json.loads(locator.read_text())['path'])
        if source.is_file():
            with source.open('rb') as incoming, (RUN_ROOT / 'vllm_server.archived.log').open('wb') as outgoing:
                while time.monotonic() < output_deadline:
                    chunk = incoming.read(1024 * 1024)
                    if not chunk:
                        break
                    outgoing.write(chunk)
    archive_path = RUN_ROOT.parent / (RUN_ID + '.zip')
    try:
        # Stored ZIP avoids compression using the remaining output budget.
        # Original files remain complete even if archive time is exhausted.
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_STORED) as archive:
            for path in sorted(RUN_ROOT.rglob('*')):
                if path.is_file():
                    with path.open('rb') as incoming, archive.open(str(path.relative_to(RUN_ROOT)), 'w') as outgoing:
                        while True:
                            if time.monotonic() >= output_deadline:
                                raise TimeoutError('Archive budget exhausted; use original per-file outputs')
                            chunk = incoming.read(1024 * 1024)
                            if not chunk:
                                break
                            outgoing.write(chunk)
    except Exception as exc:
        report['archive_error'] = str(exc)
        write_json(RUN_ROOT / 'supervisor-report.json', report)
    print(json.dumps({'status': report['status'], 'output': str(RUN_ROOT), 'archive': str(archive_path),
                      'completed_attempts': report['completed_attempts'], 'wall_seconds': report['wall_seconds']}), flush=True)
if not ok:
    raise RuntimeError('Paired experiment incomplete; see retained supervisor report, logs and per-attempt evidence')
