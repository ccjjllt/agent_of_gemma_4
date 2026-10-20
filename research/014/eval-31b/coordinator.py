"""Run the frozen P014 paired repair experiment on one continuously batched server.

This is an experiment runner, not a test suite. Inputs and answers stay outside
the Agent sandbox; only issue text and the baseline repository enter it.
The launch file supplies SETTINGS, BUNDLE and WORKER_SOURCE as frozen literals.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(data)
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def locate(name):
    found = list(Path('/kaggle/input').rglob(name))
    if len(found) != 1:
        raise RuntimeError(f'Expected one {name}, found {found[:8]}')
    return found[0]


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=4)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


def main():
    started = time.monotonic()
    run_id = 'run-014-' + SETTINGS['study'] + '-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    out = Path('/kaggle/working/results/014') / run_id
    out.mkdir(parents=True)
    deadline = started + SETTINGS.get('wall_seconds', 5100)
    tasks_path = locate('agent-tasks.json')
    panel = tasks_path.parent
    task_document = json.loads(tasks_path.read_text())
    tasks = task_document if isinstance(task_document, list) else task_document['tasks']
    by_id = {t['instance_id']: t for t in tasks}
    selected = SETTINGS.get('task_ids') or [t['instance_id'] for t in tasks][:SETTINGS.get('task_count', 8)]
    tasks = [by_id[iid] for iid in selected]
    criteria_document = json.loads((panel / 'host-criteria.json').read_text())
    criterion_items = criteria_document if isinstance(criteria_document, list) else criteria_document['items']
    criteria = {item['instance_id']: item for item in criterion_items}
    arms = list(BUNDLE)
    agents = out / 'frozen-agents'
    for arm, files in BUNDLE.items():
        for rel, text in files.items():
            save(agents / arm / rel, text)
    manifest = {'run_id': run_id, 'settings': SETTINGS, 'task_ids': selected,
                'task_input_sha256': digest(tasks_path), 'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'config_sha256': {arm: {rel: hashlib.sha256(text.encode()).hexdigest() for rel, text in files.items()} for arm, files in BUNDLE.items()},
                'planned_attempts': len(tasks) * len(arms), 'concurrency': SETTINGS.get('concurrency', 4),
                'source': 'new developer panel; host-only criteria; no reference edit enters agent',
                'versions': {name: importlib.metadata.version(name) for name in ('swegemma', 'adk-submission', 'vllm', 'torch')}}
    save(out / 'experiment-manifest.json', manifest)
    save(out / 'frozen-tasks.json', tasks)
    import torch
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    model = locate('tokenizer.json') if not SETTINGS.get('model_path_pattern') else None
    if model is None:
        model_dirs = [p.parent for p in Path('/kaggle/input').rglob('tokenizer.json') if SETTINGS['model_path_pattern'] in str(p)]
        if len(model_dirs) != 1:
            raise RuntimeError(f'Model mount ambiguous/missing: {model_dirs}')
        model = model_dirs[0]
    else:
        model = model.parent
    manifest['hardware'] = [{'name': torch.cuda.get_device_name(i), 'memory': torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    if len(manifest['hardware']) != 4 or any('L4' not in x['name'] for x in manifest['hardware']):
        raise RuntimeError(f'Requires four L4s, got {manifest["hardware"]}')
    adapter_manifest = discover_adapters(str(agents / arms[-1]), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    config = VllmConfig(model=str(model), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
                        max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90,
                        enable_auto_tool_choice=True, enable_lora=False, tensor_parallel_size=4,
                        startup_timeout=1200, extra_args=['--max-num-seqs', str(SETTINGS.get('concurrency', 4))])
    server = VllmServer(config, adapter_manifest=adapter_manifest)
    worker = out / 'attempt_worker.py'
    save(worker, WORKER_SOURCE)
    schedule = []
    for slot, task in enumerate(tasks):
        for arm in (arms if slot % 2 == 0 else list(reversed(arms))):
            ordinal = len(schedule) + 1
            entry = {'ordinal': ordinal, 'task_slot': slot + 1, 'task_id': task['instance_id'], 'variant': arm,
                     'attempt_id': f'{ordinal:02d}-{task["instance_id"]}-{arm}'}
            entry['results_dir'] = str(out / 'attempts' / entry['attempt_id'])
            schedule.append((entry, task))
    save(out / 'schedule.json', [entry for entry, task in schedule])
    active, rows = [], []
    telemetry_stop = threading.Event()
    def telemetry():
        with (out / 'gpu-utilization.jsonl').open('a') as log:
            while not telemetry_stop.is_set():
                p = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                log.write(json.dumps({'seconds': time.monotonic()-started, 'samples': p.stdout.strip().splitlines()})+'\n'); log.flush()
                telemetry_stop.wait(5)
    thread = threading.Thread(target=telemetry, daemon=True)
    def collect(record, forced=None):
        entry, process, log, launch_time = record
        log.close()
        path = Path(entry['results_dir']) / 'result.json'
        result = json.loads(path.read_text()) if path.exists() else {'resolved': None, 'error': forced or f'worker exit {process.returncode}'}
        row = {**entry, **result, 'worker_exit_code': process.returncode, 'outer_seconds': time.monotonic()-launch_time}
        if forced:
            row['error'] = forced
        rows.append(row)
        save(out / 'analysis-rows.json', rows)
        print('RESULT ' + json.dumps(row, ensure_ascii=False), flush=True)
    try:
        thread.start()
        print('PHASE model_start', flush=True)
        server.start()
        manifest['model_startup_seconds'] = time.monotonic()-started
        manifest['server_command'] = server.build_cmd()
        save(out / 'experiment-manifest.json', manifest)
        print('PHASE rolling_evaluation', flush=True)
        queue = iter(schedule)
        exhausted = False
        while active or not exhausted:
            while not exhausted and len(active) < SETTINGS.get('concurrency', 4) and time.monotonic() < deadline-150:
                try:
                    entry, task = next(queue)
                except StopIteration:
                    exhausted = True
                    break
                attempt = Path(entry['results_dir']); attempt.mkdir(parents=True)
                snapshot = panel / 'snapshots' / (task['instance_id'] + '.tar.gz')
                spec = {'entry': entry, 'agent_dir': str(agents / entry['variant']), 'task': task,
                        'snapshot': str(snapshot), 'base_url': str(server.base_url),
                        'served_model': str(getattr(server, 'effective_model_path', model)), 'model_alias': SETTINGS['model_alias'],
                        'budgets': SETTINGS['budgets'], 'judge': criteria[task['instance_id']],
                        'panel_root': str(panel), 'total_tasks': len(schedule)}
                save(attempt / 'spec.json', spec)
                log = (attempt / 'worker.log').open('w')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
                process = subprocess.Popen([sys.executable, '-u', str(worker), str(attempt/'spec.json')], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active.append((entry, process, log, time.monotonic()))
                print('LAUNCH ' + entry['attempt_id'], flush=True)
            if time.monotonic() >= deadline-150:
                exhausted = True
            for record in active[:]:
                entry, process, log, launch_time = record
                forced = None
                if process.poll() is None and (time.monotonic()-launch_time > SETTINGS.get('attempt_seconds', 1020) or time.monotonic() >= deadline-60):
                    forced = 'experiment_wall_timeout'; stop(process)
                if process.poll() is not None:
                    collect(record, forced); active.remove(record)
            if active:
                time.sleep(1)
    except BaseException:
        save(out / 'failure.txt', traceback.format_exc())
        raise
    finally:
        for record in active:
            stop(record[1]); collect(record, 'interrupted')
        telemetry_stop.set(); thread.join(timeout=10)
        server_log = getattr(server, 'log_path', None)
        if server_log and Path(server_log).exists():
            shutil.copyfile(server_log, out/'vllm-server.log')
        server.stop(timeout=10)
        summary = {'returned_attempts': len(rows), 'completed_attempts': sum(r.get('worker_exit_code') == 0 and r.get('resolved') is not None for r in rows),
                   'planned_attempts': len(schedule), 'wall_seconds': time.monotonic()-started,
                   'arms': {arm: {'resolved': sum(r.get('resolved') is True and r['variant']==arm for r in rows), 'attempts': sum(r['variant']==arm for r in rows)} for arm in arms}}
        save(out / 'summary.json', summary)
        print('FINAL ' + json.dumps(summary), flush=True)
        with zipfile.ZipFile('/kaggle/working/p014-'+SETTINGS['study']+'-output.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in out.rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(out.parent))


if __name__ == '__main__':
    main()
