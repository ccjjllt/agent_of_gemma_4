"""Run the frozen P015 paired repair experiment on one continuously batched server.

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
import urllib.error
import urllib.request
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


def prepare_capacity_diagnostic(settings, out, discover_adapters, extensions):
    record = {'purpose': settings['purpose'], 'name': settings['name'],
              'request_attempted': False, 'repair_statistics_included': False,
              'runtime_compatibility_verified': False}
    try:
        candidates = [path for path in Path('/kaggle/input').rglob('training_manifest.json')
                      if digest(path) == settings['training_manifest_sha256']]
        if len(candidates) != 1:
            raise RuntimeError('Completed capacity training manifest missing/ambiguous')
        source_manifest = candidates[0]
        actual = json.loads(source_manifest.read_text())
        if (actual['signature'] != settings['training_signature']
                or actual['adapter']['files'] != settings['files']
                or actual['status'] != 'training_completed'):
            raise RuntimeError('Mounted capacity manifest differs from frozen source')
        source = source_manifest.parent/'adapter'  # final, never a checkpoint
        root = Path('/tmp')/('p015-capacity-diagnostic-'+out.name)
        destination = root/'adapters'/settings['name']
        destination.mkdir(parents=True, exist_ok=False)
        identities = {}
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            expected = settings['files'][name]
            path = source/name
            if path.is_symlink() or path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
                raise RuntimeError('Capacity adapter file identity mismatch: '+name)
            shutil.copyfile(path, destination/name)
            if digest(destination/name) != expected['sha256']:
                raise RuntimeError('Capacity adapter copy changed: '+name)
            identities[name] = expected
        adapters = discover_adapters(str(root), adapter_extensions=extensions)
        if set(adapters.adapters) != {settings['name']} or adapters.adapters[settings['name']].rank != 8:
            raise RuntimeError('Capacity adapter discovery differs from one rank8 adapter')
        record.update(status='prepared', source_manifest=str(source_manifest),
                      source_adapter=str(source), actual_served_path=str(destination),
                      files=identities, discovered_names=list(adapters.adapters))
        return adapters, record
    except Exception:
        record.update(status='preparation_failed', error=traceback.format_exc())
        return None, record


def request_capacity_diagnostic(server, record, out):
    endpoint = str(server.base_url).rstrip('/')+'/chat/completions'
    payload = {'model': record['name'], 'messages': [{'role': 'user', 'content': 'hello'}],
               'max_tokens': 8, 'temperature': 0}
    record.update(request_attempted=True, request_url=endpoint, request=payload,
                  actual_base_model_path=str(server.config.model))
    started = time.monotonic()
    try:
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=60) as response:
            record['http_status'] = response.status
            record['response'] = json.loads(response.read())
        result = record['response']
        record['actual_response_model'] = result.get('model')
        record['runtime_compatibility_verified'] = bool(
            result.get('model') == record['name'] and result.get('choices'))
        record['status'] = 'request_completed' if record['runtime_compatibility_verified'] else 'response_model_unconfirmed'
    except Exception as exc:
        record.update(status='request_failed', error=traceback.format_exc())
        if isinstance(exc, urllib.error.HTTPError):
            record.update(http_status=exc.code, response_error_body=exc.read().decode(errors='replace'))
    record['request_seconds'] = time.monotonic()-started
    save(out/'capacity-adapter-diagnostic.json', record)


def main():
    started = time.monotonic()
    run_id = 'run-015-' + SETTINGS['study'] + '-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    out = Path('/kaggle/working/results/015') / run_id
    out.mkdir(parents=True)
    deadline = started + SETTINGS.get('wall_seconds', 5100)
    panel = Path(MATERIAL_ROOT)
    tasks_path = panel / 'agent-tasks.json'
    task_document = json.loads(tasks_path.read_text())
    tasks = task_document if isinstance(task_document, list) else task_document['tasks']
    tasks = [{**t, 'instance_id': t.get('instance_id', t.get('iid')), 'problem_statement': t.get('problem_statement', t.get('issue'))} for t in tasks]
    by_id = {t['instance_id']: t for t in tasks}
    selected = SETTINGS.get('task_ids') or [t['instance_id'] for t in tasks][:SETTINGS.get('task_count', 8)]
    tasks = [by_id[iid] for iid in selected]
    criteria_document = json.loads((panel / 'host-criteria.json').read_text())
    criterion_items = criteria_document if isinstance(criteria_document, list) else criteria_document['tasks']
    criteria = {item.get('instance_id', item.get('iid')): item for item in criterion_items}
    arms = list(BUNDLE)
    repeats = SETTINGS.get('repeats', 1)
    if type(repeats) is not int or repeats not in (1, 2):
        raise RuntimeError('repeats must be 1 or 2')
    agents = out / 'frozen-agents'
    for arm, files in BUNDLE.items():
        for rel, text in files.items():
            save(agents / arm / rel, text)
    manifest = {'run_id': run_id, 'settings': SETTINGS, 'task_ids': selected,
                'task_input_sha256': digest(tasks_path), 'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'config_sha256': {arm: {rel: hashlib.sha256(text.encode()).hexdigest() for rel, text in files.items()} for arm, files in BUNDLE.items()},
                'planned_attempts': len(tasks) * len(arms) * repeats,
                'independent_issues': len(tasks), 'draws_per_issue_per_arm': repeats,
                'concurrency': SETTINGS.get('concurrency', 4),
                'source': 'frozen natural-PR dev panel and separate train sampling; source criteria stay host-side',
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
    base_adapters = discover_adapters(str(agents / arms[-1]), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    if base_adapters.adapters:
        raise RuntimeError('Base sampling Agent must not contain an adapter')
    diagnostic_manifest, diagnostic = prepare_capacity_diagnostic(
        SETTINGS['capacity_adapter_diagnostic'], out, discover_adapters, ALLOWED_ADAPTER_EXTENSIONS)
    save(out/'capacity-adapter-diagnostic.json', diagnostic)
    def make_server(with_diagnostic):
        config = VllmConfig(model=str(model), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
                            max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90,
                            enable_auto_tool_choice=True, enable_lora=with_diagnostic,
                            max_loras=1, max_lora_rank=8, tensor_parallel_size=4,
                            startup_timeout=1200, extra_args=['--max-num-seqs', str(SETTINGS.get('concurrency', 4))])
        return VllmServer(config, adapter_manifest=diagnostic_manifest if with_diagnostic else base_adapters)
    serving_diagnostic = diagnostic_manifest is not None
    server = make_server(serving_diagnostic)
    manifest['server_launch_attempts'] = []
    worker = out / 'attempt_worker.py'
    save(worker, WORKER_SOURCE)
    schedule = []
    for repeat_index in range(1, repeats + 1):
        for slot, task in enumerate(tasks):
            for arm in (arms if slot % 2 == 0 else list(reversed(arms))):
                ordinal = len(schedule) + 1
                suffix = f'-r{repeat_index}' if repeats > 1 else ''
                entry = {'ordinal': ordinal, 'task_slot': slot + 1, 'task_id': task['instance_id'], 'variant': arm,
                         'repeat_index': repeat_index,
                         'attempt_id': f'{ordinal:02d}-{task["instance_id"]}-{arm}{suffix}',
                         'split': criteria[task['instance_id']]['split']}
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
        launch = {'enable_lora': serving_diagnostic, 'command': server.build_cmd()}
        manifest['server_launch_attempts'].append(launch)
        save(out/'experiment-manifest.json', manifest)
        try:
            server.start()
            launch['status'] = 'started'
        except Exception:
            launch.update(status='startup_failed', error=traceback.format_exc())
            if not serving_diagnostic:
                raise
            diagnostic.update(status='adapter_server_startup_failed', startup_error=launch['error'],
                              fallback_to_base=True,
                              comparison_requirement='Future weight A/B must rerun both arms with this same fallback service configuration')
            # The official start() also invokes stop() on startup failure. Call
            # it explicitly before replacing the server; preserve any remaining
            # log first because official stop() may delete its temporary file.
            failed_log = getattr(server, 'log_path', None)
            if failed_log and Path(failed_log).exists():
                shutil.copyfile(failed_log, out/'vllm-capacity-startup-failed.log')
            server.stop(timeout=10)
            serving_diagnostic = False
            save(out/'capacity-adapter-diagnostic.json', diagnostic)
            server = make_server(False)
            fallback = {'enable_lora': False, 'command': server.build_cmd(), 'reason': 'capacity adapter startup failed'}
            manifest['server_launch_attempts'].append(fallback)
            save(out/'experiment-manifest.json', manifest)
            server.start()
            fallback['status'] = 'started'
        manifest['model_startup_seconds'] = time.monotonic()-started
        manifest['server_command'] = server.build_cmd()
        manifest['serving_conditions'] = {
            'actual_base_model_path': str(server.config.model), 'enable_lora': serving_diagnostic,
            'mounted_adapters': [diagnostic['name']] if serving_diagnostic else [],
            'capacity_diagnostic_excluded_from_repair_statistics': True,
            'base_requests_use_adapter': False,
            'future_weight_ab_requirement': 'Both arms must use identical enable_lora, mounted adapters, rank, concurrency and service settings; fallback configuration requires a matching rerun'}
        save(out / 'experiment-manifest.json', manifest)
        if serving_diagnostic:
            diagnostic['server_command'] = server.build_cmd()
            request_capacity_diagnostic(server, diagnostic, out)
        print('CAPACITY_DIAGNOSTIC ' + json.dumps({
            'status': diagnostic['status'],
            'model': diagnostic.get('actual_response_model'),
            'requested_model': diagnostic['name'], 'enable_lora': serving_diagnostic}), flush=True)
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
                        'served_model': str(server.config.model), 'model_alias': SETTINGS['model_alias'],
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
                   'independent_issues_planned': len(tasks),
                   'independent_issues_attempted': len({r['task_id'] for r in rows}),
                   'draws_per_issue_per_arm': repeats,
                   'count_unit': 'attempt draw; repeated draws are not independent issues',
                   'by_split': {split: {
                       'planned_attempts': sum(entry['split'] == split for entry, _task in schedule),
                       'independent_issues_planned': len({entry['task_id'] for entry, _task in schedule if entry['split'] == split}),
                       'arms': {arm: {
                           'resolved': sum(r.get('resolved') is True and r['variant'] == arm and r['split'] == split for r in rows),
                           'attempts': sum(r['variant'] == arm and r['split'] == split for r in rows),
                           'completed_attempts': sum(r.get('worker_exit_code') == 0 and r.get('resolved') is not None and r['variant'] == arm and r['split'] == split for r in rows),
                           'independent_issues_attempted': len({r['task_id'] for r in rows if r['variant'] == arm and r['split'] == split}),
                           'issues_with_any_success': len({r['task_id'] for r in rows if r.get('resolved') is True and r['variant'] == arm and r['split'] == split}),
                       } for arm in arms},
                   } for split in sorted({entry['split'] for entry, _task in schedule})}}
        save(out / 'summary.json', summary)
        print('FINAL ' + json.dumps(summary), flush=True)
        with zipfile.ZipFile('/kaggle/working/p015-'+SETTINGS['study']+'-output.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in out.rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(out.parent))


if __name__ == '__main__':
    main()
