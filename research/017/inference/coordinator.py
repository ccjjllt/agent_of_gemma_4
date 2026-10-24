"""P017 paired evaluation; emitted after frozen SETTINGS, BUNDLE and worker source."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
import uuid
import zipfile


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


def stage_and_prove(agents, discover, extensions):
    pin = SETTINGS['adapter']
    matches = [p for p in Path('/kaggle/input').rglob('training_manifest.json')
               if digest(p) == pin['manifest_sha256']]
    if len(matches) != 1:
        raise RuntimeError('Authentic Step-68 manifest missing/ambiguous')
    training = json.loads(matches[0].read_text())
    if (training['signature'] != pin['signature'] or training['status'] != 'training_completed'
            or training['metrics']['optimizer_steps'] != 68 or training['adapter']['files'] != pin['files']):
        raise RuntimeError('Mounted manifest differs from the frozen Step-68 result')
    if pin['layout'] == 'flat':
        source = matches[0].parent
    elif pin['layout'] == 'nested':
        source = matches[0].parent / 'adapter'
    else:
        raise RuntimeError('Unknown explicitly frozen adapter layout')
    destination = agents / 'adapter' / 'adapters' / pin['name']
    destination.mkdir(parents=True, exist_ok=False)
    for name, expected in pin['files'].items():
        path = source / name
        if path.is_symlink() or path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise RuntimeError('Final adapter identity mismatch: ' + name)
        shutil.copyfile(path, destination / name)
        if digest(destination / name) != expected['sha256']:
            raise RuntimeError('Final adapter copy changed')
    adapters = discover(str(agents / 'adapter'), adapter_extensions=extensions)
    if set(adapters.adapters) != {pin['name']} or adapters.adapters[pin['name']].rank != 8:
        raise RuntimeError('Expected exactly one rank8 Step-68 adapter')
    if discover(str(agents / 'base'), adapter_extensions=extensions).adapters:
        raise RuntimeError('Base Agent must select no adapter')
    import yaml
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))
    base = yaml.load((agents / 'base/agent.yaml').read_text(), Loader=Loader)
    adapted = yaml.load((agents / 'adapter/agent.yaml').read_text(), Loader=Loader)
    if (base.get('adapter') or adapted.pop('adapter', None) != pin['name']
            or base != adapted or base['model'] != SETTINGS['model_alias']):
        raise RuntimeError('Agent arms differ beyond the official adapter field')
    for relative in BUNDLE['base']:
        if relative != 'agent.yaml' and (agents / 'base' / relative).read_bytes() != (agents / 'adapter' / relative).read_bytes():
            raise RuntimeError('Agent prompts/tools/sampling differ')
    return adapters, {'source_manifest': str(matches[0]), 'source_adapter': str(source), 'layout': pin['layout'],
                      'actual_served_path': str(destination), 'files': pin['files'],
                      'official_route_field': 'adapter', 'expected_adapter_alias': pin['name']}


def route_evidence(attempt, expected):
    budget = SETTINGS['thinking_budget']
    result = {'expected_response_model': expected, 'expected_thinking_budget': budget, 'verified': False}
    try:
        records = [json.loads(line) for line in (attempt / 'raw-wire.jsonl').read_text().splitlines()]
        requests = [r for r in records if r['kind'] == 'wire_request']
        responses = [r for r in records if r['kind'] == 'wire_response']
        requested = sorted({r['request']['model'] for r in requests})
        served = sorted({r['response']['model'] for r in responses})
        budgets = [(r['request'].get('extra_body') or {}).get('thinking_token_budget') for r in requests]
        result.update(request_count=len(requests), response_count=len(responses),
                      actual_requested_models=requested, actual_response_models=served,
                      actual_thinking_budgets=sorted(set(budgets), key=str),
                      thinking_budget_verified=bool(budgets and all(value == budget for value in budgets)))
        result['verified'] = bool(requests and responses and requested == ['openai/' + expected]
                                  and served == [expected] and result['thinking_budget_verified'])
    except Exception:
        result['error'] = traceback.format_exc()
    return result


def valid_observation(row):
    return bool(row and row.get('worker_exit_code') == 0 and row.get('judge_valid') is True
                and type(row.get('resolved')) is bool and not row.get('error')
                and row.get('route_evidence', {}).get('verified'))


def paired_summary(rows, tasks):
    lookup = {(r['task_id'], r['variant']): r for r in rows}
    if len(lookup) != len(rows):
        raise RuntimeError('Duplicate task/arm results cannot be silently replaced')
    pairs = []
    for task in tasks:
        iid = task['instance_id']
        base, adapted = [lookup.get((iid, arm)) for arm in ('base', 'adapter')]
        if not valid_observation(base) or not valid_observation(adapted):
            category = 'incomplete_or_invalid'
        elif adapted['resolved'] and not base['resolved']:
            category = 'win'
        elif base['resolved'] and not adapted['resolved']:
            category = 'regression'
        else:
            category = 'unchanged_success' if base['resolved'] else 'unchanged_failure'
        pairs.append({'task_id': iid, 'repo': task.get('repo'), 'category': category,
                      'base_resolved': base.get('resolved') if base else None,
                      'adapter_resolved': adapted.get('resolved') if adapted else None,
                      'base_p2p_pass': base.get('p2p_pass') if base else None,
                      'adapter_p2p_pass': adapted.get('p2p_pass') if adapted else None,
                      'base_valid': valid_observation(base), 'adapter_valid': valid_observation(adapted),
                      'base_error': base.get('error') if base else 'attempt missing',
                      'adapter_error': adapted.get('error') if adapted else 'attempt missing'})
    labels = ('win', 'regression', 'unchanged_success', 'unchanged_failure', 'incomplete_or_invalid')
    counts = {label: sum(p['category'] == label for p in pairs) for label in labels}
    valid = len(tasks) - counts['incomplete_or_invalid']
    wins, losses = counts['win'], counts['regression']
    return {'phase': SETTINGS['phase'], 'workers': SETTINGS['workers'], 'split': 'dev',
            'independent_issues': len(tasks), 'planned_attempts': 2 * len(tasks),
            'returned_attempts': len(rows), 'N_valid': valid, 'W': wins, 'L': losses,
            'net_new_resolved': wins - losses, 'paired_repair_rate_difference': (wins - losses) / valid if valid else None,
            'paired_outcomes': pairs, 'paired_counts': counts,
            'by_arm': {arm: {
                'attempts': sum(r['variant'] == arm for r in rows), 'planned_attempts': len(tasks),
                'missing_attempts': len(tasks) - sum(r['variant'] == arm for r in rows),
                'invalid_attempts': sum(r['variant'] == arm and not valid_observation(r) for r in rows),
                'valid_attempts': sum(r['variant'] == arm and valid_observation(r) for r in rows),
                'resolved': sum(r['variant'] == arm and r.get('resolved') is True and valid_observation(r) for r in rows),
                'raw_reported_resolved': sum(r['variant'] == arm and r.get('resolved') is True for r in rows),
                'p2p_not_observed': sum(r['variant'] == arm and r.get('p2p_pass') is None for r in rows),
            } for arm in ('base', 'adapter')},
            'historical5of12_used_as_control': False, 'capacity_adapter_mounted': False,
            'scope': 'This phase/shard/concurrency only; absent P2P is not a regression pass.'}


def attempt_budget(criterion):
    judge_seconds = float(criterion.get('timeout_seconds', criterion.get('timeout', 120)))
    if judge_seconds <= 0:
        raise ValueError('Invalid frozen judge timeout')
    judgments = 1 + bool(criterion.get('p2p_nodes') or criterion.get('p2p_command'))
    return max(SETTINGS['attempt_seconds'], 60 * float(SETTINGS['budgets']['max_time_minutes'])
               + judge_seconds * judgments + SETTINGS['attempt_setup_reserve_seconds'])


def main():
    started = time.monotonic()
    deadline = started + SETTINGS['wall_seconds']
    run_id = 'weight-ab-' + SETTINGS['phase'].lower() + '-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    out = Path('/kaggle/working/results/017') / run_id
    out.mkdir(parents=True)
    agents = Path('/tmp') / ('p017-agents-' + run_id)
    selected = [{'instance_id': iid} for iid in SETTINGS['task_ids']]
    server = None
    active, rows, admissions = [], [], []
    failure = None
    manifest = {'settings': SETTINGS}
    save(out / 'experiment-manifest.json', manifest)

    def collect(record, forced=None):
        entry, process, log, begin = record
        log.close()
        attempt = Path(entry['results_dir'])
        try:
            result = json.loads((attempt / 'result.json').read_text())
        except Exception as exc:
            result = {'resolved': None, 'error': 'worker output missing/unreadable: ' + str(exc)}
        expected = SETTINGS['adapter']['name'] if entry['variant'] == 'adapter' else str(model)
        routing = route_evidence(attempt, expected)
        save(attempt / 'route-evidence.json', routing)
        row = {**result, **entry, 'worker_exit_code': process.returncode,
               'outer_seconds': time.monotonic() - begin, 'route_evidence': routing}
        if (attempt / 'context_metrics.json').is_file():
            row['context_metrics'] = json.loads((attempt / 'context_metrics.json').read_text())
        if forced:
            row['error'] = forced
        rows.append(row)
        save(out / 'analysis-rows.json', rows)
        print('RESULT ' + json.dumps(row, ensure_ascii=False), flush=True)

    try:
        for arm, files in BUNDLE.items():
            for relative, text in files.items():
                save(agents / arm / relative, text)
        panel = Path(MATERIAL_ROOT)
        document = json.loads((panel / 'agent-tasks.json').read_text())
        tasks = document if isinstance(document, list) else document['tasks']
        tasks = {task.get('instance_id', task.get('iid')): task for task in tasks}
        selected = [{**tasks[iid], 'instance_id': iid,
                     'problem_statement': tasks[iid].get('problem_statement', tasks[iid].get('issue'))}
                    for iid in SETTINGS['task_ids']]
        criteria = json.loads((panel / 'host-criteria.json').read_text())
        criteria = criteria if isinstance(criteria, list) else criteria['tasks']
        criteria = {item.get('instance_id', item.get('iid')): item for item in criteria}
        if len(selected) != SETTINGS['task_count'] or any(criteria[t['instance_id']]['split'] != 'dev' for t in selected):
            raise RuntimeError('Only the frozen independent dev selection may run')
        import torch
        from adk_submission import VllmConfig, VllmServer, discover_adapters
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
        hardware = [{'name': torch.cuda.get_device_name(i), 'bytes': torch.cuda.get_device_properties(i).total_memory}
                    for i in range(torch.cuda.device_count())]
        if len(hardware) != 4 or any('L4' not in item['name'] for item in hardware):
            raise RuntimeError('Requires four L4s')
        models = [p.parent for p in Path('/kaggle/input').rglob('tokenizer.json') if SETTINGS['model_path_pattern'] in str(p)]
        if len(models) != 1:
            raise RuntimeError('Official31B CTv2 mount missing/ambiguous')
        model = models[0]
        adapters, stage = stage_and_prove(agents, discover_adapters, ALLOWED_ADAPTER_EXTENSIONS)
        config = VllmConfig(model=str(model), port=8000, host='127.0.0.1', max_model_len=32768,
            dtype='bfloat16', gpu_memory_utilization=0.90, tensor_parallel_size=4,
            tool_call_parser='gemma4', reasoning_parser='gemma4', enable_auto_tool_choice=True,
            enable_lora=True, max_lora_rank=8, max_loras=1, startup_timeout=1200,
            extra_args=['--max-num-seqs', str(SETTINGS['server_max_num_seqs'])])
        server = VllmServer(config, adapter_manifest=adapters)
        command = server.build_cmd()
        for flag, value in (('--max-loras', '1'), ('--max-lora-rank', '8'), ('--tensor-parallel-size', '4'),
                            ('--max-num-seqs', str(SETTINGS['server_max_num_seqs']))):
            if command[command.index(flag) + 1] != value:
                raise RuntimeError('Server command differs: ' + flag)
        if '--enable-lora' not in command:
            raise RuntimeError('Actual server command omitted LoRA')
        manifest.update(adapter_stage=stage, server_command=command, hardware=hardware,
            serving_conditions={'enable_lora': True, 'mounted_adapters': [SETTINGS['adapter']['name']],
                'actual_base_model_path': str(model), 'actual_adapter_path': stage['actual_served_path'],
                'same_server_for_both_arms': True, 'server_max_num_seqs': SETTINGS['server_max_num_seqs'],
                'active_worker_limit': SETTINGS['workers']},
            versions={name: importlib.metadata.version(name) for name in ('swegemma', 'adk-submission', 'vllm', 'torch')},
            agent_files={arm: {rel: hashlib.sha256(text.encode()).hexdigest() for rel, text in files.items()}
                         for arm, files in BUNDLE.items()})
        save(out / 'experiment-manifest.json', manifest)
        worker = out / 'attempt_worker.py'
        save(worker, WORKER_SOURCE)
        pairs, schedule = [], []
        for slot, task in enumerate(selected):
            pair = []
            # Shards preserve order in the full frozen panel, including AB/BA parity.
            global_slot = SETTINGS['task_start'] + slot
            for arm in (('base', 'adapter') if global_slot % 2 == 0 else ('adapter', 'base')):
                ordinal = len(schedule) + 1
                iid = task['instance_id']
                entry = {'ordinal': ordinal, 'task_slot': global_slot + 1, 'task_id': iid,
                         'variant': arm, 'split': 'dev', 'phase': SETTINGS['phase'],
                         'attempt_id': f'{ordinal:03d}-{iid}-{arm}',
                         'attempt_seconds': attempt_budget(criteria[iid])}
                entry['results_dir'] = str(out / 'attempts' / entry['attempt_id'])
                pair.append((entry, task))
                schedule.append(entry)
            pairs.append(pair)
        save(out / 'schedule.json', schedule)
        server.start()
        manifest['startup_seconds'] = time.monotonic() - started
        base_url = str(server.base_url).rstrip('/')
        models_url = base_url + ('/models' if base_url.endswith('/v1') else '/v1/models')
        with urllib.request.urlopen(models_url, timeout=30) as response:
            registered = json.load(response)
        save(out / 'registered-models.json', registered)
        registered_ids = {item['id'] for item in registered['data']}
        if not {str(model), SETTINGS['adapter']['name']}.issubset(registered_ids):
            raise RuntimeError('Expected base and Step-68 aliases are not registered')
        manifest['registered_model_ids'] = sorted(registered_ids)
        save(out / 'experiment-manifest.json', manifest)
        print('WEIGHT_AB_SERVER ' + json.dumps(manifest['serving_conditions']), flush=True)

        pairs_per_wave = max(1, SETTINGS['workers'] // 2)
        for offset in range(0, len(pairs), pairs_per_wave):
            wave = [item for pair in pairs[offset:offset + pairs_per_wave] for item in pair]
            # At four workers, both arms of up to two complete pairs run together.
            # At one worker, reserve the sum before launching the first arm.
            required = (sum(entry['attempt_seconds'] for entry, _ in wave) if SETTINGS['workers'] == 1
                        else max(entry['attempt_seconds'] for entry, _ in wave)) + SETTINGS['cleanup_seconds']
            remaining = deadline - time.monotonic()
            admission = {'wave': len(admissions) + 1, 'task_ids': list(dict.fromkeys(entry['task_id'] for entry, _ in wave)),
                         'required_remaining_seconds': required, 'actual_remaining_seconds': remaining,
                         'admitted': remaining >= required}
            admissions.append(admission)
            save(out / 'pair-admissions.json', admissions)
            if not admission['admitted']:
                break
            queue = iter(wave)
            exhausted = False
            while active or not exhausted:
                while not exhausted and len(active) < SETTINGS['workers']:
                    try:
                        entry, task = next(queue)
                    except StopIteration:
                        exhausted = True
                        break
                    attempt = Path(entry['results_dir'])
                    attempt.mkdir(parents=True)
                    entry['started_wall_seconds'] = time.monotonic() - started
                    entry['workers_active_at_start'] = len(active) + 1
                    entry['wave'] = admission['wave']
                    specification = {'entry': entry, 'agent_dir': str(agents / entry['variant']), 'task': task,
                        'snapshot': str(panel / 'snapshots' / (task['instance_id'] + '.tar.gz')),
                        'base_url': str(server.base_url), 'served_model': str(model), 'model_alias': SETTINGS['model_alias'],
                        'budgets': SETTINGS['budgets'], 'judge': criteria[task['instance_id']],
                        'panel_root': str(panel), 'total_tasks': len(schedule)}
                    save(attempt / 'spec.json', specification)
                    log = (attempt / 'worker.log').open('w')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
                    try:
                        process = subprocess.Popen([sys.executable, '-u', str(worker), str(attempt / 'spec.json')],
                            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        log.close()
                        raise
                    active.append((entry, process, log, time.monotonic()))
                for record in active[:]:
                    entry, process, log, begin = record
                    forced = None
                    if process.poll() is None and time.monotonic() - begin >= entry['attempt_seconds']:
                        forced = 'worker_wall_timeout'
                    elif process.poll() is None and time.monotonic() >= deadline - 60:
                        forced = 'experiment_wall_timeout'
                    if forced:
                        stop(process)
                    if process.poll() is not None:
                        collect(record, forced)
                        active.remove(record)
                if active:
                    time.sleep(1)
        save(out / 'schedule.json', schedule)
    except BaseException:
        failure = traceback.format_exc()
        save(out / 'failure.txt', failure)
        raise
    finally:
        for record in active:
            stop(record[1])
            collect(record, 'interrupted')
        if server is not None:
            log = getattr(server, 'log_path', None)
            if log and Path(log).exists():
                shutil.copyfile(log, out / 'vllm-server.log')
            try:
                server.stop(timeout=10)
            except Exception:
                save(out / 'server-stop-error.txt', traceback.format_exc())
        summary = paired_summary(rows, selected)
        summary.update(wall_seconds=time.monotonic() - started, experiment_failed=bool(failure),
                       unstarted_task_ids=[task['instance_id'] for task in selected
                                           if not any(r['task_id'] == task['instance_id'] for r in rows)],
                       admission_stop=admissions[-1] if admissions and not admissions[-1]['admitted'] else None)
        save(out / 'summary.json', summary)
        print('FINAL ' + json.dumps(summary), flush=True)
        with zipfile.ZipFile('/kaggle/working/p017-weight-ab-output.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in out.rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(out.parent))


if __name__ == '__main__':
    main()
