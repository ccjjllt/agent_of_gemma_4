"""Persistent two-replica VFARL research over a frozen train task inventory.

No GPU is allocated on import. Two model processes live for this job: both
collect, then one trains while the other waits. Every batch keeps one frozen
behavior policy. A committed adapter is synchronized before the next batch.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .run import TaskQueue, write


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(part)
    return value.hexdigest()


def emit(phase, **values):
    print(json.dumps(dict(phase=phase, **values), ensure_ascii=False), flush=True)


def atomic(path, value):
    temporary = path.with_suffix('.part')
    write(temporary, value)
    temporary.replace(path)


def mixed_course_order(entries):
    """Interleave two common tasks with one specialist using metadata only."""
    by_adapter = defaultdict(lambda: defaultdict(deque))
    for row in entries:
        by_adapter[row['primary_adapter']][row['canonical_repo']].append(row['instance_id'])

    def repository_round_robin(repositories):
        result = deque()
        queues = list(repositories.values())
        while any(queues):
            for queue in queues:
                if queue:
                    result.append(queue.popleft())
        return result

    common = repository_round_robin(by_adapter.pop('G', {}))
    specialists = [repository_round_robin(by_adapter[key]) for key in sorted(by_adapter)]
    specialist = deque()
    while any(specialists):
        for queue in specialists:
            if queue:
                specialist.append(queue.popleft())
    ordered = []
    while common or specialist:
        for _ in range(2):
            if common:
                ordered.append(common.popleft())
        if specialist:
            ordered.append(specialist.popleft())
    return ordered


def run(*, config, selection, questions, assets, prevalidation, tools, out,
        initial_adapter=None, optimizer_state=None, stop_after_seconds=0, batch_task_limit=4,
        mixed_groups=2, maximum_states=4, turns=32, tool_calls=32,
        turn_tokens=768, episode_tokens=8192, thinking_budget=256, isolation='chroot',
        worker_env=None, active_tasks=1, batch_size=1, batch_wait_ms=50,
        batch_max_tokens=32768, session_cache_tokens=0, excluded_task_ids=None, task_order='frozen',
        collection_protocol='historical-three-branch'):
    from training.train_vfarl import prepare_rows
    if collection_protocol not in {'historical-three-branch', 'live-original-pair'}:
        raise ValueError('Unknown resident focal collection protocol')
    requested_schedule = dict(batch_task_limit=batch_task_limit, mixed_groups=mixed_groups,
        maximum_states=maximum_states, active_tasks_per_replica=active_tasks)
    live_original_pair = collection_protocol == 'live-original-pair'
    if live_original_pair:
        # This opt-in protocol is a fixed small-cohort experiment. Both models
        # retain one behavior policy until at most two tasks finish. A complete
        # mixed pair is sufficient; never replenish a cohort with the remaining
        # inventory while other episodes are still running.
        batch_task_limit, mixed_groups, maximum_states, active_tasks = 2, 1, 2, 1
    branches = 2 if live_original_pair else 3
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    selected = json.loads(Path(selection).read_text())['tasks']
    if not selected or len({row['instance_id'] for row in selected}) != len(selected):
        raise ValueError('Resident task inventory must be nonempty and unique')
    if any(row['physical_split'] != 'train' for row in selected):
        raise ValueError('Resident training permits train inventory only')
    usable = {row['instance_id'] for row in (json.loads(line) for line in
        Path(prevalidation).read_text().splitlines() if line.strip()) if row.get('usable')}
    excluded = set(excluded_task_ids or [])
    if not excluded <= {row['instance_id'] for row in selected}:
        raise ValueError('Resume exclusions must belong to the original frozen task inventory')
    eligible = [row for row in selected if row['instance_id'] in usable and row['instance_id'] not in excluded]
    task_ids = (mixed_course_order(eligible) if task_order == 'mixed-course'
                else [row['instance_id'] for row in eligible])
    if task_order not in {'frozen', 'mixed-course'}:
        raise ValueError('Unknown metadata-only task ordering')
    cfg = json.loads(Path(config).read_text())
    context = cfg['model']['max_sequence_length']
    setup = dict(config=str(Path(config).resolve()), selection=str(Path(selection).resolve()),
        questions=str(Path(questions).resolve()), assets=str(Path(assets).resolve()),
        prevalidation=str(Path(prevalidation).resolve()), tools=str(Path(tools).resolve()), out=str(out.resolve()),
        initial_adapter=str(Path(initial_adapter).resolve()) if initial_adapter else None, context=context,
        optimizer_state=str(Path(optimizer_state).resolve()) if optimizer_state else None,
        turns=turns, tool_calls=tool_calls, turn_tokens=turn_tokens,
        episode_tokens=episode_tokens, thinking_budget=thinking_budget, isolation=isolation,
        active_tasks=active_tasks, batch_size=batch_size, batch_wait_ms=batch_wait_ms,
        batch_max_tokens=batch_max_tokens, session_cache_tokens=session_cache_tokens,
        collection_protocol=collection_protocol, branches=branches)
    bootstrap = out / 'resident-bootstrap.json'
    write(bootstrap, setup)
    started = time.monotonic()
    summary = dict(format='vfarl-resident-rl-v2' if batch_size > 1 else 'vfarl-resident-rl-v1', context=context,
        frozen_task_ids=[row['instance_id'] for row in selected],
        unusable_task_ids=[row['instance_id'] for row in selected if row['instance_id'] not in usable],
        claimed_task_ids=[], completed_task_ids=[], batches=[], optimizer_steps=0,
        initial_adapter=setup['initial_adapter'], model_loads=2,
        excluded_prior_claimed_task_ids=sorted(excluded), declared_task_order=task_ids,
        active_tasks_per_replica=active_tasks, maximum_generation_batch_size=batch_size,
        maximum_retained_session_tokens=session_cache_tokens,
        sampling_window='All remaining tasks until mixed-group target' if not batch_task_limit else batch_task_limit,
        budget_seconds=stop_after_seconds, stopping='Stop new claims; retain in-flight work and committed updates')
    summary.update(collection_protocol=collection_protocol,
        requested_sampling_schedule=requested_schedule,
        actual_sampling_schedule=dict(batch_task_limit=batch_task_limit,
            mixed_groups=mixed_groups, maximum_states=maximum_states,
            active_tasks_per_replica=active_tasks, focal_members_per_state=branches),
        update_boundary='Complete the fixed cohort and close both inference services before updating and synchronizing adapters'
            if live_original_pair else 'Complete the declared collection window before updating and synchronizing adapters')
    workers, sequences, positions, partial = [], [0, 0], [0, 0], ['', '']
    last_observation = 0.0

    def persist():
        summary['wall_seconds'] = time.monotonic() - started
        atomic(out / 'resident-summary.json', summary)

    def reconcile_commits():
        # Queue writes survive a collector failure before its reply. Keep the
        # actual coverage in the summary even when the current batch aborted.
        claimed, completed = [], []
        for queue in sorted(out.glob('batches/batch-*/task-queue.json')):
            state = json.loads(queue.read_text())
            claimed.extend(row['instance_id'] for row in state.get('claimed', []))
            completed.extend(row['instance_id'] for row in state.get('completed', []))
        summary['claimed_task_ids'] = list(dict.fromkeys(claimed))
        summary['completed_task_ids'] = list(dict.fromkeys(completed))
        records = []
        for path in sorted(out.glob('batches/batch-*/update/optimizer-step.json')):
            value = json.loads(path.read_text())
            if value.get('optimizer_steps') == 1:
                records.append(dict(path=str(path), source_data_sha256=value.get('source_data_sha256'),
                    adapter_export_completed=value.get('adapter_export_completed', False)))
        summary['committed_updates'] = records
        summary['optimizer_steps'] = len(records)

    def pump():
        nonlocal last_observation
        for index, (process, log, ipc) in enumerate(workers):
            with Path(log.name).open() as stream:
                stream.seek(positions[index])
                text = stream.read()
                positions[index] = stream.tell()
            partial[index] += text
            lines = partial[index].split('\n')
            partial[index] = lines.pop()
            for line in lines:
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(event, dict) and event.get('phase'):
                    print(json.dumps(dict(event, worker=index), ensure_ascii=False), flush=True)
        if time.monotonic() - last_observation >= 60:
            last_observation = time.monotonic()
            observation = dict(phase='gpu_observation', time=time.time(), workers=[])
            for process, log, _ in workers:
                worker = dict(pid=process.pid, returncode=process.poll())
                try:
                    fields = dict(line.split(':', 1) for line in
                        Path(f'/proc/{process.pid}/status').read_text().splitlines() if ':' in line)
                    worker['host_process'] = {key: fields[key].strip() for key in
                        ('State', 'VmRSS', 'VmSwap', 'Threads') if key in fields}
                    with Path(log.name).open('rb') as stream:
                        stream.seek(max(0, Path(log.name).stat().st_size - 1000))
                        worker['raw_log_tail'] = stream.read().decode(errors='replace')
                except FileNotFoundError:
                    pass
                observation['workers'].append(worker)
            try:
                gpu = subprocess.run(['nvidia-smi',
                    '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=10)
                observation.update(gpu_rows=gpu.stdout.splitlines(), error=gpu.stderr.strip(),
                                   returncode=gpu.returncode)
            except Exception as error:
                observation['error'] = repr(error)
            print(json.dumps(observation), flush=True)
            with (out / 'gpu-observations.jsonl').open('a') as stream:
                stream.write(json.dumps(observation) + '\n')

    def wait(paths):
        while not all(path.exists() for path in paths):
            pump()
            for index, (process, log, _) in enumerate(workers):
                if process.poll() is not None:
                    emit('resident_worker_exited', worker=index, exit_code=process.returncode,
                         raw_log_tail=Path(log.name).read_text(errors='replace')[-10000:])
            if any(process.poll() is not None for process, _, _ in workers):
                raise RuntimeError('A resident model exited; preserve all prior batch evidence')
            time.sleep(1)
        pump()
        values = [json.loads(path.read_text()) for path in paths]
        for value in values:
            if value.get('ok') is False:
                raise RuntimeError(value['error'])
        return values

    def command(index, operation, **values):
        sequences[index] += 1
        ipc = workers[index][2]
        atomic(ipc / f'{sequences[index]:05d}.json', dict(operation=operation, **values))
        return ipc / f'{sequences[index]:05d}.result.json'

    pending, pending_mixed, batch_number = [], 0, 0
    try:
        for index in range(2):
            ipc = out / f'ipc-{index}'
            ipc.mkdir(exist_ok=False)
            log = (out / f'resident-worker-{index}.log').open('w')
            process = subprocess.Popen([sys.executable, '-u', '-m',
                'competition.vfarl.l4x4.resident_worker', '--bootstrap', str(bootstrap),
                '--worker-index', str(index), '--ipc', str(ipc)],
                env=dict(worker_env or os.environ, CUDA_VISIBLE_DEVICES='0,1' if index == 0 else '2,3'),
                cwd=str(Path(__file__).resolve().parents[3]),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            workers.append((process, log, ipc))
        ready = wait([item[2] / 'ready.json' for item in workers])
        if ready[0]['policy'] != ready[1]['policy']:
            raise RuntimeError('Resident replicas initialized different policies')
        emit('resident_ready', context=context, task_pool=len(task_ids),
             policy_id=ready[0]['policy']['policy_id'])
        if live_original_pair:
            emit('resident_collection_protocol', collection_protocol=collection_protocol,
                 requested_schedule=requested_schedule,
                 actual_schedule=summary['actual_sampling_schedule'],
                 update_boundary=summary['update_boundary'])
        while len(summary['claimed_task_ids']) < len(task_ids):
            elapsed = time.monotonic() - started
            if stop_after_seconds and elapsed >= stop_after_seconds:
                summary['stop_reason'] = 'new-claim-time-budget-reached'
                break
            remaining = [task for task in task_ids if task not in set(summary['claimed_task_ids'])]
            batch_number += 1
            batch = out / 'batches' / f'batch-{batch_number:03d}'
            batch.mkdir(parents=True)
            queue = batch / 'task-queue.json'
            seconds = max(0.001, stop_after_seconds - elapsed) if stop_after_seconds else 0
            target = max(1, mixed_groups - pending_mixed)
            ids = remaining[:batch_task_limit] if batch_task_limit else remaining
            execution_contract = dict(config_sha256=sha(config), policy_backend='unsloth',
                context=context, draws=1, branches=branches, turns=turns, tool_calls=tool_calls,
                turn_tokens=turn_tokens, episode_tokens=episode_tokens)
            if live_original_pair:
                execution_contract['collection_protocol'] = collection_protocol
            TaskQueue(queue, ids, selection_sha256=sha(selection), maximum_states=maximum_states,
                stop_after_mixed_groups=target, stop_after_tasks=len(ids), stop_after_seconds=seconds,
                execution_contract=execution_contract)
            emit('resident_batch_started', batch=batch_number, task_ids=ids,
                 needed_mixed_groups=target, context=context)
            replies = wait([command(index, 'collect', out=str(batch / f'worker-{index}'),
                task_ids=ids, task_queue=str(queue), maximum_states=maximum_states,
                mixed_groups=target, task_limit=len(ids), stop_after_seconds=seconds)
                for index in range(2)])
            if replies[0]['policy'] != replies[1]['policy']:
                raise RuntimeError('Resident replicas collected with different policy identities')
            state = json.loads(queue.read_text())
            claimed = [row['instance_id'] for row in state['claimed']]
            completed = [row['instance_id'] for row in state['completed']]
            if not claimed:
                summary['stop_reason'] = state.get('stop_reason', 'no-new-task-claim')
                break
            summary['claimed_task_ids'].extend(claimed)
            summary['completed_task_ids'].extend(completed)
            paths = [batch / f'worker-{index}' / 'focal.jsonl' for index in range(2)]
            focal = batch / 'focal.jsonl'
            focal.write_bytes(b''.join(path.read_bytes() for path in paths if path.exists()))
            if focal.stat().st_size:
                pending.append(focal)
            data = batch / 'update-input.jsonl'
            data.write_bytes(b''.join(path.read_bytes() for path in pending))
            rows, groups = prepare_rows(data, context)
            pending_mixed = groups['mixed_group_count']
            record = dict(batch=batch_number, claimed_task_ids=claimed,
                completed_task_ids=completed, focal_sha256=sha(focal),
                pending_behavior_sha256=sha(data), pending_mixed_groups=pending_mixed,
                sampled_update_actions=len(rows), optimizer_steps=0)
            summary['batches'].append(record)
            persist()
            emit('resident_batch_collected', **record)
            if pending_mixed < mixed_groups:
                continue
            update = batch / 'update'
            result = wait([command(1, 'update', data=str(data), out=str(update))])[0]
            step = json.loads((update / 'optimizer-step.json').read_text())
            if step.get('optimizer_steps') != 1 or step.get('source_data_sha256') != sha(data):
                raise RuntimeError('Resident update has no matching committed optimizer record')
            sync = wait([command(0, 'sync', adapter_run=str(update))])[0]
            if sync['policy'] != result['policy']:
                raise RuntimeError('Resident replicas did not synchronize the committed adapter')
            summary['optimizer_steps'] += 1
            summary['latest_adapter'] = str(update)
            summary['latest_optimizer_state'] = result['optimizer_state']
            record.update(optimizer_steps=1, update=str(update),
                optimizer_update_index=summary['optimizer_steps'],
                parameter_delta_l2=step.get('global_parameter_delta_l2'),
                grad_norm=step.get('grad_norm'), synchronized_policy_id=sync['policy']['policy_id'],
                diagnostic_error=result.get('diagnostic_error'))
            pending, pending_mixed = [], 0
            # Keep every step/metric/SHA record; bound adapter output storage.
            updated_batches = [item for item in summary['batches'] if item.get('optimizer_steps')]
            retained = {item['batch'] for item in updated_batches[-2:]}
            retained.update(item['batch'] for item in updated_batches if item['optimizer_update_index'] % 10 == 0)
            for old in updated_batches:
                if old['batch'] not in retained:
                    adapter = Path(old['update']) / 'adapter'
                    if adapter.exists():
                        shutil.rmtree(adapter)
                        old['adapter_retention'] = 'Removed after verified replica synchronization; latest2 and every10th retained'
            persist()
            emit('resident_update_completed', batch=batch_number,
                 optimizer_steps_total=summary['optimizer_steps'],
                 completed_tasks=len(summary['completed_task_ids']),
                 policy_id=sync['policy']['policy_id'])
        summary.setdefault('stop_reason', 'frozen-task-pool-exhausted')
        summary['untrained_tail_files'] = [str(path) for path in pending]
        summary['untrained_tail_mixed_groups'] = pending_mixed
        persist()
        emit('resident_complete', optimizer_steps=summary['optimizer_steps'],
             claimed_tasks=len(summary['claimed_task_ids']), completed_tasks=len(summary['completed_task_ids']),
             untrained_tail_mixed_groups=pending_mixed, stop_reason=summary['stop_reason'])
        return summary
    except BaseException:
        reconcile_commits()
        persist()
        raise
    finally:
        for index, (process, _, _) in enumerate(workers):
            if process.poll() is None:
                command(index, 'close')
        deadline = time.monotonic() + 20
        for process, log, _ in workers:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.close()
        reconcile_commits()
        persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'selection', 'questions', 'assets', 'prevalidation', 'tools', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--initial-adapter')
    parser.add_argument('--optimizer-state')
    parser.add_argument('--stop-after-seconds', type=float, default=0)
    parser.add_argument('--batch-task-limit', type=int, default=4)
    parser.add_argument('--mixed-groups', type=int, default=2)
    parser.add_argument('--maximum-states', type=int, default=4)
    parser.add_argument('--collection-protocol',
        choices=('historical-three-branch', 'live-original-pair'), default='historical-three-branch',
        help='live-original-pair uses a fixed two-task cohort, one session per replica and one mixed pair per update')
    args = parser.parse_args()
    run(**vars(args))


if __name__ == '__main__':
    main()
