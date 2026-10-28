"""One resident Unsloth replica serving real collection and update commands."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

from .run import collect, digest, write


def save(path, value):
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', type=Path, required=True)
    parser.add_argument('--worker-index', type=int, choices=(0, 1), required=True)
    parser.add_argument('--ipc', type=Path, required=True)
    parser.add_argument('--session-prefix-rewind', action='store_true',
        help='Opt in to same pre-action prefix cache rewind; requires the matching resident bootstrap and a positive session-cache budget')
    args = parser.parse_args()
    setup = json.loads(args.bootstrap.read_text())
    if args.session_prefix_rewind != bool(setup.get('session_prefix_rewind', False)):
        parser.error('--session-prefix-rewind must match the explicit resident bootstrap setting')
    if args.session_prefix_rewind and setup.get('session_cache_tokens', 0) <= 0:
        parser.error('--session-prefix-rewind requires a positive session-cache budget')
    history_format = setup.get('history_format', 'raw-preserved')
    if history_format not in {'raw-preserved', 'canonical-ct-v1'} or (
            history_format != 'raw-preserved'
            and setup.get('collection_protocol') != 'live-original-pair'):
        parser.error('Canonical history requires live-original-pair and an explicit supported history format')
    if history_format != 'raw-preserved' and setup.get('history_source_sha256') != digest(
            Path(__file__).with_name('history.py').read_bytes()):
        raise RuntimeError('Canonical history source SHA differs from the explicit resident bootstrap')
    source_root = Path(__file__).resolve().parents[3]
    from .state import configure_isolation
    isolation = configure_isolation(setup['isolation'], hidden_paths=[
        Path(setup['assets']), Path(setup['out']), source_root])
    if isolation['effective_mode'] == 'observable':
        raise RuntimeError('Resident task isolation is unavailable')
    from training.vfarl_unsloth import UnslothPolicy
    print(json.dumps(dict(phase='resident_model_loading', worker=args.worker_index,
        context=setup['context'], adapter_run=setup.get('initial_adapter'))), flush=True)
    policy = UnslothPolicy(setup['config'], training=True, inference_device_count=2,
                           adapter_run=setup.get('initial_adapter'))
    policy.switch_execution_mode('inference')
    from training.vfarl_batched import configure_batched_policy
    configure_batched_policy(policy, max_batch_size=setup.get('batch_size', 2),
        max_batch_tokens=setup.get('batch_max_tokens', 32768),
        session_cache_tokens=setup.get('session_cache_tokens', 0),
        session_prefix_rewind=bool(args.session_prefix_rewind))
    selection = json.loads(Path(setup['selection']).read_text())
    entries = selection['tasks']
    if any(row['physical_split'] != 'train' for row in entries):
        raise ValueError('Resident collection permits train tasks only')
    questions = {}
    for line in Path(setup['questions']).read_text().splitlines():
        row = json.loads(line)
        if digest(row['official_initial_user_prompt'].encode()) != row['official_initial_user_prompt_sha256']:
            raise ValueError('Resident question source hash differs')
        questions[row['instance_id']] = row['official_initial_user_prompt']
    for row in entries:
        if digest(questions[row['instance_id']].encode()) != row['official_initial_user_prompt_sha256']:
            raise ValueError('Resident selected prompt differs')
    save(args.ipc / 'ready.json', dict(worker=args.worker_index, policy=policy.identity))
    print(json.dumps(dict(phase='resident_model_ready', worker=args.worker_index,
        policy_id=policy.identity['policy_id'])), flush=True)
    optimizer, sequence = None, 1
    try:
        while True:
            command_path = args.ipc / f'{sequence:05d}.json'
            if not command_path.exists():
                time.sleep(0.2)
                continue
            command = json.loads(command_path.read_text())
            response_path = args.ipc / f'{sequence:05d}.result.json'
            try:
                operation = command['operation']
                result = {}
                if operation == 'collect':
                    policy.switch_execution_mode('inference')
                    output = Path(command['out'])
                    output.mkdir(parents=True, exist_ok=False)
                    values = dict(selection=Path(setup['selection']), questions=Path(setup['questions']),
                        assets=Path(setup['assets']), out=output, config=Path(setup['config']),
                        tools=Path(setup['tools']), prevalidation=Path(setup['prevalidation']),
                        policy_backend='unsloth', policy_gpus=2, adapter_run=None,
                        score_focal_data=None, score_focal_before=False,
                        context=setup['context'], draws=1, branches=setup.get('branches', 3),
                        collection_protocol=setup.get('collection_protocol', 'historical-three-branch'),
                        complete_fixed_cohort=setup.get('complete_fixed_cohort', False),
                        focal_trigger=setup.get('focal_trigger', 'any-tool-failure'),
                        max_states=command['maximum_states'], worker_index=args.worker_index,
                        worker_count=2, task_queue=Path(command['task_queue']),
                        stop_after_mixed_groups=command['mixed_groups'],
                        stop_after_tasks=command['task_limit'],
                        stop_after_seconds=command['stop_after_seconds'],
                        turns=setup['turns'], tool_calls=setup['tool_calls'],
                        turn_tokens=setup['turn_tokens'], episode_tokens=setup['episode_tokens'],
                        thinking_budget=setup['thinking_budget'], isolation=setup['isolation'],
                        active_tasks=setup.get('active_tasks', 4),
                        resident_hidden_root=Path(setup['out']))
                    if history_format != 'raw-preserved':
                        values['history_format'] = history_format
                    task_ids = command['task_ids']
                    wanted = set(task_ids)
                    by_id = {row['instance_id']: row for row in entries}
                    if len(wanted) != len(task_ids) or not wanted <= by_id.keys():
                        raise ValueError('Resident batch contains an undeclared task')
                    # Queue task order is part of its frozen contract; a
                    # metadata-interleaved window must not fall back to the
                    # original selection-file order inside each worker.
                    subset = [by_id[task_id] for task_id in task_ids]
                    write(output / 'execution.json', dict(arguments={key: str(value)
                        if isinstance(value, Path) else value for key, value in values.items()}))
                    async def collect_sessions():
                        from training.vfarl_batched import BatchedPolicyService
                        service = BatchedPolicyService(policy,
                            max_batch_size=setup.get('batch_size', 2),
                            wait_ms=setup.get('batch_wait_ms', 10),
                            max_batch_tokens=setup.get('batch_max_tokens', 32768),
                            session_cache_tokens=setup.get('session_cache_tokens', 0),
                            session_prefix_rewind=bool(args.session_prefix_rewind))
                        try:
                            await collect(SimpleNamespace(**values), subset, questions, policy=service)
                        finally:
                            # The single model owner must finish before this
                            # worker changes adapters or enters training.
                            await service.aclose()
                    asyncio.run(collect_sessions())
                    result = dict(out=str(output), policy=policy.identity)
                elif operation == 'update':
                    from training.train_vfarl_unsloth import run
                    from training.common import sha256_file
                    from training.vfarl_optimizer import load_bound_optimizer, save_bound_optimizer
                    policy.switch_execution_mode('training')
                    optimizer_resume = None
                    if optimizer is None:
                        optimizer = policy.torch.optim.AdamW(
                            [parameter for name, parameter in policy.model.named_parameters()
                             if 'lora_' in name], lr=policy.cfg['learning_rate'], weight_decay=0.0)
                        if setup.get('optimizer_state'):
                            optimizer_resume = load_bound_optimizer(setup['optimizer_state'],
                                policy, optimizer, learning_rate=policy.cfg['learning_rate'])
                    # Loading Adam state restores param-group learning rates as
                    # well as moments. This run's declared rate takes priority;
                    # keep the existing step counters and moment tensors intact.
                    prior_learning_rates = [float(group['lr']) for group in optimizer.param_groups]
                    for group in optimizer.param_groups:
                        group['lr'] = policy.cfg['learning_rate']
                    optimizer_learning_rate = dict(declared=policy.cfg['learning_rate'],
                        before_assignment=prior_learning_rates,
                        actual=[float(group['lr']) for group in optimizer.param_groups],
                        policy='Current declared learning rate; Adam moments and step counters retained')
                    if optimizer_resume is not None:
                        optimizer_learning_rate['bound_resume'] = optimizer_resume
                    print(json.dumps(dict(phase='resident_optimizer_learning_rate',
                        **optimizer_learning_rate)), flush=True)
                    diagnostic_error = None
                    try:
                        run(SimpleNamespace(config=setup['config'], data=command['data'],
                            out=command['out'], actor_backend='unsloth', learner_backend='unsloth',
                            policy_gpus=2), policy=policy, optimizer=optimizer)
                    except Exception:
                        committed = Path(command['out']) / 'optimizer-step.json'
                        step = json.loads(committed.read_text()) if committed.exists() else {}
                        if (step.get('optimizer_steps') != 1 or not step.get('adapter_export_completed')
                                or not (Path(command['out']) / 'adapter/adapter_model.safetensors').is_file()):
                            raise
                        diagnostic_error = traceback.format_exc()
                        optimizer.zero_grad(set_to_none=True)
                        print(json.dumps(dict(phase='resident_committed_diagnostic_error',
                            out=command['out'], error=diagnostic_error)), flush=True)
                    latest = Path(setup['out']) / 'optimizer-latest.pt'
                    optimizer_binding = save_bound_optimizer(latest, policy, optimizer)
                    policy.switch_execution_mode('inference')
                    result = dict(out=command['out'], policy=policy.identity,
                        optimizer_state=str(latest), optimizer_state_sha256=sha256_file(latest),
                        optimizer_binding=optimizer_binding,
                        optimizer_learning_rate=optimizer_learning_rate,
                        diagnostic_error=diagnostic_error)
                elif operation == 'sync':
                    policy.load_adapter_in_place(command['adapter_run'])
                    result = dict(policy=policy.identity)
                elif operation != 'close':
                    raise ValueError('Unknown resident command')
                save(response_path, dict(ok=True, operation=operation, **result))
                if operation == 'close':
                    break
            except BaseException:
                error = traceback.format_exc()
                save(response_path, dict(ok=False, operation=command.get('operation'), error=error))
                print(json.dumps(dict(phase='resident_worker_error', worker=args.worker_index,
                                      error=error)), flush=True)
                raise
            sequence += 1
    finally:
        policy.close()


if __name__ == '__main__':
    main()
