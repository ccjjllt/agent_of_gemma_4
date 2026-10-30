"""Adaptive official-task collection and TRL clipped GRPO for an Unsloth LoRA.

Complete groups receive judge-derived composite rewards, low-variance groups
are filtered with a bounded refill budget, and accepted groups are reused only
within the current policy-version batch. Official assets are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import secrets
import subprocess
import sys
import threading
import time
import tomllib

from .endpoint import PolicyEndpoint
from .curriculum import TaskSampler, load_tasks
from .reward_ledger import RewardLedger
from .recovery import (CheckpointStore, CollectionInterrupted, RunBudget, WorkerAttemptError,
                       atomic_json, file_digest, run_lock, value_digest)

PROJECT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rpartition(")")[2].split()
        return int(fields[1]), fields[19]
    except (OSError, ValueError, IndexError):
        return None


def stop_process_tree(proc, grace=2, *, expected_start=None):
    """Also stop official command processes that created their own sessions."""
    snapshot = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        pid = int(path.parent.name)
        identity = process_identity(pid)
        if identity:
            snapshot[pid] = identity
    if expected_start is not None and snapshot.get(proc.pid, (None, None))[1] != expected_start:
        return
    selected = {proc.pid}
    while True:
        more = {pid for pid, (ppid, _) in snapshot.items() if ppid in selected}
        if more <= selected:
            break
        selected.update(more)

    def send(sig):
        for pid in selected:
            original = snapshot.get(pid)
            current = process_identity(pid)
            # Match process start time to avoid signaling a reused PID.
            if original is not None and current is not None and original[1] == current[1]:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass

    send(signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    send(signal.SIGKILL)
    if proc.poll() is None:
        proc.wait(timeout=5)


def write_json(path, value):
    atomic_json(path, value)


def emit(path, event, **fields):
    row = {"time_unix": time.time(), "event": event, **fields}
    text = json.dumps(row, ensure_ascii=False)
    print(text, flush=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(text + "\n")


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reward_batch_summary(trajectories):
    """Report selected reward evidence without treating unknown totals as zero."""
    exact = [item["reward"] for item in trajectories if item["reward"] is not None]
    exact_mean = sum(exact) / len(exact) if exact else None
    bounds_mean = ({
        key: sum(item["reward_bounds"][key] for item in trajectories) / len(trajectories)
        for key in ("lower", "upper")
    } if trajectories else None)
    return {
        "composite_reward_mean": exact_mean if len(exact) == len(trajectories) else None,
        "exact_composite_reward_mean": exact_mean, "exact_composite_reward_count": len(exact),
        "composite_reward_count": len(trajectories), "composite_reward_bounds_mean": bounds_mean,
        "composite_reward_population": "selected_for_update",
    }


def run_worker(command, folder, deadline, expired, processes, process_lock, *, attempt_cancelled=None):
    if expired.is_set() or time.monotonic() >= deadline:
        raise CollectionInterrupted("Run expired before launching this task worker")
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (folder / "worker.log").open("w", encoding="utf-8") as out:
        try:
            proc = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True, cwd=str(PROJECT))
        except OSError as exc:
            raise WorkerAttemptError(f"Worker launch failed: {exc}") from exc
        with process_lock:
            processes.add(proc)
        try:
            identity = process_identity(proc.pid)
            atomic_json(folder / "worker_process.json", {
                "pid": proc.pid, "start_identity": list(identity) if identity else None,
                "workspace": str(folder.resolve()), "command_sha256": value_digest(command),
            })
            # Cancellation can race Popen and registration in the shared set.
            if expired.is_set() or time.monotonic() >= deadline:
                raise CollectionInterrupted("Run expired while launching this task worker")
            while proc.poll() is None:
                if expired.is_set() or time.monotonic() >= deadline:
                    raise CollectionInterrupted("Run expired during this task worker")
                if attempt_cancelled is not None and attempt_cancelled.is_set():
                    raise WorkerAttemptError("This worker attempt was cancelled")
                try:
                    proc.wait(timeout=min(0.2, max(0.01, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            stop_process_tree(proc)
            raise
        finally:
            with process_lock:
                processes.discard(proc)
    result_path = folder / "result.json"
    if not result_path.is_file():
        raise WorkerAttemptError(f"Official worker exited {proc.returncode} without a verdict: {folder / 'worker.log'}")
    try:
        result = json.loads(result_path.read_text())
    except (ValueError, OSError) as exc:
        raise WorkerAttemptError(f"Worker result could not be read: {result_path}") from exc
    if not isinstance(result, dict):
        raise WorkerAttemptError(f"Worker result is not a verdict object: {result_path}")
    if proc.returncode or result.get("infrastructure_error"):
        raise WorkerAttemptError(f"Official worker failed; see {result_path}")
    if not result.get("verdict_valid") or not result.get("verification_complete"):
        raise WorkerAttemptError(f"Official verification is incomplete; see {result_path}")
    if result.get("reward") not in (0, 1):
        raise WorkerAttemptError("Official binary outcome is missing; refusing to invent a reward")
    return result


def collect_batch(**kwargs):
    from .collection import collect_batch as collect_recoverable_batch

    return collect_recoverable_batch(**kwargs, worker_fn=run_worker, emit_fn=emit)



def cleanup_orphan_workers(run_dir):
    """Stop only registered live attempts whose Linux process identity matches."""
    class ExistingProcess:
        def __init__(self, pid, started):
            self.pid, self.started = pid, started

        def poll(self):
            current = process_identity(self.pid)
            if current is None or current[1] != self.started:
                return 0
            try:
                status = (Path('/proc') / str(self.pid) / 'stat').read_text().rpartition(')')[2].split()[0]
                return 0 if status == 'Z' else None
            except OSError:
                return 0

        def wait(self, timeout):
            until = time.monotonic() + timeout
            while self.poll() is None:
                if time.monotonic() >= until:
                    raise subprocess.TimeoutExpired('registered worker', timeout)
                time.sleep(0.02)
            return 0

    stopped = []
    for path in run_dir.glob('batch-*-group-*-rollout-*/worker_process.json'):
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError('Worker process registry cannot be a symlink')
        record = json.loads(path.read_text())
        expected = record.get('start_identity')
        pid = record.get('pid')
        if (type(pid) is not int or pid <= 1 or pid == os.getpid()
                or record.get('workspace') != str(path.parent.resolve())):
            raise ValueError('Invalid worker process registry')
        current = process_identity(pid)
        if expected is None or current is None or current[1] != expected[1]:
            continue
        try:
            command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue  # The registered worker exited between the two reads.
        command = [item.decode() for item in command if item]
        process = ExistingProcess(pid, expected[1])
        if process.poll() is not None:
            continue
        if value_digest(command) != record['command_sha256']:
            raise ValueError('Registered live worker command identity changed; refusing to signal it')
        stop_process_tree(process, expected_start=expected[1])
        stopped.append(pid)
    return stopped


def run_learner(launch, run_dir):
    """The CPU supervisor owns the budget; its inherited lock protects both."""
    from .policy import Policy
    from .grpo_trainer import make_grpo_trainer, capture_rng_state, restore_rng_state
    from .session import TrainingSession
    import torch

    cfg = launch['recipe']
    metrics = run_dir / 'metrics.jsonl'
    started = time.monotonic()
    deadline = started + max(0.0, launch['deadline_unix'] - time.time())
    expired = threading.Event()
    processes, process_lock = set(), threading.Lock()
    policy = endpoint = session = None

    def log(event, **fields):
        emit(metrics, event, **fields)

    def cancel_run():
        expired.set()
        if policy is not None:
            policy.abort_generation.set()
        with process_lock:
            for proc in list(processes):
                stop_process_tree(proc, grace=0.5)

    timer = threading.Timer(max(0.0, deadline - time.monotonic()), cancel_run)
    timer.daemon = True
    timer.start()
    try:
        if expired.is_set():
            raise CollectionInterrupted('Run budget exhausted before model loading')
        sampler = TaskSampler(launch['training_tasks'], cfg.get('curriculum'), seed=cfg['policy'].get('seed', 3407))
        ledger = RewardLedger(run_dir, recover_incomplete_tail=launch['resume'])
        store = CheckpointStore(run_dir, torch, launch['source_binding'])
        # Reject malformed/mismatched state before allocating the model.
        if launch['resume']:
            store.load('checkpoint')
        log('loading_policy', declared_model_revision=cfg['policy'].get('model_revision'),
            eligible_tasks=len(launch['training_tasks']), resume=launch['resume'])
        policy = Policy(cfg['policy'])
        endpoint = PolicyEndpoint(policy, run_dir, int(cfg['runtime'].get('port', 0)),
                                  generation_config=cfg.get('generation', {}))
        session = TrainingSession(
            cfg=cfg, sampler=sampler, policy=policy, endpoint=endpoint, ledger=ledger, store=store,
            run_dir=run_dir, collect=collect_batch, emit=log, capture_rng=capture_rng_state,
            restore_rng=restore_rng_state, expired=expired, deadline=deadline,
            reward_summary=reward_batch_summary, torch=torch,
            collect_kwargs=dict(runtime_python=Path(cfg['runtime']['python']), data_root=Path(launch['data_root']),
                                submission=Path(launch['submission_dir']), uid_base=launch['agent_uid_base'],
                                deadline=deadline, expired=expired, processes=processes,
                                process_lock=process_lock, cancel_run=cancel_run))
        trainer_config = {**cfg['grpo'], 'learning_rate': float(cfg['run']['learning_rate']),
                          'max_grad_norm': float(cfg['run']['max_grad_norm']),
                          'total_optimizer_steps': int(cfg['run']['max_batches']) * int(cfg['grpo']['num_iterations']),
                          'num_generations': int(cfg['run']['num_generations']),
                          'seed': int(cfg['policy'].get('seed', 3407))}
        trainer = make_grpo_trainer(policy, trainer_config, run_dir / 'trainer', log,
                                    checkpoint_callback=session.checkpoint_training)
        endpoint.start()
        return session.run(trainer, resume=launch['resume'])
    except BaseException as error:
        cancel_run()
        log('run_failed', error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.monotonic() - started,
            checkpoint_id=session.checkpoint_id if session else None)
        raise
    finally:
        timer.cancel()
        if endpoint is not None:
            endpoint.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Defaults to the saved recipe on resume, otherwise configs/a800_2x80.toml')
    parser.add_argument('--describe', action='store_true', help='Print the recipe without loading data or CUDA')
    parser.add_argument('--resume', action='store_true', help='Resume the committed state in an existing --run-dir')
    parser.add_argument('--additional-walltime-seconds', type=float, default=0.0,
                        help='Explicit extra run allowance; resume otherwise uses only unspent time')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_lock-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--_invocation-id', help=argparse.SUPPRESS)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--submission-dir', type=Path, help='Defaults to the official sample_submission')
    parser.add_argument('--run-dir', type=Path)
    task = parser.add_mutually_exclusive_group()
    task.add_argument('--task-id')
    task.add_argument('--task-index', type=int, help='Index within eligible public tasks; default trains the catalog')
    args = parser.parse_args()

    def terminate(signum, frame):
        raise KeyboardInterrupt('Run termination requested')

    if args._worker:
        if args.run_dir is None or args._lock_fd is None or not args._invocation_id:
            parser.error('The internal learner requires its live CPU supervisor and inherited run lock')
        import fcntl
        run_dir = args.run_dir.resolve(strict=True)
        info, expected = os.fstat(args._lock_fd), (run_dir / '.run.lock').stat()
        if (info.st_dev, info.st_ino) != (expected.st_dev, expected.st_ino):
            parser.error('The inherited descriptor does not belong to this run')
        fcntl.flock(args._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch = json.loads((run_dir / 'recovery' / 'launch.json').read_text())
        if launch['invocation_id'] != args._invocation_id:
            parser.error('The run launch record changed')
        signal.signal(signal.SIGTERM, terminate)
        return run_learner(launch, run_dir)

    saved = None
    if args.resume:
        if args.run_dir is None:
            parser.error('--resume requires the existing --run-dir')
        saved = json.loads((args.run_dir.expanduser().resolve(strict=True) / 'run_config.json').read_text())
        if 'source_binding' not in saved or 'recipe' not in saved:
            parser.error('This older run has no recovery/source baseline; it cannot be resumed safely')
    cfg = (tomllib.loads(args.config.read_text()) if args.config else
           saved['recipe'] if saved else tomllib.loads((PROJECT / 'configs/a800_2x80.toml').read_text()))
    if args.describe:
        print(json.dumps({'recipe': cfg, 'algorithm': 'trl_clipped_grpo_composite_adaptive',
                          'data_access': 'official public tasks on server only',
                          'gpu_validation': 'not yet performed'}, ensure_ascii=False, indent=2))
        return 0
    if sys.platform != 'linux':
        parser.error('Run training on the Linux GPU server; local preparation uses --describe.')
    if not math.isfinite(args.additional_walltime_seconds) or args.additional_walltime_seconds < 0:
        parser.error('--additional-walltime-seconds must be finite and nonnegative')
    if args.additional_walltime_seconds and not args.resume:
        parser.error('Use run.walltime_seconds for a new run; additional allowance is for explicit resume')
    if saved:
        args.data_root = args.data_root or Path(saved['data_root'])
        args.model_path = args.model_path or Path(saved['recipe']['policy']['model_path'])
        args.submission_dir = args.submission_dir or Path(saved['submission_dir'])
        if args.task_id is None and args.task_index is None:
            args.task_id, args.task_index = saved.get('task_id'), saved.get('task_index')
    if args.data_root is None or args.model_path is None or args.run_dir is None:
        parser.error('New training requires --data-root, --model-path, and a new --run-dir')
    if cfg['runtime']['sandbox'] == 'subprocess' and os.geteuid() != 0:
        parser.error('The isolated subprocess backend needs root to lower the agent tool uid')
    data_root, model_path = args.data_root.expanduser().resolve(strict=True), args.model_path.expanduser().resolve(strict=True)
    # Keep a venv's executable entry point; resolving its symlink can switch
    # sys.prefix and silently inspect/run the base Python environment instead.
    runtime_python = Path(cfg['runtime']['python']).expanduser().absolute()
    cfg['runtime']['python'] = str(runtime_python)
    if not runtime_python.is_file() or not (data_root / 'tasks.jsonl').is_file() or not (model_path / 'config.json').is_file():
        parser.error('Prepare the official runtime, original public tasks.jsonl, and model directory on the server')
    counts = {name: int(cfg['run'][name]) for name in ('num_generations', 'max_batches', 'groups_per_batch',
                                                     'max_candidate_groups_per_batch', 'max_cpu_workers')}
    prefetch = int(cfg['run'].get('prefetch_groups', 2))
    iterations = int(cfg['grpo']['num_iterations'])
    if (min(*counts.values(), prefetch, iterations) < 1 or counts['num_generations'] < 2
            or counts['max_candidate_groups_per_batch'] < counts['groups_per_batch']):
        parser.error('Require K>=2, positive batch/worker/prefetch/reuse counts, and enough candidate groups')
    cfg['run']['prefetch_groups'] = min(prefetch, counts['max_candidate_groups_per_batch'])
    cfg['run'].setdefault('max_worker_retries', 1)
    cfg['run'].setdefault('max_failed_groups_per_session', 3)
    cfg['runtime'].setdefault('session_clock', 'queue_excluded')
    cfg['runtime'].setdefault('queue_timing_poll_seconds', 0.2)
    poll_seconds = float(cfg['runtime']['queue_timing_poll_seconds'])
    if (cfg['runtime']['session_clock'] not in ('queue_excluded', 'wall')
            or not math.isfinite(poll_seconds) or not 0.02 <= poll_seconds <= 1):
        parser.error('session_clock must be queue_excluded/wall; require 0.02<=poll seconds<=1')
    if int(cfg['run']['max_worker_retries']) < 0 or int(cfg['run']['max_failed_groups_per_session']) < 1:
        parser.error('Worker retries must be nonnegative; the failed-group limit must be positive')
    if cfg['sampling'] != {'temperature': 1.0, 'top_p': 1.0, 'top_k': 0}:
        parser.error('The current actor uses full categorical sampling')
    walltime, std_floor = float(cfg['run']['walltime_seconds']), float(cfg['grpo']['advantage_std_floor'])
    if not math.isfinite(walltime) or walltime <= 0 or not math.isfinite(std_floor) or std_floor <= 0:
        parser.error('Positive finite walltime and advantage_std_floor are required')
    if cfg['grpo'].get('scale_rewards', 'group') not in ('group', 'none'):
        parser.error('scale_rewards must be group or none')
    includes = [args.task_id] if args.task_id is not None else cfg.get('data', {}).get('task_ids') or None
    tasks = load_tasks(data_root / 'tasks.jsonl', include_ids=includes,
                       exclude_ids=cfg.get('data', {}).get('exclude_task_ids') or [])
    if args.task_index is not None:
        if not 0 <= args.task_index < len(tasks):
            parser.error('task-index is outside the eligible public task catalog')
        tasks = [tasks[args.task_index]]
    TaskSampler(tasks, cfg.get('curriculum'), seed=cfg['policy'].get('seed', 3407))
    submission = (args.submission_dir or data_root / 'sample_submission').expanduser().resolve(strict=True)
    cfg['policy']['model_path'] = str(model_path)
    cfg['policy']['algorithm'] = 'trl_clipped_grpo_composite_adaptive'
    if cfg['policy'].get('response_template_path'):
        cfg['policy']['response_template_path'] = str(Path(cfg['policy']['response_template_path']).expanduser().resolve(strict=True))
    os.umask(0o077)
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=args.resume, mode=0o700)
    signal.signal(signal.SIGTERM, terminate)
    from .run_context import source_binding
    with run_lock(run_dir) as lock_fd:
        stopped = cleanup_orphan_workers(run_dir) if args.resume else []
        binding = source_binding(cfg, data_root=data_root, model_path=model_path, submission=submission,
                                 runtime_python=runtime_python, tasks=tasks)
        if saved and saved['source_binding'] != binding:
            raise ValueError('Resume source/configuration differs from the committed run; refusing to mix states')
        if not saved:
            saved = {**cfg, 'recipe': cfg, 'training_tasks': tasks, 'data_root': str(data_root),
                     'submission_dir': str(submission), 'task_id': args.task_id, 'task_index': args.task_index,
                     'source_binding': binding, 'public_tasks_sha256': file_digest(data_root / 'tasks.jsonl')}
            write_json(run_dir / 'run_config.json', saved)
        budget = RunBudget(run_dir, walltime, args.additional_walltime_seconds)
        child = None
        try:
            invocation_id = secrets.token_hex(12)
            launch = {'recipe': cfg, 'training_tasks': tasks, 'data_root': str(data_root),
                      'submission_dir': str(submission), 'source_binding': binding, 'resume': args.resume,
                      'invocation_id': invocation_id, 'agent_uid_base': 100000 + secrets.randbelow(1000000),
                      'deadline_unix': time.time() + max(0.0, budget.remaining_seconds -
                                                        (time.monotonic() - budget.started))}
            write_json(run_dir / 'recovery' / 'launch.json', launch)
            emit(run_dir / 'metrics.jsonl', 'run_launch', resume=args.resume, invocation_id=invocation_id,
                 remaining_seconds=budget.remaining_seconds, additional_seconds=args.additional_walltime_seconds,
                 stopped_orphan_workers=stopped, agent_uid_base=launch['agent_uid_base'])
            child = subprocess.Popen([sys.executable, '-m', 'training.run', '--_worker', '--run-dir', str(run_dir),
                                      '--_lock-fd', str(lock_fd), '--_invocation-id', invocation_id],
                                     cwd=str(PROJECT), start_new_session=True, pass_fds=(lock_fd,))
            return child.wait(timeout=max(0.01, budget.remaining_seconds - (time.monotonic() - budget.started)))
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            print('Run interrupted or walltime exhausted; preserving its last committed state.', flush=True)
            return 124
        finally:
            if child is not None and child.poll() is None:
                stop_process_tree(child, grace=10)
            # A SIGKILLed learner cannot reap its own registered CPU workers.
            # Clean up only those exact recorded process identities.
            cleanup_orphan_workers(run_dir)
            budget.close()


if __name__ == '__main__':
    raise SystemExit(main())
