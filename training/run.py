"""Adaptive official-task collection and TRL clipped GRPO for an Unsloth LoRA.

Complete groups receive judge-derived composite rewards, low-variance groups
are filtered with a bounded refill budget, and accepted groups are reused only
within the current policy-version batch. Official assets are never rewritten.
"""
from __future__ import annotations

import argparse
from collections import deque
import concurrent.futures
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
from .rewards import compute_reward

PROJECT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rpartition(")")[2].split()
        return int(fields[1]), fields[19]
    except (OSError, ValueError, IndexError):
        return None


def stop_process_tree(proc, grace=2):
    """Also stop official command processes that created their own sessions."""
    snapshot = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        pid = int(path.parent.name)
        identity = process_identity(pid)
        if identity:
            snapshot[pid] = identity
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
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


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


def run_worker(command, folder, deadline, expired, processes, process_lock):
    if expired.is_set() or time.monotonic() >= deadline:
        raise TimeoutError("Run expired before launching this task worker")
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (folder / "worker.log").open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True, cwd=str(PROJECT))
        with process_lock:
            processes.add(proc)
        try:
            # Cancellation can race Popen and registration in the shared set.
            if expired.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("Run expired while launching this task worker")
            proc.wait(timeout=max(1, deadline - time.monotonic()))
        except BaseException:
            stop_process_tree(proc)
            raise
        finally:
            with process_lock:
                processes.discard(proc)
    result_path = folder / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"Official worker exited {proc.returncode} without a verdict: {folder / 'worker.log'}")
    result = json.loads(result_path.read_text())
    if proc.returncode or result.get("infrastructure_error"):
        raise RuntimeError(f"Official worker failed; see {result_path}")
    if not result.get("verdict_valid") or not result.get("verification_complete"):
        raise RuntimeError(f"Official verification is incomplete; see {result_path}")
    if result.get("reward") not in (0, 1):
        raise RuntimeError("Official binary outcome is missing; refusing to invent a reward")
    return result


def collect_batch(*, cfg, batch, policy_version, sampler, endpoint, run_dir,
                  runtime_python, data_root, submission, uid_base, deadline,
                  expired, processes, process_lock, cancel_run):
    """Collect bounded, overlapping groups under one unchanged learner policy.

    Worker slots are filled as soon as a trajectory ends, including from other
    groups. Completed groups are scored and committed in their selection order,
    never their completion order. That prevents fast tasks from winning the
    batch's learning slots. Every pending group reserves one remaining learning
    slot; only completed low-signal groups release a slot for another candidate.
    All started groups still finish their complete trajectories and acceptance.
    """
    generations = int(cfg["run"]["num_generations"])
    target_groups = int(cfg["run"]["groups_per_batch"])
    max_candidates = int(cfg["run"]["max_candidate_groups_per_batch"])
    prefetch_groups = min(int(cfg["run"].get("prefetch_groups", 2)), max_candidates)
    workers = int(cfg["run"]["max_cpu_workers"])
    std_floor = float(cfg["grpo"]["advantage_std_floor"])
    scaling = cfg["grpo"].get("scale_rewards", "group")
    metrics = run_dir / "metrics.jsonl"
    accepted, attempted, used_repos = [], set(), set()
    pending, runnable = deque(), deque()
    futures = {}
    completed_groups = candidate_successes = candidate_trajectories = 0
    next_candidate = 0
    catalog_exhausted = False
    sampler.set_policy_version(policy_version)

    emit(metrics, "batch_collection_started", batch=batch, policy_version=policy_version,
         target_groups=target_groups, group_size=generations, prefetch_groups=prefetch_groups,
         max_cpu_workers=workers, group_commit_order="selection_order")
    endpoint.begin([], policy_version)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)

    def select_groups():
        nonlocal next_candidate, catalog_exhausted
        while (len(pending) < min(prefetch_groups, target_groups - len(accepted))
               and next_candidate < max_candidates and not catalog_exhausted):
            selection = sampler.choose(exclude_ids=attempted, used_repos=used_repos)
            if selection is None:
                catalog_exhausted = True
                break
            candidate = next_candidate
            next_candidate += 1
            attempted.add(selection["id"])
            used_repos.add(selection["repo"])
            group_id = f"batch-{batch:04d}-group-{candidate:03d}"
            identities = [f"{group_id}-rollout-{i:03d}" for i in range(generations)]
            endpoint.register(identities, policy_version)
            state = {"candidate": candidate, "selection": selection, "group_id": group_id,
                     "identities": identities, "results": [None] * generations,
                     "next_rollout": 0, "completed": 0, "selected_at": time.monotonic()}
            pending.append(state)
            runnable.append(state)
            emit(metrics, "group_selected", batch=batch, group_id=group_id,
                 prefetched_groups=len(pending), **selection)

    def fill_worker_slots():
        # Only submit runnable work, rather than burying whole groups in the
        # executor's FIFO. Round-robin submission gives every prefetched group
        # access to CPU tools and the shared GPU request scheduler.
        while runnable and len(futures) < workers:
            state = runnable.popleft()
            index = state["next_rollout"]
            state["next_rollout"] += 1
            if state["next_rollout"] < generations:
                runnable.append(state)
            identity = state["identities"][index]
            folder = run_dir / identity
            # Overlapping groups must never share their low-privilege agent UID.
            agent_uid = uid_base + (batch * max_candidates + state["candidate"]) * generations + index
            command = [str(runtime_python), str(PROJECT / "training/official_runtime.py"),
                       "--data-root", str(data_root), "--submission-dir", str(submission),
                       "--endpoint", endpoint.url, "--workspace", str(folder / "workspace"),
                       "--result-file", str(folder / "result.json"), "--rollout-id", identity,
                       "--sandbox", cfg["runtime"]["sandbox"], "--agent-uid", str(agent_uid),
                       "--sandbox-baseline-python", cfg["runtime"].get("sandbox_baseline_python", "/opt/gemma-sandbox-baseline/bin/python"),
                       "--evaluation-overrides-json", json.dumps(cfg["runtime"].get("evaluation", {})),
                       "--task-id", state["selection"]["id"]]
            future = pool.submit(run_worker, command, folder, deadline, expired, processes, process_lock)
            futures[future] = (state, index)

    def commit_group(state):
        nonlocal completed_groups, candidate_successes, candidate_trajectories
        selection, identities, results = state["selection"], state["identities"], state["results"]
        task_id, group_id = selection["id"], state["group_id"]
        if any(endpoint.errors.get(identity) for identity in identities):
            raise RuntimeError("A candidate group has failed model calls; no partial update")
        calls = [endpoint.calls[identity] for identity in identities]
        if any(not items for items in calls):
            raise RuntimeError("An official trajectory has no recorded model actions")
        if any(result["task_id"] != task_id for result in results):
            raise RuntimeError("Official task identity differs from the selected group")
        if any(call["policy_version"] != policy_version for items in calls for call in items):
            raise RuntimeError("A collected trajectory crossed learner policy versions")

        # Use one common reward feature set for every trajectory in the group.
        reward_config = dict(cfg["rewards"])
        dense_features = [result.get("reward_features") or {} for result in results]
        dense_available = (
            all(item.get("junit_available") and item.get("dense_coverage_complete")
                for item in dense_features)
            and len({(item.get("repair_fraction") is None, item.get("regression_fraction") is None)
                     for item in dense_features}) == 1
        )
        reward_config["dense_shaping_enabled"] = dense_available
        reward_mode = "full_composite" if dense_available else "terminal_efficiency_only_missing_dense_evidence"
        # Compare recent reward statistics only when the effective formula and
        # observable dense components agree. This contains no test identifiers.
        reward_signature = json.dumps({
            "reward_config": reward_config,
            "repair_available": dense_available and dense_features[0].get("repair_fraction") is not None,
            "regression_available": dense_available and dense_features[0].get("regression_fraction") is not None,
        }, sort_keys=True, separators=(",", ":"))
        scored = [compute_reward(result, items, reward_config) for result, items in zip(results, calls)]
        rewards = [item["total"] for item in scored]
        outcomes = [bool(result["success"]) for result in results]
        candidate_successes += sum(outcomes)
        candidate_trajectories += len(outcomes)
        group_stats = sampler.observe(task_id, rewards, outcomes, policy_version=policy_version,
                                      reward_signature=reward_signature)
        completed_groups += 1
        mean_reward, deviation = group_stats["reward_mean"], group_stats["reward_std"]
        denominator = max(deviation, std_floor) if scaling == "group" else 1.0
        advantages = [(value - mean_reward) / denominator for value in rewards]
        selected = bool(group_stats["informative"])
        if selected and len(accepted) >= target_groups:
            raise RuntimeError("Pending candidate groups exceeded the remaining learning slots")
        exclusion = None if selected else "low_reward_variance"
        trajectories = []
        for identity, items, official, score, advantage in zip(identities, calls, results, scored, advantages):
            trajectories.append({
                "task_id": task_id, "group_id": group_id, "rollout_id": identity,
                "policy_version": policy_version, "calls": items,
                "reward": score["total"], "reward_components": score["components"],
                "advantage": advantage, "official_success": official["success"],
                "selection_probability": selection["selection_probability"],
            })
            write_json(run_dir / identity / "reward.json", {
                "official_reward": official["reward"], "group_reward_mode": reward_mode,
                "reward_signature": reward_signature,
                "effective_reward_weights": reward_config, **score, "advantage": advantage,
                "selected_for_update": selected, "exclusion_reason": exclusion,
            })
        emit(metrics, "group_verified", batch=batch, group_id=group_id, **group_stats,
             rewards=rewards, advantages=advantages, reward_mode=reward_mode,
             selected_for_update=selected, exclusion_reason=exclusion,
             selection_to_commit_seconds=time.monotonic() - state["selected_at"])
        write_json(run_dir / "curriculum_state.json", sampler.state_dict())
        if selected:
            accepted.append(trajectories)
        endpoint.release_rollouts(identities)

    try:
        select_groups()
        fill_worker_slots()
        while pending:
            if expired.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("Run budget exhausted during group collection")
            # Commit all currently complete predecessors before drawing again.
            # Reward observations and target decisions thus ignore completion order.
            while pending and pending[0]["completed"] == generations:
                commit_group(pending.popleft())
            select_groups()
            fill_worker_slots()
            if not pending:
                break
            if not futures:
                raise RuntimeError("Incomplete candidate group has no runnable trajectories")
            done, _ = concurrent.futures.wait(
                futures, timeout=min(0.5, max(0, deadline - time.monotonic())),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                state, index = futures.pop(future)
                state["results"][index] = future.result()
                state["completed"] += 1
                if state["completed"] == generations:
                    emit(metrics, "group_rollouts_completed", batch=batch, group_id=state["group_id"],
                         trajectories=generations, elapsed_seconds=time.monotonic() - state["selected_at"])
        if futures or runnable:
            raise RuntimeError("Collection ended with outstanding official workers")
    except BaseException:
        cancel_run()
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        # end() drains admitted GPU requests and clears retained inference KV
        # before any probability recomputation or optimizer update can begin.
        endpoint.end()
    if expired.is_set() or time.monotonic() >= deadline:
        raise TimeoutError("Run budget exhausted before collection drained")
    return {"accepted": accepted, "attempted": attempted, "completed_groups": completed_groups,
            "candidate_successes": candidate_successes, "candidate_trajectories": candidate_trajectories}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/a800_2x80.toml")
    parser.add_argument("--describe", action="store_true", help="Print the recipe without loading data or CUDA")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--submission-dir", type=Path, help="Defaults to the official sample_submission")
    parser.add_argument("--run-dir", type=Path)
    task = parser.add_mutually_exclusive_group()
    task.add_argument("--task-id")
    task.add_argument("--task-index", type=int, help="Optional index within eligible public tasks; default trains the catalog")
    args = parser.parse_args()
    cfg = tomllib.loads(args.config.read_text())
    if args.describe:
        print(json.dumps({"recipe": cfg, "algorithm": "trl_clipped_grpo_composite_adaptive",
                          "data_access": "official public tasks on server only",
                          "gpu_validation": "not yet performed"}, ensure_ascii=False, indent=2))
        return 0
    if sys.platform != "linux":
        parser.error("Run training on the Linux GPU server; local preparation uses --describe.")
    if args.data_root is None or args.model_path is None or args.run_dir is None:
        parser.error("Training requires --data-root, --model-path, and a new --run-dir.")
    if not args._worker:
        # Keep a CPU-only parent outside CUDA/imports so even a stalled model
        # load or GPU kernel cannot silently run past the configured walltime.
        child = subprocess.Popen([sys.executable, "-m", "training.run", *sys.argv[1:], "--_worker"],
                                 cwd=str(PROJECT), start_new_session=True)
        try:
            return child.wait(timeout=float(cfg["run"]["walltime_seconds"]))
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            print("Run interrupted or walltime exhausted; terminating the learner and its task workers.", flush=True)
            stop_process_tree(child, grace=10)
            return 124

    def terminate_worker(signum, frame):
        raise KeyboardInterrupt("Parent requested run termination")

    signal.signal(signal.SIGTERM, terminate_worker)
    if cfg["runtime"]["sandbox"] == "subprocess" and os.geteuid() != 0:
        parser.error("The isolated subprocess backend needs root to lower the agent tool uid.")
    data_root = args.data_root.resolve(strict=True)
    model_path = args.model_path.resolve(strict=True)
    runtime_python = Path(cfg["runtime"]["python"])
    if not runtime_python.is_file():
        parser.error("Prepare the separate official runtime environment before starting the GPU run.")
    if not (data_root / "tasks.jsonl").is_file():
        parser.error("Missing original official public tasks.jsonl under --data-root.")
    if not (model_path / "config.json").is_file():
        parser.error("Missing model config.json under --model-path.")
    generations = int(cfg["run"]["num_generations"])
    batches = int(cfg["run"]["max_batches"])
    groups_per_batch = int(cfg["run"]["groups_per_batch"])
    max_candidates = int(cfg["run"]["max_candidate_groups_per_batch"])
    workers = int(cfg["run"]["max_cpu_workers"])
    prefetch_groups = int(cfg["run"].get("prefetch_groups", 2))
    iterations = int(cfg["grpo"]["num_iterations"])
    if min(batches, groups_per_batch, workers, prefetch_groups, iterations) < 1 or generations < 2 or max_candidates < groups_per_batch:
        parser.error("Require K>=2, positive batch/worker/prefetch/reuse counts, and enough candidate groups.")
    cfg["run"]["prefetch_groups"] = min(prefetch_groups, max_candidates)
    if cfg["sampling"] != {"temperature": 1.0, "top_p": 1.0, "top_k": 0}:
        parser.error("The current actor uses full categorical sampling; other sampling distributions need an explicit adapter.")
    walltime = float(cfg["run"]["walltime_seconds"])
    std_floor = float(cfg["grpo"]["advantage_std_floor"])
    if not math.isfinite(walltime) or walltime <= 0 or not math.isfinite(std_floor) or std_floor <= 0:
        parser.error("Positive finite walltime and advantage_std_floor are required.")
    scaling = cfg["grpo"].get("scale_rewards", "group")
    if scaling not in ("group", "none"):
        parser.error("scale_rewards must be group or none.")
    includes = cfg.get("data", {}).get("task_ids") or None
    excludes = cfg.get("data", {}).get("exclude_task_ids") or []
    if args.task_id is not None:
        includes = [args.task_id]
    tasks = load_tasks(data_root / "tasks.jsonl", include_ids=includes, exclude_ids=excludes)
    if args.task_index is not None:
        if not 0 <= args.task_index < len(tasks):
            parser.error("task-index is outside the eligible public task catalog.")
        tasks = [tasks[args.task_index]]
    sampler = TaskSampler(tasks, cfg.get("curriculum"), seed=cfg["policy"].get("seed", 3407))
    os.umask(0o077)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    submission = (args.submission_dir or data_root / "sample_submission").resolve(strict=True)
    metrics = run_dir / "metrics.jsonl"
    cfg["policy"]["model_path"] = str(model_path)
    cfg["policy"]["algorithm"] = "trl_clipped_grpo_composite_adaptive"
    response_template_metadata = None
    if cfg["policy"].get("response_template_path"):
        response_template_path = Path(cfg["policy"]["response_template_path"]).expanduser().resolve(strict=True)
        cfg["policy"]["response_template_path"] = str(response_template_path)
        response_template_metadata = {
            "path": str(response_template_path),
            "sha256": file_hash(response_template_path),
            "purpose": "response parser only; does not replace checkpoint tokenizer assets",
        }
    uid_base = 100000 + secrets.randbelow(1000000)
    write_json(run_dir / "run_config.json", {**cfg, "data_root": str(data_root),
               "task_id": args.task_id, "task_index": args.task_index, "training_tasks": tasks,
               "submission_dir": str(submission), "algorithm": "trl_clipped_grpo_composite_adaptive",
               "public_tasks_sha256": file_hash(data_root / "tasks.jsonl"),
               "agent_uid_base": uid_base,
               "model_source": {
                   "declared_repo_id": cfg["policy"]["model_name"],
                   "declared_revision": cfg["policy"].get("model_revision"),
                   "declared_source_uri": cfg["policy"].get("model_source_uri"),
                   "response_template_metadata": response_template_metadata,
                   "metadata_sha256": {
                       name: file_hash(model_path / name)
                       for name in ("config.json", "tokenizer_config.json", "chat_template.jinja",
                                    "generation_config.json", "model.safetensors.index.json")
                       if (model_path / name).is_file()
                   },
               }})
    emit(metrics, "loading_policy", declared_model_revision=cfg["policy"].get("model_revision"),
         eligible_tasks=len(tasks))
    started = time.monotonic()
    deadline = started + walltime
    from .policy import Policy
    policy = Policy(cfg["policy"])
    from .grpo_trainer import make_grpo_trainer
    trainer_config = {**cfg["grpo"], "learning_rate": float(cfg["run"]["learning_rate"]),
                      "max_grad_norm": float(cfg["run"]["max_grad_norm"]),
                      "total_optimizer_steps": batches * iterations,
                      "num_generations": generations, "seed": int(cfg["policy"].get("seed", 3407))}
    trainer = make_grpo_trainer(
        policy, trainer_config, run_dir / "trainer",
        lambda event, **fields: emit(metrics, event, **fields),
    )
    import torch
    endpoint = PolicyEndpoint(policy, run_dir, int(cfg["runtime"].get("port", 0)),
                              generation_config=cfg.get("generation", {}))
    processes, process_lock = set(), threading.Lock()
    expired = threading.Event()

    def cancel_run():
        expired.set()
        policy.abort_generation.set()
        with process_lock:
            for proc in list(processes):
                stop_process_tree(proc, grace=0.5)

    timer = threading.Timer(max(0, deadline - time.monotonic()), cancel_run)
    timer.daemon = True
    timer.start()
    endpoint.start()
    policy_version = 0
    try:
        initial = run_dir / "adapters" / "initial"
        policy.save_adapter(initial)
        previous_hash = file_hash(initial / "adapter_model.safetensors")
        for batch in range(batches):
            if expired.is_set():
                raise TimeoutError("Run budget exhausted before collection")
            collection = collect_batch(
                cfg=cfg, batch=batch, policy_version=policy_version, sampler=sampler,
                endpoint=endpoint, run_dir=run_dir, runtime_python=runtime_python,
                data_root=data_root, submission=submission, uid_base=uid_base,
                deadline=deadline, expired=expired, processes=processes,
                process_lock=process_lock, cancel_run=cancel_run,
            )
            accepted, attempted = collection["accepted"], collection["attempted"]
            completed_groups = collection["completed_groups"]
            candidate_successes = collection["candidate_successes"]
            candidate_trajectories = collection["candidate_trajectories"]
            if not accepted:
                emit(metrics, "batch_no_learning_signal", batch=batch,
                     completed_groups=completed_groups, attempted_tasks=sorted(attempted),
                     sampled_official_success_rate=(candidate_successes / candidate_trajectories
                                                    if candidate_trajectories else None))
                continue
            trajectories = [trajectory for group in accepted for trajectory in group]
            emit(metrics, "grpo_update_started", batch=batch, policy_version=policy_version,
                 accepted_groups=len(accepted), trajectories=len(trajectories),
                 candidate_groups=completed_groups, reuse_iterations=iterations)
            # No actor requests are active during optimization. Old learner,
            # behavior and initial-reference probabilities remain separate.
            stats = trainer.optimize_batch(trajectories)
            if expired.is_set():
                raise TimeoutError("Run budget exhausted during GRPO updates")
            policy_version += int(stats["optimizer_steps"])
            destination = run_dir / "adapters" / f"batch-{batch + 1:04d}"
            policy.save_adapter(destination)
            current_hash = file_hash(destination / "adapter_model.safetensors")
            emit(metrics, "batch_finished", batch=batch, policy_version=policy_version,
                 accepted_groups=len(accepted), candidate_groups=completed_groups,
                 sampled_official_success_rate=candidate_successes / candidate_trajectories,
                 selected_official_success_rate=sum(item["official_success"] for item in trajectories) / len(trajectories),
                 composite_reward_mean=sum(item["reward"] for item in trajectories) / len(trajectories),
                 adapter_changed=current_hash != previous_hash, adapter_sha256=current_hash,
                 adapter_path=str(destination), grpo=stats,
                 peak_memory_gib=[torch.cuda.max_memory_allocated(i) / 2**30
                                  for i in range(torch.cuda.device_count())])
            previous_hash = current_hash
        emit(metrics, "run_finished", elapsed_seconds=time.monotonic() - started,
             optimizer_steps=policy_version)
        return 0
    except BaseException as exc:
        cancel_run()
        emit(metrics, "run_failed", error=f"{type(exc).__name__}: {exc}",
             elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        timer.cancel()
        endpoint.end()
        endpoint.close()


if __name__ == "__main__":
    raise SystemExit(main())
