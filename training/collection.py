"""Recoverable complete-group collection with bounded worker-attempt failures."""
from __future__ import annotations

from collections import deque
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import threading
import time
import uuid

from .recovery import CollectionInterrupted, WorkerAttemptError, atomic_json, file_digest, value_digest
from .rewards import score_group_rewards
from .reward_ledger import RewardLedger


def collect_batch(*, cfg, batch, policy_version, sampler, endpoint, run_dir,
                  runtime_python, data_root, submission, uid_base, deadline,
                  expired, processes, process_lock, cancel_run, reward_ledger=None,
                  recovery_state=None, persist_callback=None, worker_fn, emit_fn):
    """Keep valid completed slots; retry failed infrastructure, never failed tests.

    Selection/observation commits remain ordered, but every completed Future is
    inspected immediately. Infrastructure-exhausted groups never receive scores
    or enter the sampler's feedback. In-flight agent sessions are not resumed;
    a new attempt has a new identity and fresh workspace under the same policy.
    """
    run_dir = Path(run_dir)
    metrics = run_dir / "metrics.jsonl"
    generations = int(cfg["run"]["num_generations"])
    target = int(cfg["run"]["groups_per_batch"])
    max_candidates = int(cfg["run"]["max_candidate_groups_per_batch"])
    workers = int(cfg["run"]["max_cpu_workers"])
    prefetch = int(cfg["run"].get("prefetch_groups", 2))
    retry_limit = int(cfg["run"].get("max_worker_retries", 1))
    failure_limit = int(cfg["run"].get("max_failed_groups_per_session", 3))
    if retry_limit < 0 or failure_limit < 1:
        raise ValueError("Worker retries must be nonnegative and failure limit positive")
    ledger = reward_ledger or RewardLedger(run_dir)
    session = uuid.uuid4().hex[:10]
    state = copy.deepcopy(recovery_state) if recovery_state is not None else {
        "schema_version": 1, "batch": batch, "policy_version": policy_version,
        "groups": [], "commit_cursor": 0, "attempted": [], "used_repos": [],
        "accepted_group_ids": [], "completed_groups": 0, "candidate_successes": 0,
        "candidate_trajectories": 0, "failed_groups": 0, "catalog_exhausted": False,
        "complete": False,
    }
    if state.get("schema_version") != 1 or state["batch"] != batch or state["policy_version"] != policy_version:
        raise ValueError("Collection checkpoint belongs to a different batch or policy")
    if recovery_state is not None:
        sampler.load_state_dict(state["sampler"])
    sampler.set_policy_version(policy_version)
    calls_by_identity = {}
    accepted = []
    running = {}
    cancel_events = {}
    session_failed_groups = 0
    submission_counter = 0
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    endpoint.begin([], policy_version)

    def persist():
        state["sampler"] = sampler.state_dict()
        if persist_callback:
            persist_callback(copy.deepcopy(state))

    def check_run():
        if expired.is_set() or time.monotonic() >= deadline:
            raise CollectionInterrupted("Run budget exhausted during collection")
        if endpoint.fatal_error is not None:
            raise CollectionInterrupted(f"Generation backend failed: {endpoint.fatal_error}")

    def completed_slot(group, slot):
        identity = slot["identity"]
        path = run_dir / identity / "result.json"
        if file_digest(path) != slot["result_sha256"]:
            raise ValueError("Completed rollout result changed after its checkpoint")
        result = json.loads(path.read_text())
        if (result.get("task_id") != group["selection"]["id"] or result.get("rollout_id") != identity
                or not result.get("verdict_valid") or not result.get("verification_complete")
                or result.get("infrastructure_error")):
            raise ValueError("Completed rollout identity/verdict does not match its checkpoint")
        calls = endpoint.recover_rollout_calls(identity, policy_version)
        if not calls or value_digest(calls) != slot["calls_sha256"]:
            raise ValueError("Consumed model calls changed after their checkpoint")
        calls_by_identity[identity] = calls
        return result

    def score_group(group):
        results = [completed_slot(group, slot) for slot in group["slots"]]
        identities = [slot["identity"] for slot in group["slots"]]
        calls = [calls_by_identity[identity] for identity in identities]
        scored = score_group_rewards(results, calls, cfg["rewards"],
                                     scale_rewards=cfg["grpo"].get("scale_rewards", "group"),
                                     advantage_std_floor=float(cfg["grpo"]["advantage_std_floor"]),
                                     min_group_std=float(sampler.config["min_group_std"]))
        trajectories = []
        for identity, items, result, reward, signal, advantage, bounds in zip(
                identities, calls, results, scored["scored"], scored["learning_signal"],
                scored["advantages"], scored["advantage_bounds"]):
            trajectories.append({"task_id": group["selection"]["id"], "group_id": group["id"],
                                 "rollout_id": identity, "policy_version": policy_version, "calls": items,
                                 "reward": reward["total"], "reward_bounds": reward["reward_bounds"],
                                 "reward_components": reward["components"], "learning_signal": signal,
                                 "advantage": advantage, "advantage_bounds": bounds,
                                 "reward_mode": scored["reward_mode"], "official_success": result["success"],
                                 "selection_probability": group["selection"]["selection_probability"]})
        return results, calls, scored, trajectories

    def commit_group(group):
        results, calls, scored, trajectories = score_group(group)
        outcomes = [result["success"] for result in results]
        statistics = sampler.observe(group["selection"]["id"], scored["rewards"], outcomes,
                                     policy_version=policy_version, reward_signature=scored["reward_signature"],
                                     learning_signal=scored["learning_signal"])
        selected = bool(statistics["informative"])
        if selected and len(accepted) >= target:
            raise RuntimeError("Pending groups exceeded reserved learning slots")
        exclusion = None if selected else "no_safe_learning_signal"
        for slot, reward, trajectory in zip(group["slots"], scored["scored"], trajectories):
            atomic_json(run_dir / slot["identity"] / "reward.json", {
                "official_reward": int(trajectory["official_success"]), **reward,
                "group_reward_mode": scored["reward_mode"], "reward_signature": scored["reward_signature"],
                "effective_reward_weights": dict(cfg["rewards"]), "learning_signal": trajectory["learning_signal"],
                "advantage": trajectory["advantage"], "advantage_bounds": trajectory["advantage_bounds"],
                "group_diagnostics": scored["diagnostics"], "selected_for_update": selected,
                "exclusion_reason": exclusion})
        ledger.record_group(batch=batch, group_id=group["id"], task_id=group["selection"]["id"],
                            policy_version=policy_version, selection=group["selection"], scored=scored["scored"],
                            advantages=scored["advantages"], group_stats=statistics, results=results, calls=calls,
                            selected_for_update=selected, exclusion_reason=exclusion,
                            reward_mode=scored["reward_mode"], reward_signature=scored["reward_signature"],
                            diagnostics=scored["diagnostics"])
        group["status"] = "committed"
        group["selected"] = selected
        state["completed_groups"] += 1
        state["candidate_successes"] += sum(outcomes)
        state["candidate_trajectories"] += len(outcomes)
        if selected:
            accepted.append(trajectories)
            state["accepted_group_ids"].append(group["id"])
        emit_fn(metrics, "group_verified", batch=batch, group_id=group["id"], **statistics,
                rewards=scored["rewards"], advantages=scored["advantages"],
                learning_signal=scored["learning_signal"], advantage_bounds=scored["advantage_bounds"],
                reward_mode=scored["reward_mode"], reward_diagnostics=scored["diagnostics"],
                selected_for_update=selected, exclusion_reason=exclusion)
        atomic_json(run_dir / "curriculum_state.json", sampler.state_dict())
        endpoint.release_rollouts([slot["identity"] for slot in group["slots"]])

    def pending_groups():
        return [group for group in state["groups"] if group["status"] not in ("committed", "failed")]

    def select_groups():
        changed = False
        while (len(pending_groups()) < min(prefetch, target - len(accepted))
               and len(state["groups"]) < max_candidates and not state["catalog_exhausted"]
               and session_failed_groups < failure_limit):
            selection = sampler.choose(exclude_ids=state["attempted"], used_repos=state["used_repos"])
            if selection is None:
                state["catalog_exhausted"] = True
                break
            index = len(state["groups"])
            identifier = f"batch-{batch:04d}-group-{index:03d}"
            state["attempted"].append(selection["id"])
            state["used_repos"].append(selection["repo"])
            group = {"id": identifier, "selection": selection, "status": "pending", "selected": False,
                     "slots": [{"index": i, "attempt": 0, "failures": 0, "status": "pending"}
                               for i in range(generations)]}
            state["groups"].append(group)
            emit_fn(metrics, "group_selected", batch=batch, group_id=identifier,
                    prefetched_groups=len(pending_groups()), **selection)
            changed = True
        if changed:
            persist()  # Selection/RNG commit precedes launching any selected task.

    def fill_slots():
        nonlocal submission_counter
        # Round-robin by group, keeping retries from monopolizing worker slots.
        queue = deque(pending_groups())
        while queue and len(running) < workers:
            group = queue.popleft()
            pending = [slot for slot in group["slots"] if slot["status"] == "pending"]
            if not pending:
                continue
            slot = pending[0]
            slot["attempt"] += 1
            identity = f"{group['id']}-rollout-{slot['index']:03d}-a{slot['attempt']:03d}-{session}"
            slot.update(identity=identity, status="running")
            endpoint.register([identity], policy_version)
            folder = run_dir / identity
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            credential_path = folder / "model_auth_token"
            descriptor = os.open(credential_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as credential:
                credential.write(endpoint.rollout_token(identity))
            cancelled = threading.Event()
            cancel_events[identity] = cancelled
            submission_counter += 1
            # Each attempt has distinct agent/judge identities, including
            # across batches. Public rollout IDs are never API credentials.
            batch_uid_span = max_candidates * generations * (retry_limit + 1)
            agent_uid = uid_base + 2 * (batch * batch_uid_span + submission_counter)
            command = [str(runtime_python), str(Path(__file__).resolve().parent / "official_runtime.py"),
                       "--data-root", str(data_root), "--submission-dir", str(submission),
                       "--endpoint", endpoint.url, "--workspace", str(folder / "workspace"),
                       "--result-file", str(folder / "result.json"), "--rollout-id", identity,
                       "--sandbox", cfg["runtime"]["sandbox"], "--agent-uid", str(agent_uid),
                       "--judge-uid", str(agent_uid + 1), "--auth-token-file", str(credential_path),
                       "--parent-pid", str(os.getpid()),
                       "--sandbox-baseline-python", cfg["runtime"].get("sandbox_baseline_python", "/opt/gemma-sandbox-baseline/bin/python"),
                       "--session-clock", cfg["runtime"].get("session_clock", "queue_excluded"),
                       "--queue-timing-poll-seconds", str(cfg["runtime"].get("queue_timing_poll_seconds", 0.2)),
                       "--evaluation-overrides-json", json.dumps(cfg["runtime"].get("evaluation", {})),
                       "--task-id", group["selection"]["id"]]
            future = pool.submit(worker_fn, command, folder, deadline, expired, processes, process_lock,
                                 attempt_cancelled=cancelled)
            running[future] = (group, slot, identity)
            if len(pending) > 1:
                queue.append(group)

    def fail_attempt(group, slot, identity, error):
        nonlocal session_failed_groups
        endpoint.cancel_rollouts([identity], reason="worker_attempt_failed")
        slot["failures"] += 1
        slot["last_error"] = f"{type(error).__name__}: {error}"
        emit_fn(metrics, "worker_attempt_failed", batch=batch, group_id=group["id"], rollout_id=identity,
                slot=slot["index"], attempt=slot["attempt"], failure_count=slot["failures"], error=slot["last_error"])
        if slot["failures"] <= retry_limit:
            slot["status"] = "pending"
            return
        group["status"] = "failed"
        group["failure_reason"] = slot["last_error"]
        state["failed_groups"] += 1
        session_failed_groups += 1
        for other in group["slots"]:
            if other["status"] not in ("complete", "cancelled"):
                other["status"] = "cancelled"
            if other.get("identity") in cancel_events:
                cancel_events[other["identity"]].set()
        endpoint.cancel_rollouts([other["identity"] for other in group["slots"] if other.get("identity")],
                                reason="group_attempts_exhausted")
        emit_fn(metrics, "group_infrastructure_failed", batch=batch, group_id=group["id"],
                complete_trajectories=sum(other["status"] == "complete" for other in group["slots"]),
                observed_as_reward=False, error=group["failure_reason"])

    try:
        # Restore only records committed together with a saved generation RNG.
        # An uncommitted/in-flight attempt gets a new identity and workspace.
        for group in state["groups"]:
            for slot in group["slots"]:
                if slot["status"] == "complete":
                    completed_slot(group, slot)
                elif slot["status"] == "running":
                    slot["status"] = "pending"
                    emit_fn(metrics, "rollout_restarted_after_interruption", batch=batch,
                            group_id=group["id"], previous_rollout_id=slot.get("identity"), slot=slot["index"])
            if group["status"] == "committed" and group["selected"]:
                accepted.append(score_group(group)[3])
        if [group[0]["group_id"] for group in accepted] != state["accepted_group_ids"]:
            raise ValueError("Recovered accepted groups do not match the collection checkpoint")
        emit_fn(metrics, "batch_collection_resumed" if recovery_state is not None else "batch_collection_started",
                batch=batch, policy_version=policy_version, target_groups=target, group_size=generations,
                recovered_complete_trajectories=sum(slot["status"] == "complete" for group in state["groups"] for slot in group["slots"]))
        if not state["complete"]:
            persist()
        while not state["complete"]:
            check_run()
            while state["commit_cursor"] < len(state["groups"]):
                group = state["groups"][state["commit_cursor"]]
                if group["status"] == "failed":
                    state["commit_cursor"] += 1
                    continue
                if not all(slot["status"] == "complete" for slot in group["slots"]):
                    break
                commit_group(group)
                state["commit_cursor"] += 1
                persist()
            select_groups()
            fill_slots()
            if not pending_groups() and not running:
                state["complete"] = True
                persist()
                break
            if not running:
                raise RuntimeError("An incomplete group has no runnable workers")
            done, _ = concurrent.futures.wait(running, timeout=min(0.2, max(0, deadline - time.monotonic())),
                                              return_when=concurrent.futures.FIRST_COMPLETED)
            # Inspect failures first; scoring/ordered commits happen later.
            ordered = sorted(done, key=lambda future: future.exception() is None)
            for future in ordered:
                group, slot, identity = running.pop(future)
                if group["status"] == "failed":
                    try:
                        future.result()
                    except Exception:
                        pass
                    continue
                try:
                    result = future.result()
                    if (result.get("task_id") != group["selection"]["id"] or result.get("rollout_id") != identity):
                        raise ValueError("Worker returned another task/attempt identity")
                    calls = endpoint.recover_rollout_calls(identity, policy_version)
                    if not calls or any(call.get("policy_version") != policy_version for call in calls):
                        raise WorkerAttemptError("No confirmed consumed calls under this collection policy")
                    slot.update(status="complete", result_sha256=file_digest(run_dir / identity / "result.json"),
                                calls_sha256=value_digest(calls))
                    calls_by_identity[identity] = calls
                    endpoint.release_rollouts([identity])
                    emit_fn(metrics, "rollout_completed", batch=batch, group_id=group["id"], rollout_id=identity,
                            slot=slot["index"], attempt=slot["attempt"])
                except WorkerAttemptError as error:
                    check_run()
                    fail_attempt(group, slot, identity, error)
            if done:
                persist()
        return {"accepted": accepted, "attempted": set(state["attempted"]),
                "completed_groups": state["completed_groups"], "failed_groups": state["failed_groups"],
                "candidate_successes": state["candidate_successes"],
                "candidate_trajectories": state["candidate_trajectories"],
                "failure_limit_reached": session_failed_groups >= failure_limit, "recovery_state": state}
    except BaseException:
        # Existing committed progress is already durable. Do not overwrite its
        # RNG with a snapshot from an unsafe/failed generation backend.
        for event in cancel_events.values():
            event.set()
        endpoint.cancel_rollouts(list(cancel_events), reason="collection_interrupted")
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        endpoint.end()
