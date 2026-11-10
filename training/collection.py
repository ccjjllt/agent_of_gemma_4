"""Recoverable groups of fully verified trajectories with bounded slot failures."""
from __future__ import annotations

from collections import deque
import concurrent.futures
import copy
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from .recovery import CollectionInterrupted, WorkerAttemptError, atomic_json, file_digest, value_digest
from .rewards import reward_objective, score_group_rewards
from .reward_ledger import RewardLedger


def collect_batch(*, cfg, batch, policy_version, sampler, endpoint, run_dir,
                  runtime_python, data_root, submission, uid_base, deadline,
                  expired, processes, process_lock, cancel_run, reward_ledger=None,
                  recovery_state=None, persist_callback=None, worker_fn, emit_fn,
                  update_reserve_seconds=None):
    """Keep valid completed slots; retry failed infrastructure, never failed tests.

    Selection/observation commits remain ordered, but every completed Future is
    inspected immediately. Exhausted slots never receive scores; groups retain
    every valid trajectory if at least two remain. In-flight sessions are not resumed;
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
    reserve = (cfg["run"].get("update_reserve_seconds", 900.0)
               if update_reserve_seconds is None else update_reserve_seconds)
    if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not math.isfinite(reserve) or reserve <= 0:
        raise ValueError("Update reserve must be positive and finite")
    reserve = float(reserve)
    collection_deadline = deadline - reserve
    if retry_limit < 0 or failure_limit < 1:
        raise ValueError("Worker retries must be nonnegative and failure limit positive")
    sandbox_root = cfg["runtime"].get("sandbox_root")
    if sandbox_root is not None:
        if not isinstance(sandbox_root, str) or not Path(sandbox_root).is_absolute():
            raise ValueError("runtime.sandbox_root must be an absolute directory path")
        normalized = Path(os.path.normpath(sandbox_root))
        if normalized.parent == normalized:
            raise ValueError("runtime.sandbox_root must not be the filesystem root")
    ledger = reward_ledger or RewardLedger(run_dir)
    objective = reward_objective(cfg["rewards"])
    session = uuid.uuid4().hex[:10]
    state = copy.deepcopy(recovery_state) if recovery_state is not None else {
        "schema_version": 1, "batch": batch, "policy_version": policy_version,
        "reward_objective": objective,
        "groups": [], "commit_cursor": 0, "attempted": [], "used_repos": [],
        "accepted_group_ids": [], "completed_groups": 0, "candidate_successes": 0,
        "candidate_trajectories": 0, "failed_groups": 0, "catalog_exhausted": False,
        "complete": False,
    }
    if state.get("schema_version") != 1 or state["batch"] != batch or state["policy_version"] != policy_version:
        raise ValueError("Collection checkpoint belongs to a different batch or policy")
    if state.get("reward_objective") != objective:
        raise ValueError("Collection reward objective changed; start a new run instead of rescoring saved trajectories")
    if recovery_state is not None:
        sampler.load_state_dict(state["sampler"])
    # Accumulate active wall time across recovery without charging downtime.
    # Older snapshots have no timing evidence; retain that missingness.
    state.setdefault("elapsed_seconds", 0.0)
    state.setdefault("timing_from_start", recovery_state is None)
    state.setdefault("partial_groups", 0)
    state.setdefault("infrastructure_failed_groups", state["failed_groups"])
    if state.get("cooldown_blocked") and not state["complete"]:
        state.update(catalog_exhausted=False, cooldown_blocked=False, next_retry_unix=None)
    if not state["complete"] and state.get("stop_reason") != "update_reserve":
        state["budget_handoff"] = False
        state["deferred_groups"] = 0
    state["update_reserve_seconds"] = reserve
    timing_checkpoint = time.monotonic()
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
        nonlocal timing_checkpoint
        now = time.monotonic()
        state["elapsed_seconds"] += now - timing_checkpoint
        timing_checkpoint = now
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

    def valid_slots(group):
        return [slot for slot in group["slots"] if slot["status"] == "complete"]

    def terminal_group(group):
        return all(slot["status"] in ("complete", "exhausted") for slot in group["slots"])

    def judge_collection_blocked(group):
        """True when an exhausted slot shows the empty-patch judge cannot collect pytest."""
        if any(slot["status"] == "complete" for slot in group["slots"]):
            return False
        for slot in group["slots"]:
            if slot["status"] != "exhausted" or not slot.get("identity"):
                continue
            path = run_dir / slot["identity"] / "result.json"
            if not path.is_file():
                continue
            try:
                document = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            control = (document.get("failure_attribution") or {}).get("empty_patch_control") or {}
            if (isinstance(control, dict) and control.get("reason") == "pytest_collection_failure"
                    and control.get("healthy") is False):
                return True
        return False

    def record_health(group):
        exhausted = sum(slot["status"] == "exhausted" for slot in group["slots"])
        if not terminal_group(group) and not exhausted:
            return  # Budget cancellation alone says nothing about task health.
        previous = group.get("health") or {}
        blocked = judge_collection_blocked(group)
        if (group.get("health_recorded") and previous.get("completed_slots") == len(valid_slots(group))
                and previous.get("exhausted_slots") == exhausted
                and previous.get("judge_collection_blocked") is blocked):
            return
        group["health"] = sampler.record_health(
            group["selection"]["id"], group_id=group["id"],
            completed_slots=len(valid_slots(group)), exhausted_slots=exhausted,
            judge_collection_blocked=blocked)
        group["health_recorded"] = True
        atomic_json(run_dir / "curriculum_state.json", sampler.state_dict())
        emit_fn(metrics, "task_health_observed", batch=batch, group_id=group["id"],
                task_id=group["selection"]["id"], health=group["health"])

    def score_group(group):
        slots = valid_slots(group)
        results = [completed_slot(group, slot) for slot in slots]
        identities = [slot["identity"] for slot in slots]
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
                                 "reward_signature": scored["reward_signature"],
                                 "reward_objective": scored["objective"],
                                 "reward_normalization": scored["normalization"],
                                 "group_missingness": scored["missingness"],
                                 "selection_probability": group["selection"]["selection_probability"]})
        return results, calls, scored, trajectories

    def commit_group(group):
        record_health(group)
        results, calls, scored, trajectories = score_group(group)
        slots = valid_slots(group)
        collection_evidence = {
            "planned_trajectories": len(group["slots"]), "valid_trajectories": len(slots),
            "exhausted_trajectories": len(group["slots"]) - len(slots),
        }
        outcomes = [result["success"] for result in results]
        statistics = sampler.observe(group["selection"]["id"], scored["rewards"], outcomes,
                                     policy_version=policy_version, reward_signature=scored["reward_signature"],
                                     learning_signal=scored["learning_signal"])
        selected = bool(statistics["informative"])
        if selected and len(accepted) >= target:
            raise RuntimeError("Pending groups exceeded reserved learning slots")
        exclusion = None if selected else "no_safe_learning_signal"
        for slot, reward, trajectory in zip(slots, scored["scored"], trajectories):
            atomic_json(run_dir / slot["identity"] / "reward.json", {
                "official_reward": int(trajectory["official_success"]), **reward,
                "group_reward_mode": scored["reward_mode"], "reward_signature": scored["reward_signature"],
                "effective_reward_weights": dict(cfg["rewards"]), "learning_signal": trajectory["learning_signal"],
                "advantage": trajectory["advantage"], "advantage_bounds": trajectory["advantage_bounds"],
                "objective": scored["objective"], "normalization": scored["normalization"],
                "group_missingness": scored["missingness"],
                "group_diagnostics": scored["diagnostics"], "selected_for_update": selected,
                "exclusion_reason": exclusion})
        ledger.record_group(batch=batch, group_id=group["id"], task_id=group["selection"]["id"],
                            policy_version=policy_version, selection=group["selection"], scored=scored["scored"],
                            advantages=scored["advantages"], group_stats=statistics, results=results, calls=calls,
                            selected_for_update=selected, exclusion_reason=exclusion,
                            reward_mode=scored["reward_mode"], reward_signature=scored["reward_signature"],
                            diagnostics=scored["diagnostics"], objective=scored["objective"],
                            normalization=scored["normalization"], missingness=scored["missingness"],
                            learning_signal=scored["learning_signal"],
                            **({"collection_evidence": collection_evidence} if group.get("slot_failure_policy") else {}))
        group["status"] = "committed"
        group["selected"] = selected
        state["completed_groups"] += 1
        state["partial_groups"] += int(len(slots) < len(group["slots"]))
        state["candidate_successes"] += sum(outcomes)
        state["candidate_trajectories"] += len(outcomes)
        if selected:
            accepted.append(trajectories)
            state["accepted_group_ids"].append(group["id"])
        emit_fn(metrics, "group_verified", batch=batch, group_id=group["id"], **statistics,
                rewards=scored["rewards"], advantages=scored["advantages"],
                learning_signal=scored["learning_signal"], advantage_bounds=scored["advantage_bounds"],
                reward_mode=scored["reward_mode"], reward_diagnostics=scored["diagnostics"],
                reward_objective=scored["objective"], reward_normalization=scored["normalization"],
                reward_missingness=scored["missingness"],
                collection_evidence=collection_evidence,
                selected_for_update=selected, exclusion_reason=exclusion)
        atomic_json(run_dir / "curriculum_state.json", sampler.state_dict())
        endpoint.release_rollouts([slot["identity"] for slot in slots])

    def pending_groups():
        return [group for group in state["groups"] if group["status"] not in ("committed", "failed", "deferred")]

    def collection_time_left():
        return state.get("stop_reason") != "update_reserve" and time.monotonic() < collection_deadline

    def select_groups():
        changed = False
        while (len(pending_groups()) < min(prefetch, target - len(accepted))
               and len(state["groups"]) < max_candidates and not state["catalog_exhausted"]
               and session_failed_groups < failure_limit and collection_time_left()):
            availability = sampler.selection_distribution(exclude_ids=state["attempted"], used_repos=state["used_repos"])
            selection = sampler.choose(exclude_ids=state["attempted"], used_repos=state["used_repos"])
            if selection is None:
                state["catalog_exhausted"] = True
                state["cooldown_blocked"] = availability.get("cooldown_blocked", False)
                state["next_retry_unix"] = availability.get("next_retry_unix")
                break
            index = len(state["groups"])
            identifier = f"batch-{batch:04d}-group-{index:03d}"
            state["attempted"].append(selection["id"])
            state["used_repos"].append(selection["repo"])
            group = {"id": identifier, "selection": selection, "status": "pending", "selected": False,
                     "slot_failure_policy": "retain_verified_min_two_v1",
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
        while queue and len(running) < workers and collection_time_left():
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
                       "--event-summarizer", cfg["runtime"].get("event_summarizer", "official"),
                       "--queue-timing-poll-seconds", str(cfg["runtime"].get("queue_timing_poll_seconds", 0.2)),
                       "--evaluation-overrides-json", json.dumps(cfg["runtime"].get("evaluation", {})),
                       "--task-id", group["selection"]["id"]]
            if sandbox_root is not None:
                command.extend(["--sandbox-root", sandbox_root])
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
        slot["status"] = "exhausted"
        group.setdefault("slot_failure_policy", "retain_verified_min_two_v1")
        if not group.get("infrastructure_failed"):
            group["infrastructure_failed"] = True
            state["infrastructure_failed_groups"] += 1
            session_failed_groups += 1
        emit_fn(metrics, "rollout_infrastructure_exhausted", batch=batch, group_id=group["id"],
                rollout_id=identity, complete_trajectories=len(valid_slots(group)),
                observed_as_reward=False, error=slot["last_error"])

    def harvest(done, *, budget_stop=False):
        # Completed futures may exist behind an unfinished earlier group. Keep
        # their validated results even when no more work can be dispatched.
        ordered = sorted(done, key=lambda future: future.cancelled() or future.exception() is None)
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
                if result.get("task_id") != group["selection"]["id"] or result.get("rollout_id") != identity:
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
            except (WorkerAttemptError, concurrent.futures.CancelledError) as error:
                check_run()
                if budget_stop:
                    # This is an intentional budget cancellation, not a scored
                    # failure or an infrastructure retry. Completed slots stay.
                    slot["status"] = "pending"
                else:
                    fail_attempt(group, slot, identity, error)

    def commit_ready_groups():
        while state["commit_cursor"] < len(state["groups"]):
            group = state["groups"][state["commit_cursor"]]
            if group["status"] in ("failed", "committed", "deferred"):
                state["commit_cursor"] += 1
                continue
            if not terminal_group(group):
                break
            if len(valid_slots(group)) >= 2:
                commit_group(group)
            else:
                record_health(group)
                group.update(status="failed", failure_reason="fewer_than_two_verified_trajectories")
                state["failed_groups"] += 1
                emit_fn(metrics, "group_insufficient_trajectories", batch=batch, group_id=group["id"],
                        complete_trajectories=len(valid_slots(group)), observed_as_reward=False)
            state["commit_cursor"] += 1
            persist()

    def handoff_for_update():
        # Persist intent before cancellation: a restart must finish this handoff
        # instead of starting new rollouts with the saved behavior policy.
        state.update(stop_reason="update_reserve", budget_handoff=True)
        persist()
        harvest([future for future in running if future.done()], budget_stop=True)
        identities = []
        for future, (_, _, identity) in list(running.items()):
            cancel_events[identity].set()
            future.cancel()
            identities.append(identity)
        endpoint.cancel_rollouts(identities, reason="update_budget_reserve")
        # Worker cancellation is bounded by run_worker's process-tree shutdown;
        # do not set expired/abort_generation, since the learner still needs GPU.
        while running:
            check_run()
            done, _ = concurrent.futures.wait(running, timeout=min(0.2, max(0, deadline - time.monotonic())),
                                              return_when=concurrent.futures.FIRST_COMPLETED)
            harvest(done, budget_stop=True)
        deferred = [group for group in state["groups"] if group["status"] == "deferred"]
        for group in pending_groups():
            if not terminal_group(group):
                record_health(group)
                group["status"] = "deferred"
                deferred.append(group)
        # Skipping deferred groups unblocks complete groups later in selection
        # order. Only fully verified trajectories can reach rewards/training.
        commit_ready_groups()
        state["deferred_groups"] = len(deferred)
        state["complete"] = bool(accepted)
        if not accepted:
            # No update will occur. Preserve partial groups for a later budgeted
            # continuation instead of committing an empty batch or losing slots.
            for group in deferred:
                group["status"] = "pending"
            state["commit_cursor"] = 0
            state["stop_reason"] = None
        persist()
        emit_fn(metrics, "collection_budget_handoff", batch=batch, accepted_groups=len(accepted),
                deferred_groups=len(deferred), update_reserve_seconds=reserve,
                remaining_seconds=max(0.0, deadline - time.monotonic()),
                resume_collection=not bool(accepted))

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
            harvest([future for future in running if future.done()],
                    budget_stop=state.get("stop_reason") == "update_reserve")
            commit_ready_groups()
            if not collection_time_left():
                handoff_for_update()
                break
            select_groups()
            fill_slots()
            if not collection_time_left():
                continue
            if not pending_groups() and not running:
                if not accepted:
                    # All tasks may have failed in this very collection. The
                    # per-round exclusion set hides their cooldowns from
                    # select_groups; inspect global availability before closing
                    # an empty batch, including when the candidate cap was hit.
                    availability = sampler.selection_distribution()
                    if availability.get("cooldown_blocked"):
                        state["cooldown_blocked"] = True
                        state["next_retry_unix"] = availability.get("next_retry_unix")
                state["complete"] = bool(accepted) or not state.get("cooldown_blocked", False)
                persist()
                break
            if not running:
                if not collection_time_left():
                    continue
                raise RuntimeError("An incomplete group has no runnable workers")
            done, _ = concurrent.futures.wait(running, timeout=min(0.2, max(0, collection_deadline - time.monotonic())),
                                              return_when=concurrent.futures.FIRST_COMPLETED)
            harvest(done)
            if done:
                persist()
        return {"accepted": accepted, "attempted": set(state["attempted"]),
                "completed_groups": state["completed_groups"], "failed_groups": state["failed_groups"],
                "partial_groups": state["partial_groups"],
                "infrastructure_failed_groups": state["infrastructure_failed_groups"],
                "cooldown_blocked": state.get("cooldown_blocked", False),
                "next_retry_unix": state.get("next_retry_unix"),
                "elapsed_seconds": state["elapsed_seconds"] if state["timing_from_start"] else None,
                "observed_elapsed_seconds": state["elapsed_seconds"],
                "candidate_successes": state["candidate_successes"],
                "candidate_trajectories": state["candidate_trajectories"],
                "budget_handoff": bool(state.get("budget_handoff")),
                "deferred_groups": state.get("deferred_groups", 0),
                "update_reserve_seconds": reserve,
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
