"""Compact, checkpoint-backed training evidence; never infer post-update gains."""
from __future__ import annotations

import copy
import math
from pathlib import Path

from .recovery import atomic_json, value_digest


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _observed_sum(values):
    values = list(values)
    observed = [value for value in values if value is not None]
    return {"total": math.fsum(observed) if observed and len(observed) == len(values) else None,
            "observed_total": math.fsum(observed) if observed else None,
            "observed_batches": len(observed), "missing_batches": len(values) - len(observed)}


def make_batch_evidence(*, batch, policy_before, policy_after, collection, trajectories,
                        stats, adapter_path, adapter_sha256, adapter_changed, reward_evidence=None,
                        initial_adapter_sha256=None):
    """Keep numerical diagnostics and their semantics, never trajectory payloads."""
    completed = collection["completed_groups"]
    accepted = collection["accepted_groups"]
    attempted = len(collection["attempted_tasks"])
    steps = []
    for raw in (stats or {}).get("steps", (stats or {}).get("updates", [])):
        # The trainer's step history contains numerical metrics, not trajectories.
        steps.append(dict(raw))
    attempts = (stats or {}).get("gradient_attempts", [])
    return {
        "batch": batch, "policy_version_before": policy_before, "policy_version_after": policy_after,
        "rewards": copy.deepcopy(reward_evidence),
        "collection": {
            "completed_groups": completed, "failed_groups": collection["failed_groups"],
            "attempted_groups": attempted, "effective_advantage_groups": accepted,
            "effective_advantage_group_fraction": _ratio(accepted, completed),
            "effective_advantage_yield": _ratio(accepted, attempted),
            "elapsed_seconds": collection.get("elapsed_seconds"),
            "observed_elapsed_seconds": collection.get("observed_elapsed_seconds"),
            "budget_handoff": collection.get("budget_handoff", False),
            "deferred_groups": collection.get("deferred_groups", 0),
            "update_reserve_seconds": collection.get("update_reserve_seconds"),
            "partial_groups": collection.get("partial_groups"),
            "infrastructure_failed_groups": collection.get("infrastructure_failed_groups"),
            "cooldown_blocked": collection.get("cooldown_blocked"),
            "next_retry_unix": collection.get("next_retry_unix"),
        },
        "updates": {
            "optimizer_steps": int((stats or {}).get("optimizer_steps", 0)),
            "nonzero_lr_optimizer_steps": ((stats or {}).get("nonzero_lr_optimizer_steps") if stats else 0),
            "zero_lr_optimizer_steps": ((stats or {}).get("zero_lr_optimizer_steps") if stats else 0),
            "skipped_optimizer_steps": ((stats or {}).get("skipped_optimizer_steps") if stats else 0),
            "lr_schedule": copy.deepcopy((stats or {}).get("lr_schedule")),
            "update_wall_seconds": ((stats or {}).get("update_wall_seconds") if stats else 0.0),
            "observed_update_wall_seconds": ((stats or {}).get("observed_update_wall_seconds",
                (stats or {}).get("update_wall_seconds")) if stats else 0.0),
            "steps": steps,
            "post_update_kl": (stats or {}).get("post_update_kl"),
            "probability_correction": copy.deepcopy((stats or {}).get("probability_correction")),
            "training_prefix_reuse": copy.deepcopy((stats or {}).get("training_prefix_reuse")),
            "forward_invocations": copy.deepcopy((stats or {}).get("forward_invocations")),
            "reuse_stop_reason": (stats or {}).get("reuse_stop_reason"),
            # A rejected attempt has no optimizer-step row. Its partial KL
            # lower bound and coverage must survive without counting an update.
            "last_reuse_attempt": copy.deepcopy(attempts[-1]) if attempts else None,
        },
        "rollout_pre_update": {
            "candidate_successes": collection["candidate_successes"],
            "candidate_trajectories": collection["candidate_trajectories"],
            "candidate_official_success_rate": _ratio(collection["candidate_successes"], collection["candidate_trajectories"]),
            "selected_successes": sum(item["official_success"] for item in trajectories),
            "selected_trajectories": len(trajectories),
        },
        "adapter": {"batch": batch, "path": str(adapter_path) if adapter_path is not None else None,
                    "sha256": adapter_sha256, "changed": adapter_changed,
                    "initial_sha256": initial_adapter_sha256,
                    "changed_from_initial": (adapter_sha256 != initial_adapter_sha256
                        if adapter_sha256 is not None and initial_adapter_sha256 is not None else None)},
    }


def training_summary(cfg, batches, *, checkpoint_id, committed_batch_count=None, committed_optimizer_steps=None):
    """Aggregate numerators/denominators, not per-batch means of unequal groups."""
    rows = sorted(batches.values(), key=lambda row: row["batch"])
    collection = {name: sum(row["collection"][name] for row in rows) for name in (
        "completed_groups", "failed_groups", "attempted_groups", "effective_advantage_groups")}
    collection["effective_advantage_group_fraction"] = _ratio(
        collection["effective_advantage_groups"], collection["completed_groups"])
    collection["effective_advantage_yield"] = _ratio(
        collection["effective_advantage_groups"], collection["attempted_groups"])
    elapsed = _observed_sum(row["collection"]["elapsed_seconds"] for row in rows)
    collection.update(elapsed_seconds=elapsed["total"], time_coverage=elapsed)
    collection["effective_groups_per_collection_hour"] = (
        3600 * collection["effective_advantage_groups"] / elapsed["total"] if elapsed["total"] else None)
    steps = [{**step, "batch": row["batch"]} for row in rows for step in row["updates"]["steps"]]
    wall = _observed_sum(row["updates"]["update_wall_seconds"] for row in rows)
    nonzero = [row["updates"]["nonzero_lr_optimizer_steps"] for row in rows]
    optimizer_steps = sum(row["updates"]["optimizer_steps"] for row in rows)
    completed_batches = len(rows) if committed_batch_count is None else committed_batch_count
    updates = {
        "optimizer_steps": optimizer_steps,
        "checkpoint_optimizer_steps": committed_optimizer_steps,
        "optimizer_steps_outside_reported_batches": (
            committed_optimizer_steps - optimizer_steps if committed_optimizer_steps is not None else None),
        "nonzero_lr_optimizer_steps": sum(nonzero) if all(value is not None for value in nonzero) else None,
        "zero_lr_optimizer_steps": (sum(row["updates"]["zero_lr_optimizer_steps"] for row in rows)
            if all(row["updates"].get("zero_lr_optimizer_steps") is not None for row in rows) else None),
        "skipped_optimizer_steps": (sum(row["updates"]["skipped_optimizer_steps"] for row in rows)
            if all(row["updates"].get("skipped_optimizer_steps") is not None for row in rows) else None),
        "update_wall_seconds": wall["total"], "time_coverage": wall,
        "seconds_per_optimizer_step": _ratio(wall["total"], optimizer_steps) if wall["total"] is not None else None,
        "seconds_per_nonzero_lr_step": (_ratio(wall["total"], sum(nonzero))
            if wall["total"] is not None and all(value is not None for value in nonzero) else None),
        "lr_schedule": rows[-1]["updates"].get("lr_schedule") if rows else None,
        "steps": steps,
        "post_update_kl": rows[-1]["updates"].get("post_update_kl") if rows else None,
        "probability_correction": rows[-1]["updates"].get("probability_correction") if rows else None,
        "training_prefix_reuse": rows[-1]["updates"].get("training_prefix_reuse") if rows else None,
        "reuse_stop_evidence": [{
            "batch": row["batch"],
            "recorded": "reuse_stop_reason" in row["updates"],
            "reuse_stop_reason": row["updates"].get("reuse_stop_reason"),
            "last_reuse_attempt": row["updates"].get("last_reuse_attempt"),
        } for row in rows],
    }
    # A KL observation is at the weights *before* its optimizer step. In a
    # two-reuse cohort, the second observation measures the first update's drift;
    # the separate bounded probe reports final weights on a fixed saved subset.
    # Historical unweighted diagnostics must not become observations of the
    # corrected estimator merely because their field names are unchanged.
    by_version = {}
    for step in steps:
        version = step.get("correction_version") or "legacy_unspecified"
        by_version.setdefault(version, []).append(step)
    kl = {"by_correction_version": {}}
    for version, version_steps in sorted(by_version.items()):
        group = {"optimizer_step_observations": len(version_steps)}
        for key, old_key, estimator_key in (
                ("reference_kl_pre_update", "kl", "reference_kl_estimator"),
                ("old_policy_kl_pre_update", "approx_old_policy_kl", "old_policy_kl_estimator")):
            observed_steps = [step for step in version_steps if step.get(key, step.get(old_key)) is not None]
            observed = [step.get(key, step.get(old_key)) for step in observed_steps]
            group[key] = {
                "first": observed[0] if observed else None,
                "last": observed[-1] if observed else None,
                "max": max(observed) if observed else None,
                "observations": len(observed),
                "estimators": sorted({step.get(estimator_key) or "legacy_unspecified" for step in observed_steps}),
                "scopes": sorted({step.get("kl_scope") or "legacy_unspecified" for step in observed_steps}),
            }
        kl["by_correction_version"][version] = group
    kl["scope"] = ("pre_optimizer_gradient_forward; grouped by recorded correction_version; "
                   "cohorts have different old policies and trajectories; discarded attempts are excluded")
    kl["post_update_kl"] = updates["post_update_kl"]
    pre = {name: sum(row["rollout_pre_update"][name] for row in rows) for name in (
        "candidate_successes", "candidate_trajectories", "selected_successes", "selected_trajectories")}
    pre["candidate_official_success_rate"] = _ratio(pre["candidate_successes"], pre["candidate_trajectories"])
    pre["selected_official_success_rate"] = _ratio(pre["selected_successes"], pre["selected_trajectories"])
    return {
        "schema_version": 1, "record_type": "training_evidence", "checkpoint_id": checkpoint_id,
        "config_sha256": value_digest(cfg), "committed_batches": len(rows),
        "history_coverage": {
            "completed_batches": completed_batches, "reported_batches": len(rows),
            "unreported_completed_batches": completed_batches - len(rows),
            "reported_batch_ids": [row["batch"] for row in rows],
            "scope": "All aggregate rates and times cover reported completed batches only; checkpoint count can also include an active batch.",
        },
        "collection": collection, "updates": updates, "kl": kl, "rollout_pre_update": pre,
        "rewards": {
            "scope": "Committed completed batches; candidate and selected/optimizer exposures remain distinct",
            "by_batch": {str(row["batch"]): row.get("rewards") for row in rows},
            "observed_batches": sum(row.get("rewards") is not None for row in rows),
            "unknown_batches": sum(row.get("rewards") is None for row in rows),
            "interpretation": "Missingness partitions and realized task/advantage exposure are descriptive, not uniform-risk gradients or adapter gains; group/none requires a separate single-factor experiment.",
        },
        "adapters": [row["adapter"] for row in rows if row["adapter"]["path"]],
        "post_update_evaluation": None, "batches": rows,
        "interpretation": {
            "population": "Completed training batches only; rollout scores precede their updates and are not adapter gains.",
            "effective_groups": "Selected groups / complete valid candidate groups; yield also includes infrastructure-exhausted groups in its denominator.",
            "timing": "Recorded active wall time; excludes downtime and uncommitted crash tails. Collection timing ends at its last snapshot; update timing includes trainer checkpoints, not adapter export.",
            "nonzero_lr_steps": "Non-skipped optimizer calls with nonzero LR; adapter hashes separately establish actual weight change.",
            "probability_correction": "The summary retains the last completed batch's correction diagnostics; each step preserves its own estimator and correction version. Missing versions remain legacy_unspecified and are never pooled with corrected KL observations.",
            "reuse_stops": "Per-batch last attempts include discarded work without adding optimizer steps. A partial old_policy_kl_lower_bound with diagnostic_tokens_scored is a stopping certificate, not a complete-cohort KL observation; missing historical stop records remain unknown.",
            "evaluation": "Use fixed-task/seed paired official acceptance after exporting the adapter; missing evidence stays null.",
        },
    }


def write_training_summary(run_dir, cfg, batches, *, checkpoint_id,
                           committed_batch_count=None, committed_optimizer_steps=None):
    # No empty report at startup. This is a view of the committed run state,
    # rebuilt on resume so append-only event duplication cannot inflate results.
    path = Path(run_dir) / "training_summary.json"
    if batches or committed_batch_count:
        atomic_json(path, training_summary(cfg, batches, checkpoint_id=checkpoint_id,
            committed_batch_count=committed_batch_count, committed_optimizer_steps=committed_optimizer_steps))
    elif path.exists():
        path.unlink()
