"""Compact, checkpoint-backed training evidence; never infer post-update gains."""
from __future__ import annotations

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
                        stats, adapter_path, adapter_sha256, adapter_changed):
    """Only scalar evidence and identifiers enter the recoverable run state."""
    completed = collection["completed_groups"]
    accepted = collection["accepted_groups"]
    attempted = len(collection["attempted_tasks"])
    steps = []
    for raw in (stats or {}).get("steps", (stats or {}).get("updates", [])):
        # The trainer's step history contains numerical metrics, not trajectories.
        steps.append(dict(raw))
    return {
        "batch": batch, "policy_version_before": policy_before, "policy_version_after": policy_after,
        "collection": {
            "completed_groups": completed, "failed_groups": collection["failed_groups"],
            "attempted_groups": attempted, "effective_advantage_groups": accepted,
            "effective_advantage_group_fraction": _ratio(accepted, completed),
            "effective_advantage_yield": _ratio(accepted, attempted),
            "elapsed_seconds": collection.get("elapsed_seconds"),
            "observed_elapsed_seconds": collection.get("observed_elapsed_seconds"),
        },
        "updates": {
            "optimizer_steps": int((stats or {}).get("optimizer_steps", 0)),
            "nonzero_lr_optimizer_steps": ((stats or {}).get("nonzero_lr_optimizer_steps") if stats else 0),
            "update_wall_seconds": ((stats or {}).get("update_wall_seconds") if stats else 0.0),
            "observed_update_wall_seconds": ((stats or {}).get("observed_update_wall_seconds",
                (stats or {}).get("update_wall_seconds")) if stats else 0.0),
            "steps": steps,
            "post_update_kl": (stats or {}).get("post_update_kl"),
        },
        "rollout_pre_update": {
            "candidate_successes": collection["candidate_successes"],
            "candidate_trajectories": collection["candidate_trajectories"],
            "candidate_official_success_rate": _ratio(collection["candidate_successes"], collection["candidate_trajectories"]),
            "selected_successes": sum(item["official_success"] for item in trajectories),
            "selected_trajectories": len(trajectories),
        },
        "adapter": {"path": str(adapter_path) if adapter_path is not None else None,
                    "sha256": adapter_sha256, "changed": adapter_changed},
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
        "update_wall_seconds": wall["total"], "time_coverage": wall,
        "seconds_per_optimizer_step": _ratio(wall["total"], optimizer_steps) if wall["total"] is not None else None,
        "steps": steps,
        "post_update_kl": rows[-1]["updates"].get("post_update_kl") if rows else None,
    }
    # A KL observation is at the weights *before* its optimizer step. In a
    # two-reuse cohort, the second observation measures the first update's drift;
    # the separate bounded probe reports final weights on a fixed saved subset.
    kl = {}
    for key, old_key in (("reference_kl_pre_update", "kl"),
                         ("old_policy_kl_pre_update", "approx_old_policy_kl")):
        observed = [step.get(key, step.get(old_key)) for step in steps]
        observed = [value for value in observed if value is not None]
        kl[key] = {"first": observed[0] if observed else None,
                   "last": observed[-1] if observed else None,
                   "max": max(observed) if observed else None, "observations": len(observed)}
    kl["scope"] = "pre_optimizer_gradient_forward; cohorts have different old policies and trajectories"
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
        "adapters": [row["adapter"] for row in rows if row["adapter"]["path"]],
        "post_update_evaluation": None, "batches": rows,
        "interpretation": {
            "population": "Completed training batches only; rollout scores precede their updates and are not adapter gains.",
            "effective_groups": "Selected groups / complete valid candidate groups; yield also includes infrastructure-exhausted groups in its denominator.",
            "timing": "Recorded active wall time; excludes downtime and uncommitted crash tails. Collection timing ends at its last snapshot; update timing includes trainer checkpoints, not adapter export.",
            "nonzero_lr_steps": "Non-skipped optimizer calls with nonzero LR; adapter hashes separately establish actual weight change.",
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
