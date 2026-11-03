"""Measured complete-group reward distributions and collection-cost accounting.

The ledger exports only scalar evidence and identifiers. Raw prompts, token IDs,
test names, XML, patches and model traces are never copied into these files.
Unknown rewards remain unknown; selection is distinct from an observed update.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile

from .recovery import atomic_json, _sync_directory


_COMPONENTS = ("resolved", "repair", "regression", "efficiency", "repetition")
_FLAGS = (
    "legacy_group_fallback_triggered", "legacy_all_failure_zero",
    "new_signal_recovered_vs_legacy", "no_provable_signal_with_missing",
    "complete_evidence_zero_signal", "would_have_fallback",
)
_LABEL = re.compile(r"[A-Za-z0-9_.:-]{1,160}")
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,255}")


def _number(value, *, nonnegative=False):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Reward diagnostics require finite numeric values or None")
    if nonnegative and value < 0:
        raise ValueError("Cost and count values must be nonnegative")
    return value


def _optional_cost(value):
    # Missing or invalid instrumentation is missing evidence, never a measured 0.
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def _label(value):
    if value is None:
        return None
    return value if isinstance(value, str) and _LABEL.fullmatch(value) else "unclassified"


def _identity(value):
    if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
        raise ValueError("Invalid reward-ledger identifier")
    return value


def _integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("Batch, policy and optimizer counters must be nonnegative integers")
    return value


def _bounds(value):
    if value is None:
        return None
    lower, upper = _number(value.get("lower")), _number(value.get("upper"))
    if lower is None or upper is None or lower > upper:
        raise ValueError("Reward bounds must be finite and ordered")
    return {"lower": lower, "upper": upper}


def _distribution(values):
    values = list(values)
    observed = sorted(float(value) for value in values if value is not None)
    n = len(observed)

    def percentile(q):
        if not n:
            return None
        position = (n - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return observed[low] + (observed[high] - observed[low]) * (position - low)

    # Non-overlapping fixed bins; exact scalar observations remain in the journal.
    edges = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0, 1.3, 2.0]
    counts = [0] * (len(edges) + 1)
    for value in observed:
        slot = next((index for index, edge in enumerate(edges) if value <= edge), len(edges))
        counts[slot] += 1
    return {
        "observed": n, "missing": len(values) - n,
        "min": observed[0] if n else None, "max": observed[-1] if n else None,
        "mean": statistics.fmean(observed) if n else None,
        "std": statistics.pstdev(observed) if n else None,
        "p05": percentile(0.05), "p50": percentile(0.5), "p95": percentile(0.95),
        "zero_count": sum(value == 0 for value in observed),
        "histogram": {"upper_edges": edges, "counts": counts,
                      "intervals": "(-infinity,edge0], (edge0,edge1], ..., (last_edge,infinity)"},
    }


def _cost_sum(values):
    values = list(values)
    observed = [value for value in values if value is not None]
    return {"observed_sum": math.fsum(observed) if observed else None,
            "observed": len(observed), "missing": len(values) - len(observed)}


def _atomic_json(path, payload):
    atomic_json(path, payload)


def _atomic_bytes(path, payload):
    """Preserve exact journal bytes, including an undecodable interrupted tail."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=".reward-journal-", delete=False) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RewardLedger:
    """One ledger per run, containing only complete, valid candidate groups.

    Group records are append-only. A reconstructed instance reloads their scalar
    records and previously acknowledged batch updates from the atomic summary.
    Interrupted/infrastructure-failed groups are outside this ledger's cost scope.
    """

    def __init__(self, run_dir, *, recover_incomplete_tail=False):
        """Load the ledger; tail recovery requires the caller's exclusive run_lock.

        The default never repairs a journal that another writer may still own.
        Recovery preserves the exact incomplete bytes before replacing the file
        with its fully validated newline-terminated prefix.
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.group_path = self.run_dir / "reward_groups.jsonl"
        self.summary_path = self.run_dir / "reward_summary.json"
        self.groups = []
        self.batch_updates = {}
        self._group_ids = set()
        journal = prefix = tail = b""
        if self.group_path.exists():
            journal = self.group_path.read_bytes()
            committed_end = journal.rfind(b"\n") + 1
            prefix, tail = journal[:committed_end], journal[committed_end:]
            # Validate every complete row before any recovery mutation. A bad
            # interior record or duplicate is never discarded as a partial tail.
            for line in prefix.splitlines(keepends=True):
                row = json.loads(line)
                if (not isinstance(row, dict) or row.get("schema_version") != 1
                        or row.get("record_type") != "complete_group"):
                    raise ValueError("Unsupported reward-journal record")
                if row["group_id"] in self._group_ids:
                    raise ValueError("Reward journal contains a duplicate group")
                self._group_ids.add(row["group_id"])
                self.groups.append(row)
            if tail and not recover_incomplete_tail:
                raise ValueError("Reward journal has an incomplete tail; exclusive run recovery is required")
        if self.summary_path.exists():
            previous = json.loads(self.summary_path.read_text())
            if previous.get("schema_version") != 1:
                raise ValueError("Unsupported reward-summary schema")
            self.batch_updates = dict(previous.get("batch_updates") or {})
        if tail:
            evidence = self.group_path.with_name(
                "reward_groups.incomplete-" + hashlib.sha256(journal).hexdigest() + ".tail"
            )
            if evidence.exists() and evidence.read_bytes() != tail:
                raise ValueError("Interrupted journal tail conflicts with preserved recovery evidence")
            _atomic_bytes(evidence, tail)
            # run_lock owns the interval between this check and replacement.
            # Reject an unexpected concurrent modification instead of truncating it.
            if self.group_path.read_bytes() != journal:
                raise RuntimeError("Reward journal changed during exclusive recovery")
            _atomic_bytes(self.group_path, prefix)
        self._write_summary()

    @staticmethod
    def _selection(selection):
        output = {}
        for key in ("selection_probability", "eligible_count", "candidate_count", "draw", "feedback_revision",
                    "observed_candidate_count", "unseen_candidate_count", "adaptive_candidate_count",
                    "suppressed_zero_signal_count", "observed_probability_mass", "unseen_probability_mass",
                    "policy_version", "last_observed_policy_version", "feedback_policy_age"):
            if key in selection:
                output[key] = _number(selection[key], nonnegative=True)
        for key in ("selection_phase", "selected_pool", "feedback_status"):
            if key in selection:
                output[key] = _label(selection[key])
        for key in ("selection_probability_components", "effective_pool_masses"):
            if key in selection:
                output[key] = {name: _number(selection[key].get(name), nonnegative=True)
                               for name in ("uniform", "coverage", "adaptive")}
        return output

    @staticmethod
    def _evidence(evidence):
        output = {"status": _label(evidence.get("status")), "reason": _label(evidence.get("reason"))}
        for key in ("fraction", "lower", "upper", "expected", "passed", "nonpassed", "missing", "observed", "weight"):
            if key in evidence:
                output[key] = _number(evidence[key])
        for key in ("applicable", "coverage_complete", "available"):
            if key in evidence and isinstance(evidence[key], bool):
                output[key] = evidence[key]
        if "fraction_bounds" in evidence:
            output["fraction_bounds"] = _bounds(evidence["fraction_bounds"])
        if isinstance(evidence.get("counts"), dict):
            output["counts"] = {key: _number(evidence["counts"].get(key), nonnegative=True)
                                for key in ("expected", "passed", "nonpassed", "missing")}
        return output

    @staticmethod
    def _diagnostics(diagnostics):
        diagnostics = diagnostics or {}
        output = {key: diagnostics.get(key) if isinstance(diagnostics.get(key), bool) else None for key in _FLAGS}
        for key in ("learning_signal_rms", "strict_signal_rms", "common_signal_rms", "legacy_signal_rms",
                    "nonzero_advantages", "reward_std", "selection_to_commit_seconds", "group_size",
                    "point_reward_count", "missing_reward_count", "provable_pair_count",
                    "normalizer", "std_upper_bound", "advantage_rms"):
            if key in diagnostics:
                output[key] = _number(diagnostics[key], nonnegative=True)
        output["signal_source"] = _label(diagnostics.get("signal_source"))
        output["missing_mode"] = _label(diagnostics.get("missing_mode"))
        output["common_components"] = [name for name in diagnostics.get("common_components", []) if name in _COMPONENTS]
        output["ablation"] = {}
        for name, value in (diagnostics.get("ablation") or {}).items():
            label = _label(name)
            if label == "unclassified" or not isinstance(value, dict):
                continue
            output["ablation"][label] = {
                "learning_signal_rms": _number(value.get("learning_signal_rms"), nonnegative=True),
                "nonzero_advantages": _number(value.get("nonzero_advantages", value.get("nonzero_count")), nonnegative=True),
                "would_update": value.get("would_update") if isinstance(value.get("would_update"), bool) else None,
                "signal_source": _label(value.get("signal_source")),
            }
        return output

    @staticmethod
    def _objective(value):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("Reward objective must be a scalar declaration")
        return {"algorithm": _label(value.get("algorithm")),
                "weights": {name: _number((value.get("weights") or {}).get(name), nonnegative=True)
                            for name in _COMPONENTS},
                "success_token_budget": _number(value.get("success_token_budget"), nonnegative=True),
                "dense_shaping_enabled": (value.get("dense_shaping_enabled")
                    if isinstance(value.get("dense_shaping_enabled"), bool) else None)}

    @staticmethod
    def _normalization(value):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("Reward normalization must be a scalar declaration")
        signature = value.get("normalization_signature")
        if signature is not None:
            decoded = json.loads(signature)
            allowed = {"algorithm", "scale_rewards", "advantage_std_floor", "min_group_std"}
            if not isinstance(decoded, dict) or set(decoded) - allowed:
                raise ValueError("Unexpected normalization signature fields")
            decoded = {key: _label(item) if key in {"algorithm", "scale_rewards"}
                       else _number(item, nonnegative=True) for key, item in decoded.items()}
            signature = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
        return {"mode": _label(value.get("mode")), "algorithm": _label(value.get("algorithm")),
                **{key: _number(value.get(key), nonnegative=True)
                   for key in ("denominator", "std_upper_bound", "multiplier")},
                "normalization_signature": signature}

    @staticmethod
    def _missingness(value):
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("Reward missingness must be a scalar declaration")
        patterns = {}
        for pattern, count in (value.get("trajectory_patterns") or {}).items():
            parts = [part.split(":", 1) for part in pattern.split("|")]
            if (any(len(part) != 2 or part[0] not in _COMPONENTS or part[1] not in
                    {"complete", "bounded_partial", "unavailable", "not_applicable", "disabled"}
                    for part in parts) or len({part[0] for part in parts}) != len(parts)):
                raise ValueError("Unexpected component missingness pattern")
            patterns[pattern] = _integer(count)
        reasons = {}
        for reason, count in (value.get("reason_counts") or {}).items():
            reason = _label(reason)
            reasons[reason] = reasons.get(reason, 0) + _integer(count)
        return {"trajectory_patterns": patterns,
                "missing_component_counts": {name: _integer(count) for name, count in
                    (value.get("missing_component_counts") or {}).items() if name in _COMPONENTS},
                "reason_counts": reasons, "group_pattern": _label(value.get("group_pattern")),
                **{key: _number(value.get(key), nonnegative=True)
                   for key in ("uncertainty_width_mean", "uncertainty_width_max")}}

    def record_group(self, *, batch, group_id, task_id, policy_version, selection,
                     scored, advantages, group_stats, results, calls, selected_for_update,
                     exclusion_reason, reward_mode, reward_signature, diagnostics=None,
                     objective=None, normalization=None, missingness=None, learning_signal=None):
        batch, policy_version = _integer(batch), _integer(policy_version)
        group_id, task_id = _identity(group_id), _identity(task_id)
        if not isinstance(selected_for_update, bool):
            raise ValueError("selected_for_update must be explicit")
        n = len(scored)
        if n < 2 or any(len(values) != n for values in (advantages, results, calls)):
            raise ValueError("Reward ledger requires aligned complete-group records")
        if not isinstance(reward_signature, str) or not reward_signature:
            raise ValueError("A reward signature is required")
        if learning_signal is not None and len(learning_signal) != n:
            raise ValueError("Learning signals must align with complete-group trajectories")
        objective = self._objective(objective)
        normalization = self._normalization(normalization)
        missingness = self._missingness(missingness)
        trajectories = []
        for index, (score, advantage, result, invocation) in enumerate(zip(scored, advantages, results, calls)):
            if (result.get("verdict_valid") is not True or result.get("verification_complete") is not True
                    or result.get("infrastructure_error") or not isinstance(result.get("success"), bool)):
                raise ValueError("Only complete, valid official verdicts enter reward distributions")
            if not invocation or any(not isinstance(call.get("completion_ids"), list) or not call["completion_ids"]
                                     for call in invocation):
                raise ValueError("Reward costs require the actual nonempty generated calls")
            features = result.get("reward_features") or {}
            generation_times = [_optional_cost(call.get("generation_elapsed_seconds")) for call in invocation]
            queue_times = [_optional_cost(call.get("queue_wait_seconds")) for call in invocation]
            trajectories.append({
                "index": index, "official_success": result["success"],
                "total": _number(score.get("total")), "reward_bounds": _bounds(score.get("reward_bounds")),
                "advantage": _number(advantage),
                "learning_signal": _number(learning_signal[index]) if learning_signal is not None else None,
                "components": {name: _number((score.get("components") or {}).get(name)) for name in _COMPONENTS},
                "component_bounds": {name: _bounds((score.get("component_bounds") or {}).get(name)) for name in _COMPONENTS},
                "component_evidence": {name: self._evidence((score.get("component_evidence") or {}).get(name) or {})
                                       for name in _COMPONENTS},
                "cost": {"generated_tokens": sum(len(call["completion_ids"]) for call in invocation),
                         "model_calls": len(invocation),
                         "worker_seconds": _optional_cost(result.get("duration_seconds")),
                         "verification_worker_seconds": _optional_cost(features.get("verification_duration_seconds")),
                         "generation_request_seconds": _cost_sum(generation_times),
                         "queue_request_seconds": _cost_sum(queue_times)},
            })
        statistics_row = {}
        for key in ("reward_mean", "reward_std", "learning_signal_rms", "learning_signal_second_moment", "official_success_rate",
                    "feedback_revision", "cumulative_groups", "cumulative_success_rate", "success_rate_ema"):
            if key in group_stats:
                statistics_row[key] = _number(group_stats[key])
        for key in ("informative", "reward_complete"):
            if isinstance(group_stats.get(key), bool):
                statistics_row[key] = group_stats[key]
        statistics_row["signal_status"] = _label(group_stats.get("signal_status"))
        row = {
            "schema_version": 1, "record_type": "complete_group", "batch": batch,
            "group_id": group_id, "task_id": task_id, "policy_version": policy_version,
            "selection": self._selection(selection), "trajectories": trajectories,
            "selected_for_update": selected_for_update, "exclusion_reason": _label(exclusion_reason),
            "reward_mode": _label(reward_mode),
            "reward_signature_sha256": hashlib.sha256(reward_signature.encode()).hexdigest(),
            "group_statistics": statistics_row, "diagnostics": self._diagnostics(diagnostics),
            "objective": objective, "normalization": normalization, "missingness": missingness,
        }
        payload = json.dumps(row, ensure_ascii=False, allow_nan=False)
        if group_id in self._group_ids:
            existing = next(item for item in self.groups if item["group_id"] == group_id)
            if existing != row:
                raise ValueError("Recovered group differs from its already committed reward evidence")
            return existing
        with self.group_path.open("a", encoding="utf-8") as output:
            output.write(payload + "\n")
            output.flush()
            os.fsync(output.fileno())
        _sync_directory(self.group_path.parent)
        self.groups.append(row)
        self._group_ids.add(group_id)
        self._write_summary()
        return row

    @staticmethod
    def _nonzero_lr_steps(value, optimizer_steps):
        if value is None:
            return None
        value = _integer(value)
        if value > optimizer_steps:
            raise ValueError("Nonzero-LR steps cannot exceed optimizer calls")
        return value

    def mark_batch_update(self, batch, optimizer_steps, adapter_changed=None, *,
                          nonzero_lr_optimizer_steps=None):
        batch, optimizer_steps = _integer(batch), _integer(optimizer_steps)
        nonzero_lr_optimizer_steps = self._nonzero_lr_steps(nonzero_lr_optimizer_steps, optimizer_steps)
        if adapter_changed is not None and not isinstance(adapter_changed, bool):
            raise ValueError("adapter_changed must be a boolean or unknown")
        value = {"optimizer_steps": optimizer_steps, "adapter_changed": adapter_changed,
                 "nonzero_lr_optimizer_steps": nonzero_lr_optimizer_steps}
        previous = self.batch_updates.get(str(batch))
        if previous is not None:
            if previous["optimizer_steps"] != optimizer_steps:
                raise ValueError("An acknowledged batch update cannot be silently rewritten")
            # Evidence can become known, but cannot be erased or changed by a
            # later acknowledgment. Missing historical fields remain unknown.
            for key in ("adapter_changed", "nonzero_lr_optimizer_steps"):
                if previous.get(key) is not None and previous[key] != value[key]:
                    raise ValueError("An acknowledged batch update cannot be silently rewritten")
        self.batch_updates[str(batch)] = value
        self._write_summary()

    def restore_updates(self, updates, checkpoint_id):
        """Reconcile optimizer membership to the loaded durable model state.

        A crash can occur after an in-memory optimizer update but before its
        checkpoint commits. Keep a recovery event instead of silently treating
        that lost state as part of the resumed learner.
        """
        restored = {}
        for batch, value in updates.items():
            batch_number = int(batch)
            _integer(batch_number)
            steps = _integer(value["optimizer_steps"])
            changed = value.get("adapter_changed")
            if changed is not None and not isinstance(changed, bool):
                raise ValueError("Invalid saved update membership")
            nonzero = self._nonzero_lr_steps(value.get("nonzero_lr_optimizer_steps"), steps)
            restored[str(batch_number)] = {"optimizer_steps": steps, "adapter_changed": changed,
                                          "nonzero_lr_optimizer_steps": nonzero}
        if restored != self.batch_updates:
            with (self.run_dir / "reward_update_recovery.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps({"checkpoint_id": checkpoint_id,
                                         "before": self.batch_updates, "restored": restored}) + "\n")
                output.flush()
                os.fsync(output.fileno())
            _sync_directory(self.run_dir)
        self.batch_updates = restored
        self._write_summary()

    @staticmethod
    def _moments(values):
        """Compact counts and moments, without repeated histogram arrays."""
        values = list(values)
        observed = [float(value) for value in values if value is not None]
        return {"observed": len(observed), "missing": len(values) - len(observed),
                "mean": statistics.fmean(observed) if observed else None,
                "min": min(observed) if observed else None,
                "max": max(observed) if observed else None}

    @staticmethod
    def _mass(values):
        values = list(values)
        observed = [value for value in values if value is not None]
        missing = len(values) - len(observed)
        return {"total": math.fsum(observed) if not missing else None,
                "observed_total": math.fsum(observed) if observed else None,
                "observed": len(observed), "missing": missing}

    @staticmethod
    def _component_coverage(groups, name):
        evidence = [row["component_evidence"].get(name, {})
                    for group in groups for row in group["trajectories"]]
        statuses = Counter(item.get("status") or "unreported" for item in evidence)
        observed = [item.get("fraction") if item.get("status") == "complete" else None
                    for item in evidence]
        known_applicable = sum((item.get("counts") or {}).get("expected", 0) > 0
                               or item.get("status") in {"complete", "bounded_partial"}
                               for item in evidence)
        return {"trajectories": len(evidence), "statuses": dict(statuses),
                "measured_rates": RewardLedger._moments(observed),
                "known_applicable_trajectories": known_applicable,
                "not_applicable_trajectories": statuses["not_applicable"],
                "disabled_trajectories": statuses["disabled"],
                "interpretation": "Rates require complete applicable evidence; disabled and not-applicable reward zeros are not observed zero rates."}

    def _weighting(self, groups):
        """Realized scalar exposure, not gradients or inverse-propensity claims."""
        rows = [row for group in groups for row in group["trajectories"]]
        raw, normalized, raw_tokens, normalized_tokens = [], [], [], []
        step_raw_tokens, step_normalized_tokens, step_groups = [], [], []
        nonzero_raw_tokens, nonzero_normalized_tokens, nonzero_groups = [], [], []
        for group in groups:
            update = self.batch_updates.get(str(group["batch"]))
            steps = (update.get("optimizer_steps") if update is not None else None) if group["selected_for_update"] else 0
            nonzero = (update.get("nonzero_lr_optimizer_steps") if update is not None else None) if group["selected_for_update"] else 0
            step_groups.append(steps)
            nonzero_groups.append(nonzero)
            for row in group["trajectories"]:
                signal, advantage = row.get("learning_signal"), row.get("advantage")
                token_count = row["cost"]["generated_tokens"]
                a, b = abs(signal) if signal is not None else None, abs(advantage) if advantage is not None else None
                raw.append(a)
                normalized.append(b)
                raw_tokens.append(a * token_count if a is not None else None)
                normalized_tokens.append(b * token_count if b is not None else None)
                step_raw_tokens.append(0.0 if steps == 0 else a * token_count * steps if a is not None and steps is not None else None)
                step_normalized_tokens.append(0.0 if steps == 0 else b * token_count * steps if b is not None and steps is not None else None)
                nonzero_raw_tokens.append(0.0 if nonzero == 0 else a * token_count * nonzero if a is not None and nonzero is not None else None)
                nonzero_normalized_tokens.append(0.0 if nonzero == 0 else b * token_count * nonzero if b is not None and nonzero is not None else None)
        return {
            "groups": len(groups), "trajectories": len(rows),
            "selection_probability": self._moments(group.get("selection", {}).get("selection_probability") for group in groups),
            "normalization_denominator": self._moments((group.get("normalization") or {}).get("denominator") for group in groups),
            "normalization_multiplier": self._moments((group.get("normalization") or {}).get("multiplier") for group in groups),
            "raw_signal_rms": self._moments(group.get("diagnostics", {}).get("learning_signal_rms") for group in groups),
            "normalized_advantage_rms": self._moments(group.get("diagnostics", {}).get("advantage_rms") for group in groups),
            "raw_signal_l1": self._mass(raw), "normalized_advantage_l1": self._mass(normalized),
            "raw_token_signal_l1": self._mass(raw_tokens),
            "normalized_token_advantage_l1": self._mass(normalized_tokens),
            "acknowledged_optimizer_group_exposures": self._mass(step_groups),
            "optimizer_call_weighted_raw_token_signal_l1": self._mass(step_raw_tokens),
            "optimizer_call_weighted_normalized_token_advantage_l1": self._mass(step_normalized_tokens),
            "acknowledged_nonzero_lr_group_exposures": self._mass(nonzero_groups),
            "nonzero_lr_step_weighted_raw_token_signal_l1": self._mass(nonzero_raw_tokens),
            "nonzero_lr_step_weighted_normalized_token_advantage_l1": self._mass(nonzero_normalized_tokens),
            "interpretation": "Absolute scalar/token exposure before PPO ratios, clipping, parameter Jacobians and trajectory/fixed-length loss denominators. Optimizer-call counts include zero-LR or skipped calls; nonzero-LR counts require a non-skipped call with a nonzero used LR. Neither proves parameter change, actual gradient magnitude, uniform-task correction, or repair gain; adapter_changed remains separate evidence.",
        }

    def _nonzero_lr_partitions(self, groups):
        selected = [group for group in groups if group["selected_for_update"]]
        partitions = {"confirmed_used_in_nonzero_lr_step": [],
                      "selected_confirmed_no_nonzero_lr_step": [],
                      "selected_effective_update_unknown": []}
        for group in selected:
            value = self.batch_updates.get(str(group["batch"]), {}).get("nonzero_lr_optimizer_steps")
            key = ("selected_effective_update_unknown" if value is None else
                   "confirmed_used_in_nonzero_lr_step" if value > 0 else
                   "selected_confirmed_no_nonzero_lr_step")
            partitions[key].append(group)
        return partitions

    def _compact_population(self, groups):
        rows = [row for group in groups for row in group["trajectories"]]
        return {"groups": len(groups), "trajectories": len(rows),
                "official_successes": sum(row["official_success"] for row in rows),
                "official_success_rate": sum(row["official_success"] for row in rows) / len(rows) if rows else None,
                "component_coverage": {name: self._component_coverage(groups, name) for name in ("repair", "regression")},
                "weighting": self._weighting(groups)}

    def _compact_partitions(self, groups):
        selected = [group for group in groups if group["selected_for_update"]]
        used = [group for group in selected if self.batch_updates.get(str(group["batch"]), {}).get("optimizer_steps", 0) > 0]
        unconfirmed = [group for group in selected if str(group["batch"]) not in self.batch_updates]
        return {"all_candidates": self._compact_population(groups),
                "selected": self._compact_population(selected),
                "confirmed_used_in_optimizer_step": self._compact_population(used),
                **{key: self._compact_population(items) for key, items in self._nonzero_lr_partitions(groups).items()},
                "selected_update_unconfirmed_groups": len(unconfirmed)}

    @staticmethod
    def _pattern_identity(group):
        missingness = group.get("missingness")
        if missingness is None:
            return "legacy_unreported", None
        # Include the exact component status/count map, not only the coarse
        # group class: repair-missing and regression-missing must stay separate.
        pattern = {"group_pattern": missingness.get("group_pattern"),
                   "trajectory_patterns": missingness.get("trajectory_patterns")}
        encoded = json.dumps(pattern, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest(), pattern

    def _reward_evidence(self, groups):
        patterns, objectives, normalizations = {}, {}, {}
        for group in groups:
            key, pattern = self._pattern_identity(group)
            patterns.setdefault(key, {"pattern": pattern, "groups": []})["groups"].append(group)
            objective = group.get("objective")
            key = (hashlib.sha256(json.dumps(objective, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                   if objective is not None else "legacy_unreported")
            entry = objectives.setdefault(key, {"objective": objective, "candidate_groups": 0, "selected_groups": 0})
            entry["candidate_groups"] += 1
            entry["selected_groups"] += int(group["selected_for_update"])
            normalization = group.get("normalization") or {}
            signature = normalization.get("normalization_signature")
            key = hashlib.sha256(signature.encode()).hexdigest() if signature is not None else "legacy_unreported"
            entry = normalizations.setdefault(key, {"normalization_signature": signature,
                "mode": normalization.get("mode"), "algorithm": normalization.get("algorithm"),
                "candidate_groups": 0, "selected_groups": 0})
            entry["candidate_groups"] += 1
            entry["selected_groups"] += int(group["selected_for_update"])
        return {
            **self._compact_partitions(groups),
            "objectives": objectives, "normalizations": normalizations,
            "by_missingness_pattern": {key: {"pattern": value["pattern"], **self._compact_partitions(value["groups"])}
                                      for key, value in sorted(patterns.items())},
            "by_task": {task_id: self._compact_partitions([group for group in groups if group["task_id"] == task_id])
                        for task_id in sorted({group["task_id"] for group in groups})},
            "interpretation": "Adaptive selection, signal filtering and group scaling jointly determine realized exposure. Reported probabilities are conditional draws; 1/q cannot undo acceptance filtering or recover a uniform-task objective. Missing historical declarations/signals remain unknown.",
        }

    def batch_evidence(self, batch):
        """Compact committed group evidence for one recoverable batch summary."""
        batch = _integer(batch)
        return {"schema_version": 1, "batch": batch,
                **self._reward_evidence([group for group in self.groups if group["batch"] == batch])}

    def _aggregate(self, groups):
        rows = [row for group in groups for row in group["trajectories"]]
        counters = {}
        for flag in _FLAGS:
            observations = [group["diagnostics"].get(flag) for group in groups]
            observed = sum(isinstance(value, bool) for value in observations)
            true_count = sum(value is True for value in observations)
            counters[flag] = {"true_groups": true_count, "observed_groups": observed,
                              "unknown_groups": len(groups) - observed,
                              "rate_among_observed": true_count / observed if observed else None}
        components = {}
        for name in _COMPONENTS:
            components[name] = {
                "values": _distribution(row["components"][name] for row in rows),
                "statuses": dict(Counter(row["component_evidence"][name].get("status") or "unreported" for row in rows)),
                "reasons": dict(Counter(row["component_evidence"][name]["reason"] for row in rows
                                        if row["component_evidence"][name].get("reason"))),
            }
        ablations = {}
        names = sorted({name for group in groups for name in group["diagnostics"].get("ablation", {})})
        for name in names:
            ablations[name] = {key: _distribution(group["diagnostics"].get("ablation", {}).get(name, {}).get(key)
                                                  for group in groups)
                               for key in ("learning_signal_rms", "nonzero_advantages")}
            observed = [group["diagnostics"].get("ablation", {}).get(name, {}) for group in groups]
            ablations[name]["would_update_groups"] = sum(value.get("would_update") is True for value in observed)
            ablations[name]["update_decision_observed_groups"] = sum(isinstance(value.get("would_update"), bool) for value in observed)
            ablations[name]["signal_sources"] = dict(Counter(value["signal_source"] for value in observed if value.get("signal_source")))
        lower = [row["reward_bounds"]["lower"] if row["reward_bounds"] is not None else None for row in rows]
        upper = [row["reward_bounds"]["upper"] if row["reward_bounds"] is not None else None for row in rows]
        cost = {"generated_tokens": sum(row["cost"]["generated_tokens"] for row in rows),
                "model_calls": sum(row["cost"]["model_calls"] for row in rows)}
        for name in ("worker_seconds", "verification_worker_seconds"):
            cost[name] = _cost_sum(row["cost"][name] for row in rows)
        for name in ("generation_request_seconds", "queue_request_seconds"):
            observed = sum(row["cost"][name]["observed"] for row in rows)
            cost[name] = {"observed_sum": (math.fsum(row["cost"][name]["observed_sum"] or 0.0 for row in rows)
                                           if observed else None),
                          "observed_calls": observed,
                          "missing_calls": sum(row["cost"][name]["missing"] for row in rows)}
        return {
            "groups": len(groups), "trajectories": len(rows),
            "official_successes": sum(row["official_success"] for row in rows),
            "official_success_rate": sum(row["official_success"] for row in rows) / len(rows) if rows else None,
            "point_rewards": _distribution(row["total"] for row in rows),
            "reward_lower_bounds": _distribution(lower), "reward_upper_bounds": _distribution(upper),
            "reward_interval_widths": _distribution(None if lo is None or hi is None else hi - lo
                                                       for lo, hi in zip(lower, upper)),
            "advantages": _distribution(row["advantage"] for row in rows),
            "learning_signal_rms": _distribution(group["group_statistics"].get("learning_signal_rms") for group in groups),
            "components": components, "diagnostic_rates": counters,
            "component_rate_coverage": {name: self._component_coverage(groups, name) for name in ("repair", "regression")},
            "offline_weight_ablation": ablations, "cost": cost,
        }

    def _partitions(self, groups):
        selected = [group for group in groups if group["selected_for_update"]]
        filtered = [group for group in groups if not group["selected_for_update"]]
        applied = [group for group in selected if self.batch_updates.get(str(group["batch"]), {}).get("optimizer_steps", 0) > 0]
        confirmed_no_update = [group for group in selected if str(group["batch"]) in self.batch_updates
                               and self.batch_updates[str(group["batch"])]["optimizer_steps"] == 0]
        unconfirmed = [group for group in selected if str(group["batch"]) not in self.batch_updates]
        return {"all_candidates": self._aggregate(groups), "selected": self._aggregate(selected),
                "filtered": self._aggregate(filtered), "confirmed_used_in_optimizer_step": self._aggregate(applied),
                **{key: self._aggregate(items) for key, items in self._nonzero_lr_partitions(groups).items()},
                "selected_confirmed_no_update": self._aggregate(confirmed_no_update),
                "selected_update_unconfirmed": self._aggregate(unconfirmed)}

    def _write_summary(self):
        signatures = sorted({group["reward_signature_sha256"] for group in self.groups})
        by_signature = {}
        for signature in signatures:
            groups = [group for group in self.groups if group["reward_signature_sha256"] == signature]
            by_signature[signature] = {"reward_modes": sorted({group["reward_mode"] for group in groups}),
                                       **self._partitions(groups)}
        summary = {
            "schema_version": 1,
            "scope": "Complete valid candidate groups only, including filtered groups; interrupted/infrastructure-failed groups are not included.",
            "interpretation": {
                "unknown_rewards": "Missing totals/components remain None and are excluded from observed-value statistics, not counted as zero.",
                "costs": "Worker, verification and request seconds can overlap. They are not GPU wall time or billing; use generation_metrics for decoder throughput.",
                "updates": "Selection is not completion. Optimizer-call membership is separate from non-skipped nonzero-LR membership. Zero-LR/skipped calls do not enter the latter; missing historical nonzero-LR counts remain unknown. A nonzero-LR call still does not prove changed parameters; adapter_changed records that separately.",
                "sampling": "These are adaptively sampled training candidates, not an unbiased uniform-task evaluation.",
                "ablation": "Weight ablations reuse the same observed evidence; sensitivity is not training effectiveness or repair gain.",
                "missingness": "No provable signal with missing evidence does not establish that missingness caused zero signal.",
            },
            "batch_updates": dict(self.batch_updates),
            **self._partitions(self.groups), "by_reward_signature": by_signature,
            "reward_evidence": self._reward_evidence(self.groups),
        }
        _atomic_json(self.summary_path, summary)
        return summary
