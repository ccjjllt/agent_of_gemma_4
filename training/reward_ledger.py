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
    "complete_evidence_zero_signal",
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
                    "point_reward_count", "missing_reward_count", "provable_pair_count"):
            if key in diagnostics:
                output[key] = _number(diagnostics[key], nonnegative=True)
        output["signal_source"] = _label(diagnostics.get("signal_source"))
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

    def record_group(self, *, batch, group_id, task_id, policy_version, selection,
                     scored, advantages, group_stats, results, calls, selected_for_update,
                     exclusion_reason, reward_mode, reward_signature, diagnostics=None):
        batch, policy_version = _integer(batch), _integer(policy_version)
        group_id, task_id = _identity(group_id), _identity(task_id)
        if not isinstance(selected_for_update, bool):
            raise ValueError("selected_for_update must be explicit")
        n = len(scored)
        if n < 2 or any(len(values) != n for values in (advantages, results, calls)):
            raise ValueError("Reward ledger requires aligned complete-group records")
        if not isinstance(reward_signature, str) or not reward_signature:
            raise ValueError("A reward signature is required")
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

    def mark_batch_update(self, batch, optimizer_steps, adapter_changed=None):
        batch, optimizer_steps = _integer(batch), _integer(optimizer_steps)
        if adapter_changed is not None and not isinstance(adapter_changed, bool):
            raise ValueError("adapter_changed must be a boolean or unknown")
        value = {"optimizer_steps": optimizer_steps, "adapter_changed": adapter_changed}
        previous = self.batch_updates.get(str(batch))
        if previous is not None and previous != value:
            if (previous["optimizer_steps"] != optimizer_steps
                    or previous.get("adapter_changed") is not None or adapter_changed is None):
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
            restored[str(batch_number)] = {"optimizer_steps": steps, "adapter_changed": changed}
        if restored != self.batch_updates:
            with (self.run_dir / "reward_update_recovery.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps({"checkpoint_id": checkpoint_id,
                                         "before": self.batch_updates, "restored": restored}) + "\n")
                output.flush()
                os.fsync(output.fileno())
            _sync_directory(self.run_dir)
        self.batch_updates = restored
        self._write_summary()

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
                "updates": "Selection is not completion. Confirmed update membership requires an acknowledged optimizer step; unconfirmed groups remain separate.",
                "sampling": "These are adaptively sampled training candidates, not an unbiased uniform-task evaluation.",
                "ablation": "Weight ablations reuse the same observed evidence; sensitivity is not training effectiveness or repair gain.",
                "missingness": "No provable signal with missing evidence does not establish that missingness caused zero signal.",
            },
            "batch_updates": dict(self.batch_updates),
            **self._partitions(self.groups), "by_reward_signature": by_signature,
        }
        _atomic_json(self.summary_path, summary)
        return summary
