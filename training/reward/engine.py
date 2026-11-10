"""VerificationBundle -> RewardEngine -> RewardRecord.

The official runtime can call :func:`compute_reward_record` after loading its
private candidate node report.  Only aggregate rewards/evidence leave this API.
"""
from __future__ import annotations

from collections.abc import Mapping

from .baseline import EmptyPatchBaselineStore, TaskKey
from .critic import ProcessCritic
from .differential import pair_node_reports
from .progress import DEFAULT_EFFICIENCY_WEIGHT, DEFAULT_SCALES, composite_reward


REWARD_RECORD_FIELDS = (
    "resolved", "P", "G", "progress", "regression", "validity", "base_reward",
    "efficiency", "efficiency_weight", "total_reward", "eligible",
    "exclusion_reason", "objective_version",
)
_REPORT_KEYS = (
    "expected_universes", "node_statuses", "nodes", "artifact_hashes", "junit_available",
    "verification_complete", "verdict_valid", "infrastructure_error", "resolved", "success", "validity",
)
_METRIC_KEYS = ("gpu_seconds", "action_tokens", "tool_calls")


def _candidate_report(bundle: Mapping[str, object]) -> dict:
    report: dict = {}
    for name in ("candidate_report", "candidate", "candidate_summary"):
        value = bundle.get(name)
        if isinstance(value, Mapping):
            report.update(value)
    verdict = bundle.get("candidate_verdict")
    if isinstance(verdict, Mapping):
        for key, value in verdict.items():
            report.setdefault(key, value)
    elif type(verdict) is bool:
        report.setdefault("resolved", verdict)
    for key in _REPORT_KEYS:
        if key in bundle:
            report.setdefault(key, bundle[key])
    return report


def _resolved(bundle: Mapping[str, object], candidate: Mapping[str, object]) -> object:
    if "resolved" in candidate:
        return candidate["resolved"]
    if "success" in candidate:
        return candidate["success"]
    if type(bundle.get("candidate_verdict")) is bool:
        return bundle["candidate_verdict"]
    return bundle.get("resolved", bundle.get("success"))


def _action_tokens(calls: object) -> tuple[int | None, str | None]:
    if calls is None:
        return None, None
    if not isinstance(calls, (list, tuple)) or not calls:
        return None, "action_token_calls_missing_or_empty"
    total = 0
    for call in calls:
        if not isinstance(call, Mapping) or not isinstance(call.get("action_mask"), (list, tuple)):
            return None, "action_mask_missing"
        mask = call["action_mask"]
        if any(type(value) is not int or value not in (0, 1) for value in mask):
            return None, "action_mask_invalid"
        total += sum(mask)
    return total, None


def _metrics(bundle: Mapping[str, object], supplied: Mapping[str, object] | None) -> tuple[dict, list[str]]:
    source: dict = {}
    for name in ("resource_metrics", "actual_metrics", "efficiency_metrics"):
        value = bundle.get(name)
        if isinstance(value, Mapping):
            source.update(value)
    for key in _METRIC_KEYS:
        if key in bundle:
            source[key] = bundle[key]
    if isinstance(supplied, Mapping):
        source.update(supplied)
    metrics = {key: source[key] for key in _METRIC_KEYS if key in source}
    errors: list[str] = []
    if "action_tokens" not in metrics:
        tokens, error = _action_tokens(bundle.get("calls"))
        if tokens is not None:
            metrics["action_tokens"] = tokens
        if error:
            errors.append(error)
    return metrics, errors


class RewardEngine:
    """Combine judge-private differential evidence into one bounded record."""

    def __init__(
        self,
        baseline_store: EmptyPatchBaselineStore | None = None,
        *,
        efficiency_weight: float = DEFAULT_EFFICIENCY_WEIGHT,
        scales: Mapping[str, float] | None = None,
        critic: ProcessCritic | None = None,
    ):
        self.baseline_store = baseline_store
        self.efficiency_weight = efficiency_weight
        self.scales = dict(DEFAULT_SCALES if scales is None else scales)
        self.critic = critic or ProcessCritic()

    def evaluate(
        self,
        verification_bundle: Mapping[str, object],
        *,
        task_key: TaskKey | Mapping[str, object] | None = None,
        baseline: Mapping[str, object] | None = None,
        metrics: Mapping[str, object] | None = None,
    ) -> dict:
        if not isinstance(verification_bundle, Mapping):
            raise TypeError("verification_bundle must be a mapping")
        bundle = verification_bundle
        candidate = _candidate_report(bundle)
        key_value = task_key if task_key is not None else bundle.get("task_key")
        key: TaskKey | None = None
        key_error: str | None = None
        if key_value is not None:
            try:
                key = TaskKey.from_dict(key_value)
            except (TypeError, ValueError) as error:
                key_error = type(error).__name__

        if baseline is None:
            for name in ("baseline_report", "baseline", "baseline_summary"):
                value = bundle.get(name)
                if isinstance(value, Mapping):
                    baseline = value
                    break
        baseline_error: str | None = None
        if baseline is None and self.baseline_store is not None and key is not None:
            try:
                baseline = self.baseline_store.get(key)
            except (FileNotFoundError, OSError, ValueError, TypeError) as error:
                baseline_error = type(error).__name__
        if baseline is None:
            baseline = {
                "verification_complete": False,
                "verdict_valid": False,
                "infrastructure_error": baseline_error or "baseline_report_missing",
            }

        differential = pair_node_reports(candidate, baseline)
        extra_reasons = []
        if key_error:
            extra_reasons.append("task_key_invalid")
        if baseline_error:
            extra_reasons.append("baseline_unavailable")
        if extra_reasons:
            differential = dict(differential)
            differential["eligible"] = False
            differential["exclusion_reasons"] = list(dict.fromkeys(
                differential["exclusion_reasons"] + extra_reasons,
            ))

        actual_metrics, metric_errors = _metrics(bundle, metrics)
        validity = candidate.get("validity", bundle.get("validity"))
        if type(validity) is bool:
            validity = float(validity)
        record = composite_reward(
            differential,
            resolved=_resolved(bundle, candidate),
            validity=validity,
            metrics=actual_metrics,
            efficiency_weight=self.efficiency_weight,
            scales=self.scales,
        )
        # Metric gaps only disable the efficiency auxiliary (handled inside
        # composite_reward); they never overturn official differential eligibility.

        process = self.critic.evaluate(
            tool_events=bundle.get("tool_events", bundle.get("events")),
            edits=bundle.get("edits"),
            tests=bundle.get("test_events", bundle.get("tests")),
            differential=differential,
            private_targets=bundle.get("private_targets", bundle.get("judge_private_targets")),
        )
        artifacts = differential.get("artifact_hashes", {})
        record["audit"] = {
            "differential": differential,
            "efficiency_evidence": record["efficiency_evidence"],
            "process_evidence": process,
            "task_key_hash": key.cache_key if key else None,
            "baseline_cache_key": baseline.get("cache_key") if isinstance(baseline, Mapping) else None,
            "candidate_summary_ref": bundle.get("candidate_summary_ref"),
            "baseline_summary_ref": bundle.get("baseline_summary_ref"),
            "patch_sha256": bundle.get("patch_sha256"),
            "artifact_hashes": artifacts,
            "actual_metric_fields": sorted(actual_metrics),
            "metric_errors": list(metric_errors),
            "private_node_map_in_record": False,
        }
        return record

    score = evaluate
    __call__ = evaluate

    def spo_record(self, verification_bundle: Mapping[str, object], **kwargs: object) -> dict:
        return spo_reward_record(self.evaluate(verification_bundle, **kwargs))


def spo_reward_record(record: Mapping[str, object]) -> dict:
    """Drop audit-only fields before constructing trajectory.RewardRecord."""
    return {name: record.get(name) for name in REWARD_RECORD_FIELDS}


def compute_reward_record(
    verification_bundle: Mapping[str, object],
    *,
    baseline_store: EmptyPatchBaselineStore | None = None,
    baseline: Mapping[str, object] | None = None,
    task_key: TaskKey | Mapping[str, object] | None = None,
    metrics: Mapping[str, object] | None = None,
    efficiency_weight: float = DEFAULT_EFFICIENCY_WEIGHT,
    scales: Mapping[str, float] | None = None,
) -> dict:
    """Functional entry point suitable for ``official_runtime`` judge code."""
    engine = RewardEngine(baseline_store, efficiency_weight=efficiency_weight, scales=scales)
    return engine.evaluate(verification_bundle, task_key=task_key, baseline=baseline, metrics=metrics)
