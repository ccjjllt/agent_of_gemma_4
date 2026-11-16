"""Bounded official-differential utility and measured-success efficiency."""
from __future__ import annotations

import math
import statistics
from typing import Mapping


LEGACY_OBJECTIVE_VERSION = "official_differential_spo_v2"
OBJECTIVE_VERSION = "official_differential_spo_v3"
SUPPORTED_OBJECTIVE_VERSIONS = (LEGACY_OBJECTIVE_VERSION, OBJECTIVE_VERSION)
DEFAULT_EFFICIENCY_WEIGHT = 0.05
DEFAULT_SCALES = {"gpu_seconds": 3600.0, "action_tokens": 32768.0, "tool_calls": 100.0}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def efficiency_evidence(metrics: Mapping[str, object] | None, scales: Mapping[str, float] | None = None) -> dict:
    """Consume only measured GPU seconds, action tokens and actual tool calls.

    Available components are averaged; absent components remain absent.  If no
    usable measurement exists, E is None (unknown), never zero or one.
    """
    scales = dict(DEFAULT_SCALES if scales is None else scales)
    for name in DEFAULT_SCALES:
        scale = _number(scales.get(name))
        if scale is None or scale <= 0:
            raise ValueError(f"Efficiency scale {name} must be positive and finite")
        scales[name] = scale
    source = metrics if isinstance(metrics, Mapping) else {}
    components: dict[str, dict] = {}
    values: list[float] = []
    invalid: list[str] = []
    for name in DEFAULT_SCALES:
        raw = source.get(name)
        value = _number(raw)
        integer_required = name in ("action_tokens", "tool_calls")
        if raw is None:
            components[name] = {"status": "missing", "actual": None, "score": None}
            continue
        if value is None or value < 0 or (integer_required and int(value) != value):
            components[name] = {"status": "invalid", "actual": None, "score": None}
            invalid.append(name)
            continue
        actual = int(value) if integer_required else value
        score = math.exp(-value / scales[name])
        components[name] = {"status": "measured", "actual": actual, "score": score,
                            "scale": scales[name]}
        values.append(score)
    if invalid:
        status, efficiency = "invalid", None
    elif values:
        status, efficiency = ("complete" if len(values) == 3 else "partial"), statistics.fmean(values)
    else:
        status, efficiency = "unknown", None
    return {
        "status": status,
        "efficiency": efficiency,
        "components": components,
        "available_components": len(values),
        "invalid_components": invalid,
        "formula": "mean(exp(-actual/tau)) over available actual measurements",
    }


def composite_reward(
    differential: Mapping[str, object],
    *,
    resolved: object,
    validity: object,
    metrics: Mapping[str, object] | None = None,
    efficiency_weight: float = DEFAULT_EFFICIENCY_WEIGHT,
    scales: Mapping[str, float] | None = None,
    objective_version: str = OBJECTIVE_VERSION,
) -> dict:
    """Use complete differentials or, in v3, a valid sparse official verdict.

    v2 keeps its original strict differential eligibility and formula. In v3,
    unavailable P/G stay unknown; the sparse branch uses only resolved and
    validity, with the same utility weights and bounded base offset.
    """
    if objective_version not in SUPPORTED_OBJECTIVE_VERSIONS:
        raise ValueError(f"unsupported reward objective_version: {objective_version}")
    weight = _number(efficiency_weight)
    if weight is None or not 0 <= weight <= 1:
        raise ValueError("efficiency_weight must be finite and in [0, 1]")
    differential_reasons = list(differential.get("exclusion_reasons") or []) if isinstance(differential, Mapping) else ["differential_missing"]
    if not isinstance(differential, Mapping) or differential.get("eligible") is not True:
        if not differential_reasons:
            differential_reasons = ["differential_ineligible"]
    verdict_reasons: list[str] = []
    if type(resolved) is not bool:
        verdict_reasons.append("resolved_verdict_missing")
    validity_value = _number(validity)
    if validity_value is None or not 0 <= validity_value <= 1:
        verdict_reasons.append("validity_evidence_missing_or_invalid")

    p = _number(differential.get("P")) if isinstance(differential, Mapping) else None
    g = _number(differential.get("G")) if isinstance(differential, Mapping) else None
    component_reasons: list[str] = []
    if p is None or not 0 <= p <= 1:
        component_reasons.append("P_unavailable")
    if g is None or not 0 <= g <= 1:
        component_reasons.append("G_unavailable")
    differential_available = not differential_reasons and not component_reasons
    if objective_version == LEGACY_OBJECTIVE_VERSION:
        reasons = differential_reasons + verdict_reasons + component_reasons
    else:
        candidate_reasons = list(differential.get("candidate_exclusion_reasons") or []) if isinstance(differential, Mapping) else []
        candidate_eligible = (
            isinstance(differential, Mapping)
            and differential.get("candidate_eligible", differential.get("eligible")) is True
        )
        if not candidate_eligible and not candidate_reasons:
            candidate_reasons.append("candidate_verification_evidence_missing")
        reasons = candidate_reasons + verdict_reasons
    reasons = list(dict.fromkeys(reasons))
    sparse = objective_version == OBJECTIVE_VERSION and not differential_available
    if sparse:
        # Keep any partial observations in the private differential audit;
        # the learner record must not imply that missing components were zero.
        p = g = None

    base = utility = progress = efficiency = total = None
    efficiency_info = efficiency_evidence(metrics, scales)
    if not reasons:
        if sparse:
            utility = 0.60 * float(resolved) + 0.05 * validity_value
        else:
            progress = max(0.0, p - g)
            utility = 0.60 * float(resolved) + 0.25 * progress - 0.10 * g + 0.05 * validity_value
        base = min(1.0, max(0.0, utility + 0.10))
        if sparse:
            total = base
            efficiency_info = {**efficiency_info,
                               "auxiliary_status": "disabled_for_sparse_official_reward"}
        elif not resolved:
            # A failure's reward is completely independent of all cost fields.
            efficiency_info = {
                "status": "not_applicable", "efficiency": None, "components": {},
                "available_components": 0, "invalid_components": [],
                "reason": "efficiency_requires_resolved",
            }
            total = base
        elif weight == 0:
            efficiency = efficiency_info["efficiency"]
            total = base
        elif efficiency_info["efficiency"] is None:
            total = base
            efficiency_info = {**efficiency_info,
                               "auxiliary_status": "disabled_until_efficiency_is_measured"}
        else:
            efficiency = efficiency_info["efficiency"]
            total = base + weight * float(resolved) * (efficiency - base)
            total = min(1.0, max(0.0, total))
    eligible = not reasons
    return {
        "resolved": resolved if type(resolved) is bool else None,
        "P": p,
        "G": g,
        "progress": progress,
        "regression": None if g is None else -g,
        "validity": validity_value if validity_value is not None and 0 <= validity_value <= 1 else None,
        "utility": utility,
        "base_reward": base,
        "efficiency": efficiency,
        "efficiency_weight": weight,
        "total_reward": total,
        "eligible": eligible,
        "exclusion_reason": ";".join(reasons) if reasons else None,
        "objective_version": objective_version,
        "efficiency_evidence": efficiency_info,
        "reward_branch": "excluded" if reasons else "sparse_official" if sparse else "differential",
        "sparse_reasons": list(dict.fromkeys(differential_reasons + component_reasons)) if sparse and not reasons else [],
    }
