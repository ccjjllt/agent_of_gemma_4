"""Composite trajectory rewards from completed official verification.

Only the judge process sees required test identifiers and JUnit XML. The learner
receives aggregate counts, rates and fingerprints; missing evidence stays None.
No reference patch similarity or answer text is used as a reward.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import statistics
from typing import Callable
import xml.etree.ElementTree as ET


def junit_reward_features(
    xml_text: str,
    *,
    fail_to_pass: tuple[str, ...],
    pass_to_pass: tuple[str, ...],
    fallback_requirements: tuple[str, ...] = (),
    node_matcher: Callable[[str, str, str, str], bool],
) -> dict:
    """Aggregate a final JUnit report with the official requirement matcher.

    The caller reads the report only after the official verification process has
    exited. This function does not execute, select, shorten or repeat tests.
    """
    # This source choice matches the official verifier. Knowing that a component
    # has no requirements is different from being unable to observe its tests.
    use_fallback = not fail_to_pass and not pass_to_pass
    repair_requirements = fallback_requirements if use_fallback else fail_to_pass

    def unseen_counts(requirements):
        expected = len(tuple(dict.fromkeys(requirements)))
        return {"expected": expected, "passed": 0, "nonpassed": 0, "missing": expected}

    expected_counts = {
        "repair": unseen_counts(repair_requirements),
        "regression": unseen_counts(pass_to_pass),
    }
    output = {
        "junit_available": False,
        "junit_sha256": hashlib.sha256(xml_text.encode()).hexdigest() if xml_text else None,
        "repair_fraction": None,
        "regression_fraction": None,
        "repair_requirements_source": "test_patch_functions" if use_fallback else "FAIL_TO_PASS",
        "dense_coverage_complete": False,
    }
    for name, counts in expected_counts.items():
        output[name + "_counts"] = counts
        output[name + "_coverage_complete"] = counts["expected"] == 0
        output[name + "_available"] = False
        output[name + "_unavailable_reason"] = "not_applicable" if counts["expected"] == 0 else None

    def unavailable(reason):
        output["unavailable_reason"] = reason
        for name, counts in expected_counts.items():
            if counts["expected"]:
                output[name + "_unavailable_reason"] = reason
        return output

    if not xml_text.strip():
        return unavailable("missing_junit_report")
    if "<!DOCTYPE" in xml_text.upper() or "<!ENTITY" in xml_text.upper():
        return unavailable("unsupported_junit_declaration")
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return unavailable("malformed_junit_report")
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    cases = list(root.findall(".//testcase"))
    if not suites or not cases:
        return unavailable("no_testcases_in_junit")

    passed_names, nonpassed_names = set(), set()
    case_counts = Counter()
    for case in cases:
        name, classname = case.get("name", ""), case.get("classname", "")
        candidates = {name}
        if "[" in name:
            candidates.add(name.split("[", 1)[0])
        if classname:
            candidates.update((f"{classname}::{name}", f"{classname}.{name}",
                               f"{classname.replace('.', '/')}.py::{name}"))
        status = ("failure" if case.find("failure") is not None else
                  "error" if case.find("error") is not None else
                  "skipped" if case.find("skipped") is not None else "passed")
        case_counts[status] += 1
        (passed_names if status == "passed" else nonpassed_names).update(candidates)

    def count(requirements):
        unique = tuple(dict.fromkeys(requirements))
        counts = {"expected": len(unique), "passed": 0, "nonpassed": 0, "missing": 0}
        for required in unique:
            short = required.split("::")[-1]
            base = short.split("[", 1)[0]
            has_pass = any(node_matcher(node, required, short, base) for node in passed_names)
            has_nonpass = any(node_matcher(node, required, short, base) for node in nonpassed_names)
            key = "nonpassed" if has_nonpass else "passed" if has_pass else "missing"
            counts[key] += 1
        return counts

    repair = count(repair_requirements)
    regression = count(pass_to_pass)
    dense_complete = repair["missing"] == 0 and regression["missing"] == 0
    output.update({
        "junit_available": True,
        "testcase_counts": {key: case_counts[key] for key in ("passed", "failure", "error", "skipped")},
        "repair_counts": repair,
        "regression_counts": regression,
        "dense_coverage_complete": dense_complete,
    })
    # Preserve each independently verified component. A missing PASS_TO_PASS
    # node must not erase a fully observed FAIL_TO_PASS fraction, or vice versa.
    for name, counts, numerator in (("repair", repair, "passed"),
                                    ("regression", regression, "nonpassed")):
        complete = counts["missing"] == 0
        available = complete and counts["expected"] > 0
        output[name + "_coverage_complete"] = complete
        output[name + "_available"] = available
        output[name + "_unavailable_reason"] = (
            "not_applicable" if counts["expected"] == 0 else
            "required_test_nodes_not_observed" if not complete else None
        )
        if available:
            output[name + "_fraction"] = counts[numerator] / counts["expected"]
    if not dense_complete:
        output["unavailable_reason"] = "required_test_nodes_not_observed"
    return output


def _weights(config):
    weights = {"resolved": float(config.get("resolved_weight", 1.0)),
               "repair": float(config.get("repair_weight", 0.3)),
               "regression": float(config.get("regression_weight", 0.5)),
               "efficiency": float(config.get("efficiency_weight", 0.02)),
               "repetition": float(config.get("repetition_weight", 0.0))}
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Reward weights must be finite and nonnegative")
    if weights["resolved"] - sum(weights[key] for key in ("regression", "efficiency", "repetition")) <= weights["repair"]:
        raise ValueError("resolved_weight must dominate shaping and preserve success ordering")
    return weights


def _rate_evidence(features, name):
    """Return a measured rate or a valid interval, never impute a missing rate."""
    value = features.get(name + "_fraction")
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                              or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"Invalid judge statistic: {name}_fraction")
    counts = features.get(name + "_counts")
    reliable_report = bool(features.get("junit_available"))
    if counts is not None:
        if (not isinstance(counts, dict) or any(type(counts.get(key)) is not int or counts[key] < 0
                for key in ("expected", "passed", "nonpassed", "missing"))
                or counts["passed"] + counts["nonpassed"] + counts["missing"] != counts["expected"]):
            raise ValueError(f"Inconsistent judge counts: {name}")
        expected = counts["expected"]
        if expected == 0:
            if value is not None:
                raise ValueError("A not-applicable component cannot have a measured fraction")
            return {"status": "not_applicable", "reason": "not_applicable", "fraction": None,
                    "lower": 0.0, "upper": 0.0, "counts": dict(counts)}
        if reliable_report:
            numerator = counts["passed" if name == "repair" else "nonpassed"]
            lower, upper = numerator / expected, (numerator + counts["missing"]) / expected
            if value is not None and (counts["missing"] or not math.isclose(value, lower, abs_tol=1e-12)):
                raise ValueError(f"Reported {name} fraction conflicts with observed counts")
            return {"status": "complete" if lower == upper else "bounded_partial",
                    "reason": None if lower == upper else "required_test_nodes_not_observed",
                    "fraction": lower if lower == upper else None,
                    "lower": lower, "upper": upper, "counts": dict(counts)}
    complete = features.get(name + "_coverage_complete", features.get("dense_coverage_complete", False))
    if reliable_report and complete and value is not None:
        return {"status": "complete", "reason": None, "fraction": float(value),
                "lower": float(value), "upper": float(value)}
    # No counts does not establish expected=0. This includes runtime captures
    # that failed before producing a JUnit aggregate.
    return {"status": "unavailable", "fraction": None, "lower": 0.0, "upper": 1.0,
            "reason": features.get(name + "_unavailable_reason") or
                      features.get("unavailable_reason") or "missing_component_evidence"}


def compute_reward(result: dict, calls: list[dict], config: dict) -> dict:
    """Bound the full configured reward after a valid official acceptance.

    Unknown contributions keep None and an interval. Only point-identified
    totals have a numerical `total`; missing totals never become measured zero.
    """
    if (not result.get("verdict_valid") or not result.get("verification_complete")
            or result.get("infrastructure_error")):
        raise ValueError("Composite reward requires a complete, valid official verdict")
    success = result.get("success")
    if not isinstance(success, bool):
        raise ValueError("Official success must be an explicit boolean")
    weights = _weights(config)
    budget = int(config.get("success_token_budget", 65536))
    if budget < 1:
        raise ValueError("success_token_budget must be positive")
    if not calls or any(not call.get("completion_ids") for call in calls):
        raise ValueError("Reward needs recorded, nonempty model calls")
    generated_tokens = sum(len(call["completion_ids"]) for call in calls)
    counts = Counter(tuple(call["completion_ids"]) for call in calls)
    duplicate_fraction = sum(value - 1 for value in counts.values()) / len(calls)
    features = dict(result.get("reward_features") or {})
    features.setdefault("repair_fraction", None)
    features.setdefault("regression_fraction", None)
    dense_enabled = bool(config.get("dense_shaping_enabled", True))
    components = {
        "resolved": weights["resolved"] * int(success),
        "efficiency": -weights["efficiency"] * min(generated_tokens / budget, 1.0) if success else 0.0,
        "repetition": -weights["repetition"] * duplicate_fraction if success else 0.0,
    }
    bounds = {name: {"lower": value, "upper": value} for name, value in components.items()}
    evidence = {name: {"status": "complete", "reason": None} for name in components}
    for name in ("repair", "regression"):
        item = _rate_evidence(features, name)
        if name == "regression" and success and item["lower"] > 0:
            raise ValueError("Official success conflicts with observed required-test regressions")
        if not dense_enabled or weights[name] == 0:
            evidence[name] = {**item, "status": "disabled", "reason": "configured_component_disabled"}
            components[name] = 0.0
            bounds[name] = {"lower": 0.0, "upper": 0.0}
            continue
        evidence[name] = item
        coefficient = weights[name] * (-1 if name == "regression" else 1)
        endpoints = (coefficient * item["lower"], coefficient * item["upper"])
        lower, upper = min(endpoints), max(endpoints)
        bounds[name] = {"lower": lower, "upper": upper}
        components[name] = lower if lower == upper else None
    lower = math.fsum(item["lower"] for item in bounds.values())
    upper = math.fsum(item["upper"] for item in bounds.values())
    complete = all(value is not None for value in components.values())
    features.update({
        "resolved": success, "generated_tokens": generated_tokens, "model_calls": len(calls),
        "duplicate_completion_fraction": duplicate_fraction, "success_token_budget": budget,
        "dense_shaping_enabled": dense_enabled, "reward_complete": complete,
        "missing_components": [name for name, value in components.items() if value is None],
    })
    return {"total": math.fsum(components.values()) if complete else None,
            "reward_bounds": {"lower": lower, "upper": upper}, "components": components,
            "component_bounds": bounds, "component_evidence": evidence, "features": features}


def _rms(values):
    return math.sqrt(statistics.fmean(value * value for value in values))


def _centered_bounds(bounds):
    """Closest-to-zero certified centered advantages, not pairwise ranks.

    For any completion of unknown reward values inside their intervals, every
    nonzero signal has the true group-centered advantage's sign. Do NOT center
    these signals again: they need not sum to zero when evidence is partial.
    """
    size = len(bounds)
    intervals, signals = [], []
    if all(item["lower"] == item["upper"] for item in bounds):
        mean = statistics.fmean(item["lower"] for item in bounds)
        values = [item["lower"] - mean for item in bounds]
        return [{"lower": value, "upper": value} for value in values], values
    for i, item in enumerate(bounds):
        lower = ((size - 1) * item["lower"] - math.fsum(other["upper"] for j, other in enumerate(bounds) if j != i)) / size
        upper = ((size - 1) * item["upper"] - math.fsum(other["lower"] for j, other in enumerate(bounds) if j != i)) / size
        intervals.append({"lower": lower, "upper": upper})
        signals.append(lower if lower > 0 else upper if upper < 0 else 0.0)
    return intervals, signals


def _select_signal(scored, threshold, omit=None):
    names = [name for name in scored[0]["components"] if name != omit]
    bounds = [{"lower": math.fsum(row["component_bounds"][name]["lower"] for name in names),
               "upper": math.fsum(row["component_bounds"][name]["upper"] for name in names)} for row in scored]
    intervals, strict = _centered_bounds(bounds)
    common = [name for name in names if all(row["components"][name] is not None for row in scored)]
    common_rewards = [math.fsum(row["components"][name] for name in common) for row in scored]
    common_mean = statistics.fmean(common_rewards)
    common_signal = [value - common_mean for value in common_rewards]
    complete = all(item["lower"] == item["upper"] for item in bounds)
    # Retain the old reliable terminal comparison when wide dense intervals
    # obscure EVERY full-objective sign. This is an explicitly reduced objective.
    fallback = not complete and _rms(strict) <= threshold and _rms(common_signal) > threshold
    signal = common_signal if fallback else strict
    mode = "exact_composite" if complete else "common_components_fallback" if fallback else "interval_centered_advantage"
    return {"signal": signal, "advantage_bounds": intervals, "strict_signal": strict,
            "signal_mode": mode, "common_components": common, "common_signal": common_signal,
            "common_rewards": common_rewards, "signal_rms": _rms(signal)}


def score_group_rewards(results, calls, config, *, scale_rewards="group", advantage_std_floor=0.05,
                        min_group_std=1e-4):
    """Score a complete group, preserving evidence and the chosen loss target.

    This does not claim an unbiased full-reward estimate under arbitrary missing
    evidence. Widening an interval can erase a negative signal; missingness and
    fallback rates must remain visible. There is no early trajectory stopping.
    """
    if len(results) < 2 or len(results) != len(calls):
        raise ValueError("A complete group needs matching results and calls for at least two trajectories")
    if (scale_rewards not in ("group", "none") or not math.isfinite(advantage_std_floor)
            or advantage_std_floor <= 0 or not math.isfinite(min_group_std) or min_group_std < 0):
        raise ValueError("Invalid group reward normalization")
    scored = [compute_reward(result, items, config) for result, items in zip(results, calls)]
    chosen = _select_signal(scored, min_group_std)
    signal = chosen["signal"]
    denominator = max(chosen["signal_rms"], advantage_std_floor) if scale_rewards == "group" else 1.0
    advantages = [value / denominator for value in signal]
    features = [row["features"] for row in scored]
    legacy_dense = (bool(config.get("dense_shaping_enabled", True))
                    and all(item.get("junit_available") and item.get("dense_coverage_complete") for item in features)
                    and len({(item.get("repair_fraction") is None, item.get("regression_fraction") is None)
                             for item in features}) == 1)
    weights = _weights(config)
    legacy_rewards = []
    for row in scored:
        parts = [row["components"][name] for name in ("resolved", "efficiency", "repetition")]
        if legacy_dense:
            for name, sign in (("repair", 1), ("regression", -1)):
                observed = row["features"].get(name + "_fraction")
                if observed is not None:
                    parts.append(sign * weights[name] * observed)
        legacy_rewards.append(math.fsum(parts))
    legacy_std = statistics.pstdev(legacy_rewards)
    missing = sum(row["total"] is None for row in scored)
    outcomes = [result["success"] for result in results]
    diagnostics = {
        "group_size": len(scored), "point_reward_count": len(scored) - missing,
        "missing_reward_count": missing, "signal_source": chosen["signal_mode"],
        "common_components": chosen["common_components"],
        "learning_signal_rms": chosen["signal_rms"], "strict_signal_rms": _rms(chosen["strict_signal"]),
        "common_signal_rms": _rms(chosen["common_signal"]),
        "nonzero_advantages": sum(value != 0 for value in advantages),
        "legacy_group_fallback_triggered": bool(config.get("dense_shaping_enabled", True)) and not legacy_dense,
        "legacy_signal_rms": legacy_std,
        "legacy_all_failure_zero": not any(outcomes) and all(value == 0 for value in legacy_rewards),
        "new_signal_recovered_vs_legacy": chosen["signal_rms"] > min_group_std and legacy_std <= min_group_std,
        "no_provable_signal_with_missing": bool(missing and chosen["signal_rms"] <= min_group_std),
        "complete_evidence_zero_signal": not missing and chosen["signal_rms"] <= min_group_std,
        "ablation": {},
    }
    for name in ("repair", "regression", "efficiency"):
        alternative = _select_signal(scored, min_group_std, omit=name)
        diagnostics["ablation"]["without_" + name] = {
            "learning_signal_rms": alternative["signal_rms"],
            "nonzero_advantages": sum(value != 0 for value in alternative["signal"]),
            "would_update": alternative["signal_rms"] > min_group_std,
            "signal_source": alternative["signal_mode"],
        }
    signature = json.dumps({"algorithm": "centered_reward_intervals_v1", "weights": _weights(config),
                            "success_token_budget": int(config.get("success_token_budget", 65536)),
                            "dense_shaping_enabled": bool(config.get("dense_shaping_enabled", True)),
                            "signal_source": chosen["signal_mode"],
                            "common_components": (chosen["common_components"]
                                                  if chosen["signal_mode"] == "common_components_fallback" else None)},
                           sort_keys=True, separators=(",", ":"))
    return {"scored": scored, "rewards": [row["total"] for row in scored],
            "learning_signal": signal, "advantages": advantages,
            "advantage_bounds": chosen["advantage_bounds"],
            "reward_mode": chosen["signal_mode"], "reward_signature": signature,
            "diagnostics": diagnostics}
