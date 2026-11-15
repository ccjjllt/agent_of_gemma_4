"""Pair private candidate and empty-patch JUnit node statuses.

No node identifier or expected universe is returned.  Missing reports are not
empty universes and zero-denominator measured components are explicitly N/A.
"""
from __future__ import annotations

from typing import Mapping


_COMPONENTS = ("repair", "regression")
_PASSED = "passed"
_NONPASSED = {"failure", "error", "skipped", "nonpassed"}
_ALLOWED = {_PASSED, "missing", *_NONPASSED}


def _sequence(value: object) -> list[str] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        return None
    if any(not isinstance(node, str) or not node for node in value) or len(set(value)) != len(value):
        return None
    return list(value)


def _inspect(report: object, label: str) -> dict:
    reasons: list[str] = []
    if not isinstance(report, Mapping):
        return {"reasons": [f"{label}_report_missing"], "expected": None, "nodes": None,
                "coverage": None, "artifact_hashes": {}}
    if report.get("verification_complete") is not True:
        reasons.append(f"{label}_verification_incomplete")
    if report.get("verdict_valid") is not True:
        reasons.append(f"{label}_verdict_invalid")
    if report.get("infrastructure_error"):
        reasons.append(f"{label}_infrastructure_error")
    if report.get("junit_available") is False:
        reasons.append(f"{label}_junit_missing")

    raw_expected = report.get("expected_universes")
    expected: dict[str, list[str]] | None = None
    if isinstance(raw_expected, Mapping):
        parsed_expected: dict[str, list[str]] = {}
        expected_complete = True
        for component in _COMPONENTS:
            nodes = _sequence(raw_expected.get(component))
            if nodes is None:
                reasons.append(f"{label}_{component}_expected_universe_missing")
                expected_complete = False
            else:
                parsed_expected[component] = nodes
        expected = parsed_expected if expected_complete else None
    else:
        reasons.append(f"{label}_expected_universes_missing")

    raw_nodes = report.get("node_statuses", report.get("nodes"))
    nodes: dict[str, dict[str, str]] = {}
    coverage: dict[str, dict] = {}
    for component in _COMPONENTS:
        universe = expected.get(component, []) if expected is not None else []
        rows = raw_nodes.get(component) if isinstance(raw_nodes, Mapping) else None
        if not isinstance(rows, Mapping):
            reasons.append(f"{label}_{component}_node_statuses_missing")
            rows = {}
        normalized: dict[str, str] = {}
        invalid = 0
        for node, row in rows.items():
            if not isinstance(node, str):
                invalid += 1
                continue
            status = row.get("status") if isinstance(row, Mapping) else row
            if status == "failed":
                status = "failure"
            if status not in _ALLOWED:
                invalid += 1
            else:
                normalized[node] = str(status)
        expected_set = set(universe)
        missing = [node for node in universe if node not in normalized or normalized[node] == "missing"]
        unexpected = [node for node in normalized if node not in expected_set]
        if invalid:
            reasons.append(f"{label}_{component}_invalid_node_status")
        if unexpected:
            reasons.append(f"{label}_{component}_unexpected_nodes")
        if missing and universe:
            reasons.append(f"{label}_{component}_required_nodes_missing")
        if not universe and normalized:
            reasons.append(f"{label}_{component}_not_applicable_nodes_present")
        nodes[component] = normalized
        observed = len(universe) - len(missing)
        coverage[component] = {
            "expected": len(universe),
            "observed": observed,
            "missing": len(missing),
            "unexpected": len(unexpected),
            "invalid": invalid,
            "fraction": observed / len(universe) if universe else None,
            "complete": not missing and not unexpected and not invalid,
            "status": "not_applicable" if not universe else "complete" if not missing and not unexpected and not invalid else "incomplete",
        }
    artifacts = report.get("artifact_hashes")
    return {
        "reasons": reasons,
        "expected": expected,
        "nodes": nodes,
        "coverage": coverage,
        "artifact_hashes": dict(artifacts) if isinstance(artifacts, Mapping) else {},
    }


def shadow_regression(candidate: Mapping[str, object],
                      baseline: Mapping[str, object]) -> dict:
    """Exact baseline-passed regression observation, judge-private only.

    This is deliberately separate from official ``G``.  Official ``G`` uses
    PASS_TO_PASS requirements; public tasks here do not ship them.  Shadow G
    uses every exact ``classname::name`` node that passed in the empty-patch
    baseline.  It never enters eligibility or the reward formula.
    """
    definition = "exact_baseline_passed_regression_v1"
    base = {
        "definition": definition, "diagnostic_only": True,
        "official_G_replaced": False, "used_in_reward": False,
        "visibility": "judge_private_only",
    }
    if not isinstance(candidate, Mapping) or not isinstance(baseline, Mapping):
        return {**base, "status": "unavailable", "value": None, "numerator": None,
                "denominator": None, "reason": "report_missing"}
    if baseline.get("observed_nodes_usable") is False or candidate.get("observed_nodes_usable") is False:
        return {**base, "status": "unavailable", "value": None, "numerator": None,
                "denominator": None, "reason": "duplicate_or_ambiguous_exact_nodes"}
    baseline_nodes = baseline.get("observed_nodes")
    candidate_nodes = candidate.get("observed_nodes")
    if not isinstance(baseline_nodes, Mapping) or not isinstance(candidate_nodes, Mapping):
        return {**base, "status": "unavailable", "value": None, "numerator": None,
                "denominator": None, "reason": "exact_observed_nodes_not_retained"}
    if any(not isinstance(key, str) or not key or status not in _ALLOWED
           for key, status in baseline_nodes.items()):
        return {**base, "status": "unavailable", "value": None, "numerator": None,
                "denominator": None, "reason": "invalid_baseline_node_status"}
    if any(not isinstance(key, str) or not key or status not in _ALLOWED
           for key, status in candidate_nodes.items()):
        return {**base, "status": "unavailable", "value": None, "numerator": None,
                "denominator": None, "reason": "invalid_candidate_node_status"}
    baseline_passed = [key for key, status in baseline_nodes.items() if status == _PASSED]
    missing = [key for key in baseline_passed if key not in candidate_nodes]
    if missing:
        matched = len(baseline_passed) - len(missing)
        return {**base, "status": "unavailable", "value": None, "numerator": None,
                "denominator": None, "baseline_nodes": len(baseline_nodes),
                "baseline_passed_nodes": len(baseline_passed),
                "matching_nodes": matched,
                "candidate_only_nodes": len(set(candidate_nodes) - set(baseline_nodes)),
                "coverage_complete": False,
                "reason": "candidate_missing_baseline_passed_nodes"}
    denominator = len(baseline_passed)
    if denominator == 0:
        return {**base, "status": "not_applicable", "value": 0.0, "numerator": 0,
                "denominator": 0, "baseline_nodes": len(baseline_nodes),
                "baseline_passed_nodes": 0, "matching_nodes": 0,
                "candidate_only_nodes": len(set(candidate_nodes) - set(baseline_nodes)),
                "coverage_complete": True, "reason": "no_baseline_passed_nodes"}
    numerator = sum(candidate_nodes[key] != _PASSED for key in baseline_passed)
    return {**base, "status": "measured", "value": numerator / denominator,
            "numerator": numerator, "denominator": denominator,
            "baseline_nodes": len(baseline_nodes),
            "baseline_passed_nodes": denominator,
            "matching_nodes": denominator,
            "candidate_only_nodes": len(set(candidate_nodes) - set(baseline_nodes)),
            "coverage_complete": True, "reason": None}


def shadow_repair(candidate: Mapping[str, object],
                  baseline: Mapping[str, object]) -> dict:
    """Exact baseline-nonpassed repair observation, judge-private only.

    Symmetric to :func:`shadow_regression`, but the denominator is every exact
    node that did not pass in the empty-patch baseline.  A candidate that
    repairs at least one such node has a real positive signal even when the
    official repair universe is unavailable.  Missing candidate nodes keep the
    result unavailable rather than imputing a repair.
    """
    definition = "exact_baseline_nonpassed_repair_v1"
    base = {
        "definition": definition, "diagnostic_only": True,
        "official_P_replaced": False, "used_in_reward": False,
        "visibility": "judge_private_only",
    }
    if not isinstance(candidate, Mapping) or not isinstance(baseline, Mapping):
        return {**base, "status": "unavailable", "value": None, "numerator": None,
                "denominator": None, "reason": "report_missing"}
    if baseline.get("observed_nodes_usable") is False or candidate.get("observed_nodes_usable") is False:
        return {**base, "status": "unavailable", "value": None, "numerator": None,
                "denominator": None, "reason": "duplicate_or_ambiguous_exact_nodes"}
    baseline_nodes = baseline.get("observed_nodes")
    candidate_nodes = candidate.get("observed_nodes")
    if not isinstance(baseline_nodes, Mapping) or not isinstance(candidate_nodes, Mapping):
        return {**base, "status": "unavailable", "value": None, "numerator": None,
                "denominator": None, "reason": "exact_observed_nodes_not_retained"}
    if any(not isinstance(key, str) or not key or status not in _ALLOWED
           for key, status in baseline_nodes.items()):
        return {**base, "status": "unavailable", "value": None, "numerator": None,
                "denominator": None, "reason": "invalid_baseline_node_status"}
    if any(not isinstance(key, str) or not key or status not in _ALLOWED
           for key, status in candidate_nodes.items()):
        return {**base, "status": "unavailable", "value": None, "numerator": None,
                "denominator": None, "reason": "invalid_candidate_node_status"}
    baseline_nonpassed = [key for key, status in baseline_nodes.items() if status != _PASSED]
    missing = [key for key in baseline_nonpassed if key not in candidate_nodes]
    if missing:
        matched = len(baseline_nonpassed) - len(missing)
        return {**base, "status": "unavailable", "value": None, "numerator": None,
                "denominator": None, "baseline_nodes": len(baseline_nodes),
                "baseline_nonpassed_nodes": len(baseline_nonpassed),
                "matching_nodes": matched,
                "candidate_only_nodes": len(set(candidate_nodes) - set(baseline_nodes)),
                "coverage_complete": False,
                "reason": "candidate_missing_baseline_nonpassed_nodes"}
    denominator = len(baseline_nonpassed)
    if denominator == 0:
        return {**base, "status": "not_applicable", "value": 0.0, "numerator": 0,
                "denominator": 0, "baseline_nodes": len(baseline_nodes),
                "baseline_nonpassed_nodes": 0, "matching_nodes": 0,
                "candidate_only_nodes": len(set(candidate_nodes) - set(baseline_nodes)),
                "coverage_complete": True, "reason": "no_baseline_nonpassed_nodes"}
    numerator = sum(candidate_nodes[key] == _PASSED for key in baseline_nonpassed)
    return {**base, "status": "measured", "value": numerator / denominator,
            "numerator": numerator, "denominator": denominator,
            "baseline_nodes": len(baseline_nodes),
            "baseline_nonpassed_nodes": denominator,
            "matching_nodes": denominator,
            "candidate_only_nodes": len(set(candidate_nodes) - set(baseline_nodes)),
            "coverage_complete": True, "reason": None}


def pair_node_reports(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> dict:
    """Compute official P and G without imputing unavailable observations.

    P = (baseline non-passed repair -> candidate passed) / baseline non-passed repair.
    G = (baseline passed regression -> candidate non-passed) / baseline passed regression.
    """
    left = _inspect(candidate, "candidate")
    right = _inspect(baseline, "baseline")
    reasons = list(dict.fromkeys(left["reasons"] + right["reasons"]))
    universes_match = left["expected"] is not None and left["expected"] == right["expected"]
    if not universes_match:
        reasons.append("candidate_baseline_expected_universe_mismatch")

    components: dict[str, dict] = {}
    counts = {"fixed": 0, "regressed": 0}
    for component, rate_name in (("repair", "P"), ("regression", "G")):
        value = numerator = denominator = None
        status = "unavailable"
        if (universes_match and left["coverage"] and right["coverage"]
                and left["coverage"][component]["complete"] and right["coverage"][component]["complete"]):
            universe = left["expected"][component]
            candidate_nodes = left["nodes"][component]
            baseline_nodes = right["nodes"][component]
            if component == "repair":
                paired = [node for node in universe if baseline_nodes[node] != _PASSED]
                numerator = sum(candidate_nodes[node] == _PASSED for node in paired)
                counts["fixed"] = numerator
            else:
                paired = [node for node in universe if baseline_nodes[node] == _PASSED]
                numerator = sum(candidate_nodes[node] != _PASSED for node in paired)
                counts["regressed"] = numerator
            denominator = len(paired)
            if denominator == 0:
                value, status = 0.0, "not_applicable"
            else:
                value, status = numerator / denominator, "measured"
        components[rate_name] = {
            "value": value,
            "status": status,
            "numerator": numerator,
            "denominator": denominator,
            "reason": None if status != "unavailable" else f"{component}_pairing_evidence_incomplete",
        }

    coverage = {}
    for component in _COMPONENTS:
        candidate_coverage = left["coverage"].get(component, {}) if left["coverage"] else {}
        baseline_coverage = right["coverage"].get(component, {}) if right["coverage"] else {}
        coverage[component] = {
            "expected": candidate_coverage.get("expected"),
            "candidate": candidate_coverage,
            "baseline": baseline_coverage,
            "complete": bool(candidate_coverage.get("complete") and baseline_coverage.get("complete") and universes_match),
        }
    regression = shadow_regression(candidate, baseline)
    repair = shadow_repair(candidate, baseline)
    positive_signal = (
        (components["P"]["value"] is not None and components["P"]["value"] > 0)
        or (repair["status"] == "measured" and (repair["value"] or 0) > 0)
    )
    return {
        "P": components["P"]["value"],
        "G": components["G"]["value"],
        "components": components,
        "counts": counts,
        "coverage": coverage,
        "eligible": not reasons,
        "exclusion_reasons": list(dict.fromkeys(reasons)),
        "artifact_hashes": {"candidate": left["artifact_hashes"], "baseline": right["artifact_hashes"]},
        "shadow_regression": regression,
        "shadow_repair": repair,
        "positive_repair_signal": positive_signal,
        "node_detail_visibility": "judge_private_only",
    }


compute_differential = pair_node_reports
