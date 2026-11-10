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
    return {
        "P": components["P"]["value"],
        "G": components["G"]["value"],
        "components": components,
        "counts": counts,
        "coverage": coverage,
        "eligible": not reasons,
        "exclusion_reasons": list(dict.fromkeys(reasons)),
        "artifact_hashes": {"candidate": left["artifact_hashes"], "baseline": right["artifact_hashes"]},
        "node_detail_visibility": "judge_private_only",
    }


compute_differential = pair_node_reports
