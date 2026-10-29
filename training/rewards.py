"""Composite trajectory rewards from completed official verification.

Only the judge process sees required test identifiers and JUnit XML. The learner
receives aggregate counts, rates and fingerprints; missing evidence stays None.
No reference patch similarity or answer text is used as a reward.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
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
    output = {
        "junit_available": False,
        "junit_sha256": hashlib.sha256(xml_text.encode()).hexdigest() if xml_text else None,
        "repair_fraction": None,
        "regression_fraction": None,
        "dense_coverage_complete": False,
    }
    if not xml_text.strip():
        return {**output, "unavailable_reason": "missing_junit_report"}
    if "<!DOCTYPE" in xml_text.upper() or "<!ENTITY" in xml_text.upper():
        return {**output, "unavailable_reason": "unsupported_junit_declaration"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {**output, "unavailable_reason": "malformed_junit_report"}
    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    cases = list(root.findall(".//testcase"))
    if not suites or not cases:
        return {**output, "unavailable_reason": "no_testcases_in_junit"}

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

    # The official verifier falls back to test_patch function names only if
    # neither explicit test list exists. Preserve exactly that source choice.
    use_fallback = not fail_to_pass and not pass_to_pass
    repair = count(fallback_requirements if use_fallback else fail_to_pass)
    regression = count(pass_to_pass)
    dense_complete = repair["missing"] == 0 and regression["missing"] == 0
    output.update({
        "junit_available": True,
        "testcase_counts": {key: case_counts[key] for key in ("passed", "failure", "error", "skipped")},
        "repair_requirements_source": "test_patch_functions" if use_fallback else "FAIL_TO_PASS",
        "repair_counts": repair,
        "regression_counts": regression,
        "dense_coverage_complete": dense_complete,
    })
    if dense_complete:
        if repair["expected"]:
            output["repair_fraction"] = repair["passed"] / repair["expected"]
        if regression["expected"]:
            output["regression_fraction"] = regression["nonpassed"] / regression["expected"]
    else:
        output["unavailable_reason"] = "required_test_nodes_not_observed"
    return output


def compute_reward(result: dict, calls: list[dict], config: dict) -> dict:
    """Score one fully verified trajectory, preserving a dominant task reward.

    Missing dense statistics omit their shaping term and remain explicitly
    unavailable in the returned features. They are never represented as a
    measured zero pass/regression rate. Scores are computed after completion,
    so efficiency terms cannot shorten an in-progress task or its acceptance.
    """
    if (not result.get("verdict_valid") or not result.get("verification_complete")
            or result.get("infrastructure_error")):
        raise ValueError("Composite reward requires a complete, valid official verdict")
    success = result.get("success")
    if not isinstance(success, bool):
        raise ValueError("Official success must be an explicit boolean")
    weights = {
        "resolved": float(config.get("resolved_weight", 1.0)),
        "repair": float(config.get("repair_weight", 0.3)),
        "regression": float(config.get("regression_weight", 0.5)),
        "efficiency": float(config.get("efficiency_weight", 0.02)),
        "repetition": float(config.get("repetition_weight", 0.0)),
    }
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Reward weights must be finite and nonnegative")
    # Conservative bound includes every possible successful-trajectory penalty.
    # Even if optional statistics are unavailable, no failure may outrank any
    # success merely by exploiting auxiliary shaping or a short output.
    if weights["resolved"] - sum(weights[key] for key in ("regression", "efficiency", "repetition")) <= weights["repair"]:
        raise ValueError("resolved_weight must dominate all shaping terms and preserve success ordering")
    budget = int(config.get("success_token_budget", 65536))
    if budget < 1:
        raise ValueError("success_token_budget must be positive")
    if not calls or any(not call.get("completion_ids") for call in calls):
        raise ValueError("Reward needs the recorded, nonempty model calls for this trajectory")
    generated_tokens = sum(len(call["completion_ids"]) for call in calls)
    counts = Counter(tuple(call["completion_ids"]) for call in calls)
    duplicate_fraction = sum(value - 1 for value in counts.values()) / len(calls)
    features = dict(result.get("reward_features") or {})
    features.setdefault("repair_fraction", None)
    features.setdefault("regression_fraction", None)
    observed_repair = features.get("repair_fraction")
    observed_regression = features.get("regression_fraction")
    for name, value in (("repair_fraction", observed_repair), ("regression_fraction", observed_regression)):
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"Invalid judge statistic: {name}")
    dense_enabled = bool(config.get("dense_shaping_enabled", True))
    usable = dense_enabled and bool(features.get("dense_coverage_complete"))
    repair = observed_repair if usable else None
    regression = observed_regression if usable else None
    if success and observed_regression is not None and observed_regression != 0:
        raise ValueError("Official success conflicts with observed required-test regressions")
    components = {
        "resolved": weights["resolved"] * int(success),
        "repair": weights["repair"] * repair if repair is not None else None,
        "regression": -weights["regression"] * regression if regression is not None else None,
        "efficiency": -weights["efficiency"] * min(generated_tokens / budget, 1.0) if success else 0.0,
        "repetition": -weights["repetition"] * duplicate_fraction if success else 0.0,
    }
    features.update({
        "resolved": success,
        "generated_tokens": generated_tokens,
        "model_calls": len(calls),
        "duplicate_completion_fraction": duplicate_fraction,
        "success_token_budget": budget,
        "dense_shaping_enabled": dense_enabled,
        "group_reward_mode": "composite" if dense_enabled else "terminal_efficiency",
    })
    total = sum(value for value in components.values() if value is not None)
    return {"total": total, "components": components, "features": features}
