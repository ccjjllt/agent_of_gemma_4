"""Judge JUnit XML -> differential node reports (pure, CPU-safe).

The official runtime already aggregates JUnit into fractions; SPO differential
rewards need per-requirement pairing against the empty-patch baseline.  This
module reuses the official requirement-matching semantics without executing
anything: each FAIL_TO_PASS / PASS_TO_PASS requirement maps to
passed / failure / missing from the same final JUnit report.

``node_matcher`` is injected (server: ``verification._node_matches_requirement``).
Without it, exact requirement equality is used.  No test names leave this API
beyond the aggregate pairing consumed by :mod:`training.reward.differential`.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence


def parse_junit_names(xml_text: str) -> tuple[set[str], set[str], bool]:
    """Return (passed_names, nonpassed_names, usable). Never raises on bad XML."""
    if not isinstance(xml_text, str) or not xml_text.strip():
        return set(), set(), False
    if "<!DOCTYPE" in xml_text.upper() or "<!ENTITY" in xml_text.upper():
        return set(), set(), False
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return set(), set(), False
    cases = list(root.findall(".//testcase"))
    if not cases:
        return set(), set(), False
    passed: set[str] = set()
    nonpassed: set[str] = set()
    for case in cases:
        name, classname = case.get("name", ""), case.get("classname", "")
        candidates = {name}
        if "[" in name:
            candidates.add(name.split("[", 1)[0])
        if classname:
            candidates.update((
                f"{classname}::{name}", f"{classname}.{name}",
                f"{classname.replace('.', '/')}.py::{name}",
            ))
        status = ("failure" if case.find("failure") is not None else
                  "error" if case.find("error") is not None else
                  "skipped" if case.find("skipped") is not None else "passed")
        (passed if status == "passed" else nonpassed).update(candidates)
    return passed, nonpassed, True


def _match(node_matcher: Callable[..., bool] | None, node: str,
           required: str, short: str, base: str) -> bool:
    if node_matcher is None:
        return node == required
    return bool(node_matcher(node, required, short, base))


def _component_statuses(requirements: Sequence[str], passed: set[str],
                        nonpassed: set[str],
                        node_matcher: Callable[..., bool] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for required in dict.fromkeys(requirements):
        short = required.split("::")[-1]
        base = short.split("[", 1)[0]
        has_nonpass = any(_match(node_matcher, node, required, short, base) for node in nonpassed)
        has_pass = any(_match(node_matcher, node, required, short, base) for node in passed)
        result[required] = "failure" if has_nonpass else "passed" if has_pass else "missing"
    return result


def build_node_report(fail_to_pass: Sequence[str], pass_to_pass: Sequence[str],
                      xml_text: str, node_matcher: Callable[..., bool] | None = None, *,
                      verification_complete: bool = False, verdict_valid: bool = False,
                      infrastructure_error: str | None = None) -> dict:
    """Build a differential-compatible report from one final JUnit XML."""
    repair_reqs = list(dict.fromkeys(fail_to_pass or ()))
    regression_reqs = list(dict.fromkeys(pass_to_pass or ()))
    passed, nonpassed, usable = parse_junit_names(xml_text)
    return {
        "expected_universes": {"repair": repair_reqs, "regression": regression_reqs},
        "node_statuses": {
            "repair": _component_statuses(repair_reqs, passed, nonpassed, node_matcher),
            "regression": _component_statuses(regression_reqs, passed, nonpassed, node_matcher),
        },
        "junit_available": usable,
        "verification_complete": verification_complete,
        "verdict_valid": verdict_valid,
        "infrastructure_error": infrastructure_error,
    }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def task_key_for(task_id: str, repo: str, base_commit: str,
                 test_patch: str, judge_environment_fingerprint: str) -> dict[str, str]:
    """Build a TaskKey mapping from official task text and judge fingerprint."""
    if not task_id or not repo or not base_commit:
        raise ValueError("task identity fields are required")
    return {
        "task_id": task_id,
        "repo": repo,
        "base_commit": base_commit,
        "test_patch_sha256": sha256_text(test_patch or ""),
        "judge_environment_hash": sha256_text(judge_environment_fingerprint or ""),
    }
