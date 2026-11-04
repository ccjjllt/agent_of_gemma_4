"""Judge-side attribution of terminal model failures, separate from worker faults."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET


def _environment_failure(detail: str) -> bool:
    return bool(re.search(
        r"permission denied|permissionerror|operation not permitted|read-only file system|"
        r"no space left|disk quota|cannot allocate memory|out of memory|memoryerror|"
        r"input/output error|too many open files|resource temporarily unavailable|"
        r"command not found|(?:git|patch|python3?)\b[^\n]*: not found|(?:git|patch) not found:|"
        r"patch file not found:|not a git repository|dubious ownership|index\.lock|"
        r"timed out|timeouterror|\bkilled\b|segmentation fault|"
        r"failed \((?:-\d+|12[4-9]|1[3-9]\d)\)", detail, re.IGNORECASE,
    ))


def junit_execution_evidence(xml_text: str) -> dict:
    """Keep execution categories only; test names and output stay with the judge."""
    counts = {"collection_errors": 0, "other_errors": 0}
    if "<!DOCTYPE" in xml_text.upper() or "<!ENTITY" in xml_text.upper():
        return counts
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return counts
    for case in root.findall(".//testcase"):
        for error in case.findall("error"):
            key = "collection_errors" if error.get("message") == "collection failure" else "other_errors"
            counts[key] += 1
    return counts


def patch_rejection(captured: dict) -> bool:
    """Require an observed patch-content rejection, never just a nonzero exit."""
    applications = captured.get("patch_applications", [])
    if len(applications) != 1 or applications[0]["exit_code"] != 1:
        return False
    detail = applications[0]["diagnostic"].lower()
    if _environment_failure(detail) or re.search(r"filenotfounderror|traceback \(most recent call last\)", detail):
        return False
    return bool(re.search(
        r"corrupt patch|malformed patch|patch fragment without header|"
        r"no valid patches|only garbage was found|patch failed:|patch does not apply|"
        r"hunk #\d+ failed|can't find file to patch|does not exist in index|"
        r"already exists in working directory|unsafe path traversal in patch:|"
        r"unsafe symlink/path escape in patch:", detail,
    ))


def failure_control_reason(result, captured: dict) -> str | None:
    """Which abnormal verdict could be attributable to model code?

    Exit 2 alone also means interruption; exits 3/5, signals, timeouts and
    arbitrary runner errors are deliberately not converted into model failures.
    """
    if not result.agent_patch.strip() or result.resolved:
        return None
    code = captured.get("pytest_exit_code")
    if code is None or code != result.test_exit_code or not result.test_output.startswith("STDOUT:\n"):
        return None
    error = result.error or ""
    # A later healthy sandbox does not rule out transient resource/I/O faults.
    if _environment_failure(result.test_output + "\n" + error):
        return None
    if code == 2 and (not error or error.startswith("Agent ")):
        evidence = captured.get("pytest_execution", {})
        if evidence.get("collection_errors", 0) and not evidence.get("other_errors", 0):
            return "pytest_collection_failure"
    if code == 4 and (not error or error.startswith("Agent ")):
        if "ImportError while loading conftest" in result.test_output:
            return "pytest_import_failure"
    # The official harness can reject exit 0 (e.g. premature os._exit(0)).
    if code == 0 and error.startswith((
        "Missing JUnit XML report", "Missing or empty JUnit XML report",
        "Malformed JUnit XML report:", "No <testsuite>", "No passing tests recorded",
        "Test failures/errors recorded", "Required test node did not pass:",
        "Pytest stdout summary indicates",
    )):
        return "junit_rejected"
    return None


def healthy_control(result, captured: dict) -> bool:
    """An unfixed base may fail assertions; it must actually execute tests."""
    features = captured.get("reward_features", {})
    counts = features.get("testcase_counts", {})
    return bool(
        result.error is None and result.test_exit_code in (0, 1)
        and captured.get("pytest_exit_code") == result.test_exit_code
        and result.test_output.startswith("STDOUT:\n") and features.get("junit_available")
        and counts.get("passed", 0) + counts.get("failure", 0) > 0
        and counts.get("error", 0) == 0
    )


def classify_verification(result, captured: dict, *, delivery_error=None, timing_error=None) -> dict:
    """A complete verdict can be a terminal patch rejection without running pytest.

    Dense rewards remain based on the candidate's actual JUnit report. A healthy
    empty-patch control supplies attribution only, never reward observations.
    """
    error = result.error or None
    pytest_ran = (captured.get("pytest_exit_code") is not None
                  and captured["pytest_exit_code"] == result.test_exit_code
                  and result.test_output.startswith("STDOUT:\n"))
    agent_error = error if error and error.startswith("Agent ") else None
    kind, stage = None, None
    if pytest_ran and result.test_exit_code in (0, 1) and (not error or agent_error):
        kind = "tests_completed"
    elif error and error.startswith("Failed to apply agent patch:") and patch_rejection(captured):
        kind, stage, agent_error = "agent_patch_rejected", "patch_application", error
    else:
        reason = failure_control_reason(result, captured)
        control = captured.get("failure_control", {})
        if reason and control.get("reason") == reason and control.get("healthy") is True:
            kind, stage = "agent_" + reason, "pytest"
            agent_error = error or f"Agent patch caused {reason} (healthy empty-patch control)."
    terminal = kind is not None
    valid = terminal and delivery_error is None and timing_error is None
    if not valid:
        kind = ("unconfirmed_session_timing" if timing_error else
                "unconfirmed_model_consumption" if delivery_error else "unclassified_verification_failure")
        stage = ("session_timing" if timing_error else "model_delivery" if delivery_error else
                 "patch_application" if error and error.startswith("Failed to apply agent patch:") else
                 "test_patch_application" if error and error.startswith("Failed to apply test_patch:") else
                 "pytest" if pytest_ran else "setup_or_agent")
    success = bool(result.resolved) if valid and kind == "tests_completed" else False if valid else None
    return {
        "success": success, "reward": float(success) if success is not None else None,
        "verification_complete": terminal, "pytest_ran": pytest_ran,
        "verdict_valid": valid, "verdict_kind": kind, "failure_stage": stage,
        "infrastructure_error": None if valid else (
            timing_error or delivery_error or error or
            f"No attributable official verification verdict (exit={result.test_exit_code})."
        ),
        "agent_error": agent_error,
    }
