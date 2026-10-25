"""Normalize actual CPU replay artifacts into provenance-preserving SFT records.

This is a data export: it does not execute tasks, change a verifier, or repair a
source report. Ineligible records retain their evidence and concrete reasons.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from swegemma.harness.verification import _validate_junit_xml
from swegemma.models.task import _normalize_test_nodes
from swegemma.models import Task


ROOT = Path(__file__).resolve().parents[3]
REPLAY = Path(__file__).resolve().parent
SIX_TOOLS = {"run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch"}
EXCLUDED_REPOS = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}
SOURCE_PIN_FIELDS = ("instance_id", "source_task_path", "source_task_sha256", "execution_commit",
                     "source_review_sha256", "primary_adapter", "submechanism",
                     "visible_prompt_profile")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def resolve(value: str | Path | None, near: Path = REPLAY) -> Path | None:
    if not value:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [ROOT / path, near / path]
    return next((p.resolve() for p in candidates if p.exists()), candidates[0].resolve())


def read_json(path: Path | None):
    return json.loads(path.read_text()) if path and path.is_file() else None


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def unwrap_task(payload, instance_id: str) -> dict:
    """Find the exact task row in saved task-provider response envelopes."""
    if isinstance(payload, dict):
        if payload.get("instance_id") == instance_id and "FAIL_TO_PASS" in payload:
            return payload
        for key in ("row", "task", "rows"):
            if key in payload:
                found = unwrap_task(payload[key], instance_id)
                if found:
                    return found
    elif isinstance(payload, list):
        for item in payload:
            found = unwrap_task(item, instance_id)
            if found:
                return found
    return {}


def find_run(report: dict) -> Path | None:
    for key in ("current_replay", "run_directory", "replay_output", "run_dir", "run_path"):
        path = resolve(report.get(key))
        if path and path.is_dir():
            return path
    for phase in ("repaired", "baseline"):
        value = report.get("reports", {}).get(phase, {})
        entries = value if isinstance(value, list) else [value]
        for entry in entries:
            path = resolve(entry.get("path")) if isinstance(entry, dict) else None
            if path:
                return path.parent
    return None


def exact_required_case(case: dict, requirement: str, source_root: str = "") -> bool:
    """Match baseline causality without the official matcher's prefix fallback.

    The unmodified official matcher still determines repaired acceptance. Here,
    a similarly named failing test cannot substitute for the actual F2P case.
    """
    parts = requirement.split("::")
    if case["name"] != parts[-1]:
        return False
    if len(parts) == 1:
        return True
    source_file = parts[0]
    if source_root:
        source_file = str(Path(source_file).relative_to(source_root))
    module = source_file.removesuffix(".py").replace("/", ".")
    expected = ".".join([module, *parts[1:-1]])
    return case["classname"] == expected


def junit_report(path: Path, original: dict, adapted: dict, source_root: str = "") -> dict:
    raw = path.read_text()
    tree = ET.fromstring(raw)
    suites = [tree] if tree.tag == "testsuite" else list(tree.findall(".//testsuite"))
    cases = []
    for element in tree.findall(".//testcase"):
        outcome = next((kind for kind in ("failure", "error", "skipped")
                        if element.find(kind) is not None), "passed")
        detail = element.find(outcome) if outcome != "passed" else None
        cases.append({"classname": element.attrib.get("classname", ""),
                      "name": element.attrib.get("name", ""), "outcome": outcome,
                      "message": detail.attrib.get("message", "") if detail is not None else None})
    f2p = _normalize_test_nodes(original.get("FAIL_TO_PASS"))
    evidence = []
    for required in f2p:
        matching = [case for case in cases if exact_required_case(case, required, source_root)]
        evidence.append({"required_node": required, "matching_cases": matching,
                         "actual_failure_observed": bool(matching) and all(
                             case["outcome"] in {"failure", "error"} for case in matching)})
    official = _validate_junit_xml(raw, f2p,
                                   _normalize_test_nodes(original.get("PASS_TO_PASS")),
                                   adapted.get("test_patch", ""))
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    return {"path": label(path), "sha256": sha(path), **counts,
            "skips": counts["skipped"], "passed": counts["tests"] - counts["failures"]
            - counts["errors"] - counts["skipped"],
            "original_official_junit_validation": list(official),
            "original_F2P_evidence": evidence,
            "baseline_case_address_source_root": source_root or ".",
            "all_original_F2P_observed_failed": bool(evidence) and all(
                item["actual_failure_observed"] for item in evidence),
            "failed_nodes": [f'{case["classname"]}::{case["name"]}' for case in cases
                             if case["outcome"] in {"failure", "error"}],
            "skipped_nodes": [f'{case["classname"]}::{case["name"]}' for case in cases
                              if case["outcome"] == "skipped"]}


def parent_compatibility(summary: dict) -> dict:
    packages = {canonicalize_name(key): value
                for key, value in summary.get("runtime_packages", {}).items()}
    pinned = {name: importlib.metadata.version(name)
              for name in ("swegemma", "adk-eval-core", "adk-submission", "google-adk")}
    reasons = [f"{name} runtime identity missing or different from pinned CPU runtime"
               for name, version in pinned.items() if packages.get(name) != version]
    bindings = summary.get("parent_runtime_import_bindings", {})
    requirements = [Requirement(value) for parent in ("google-adk", "adk-eval-core")
                    for value in (importlib.metadata.requires(parent) or [])
                    if canonicalize_name(Requirement(value).name) in {"pydantic", "numpy"}]
    checks = []
    for requirement in requirements:
        name = canonicalize_name(requirement.name)
        actual = bindings.get(name, {}).get("version", packages.get(name))
        checks.append({"requirement": str(requirement), "actual_version": actual,
                       "evidence": "actual_parent_import" if name in bindings else
                                   "recorded_distribution_metadata_historical_import_not_captured"})
        if not actual or not requirement.specifier.contains(actual):
            reasons.append(f"parent {name} {actual!r} does not satisfy {requirement}")
        if bindings.get(name, {}).get("import_error"):
            reasons.append(f"parent {name} actual import failed")
    return {"compatible": not reasons, "reasons": reasons, "pinned_runtime": pinned,
            "parent_google_adk": packages.get("google-adk"),
            "parent_pydantic": bindings.get("pydantic", {}).get("version", packages.get("pydantic")),
            "parent_numpy": bindings.get("numpy", {}).get("version", packages.get("numpy")),
            "critical_requirement_source": "installed pinned google-adk and adk-eval-core METADATA",
            "critical_dependency_checks": checks,
            "actual_parent_import_bindings": bindings,
            "target_dependency_profile_is_separate": bool(summary.get("task_site_packages_profile"))}


def approved_mapping(instance_id: str, original: dict, adapted: dict, approvals: dict) -> dict | None:
    """Apply only a separately reviewed, hash-pinned source identity mapping."""
    approval = approvals.get(instance_id)
    if not approval:
        return None
    path = resolve(approval["mapping_path"])
    if not path or sha(path) != approval["mapping_sha256"]:
        raise ValueError("Audited source mapping differs from its explicitly reviewed hash")
    mapping = read_json(path)
    source = resolve(mapping["source_test_path"])
    if not source or sha(source) != mapping["test_file_sha256"]:
        raise ValueError("Audited source mapping test asset differs from its frozen source")
    tree = ast.parse(source.read_text())
    methods = {(cls.name, method.name): method for cls in tree.body if isinstance(cls, ast.ClassDef)
               for method in cls.body if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))}
    expected = [(group, req) for group in ("FAIL_TO_PASS", "PASS_TO_PASS")
                for req in _normalize_test_nodes(original.get(group))]
    rows = mapping["mapping"]
    if [(row["original_group"], row["original_requirement"]) for row in rows] != expected:
        raise ValueError("Audited mapping does not preserve every original requirement")
    mapped = {"FAIL_TO_PASS": [], "PASS_TO_PASS": []}
    for row in rows:
        if not row["mapped_original_tests"]:
            raise ValueError("An original required test has no mapped source method")
        for target in row["mapped_original_tests"]:
            method = methods.get((target["class"], target["method"]))
            if method is None or method.lineno != target["line"] or method.end_lineno != target["end_line"]:
                raise ValueError("Mapped test does not identify the recorded source AST method")
            description = ast.get_docstring(method)
            short_description = description.strip().split("\n")[0] if description else None
            if row["mapping_kind"] == "original_unittest_shortDescription":
                if short_description != row["original_requirement"]:
                    raise ValueError("Mapped unittest description differs from its source method")
            else:
                required = f'{target["method"]} (__main__.{target["class"]})'
                if required != row["original_requirement"]:
                    raise ValueError("Mapped class/method differs from its original unittest ID")
            pytest_id = f'{source.name}::{target["class"]}::{target["method"]}'
            if pytest_id != target["pytest_id"]:
                raise ValueError("Mapped pytest ID differs from the original source method")
            if pytest_id not in mapped[row["original_group"]]:
                mapped[row["original_group"]].append(pytest_id)
    if any(tuple(mapped[group]) != _normalize_test_nodes(adapted.get(group)) for group in mapped):
        raise ValueError("Actual adapted requirements differ from the conservative reviewed mapping")
    if len(expected) != approval["original_requirement_count"] or sum(map(len, mapped.values())) != approval["adapted_required_node_count"]:
        raise ValueError("Audited requirement mapping count changed")
    return {"approval": approval, "mapping_path": label(path), "mapping_sha256": sha(path),
            "source_test_path": label(source), "source_test_sha256": sha(source),
            "mapped_requirements": mapped, "original_requirement_count": len(expected),
            "adapted_requirement_count": sum(map(len, mapped.values()))}


def normalize(report_path: Path, catalog: dict, approvals: dict) -> dict:
    report = read_json(report_path)
    if not isinstance(report, dict):
        raise ValueError("A replay report must be a JSON object")
    instance_id = report.get("instance_id", "")
    entry = catalog.get(instance_id)
    reasons = []
    run = find_run(report)
    summary_path = run / "run-summary.json" if run else None
    summary = read_json(summary_path) or {}
    if not summary:
        reasons.append("actual_run_summary_missing")
    if not entry:
        reasons.append("outside_reviewed_training_selection")
    original_path = resolve(entry["source_task_path"] if entry else next((report.get(key)
                            for key in ("original_task", "raw_original_task", "source_task_path",
                                        "original_source_task_path") if report.get(key)), None))
    acquisition_root = ROOT / "competition/vfarl/acquisition"
    permitted_source = bool(original_path and original_path.is_relative_to(acquisition_root))
    if not permitted_source:
        reasons.append("source_task_outside_allowed_training_acquisition")
    original = unwrap_task(read_json(original_path), instance_id) if permitted_source else {}
    if not original:
        reasons.append("exact_original_task_row_unavailable")
    if entry and original_path and sha(original_path) != entry["source_task_sha256"]:
        reasons.append("original_task_blob_differs_from_reviewed_source_pin")
    if entry:
        review_path = resolve(entry.get("source_review_path"))
        if not review_path or not review_path.is_file() or sha(review_path) != entry.get("source_review_sha256"):
            reasons.append("mechanism_source_review_changed_or_unavailable")
    adapted_path = resolve(report.get("adapted_task"))
    if not adapted_path and run:
        adapted_path = resolve(run.parent / "adapted-task.json")
    adapted = read_json(adapted_path) or {}
    if not adapted:
        reasons.append("actual_adapted_task_unavailable")
    elif summary.get("source_task_sha256") != sha(adapted_path):
        reasons.append("adapted_task_changed_since_recorded_run")
    if adapted.get("instance_id") != instance_id or summary.get("instance_id") != instance_id:
        reasons.append("instance_id_binding_missing_or_different")
    original_description = original.get("problem_statement", "")
    adapted_description = adapted.get("problem_statement", "")
    if not isinstance(original_description, str) or not isinstance(adapted_description, str):
        raise ValueError("Task descriptions must be explicit text")
    if not adapted_description.strip():
        reasons.append("empty_policy_task_description_requires_explicit_curation_and_actual_replay")
    visible_prompt_profile = report.get("visible_prompt_profile")
    if original_description != adapted_description:
        profile = visible_prompt_profile or {}
        # A report cannot approve its own replacement task description. Bind
        # the complete versioned disclosure to the separately reviewed catalog.
        reviewed_profile = entry.get("visible_prompt_profile") if entry else None
        declared = (profile.get("kind") == "curator_authored_behavioral_clarification"
            and isinstance(reviewed_profile, dict) and profile == reviewed_profile
            and profile.get("reviewed_by")
            and profile.get("original_problem_statement_sha256") == hashlib.sha256(original_description.encode()).hexdigest()
            and profile.get("curated_problem_statement_sha256") == hashlib.sha256(adapted_description.encode()).hexdigest()
            and profile.get("curated_prompt") == adapted_description)
        if not declared:
            reasons.append("policy_description_changed_without_versioned_source_disclosure")
    else:
        visible_prompt_profile = {"kind": "verbatim_publisher_description",
                                  "original_problem_statement_sha256": hashlib.sha256(original_description.encode()).hexdigest()}
    original_requirements = {key: _normalize_test_nodes(original.get(key))
                             for key in ("FAIL_TO_PASS", "PASS_TO_PASS")}
    unchanged = bool(original) and all(_normalize_test_nodes(adapted.get(key)) == value
                                       for key, value in original_requirements.items())
    mapping = approved_mapping(instance_id, original, adapted, approvals)
    requirements_preserved = unchanged or bool(mapping)
    if not requirements_preserved or report.get("original_verifier_requirements_relaxed") is True:
        reasons.append("original_required_nodes_not_preserved_exactly")
    if entry and adapted.get("base_commit") != entry["execution_commit"]:
        reasons.append("adapted_execution_commit_differs_from_source_pin")
    if entry and entry["canonical_repo"].lower() in EXCLUDED_REPOS:
        reasons.append("competition_repository_excluded_from_training")
    targets = summary.get("verification_target_files")
    target_evidence = "recorded_actual_run_summary"
    if targets is None and adapted and summary.get("task_data_adapter") in {None, "Task"}:
        # Early official runs predate this explicit metadata field. Their pinned,
        # unchanged task bytes determine the original official Task property.
        fields = {field.name for field in dataclasses.fields(Task)}
        targets = Task(**{key: value for key, value in adapted.items() if key in fields}).test_files
        target_evidence = "official_Task_property_from_hash_matched_historical_task"
    targets = targets or []
    file_scope = bool(targets) and all(isinstance(target, str) and "::" not in target
                                      and not target.startswith("-") for target in targets)
    explicit_scope = summary.get("verification_execution_scope", report.get("verification_execution_scope"))
    if explicit_scope and explicit_scope not in {"file_scope", "official_file_scope"}:
        file_scope = False
    if not file_scope:
        reasons.append("not_strict_official_file_scope")
    actual_targets = [p for p in targets if p.endswith(".py") and Path(p).name != "conftest.py"] or targets
    required_files = {node.split("::", 1)[0] for nodes in original_requirements.values()
                      for node in nodes if "::" in node}
    missing_files = sorted(required_files - set(actual_targets))
    if missing_files:
        reasons.append("original_required_source_files_not_executed")
    compatibility = parent_compatibility(summary)
    if not compatibility["compatible"]:
        reasons.append("parent_ADK_runtime_incompatible_or_unrecorded")
    recipe_path = run / "recipe.json" if run else None
    messages_path = run / "messages.json" if run else None
    recipe = read_json(recipe_path) or {}
    messages = read_json(messages_path) or []
    if recipe.get("evidence_kind") != "human_reviewed_off_policy_demonstration" or not all(
            recipe.get(key) for key in ("reviewed_by", "review_notes")):
        reasons.append("reviewed_human_recipe_missing")
    if not messages:
        reasons.append("actual_messages_missing")
    events_path = run / "tool-events.jsonl" if run else None
    events = [json.loads(line) for line in events_path.read_text().splitlines()
              if line.strip()] if events_path and events_path.exists() else []
    actions = recipe.get("actions", [])
    calls = [call for message in messages for call in message.get("tool_calls", [])]
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    paired = len(events) == len(actions) == len(calls) == len(tool_messages) and bool(events)
    if paired:
        paired = all(event["tool"] == action["tool"] == call["function"]["name"]
                     and event["arguments"] == action["arguments"]
                     and json.loads(call["function"]["arguments"]) == action["arguments"]
                     and call["id"] == event["tool_call_id"] == observation["tool_call_id"]
                     and json.loads(observation["content"]) == event["response"]
                     for event, action, call, observation in zip(events, actions, calls, tool_messages))
    if not paired:
        reasons.append("actual_recipe_events_and_messages_do_not_align")
    if not summary.get("six_tools_only") or any(event.get("tool") not in SIX_TOOLS for event in events):
        reasons.append("official_six_tool_execution_not_established")
    if summary.get("backend") != "official_SubprocessManager_noncontainer":
        reasons.append("actual_local_official_backend_unrecorded")
    reports = {}
    # A nested original pytest.ini can make JUnit addresses relative to a test
    # directory. This collector annotation preserves every original ID; it is
    # accepted only with the exact config bytes in the actual source snapshot.
    junit_root_profile = report.get("baseline_junit_source_root")
    junit_source_root = ""
    if junit_root_profile:
        candidate_root = Path(junit_root_profile["root"])
        if candidate_root.is_absolute() or ".." in candidate_root.parts or not junit_root_profile.get("reviewed_by"):
            raise ValueError("JUnit source root requires a reviewed relative source directory")
        config_file = str(candidate_root / "pytest.ini")
        with tarfile.open(run / "source-snapshot.tgz") as snapshot:
            config_bytes = snapshot.extractfile(config_file).read()
        if hashlib.sha256(config_bytes).hexdigest() != junit_root_profile["config_sha256"]:
            raise ValueError("Original nested pytest config differs from the actual snapshot")
        if not all(Path(target).is_relative_to(candidate_root) for target in targets):
            raise ValueError("Reviewed JUnit root must contain every complete verification target")
        junit_source_root = str(candidate_root)
    for phase in ("baseline", "repaired"):
        paths = list(run.glob(f"{phase}-_swegemma_junit_*.xml")) if run else []
        if len(paths) != 1:
            reasons.append(f"{phase}_complete_JUnit_unavailable_or_ambiguous")
            continue
        reports[phase] = junit_report(paths[0], original, adapted, junit_source_root)
        if mapping:
            mapped_original = dict(original, **mapping["mapped_requirements"])
            mapped_report = junit_report(paths[0], mapped_original, adapted)
            reports[phase]["audited_source_id_mapping_official_junit_validation"] = mapped_report[
                "original_official_junit_validation"]
            reports[phase]["audited_source_id_mapping_F2P_evidence"] = mapped_report["original_F2P_evidence"]
            reports[phase]["all_audited_mapped_F2P_observed_failed"] = mapped_report[
                "all_original_F2P_observed_failed"]
    baseline = read_json(run / "baseline-official-verification.json") if run else None
    repaired = read_json(run / "official-verification.json") if run else None
    if not baseline or baseline.get("resolved") is not False or not summary.get("baseline_official_verify_task_invoked"):
        reasons.append("actual_official_baseline_failure_unavailable")
    baseline_f2p_failed = reports.get("baseline", {}).get("all_original_F2P_observed_failed")
    if mapping:
        baseline_f2p_failed = reports.get("baseline", {}).get("all_audited_mapped_F2P_observed_failed")
    baseline_evidence_accepted = bool(baseline_f2p_failed)
    baseline_label_review = entry.get("baseline_label_review") if entry else None
    baseline_evidence_profile = "all_original_F2P_failed" if not mapping else "all_audited_mapped_F2P_failed"
    if not baseline_evidence_accepted and baseline_label_review:
        evidence = reports.get("baseline", {}).get("original_F2P_evidence", [])
        counts = {"failed": sum(item["actual_failure_observed"] for item in evidence),
                  "passed": sum(bool(item["matching_cases"]) and all(case["outcome"] == "passed"
                                for case in item["matching_cases"]) for item in evidence)}
        counts["unmatched_or_other"] = len(evidence) - counts["failed"] - counts["passed"]
        # Source labels can mark already-passing cases F2P. Only a specifically
        # reviewed task and exact baseline XML may use this evidence profile;
        # every original repaired requirement is still checked by the verifier.
        baseline_evidence_accepted = bool(unchanged and baseline_label_review.get("reviewed_by")
            and baseline_label_review.get("original_task_sha256") == sha(original_path)
            and baseline_label_review.get("baseline_xml_sha256") == reports.get("baseline", {}).get("sha256")
            and counts == baseline_label_review.get("actual_original_F2P_counts")
            and len(evidence) == len(original_requirements["FAIL_TO_PASS"])
            and counts["failed"] > 0 and counts["unmatched_or_other"] == 0)
        if baseline_evidence_accepted:
            baseline_evidence_profile = "audited_source_F2P_label_noise"
    if not baseline_evidence_accepted:
        reasons.append("original_F2P_failure_not_observed_in_baseline_JUnit")
    repaired_ok = bool(repaired and repaired.get("resolved") is True
                       and summary.get("resolved_by_local_official_verify_task") is True
                       and summary.get("official_verify_task_invoked"))
    if not repaired_ok:
        reasons.append("actual_local_official_repair_not_resolved")
    strict_ids_passed = bool(unchanged and reports.get("repaired", {}).get(
        "original_official_junit_validation", [False])[0])
    mapped_ids_passed = bool(mapping and reports.get("repaired", {}).get(
        "audited_source_id_mapping_official_junit_validation", [False])[0])
    if mapping and reports.get("repaired", {}).get("tests") != mapping["approval"]["complete_file_case_count"]:
        reasons.append("audited_source_mapping_full_file_collection_changed")
    if not strict_ids_passed and not mapped_ids_passed:
        reasons.append("original_official_JUnit_matcher_rejects_repaired_report")
    result = dict(summary)
    result.update({"schema_version": "canonical-official-replay-v1", "instance_id": instance_id,
                   "original_report": label(report_path), "original_report_sha256": sha(report_path),
                   "actual_run_directory": label(run), "actual_run_summary": label(summary_path),
                   "actual_run_summary_sha256": sha(summary_path) if summary_path and summary_path.exists() else None,
                   "original_task": label(original_path),
                   "original_task_sha256": sha(original_path) if original_path and original_path.exists() else None,
                   "adapted_task": label(adapted_path), "recipe": label(recipe_path),
                   "messages": label(messages_path), "reports": reports,
                   "original_verifier_requirements_relaxed": not requirements_preserved,
                   "required_node_identifiers_verbatim_preserved": unchanged,
                   "strict_verbatim_required_ids_passed": strict_ids_passed,
                   "source_requirement_profile": "audited_unittest_to_pytest" if mapping else "verbatim_required_ids",
                   "audited_source_id_mapping": mapping,
                   "adapted_FAIL_TO_PASS_count": len(_normalize_test_nodes(adapted.get("FAIL_TO_PASS"))),
                   "adapted_PASS_TO_PASS_count": len(_normalize_test_nodes(adapted.get("PASS_TO_PASS"))),
                   "source_FAIL_TO_PASS_count": len(original_requirements["FAIL_TO_PASS"]),
                   "source_PASS_TO_PASS_count": len(original_requirements["PASS_TO_PASS"]),
                   "verification_execution_scope": "official_file_scope" if file_scope else "other_or_unknown",
                   "verification_target_files": targets,
                   "verification_target_evidence": target_evidence,
                   "baseline_junit_source_root": junit_root_profile,
                   "required_source_files_missing_from_actual_pytest_targets": missing_files,
                   "parent_ADK_compatibility": compatibility,
                   "mechanism": ({"adapter": entry["primary_adapter"],
                       "submechanism": entry["submechanism"], "evidence": entry["mechanism_evidence"]}
                       if entry else report.get("mechanism")),
                   "source_report_mechanism": report.get("mechanism"),
                   "evidence_kind": summary.get("evidence_kind", recipe.get("evidence_kind")),
                   "demonstration_authorship": "Codex agents specified and reviewed the scripted actions; these are not Gemma policy samples or end-user manual annotations. The legacy human_reviewed schema label is retained for raw-format compatibility.",
                   "source_reported_eligibility": report.get("human_off_policy_sft_demonstration_available",
                       report.get("training_eligibility")),
                   "source_runtime_profile": report.get("source_runtime_profile"),
                   "source_runtime_profile_sha256": report.get("source_runtime_profile_sha256"),
                   "runtime_equivalence_limit": report.get("runtime_equivalence_limit"),
                   "independent_runtime_followup_review": report.get("independent_runtime_followup_review"),
                   "complete_official_replay": bool(summary and baseline and repaired
                       and reports.get("baseline") and reports.get("repaired")
                       and summary.get("official_verify_task_invoked")
                       and summary.get("baseline_official_verify_task_invoked")),
                   "baseline_verified_original_F2P_failure": bool(baseline
                       and baseline.get("resolved") is False
                       and baseline_f2p_failed),
                   "baseline_evidence_profile": baseline_evidence_profile,
                   "baseline_label_review": baseline_label_review,
                   "baseline_repair_causality_evidence_accepted": baseline_evidence_accepted,
                   "primary_adapter": entry["primary_adapter"] if entry else report.get("primary_adapter"),
                   "submechanism": entry["submechanism"] if entry else report.get("submechanism"),
                   "repair_family_id": entry.get("repair_family_id") if entry else None,
                   "snapshot_group_id": entry.get("snapshot_group_id") if entry else None,
                   "canonical_repo": entry.get("canonical_repo") if entry else report.get("canonical_repo"),
                   "source_dataset": entry.get("source_dataset") if entry else None,
                   "source_revision": entry.get("source_revision") if entry else None,
                   "physical_split": entry.get("physical_split") if entry else None,
                   "source_review_path": entry.get("source_review_path") if entry else None,
                   "source_review_sha256": entry.get("source_review_sha256") if entry else None,
                   "source_curator_notes": (entry.get("curator_limitations_and_notes") or entry.get("root_second_read")) if entry else None,
                   "visible_prompt_profile": visible_prompt_profile,
                   "tool_events": len(events),
                   "eligible_for_human_off_policy_SFT": not reasons,
                   "eligible_with_audited_source_id_mapping": bool(mapping and not reasons),
                   "ineligibility_reasons": list(dict.fromkeys(reasons)),
                   "training_eligibility": {"human_off_policy_tool_demonstration": not reasons,
                       "strict_official_file_scope": file_scope,
                       "original_required_test_identities_preserved": requirements_preserved,
                       "required_node_identifiers_verbatim_preserved": unchanged,
                       "model_on_policy_RL": False, "same_state_action_preference_pair": False,
                       "competition_Docker_equivalence": False, "GPU_training_started": False}})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=REPLAY)
    parser.add_argument("--reports", nargs="+", type=Path,
                        help="Only these explicitly selected train replay reports are opened")
    parser.add_argument("--catalog", type=Path, default=REPLAY.parent / "pilot96/catalog.json")
    parser.add_argument("--supplemental-catalog", type=Path,
                        default=REPLAY.parent / "acquisition/supplemental/catalog.json",
                        help="Explicit independently reviewed additional train identities; never inferred from a replay report")
    parser.add_argument("--out", type=Path, default=REPLAY / "artifacts/canonical-results")
    parser.add_argument("--manifest", type=Path, default=REPLAY / "canonical-results-manifest.json")
    parser.add_argument("--approved-mappings", type=Path,
                        default=REPLAY / "approved-source-requirement-mappings.json")
    args = parser.parse_args()
    catalog_blob = read_json(args.catalog)
    supplemental_blob = read_json(args.supplemental_catalog) or {"tasks": []}
    all_entries = list(catalog_blob["tasks"])
    existing_ids = {row["instance_id"] for row in all_entries}
    for row in supplemental_blob["tasks"]:
        if row["instance_id"] in existing_ids or row.get("physical_split") != "train" or row.get("use_partition") != "train_only":
            raise ValueError("Supplemental entries must be distinct, explicitly reviewed training identities")
        existing_ids.add(row["instance_id"])
        all_entries.append(row)
    catalog = {row["instance_id"]: row for row in all_entries}
    source_pins = [{key: row.get(key) for key in SOURCE_PIN_FIELDS}
                   for row in sorted(all_entries, key=lambda row: row["instance_id"])]
    source_pins_hash = hashlib.sha256(json.dumps(source_pins, ensure_ascii=False, sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest()
    approval_blob = read_json(args.approved_mappings) or {"approvals": []}
    approvals = {row["instance_id"]: row for row in approval_blob["approvals"]}
    records = []
    report_paths = args.reports if args.reports is not None else args.reports_dir.glob("*-replay-result.json")
    for path in sorted(set(report_paths)):
        if args.reports is not None:
            named = read_json(path)
            if not isinstance(named, dict) or named.get("instance_id") not in catalog:
                raise ValueError("An explicitly named report must belong to this reviewed train selection")
        try:
            result = normalize(path, catalog, approvals)
        except (OSError, ValueError, KeyError, TypeError, ET.ParseError) as error:
            result = {"schema_version": "canonical-official-replay-v1", "original_report": label(path),
                      "eligible_for_human_off_policy_SFT": False,
                      "ineligibility_reasons": ["actual_artifact_normalization_incomplete"],
                      "normalization_error": f"{type(error).__name__}: {error}"}
        target = args.out / path.name
        eligible = result["eligible_for_human_off_policy_SFT"]
        if eligible:
            write_json(target, result)
        elif target.exists():
            previous = read_json(target)
            if previous.get("schema_version") == "canonical-official-replay-v1":
                target.unlink()
        records.append({"instance_id": result.get("instance_id"),
                        "original_result_path": label(path),
                        "canonical_result_path": label(target) if eligible else None,
                        "canonical_result_sha256": sha(target) if eligible else None,
                        "actual_run_path": result.get("actual_run_directory"),
                        "eligible": result["eligible_for_human_off_policy_SFT"],
                        "ineligibility_reasons": result["ineligibility_reasons"],
                        "complete_official_replay": result.get("complete_official_replay", False),
                        "baseline_verified_original_F2P_failure": result.get(
                            "baseline_verified_original_F2P_failure", False),
                        "resolved_by_local_official_verify_task": result.get(
                            "resolved_by_local_official_verify_task"),
                        "tool_events": result.get("tool_events", 0),
                        "primary_adapter": result.get("primary_adapter"),
                        "submechanism": result.get("submechanism"),
                        "strict_verbatim_required_ids_passed": result.get("strict_verbatim_required_ids_passed", False),
                        "eligible_with_audited_source_id_mapping": result.get("eligible_with_audited_source_id_mapping", False),
                        "source_requirement_profile": result.get("source_requirement_profile"),
                        "baseline_evidence_profile": result.get("baseline_evidence_profile"),
                        "visible_prompt_profile": result.get("visible_prompt_profile", {}).get("kind"),
                        "normalization_error": result.get("normalization_error")})
    # Retries may publish a new report while retaining the earlier negative report.
    # Each task still contributes at most one episode. Prefer fully admitted
    # evidence, then complete double-Phase evidence, then the newest report.
    report_records = records
    grouped = {}
    for row in report_records:
        identity = row.get("instance_id") or row["original_result_path"]
        grouped.setdefault(identity, []).append(row)
    records = []
    for identity, candidates in sorted(grouped.items()):
        selected = max(candidates, key=lambda row: (
            row["eligible"], row["complete_official_replay"],
            (ROOT / row["original_result_path"]).stat().st_mtime_ns))
        selected = dict(selected)
        selected["retained_report_candidates"] = [row["original_result_path"] for row in candidates]
        records.append(selected)
    manifest = {"schema_version": "canonical-official-replay-manifest-v1",
                "catalog": label(args.catalog.resolve()),
                "primary_catalog_task_count": len(catalog_blob["tasks"]),
                "supplemental_catalog": label(args.supplemental_catalog.resolve()) if args.supplemental_catalog.exists() else None,
                "supplemental_catalog_sha256": sha(args.supplemental_catalog) if args.supplemental_catalog.exists() else None,
                "supplemental_catalog_task_count": len(supplemental_blob["tasks"]),
                "catalog_source_pins_sha256": source_pins_hash,
                "catalog_source_pin_fields": list(SOURCE_PIN_FIELDS),
                "catalog_source_pins": source_pins,
                "catalog_hash_scope": "Sorted immutable source identities, mechanism assignments and reviewed visible-prompt profiles; mutable replay progress excluded to avoid a catalog/manifest hash cycle",
                "raw_reports_unchanged": True, "original_official_verifier_unchanged": True,
                "source_report_count": len(report_records), "unique_reported_tasks": len(records),
                "task_selection_policy": "One episode per instance ID; admitted evidence preferred, then complete dual-Phase evidence, then newest report. Earlier reports and raw observations remain unchanged.",
                "source_reports": report_records,
                "eligible_count": sum(row["eligible"] for row in records),
                "strict_verbatim_eligible_count": sum(row["eligible"] and row["strict_verbatim_required_ids_passed"] for row in records),
                "audited_mapping_eligible_count": sum(row["eligible_with_audited_source_id_mapping"] for row in records),
                "baseline_evidence_profile_counts_among_admitted": dict(Counter(row["baseline_evidence_profile"] for row in records if row["eligible"])),
                "visible_prompt_profile_counts_among_admitted": dict(Counter(row["visible_prompt_profile"] for row in records if row["eligible"])),
                "approved_source_mappings": label(args.approved_mappings.resolve()),
                "approved_source_mappings_sha256": sha(args.approved_mappings),
                "ineligibility_reason_counts": dict(Counter(reason for row in records
                    for reason in row["ineligibility_reasons"])),
                "normalizer": label(Path(__file__).resolve()),
                "normalizer_sha256": sha(Path(__file__).resolve()),
                "eligible_results": [row["canonical_result_path"] for row in records if row["eligible"]],
                "tasks": records, "model_rollouts": 0, "verified_focal_pairs": 0,
                "GPU_training_started": False}
    write_json(args.manifest, manifest)
    print(json.dumps({key: manifest[key] for key in ("source_report_count", "eligible_count",
                                                  "ineligibility_reason_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
