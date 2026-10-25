"""Publish actual R4 replay evidence and source review, without rewriting raw runs."""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
REPLAY = ROOT / "competition/vfarl/replay"
sys.path.insert(0, str(REPLAY))
from swegemma.harness.verification import _validate_junit_xml
from normalize_results import exact_required_case

HERE = Path(__file__).resolve().parent
ART = REPLAY / "artifacts/expand-compat"
KEYS = ("cloudpickle", "apispec-ref", "apispec-null")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def publish():
    reviews, reports = [], []
    for key in KEYS:
        prep_path = REPLAY / ("expand-r4-" + key + "-source-preparation.json")
        prep = json.loads(prep_path.read_text())
        output = ART / (key + "-run-001")
        summary_path = output / "run-summary.json"
        if not summary_path.exists():
            raise ValueError("Actual official run has not completed: " + key)
        summary = json.loads(summary_path.read_text())
        adapted = json.loads((ROOT / prep["adapted_task"]).read_text())
        source = json.loads((ROOT / prep["original_task"]).read_text())["row"]
        if any(source[name] != adapted[name] for name in ("instance_id", "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS")):
            raise ValueError("Original task visible text or requirements changed")
        phase_reports = {}
        for phase in ("baseline", "repaired"):
            found = list(output.glob(phase + "-_swegemma_junit_*.xml"))
            if len(found) != 1:
                raise ValueError("A complete original JUnit report is required: " + phase)
            path = found[0]
            tree = ET.fromstring(path.read_text())
            suites = [tree] if tree.tag == "testsuite" else tree.findall(".//testsuite")
            counts = {name: sum(int(s.attrib.get(name, 0)) for s in suites)
                      for name in ("tests", "failures", "errors", "skipped")}
            cases = [{"classname": c.get("classname", ""), "name": c.get("name", ""),
                      "outcome": next((o for o in ("failure", "error", "skipped") if c.find(o) is not None), "passed")}
                     for c in tree.findall(".//testcase")]
            evidence = []
            for requirement in source["FAIL_TO_PASS"]:
                matched = [c for c in cases if exact_required_case(c, requirement)]
                evidence.append({"requirement": requirement, "cases": matched,
                                 "actual_failure": bool(matched) and all(c["outcome"] in {"failure", "error"} for c in matched)})
            phase_reports[phase] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), **counts,
                                    "original_official_junit_validation": list(_validate_junit_xml(path.read_text(), tuple(source["FAIL_TO_PASS"]), tuple(source["PASS_TO_PASS"]), adapted["test_patch"])),
                                    "original_F2P_evidence": evidence,
                                    "all_original_F2P_observed_failed": all(e["actual_failure"] for e in evidence)}
        recipe_path = ROOT / prep["recipe"]
        events_path = output / "tool-events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        observations = [{"index": e["index"], "actual_response": e["response"]}
                        for e in events if e["tool"] == "run_command"]
        extra = {"reports": phase_reports, "original_task": prep["original_task"], "original_task_sha256": prep["original_task_sha256"],
                 "adapted_task": prep["adapted_task"], "recipe": prep["recipe"], "replay_output": str(output.relative_to(ROOT)),
                 "source_preparation": str(prep_path.relative_to(ROOT)), "source_train_revision": prep["source_revision"],
                 "source_revision": prep["source_revision"], "canonical_repo": prep["canonical_repo"],
                 "source_dataset": prep["source_dataset"], "physical_split": "train", "primary_adapter": "R", "submechanism": "R4",
                 "execution_commit": prep["execution_commit"], "bug_branch_revision": prep["execution_commit"],
                 "source_FAIL_TO_PASS_count": prep["source_FAIL_TO_PASS_count"], "source_PASS_TO_PASS_count": prep["source_PASS_TO_PASS_count"],
                 "original_verifier_requirements_relaxed": False, "required_node_identifiers_verbatim_preserved": True,
                 "repair_family_id": prep["repair_family_id"], "snapshot_group_id": prep["snapshot_group_id"],
                 "mechanism": {"adapter": "R", "submechanism": "R4", "evidence": prep["mechanism_evidence"]},
                 "actual_mechanism_observations": observations, "patch_sha256": sha(output / "submitted.patch"),
                 "source_runtime_profile": ("Independent taskPython3.11 source-supported runtime and copied task dependencies; source-bound originalCodeType3.11 semantics, modernADK parent unaffected." if key == "cloudpickle" else "SourcePython3.12 with existing read-only task dependencies from known-good APISpec profile; actual OAS2/3.0/3.1 APIs executed."),
                 "training_eligibility": {"human_off_policy_tool_demonstration": bool(summary["resolved_by_local_official_verify_task"] and phase_reports["repaired"]["original_official_junit_validation"][0] and phase_reports["baseline"]["all_original_F2P_observed_failed"]),
                                          "model_on_policy_RL": False, "same_state_action_preference_pair": False, "GPU_training_started": False}}
        summary.update(extra)
        if key == "cloudpickle":
            summary["source_configuration_conflict"] = {
                "source_config": str((ROOT / prep["workspace"] / "tox.ini").relative_to(ROOT)),
                "source_config_sha256": sha(ROOT / prep["workspace"] / "tox.ini"),
                "original_setting": "[pytest] addopts = -s",
                "actual_residual_failure": "Official workspace pytest.ini overrides original tox.ini capture setting. Three original required file-handle cases receive pytest EncodedFile stdout/stderr, while this source reducer is registered for exact io.TextIOWrapper.",
                "decision": "Preserve full official False; do not change official flags/capture, tests or requirements to obtain True.",
            }
        target = REPLAY / ("expand-r4-" + key + "-replay-result.json")
        reports.append((target, summary))
        reviews.append({**prep, "source_review_complete": True,
                        "actual_current_run": str(output.relative_to(ROOT)), "actual_run_summary_sha256": sha(summary_path),
                        "actual_recipe_sha256": sha(recipe_path), "actual_events_sha256": sha(events_path),
                        "actual_observations": observations, "actual_reports": phase_reports,
                        "resolved_by_local_official_verify_task": summary["resolved_by_local_official_verify_task"],
                        "source_requirements_verbatim": True})
        reviews[-1]["source_license"]["declared_license"] = "BSD-3-Clause" if key == "cloudpickle" else "MIT"
        if key == "cloudpickle":
            reviews[-1]["source_configuration_conflict"] = summary["source_configuration_conflict"]
    review_path = HERE / "reviewed-candidates.json"
    review = {"date": "2026-10-25", "scope": "Explicit supplemental train R4 candidates, original96 unchanged.",
              "classification_basis": "Different required old/new runtime or OpenAPI version contracts, not version keywords. Two APISpec functions are distinct but share the upstream snapshot and converter pipeline; never split these by taskID for evaluation.",
              "rejected_candidates": ["MarkupSafe348 deletes helper and leaves incomplete def; not clean R4", "Colorlog constructor deletion is generic capability restoration, not evidence of version compatibility", "Cloudpickle0lun00yt same reducer family and broader name-reversal corruption, not additional independent task", "Environs320 mixes undefined variables with legacy default gate; not used to fill R4 by keyword"],
              "tasks": reviews}
    save(review_path, review)
    for path, report in reports:
        report.update({"source_review_path": str(review_path.relative_to(ROOT)), "source_review_sha256": sha(review_path)})
        save(path, report)
        print(json.dumps({"instance_id": report["instance_id"], "resolved": report["resolved_by_local_official_verify_task"],
                          "baseline_all_original_F2P_failed": report["reports"]["baseline"]["all_original_F2P_observed_failed"], "report": str(path.relative_to(ROOT))}), flush=True)


if __name__ == "__main__":
    publish()
