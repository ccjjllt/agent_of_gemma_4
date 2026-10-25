"""Prepare reviewed Moto train demonstrations and retain actual CPU run evidence.

Publisher solution hunks are collector-only recipe material. Original task tests,
requirements and observations are never authored or weakened by this collector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
REPLAY = Path(__file__).resolve().parent
ART = REPLAY / "artifacts/moto-cpu"
REVIEWS = {
    "7335": "Cognito Identity GetId/GetOpenIdToken are public operations; include both exact service operations in the authentication exemption list.",
    "5200": "Secrets Manager ARN random suffix must contain six ASCII letters; repair the generator length rather than hardcoding an ARN.",
    "5348": "DynamoDB get_item must reject an empty key AttributeValue before dispatch; retain the original key name in its validation error.",
    "6492": "Route53 create hosted zone must return the Location header derived from the newly created zone identifier.",
    "5486": "EC2 must reject a missing security group before creating or storing a reservation, with the service-specific not-found exception.",
    "6159": "API Gateway resource path validation must admit colon while retaining the existing anchored character and path-parameter constraints.",
    "4934": "Lambda publish-version must report an unknown function as a region-aware ResourceNotFoundException/404; retain ARN decoding and routing for both names and ARNs.",
    "6755": "SES MAIL-FROM validation must use verified identities and the domain component of an email identity, also accepting domain-only identities.",
    "7102": "KMS encrypt plaintext accepts lengths 1 through 4096; reject an oversized payload before allocating encryption state.",
    "6609": "KMS signing algorithm metadata and validation must include the original supported SM2DSA value.",
    "7168": "Scheduler missing get/delete operations must raise ScheduleNotFound with the requested schedule name, including an explicit delete guard.",
    "7061": "Autoscaling describe_tags must apply key and value filters to the existing tag stream, preserving prior filters and their conjunction.",
    "7111": "Lambda omitted PackageType defaults to Zip at construction, so list/get responses share the correct package metadata.",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def reviewed_hunks(patch, workspace):
    """Convert already reviewed publisher hunks to exact, unique tool edits."""
    edits, path, old, new, in_hunk = [], None, [], [], False

    def flush():
        if not in_hunk or old == new:
            return
        before, after = "".join(old), "".join(new)
        current = (workspace / path).read_text()
        if not before or current.count(before) != 1:
            raise ValueError(f"Reviewed hunk is not a unique source anchor: {path}")
        edits.append({"tool": "edit_file", "arguments": {
            "filepath": path, "old_string": before, "new_string": after}, "expected_status": "ok"})

    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            flush()
            in_hunk, old, new = False, [], []
        elif line.startswith("+++ b/"):
            path = line[6:].strip()
        elif line.startswith("@@ "):
            flush()
            in_hunk, old, new = True, [], []
        elif in_hunk and not line.startswith("\\"):
            if line.startswith(" "):
                old.append(line[1:]); new.append(line[1:])
            elif line.startswith("-"):
                old.append(line[1:])
            elif line.startswith("+"):
                new.append(line[1:])
    flush()
    if not edits:
        raise ValueError("No reviewed production edit")
    # Apply to collector memory only, to catch adjacent/overlapping anchors.
    simulated = {}
    for edit in edits:
        a = edit["arguments"]
        text = simulated.setdefault(a["filepath"], (workspace / a["filepath"]).read_text())
        if text.count(a["old_string"]) != 1:
            raise ValueError("Sequential edit anchor changed")
        simulated[a["filepath"]] = text.replace(a["old_string"], a["new_string"], 1)
    return edits


def prepare(number):
    sources = json.loads((ART / "sources.json").read_text())
    source = next(s for s in sources if s["instance_id"] == f"getmoto__moto-{number}")
    raw_path = Path(source["source_task"])
    if sha(raw_path.read_bytes()) != source["source_task_sha256"]:
        raise ValueError("Original source task changed")
    row = json.loads(raw_path.read_text())["row"]
    workspace = Path(source["workspace_source"])
    requirements = list(row["FAIL_TO_PASS"]) + list(row["PASS_TO_PASS"])
    task = {k: row[k] for k in ("instance_id", "repo", "base_commit", "problem_statement", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS")}
    task["patch"] = ""
    task["verification_target_files"] = sorted({node.split("::", 1)[0] for node in requirements})
    folder = ART / row["instance_id"]
    save(folder / "task.json", task)
    edits = reviewed_hunks(row["patch"], workspace)
    paths = list(dict.fromkeys(e["arguments"]["filepath"] for e in edits))
    # A real source import and compilation is a bounded diagnostic; the original
    # baseline/repaired Phase B runs supply task outcome evidence separately.
    diagnostic = "python3 - <<'PYREPLAY'\nfrom pathlib import Path\nimport hashlib, importlib, py_compile\n"
    for path in paths:
        module = path.removesuffix(".py").replace("/", ".")
        diagnostic += (f"m=importlib.import_module({module!r})\nactual=Path(m.__file__).resolve()\n"
                       f"expected=(Path.cwd()/{path!r}).resolve()\nassert actual==expected,(actual,expected)\n"
                       "print('source_binding',actual,'sha256',hashlib.sha256(actual.read_bytes()).hexdigest())\n"
                       "py_compile.compile(str(actual),doraise=True)\n")
    diagnostic += "PYREPLAY"
    action = {"tool": "run_command", "arguments": {"command": diagnostic},
              "expected_status": "ok", "reviewed_command_sha256": sha(diagnostic.encode())}
    reads = [{"tool": "read_file", "arguments": {"filepath": path,
              "start_line": 1,
              "end_line": min(200, len((workspace / path).read_text().splitlines()))}, "expected_status": "ok"}
             for path in paths]
    # Show the actual hunk locations rather than only the start of long modules.
    for read, path in zip(reads, paths):
        anchor = next(e["arguments"]["old_string"] for e in edits if e["arguments"]["filepath"] == path)
        start = (workspace / path).read_text().split(anchor, 1)[0].count("\n") + 1
        read["arguments"].update(start_line=max(1,start-5), end_line=start+min(120,anchor.count("\n")+8))
    recipe = {"evidence_kind": "human_reviewed_off_policy_demonstration",
              "reviewed_by": "Codex manual source hunk and original task requirement review 2026-10-25",
              "review_notes": REVIEWS[number] + " Publisher solution is collector-only. Original F2P/P2P and test_patch are retained verbatim; no model inference. Baseline and repaired clean Phase B determine eligibility.",
              "actions": [{"tool":"get_status","arguments":{},"expected_status":"ok"}] + reads + [action] + edits + [action, {"tool":"get_status","arguments":{},"expected_status":"ok"}, {"tool":"submit_patch","arguments":{},"expected_status":"ok"}],
              "final_response": REVIEWS[number] + " Patch submitted; source imports and compilation completed. Full task verification is performed separately by the harness."}
    save(REPLAY / f"moto-{number}-reviewed-recipe.json", recipe)
    save(REPLAY / f"moto-{number}-source-preparation.json", {
        **source, "original_task": str(raw_path.relative_to(ROOT)),
        "original_task_sha256": sha(raw_path.read_bytes()),
        "adapted_task": str((folder / "task.json").relative_to(ROOT)),
        "source_train_revision": "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb",
        "original_verifier_requirements_relaxed": False,
        "required_node_identifiers_verbatim_preserved": True,
        "test_patch_sha256": sha(row["test_patch"].encode()),
        "verification_target_files": task["verification_target_files"],
        "original_FAIL_TO_PASS_count": len(row["FAIL_TO_PASS"]),
        "original_PASS_TO_PASS_count": len(row["PASS_TO_PASS"]),
        "mechanism_review": REVIEWS[number], "publisher_patch_semantics": "repair_solution_collector_only"})
    return folder


def record(number, run):
    from swegemma.harness.verification import _validate_junit_xml
    prep = json.loads((REPLAY / f"moto-{number}-source-preparation.json").read_text())
    task = json.loads((ROOT / prep["adapted_task"]).read_text())
    output = ART / f"getmoto__moto-{number}" / run
    result = json.loads((output / "run-summary.json").read_text())
    reports = {}
    for label in ("baseline", "repaired"):
        paths = list(output.glob(label + "-*_junit_*.xml"))
        if len(paths) != 1:
            raise ValueError(f"Need actual complete JUnit evidence for {label}")
        path = paths[0]
        xml = ET.fromstring(path.read_text())
        suites = [xml] if xml.tag == "testsuite" else xml.findall(".//testsuite")
        counts = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
        reports[label] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path.read_bytes()), **counts,
            "original_official_junit_validation": _validate_junit_xml(path.read_text(), tuple(task["FAIL_TO_PASS"]), tuple(task["PASS_TO_PASS"]), task["test_patch"])}
    result.update({"reports": reports, "original_task": prep["original_task"],
        "adapted_task": prep["adapted_task"], "source_preparation": f"competition/vfarl/replay/moto-{number}-source-preparation.json",
        "recipe": f"competition/vfarl/replay/moto-{number}-reviewed-recipe.json",
        "replay_output": str(output.relative_to(ROOT)),
        "source_train_revision": prep["source_train_revision"], "bug_branch_revision": prep["execution_commit"],
        "original_verifier_requirements_relaxed": False,
        "required_node_identifiers_verbatim_preserved": True,
        "source_FAIL_TO_PASS_count": prep["original_FAIL_TO_PASS_count"],
        "source_PASS_TO_PASS_count": prep["original_PASS_TO_PASS_count"],
        "mechanism_review": prep["mechanism_review"],
        "training_eligibility": {"human_off_policy_tool_demonstration": result["resolved_by_local_official_verify_task"] is True and reports["baseline"]["original_official_junit_validation"][0] is False and reports["repaired"]["original_official_junit_validation"][0] is True,
                                  "model_on_policy_RL": False,"same_state_action_preference_pair":False,"GPU_training_started":False}})
    save(REPLAY / f"moto-{number}-replay-result.json", result)
    print(json.dumps({"instance_id": result["instance_id"], "resolved": result["resolved_by_local_official_verify_task"], "reports": reports}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", choices=REVIEWS)
    parser.add_argument("--record-run")
    args = parser.parse_args()
    record(args.number, args.record_run) if args.record_run else prepare(args.number)
