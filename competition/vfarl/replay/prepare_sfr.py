"""Prepare pinned public-training workspaces and original validation assets.

This is acquisition/data preparation, not a new validation suite. It never
executes publisher commands or uses publisher trajectory solution columns.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import tarfile
import xml.etree.ElementTree as ET
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPLAY = Path(__file__).resolve().parent
ART = REPLAY / "artifacts" / "sfr"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def acquire(repo: str, revision: str, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination.with_suffix(".tgz")
    resolution = None
    resolved = revision
    if len(revision) != 40:
        resolution_path = destination.with_suffix(".revision.json")
        resolution_url = f"https://api.github.com/repos/{repo}/commits/{revision}"
        if resolution_path.exists():
            resolution = json.loads(resolution_path.read_text())
        else:
            try:
                with urllib.request.urlopen(resolution_url, timeout=120) as response:
                    body = response.read()
                resolved_sha = json.loads(body)["sha"]
            except urllib.error.HTTPError as error:
                if error.code != 403:
                    raise
                resolution_url = f"https://github.com/{repo}/commit/{revision}.patch"
                with urllib.request.urlopen(resolution_url, timeout=120) as response:
                    body = response.read()
                match = re.match(rb"From ([0-9a-f]{40}) ", body)
                if not match:
                    raise ValueError("Commit patch did not expose a complete immutable commit")
                resolved_sha = match.group(1).decode()
            resolution = {"url": resolution_url, "sha": resolved_sha,
                          "response_sha256": digest(body)}
            save(resolution_path, resolution)
        resolved = resolution["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", resolved) or not resolved.startswith(revision):
            raise ValueError("Commit resolution differs from the source snapshot prefix")
    url = f"https://codeload.github.com/{repo}/tar.gz/{revision}"
    if not archive.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            archive.write_bytes(response.read())
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        top = members[0].name.split("/")[0]
        archived_ref = top.rsplit("-", 1)[-1]
        if archived_ref not in {revision, resolved}:
            raise ValueError(f"Archive revision mismatch: {top}, requested {revision}")
        for member in members:
            parts = Path(member.name).parts
            if not parts or parts[0] != top or ".." in parts:
                raise ValueError("Unsafe archive member")
            relative = Path(*parts[1:])
            if not relative.parts:
                continue
            target = destination / relative
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Archive member resolves outside its source workspace")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                payload = bundle.extractfile(member).read()
                if target.exists() and target.read_bytes() != payload:
                    raise ValueError(f"Existing acquisition was altered: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                target.chmod(member.mode)
            elif member.issym():
                if (Path(member.linkname).is_absolute()
                        or not (target.parent / member.linkname).resolve().is_relative_to(destination.resolve())):
                    raise ValueError("Unsafe archive symlink")
                if not target.exists():
                    target.symlink_to(member.linkname)
            else:
                raise ValueError(f"Unsupported archive member: {member.name}")
    return {"repo": repo, "requested_revision": revision, "resolved_revision": resolved,
            "archive_path": str(archive.relative_to(ROOT)), "archive_sha256": digest(archive.read_bytes()),
            "download_url": url, "immutable_url": f"https://codeload.github.com/{repo}/tar.gz/{resolved}",
            "commit_resolution": resolution}


def prepare(instance_id: str, key: str, required_node_mapping: dict[str, list[str]] | None = None) -> dict:
    catalog = json.loads((ROOT / "competition/vfarl/pilot96/catalog.json").read_text())
    entry = next(r for r in catalog["tasks"] if r["instance_id"] == instance_id)
    payload = json.loads((ROOT / entry["source_task_path"]).read_text())
    row = payload["row"] if "row" in payload else payload["rows"][0]["row"]
    smith = entry["source_dataset"] == "SWE-bench/SWE-smith-py"
    if not smith and entry["source_dataset"] != "nebius/SWE-rebench-V2":
        raise ValueError("This profile supports only explicitly audited Smith/V2 public training tasks")
    source = ART / (key + "-source")
    buggy = acquire(row["repo"], entry["execution_commit"], source)
    if smith:
        prefix = row["repo"].rsplit(".", 1)[-1]
        base_dest = ART / "upstream" / (entry["canonical_repo"].replace("/", "__") + "." + prefix)
        base = acquire(entry["canonical_repo"], prefix, base_dest)
    else:
        base_dest, base = source, buggy
    original_required = list(row["FAIL_TO_PASS"]) + list(row["PASS_TO_PASS"])
    if required_node_mapping is None:
        fail_to_pass, pass_to_pass = row["FAIL_TO_PASS"], row["PASS_TO_PASS"]
    else:
        if set(required_node_mapping) != set(original_required) or not all(required_node_mapping.values()):
            raise ValueError("Explicit node mapping must cover every original requirement")
        fail_to_pass = [mapped for node in row["FAIL_TO_PASS"] for mapped in required_node_mapping[node]]
        pass_to_pass = [mapped for node in row["PASS_TO_PASS"] for mapped in required_node_mapping[node]]
        if len(set(fail_to_pass + pass_to_pass)) != len(fail_to_pass + pass_to_pass):
            raise ValueError("The mapped required IDs must remain distinct")
    required = list(fail_to_pass) + list(pass_to_pass)
    targets = sorted({node.split("::", 1)[0] for node in required})
    files = []
    missing_patch = []
    for path in sorted(base_dest.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(base_dest)
        # Preserve complete original verification trees/fixtures and root
        # conftest. Never substitute production code for a validation target.
        if not any(part in {"tests", "test", "testing"} for part in relative.parts) and str(relative) not in {"conftest.py", "test.py"}:
            continue
        raw = path.read_bytes()
        existing = source / relative
        exists = existing.exists()
        if exists and existing.read_bytes() != raw:
            raise ValueError(f"Original verification asset differs at {relative}")
        files.append({"path": str(relative), "bytes": len(raw), "sha256": digest(raw),
                      "source_revision": base["resolved_revision"], "already_present": exists})
        if not exists:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Missing binary original asset requires explicit source hydration: {relative}") from exc
            missing_patch.extend(difflib.unified_diff([], text.splitlines(keepends=True),
                                                     fromfile="/dev/null", tofile="b/" + str(relative)))
    if not all((source / t).exists() or any(f["path"] == t for f in files) for t in targets):
        raise ValueError("An original required verification file is unavailable")
    task = {"instance_id": instance_id, "repo": row["repo"], "base_commit": entry["execution_commit"],
            "problem_statement": row["problem_statement"], "patch": "", "test_patch": "".join(missing_patch) if smith else row["test_patch"],
            "FAIL_TO_PASS": fail_to_pass, "PASS_TO_PASS": pass_to_pass,
            "verification_target_files": targets}
    adapted = ART / (key + "-task.json")
    save(adapted, task)
    (ART / (key + ("-bug-introducing.patch" if smith else "-publisher-solution.patch"))).write_text(row["patch"])
    result = {"instance_id": instance_id, "canonical_repo": entry["canonical_repo"],
              "primary_adapter": entry["primary_adapter"], "submechanism": entry["submechanism"],
              "original_task": entry["source_task_path"], "original_task_sha256": entry["source_task_sha256"],
              "source_train_revision": entry["source_revision"], "buggy_source": buggy, "upstream_source": base,
              "workspace": str(source.relative_to(ROOT)), "adapted_task": str(adapted.relative_to(ROOT)),
              "verification_target_files": targets, "verification_assets": files,
              "F2P_count": len(row["FAIL_TO_PASS"]), "P2P_count": len(row["PASS_TO_PASS"]),
              "adapted_F2P_count": len(fail_to_pass), "adapted_P2P_count": len(pass_to_pass),
              "required_node_identifiers_verbatim_preserved": required_node_mapping is None,
              "required_node_mapping": required_node_mapping,
              "source_requirements_relaxed": False, "source_bug_patch_is_solution": not smith,
              "original_patch_semantics": "bug_introducing" if smith else "solution",
              "test_patch_provenance": "frozen_original_upstream_validation_assets" if smith else "verbatim_original_public_train_task_test_patch"}
    save(REPLAY / (key + "-source-preparation.json"), result)
    print(json.dumps({k: v for k, v in result.items() if k != "verification_assets"}, ensure_ascii=False))
    return result


def write_reviewed_recipe(key: str, path: str, module: str, start: int, end: int,
                          edits: list[tuple[str, str]], observation: str,
                          review: str, final_response: str) -> Path:
    """Serialize explicitly reviewed actions; never infer a repair or reward."""
    def action(tool: str, arguments: dict, expected: str = "ok") -> dict:
        value = {"tool": tool, "arguments": arguments, "expected_status": expected}
        if tool == "run_command":
            value["reviewed_command_sha256"] = digest(arguments["command"].encode())
        return value

    binding = (
        "from pathlib import Path\nimport hashlib, importlib\n"
        f"expected=(Path.cwd() / {path!r}).resolve()\n"
        "try:\n"
        f" module = importlib.import_module({module!r})\n"
        "except Exception as import_error:\n"
        " traceback=import_error.__traceback__\n"
        " while traceback is not None:\n"
        "  file=traceback.tb_frame.f_globals.get('__file__')\n"
        "  if file and Path(file).resolve()==expected:\n"
        "   print('source_binding', str(expected), 'sha256', hashlib.sha256(expected.read_bytes()).hexdigest(), 'actual_import_failure_frame')\n"
        "  traceback=traceback.tb_next\n"
        " raise\n"
        "actual=Path(module.__file__).resolve()\n"
        'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
        "assert actual == expected, (actual, expected)\n"
    )
    command = "python3 - <<'PYREPLAY'\n" + binding + observation + "\nPYREPLAY"
    actions = [action("get_status", {}),
               action("read_file", {"filepath": path, "start_line": start, "end_line": end}),
               action("run_command", {"command": command}, "error")]
    current = (ART / (key + "-source") / path).read_text()
    for old, new in edits:
        if current.count(old) != 1:
            raise ValueError(f"The reviewed edit is not unique in {key}: {old!r}")
        current = current.replace(old, new)
        actions.append(action("edit_file", {"filepath": path, "old_string": old, "new_string": new}))
    actions.extend([action("run_command", {"command": command}), action("get_status", {}),
                    action("submit_patch", {})])
    target = REPLAY / (key + "-reviewed-recipe.json")
    save(target, {"evidence_kind": "human_reviewed_off_policy_demonstration",
                  "reviewed_by": "Codex manual causal source, pinned original tests, and command review 2026-10-25",
                  "review_notes": review + " Original F2P/P2P unchanged. Baseline diagnostic failure is expected task reproduction, not a policy error. Source imports asserted against the workspace before and after the repair.",
                  "actions": actions, "final_response": final_response})
    return target


def record_result(key: str, run: str) -> Path:
    """Publish actual official-run evidence beside its acquisition record."""
    from swegemma.harness.verification import _validate_junit_xml

    preparation = json.loads((REPLAY / (key + "-source-preparation.json")).read_text())
    output = ART / run
    summary = json.loads((output / "run-summary.json").read_text())
    task = json.loads((ROOT / preparation["adapted_task"]).read_text())
    reports = {}
    for label in ("baseline", "repaired"):
        found = list(output.glob(label + "-*_junit_*.xml"))
        if len(found) != 1:
            raise ValueError(f"Expected one real original JUnit report for {label}: {found}")
        path = found[0]
        text = path.read_text()
        xml = ET.fromstring(text)
        suites = [xml] if xml.tag == "testsuite" else xml.findall(".//testsuite")
        counts = {k: sum(int(s.attrib.get(attr, 0)) for s in suites)
                  for k, attr in (("tests", "tests"), ("failures", "failures"),
                                  ("errors", "errors"), ("skips", "skipped"))}
        cases = xml.findall(".//testcase")
        reports[label] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path.read_bytes()),
                          **counts, "passed": counts["tests"] - counts["failures"] - counts["errors"] - counts["skips"],
                          "failed_nodes": [c.get("classname", "") + "::" + c.get("name", "")
                                           for c in cases if c.find("failure") is not None or c.find("error") is not None],
                          "skipped_nodes": [c.get("classname", "") + "::" + c.get("name", "")
                                            for c in cases if c.find("skipped") is not None],
                          "original_official_junit_validation": _validate_junit_xml(text, tuple(task["FAIL_TO_PASS"]), tuple(task["PASS_TO_PASS"]), task["test_patch"])}
    events = [json.loads(line) for line in (output / "tool-events.jsonl").read_text().splitlines()]
    bindings = []
    for event in events:
        if event["tool"] == "run_command":
            response = event["response"]
            stdout = response.get("stdout", response.get("details", {}).get("stdout", ""))
            binding_lines = [line for line in stdout.splitlines() if line.startswith("source_binding ")]
            bindings.append({"tool_event": event["index"], "status": response["status"],
                             "actual_binding_and_hash": binding_lines,
                             "binding_assertion_executed_before_mechanism_observation": bool(binding_lines)})
    summary.update({"reports": reports, "original_task": preparation["original_task"],
                    "adapted_task": preparation["adapted_task"], "source_preparation": str((REPLAY / (key + "-source-preparation.json")).relative_to(ROOT)),
                    "recipe": str((REPLAY / (key + "-reviewed-recipe.json")).relative_to(ROOT)),
                    "replay_output": str(output.relative_to(ROOT)),
                    "source_train_revision": preparation["source_train_revision"],
                    "upstream_base_revision": preparation["upstream_source"]["resolved_revision"],
                    "bug_branch_revision": preparation["buggy_source"]["resolved_revision"],
                    "source_FAIL_TO_PASS_count": preparation["F2P_count"], "source_PASS_TO_PASS_count": preparation["P2P_count"],
                    "original_verifier_requirements_relaxed": False,
                    "required_node_identifiers_verbatim_preserved": preparation.get("required_node_identifiers_verbatim_preserved", True),
                    "adapted_FAIL_TO_PASS_count": len(task["FAIL_TO_PASS"]),
                    "adapted_PASS_TO_PASS_count": len(task["PASS_TO_PASS"]),
                    "required_node_mapping": preparation.get("required_node_mapping"),
                    "mechanism": {"adapter": preparation["primary_adapter"], "submechanism": preparation["submechanism"]},
                    "visible_source_binding_observations": bindings,
                    "patch_sha256": digest((output / "submitted.patch").read_bytes()),
                    "training_eligibility": {"human_off_policy_tool_demonstration": bool(summary["resolved_by_local_official_verify_task"] and reports["repaired"]["original_official_junit_validation"][0]),
                                             "task_repair_verified_in_local_official_subprocess": summary["resolved_by_local_official_verify_task"],
                                             "competition_Docker_equivalence": False, "model_on_policy_RL": False,
                                             "same_state_action_preference_pair": False, "GPU_training_started": False},
                    "timing_note": "Human-scripted CPU latency is not model inference or policy budget performance."})
    target = REPLAY / (key + "-replay-result.json")
    save(target, summary)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id")
    parser.add_argument("key")
    options = parser.parse_args()
    prepare(options.instance_id, options.key)
