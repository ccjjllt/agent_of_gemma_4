"""Produce explicitly reviewed common-course source records and real tool runs.

The input manifest contains individually selected contracts and reproduction
commands. This workflow does not invent classifications, tests or policy data.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REPLAY = ROOT / "competition/vfarl/replay"
ART = REPLAY / "artifacts/s600-common-root"
REV = "77cab9055d42ab4a5c25c89a8f937096db13558e"
REPORT_PREFIX = "s600-root-"
REVIEW_SCOPE = "Individually selected common repair contracts for S600, not relabeled S specialist data. Correct mechanism labels retained."
DEFAULT_COURSE = "S600_common"
sys.path.insert(0, str(REPLAY))
from prepare_sfr import acquire
from normalize_results import junit_report


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(path):
    return str(path.relative_to(ROOT))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def preparation_path(config):
    return HERE / (config["key"] + "-preparation" + config.get("task_version_suffix", "") + ".json")


def source_row(iid):
    index = {r["instance_id"]: r for r in (json.loads(line) for line in
        (ROOT / "competition/vfarl/acquisition/general/smith-task-index.jsonl").read_text().splitlines())}
    expected = index[iid]
    if expected["split"] != "train" or expected["revision"] != REV:
        raise ValueError("Task identity is outside the fixed allowed train source")
    for path in (HERE / "artifacts/pages").glob("*.json"):
        if path.name.endswith(".receipt.json"):
            continue
        receipt_path = path.with_suffix(".receipt.json")
        proof = json.loads(receipt_path.read_text())
        if proof["sha256"] != sha(path) or proof["revision"] != REV or proof["split"] != "train":
            raise ValueError("Cached source differs from fixed train receipt")
        for record in json.loads(path.read_text())["rows"]:
            row = record["row"]
            if row["instance_id"] == iid:
                if record["row_idx"] != expected["row_idx"] or row["repo"] != expected["repo"]:
                    raise ValueError("Original source body identity differs from allowed index")
                return row, {"source_page": label(path), "source_page_sha256": sha(path),
                    "source_page_receipt": label(receipt_path), "row_idx": record["row_idx"]}
    raise ValueError("Explicit selected task not present in its source cache")


def resolve_branch(repo, iid):
    path = HERE / "artifacts/refs" / (repo.replace("/", "__") + ".txt")
    if not path.exists():
        result = subprocess.run(["git", "ls-remote", "https://github.com/" + repo + ".git"],
            capture_output=True, text=True, check=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.stdout)
    refs = dict((fields[1], fields[0]) for fields in
        (line.split() for line in path.read_text().splitlines()) if len(fields) == 2)
    commit = refs["refs/heads/" + iid]
    if len(commit) != 40:
        raise ValueError("Complete immutable execution commit required")
    return commit, {"url": "https://github.com/" + repo + ".git", "ref": "refs/heads/" + iid,
                    "response_path": label(path), "response_sha256": sha(path)}


def function_text(path, qualified):
    text = path.read_text()
    scope = ast.parse(text)
    for component in qualified.split("."):
        nodes = [node for node in scope.body if isinstance(node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == component]
        if len(nodes) != 1:
            raise ValueError("Production symbol does not resolve uniquely: " + qualified)
        scope = nodes[0]
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("Reviewed repair must target an explicit production function")
    return "".join(text.splitlines(keepends=True)[scope.lineno-1:scope.end_lineno]), scope.lineno, scope.end_lineno


def prepare(config):
    key = config["key"]
    compile(config["body"], "<reviewed observation " + key + ">", "exec")
    row, provenance = source_row(config["instance_id"])
    original_description = row.get("problem_statement", "")
    curated = config.get("curated_prompt")
    if curated is not None and (not isinstance(curated, str) or not curated.strip()):
        raise ValueError("A curated description must be explicit nonempty reviewed text")
    if not original_description.strip() and curated is None:
        raise ValueError("An empty publisher description needs an explicit reviewed curated_prompt")
    if curated is not None and original_description.strip() and not config.get("curation_reason"):
        raise ValueError("Replacing a nonempty publisher description needs an explicit curation_reason")
    task_path = HERE / (row["instance_id"] + ".task.json")
    save(task_path, {"source_dataset": "SWE-bench/SWE-smith-py", "revision": REV,
                     "split": "train", **provenance, "row": row})
    commit, ref = resolve_branch(row["repo"], row["instance_id"])
    source = ART / (key + "-source")
    buggy = acquire(row["repo"], commit, source)
    upstream = ART / (config["canonical_repo"].replace("/", "__") + "-upstream")
    up = acquire(config["canonical_repo"], config["snapshot"], upstream)
    files, missing = [], []
    for path in sorted(upstream.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(upstream)
        if (not any(part in {"test", "tests", "testing"} for part in rel.parts)
                and str(rel) not in {"conftest.py", "test.py"}
                and not (rel.name.startswith("test_") and rel.suffix == ".py")
                and not rel.name.endswith("_test.py")):
            continue
        actual, raw = source / rel, path.read_bytes()
        if actual.exists() and actual.read_bytes() != raw:
            raise ValueError("Original verification source differs: " + str(rel))
        files.append({"path": str(rel), "sha256": sha(path), "source_revision": up["resolved_revision"],
                      "present_in_buggy_source": actual.exists()})
        if not actual.exists():
            for patch_line in difflib.unified_diff([], raw.decode().splitlines(keepends=True),
                    fromfile="/dev/null", tofile="b/" + str(rel)):
                # Preserve an upstream file whose final byte is not a newline.
                # Without the standard marker, the next file header joins its
                # final source line and corrupts the restored native asset.
                missing.append(patch_line if patch_line.endswith("\n") else
                    patch_line + "\n\\ No newline at end of file\n")
    targets = sorted({node.split("::", 1)[0] for node in row["FAIL_TO_PASS"] + row["PASS_TO_PASS"]})
    if not targets or not all((source / p).exists() or any(f["path"] == p for f in files) for p in targets):
        raise ValueError("Full original required file scope unavailable")
    task_payload = {"instance_id": row["instance_id"], "repo": row["repo"], "base_commit": commit,
        "problem_statement": row["problem_statement"], "patch": "", "test_patch": "".join(missing),
        "FAIL_TO_PASS": row["FAIL_TO_PASS"], "PASS_TO_PASS": row["PASS_TO_PASS"],
        "verification_target_files": targets}
    adapted = ART / (key + "-task" + config.get("task_version_suffix", "") + ".json")
    visible_profile = None
    if curated is not None:
        original_adapted = ART / (key + "-task-original-description.json")
        save(original_adapted, task_payload)
        adapted = ART / (key + "-task" + config.get("task_version_suffix", "-curated-v2") + ".json")
        task_payload = {**task_payload, "problem_statement": curated}
        visible_profile = {
            "kind": "curator_authored_behavioral_clarification",
            "reviewed_by": "Codex explicit fixed-source public API contract and mutation review",
            "reason": config.get("curation_reason", "publisher description empty"),
            "original_problem_statement_sha256": hashlib.sha256(original_description.encode()).hexdigest(),
            "curated_problem_statement_sha256": hashlib.sha256(curated.encode()).hexdigest(),
            "curated_prompt": curated,
            "source_basis": ["Fixed upstream public API contract", "Reviewed allowed-train production mutation"],
            "original_adapted_task": label(original_adapted),
            "original_adapted_task_sha256": sha(original_adapted),
            "reference_patch_or_required_list_in_prompt": False,
        }
    save(adapted, task_payload)
    rel = config["relative_path"]
    before, line, end = function_text(source / rel, config["symbol"])
    after, _, _ = function_text(upstream / rel, config["symbol"])
    original_text = (source / rel).read_text()
    if before == after or original_text.count(before) != 1:
        raise ValueError("No unique source-supported production edit")
    if original_text.replace(before, after) != (upstream / rel).read_text():
        raise ValueError("Source includes additional changes outside this reviewed function")
    module = rel.removeprefix("src/").removesuffix(".py").replace("/", ".")
    command = ("python3 - <<'PYREPLAY'\nfrom pathlib import Path\nimport importlib,hashlib\n"
        + "m=importlib.import_module(" + repr(module) + ")\nactual=Path(m.__file__).resolve();expected=(Path.cwd()/"
        + repr(rel) + ").resolve()\nassert actual==expected,(actual,expected)\n"
        + "print('source_binding',actual,'sha256',hashlib.sha256(actual.read_bytes()).hexdigest())\n"
        + config["body"] + "\nPYREPLAY")
    def run(status):
        return {"tool": "run_command", "arguments": {"command": command}, "expected_status": status,
                "reviewed_command_sha256": hashlib.sha256(command.encode()).hexdigest()}
    recipe = {"evidence_kind": "human_reviewed_off_policy_demonstration",
        "reviewed_by": "Codex source contract and explicit command review2026-10-25",
        "review_notes": config["contract"] + " The complete production-file delta is exactly this reviewed function. Original full-file verification and all requirements retained. Actual diagnosis is not a rejected policy action.",
        "actions": [{"tool": "get_status", "arguments": {}, "expected_status": "ok"},
            {"tool": "read_file", "arguments": {"filepath": rel, "start_line": max(1, line-4),
                "end_line": min(end+4, line+120)}, "expected_status": "ok"}, run("error"),
            {"tool": "edit_file", "arguments": {"filepath": rel, "old_string": before, "new_string": after}, "expected_status": "ok"},
            run("ok"), {"tool": "get_status", "arguments": {}, "expected_status": "ok"},
            {"tool": "submit_patch", "arguments": {}, "expected_status": "ok"}],
        "final_response": config["contract"] + " The source-bound reproduction passes after the repair. Patch submitted."}
    recipe_path = REPLAY / (REPORT_PREFIX + key + "-reviewed-recipe.json")
    save(recipe_path, recipe)
    license_names = (config["license_file"],) if config.get("license_file") else ("LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE.rst", "LICENSE-MIT")
    license_path = next((upstream / name for name in license_names
                         if (upstream / name).exists()), None)
    if license_path is None or not license_path.resolve().is_relative_to(upstream.resolve()):
        raise ValueError("Original source license unavailable")
    copied = HERE / (config["canonical_repo"].replace("/", "__") + ".LICENSE")
    copied.write_bytes(license_path.read_bytes())
    dataset = ROOT / "competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt"
    prep = {k: v for k, v in config.items() if k != "body"}
    prep.update({"source_dataset": "SWE-bench/SWE-smith-py", "source_revision": REV, "physical_split": "train",
        "use_partition": "train_only", "course_membership": config.get("course_membership", DEFAULT_COURSE),
        "original_task": label(task_path), "original_task_sha256": sha(task_path), "execution_commit": commit,
        "mirror_ref": ref, "buggy_source": buggy, "upstream_source": up, "workspace": label(source),
        "adapted_task": label(adapted), "adapted_task_sha256": sha(adapted), "recipe": label(recipe_path),
        "recipe_sha256": sha(recipe_path), "snapshot_group_id": config["canonical_repo"] + "::" + up["resolved_revision"],
        "repair_family_id": config["canonical_repo"] + "::" + config["symbol"] + "::" + config["family_contract"],
        "verification_target_files": targets, "verification_assets": files,
        "original_FAIL_TO_PASS_count": len(row["FAIL_TO_PASS"]), "original_PASS_TO_PASS_count": len(row["PASS_TO_PASS"]),
        "source_license": {"path": label(copied), "sha256": sha(copied), "revision": up["resolved_revision"], "declared_license": config["source_license"]},
        "dataset_license": {"path": label(dataset), "sha256": sha(dataset), "revision": REV, "declared_card_license": "MIT"},
        "mechanism_evidence": {"primary_mechanism": config["submechanism"], "production_symbol": config["symbol"],
            "causal_contract": config["contract"], "source_diff_reviewed": True,
            "bug_patch_direction": "introduces_defect_not_repair_target"},
        "original_patch_semantics": "bug_introducing", "original_verifier_requirements_relaxed": False,
        "source_mechanism_review_complete": True, **provenance})
    if visible_profile is not None:
        prep["visible_prompt_profile"] = visible_profile
    save(preparation_path(config), prep)


def replay(config, attempt="001"):
    key = config["key"]
    prep = json.loads(preparation_path(config).read_text())
    out = ART / (key + "-run-" + attempt)
    if (out / "run-summary.json").exists():
        return
    if out.exists():
        raise ValueError("An incomplete actual attempt already exists; choose a new attempt identifier")
    with (ART / (key + "-run-" + attempt + ".log")).open("w") as log:
        command = [str(REPLAY / ".venv/bin/python"), str(REPLAY / "run_official.py"),
            "--task", str(ROOT / prep["adapted_task"]), "--workspace", str(ROOT / prep["workspace"]),
            "--recipe", str(ROOT / prep["recipe"]), "--output", str(out), "--verify", "--verify-baseline",
            "--source-layout", config.get("source_layout", "."), "--task-site-packages", str(REPLAY / config["task_profile"]),
            "--task-site-packages-mode", "copy", "--command-timeout", "90", "--verification-timeout", "180"]
        for name, path in config.get("sandbox_executables", {}).items():
            command.extend(["--sandbox-executable", name + "=" + str(ROOT / path)])
        if config.get("task_python"):
            command.extend(["--task-python", str(ROOT / config["task_python"])])
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def publish(configs, attempt="001", review_name="reviewed-candidates"):
    reviews, reports = [], []
    for config in configs:
        key = config["key"]
        prep_path = preparation_path(config)
        out = ART / (key + "-run-" + attempt)
        if not (out / "run-summary.json").exists():
            continue
        prep = json.loads(prep_path.read_text())
        summary_path = out / "run-summary.json"
        result = json.loads(summary_path.read_text())
        original = json.loads((ROOT / prep["original_task"]).read_text())["row"]
        adapted = json.loads((ROOT / prep["adapted_task"]).read_text())
        phases = {}
        for phase in ("baseline", "repaired"):
            matches = list(out.glob(phase + "-_swegemma_junit_*.xml"))
            if len(matches) != 1:
                # Keep the actual verifier failure, with no invented case counts.
                # The normalizer will refuse incomplete baseline/repair evidence.
                phases[phase] = None
                continue
            phases[phase] = junit_report(matches[0], original, adapted)
        events_path = out / "tool-events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        observations = [{"index": e["index"], "actual_response": e["response"]} for e in events if e["tool"] == "run_command"]
        result.update({"original_task": prep["original_task"], "original_task_sha256": prep["original_task_sha256"],
            "adapted_task": prep["adapted_task"], "recipe": prep["recipe"], "replay_output": label(out),
            "reports": phases, "canonical_repo": prep["canonical_repo"], "source_dataset": prep["source_dataset"],
            "source_revision": REV, "physical_split": "train", "primary_adapter": prep["primary_adapter"],
            "submechanism": prep["submechanism"], "course_membership": prep["course_membership"], "execution_commit": prep["execution_commit"],
            "repair_family_id": prep["repair_family_id"], "snapshot_group_id": prep["snapshot_group_id"],
            "mechanism": {"adapter": prep["primary_adapter"], "submechanism": prep["submechanism"], "evidence": prep["mechanism_evidence"]},
            "original_verifier_requirements_relaxed": False, "required_node_identifiers_verbatim_preserved": True,
            "actual_mechanism_observations": observations, "source_preparation": label(prep_path)})
        if prep.get("visible_prompt_profile") is not None:
            result["visible_prompt_profile"] = prep["visible_prompt_profile"]
        reviews.append({**prep, "source_review_complete": True, "actual_current_run": label(out),
            "actual_run_summary_sha256": sha(summary_path), "actual_recipe_sha256": sha(ROOT / prep["recipe"]),
            "actual_events_sha256": sha(events_path), "actual_reports": phases, "actual_observations": observations,
            "resolved_by_local_official_verify_task": result["resolved_by_local_official_verify_task"]})
        reports.append((REPLAY / (REPORT_PREFIX + key + "-attempt" + attempt + "-replay-result.json"), result))
    path = HERE / (review_name + ".json")
    if path.exists() and json.loads(path.read_text()).get("tasks") != reviews:
        raise ValueError("Use a new review-name to preserve a frozen earlier source review")
    save(path, {"date": "2026-10-25", "source_revision": REV, "physical_split": "train",
        "scope": REVIEW_SCOPE,
        "independence_basis": "One reviewed production-function contract per selected mutation, with full source delta confirmation; same upstream snapshot grouped.",
        "tasks": reviews})
    for target, result in reports:
        result.update({"source_review_path": label(path), "source_review_sha256": sha(path)})
        save(target, result)
        print(json.dumps({"instance_id": result["instance_id"], "resolved": result["resolved_by_local_official_verify_task"],
            "baseline_F2P_failed": (result["reports"]["baseline"] or {}).get("all_original_F2P_observed_failed")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=HERE / "selected-contracts.json")
    parser.add_argument("--keys", nargs="+")
    parser.add_argument("--stage", choices=("prepare", "replay", "publish", "all"), default="all")
    parser.add_argument("--attempt", default="001")
    parser.add_argument("--review-name", default="reviewed-candidates")
    parser.add_argument("--keep-going", action="store_true",
                        help="Retain failed attempts and continue independent source tasks; never admit missing evidence")
    args = parser.parse_args()
    if not args.attempt.isdecimal() or not args.review_name.replace("-", "").isalnum():
        raise ValueError("Explicit safe attempt and review identifiers required")
    configs = json.loads(args.selection.read_text())["tasks"]
    if args.keys:
        configs = [c for c in configs if c["key"] in args.keys]
    failures = []
    for config in configs:
        try:
            if args.stage in {"prepare", "all"}:
                prepare(config)
                print("prepared", config["key"], flush=True)
            if args.stage in {"replay", "all"}:
                replay(config, args.attempt)
                print("replayed", config["key"], flush=True)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            if not args.keep_going:
                raise
            failures.append(config["key"])
            print("Retained actual failure", config["key"], repr(error), file=sys.stderr, flush=True)
    if args.stage in {"publish", "all"}:
        publish(configs, args.attempt, args.review_name)
    if failures:
        raise SystemExit(1)
