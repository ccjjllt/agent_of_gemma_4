"""Replace the six CPU-confirmed PTY-incompatible S600 tasks before sampling.

Keeps the previous inventory intact, all other 94 identities unchanged and each
replaced task's course/submechanism slot. New tasks still need actual Linux
baseline/reviewed-repair validation; a compatible neighbour is not their result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

try:
    from .prepare import ROOT, SOURCE, load_candidates, sha_file, source_path, verified_file
except ImportError:
    from prepare import ROOT, SOURCE, load_candidates, sha_file, source_path, verified_file


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "kaggle/prevalidate-evidence-v1/vfarl-prevalidate/prevalidation.jsonl"
UNAVAILABLE_REPOSITORY = "theskumar/python-dotenv"


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def relative(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def report_for(task):
    return json.loads(verified_file(task["actual_replay_result"], task["actual_replay_result_sha256"]).read_text())


def replacement_metadata(candidate, position):
    question = candidate["question"]
    report = report_for(question)
    if (report["instance_id"] != question["instance_id"] or not report["resolved_by_local_official_verify_task"]
            or not report["eligible_for_human_off_policy_SFT"]):
        raise ValueError("Replacement lacks its original complete admitted coldstart evidence")
    for key in ("task_input", "source_task", "source_review"):
        path_key = key if key == "task_input" else key + "_path"
        verified_file(question[path_key], question[key + "_sha256"])
    recipe = source_path(report["recipe"])
    record = {key: value for key, value in question.items()
              if key not in ("official_initial_user_prompt", "metadata_is_policy_prompt", "policy_input_field")}
    record.update({"question_row": position + 1, "role": candidate["role"],
                   "recipe": relative(recipe), "recipe_sha256": sha_file(recipe),
                   "old_thinking_zero_max_context_tokens": candidate["old_max_context_tokens"],
                   "old_thinking_zero_headroom_to_4k": 4096 - candidate["old_max_context_tokens"],
                   "old_decision_examples": candidate["old_decision_examples"],
                   "old_assistant_loss_tokens": candidate["old_assistant_loss_tokens"],
                   "metadata_is_policy_prompt": False})
    return record, {key: question[key] for key in (
        "instance_id", "official_initial_user_prompt", "official_initial_user_prompt_sha256")}, report


def replace(previous_path, evidence_path, output):
    if output.resolve() == previous_path.parent.resolve():
        raise ValueError("A replacement version must not overwrite the previous inventory")
    previous = json.loads(previous_path.read_text())
    question_path = previous_path.parent / previous["questions"]["path"]
    if sha_file(question_path) != previous["questions"]["sha256"]:
        raise ValueError("Previous questions differ from their frozen manifest")
    old_questions = {row["instance_id"]: row for row in read_rows(question_path)}
    old_tasks = {row["instance_id"]: row for row in previous["tasks"]}
    if len(old_tasks) != 100 or set(old_questions) != set(old_tasks):
        raise ValueError("Expected the previous complete 100-task inventory")
    evidence_rows = read_rows(evidence_path)
    evidence = {row["instance_id"]: row for row in evidence_rows}
    if len(evidence) != len(evidence_rows) or set(evidence) != set(old_tasks):
        raise ValueError("Expected exactly one real CPU result for each previous task")
    removed = {iid: task for iid, task in old_tasks.items()
               if task["canonical_repo"].lower() == UNAVAILABLE_REPOSITORY}
    if len(removed) != 6:
        raise ValueError("This version addresses exactly the six observed dotenv PTY failures")
    for iid, task in removed.items():
        result = evidence[iid]
        if result["task_input_sha256"] != task["task_input_sha256"] or result.get("usable"):
            raise ValueError("CPU evidence does not identify the unavailable frozen task")
        if result.get("execution_isolation") != "chroot" or any(
                "OSError: out of pty devices" not in result[phase].get("test_output", "")
                for phase in ("baseline", "repaired")):
            raise ValueError("Both original CPU executions must contain the actual PTY incompatibility")
    manifest, candidates, _ = load_candidates()
    if sha_file(SOURCE) != previous["source_course_manifest_sha256"]:
        raise ValueError("The frozen S600 whitelist changed since v1 selection")
    usable_repositories, usable_profiles = set(), {}
    for iid, result in evidence.items():
        if result.get("usable"):
            task = old_tasks[iid]
            if result["task_input_sha256"] != task["task_input_sha256"]:
                raise ValueError("A usable CPU result does not match its source identity")
            report = report_for(task)
            usable_repositories.add(task["canonical_repo"].lower())
            profile = report.get("task_site_packages_profile")
            if profile:
                usable_profiles.setdefault(profile, []).append(iid)
    used_ids = set(old_tasks)
    used_families = {task["repair_family_id"] for iid, task in old_tasks.items() if iid not in removed}
    reports, replacements = {}, []
    new_tasks, new_questions = [], []
    for position, old in enumerate(previous["tasks"]):
        if old["instance_id"] not in removed:
            new_tasks.append(copy.deepcopy(old))
            new_questions.append(copy.deepcopy(old_questions[old["instance_id"]]))
            continue
        eligible = [item for item in candidates if item["role"] == old["role"]
                    and item["question"]["submechanism"] == old["submechanism"]
                    and item["old_max_context_tokens"] <= 4096
                    and item["question"]["instance_id"] not in used_ids
                    and item["question"]["repair_family_id"] not in used_families
                    and item["question"]["canonical_repo"].lower() != UNAVAILABLE_REPOSITORY]
        if not eligible:
            raise ValueError("No admitted, unique-family 4K replacement for " + old["instance_id"])

        def priority(item):
            question = item["question"]
            iid = question["instance_id"]
            if iid not in reports:
                reports[iid] = report_for(question)
            profile = reports[iid].get("task_site_packages_profile")
            seed_tie = hashlib.sha256(f"{previous['seed']}:{iid}".encode()).hexdigest()
            return (profile not in usable_profiles,
                    question["canonical_repo"].lower() not in usable_repositories,
                    item["old_max_context_tokens"], seed_tie)

        chosen = min(eligible, key=priority)
        task, question, report = replacement_metadata(chosen, position)
        used_ids.add(task["instance_id"])
        used_families.add(task["repair_family_id"])
        new_tasks.append(task)
        new_questions.append(question)
        profile = report.get("task_site_packages_profile")
        replacements.append({
            "removed_instance_id": old["instance_id"], "replacement_instance_id": task["instance_id"],
            "role": old["role"], "submechanism": old["submechanism"],
            "reason": "Original and reviewed-repair Linux chroot runs both fail required upstream paths with OSError: out of pty devices; no PTY backend is available. Source tests and their expectations remain unchanged.",
            "removed_task_input_sha256": old["task_input_sha256"],
            "cpu_evidence_path": relative(evidence_path), "cpu_evidence_sha256": sha_file(evidence_path),
            "replacement_source": {key: task[key] for key in (
                "canonical_repo", "physical_split", "source_task_path", "source_task_sha256",
                "task_input", "task_input_sha256", "source_review_path", "source_review_sha256",
                "actual_replay_result", "actual_replay_result_sha256", "recipe", "recipe_sha256",
                "official_initial_user_prompt_sha256", "repair_family_id", "snapshot_group_id",
                "old_thinking_zero_max_context_tokens")},
            "selection_evidence": {"eligible_same_slot_candidates": len(eligible),
                "repository_has_usable_cpu_peer": task["canonical_repo"].lower() in usable_repositories,
                "source_dependency_profile": profile,
                "profile_has_usable_cpu_peer": profile in usable_profiles,
                "usable_cpu_peer_instance_ids": sorted(usable_profiles.get(profile, [])),
                "method": "Prefer an already usable CPU dependency profile, then an already usable repository, then shorter complete thinking-zero demonstrations; fixed-seed hash breaks ties."},
            "replacement_linux_prevalidation_required": True,
            "replacement_linux_success_claimed": False,
        })
    counts = lambda key: dict(sorted(Counter(task[key] for task in new_tasks).items()))
    if (len({task["instance_id"] for task in new_tasks}) != 100
            or len({task["repair_family_id"] for task in new_tasks}) != 100
            or counts("role") != {"common": 67, "specialist": 33}
            or counts("submechanism") != previous["counts"]["submechanisms"]
            or any(task["physical_split"] != "train" or task["old_thinking_zero_max_context_tokens"] > 4096 for task in new_tasks)):
        raise ValueError("Replacement changed a frozen task, mechanism, family or length constraint")
    result = copy.deepcopy(previous)
    result.update({"schema_version": "vfarl-l4x4-s600-train100-v2",
                   "parent_selection": relative(previous_path), "parent_selection_sha256": sha_file(previous_path),
                   "tasks": new_tasks, "substitutions": "substitutions.json"})
    result["selection_policy"]["environment_replacement"] = "Six proven PTY-incompatible dotenv tasks replaced before any policy sampling; 94 original records retained, same role and submechanism slots. New tasks require Linux validation."
    result["counts"].update({"roles": counts("role"), "submechanisms": counts("submechanism"),
        "repositories": counts("canonical_repo"), "unique_repositories": len(counts("canonical_repo")),
        "unique_snapshot_groups": len(counts("snapshot_group_id")),
        "old_complete_episodes_at_most_3072": sum(task["old_thinking_zero_max_context_tokens"] <= 3072 for task in new_tasks),
        "old_max_context_tokens": max(task["old_thinking_zero_max_context_tokens"] for task in new_tasks)})
    substitutions = {"schema_version": "vfarl-environment-substitutions-v1",
        "selection_stage": "CPU environment preparation before any model rollout in this experiment",
        "parent_selection": relative(previous_path), "parent_selection_sha256": sha_file(previous_path),
        "source_course_manifest": relative(SOURCE), "source_course_manifest_sha256": sha_file(SOURCE),
        "cpu_evidence_path": relative(evidence_path), "cpu_evidence_sha256": sha_file(evidence_path),
        "model_scores_used_for_selection": False, "retained_original_tasks": 94,
        "preserved_constraints": {"unique_tasks": 100, "common": 67, "specialist": 33,
            "same_submechanism_slot_for_each_replacement": True, "unique_repair_families": 100,
            "historical_complete_context_cap": 4096},
        "replacements": replacements}
    output.mkdir(parents=True, exist_ok=False)
    output_questions = output / "questions.jsonl"
    output_questions.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in new_questions))
    result["questions"]["sha256"] = sha_file(output_questions)
    (output / "substitutions.json").write_text(json.dumps(substitutions, ensure_ascii=False, indent=2) + "\n")
    result["substitutions_sha256"] = sha_file(output / "substitutions.json")
    (output / "selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return substitutions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=HERE / "selection.json")
    parser.add_argument("--prevalidation", type=Path, default=EVIDENCE)
    parser.add_argument("--out", type=Path, default=HERE / "v2")
    args = parser.parse_args()
    result = replace(args.previous.resolve(), args.prevalidation.resolve(), args.out.resolve())
    print(json.dumps([{key: row[key] for key in ("removed_instance_id", "replacement_instance_id", "role", "submechanism")}
                      for row in result["replacements"]], ensure_ascii=False, indent=2))
