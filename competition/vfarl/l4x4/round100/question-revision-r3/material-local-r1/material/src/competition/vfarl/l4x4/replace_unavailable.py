"""Replace CPU-proven incompatible S600 environments without changing labels.

The dotenv case reproduces v2 unchanged. The boltons case keeps the other 93
v2 identities and replaces seven uncollected network-incompatible tasks. One
explicit common-G coverage gap has no same-label S600 alternative; its chosen
task retains its actual G-input_precondition label. The yamllint case keeps 99
v3 records and replaces the single PTY-incompatible S2 task using the original
CPU evidence. New tasks still need their own full Linux validation.
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
CASES = {
    "dotenv-pty": {
        "repository": "theskumar/python-dotenv", "count": 6, "version": "v2",
        "reason": "Original and reviewed-repair Linux chroot runs both fail required upstream paths with OSError: out of pty devices; no PTY backend is available. Source tests and their expectations remain unchanged.",
        "policy": "Six proven PTY-incompatible dotenv tasks replaced before any policy sampling; 94 original records retained, same role and submechanism slots. New tasks require Linux validation.",
    },
    "boltons-network": {
        "repository": "mahmoud/boltons", "count": 7, "version": "v3",
        "reason": "Original and reviewed-repair Linux chroot runs both fail required tests/test_socketutils.py paths with PermissionError: [Errno 1] Operation not permitted. Network creation is unavailable in this task environment; these are environment failures, not model outcomes. Original tests and expectations remain unchanged.",
        "policy": "Seven CPU-proven network-incompatible boltons tasks replaced using no model scores; other 93 v2 records retained. Six slots retain their exact role/submechanism. The unique common G-optional_state_contract slot has no alternative in all S600 and is replaced by a real G-input_precondition task without relabeling. Specialist S1-S5 quotas remain unchanged. New tasks require Linux validation.",
    },
    "yamllint-pty": {
        "repository": "adrienverge/yamllint", "count": 1, "version": "v4",
        "reason": "Original and reviewed-repair R2 Linux chroot executions both fail the required upstream tests/test_cli.py PTY path with OSError: out of pty devices. This is an environment capability failure, not a model result. Original tests and expectations remain unchanged.",
        "policy": "One CPU-proven PTY-incompatible yamllint S2 task replaced without model-score input; all other 99 v3 task records retained. The replacement retains its real specialist/S2 source label and every S1-S5 quota remains unchanged. Prior common-G coverage gap is unchanged. New task requires Linux validation.",
    },
}
KNOWN_UNAVAILABLE_REPOSITORIES = {"theskumar/python-dotenv", "mahmoud/boltons"}
OPTIONAL_STATE_REPLACEMENT = "kayak__pypika.1c9646f0.func_basic__w1s47eoh"


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


def replace(previous_path, evidence_path, output, case="dotenv-pty"):
    specification = CASES[case]
    unavailable_repository = specification["repository"]
    if output.resolve() == previous_path.parent.resolve():
        raise ValueError("A replacement version must not overwrite the previous inventory")
    previous = json.loads(previous_path.read_text())
    if case == "boltons-network" and previous["schema_version"] != "vfarl-l4x4-s600-train100-v2":
        raise ValueError("The boltons replacement must start from the frozen v2 inventory")
    if case == "yamllint-pty" and previous["schema_version"] != "vfarl-l4x4-s600-train100-v3":
        raise ValueError("The yamllint replacement must start from the frozen v3 inventory")
    question_path = previous_path.parent / previous["questions"]["path"]
    if sha_file(question_path) != previous["questions"]["sha256"]:
        raise ValueError("Previous questions differ from their frozen manifest")
    old_questions = {row["instance_id"]: row for row in read_rows(question_path)}
    old_tasks = {row["instance_id"]: row for row in previous["tasks"]}
    if len(old_tasks) != 100 or set(old_questions) != set(old_tasks):
        raise ValueError("Expected the previous complete 100-task inventory")
    evidence_rows = read_rows(evidence_path)
    evidence = {row["instance_id"]: row for row in evidence_rows}
    evidence_selection_path = previous_path
    evidence_tasks = old_tasks
    if case == "yamllint-pty":
        # R2 evaluated v2. The yaml task is unchanged in v3, while v3's seven
        # new tasks have no R2 outcomes; do not fabricate results for them.
        evidence_selection_path = verified_file(previous["parent_selection"], previous["parent_selection_sha256"])
        evidence_selection = json.loads(evidence_selection_path.read_text())
        if evidence_selection["schema_version"] != "vfarl-l4x4-s600-train100-v2":
            raise ValueError("Expected v2 as the source of the actual R2 CPU evidence")
        evidence_tasks = {task["instance_id"]: task for task in evidence_selection["tasks"]}
    if len(evidence) != len(evidence_rows) or set(evidence) != set(evidence_tasks):
        raise ValueError("Expected exactly one real CPU result for every task in the declared evidence inventory")
    removed = {iid: task for iid, task in old_tasks.items()
               if task["canonical_repo"].lower() == unavailable_repository}
    if len(removed) != specification["count"]:
        raise ValueError("Unexpected number of tasks for the explicit CPU environment replacement")
    for iid, task in removed.items():
        result = evidence[iid]
        if result["task_input_sha256"] != task["task_input_sha256"] or result.get("usable"):
            raise ValueError("CPU evidence does not identify the unavailable frozen task")
        required = (["tests/test_socketutils.py", "PermissionError: [Errno 1] Operation not permitted"]
                    if case == "boltons-network" else ["OSError: out of pty devices"])
        if case == "yamllint-pty":
            required.append("tests/test_cli.py")
        if result.get("execution_isolation") != "chroot" or any(
                marker not in result[phase].get("test_output", "")
                for phase in ("baseline", "repaired") for marker in required):
            raise ValueError("Both original CPU executions must show the declared environment incompatibility")
    manifest, candidates, _ = load_candidates()
    if sha_file(SOURCE) != previous["source_course_manifest_sha256"]:
        raise ValueError("The frozen S600 whitelist changed since v1 selection")
    usable_repositories, usable_profiles = set(), {}
    for iid, result in evidence.items():
        if result.get("usable") and iid in old_tasks:
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
        coverage_exception = case == "boltons-network" and old["submechanism"] == "G-optional_state_contract"
        if coverage_exception:
            exact_label_inventory = [item for item in candidates
                                     if item["question"]["submechanism"] == old["submechanism"]]
            if len(exact_label_inventory) != 1 or exact_label_inventory[0]["question"]["instance_id"] != old["instance_id"]:
                raise ValueError("The declared unique-label coverage exception no longer matches S600")
        eligible = [item for item in candidates if item["role"] == old["role"]
                    and (item["question"]["primary_adapter"] == "G" and item["role"] == "common"
                         if coverage_exception else item["question"]["submechanism"] == old["submechanism"])
                    and item["old_max_context_tokens"] <= 4096
                    and item["question"]["instance_id"] not in used_ids
                    and item["question"]["repair_family_id"] not in used_families
                    and item["question"]["canonical_repo"].lower() not in
                    (KNOWN_UNAVAILABLE_REPOSITORIES | {"adrienverge/yamllint"} if case == "yamllint-pty"
                     else KNOWN_UNAVAILABLE_REPOSITORIES if case == "boltons-network" else {unavailable_repository})]
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
                    coverage_exception and iid != OPTIONAL_STATE_REPLACEMENT,
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
            "reason": specification["reason"],
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
        if coverage_exception:
            replacements[-1]["selection_evidence"]["eligible_common_G_candidates"] = replacements[-1]["selection_evidence"].pop("eligible_same_slot_candidates")
            replacements[-1]["selection_evidence"]["eligible_same_submechanism_candidates"] = 0
            replacements[-1]["selection_evidence"]["method"] = "Prefer an already usable CPU profile/repository, then the inspected Table.for_portion attribute-state guard contract, then shorter complete demonstrations and fixed-seed hash; retain the chosen task's original G label."
            replacements[-1].update({
                "replacement_submechanism": task["submechanism"],
                "same_submechanism_preserved": False,
                "coverage_exception": {
                    "original_label_count_in_all_S600": 1,
                    "same_label_replacement_candidates_before_length_filter": 0,
                    "new_label_is_original_source_label": True,
                    "reason": "No alternate G-optional_state_contract task exists in the frozen S600 course. Select another genuine common-G task and preserve its own label; do not manufacture mechanism coverage.",
                    "semantic_preference": "Table.for_portion validates whether an attribute state is already occupied before assigning it, a nearby optional-state/attribute-transition contract; its source classifies it as G-input_precondition and that label is retained.",
                    "source_question_and_family_used_only": True,
                },
            })
        if case in {"boltons-network", "yamllint-pty"}:
            replacements[-1]["replacement_source"].update({
                "primary_adapter": task["primary_adapter"], "submechanism": task["submechanism"],
                "role": task["role"]})
    counts = lambda key: dict(sorted(Counter(task[key] for task in new_tasks).items()))
    label_changes = [{"instance_id": old["instance_id"], "from": old["submechanism"],
                      "to": new["submechanism"], "replacement_instance_id": new["instance_id"]}
                     for old, new in zip(previous["tasks"], new_tasks)
                     if old["submechanism"] != new["submechanism"]]
    expected_counts = Counter(previous["counts"]["submechanisms"])
    if case == "boltons-network":
        if (len(label_changes) != 1 or label_changes[0]["from"] != "G-optional_state_contract"
                or label_changes[0]["to"] != "G-input_precondition"):
            raise ValueError("Only the declared original-label common-G coverage gap may change")
        expected_counts.subtract({"G-optional_state_contract": 1})
        expected_counts.update({"G-input_precondition": 1})
        expected_counts = +expected_counts
    elif label_changes:
        raise ValueError("This replacement preserves every original mechanism slot")
    if (len({task["instance_id"] for task in new_tasks}) != 100
            or len({task["repair_family_id"] for task in new_tasks}) != 100
            or counts("role") != {"common": 67, "specialist": 33}
            or counts("submechanism") != dict(expected_counts)
            or any(task["physical_split"] != "train" or task["old_thinking_zero_max_context_tokens"] > 4096 for task in new_tasks)):
        raise ValueError("Replacement changed a frozen task, mechanism, family or length constraint")
    result = copy.deepcopy(previous)
    result.update({"schema_version": "vfarl-l4x4-s600-train100-" + specification["version"],
                   "parent_selection": relative(previous_path), "parent_selection_sha256": sha_file(previous_path),
                   "tasks": new_tasks, "substitutions": "substitutions.json"})
    result["selection_policy"]["environment_replacement"] = specification["policy"]
    result["counts"].update({"roles": counts("role"), "submechanisms": counts("submechanism"),
        "repositories": counts("canonical_repo"), "unique_repositories": len(counts("canonical_repo")),
        "unique_snapshot_groups": len(counts("snapshot_group_id")),
        "old_complete_episodes_at_most_3072": sum(task["old_thinking_zero_max_context_tokens"] <= 3072 for task in new_tasks),
        "old_max_context_tokens": max(task["old_thinking_zero_max_context_tokens"] for task in new_tasks)})
    substitutions = {"schema_version": "vfarl-environment-substitutions-v1",
        "selection_stage": ("CPU environment preparation before any model rollout in this experiment"
                            if case == "dotenv-pty" else "CPU environment-only replacement of never-sampled unavailable tasks; no model outcomes used"),
        "parent_selection": relative(previous_path), "parent_selection_sha256": sha_file(previous_path),
        "source_course_manifest": relative(SOURCE), "source_course_manifest_sha256": sha_file(SOURCE),
        "cpu_evidence_path": relative(evidence_path), "cpu_evidence_sha256": sha_file(evidence_path),
        "model_scores_used_for_selection": False, "retained_original_tasks": 100 - len(removed),
        "preserved_constraints": {"unique_tasks": 100, "common": 67, "specialist": 33,
            "same_submechanism_slot_for_each_replacement": not label_changes, "unique_repair_families": 100,
            "historical_complete_context_cap": 4096},
        "replacements": replacements}
    if case == "boltons-network":
        substitutions.update({"case": case, "common_submechanism_changes": label_changes,
                              "coverage_gap": "G-optional_state_contract is no longer represented because its sole S600 task needs unavailable socket capabilities.",
                              "specialist_submechanism_quotas_unchanged": True,
                              "new_tasks_not_yet_model_sampled": True})
        result["selection_policy"]["common_submechanism_changes"] = label_changes
        result["selection_policy"]["coverage_gap"] = substitutions["coverage_gap"]
    if case == "yamllint-pty":
        substitutions.update({"case": case,
                              "cpu_evidence_selection": relative(evidence_selection_path),
                              "cpu_evidence_selection_sha256": sha_file(evidence_selection_path),
                              "cpu_evidence_boundary": "R2 covered v2; the replaced yamllint task is identical in v3. No R2 success is inferred for the seven retained v3 replacements.",
                              "common_submechanism_distribution_unchanged_from_parent": True,
                              "specialist_submechanism_quotas_unchanged": True,
                              "replacement_tasks_not_yet_model_sampled": True})
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
    parser.add_argument("--case", choices=tuple(CASES), default="dotenv-pty")
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--prevalidation", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.case != "dotenv-pty" and args.prevalidation is None:
        parser.error("This replacement requires the complete actual v2 CPU --prevalidation report")
    parent = {"dotenv-pty": "selection.json", "boltons-network": "v2/selection.json", "yamllint-pty": "v3/selection.json"}
    previous = args.previous or HERE / parent[args.case]
    evidence = args.prevalidation or EVIDENCE
    output = args.out or HERE / CASES[args.case]["version"]
    result = replace(previous.resolve(), evidence.resolve(), output.resolve(), args.case)
    print(json.dumps([{**{key: row[key] for key in ("removed_instance_id", "replacement_instance_id", "role", "submechanism")},
                       "replacement_submechanism": row.get("replacement_submechanism", row["submechanism"])}
                      for row in result["replacements"]], ensure_ascii=False, indent=2))
