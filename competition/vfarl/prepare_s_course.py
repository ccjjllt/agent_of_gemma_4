"""Export the admitted, disjoint common and specialist parts of the S course.

This prepares data only. Source candidates and repeated decisions cannot fill
the 600-task objective. No training or model invocation is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VFARL = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(path):
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def export(manifest_path, output, summary_path):
    sft = read(manifest_path)
    actions_path = ROOT / sft["recommended_training_input"]
    if sha(actions_path) != sft["action_output_sha256"]:
        raise ValueError("Admitted action file differs from its source manifest")
    actions = [json.loads(line) for line in actions_path.read_text().splitlines() if line.strip()]
    primary_path = VFARL / "pilot96/catalog.json"
    supplemental_path = VFARL / "acquisition/supplemental/catalog.json"
    entries = read(primary_path)["tasks"] + read(supplemental_path)["tasks"]
    catalog = {entry["instance_id"]: entry for entry in entries}
    episodes = {}
    for action in actions:
        episodes.setdefault(action["instance_id"], action)
    if len(episodes) != sft["unique_task_episodes"]:
        raise ValueError("Master action identities differ from the admitted episode manifest")
    excluded_repositories = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}
    if any(row["canonical_repo"].lower() in excluded_repositories for row in episodes.values()):
        raise ValueError("A competition repository cannot enter this external training course")
    prompts_path = ROOT / sft["official_initial_prompts"]
    if sha(prompts_path) != sft["official_initial_prompts_sha256"]:
        raise ValueError("Question prompts differ from the actual official prompt export")
    prompt_rows = read(prompts_path)["tasks"]
    prompts = {row["instance_id"]: row for row in prompt_rows}
    if len(prompts) != len(prompt_rows) or set(prompts) != set(episodes):
        raise ValueError("Question identities must match the admitted episodes exactly")
    membership = {}
    for iid, row in episodes.items():
        entry = catalog[iid]
        declared = entry.get("course_membership")
        # Pre-S600 source records use legacy course labels. Initial seed is
        # deliberately partitioned as65 non-S common plus16 S specialist.
        if declared == "S600_common":
            role = "common"
        elif declared == "S600_specialist":
            role = "specialist"
        else:
            role = "specialist" if row["primary_adapter"] == "S" else "common"
        if role == "specialist" and row["submechanism"] not in {"S1", "S2", "S3", "S4", "S5"}:
            raise ValueError("A specialist slot needs reviewed representation/algorithm evidence")
        membership[iid] = role
    output.mkdir(parents=True, exist_ok=True)
    courses = {}
    for role, target in (("common", 400), ("specialist", 200)):
        chosen = [row for row in actions if membership[row["instance_id"]] == role]
        tasks = {row["instance_id"]: row for row in chosen}
        if len(tasks) > target:
            raise ValueError("Freeze a reviewed selection before exceeding the " + role + " task target")
        path = output / ("S_" + role + ("" if len(tasks) == target else "_partial") + ".jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in chosen))
        questions = []
        for iid in sorted(tasks):
            source, prompt, action = catalog[iid], prompts[iid], tasks[iid]
            if (hashlib.sha256(prompt["prompt"].encode()).hexdigest() != prompt["prompt_sha256"] or
                    prompt["prompt_sha256"] != action["official_initial_user_prompt_sha256"]):
                raise ValueError("Question text differs from the actual action prefix: " + iid)
            questions.append({"instance_id": iid, "course": "S600_" + role,
                **{key: action[key] for key in ("canonical_repo", "primary_adapter", "submechanism",
                    "repair_family_id", "snapshot_group_id", "source_dataset", "source_revision", "physical_split")},
                "official_initial_user_prompt": prompt["prompt"],
                "official_initial_user_prompt_sha256": prompt["prompt_sha256"],
                "task_input": prompt["task_input"], "task_input_sha256": prompt["task_input_sha256"],
                "source_task_path": source.get("source_task_path"),
                "source_task_sha256": source.get("source_task_sha256"),
                "source_review_path": source.get("source_review_path"),
                "source_review_sha256": source.get("source_review_sha256"),
                "visible_prompt_profile": action["visible_prompt_profile"],
                "source_requirement_profile": action["source_requirement_profile"],
                "baseline_evidence_profile": action["baseline_evidence_profile"],
                "actual_replay_result": prompt["replay_result"],
                "actual_replay_result_sha256": prompt["replay_result_sha256"],
                "metadata_is_policy_prompt": False,
                "policy_input_field": "official_initial_user_prompt"})
        questions_path = output / ("S_" + role + "_questions.jsonl")
        questions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions))
        courses[role] = {"path": label(path), "sha256": sha(path), "target_unique_tasks": target,
            "questions_path": label(questions_path), "questions_sha256": sha(questions_path),
            "question_records": len(questions),
            "admitted_unique_tasks": len(tasks), "remaining_unique_tasks": target-len(tasks),
            "decision_examples": len(chosen),
            "assistant_loss_tokens": sum(row["assistant_loss_tokens"] for row in chosen),
            "submechanism_task_counts": dict(Counter(row["submechanism"] for row in tasks.values())),
            "repair_family_count": len({row["repair_family_id"] for row in tasks.values()}),
            "snapshot_group_count": len({row["snapshot_group_id"] for row in tasks.values()}),
            "instance_ids": sorted(tasks)}
    common, specialist = (set(courses[role]["instance_ids"]) for role in ("common", "specialist"))
    if common & specialist or common | specialist != set(episodes):
        raise ValueError("S course parts must be disjoint and cover the admitted selected identities")
    result = {"schema_version": "S600-disjoint-course-v1", "date": "2026-10-25",
        "objective_unique_tasks": 600, "admitted_unique_tasks": len(episodes),
        "remaining_unique_tasks": 600-len(episodes),
        "all_600_prepared": len(common) == 400 and len(specialist) == 200,
        "source_sft_manifest": label(manifest_path), "source_sft_manifest_sha256": sha(manifest_path),
        "source_action_file": label(actions_path), "source_action_file_sha256": sha(actions_path),
        "source_official_prompts": label(prompts_path), "source_official_prompts_sha256": sha(prompts_path),
        "source_dataset_task_counts": dict(Counter(row["source_dataset"] for row in episodes.values())),
        "physical_split_task_counts": dict(Counter(row["physical_split"] for row in episodes.values())),
        "source_repository_task_counts": dict(Counter(row["canonical_repo"].lower() for row in episodes.values())),
        "excluded_competition_repositories": sorted(excluded_repositories),
        "source_catalog": label(primary_path), "source_catalog_sha256": sha(primary_path),
        "supplemental_catalog": label(supplemental_path), "supplemental_catalog_sha256": sha(supplemental_path),
        "courses": courses, "common_specialist_overlap": 0,
        "counting_policy": "Distinct admitted task identities; repeated actions and shared-course reuse do not add tasks. Family/snapshot groupings retained, not random evaluation splits.",
        "initial_seed_policy": "All65 non-S admitted seeds are common;16 S seeds are specialist. New explicit S600_common/S600_specialist catalog membership overrides owner labels without relabeling the actual mechanism.",
        "specialist_search_targets": {s: 40 for s in ("S1", "S2", "S3", "S4", "S5")},
        "search_quota_is_admission_override": False,
        "evidence_kind": "Codex-authored actual official-tool off-policy coldstart demonstrations",
        "competition_Docker_environment_equivalent": False,
        "model_rollouts": 0, "verified_focal_action_pairs": 0,
        "training_mixture_weights": None, "training_started": False,
        "latest_user_scope": "Prepare questions/data only; training deferred."}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=VFARL / "replay/gemma-sft-manifest.json")
    parser.add_argument("--out", type=Path, default=VFARL / "replay/artifacts/s600")
    parser.add_argument("--summary", type=Path, default=VFARL / "s600/course-manifest.json")
    args = parser.parse_args()
    result = export(args.manifest.resolve(), args.out.resolve(), args.summary.resolve())
    print(json.dumps({"admitted_unique_tasks": result["admitted_unique_tasks"],
        "remaining_unique_tasks": result["remaining_unique_tasks"],
        "common": result["courses"]["common"]["admitted_unique_tasks"],
        "specialist": result["courses"]["specialist"]["admitted_unique_tasks"]}))
