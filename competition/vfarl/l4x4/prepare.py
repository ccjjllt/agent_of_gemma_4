"""Freeze a small S600 train-only inventory for the L4 VFARL experiment.

This exports questions and host-side source links, not model rollouts or RL
examples. The frozen thinking-zero demonstrations are read only for their
lengths; their messages, answers and control prefixes are never rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "competition/vfarl/s600/course-manifest.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent
SPECIALIST_QUOTAS = {"S1": 7, "S2": 7, "S3": 7, "S4": 6, "S5": 6}


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("Source link is not an existing repository file: " + str(value))
    return path


def verified_file(value, expected):
    path = source_path(value)
    if not expected or sha_file(path) != expected:
        raise ValueError("Frozen source hash mismatch: " + str(value))
    return path


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_candidates():
    manifest = json.loads(SOURCE.read_text())
    if manifest["admitted_unique_tasks"] != 600 or not manifest["all_600_prepared"]:
        raise ValueError("Expected the completed frozen S600 course")
    excluded = {repo.lower() for repo in manifest["excluded_competition_repositories"]}
    candidates, input_files, identities = [], [], set()
    for role, count in (("common", 400), ("specialist", 200)):
        course = manifest["courses"][role]
        questions_path = verified_file(course["questions_path"], course["questions_sha256"])
        actions_path = verified_file(course["path"], course["sha256"])
        input_files.extend([
            {"role": role + "_questions", "path": course["questions_path"], "sha256": course["questions_sha256"]},
            {"role": role + "_thinking_zero_actions", "path": course["path"], "sha256": course["sha256"]},
        ])
        questions = {row["instance_id"]: row for row in rows(questions_path)}
        allowed = set(course["instance_ids"])
        if len(allowed) != count or set(questions) != allowed or allowed & identities:
            raise ValueError("Question identities do not match the disjoint S600 whitelist")
        identities.update(allowed)
        lengths, decisions, supervision = {}, Counter(), Counter()
        for action in rows(actions_path):
            iid = action["instance_id"]
            if iid not in allowed or action["physical_split"] != "train":
                raise ValueError("Action outside this course's train whitelist")
            if action["generation_configuration_for_this_export"]["thinking_budget"] != 0:
                raise ValueError("Unexpected historical thinking policy")
            size = len(action["input_ids"])
            if size != action["context_tokens"]:
                raise ValueError("Historical context length differs from actual token IDs")
            lengths[iid] = max(lengths.get(iid, 0), size)
            decisions[iid] += 1
            supervision[iid] += action["assistant_loss_tokens"]
        if set(lengths) != allowed:
            raise ValueError("Incomplete historical demonstration inventory")
        for iid, question in questions.items():
            if question["physical_split"] != "train" or question["canonical_repo"].lower() in excluded:
                raise ValueError("Question is not an allowed external training task")
            if question["course"] != "S600_" + role:
                raise ValueError("Question course label differs from its frozen membership")
            prompt_hash = hashlib.sha256(question["official_initial_user_prompt"].encode()).hexdigest()
            if prompt_hash != question["official_initial_user_prompt_sha256"]:
                raise ValueError("Question prompt hash mismatch")
            candidates.append({"question": question, "role": role,
                               "old_max_context_tokens": lengths[iid],
                               "old_decision_examples": decisions[iid],
                               "old_assistant_loss_tokens": supervision[iid]})
    return manifest, candidates, input_files


def choose(candidates, seed):
    eligible = [item for item in candidates if item["old_max_context_tokens"] <= 4096]
    selected, used_ids, used_families = [], set(), set()
    repo_counts, mechanism_counts = Counter(), Counter()

    def take(pool):
        options = [item for item in pool
                   if item["question"]["instance_id"] not in used_ids
                   and item["question"]["repair_family_id"] not in used_families]
        if not options:
            raise ValueError("Insufficient unique repair families for the declared selection")

        def order(item):
            question = item["question"]
            tie = hashlib.sha256(f"{seed}:{question['instance_id']}".encode()).hexdigest()
            return (item["old_max_context_tokens"] > 3072,
                    mechanism_counts[question["submechanism"]],
                    repo_counts[question["canonical_repo"].lower()],
                    item["old_max_context_tokens"], tie)

        item = min(options, key=order)
        question = item["question"]
        selected.append(item)
        used_ids.add(question["instance_id"])
        used_families.add(question["repair_family_id"])
        repo_counts[question["canonical_repo"].lower()] += 1
        mechanism_counts[question["submechanism"]] += 1

    # Reserve the smaller specialist pool first; round-robin makes the balance
    # explicit instead of allowing repository popularity to decide it.
    for position in range(max(SPECIALIST_QUOTAS.values())):
        for mechanism, count in SPECIALIST_QUOTAS.items():
            if position < count:
                take([item for item in eligible if item["role"] == "specialist"
                      and item["question"]["submechanism"] == mechanism])
    for _ in range(67):
        take([item for item in eligible if item["role"] == "common"])
    return sorted(selected, key=lambda item: (item["role"], item["question"]["instance_id"]))


def prepare(output, seed):
    manifest, candidates, input_files = load_candidates()
    chosen = choose(candidates, seed)
    questions, metadata = [], []
    for index, item in enumerate(chosen):
        question = item["question"]
        report_path = verified_file(question["actual_replay_result"], question["actual_replay_result_sha256"])
        report = json.loads(report_path.read_text())
        if (report["instance_id"] != question["instance_id"]
                or not report["resolved_by_local_official_verify_task"]
                or not report["eligible_for_human_off_policy_SFT"]):
            raise ValueError("Selected task no longer matches its admitted replay evidence")
        for key in ("task_input", "source_task", "source_review"):
            path_key = key if key == "task_input" else key + "_path"
            verified_file(question[path_key], question[key + "_sha256"])
        recipe = source_path(report["recipe"])
        questions.append({key: question[key] for key in (
            "instance_id", "official_initial_user_prompt", "official_initial_user_prompt_sha256")})
        record = {key: value for key, value in question.items()
                  if key not in ("official_initial_user_prompt", "metadata_is_policy_prompt", "policy_input_field")}
        record.update({"question_row": index + 1, "role": item["role"],
                       "recipe": str(recipe.relative_to(ROOT)), "recipe_sha256": sha_file(recipe),
                       "old_thinking_zero_max_context_tokens": item["old_max_context_tokens"],
                       "old_thinking_zero_headroom_to_4k": 4096 - item["old_max_context_tokens"],
                       "old_decision_examples": item["old_decision_examples"],
                       "old_assistant_loss_tokens": item["old_assistant_loss_tokens"],
                       "metadata_is_policy_prompt": False})
        metadata.append(record)
    output.mkdir(parents=True, exist_ok=True)
    question_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions)
    question_path = output / "questions.jsonl"
    question_path.write_text(question_text)
    counts = lambda key: dict(sorted(Counter(row[key] for row in metadata).items()))
    result = {
        "schema_version": "vfarl-l4x4-s600-train100-v1",
        "purpose": "Training-only VFARL loop validation inventory; not an evaluation split or collected RL dataset.",
        "seed": seed,
        "source_course_manifest": str(SOURCE.relative_to(ROOT)),
        "source_course_manifest_sha256": sha_file(SOURCE),
        "source_inputs": input_files,
        "selection_policy": {
            "common_tasks": 67, "specialist_tasks": 33, "specialist_quotas": SPECIALIST_QUOTAS,
            "require_old_complete_episode_max_tokens_at_most": 4096,
            "preference": "First prefer old complete episodes <=3072 tokens, then underrepresented submechanisms and repositories, then shorter episodes; seeded SHA256 breaks exact ties.",
            "unique_repair_families": True,
            "specialist_pool_reserved_first": True,
            "family_and_snapshot_metadata_preserved": True,
            "evaluation_split_created": False,
        },
        "counts": {"unique_tasks": len(metadata), "roles": counts("role"),
                   "submechanisms": counts("submechanism"), "repositories": counts("canonical_repo"),
                   "unique_repositories": len(counts("canonical_repo")),
                   "unique_repair_families": len(counts("repair_family_id")),
                   "unique_snapshot_groups": len(counts("snapshot_group_id")),
                   "old_complete_episodes_at_most_3072": sum(row["old_thinking_zero_max_context_tokens"] <= 3072 for row in metadata),
                   "old_complete_episodes_at_most_4096": len(metadata),
                   "old_max_context_tokens": max(row["old_thinking_zero_max_context_tokens"] for row in metadata)},
        "historical_length_census": {str(limit): {
            role: sum(item["role"] == role and item["old_max_context_tokens"] <= limit for item in candidates)
            for role in ("common", "specialist")}
            for limit in (3072, 4096, 8192)},
        "length_boundary": "All lengths above are exact historical thinking=0 demonstration lengths. They are only a selection heuristic: authentic thinking=1024 sampling changes histories, and repeated turns can exceed 4K/8K. Tokenize every actual complete prefix, reserve output inside the context cap, never silently truncate histories, and report context exhaustion separately from verified task failure.",
        "historical_demonstrations_rewritten": False,
        "new_policy_samples_created": False,
        "physical_split": "train",
        "only_training_whitelist_read": True,
        "questions": {"path": question_path.name, "sha256": sha_file(question_path),
                      "records": len(questions), "policy_input_field": "official_initial_user_prompt",
                      "allowed_fields": ["instance_id", "official_initial_user_prompt", "official_initial_user_prompt_sha256"]},
        "metadata_boundary": "Only questions.jsonl official_initial_user_prompt enters the policy. This selection file contains privileged host-side task/verifier/recipe source links; do not serialize it or referenced gold/verifier assets into model context.",
        "excluded_competition_repositories": manifest["excluded_competition_repositories"],
        "tasks": metadata,
    }
    if len({row["instance_id"] for row in metadata}) != 100 or counts("role") != {"common": 67, "specialist": 33}:
        raise ValueError("Selection does not contain exactly the requested 100 distinct tasks")
    (output / "selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = prepare(args.out.resolve(), args.seed)
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))
