"""Freeze a small S600 train-only inventory for the L4 VFARL experiment.

This exports questions and host-side source links, not model rollouts or RL
examples. The frozen thinking-zero demonstrations are read only for their
lengths; their messages, answers and control prefixes are never rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace


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


def _revision_parent(previous_path):
    """Read and bind the fixed parent once, retaining original question/task bytes."""
    previous_path = Path(previous_path).resolve()
    selection_bytes = previous_path.read_bytes()
    previous = json.loads(selection_bytes)
    entries = previous["tasks"]
    ids = [row["instance_id"] for row in entries]
    if len(ids) != 100 or len(set(ids)) != 100 or any(row.get("physical_split") != "train" for row in entries):
        raise ValueError("A problem revision requires the fixed 100 distinct train identities")
    questions_path = previous_path.parent / previous["questions"]["path"]
    question_bytes = questions_path.read_bytes()
    if hashlib.sha256(question_bytes).hexdigest() != previous["questions"]["sha256"]:
        raise ValueError("The parent questions changed")
    question_lines = question_bytes.splitlines(keepends=True)
    question_rows = [json.loads(line) for line in question_lines]
    if [row["instance_id"] for row in question_rows] != ids:
        raise ValueError("Parent questions and fixed inventory order differ")
    substitutions_source = None
    substitutions_relative = None
    if previous.get("substitutions"):
        substitutions_relative = Path(previous["substitutions"])
        if substitutions_relative.is_absolute() or ".." in substitutions_relative.parts:
            raise ValueError("Substitutions must have a local selection-relative path")
        substitutions_source = (previous_path.parent / substitutions_relative).resolve()
        verified_file(str(substitutions_source.relative_to(ROOT)), previous.get("substitutions_sha256"))
    inherited_revisions = []
    for reference in [*previous.get("problem_revisions", []), previous.get("problem_revision"),
                      *(row.get("problem_revision") for row in entries)]:
        if reference is None or any(item["sha256"] == reference["sha256"] for item in inherited_revisions):
            continue
        verified_file(reference["path"], reference["sha256"])
        inherited_revisions.append(copy.deepcopy(reference))

    # Locate raw top-level task objects with JSON's decoder, not text matching.
    # Reuse their original bytes when writing unaffected task records.
    text = selection_bytes.decode("utf-8")
    decoder = json.JSONDecoder()
    def space(position):
        while position < len(text) and text[position].isspace():
            position += 1
        return position
    position = space(0) + 1
    while True:
        key, key_end = decoder.raw_decode(text, space(position))
        value_start = space(space(key_end) + 1)
        _, value_end = decoder.raw_decode(text, value_start)
        if key == "tasks":
            array_start, array_end = value_start, value_end
            break
        position = space(value_end) + 1
    task_spans = []
    position = space(array_start + 1)
    for _ in entries:
        _, end = decoder.raw_decode(text, position)
        task_spans.append((position - array_start, end - array_start))
        position = space(end)
        if text[position] == ",":
            position = space(position + 1)
    return dict(path=previous_path, sha256=hashlib.sha256(selection_bytes).hexdigest(),
        previous=previous, entries=entries, ids=ids, question_lines=question_lines,
        question_rows=question_rows, substitutions_source=substitutions_source,
        substitutions_relative=substitutions_relative, inherited_revisions=inherited_revisions,
        task_array_text=text[array_start:array_end], task_spans=task_spans)


def _problem_change(parent, index, statement_path, expected_statement_sha=None):
    from swegemma.harness.agent_runner import build_agent_prompt

    entry = parent["entries"][index]
    old_task = json.loads(verified_file(entry["task_input"], entry["task_input_sha256"]).read_text())
    statement_path = Path(statement_path).resolve()
    statement_bytes = statement_path.read_bytes()
    statement_sha = hashlib.sha256(statement_bytes).hexdigest()
    if expected_statement_sha is not None and statement_sha != expected_statement_sha:
        raise ValueError("Planned statement changed: " + str(statement_path))
    statement = statement_bytes.decode("utf-8")
    if not statement.strip() or statement == old_task["problem_statement"]:
        raise ValueError("A revision must contain a new, nonempty behavioral statement")
    report = json.loads(verified_file(entry["actual_replay_result"], entry["actual_replay_result_sha256"]).read_text())
    config = SimpleNamespace(budget=SimpleNamespace(**report["evaluation_budget"]),
        harness=SimpleNamespace(**report["harness_limits"]), graph_dir="", embeddings_dir="",
        enable_sandbox_testing=True)
    tools = report.get("initial_prompt_declared_tools")
    tools_source = "Historical report initial_prompt_declared_tools"
    if tools is None:
        imports = report.get("tool_imports", {})
        if (report.get("six_tools_only") is not True
                or set(imports) != {"run_command", "read_file", "write_file", "edit_file", "get_status", "submit_patch"}):
            raise ValueError("Historical prompt tool declarations are unavailable")
        tools = sorted(imports)
        tools_source = "Historical six_tools_only report's actual tool_imports; missing declaration field is not rewritten"
    old_prompt = build_agent_prompt(SimpleNamespace(**old_task), config,
        declared_tools=tools, enable_sandbox_testing=True)
    old_sha = hashlib.sha256(old_prompt.encode()).hexdigest()
    if (old_prompt != parent["question_rows"][index]["official_initial_user_prompt"]
            or old_sha != entry["official_initial_user_prompt_sha256"]
            or old_sha != parent["question_rows"][index]["official_initial_user_prompt_sha256"]):
        raise ValueError("The official prompt builder does not reproduce the frozen parent prompt")
    new_task = {**old_task, "problem_statement": statement}
    new_prompt = build_agent_prompt(SimpleNamespace(**new_task), config,
        declared_tools=tools, enable_sandbox_testing=True)
    return dict(index=index, entry=entry, statement=statement, statement_sha256=statement_sha,
        statement_path=statement_path, new_task=new_task, new_prompt=new_prompt, tools_source=tools_source,
        new_prompt_sha256=hashlib.sha256(new_prompt.encode()).hexdigest())


def _write_problem_change(parent, change, destination):
    destination.mkdir(parents=True, exist_ok=True)
    entry, index = change["entry"], change["index"]
    (destination / "problem-statement.txt").write_bytes(change["statement"].encode("utf-8"))
    task_path = destination / "task-v2.json"
    new_task = change["new_task"]
    task_path.write_text(json.dumps(new_task, ensure_ascii=False, indent=2) + "\n")
    revision_path = destination / "revision.json"
    revision = {
        "format": "vfarl-fixed100-problem-revision-v1", "instance_id": entry["instance_id"],
        "parent_selection": str(parent["path"].relative_to(ROOT)),
        "parent_selection_sha256": parent["sha256"],
        "old_task_input": entry["task_input"], "old_task_input_sha256": entry["task_input_sha256"],
        "new_task_input": str(task_path.resolve().relative_to(ROOT)), "new_task_input_sha256": sha_file(task_path),
        "old_initial_prompt_sha256": entry["official_initial_user_prompt_sha256"],
        "new_initial_prompt_sha256": change["new_prompt_sha256"],
        "problem_statement_sha256": change["statement_sha256"],
        "prompt_builder": "Actual swegemma.harness.agent_runner.build_agent_prompt; parent prompt reproduced byte-for-byte",
        "prompt_tools_source": change["tools_source"],
        "preserved": "Same 100 IDs, order, role, mechanism, family, source snapshot, reference repair, original F2P/P2P and full target files; this record changes only this task's problem statement",
        "historical_demonstration": {"report": entry["actual_replay_result"],
            "sha256": entry["actual_replay_result_sha256"], "applies_to": "Original problem version only"},
        "new_demonstration": "Not replayed; old action traces and supervision are not relabeled as new-prompt demonstrations",
        "new_cpu_evidence": "Requires actual baseline/reference execution on the unchanged full oracle; old usable is not copied to the revised task hash",
    }
    revision_path.write_text(json.dumps(revision, ensure_ascii=False, indent=2) + "\n")
    reference = {"path": str(revision_path.resolve().relative_to(ROOT)), "sha256": sha_file(revision_path)}
    question_line = (json.dumps({**parent["question_rows"][index],
        "official_initial_user_prompt": change["new_prompt"],
        "official_initial_user_prompt_sha256": change["new_prompt_sha256"]}, ensure_ascii=False) + "\n").encode("utf-8")
    return revision, reference, question_line


def _finish_problem_revisions(parent, changes, output, *, batch=False):
    output.mkdir(parents=True)
    if parent["substitutions_source"] is not None:
        destination = output / parent["substitutions_relative"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent["substitutions_source"], destination)
    result = copy.deepcopy(parent["previous"])
    result["schema_version"] = "vfarl-l4x4-s600-train100-question-revision-v1"
    result["parent_selection"] = str(parent["path"].relative_to(ROOT))
    result["parent_selection_sha256"] = parent["sha256"]
    result["problem_revisions"] = copy.deepcopy(parent["inherited_revisions"])
    result.pop("problem_revision", None)
    result.pop("problem_revision_batch", None)
    question_lines = list(parent["question_lines"])
    revisions = []
    for change in changes:
        index, entry = change["index"], change["entry"]
        destination = output / "tasks" / entry["instance_id"] if batch else output
        if not destination.resolve().is_relative_to(output.resolve()):
            raise ValueError("Task identifier leaves the revision output directory")
        revision, reference, question_lines[index] = _write_problem_change(parent, change, destination)
        revisions.append(revision)
        result["problem_revisions"].append(copy.deepcopy(reference))
        revised = result["tasks"][index]
        revised.update(task_input=revision["new_task_input"], task_input_sha256=revision["new_task_input_sha256"],
            official_initial_user_prompt_sha256=change["new_prompt_sha256"],
            visible_prompt_profile="curator_authored_behavioral_clarification",
            actual_replay_result_applies_to="Historical problem version only; new-prompt demonstration not replayed",
            problem_revision=reference)
        if not batch:
            result["problem_revision"] = copy.deepcopy(reference)
    question_path = output / "questions.jsonl"
    question_path.write_bytes(b"".join(question_lines))
    result["questions"] = {**result["questions"], "path": question_path.name, "sha256": sha_file(question_path)}
    if batch:
        result["problem_revision_batch"] = {"count": len(changes), "array_indices": [c["index"] for c in changes]}
    raw_tasks = parent["task_array_text"]
    for change in sorted(changes, key=lambda c: c["index"], reverse=True):
        start, end = parent["task_spans"][change["index"]]
        replacement = json.dumps(result["tasks"][change["index"]], ensure_ascii=False, indent=2).replace("\n", "\n    ")
        raw_tasks = raw_tasks[:start] + replacement + raw_tasks[end:]
    marker = "__vfarl_preserve_original_task_object_bytes__"
    rendered = json.dumps({**result, "tasks": marker}, ensure_ascii=False, indent=2)
    if rendered.count(json.dumps(marker)) != 1:
        raise ValueError("Task-array serialization marker is ambiguous")
    (output / "selection.json").write_bytes((rendered.replace(json.dumps(marker), raw_tasks, 1) + "\n").encode("utf-8"))
    return revisions


def revise_problem(previous_path, instance_id, statement_path, output):
    """Version one question; preserve the existing single-task API and layout."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve the previous revision directory: " + str(output))
    parent = _revision_parent(previous_path)
    if instance_id not in parent["ids"]:
        raise ValueError("Question is outside the frozen training inventory")
    change = _problem_change(parent, parent["ids"].index(instance_id), statement_path)
    return _finish_problem_revisions(parent, [change], output)[0]


def revise_problems(previous_path, plan_path, output):
    """Apply explicit index/ID/hash-bound statements, writing the 100-question parent once."""
    output, plan_path = Path(output).resolve(), Path(plan_path).resolve()
    if output.exists():
        raise FileExistsError("Preserve the previous revision directory: " + str(output))
    parent = _revision_parent(previous_path)
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    entries = plan.get("entries")
    if not isinstance(entries, list) or not entries or len(entries) > 100:
        raise ValueError("A revision plan needs 1-100 explicit fixed-inventory entries")
    for key in ("parent_selection_sha256", "previous_selection_sha256"):
        if key in plan and plan[key] != parent["sha256"]:
            raise ValueError("Revision plan belongs to a different parent selection")
    for key in ("parent_selection", "previous_selection"):
        if key in plan and (ROOT / plan[key]).resolve() != parent["path"]:
            raise ValueError("Revision plan references a different parent selection path")
    changes, seen = [], set()
    for item in entries:
        index = item["index"]
        if type(index) is not int or index < 0 or index >= 100 or index in seen:
            raise ValueError("Revision plan contains an invalid or repeated array index")
        entry = parent["entries"][index]
        if (item["instance_id"] != entry["instance_id"]
                or item["old_task_input_sha256"] != entry["task_input_sha256"]):
            raise ValueError("Revision plan index/ID/old task identity differs from the fixed parent")
        statement_path = Path(item["statement_path"])
        if not isinstance(item["statement_sha256"], str):
            raise ValueError("Revision plan must explicitly bind a statement SHA256")
        if not statement_path.is_absolute():
            statement_path = ROOT / statement_path
        changes.append(_problem_change(parent, index, statement_path, item["statement_sha256"]))
        seen.add(index)
    revisions = _finish_problem_revisions(parent, changes, output, batch=True)
    summary = dict(format="vfarl-fixed100-problem-revision-batch-v1",
        parent_selection=str(parent["path"].relative_to(ROOT)), parent_selection_sha256=parent["sha256"],
        plan_path=str(plan_path), plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        changed_questions=len(revisions), fixed_inventory=100,
        selection_sha256=sha_file(output / "selection.json"), questions_sha256=sha_file(output / "questions.jsonl"),
        revisions=revisions, new_demonstrations="Not replayed; no historical actions relabeled",
        new_cpu_evidence="Each new task version requires applicable actual CPU evidence; unchanged oracle is retained")
    (output / "batch-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revise-selection", type=Path)
    parser.add_argument("--revise-task")
    parser.add_argument("--problem-statement", type=Path)
    parser.add_argument("--revision-plan", type=Path,
                        help="Batch entries: index, instance_id, old_task_input_sha256, statement_path (repo-relative or absolute), statement_sha256")
    args = parser.parse_args()
    if args.revision_plan:
        if not args.revise_selection or args.revise_task or args.problem_statement:
            parser.error("Batch revision requires --revise-selection and --revision-plan; omit single-task arguments")
        result = revise_problems(args.revise_selection.resolve(), args.revision_plan.resolve(), args.out.resolve())
    elif any((args.revise_selection, args.revise_task, args.problem_statement)):
        if not all((args.revise_selection, args.revise_task, args.problem_statement)):
            parser.error("Problem revision requires --revise-selection, --revise-task and --problem-statement")
        result = revise_problem(args.revise_selection.resolve(), args.revise_task,
            args.problem_statement.resolve(), args.out.resolve())
    else:
        result = prepare(args.out.resolve(), args.seed)["counts"]
    print(json.dumps(result, ensure_ascii=False, indent=2))
