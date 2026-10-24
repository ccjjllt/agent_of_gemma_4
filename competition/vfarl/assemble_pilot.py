"""Assemble the reviewed 96-task course, keeping policy and privileged data apart."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "competition/vfarl"
EXCLUDED = {"fastapi", "rich", "requests", "httpx"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def task_row(path):
    value = read(path)
    if "row" in value:
        return value["row"]
    if "rows" in value and len(value["rows"]) == 1:
        return value["rows"][0]["row"]
    raise ValueError("Unsupported acquired task wrapper: " + str(path))


def nodes(value):
    value = json.loads(value) if isinstance(value, str) else value
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("Original verification nodes must remain an explicit list")
    return value


def normalize(entry, group, review_path, licenses):
    if group == "G":
        source = entry["task_source"]
        path, expected = entry["task_blob_path"], entry["task_blob_sha256"]
        dataset, revision = source["source"], source["revision"]
        repo, sub = entry["canonical_repo"], entry["submechanism"]
        evidence, license_evidence = entry["causal_evidence"], entry["license_evidence"]
    else:
        source = entry["source_task"]
        path = source.get("path") or source["raw_path"]
        expected = source.get("sha256") or source["raw_sha256"]
        dataset, revision = source["dataset"], source["revision"]
        repo = entry.get("canonical_repo") or entry["environment"]["upstream_repo"]
        sub = entry.get("primary_mechanism") or entry["submechanism"]
        evidence = entry.get("causal_evidence") or {k: entry[k] for k in (
            "agent_visible_trigger", "base_wrong_behavior", "production_symbols",
            "causal_change", "discriminating_observation")}
        license_evidence = entry.get("license") or licenses[repo.lower()]
    if digest(path) != expected:
        raise ValueError("Acquired source changed: " + str(path))
    task = task_row(path)
    if task["instance_id"] != entry["instance_id"]:
        raise ValueError("Task/review identity mismatch")
    if repo.rsplit("/", 1)[-1].lower() in EXCLUDED:
        raise ValueError("Competition repository entered training course")
    environment = entry["environment"]
    commit = environment.get("execution_commit") or environment.get("buggy_commit") or environment.get("base_commit")
    if not commit or not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Missing complete immutable execution commit: " + entry["instance_id"])
    failed, passed = nodes(task["FAIL_TO_PASS"]), nodes(task["PASS_TO_PASS"])
    if not failed or set(failed) & set(passed):
        raise ValueError("Empty or contradictory original required-node lists")
    kind = "bug_introducing" if dataset == "SWE-bench/SWE-smith-py" else "solution"
    record = {
        "instance_id": entry["instance_id"], "canonical_repo": repo,
        "primary_adapter": entry["primary_adapter"], "submechanism": sub,
        "secondary_mechanisms": entry.get("secondary_mechanisms", []),
        "repair_family_id": entry["repair_family_id"],
        "snapshot_group_id": entry.get("snapshot_group_id") or repo + "::" + commit,
        "use_partition": "train_only", "source_dataset": dataset,
        "source_revision": revision, "physical_split": "train",
        "source_task_path": relative(path), "source_task_sha256": expected,
        "source_review_path": relative(review_path), "source_review_sha256": digest(review_path),
        "environment": environment, "execution_commit": commit,
        "license_evidence": license_evidence, "mechanism_evidence": evidence,
        "curator_limitations_and_notes": entry.get("source_limitations") or entry.get("review_notes") or entry.get("source_consistency"),
        "source_verification_evidence": entry.get("source_verification") or {
            "specification": "Original task FAIL_TO_PASS and PASS_TO_PASS retained verbatim in privileged export",
            "source_task_path": relative(path), "source_task_sha256": expected,
        },
        "original_patch_semantics": kind,
        "original_FAIL_TO_PASS_count": len(failed), "original_PASS_TO_PASS_count": len(passed),
        "required_node_format": "pytest" if all("::" in n for n in failed+passed) else "mixed_or_unittest_adapter_required",
        "original_verifier_requirements_relaxed": False,
        "source_mechanism_review_complete": True,
        "target_tool_replayed": False, "task_resolved_verified_in_local_official_backend": None,
        "human_off_policy_sft_demonstration_available": False,
        "competition_Docker_environment_equivalent": False,
        "on_policy_rollout_available": False, "verified_critical_pair_available": False,
    }
    return record, task


def main():
    out = BASE / "pilot96"
    blobs = BASE / "artifacts/pilot96"
    out.mkdir(exist_ok=True)
    blobs.mkdir(parents=True, exist_ok=True)
    g = BASE / "acquisition/general/general-36-review.json"
    sf = BASE / "acquisition/s2-f2/reviewed-sf40.json"
    r = BASE / "acquisition/r2-r5/mechanism-reviews.json"
    licenses = {v["upstream_repo"].lower(): v for v in read(BASE / "acquisition/r2-r5/license-evidence.json")["rows"]}
    groups = [(read(g)["tasks"], "G", g), (read(sf)["tasks"], "SF", sf),
              (read(r)["task_reviews"], "R", r)]
    records, original_tasks = [], []
    for entries, group, review in groups:
        for entry in entries:
            record, task = normalize(entry, group, review, licenses)
            records.append(record)
            original_tasks.append({"instance_id": record["instance_id"], "source_dataset": record["source_dataset"],
                                   "source_revision": record["source_revision"], "physical_split": "train",
                                   "patch_semantics": record["original_patch_semantics"], "task": task})
    ids = [r["instance_id"] for r in records]
    if len(ids) != 96 or len(set(ids)) != 96:
        raise ValueError("The reviewed task identities must be exactly 96 unique records")
    # Family declarations are retained with their original evidence; human cross-review
    # of production symbols, not equality of these strings alone, supplies independence.
    if len({r["repair_family_id"] for r in records}) != 96:
        raise ValueError("A declared repair family appears twice")
    lookup = {r["instance_id"]: r for r in records}
    for result_path in sorted((BASE / "replay").glob("*-replay-result.json")):
        result = read(result_path)
        record = lookup.get(result["instance_id"])
        if record is None:
            continue  # Audit/reserve tasks must not silently re-enter the 96.
        record["target_tool_replayed"] = True
        record["replay_result_path"] = relative(result_path)
        record["replay_result_sha256"] = digest(result_path)
        record["task_resolved_verified_in_local_official_backend"] = result["resolved_by_local_official_verify_task"]
        record["human_off_policy_sft_demonstration_available"] = bool(result["resolved_by_local_official_verify_task"])
    owners = Counter(r["primary_adapter"] for r in records)
    submechanisms = Counter(r["submechanism"] for r in records if r["primary_adapter"] != "G")
    if owners != {"G": 36, "S": 20, "F": 20, "R": 20}:
        raise ValueError("Adapter course allocation drifted")
    if any(submechanisms.get(a+str(i)) != 4 for a in "SFR" for i in range(1, 6)):
        raise ValueError("Each specialist submechanism needs four reviewed tasks")
    privileged = blobs / "tasks.privileged.jsonl"
    privileged.write_text("".join(json.dumps(t, ensure_ascii=False)+"\n" for t in original_tasks))
    # This is a staging export of source-visible descriptions. It intentionally does
    # not append hindsight causal labels, gold edits, verifier lists or evaluator logs.
    policy = blobs / "policy-tasks.jsonl"
    policy.write_text("".join(json.dumps({
        "instance_id": r["instance_id"], "problem_statement": t["task"]["problem_statement"],
        "workspace_source_commit": r["execution_commit"], "tools_contract": "swegemma-0.2.11 official six tools",
        "prompt_status": "Original publisher description; known spec/trigger limitations remain in collector-side review. Human clarification must be explicitly versioned before model sampling.",
    }, ensure_ascii=False)+"\n" for r, t in zip(records, original_tasks)))
    manifest = {
        "date": "2026-10-24", "tasks": 96, "unique_instance_ids": 96,
        "declared_repair_families": 96, "independence_basis": "Different production-symbol repair contracts and cross-review; not trajectory-row counts.",
        "primary_adapter_counts": dict(owners), "specialist_submechanism_counts": dict(submechanisms),
        "repository_count": len({r["canonical_repo"].lower() for r in records}),
        "repository_counts": dict(Counter(r["canonical_repo"].lower() for r in records)),
        "source_dataset_counts": dict(Counter(r["source_dataset"] for r in records)),
        "source_tasks_with_complete_execution_commit": 96,
        "task_environment_instantiation_complete_count": sum(r["target_tool_replayed"] for r in records),
        "locally_official_verified_human_demonstrations": sum(r["human_off_policy_sft_demonstration_available"] for r in records),
        "competition_Docker_equivalent_count": 0, "on_policy_rollouts": 0, "verified_critical_pairs": 0,
        "GPU_training_started": False, "privileged_tasks_path": relative(privileged),
        "privileged_tasks_sha256": digest(privileged), "policy_descriptions_path": relative(policy),
        "policy_descriptions_sha256": digest(policy),
        "limitations": ["Static mechanism/source completion is not 96 executable successes or 96 training-admitted trajectories.",
                        "Image tags and source pins do not prove dependencies and image digests are fully frozen.",
                        "Known source prompt inaccuracies remain explicit; policy clarification is separately versioned.",
                        "Unittest-style nodes need an explicit source adapter before official verification.",
                        "Local official subprocess evidence does not establish competition Docker equivalence."],
    }
    (out / "catalog.json").write_text(json.dumps({"manifest": manifest, "tasks": records}, ensure_ascii=False, indent=2)+"\n")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    lines = ["# 首批96题：机制与来源目录", "", "静态机制/来源已完成；实际目标重放与完整验证另外计数。所有任务仅用于训练，未摄入比赛四仓/129公共题或heldout。", "",
             "| Adapter | 机制 | 任务 | 仓库 | 当前本地官方验证 |", "| --- | --- | --- | --- | --- |"]
    for r in records:
        if r["human_off_policy_sft_demonstration_available"]:
            status = "已通过（人工off-policy）"
        elif r["target_tool_replayed"]:
            status = "已重放，整题未通过"
        else:
            status = "未重放"
        lines.append("| {primary_adapter} | {submechanism} | `{instance_id}` | {canonical_repo} | ".format(**r)+status+" |")
    lines += ["", "原始任务、完整F2P/P2P与补丁在本地privileged JSONL保存；只给policy导出原始题面。所有因果标签/参考修复/验证输出留在collector侧。", "",
              "已知环境、路径保护或required节点匹配不适配的审计题按reserve决定保留，不以删要求换成功。"]
    (out / "catalog.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
