"""Build the F5 supplemental training catalog for normalize_results.

Reads acquisition/f5-new/reviewed-candidates.json (written by publish) and
emits acquisition/f5-new/supplemental-f5.json in the reviewed-supplemental
schema subset consumed by replay/normalize_results.py. New file only; the
S600 supplemental catalog is never modified.
"""
import hashlib
import json
from pathlib import Path

from swegemma.models.task import _normalize_test_nodes

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(path):
    return str(path.relative_to(ROOT))


def main():
    review_path = HERE / "reviewed-candidates.json"
    review = json.loads(review_path.read_text())
    review_sha = sha(review_path)
    entries = []
    for task in review["tasks"]:
        original = json.loads((ROOT / task["original_task"]).read_text())["row"]
        adapted = json.loads((ROOT / task["adapted_task"]).read_text())
        assert original["problem_statement"] == adapted["problem_statement"]
        assert original["problem_statement"].strip()
        assert task["original_verifier_requirements_relaxed"] is False
        assert (_normalize_test_nodes(adapted.get("FAIL_TO_PASS")) ==
                _normalize_test_nodes(original.get("FAIL_TO_PASS")))
        assert (_normalize_test_nodes(adapted.get("PASS_TO_PASS")) ==
                _normalize_test_nodes(original.get("PASS_TO_PASS")))
        profile = {"kind": "verbatim_publisher_description",
                   "original_problem_statement_sha256": hashlib.sha256(
                       original["problem_statement"].encode()).hexdigest()}
        entries.append({
            "instance_id": task["instance_id"],
            "canonical_repo": task["canonical_repo"],
            "primary_adapter": task["primary_adapter"],
            "submechanism": task["submechanism"],
            "repair_family_id": task["repair_family_id"],
            "snapshot_group_id": task["snapshot_group_id"],
            "use_partition": "train_only",
            "source_dataset": task["source_dataset"],
            "source_revision": task["source_revision"],
            "physical_split": task["physical_split"],
            "source_task_path": task["original_task"],
            "source_task_sha256": task["original_task_sha256"],
            "source_review_path": label(review_path),
            "source_review_sha256": review_sha,
            "execution_commit": task["execution_commit"],
            "visible_prompt_profile": profile,
            "mechanism_evidence": task["mechanism_evidence"],
            "original_FAIL_TO_PASS_count": task["original_FAIL_TO_PASS_count"],
            "original_PASS_TO_PASS_count": task["original_PASS_TO_PASS_count"],
            "original_patch_semantics": task.get("original_patch_semantics", "bug_introducing"),
            "course_membership": task.get("course_membership", "F600_F5"),
        })
    entries.sort(key=lambda e: e["instance_id"])
    ids = [e["instance_id"] for e in entries]
    assert len(set(ids)) == len(ids) and len(ids) >= 1, [len(ids), len(set(ids))]
    out = {"schema_version": "reviewed-supplemental-training-catalog-v1",
           "date": "2026-10-25",
           "scope": "F600 F5 event/stream/lifecycle timing contracts; Smith-py train only; four competition repositories excluded.",
           "selection_policy": "Explicit train-only F5 mechanism sources with production-to-consumption-to-end event-sequence observations; single-point values demoted to S/G and SSE-string escapes rejected during review.",
           "tasks": entries}
    target = HERE / "supplemental-f5.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"supplemental_f5_tasks": len(entries),
                      "catalog": label(target), "sha256": sha(target)}))


if __name__ == "__main__":
    main()
