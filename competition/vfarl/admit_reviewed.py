"""Join an explicitly read source-review batch to actual admitted replay data.

This ingests a named, hash-pinned review after its mechanism review. It performs
no source execution, generates no review verdict and never relaxes the verifier.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VFARL = Path(__file__).resolve().parent
REPLAY = VFARL / "replay"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--second-read", required=True)
    parser.add_argument("--course", choices=("S600_common", "S600_specialist", "by_primary"), required=True)
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args()
    review_path = args.review.resolve()
    if not review_path.is_relative_to(VFARL / "acquisition") or sha(review_path) != args.review_sha256:
        raise ValueError("Named review differs from its explicitly read source pin")
    catalog_path = VFARL / "acquisition/supplemental/catalog.json"
    supplemental = read(catalog_path)
    known = {entry["instance_id"] for entry in supplemental["tasks"]}
    known.update(entry["instance_id"] for entry in read(VFARL / "pilot96/catalog.json")["tasks"])
    spec = importlib.util.spec_from_file_location("admission_source_reader", REPLAY / "normalize_results.py")
    normalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer)
    approvals = {entry["instance_id"]: entry for entry in read(REPLAY / "approved-source-requirement-mappings.json")["approvals"]}
    reports = {}
    for path in REPLAY.glob("*-replay-result.json"):
        raw = read(path)
        reports.setdefault(raw["instance_id"], []).append(path)
    added, not_admitted = [], []
    for task in read(review_path)["tasks"]:
        iid = task["instance_id"]
        if iid in known or iid in args.exclude:
            continue
        original = task.get("source_task_path", task.get("original_task"))
        original_sha = task.get("source_task_sha256", task.get("original_task_sha256"))
        if not original or sha(ROOT / original) != original_sha:
            raise ValueError("Original task no longer matches the reviewed source: " + iid)
        license_evidence = task.get("license_evidence") or {
            "source": task["source_license"], "dataset": task["dataset_license"]}
        for proof in license_evidence.values():
            if isinstance(proof, dict) and proof.get("path") and sha(ROOT / proof["path"]) != proof["sha256"]:
                raise ValueError("License source changed: " + iid)
        entry = {key: task[key] for key in ("instance_id", "canonical_repo", "primary_adapter", "submechanism",
            "repair_family_id", "snapshot_group_id", "execution_commit", "source_dataset", "source_revision",
            "physical_split", "original_FAIL_TO_PASS_count", "original_PASS_TO_PASS_count", "mechanism_evidence",
            "original_patch_semantics")}
        default_course = ("S600_specialist" if entry["primary_adapter"] == "S" else "S600_common") if args.course == "by_primary" else args.course
        course = task.get("course_membership", default_course)
        if course not in {"S600_common", "S600_specialist"}:
            raise ValueError("An explicit disjoint S600 course membership is required")
        entry.update({"use_partition": "train_only", "course_membership": course,
            "source_task_path": original, "source_task_sha256": original_sha,
            "source_review_path": str(review_path.relative_to(ROOT)), "source_review_sha256": args.review_sha256,
            "environment": task.get("environment", {"buggy_source": task.get("buggy_source"),
                "upstream_source": task.get("upstream_source"), "source_build_profile": task.get("source_build_profile")}),
            "license_evidence": license_evidence, "source_mechanism_review_complete": True,
            "root_second_read": args.second_read})
        if task.get("visible_prompt_profile"):
            entry["visible_prompt_profile"] = task["visible_prompt_profile"]
        results = [(path, normalizer.normalize(path, {iid: entry}, approvals)) for path in reports.get(iid, [])]
        admitted = [(path, result) for path, result in results if result["eligible_for_human_off_policy_SFT"]]
        if not admitted:
            not_admitted.append({"instance_id": iid,
                "actual_reasons": [result["ineligibility_reasons"] for _, result in results]})
            continue
        entry["root_admission_read_result"] = str(admitted[0][0].relative_to(ROOT))
        supplemental["tasks"].append(entry)
        known.add(iid)
        added.append(entry)
    catalog_path.write_text(json.dumps(supplemental, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"added": len(added), "courses": dict(Counter(e["course_membership"] for e in added)),
        "mechanisms": dict(Counter(e["submechanism"] for e in added)), "not_admitted": not_admitted}, ensure_ascii=False))


if __name__ == "__main__":
    main()
