"""Serialize the explicitly reviewed F course without changing the frozen S course.

Only named, catalog-bound train reports are read. This workflow executes no
tasks, model calls, training or GPU work and creates no mechanism verdicts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
VFARL = Path(__file__).resolve().parent
REPLAY = VFARL / "replay"
COURSE = VFARL / "f600"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    common_path = COURSE / "common-catalog.json"
    specialist_path = COURSE / "specialist-catalog.json"
    entries = read(common_path)["tasks"] + read(specialist_path)["tasks"]
    ids = [row["instance_id"] for row in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Common and specialist selections must be disjoint")
    if any(row.get("physical_split") != "train" or row.get("use_partition") != "train_only"
           for row in entries):
        raise ValueError("Only explicitly selected train source records may be exported")
    reports = []
    for entry in entries:
        report = ROOT / entry["root_admission_read_result"]
        if read(report).get("instance_id") != entry["instance_id"]:
            raise ValueError("Selected report identity differs from its reviewed catalog")
        reports.append(report)
    python = REPLAY / ".venv/bin/python"
    run = lambda args, **kwargs: subprocess.run([str(x) for x in args], cwd=ROOT, check=True, **kwargs)
    canonical_path = COURSE / "canonical-results-manifest.json"
    run([python, REPLAY / "normalize_results.py", "--catalog", common_path,
         "--supplemental-catalog", specialist_path, "--reports", *reports,
         "--out", REPLAY / "artifacts/f600-canonical-results", "--manifest", canonical_path])
    canonical = read(canonical_path)
    if canonical["eligible_count"] != len(ids):
        raise ValueError("Selected F course includes unadmitted replay evidence; inspect the canonical manifest")
    artifacts = REPLAY / "artifacts/gemma-f"
    artifacts.mkdir(parents=True, exist_ok=True)
    prompts = artifacts / "official-task-prompts.json"
    declarations = REPLAY / "adk-six-tool-declarations.json"
    results = canonical["eligible_results"]
    run([python, REPLAY / "export_prompts.py", "--results", *results,
         "--declarations", declarations.relative_to(ROOT), "--out", prompts.relative_to(ROOT)])
    with (artifacts / "export-current.log").open("w") as log:
        run([REPLAY / ".venv-format/bin/python", VFARL / "format_replays.py", "--results", *results,
             "--tokenizer", "research/015/tokenizer-assets/official-31b-ct-v2-tokenizer",
             "--declarations", declarations.relative_to(ROOT), "--official-prompts", prompts.relative_to(ROOT),
             "--out", artifacts.relative_to(ROOT)],
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            stdout=log, stderr=subprocess.STDOUT)
    sft_path = REPLAY / "gemma-f-manifest.json"
    shutil.copyfile(artifacts / "manifest.json", sft_path)
    run([python, VFARL / "prepare_f_course.py", "--manifest", sft_path,
         "--common-catalog", common_path, "--specialist-catalog", specialist_path])
    course_path = COURSE / "course-manifest.json"
    course = read(course_path)
    summary = {"schema_version": "F600-export-summary-v1", "date": "2026-10-26",
        "course_manifest": str(course_path.relative_to(ROOT)), "course_manifest_sha256": sha(course_path),
        "canonical_manifest": str(canonical_path.relative_to(ROOT)), "canonical_manifest_sha256": sha(canonical_path),
        "sft_manifest": str(sft_path.relative_to(ROOT)), "sft_manifest_sha256": sha(sft_path),
        "admitted_unique_tasks": course["admitted_unique_tasks"],
        "remaining_unique_tasks": course["remaining_unique_tasks"],
        "common_tasks": course["courses"]["common"]["admitted_unique_tasks"],
        "specialist_tasks": course["courses"]["specialist"]["admitted_unique_tasks"],
        "specialist_submechanisms": course["courses"]["specialist"]["submechanism_task_counts"],
        "scope": "Explicit F course only; existing S600 artifacts and manifests are not rewritten",
        "model_rollouts": 0, "verified_focal_action_pairs": 0, "GPU_training_started": False}
    (COURSE / "export-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
