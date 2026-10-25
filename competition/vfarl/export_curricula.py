"""Partition admitted action data into four primary courses and common replay.

This performs data preparation only. It neither sets training mixture weights nor
starts training. The source action prefixes and observations remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(manifest_path: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    actions_path = Path(manifest["recommended_training_input"])
    if sha(actions_path) != manifest["action_output_sha256"]:
        raise ValueError("The admitted master action file differs from its export manifest")
    rows = [json.loads(line) for line in actions_path.read_text().splitlines() if line.strip()]
    if len(rows) != manifest["action_examples"]:
        raise ValueError("Master action count differs from its export manifest")
    output.mkdir(parents=True, exist_ok=True)
    courses = {}
    for name in ("G_primary", "S_primary", "F_primary", "R_primary", "shared_coldstart"):
        selected = rows if name == "shared_coldstart" else [
            row for row in rows if row["primary_adapter"] == name[0]]
        target = output / (name + ".jsonl")
        # Keep the complete action record byte semantics; no new prompt text,
        # hindsight labels or invented termination is inserted here.
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
        tasks = {row["instance_id"]: row for row in selected}
        courses[name] = {"path": str(target), "sha256": sha(target),
            "task_episodes": len(tasks), "action_examples": len(selected),
            "assistant_loss_tokens": sum(row["assistant_loss_tokens"] for row in selected),
            "submechanism_task_counts": dict(Counter(row["submechanism"] for row in tasks.values())),
            "instance_ids": sorted(tasks),
            "maximum_action_context_tokens": max((row["context_tokens"] for row in selected), default=0)}
    primary_ids = [iid for name, course in courses.items() if name != "shared_coldstart"
                   for iid in course["instance_ids"]]
    if len(primary_ids) != len(set(primary_ids)) or set(primary_ids) != set(courses["shared_coldstart"]["instance_ids"]):
        raise ValueError("Primary courses must partition the shared admitted task identities")
    result = {"schema_version": "admitted-adapter-curricula-v1",
        "source_manifest": str(manifest_path), "source_manifest_sha256": sha(manifest_path),
        "source_action_file": str(actions_path), "source_action_file_sha256": sha(actions_path),
        "courses": courses,
        "use_partition": "train_only",
        "selection_policy": "Four disjoint primary mechanism courses plus the complete shared six-tool coldstart course. The shared course repeats these same tasks, not additional independent examples.",
        "mixture_weights": None,
        "mixture_note": "G may use common repair mechanisms; experts need general replay/KL retention. This export does not decide their sampling proportions or claim that current mechanism coverage is sufficient.",
        "evidence_kind": "Codex-scripted off-policy demonstrations with actual official-tool observations and complete repaired verification",
        "model_rollouts": 0, "verified_focal_action_pairs": 0,
        "GPU_training_started": False}
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.manifest, args.out)
    print(json.dumps({name: {key: row[key] for key in ("task_episodes", "action_examples", "assistant_loss_tokens")}
                      for name, row in result["courses"].items()}, ensure_ascii=False))
