"""Version official initial prompts separately from raw human replay messages."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

from swegemma.harness.agent_runner import build_agent_prompt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--declarations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    names = [d["name"] for d in json.loads(args.declarations.read_text())]
    rows = []
    for result_path in args.results:
        result = json.loads(result_path.read_text())
        if not result.get("resolved_by_local_official_verify_task"):
            raise ValueError("Only locally verified demonstrations are eligible")
        source_path = Path(result["adapted_task"])
        source = json.loads(source_path.read_text())
        # Explicit input allowlist: no gold, test_patch, required lists or hindsight.
        task = SimpleNamespace(**{key: source.get(key) for key in (
            "repo", "problem_statement", "hints_text", "base_commit")})
        config = SimpleNamespace(
            budget=SimpleNamespace(**result["evaluation_budget"]),
            harness=SimpleNamespace(**result["harness_limits"]),
        )
        prompt = build_agent_prompt(task, config, workspace_tree="",
                                    enable_sandbox_testing=True, declared_tools=names)
        messages_path = Path(result["reports"]["repaired"]["path"]).parent / "messages.json"
        raw_messages = json.loads(messages_path.read_text())
        already_official = raw_messages[0]["content"] == prompt
        rows.append({
            "instance_id": result["instance_id"], "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "replay_result": str(result_path), "replay_result_sha256": sha(result_path),
            "task_input": str(source_path), "task_input_sha256": sha(source_path),
            "input_allowlist": ["repo", "problem_statement", "hints_text", "base_commit"],
            "budget": result["evaluation_budget"], "harness": result["harness_limits"],
            "generator": "unmodified swegemma.harness.agent_runner.build_agent_prompt",
            "swegemma_version": importlib.metadata.version("swegemma"),
            "workspace_tree": "omitted optional field; no reconstructed future tree",
            "prompt_change_kind": ("raw initial prompt already matches official builder" if already_official else
                                   "explicit collector-side official-wrapper augmentation; raw replay messages unchanged"),
            "model_sampled_under_augmented_prompt": False,
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"schema_version": "official-task-prompts-v1",
                                    "tasks": rows}, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"output": str(args.out), "sha256": sha(args.out), "tasks": len(rows)}))


if __name__ == "__main__":
    main()
