"""Package the P014 12B base/behavior comparison into one Kaggle code_file."""
from pathlib import Path
import ast
import hashlib
import json
import subprocess


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    # Use the same runtime setup as the 31B study, without importing/executing it.
    shared_builder = ROOT / "research/014/eval-31b/build_carrier.py"
    parsed = ast.parse(shared_builder.read_text())
    bootstrap = next(ast.literal_eval(node.value) for node in parsed.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "BOOTSTRAP"
                             for target in node.targets))
    bootstrap = bootstrap.replace("os.environ.update(", "P014_SESSION_START = time.monotonic()\nos.environ.update(", 1)
    subset = json.loads((HERE / "subset-freeze.json").read_text())
    panel_path = ROOT / "research/014/panel/material/agent-tasks.json"
    document = json.loads(panel_path.read_text())
    tasks = document if isinstance(document, list) else document["tasks"]
    by_id = {task["instance_id"]: task for task in tasks}
    selected = subset["task_ids"]
    missing = sorted(set(selected) - set(by_id))
    if missing:
        raise RuntimeError("Preselected 12B issues absent from prepared panel: " + str(missing))
    bundles = {
        arm: {str(path.relative_to(HERE / "agents" / arm)): path.read_text()
              for path in sorted((HERE / "agents" / arm).rglob("*")) if path.is_file()}
        for arm in ("base", "adapter")
    }
    settings = {
        "study": "12b-behavior-pair-v1",
        "model_alias": "gemma-4-12b-it-qat-w4a16-ct",
        "model_path_pattern": "gemma-4-12b-it-qat-w4a16-ct/2",
        "task_ids": selected,
        "subset_selection": subset,
        "budgets": {"timeout_seconds": 240, "max_tool_calls": 28,
                    "max_time_minutes": 8, "max_turns": 80},
        "concurrency": 4,
        "wall_seconds": 6900,
        "attempt_seconds": 1020,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "quota_estimate_upper_hours": 4.0,
        "purpose": "Same 12B agent and budget; only repair LoRA differs",
    }
    worker = (HERE / "attempt_worker.py").read_text()
    coordinator = (HERE / "coordinator.py").read_text()
    source = (
        "# Frozen P014 12B repair-only LoRA comparison; no training or submission.\n"
        + bootstrap + "\nSETTINGS = " + repr(settings)
        + "\nBUNDLE = " + repr(bundles)
        + "\nWORKER_SOURCE = " + repr(worker) + "\n" + coordinator
    )
    ast.parse(source)
    if len(source.encode()) >= 1000000:
        raise RuntimeError("Single code_file unexpectedly exceeds Kaggle text limit")
    kernel = HERE / "kernel"
    kernel.mkdir(exist_ok=True)
    (kernel / "run.py").write_text(source)
    metadata = json.loads((ROOT / "research/013/behavior-sft/carrier/kernel/kernel-metadata.json").read_text())
    metadata.update(
        id="ccjjllt/p014-12b-behavior-pair-v1",
        title="P014 12B Behavior Pair V1",
        code_file="run.py",
        dataset_sources=["metric/gemma-4-developer-agent-wheelhouse"],
        kernel_sources=["ccjjllt/p014-frozen-eight-material-cpu", "ccjjllt/p013-12b-behavior-sft-34"],
        model_sources=["google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2"],
    )
    (kernel / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {
        "settings": settings,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_bytes": len(source.encode()),
        "worker_sha256": hashlib.sha256(worker.encode()).hexdigest(),
        "coordinator_sha256": hashlib.sha256(coordinator.encode()).hexdigest(),
        "agent_task_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "shared_bootstrap_source": str(shared_builder.relative_to(ROOT)),
        "adapter_source": json.loads((HERE / "adapter-source.json").read_text()),
        "model_pin": json.loads((HERE / "model-pin.json").read_text()),
        "members": {arm: {rel: hashlib.sha256(text.encode()).hexdigest()
                          for rel, text in files.items()} for arm, files in bundles.items()},
    }
    (HERE / "run-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"kernel": str(kernel), "bytes": manifest["source_bytes"],
                      "sha256": manifest["source_sha256"], "task_ids": selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
