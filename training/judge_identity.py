"""CPU-only identity shared by dispatch, official verdicts and baseline probes.

Only small runtime metadata and the official setup source are read. Model
weights, task content and reference patches are never accessed here.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


SCHEMA = "gemma.judge_environment.v1"
_PACKAGES = ("swegemma", "adk-eval-core", "adk-submission", "google-adk", "litellm")


def effective_evaluation_settings(submission: Path, overrides: Mapping[str, Any] | None = None) -> dict:
    import yaml

    path = next((submission / name for name in ("eval_config.yaml", "eval_config.yml")
                 if (submission / name).is_file()), None)
    document = yaml.safe_load(path.read_text(encoding="utf-8")) if path else {}
    document = document or {}
    section = document.get("evaluation", document)
    turns = section.get("max_turns", section.get("max_llm_calls"))
    settings = {
        "timeout_seconds": int(section.get("timeout_seconds", 300)),
        "verification_timeout_seconds": int(section.get("verification_timeout_seconds", 300)),
        "max_tool_calls": int(section.get("max_tool_calls", 100)),
        "max_time_minutes": float(section.get("max_time_minutes", 60.0)),
        "max_turns": int(turns) if turns is not None else None,
    }
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("evaluation overrides must be a JSON object")
        unknown = set(overrides) - set(settings)
        if unknown:
            raise ValueError(f"Unknown evaluation override keys: {sorted(unknown)}")
        for key, value in overrides.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Evaluation override {key} must be a positive finite number")
            if key != "max_time_minutes" and int(value) != value:
                raise ValueError(f"Evaluation override {key} must be an integer")
            settings[key] = float(value) if key == "max_time_minutes" else int(value)
    return settings


def canonical_environment(environment: Mapping[str, Any]) -> str:
    if not isinstance(environment, Mapping) or environment.get("schema") != SCHEMA:
        raise ValueError("unsupported or missing judge environment identity")
    return json.dumps(dict(environment), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def describe_judge_environment(*, data_root: str | Path, submission_dir: str | Path,
                               baseline_python: str | Path, sandbox: str = "subprocess",
                               evaluation_overrides: Mapping[str, Any] | None = None,
                               settings: Mapping[str, Any] | None = None,
                               baseline_info: Mapping[str, Any] | None = None) -> dict:
    """Run inside the official runtime, not the learner's Python environment."""
    if sandbox != "subprocess":
        raise ValueError("judge identity currently supports the subprocess training backend")
    if baseline_info is None:
        result = subprocess.run([
            str(baseline_python), "-I", "-c",
            "import importlib.metadata as m,json,sys;"
            "print(json.dumps({'version':list(sys.version_info[:2]),"
            "'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))",
        ], check=True, capture_output=True, text=True, timeout=15,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        baseline_info = json.loads(result.stdout)
    # Reuse the same cheap distribution metadata already read by the sandbox
    # manager. Do not hash site-packages, wheels or model weight trees.
    environment = {
        "schema": SCHEMA,
        "packages": {name: importlib.metadata.version(name) for name in _PACKAGES},
        "runtime_python": list(sys.version_info[:3]),
        "evaluation": dict(settings) if settings is not None else effective_evaluation_settings(
            Path(submission_dir), evaluation_overrides),
        "sandbox": sandbox,
        "baseline_python": {"version": list(baseline_info["version"]),
                            "packages": dict(baseline_info["packages"])},
        "setup_sha256": hashlib.sha256((Path(data_root) / "sandbox" / "setup.py").read_bytes()).hexdigest(),
    }
    canonical_environment(environment)
    return environment


def read_runtime_identity(runtime_python: str | Path, **settings: Any) -> dict:
    """One metadata-only subprocess before catalog construction or GPU work."""
    project_root = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([
        str(runtime_python), "-I", "-c",
        "import json,sys;sys.path.insert(0,sys.argv[1]);"
        "from training.judge_identity import describe_judge_environment;"
        "print(json.dumps(describe_judge_environment(**json.loads(sys.argv[2])),sort_keys=True))",
        project_root, json.dumps(settings, default=str, allow_nan=False),
    ], check=True, capture_output=True, text=True, timeout=30,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    environment = json.loads(result.stdout)
    canonical_environment(environment)
    return environment


def bind_run_identity(run_dir: Path, environment: Mapping[str, Any], *, resume: bool) -> str:
    """Keep legacy runs immutable; never relabel their cached evidence."""
    fingerprint = canonical_environment(environment)
    path = Path(run_dir) / "judge_identity.json"
    if resume:
        try:
            saved = canonical_environment(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError(
                "resume requires a recorded canonical judge identity; legacy runs are read-only, "
                "start a new run instead of relabeling old trajectories or baselines") from error
        if saved != fingerprint:
            raise RuntimeError(
                "judge environment differs from the recorded run; start a new run, "
                "old trajectories and baseline caches cannot be reused under a changed judge")
    else:
        # The caller has rejected existing run state before reaching here.
        with path.open("x", encoding="utf-8") as output:
            output.write(fingerprint + "\n")
    return fingerprint
