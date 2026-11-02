r"""Paired public-task evaluation of a BF16 base and an exported LoRA.

Run explicitly on a separately authorized GPU host with the existing training
environment. Importing this module or requesting --help never loads a model.
This reuses official_runtime's complete judge and training sampling contract;
it does not claim to measure the competition's W4A16 CT serving backend.

Example (all paths are already prepared on the server)::

    python -m training.evaluation --config configs/a800_2x80.toml \
        --data-root /data/public --model-path /models/qat-unquantized \
        --adapter /runs/train/adapters/batch-0001 --task-id public-task-id \
        --seed 3407 --seed 3408 --walltime-seconds 7200 \
        --output-dir /runs/evaluation-01 --scratch-dir /scratch/evaluation-01

Both arms use identical tasks, RNG seeds, submission, budgets and wall clocks.
The CPU-only summarize_pairs function also accepts synthetic result records.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sys
import threading
import time
import tomllib

from .curriculum import load_tasks
from .recovery import atomic_json, file_digest, value_digest


PROJECT = Path(__file__).resolve().parents[1]
_PRIVATE_DATA = re.compile(r"(^|[/_.-])(heldout|holdout|reserved|secret|hidden)($|[/_.-])", re.I)
_CLAIMS = {
    "in_sample": "Public training-task evaluation; not independent generalization.",
    "excluded_public": (
        "Public tasks excluded from the supplied training catalog; not official heldout "
        "evaluation or proof of independent generalization. Earlier training exposure is not inferred."
    ),
}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _mean(values, denominator):
    values = [float(value) for value in values if _finite(value)]
    return {"mean": sum(values) / len(values) if values else None,
            "available_count": len(values),
            "coverage": len(values) / denominator if denominator else None}


def lightweight_result(result, calls=(), *, elapsed_seconds=None):
    """Keep judge aggregates and source fingerprints, never task/answer content."""
    valid = (result.get("verdict_valid") is True
             and result.get("verification_complete") is True
             and not result.get("infrastructure_error")
             and type(result.get("success")) is bool)
    features = result.get("reward_features") or {}
    timing = result.get("session_timing") or {}
    provenance = result.get("provenance") or {}
    components = {}
    for name in ("repair", "regression"):
        fraction = features.get(name + "_fraction")
        available = (valid and _finite(fraction) and 0 <= fraction <= 1
                     and features.get(name + "_available", True)
                     and features.get(name + "_coverage_complete", features.get("dense_coverage_complete", False)))
        counts = features.get(name + "_counts")
        components[name] = {
            "fraction": float(fraction) if available else None,
            "available": bool(available), "counts": counts,
            "unavailable_reason": None if available else (
                features.get(name + "_unavailable_reason") or
                features.get("unavailable_reason") or "no_complete_component_evidence"),
        }
    return {
        "valid": bool(valid), "success": result["success"] if valid else None,
        "verification_complete": result.get("verification_complete") is True,
        "test_exit_code": result.get("test_exit_code"),
        "agent_error": result.get("agent_error"),
        "infrastructure_error": result.get("infrastructure_error"),
        **components,
        "costs": {
            "wall_seconds": timing.get("wall_seconds"),
            "charged_seconds": timing.get("charged_seconds"),
            "worker_seconds": result.get("duration_seconds", elapsed_seconds),
            "verification_seconds": features.get("verification_duration_seconds"),
            "generated_tokens": sum(len(call["completion_ids"]) for call in calls) if calls else None,
            "model_calls": len(calls) if calls else result.get("llm_calls"),
            "tool_calls": result.get("tool_calls"),
        },
        "evidence": {"patch_sha256": features.get("patch_sha256"),
                     "junit_sha256": features.get("junit_sha256"),
                     "public_tasks_sha256": provenance.get("public_tasks_sha256"),
                     "harness_packages": provenance.get("packages"),
                     "official_starter_source_sha256": provenance.get("official_starter_source_sha256"),
                     "effective_evaluation": provenance.get("evaluation"),
                     "context_management": provenance.get("context_management"),
                     "sampling": provenance.get("sampling"),
                     "session_clock": provenance.get("session_clock")},
    }


def summarize_pairs(rows, *, planned_pairs=None, population="in_sample"):
    """Report common-pair comparisons without treating absent verdicts as losses.

    One row is {task_id, seed, base: lightweight_result, adapter: ...}; either
    arm may be absent after an interruption. Every seed is a separate episode,
    not an independent task or a pass@k attempt from which to select the best.
    """
    rows = list(rows)
    planned = len(rows) if planned_pairs is None else planned_pairs
    if type(planned) is not int or planned < len(rows) or population not in _CLAIMS:
        raise ValueError("Invalid planned pair count or evaluation population")
    identities = [(row["task_id"], row["seed"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate task/seed pairs would bias evaluation")

    def valid(arm):
        return isinstance(arm, dict) and arm.get("valid") is True and type(arm.get("success")) is bool

    arms = {}
    for name in ("base", "adapter"):
        attempted = [row[name] for row in rows if isinstance(row.get(name), dict)]
        observed = [row[name] for row in rows if valid(row.get(name))]
        successes = sum(arm["success"] for arm in observed)
        arms[name] = {
            "attempted_count": len(attempted), "valid_count": len(observed), "resolved_count": successes,
            "resolution_rate": successes / planned if planned and len(observed) == planned else None,
            "observed_resolution_rate": successes / len(observed) if observed else None,
            "coverage": len(observed) / planned if planned else None,
            **{component: _mean([arm.get(component, {}).get("fraction") for arm in observed], planned)
               for component in ("repair", "regression")},
            "cost_population": "all recorded arms, including incomplete or infrastructure-failed attempts",
            "costs": {cost: _mean([arm.get("costs", {}).get(cost) for arm in attempted], planned)
                      for cost in ("wall_seconds", "charged_seconds", "worker_seconds", "verification_seconds",
                                   "generated_tokens", "model_calls", "tool_calls")},
            "valid_costs": {cost: _mean([arm.get("costs", {}).get(cost) for arm in observed], planned)
                            for cost in ("wall_seconds", "charged_seconds", "worker_seconds", "verification_seconds",
                                         "generated_tokens", "model_calls", "tool_calls")},
        }
    complete = [row for row in rows if valid(row.get("base")) and valid(row.get("adapter"))
                and not row.get("pairing_error")]
    transitions = {"fixed": 0, "regressed": 0, "both_resolved": 0, "both_failed": 0}
    for row in complete:
        base, adapter = row["base"]["success"], row["adapter"]["success"]
        key = ("both_resolved" if base else "fixed") if adapter else ("regressed" if base else "both_failed")
        transitions[key] += 1

    def delta(section, key):
        values = []
        for row in complete:
            left, right = row["base"].get(section, {}).get(key), row["adapter"].get(section, {}).get(key)
            if _finite(left) and _finite(right):
                values.append(right - left)
        return _mean(values, planned)

    count = len(complete)
    return {"schema_version": 1, "record_type": "paired_evaluation",
            "planned_pairs": planned, "recorded_pairs": len(rows),
            "valid_pairs": count, "paired_coverage": count / planned if planned else None,
            "population": population, "claim": _CLAIMS[population], "arms": arms,
            "unit": "one fixed task and seed; no best-of-seeds selection",
            "paired": {
                "base_resolution_rate": sum(row["base"]["success"] for row in complete) / count if count else None,
                "adapter_resolution_rate": sum(row["adapter"]["success"] for row in complete) / count if count else None,
                "resolution_delta": (transitions["fixed"] - transitions["regressed"]) / count if count else None,
                "transitions": transitions,
                "repair_delta": delta("repair", "fraction"),
                "regression_delta": delta("regression", "fraction"),
                "wall_seconds_delta": delta("costs", "wall_seconds"),
                "generated_tokens_delta": delta("costs", "generated_tokens"),
            }}


def _public(path):
    path = Path(path).expanduser().resolve(strict=True)
    if _PRIVATE_DATA.search(str(path)):
        raise ValueError("Evaluation accepts explicitly prepared public competition assets only")
    return path


def _fingerprint(path):
    path = Path(path)
    info = path.stat()
    return {"path": str(path), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
            "sha256": file_digest(path)}


def _submission_identity(root):
    return [{"name": path.relative_to(root).as_posix(), "sha256": file_digest(path)}
            for path in sorted(root.rglob("*"))
            if path.is_file() and "adapters" not in path.relative_to(root).parts
            and path.suffix in {".yaml", ".yml", ".md", ".txt", ".py", ".json"}]


def _model_identity(path, policy):
    # Full model weights are never opened or hashed during evaluation setup.
    weights = []
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix in {".safetensors", ".bin"}:
            info = item.stat()
            weights.append({"name": item.relative_to(path).as_posix(),
                            "bytes": info.st_size, "mtime_ns": info.st_mtime_ns})
    return {"path": str(path), "declared_name": policy.get("model_name"),
            "declared_revision": policy.get("model_revision"),
            "declared_source_uri": policy.get("model_source_uri"),
            "config": _fingerprint(path / "config.json"),
            "weights": weights, "weight_fingerprint_method": "size_mtime_ns_only"}


def _load_exported_adapter(policy, path):
    """Load the existing export into the same ordinary LoRA layout, without merge."""
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    supplied, current = LoraConfig.from_pretrained(str(path)), policy.model.peft_config["default"]
    for field in ("r", "lora_alpha", "lora_dropout", "bias", "target_modules", "rank_pattern",
                  "alpha_pattern", "modules_to_save", "use_rslora", "use_dora", "fan_in_fan_out"):
        if getattr(supplied, field) != getattr(current, field):
            raise ValueError(f"Exported adapter {field} differs from the configured policy layout")
    expected = get_peft_model_state_dict(policy.model, adapter_name="default")
    tensors = load_file(str(path / "adapter_model.safetensors"), device="cpu")
    if tensors.keys() != expected.keys():
        raise ValueError("Exported adapter tensor names differ from the configured text projections")
    for name, value in tensors.items():
        if value.shape != expected[name].shape or not bool(policy.torch.isfinite(value).all()):
            raise ValueError(f"Invalid exported adapter tensor: {name}")
    loaded = set_peft_model_state_dict(policy.model, tensors, adapter_name="default")
    if loaded.unexpected_keys or any(".lora_" in name for name in loaded.missing_keys):
        raise ValueError("PEFT did not load the entire exported adapter")
    policy._mode = None
    # Unsloth's mode switch runs before disable_adapter, so inference setup
    # cannot re-enable an adapter inside the base arm's generation thread.
    policy._for_inference()
    policy.model.requires_grad_(False)


def _seed_policy(policy, seed):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    policy.torch.manual_seed(seed)
    policy.torch.cuda.manual_seed_all(seed)


def _run_arm(*, policy, endpoint, cfg, task, seed, arm, ordinal, scratch,
             data_root, submission, uid_base, deadline, expired, processes, process_lock):
    from .run import run_worker

    identity = f"eval-{ordinal:06d}-{arm}"
    folder = scratch / identity
    folder.mkdir(mode=0o700)
    context = policy.model.disable_adapter() if arm == "base" else nullcontext()
    calls, result = [], {}
    started = time.monotonic()
    with context:
        _seed_policy(policy, seed)
        # A fresh engine for each arm removes all KV/prefix state from its peer.
        try:
            endpoint.begin([identity], ordinal)
            credential = folder / "model_auth_token"
            descriptor = os.open(credential, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w") as out:
                out.write(endpoint.rollout_token(identity))
            runtime = cfg["runtime"]
            command = [str(Path(runtime["python"]).expanduser().absolute()),
                       str(PROJECT / "training" / "official_runtime.py"),
                       "--data-root", str(data_root), "--submission-dir", str(submission),
                       "--endpoint", endpoint.url, "--workspace", str(folder / "workspace"),
                       "--result-file", str(folder / "result.json"), "--rollout-id", identity,
                       "--sandbox", runtime.get("sandbox", "subprocess"),
                       "--agent-uid", str(uid_base + 2 * ordinal),
                       "--judge-uid", str(uid_base + 2 * ordinal + 1),
                       "--auth-token-file", str(credential), "--parent-pid", str(os.getpid()),
                       "--sandbox-baseline-python", runtime.get("sandbox_baseline_python", "/opt/gemma-sandbox-baseline/bin/python"),
                       "--session-clock", "wall", "--evaluation-overrides-json", json.dumps(runtime.get("evaluation", {})),
                       "--task-id", task["id"]]
            if runtime.get("sandbox_root"):
                command.extend(["--sandbox-root", runtime["sandbox_root"]])
            result = run_worker(command, folder, deadline, expired, processes, process_lock)
            if result.get("task_id") != task["id"] or result.get("rollout_id") != identity:
                raise ValueError("Official result does not belong to this evaluation arm")
            calls = endpoint.recover_rollout_calls(identity, ordinal)
            if not calls:
                raise ValueError("No confirmed consumed model calls for this evaluation arm")
        except Exception as error:
            result_path = folder / "result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
            result = {**result, "verdict_valid": False, "success": None,
                      "infrastructure_error": result.get("infrastructure_error") or f"{type(error).__name__}: {error}"}
        finally:
            endpoint.cancel_rollouts([identity], reason="evaluation_arm_finished")
            endpoint.end(raise_errors=False)
            credential = folder / "model_auth_token"
            credential.unlink(missing_ok=True)
    return {**lightweight_result(result, calls, elapsed_seconds=time.monotonic() - started),
            "rollout_id": identity, "scratch_path": str(folder)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/a800_2x80.toml")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--submission-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--walltime-seconds", type=float, required=True,
                        help="Total allowance including model loading; no automatic resume or budget renewal.")
    parser.add_argument("--population", choices=tuple(_CLAIMS), default="in_sample")
    parser.add_argument("--training-run-config", type=Path,
                        help="Original run_config.json, required to substantiate excluded_public membership.")
    args = parser.parse_args(argv)
    started = time.monotonic()
    if sys.platform != "linux":
        parser.error("Execution requires the separately authorized Linux GPU host; --help and summary imports are CPU-only")
    if (not math.isfinite(args.walltime_seconds) or args.walltime_seconds <= 0
            or len(set(args.task_id)) != len(args.task_id) or len(set(args.seed)) != len(args.seed)
            or any(not 0 <= seed < 2**32 for seed in args.seed)):
        parser.error("Require positive walltime, unique task IDs and unique seeds in [0, 2**32)")
    if args.population == "excluded_public" and args.training_run_config is None:
        parser.error("excluded_public requires the original --training-run-config")
    cfg = tomllib.loads(args.config.read_text())
    if cfg["runtime"].get("sandbox", "subprocess") == "subprocess" and os.geteuid() != 0:
        parser.error("The isolated subprocess agent/judge runtime requires root on the GPU host")
    if cfg.get("sampling") != {"temperature": 1.0, "top_p": 1.0, "top_k": 0}:
        parser.error("The existing policy endpoint supports full categorical sampling only")
    data_root, model_path, adapter = _public(args.data_root), _public(args.model_path), _public(args.adapter)
    tasks_by_id = {row["id"]: row for row in load_tasks(data_root / "tasks.jsonl", include_ids=args.task_id)}
    tasks = [tasks_by_id[identifier] for identifier in args.task_id]
    submission = _public(args.submission_dir or data_root / "sample_submission")
    training_identity = None
    if args.training_run_config:
        original_path = _public(args.training_run_config)
        original = json.loads(original_path.read_text())
        if not isinstance(original.get("training_tasks"), list):
            parser.error("Training run config lacks the original eligible public task catalog")
        catalog = {row["id"] for row in original["training_tasks"]}
        if not adapter.is_relative_to(original_path.parent / "adapters"):
            parser.error("The adapter must belong to the supplied training run's adapters directory")
        if args.population == "excluded_public" and catalog.intersection(args.task_id):
            parser.error("An excluded_public task belongs to the supplied training catalog")
        training_identity = {**_fingerprint(original_path),
                             "source_binding_sha256": value_digest(original.get("source_binding")),
                             "original_recipe": original.get("recipe"), "eligible_task_ids": sorted(catalog)}
    cfg = copy.deepcopy(cfg)
    cfg["policy"]["model_path"] = str(model_path)
    cfg["runtime"]["session_clock"] = "wall"
    cfg.setdefault("generation", {}).update(max_batch_size=1, batch_wait_ms=0)
    output, scratch = args.output_dir.expanduser().absolute(), args.scratch_dir.expanduser().absolute()
    if output.exists() or scratch.exists() or output == scratch:
        parser.error("Use fresh distinct output and scratch directories")
    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    scratch.mkdir(parents=True, mode=0o700)
    model_identity = _model_identity(model_path, cfg["policy"])
    settings = {"policy": {key: cfg["policy"].get(key) for key in ("max_context", "response_parser")},
                "sampling": cfg["sampling"], "generation": cfg["generation"],
                "runtime": {key: cfg["runtime"].get(key) for key in ("sandbox", "evaluation", "session_clock")},
                "base_model": {key: model_identity[key] for key in ("declared_name", "declared_revision")},
                "base_model_config_sha256": model_identity["config"]["sha256"],
                "submission": _submission_identity(submission),
                "runtime_source_sha256": file_digest(PROJECT / "training/official_runtime.py"),
                "tasks": tasks, "seeds": args.seed,
                "policy_execution": "sequential; reset Python/NumPy/Torch/CUDA RNG per arm; fresh generation engine",
                "seed_scope": "model sampling RNG; repository tools and CUDA kernels may remain nondeterministic"}
    manifest = {"schema_version": 1, "backend": "qat_unquantized_bf16",
                "competition_ct_evaluated": False, "settings": settings, "settings_sha256": value_digest(settings),
                "population": args.population, "claim": _CLAIMS[args.population],
                "model": model_identity, "evaluation_recipe": cfg,
                "recipe_source": _fingerprint(args.config),
                "adapter": {"config": _fingerprint(adapter / "adapter_config.json"),
                            "weights": _fingerprint(adapter / "adapter_model.safetensors")},
                "training_run_config": training_identity,
                "public_tasks_sha256": file_digest(data_root / "tasks.jsonl"),
                "walltime_seconds": args.walltime_seconds, "scratch_dir": str(scratch),
                "planned_pairs": len(tasks) * len(args.seed)}
    atomic_json(output / "manifest.json", manifest)
    summary_identity = {
        "settings_sha256": manifest["settings_sha256"],
        "adapter_sha256": manifest["adapter"]["weights"]["sha256"],
        "training_run_config_sha256": training_identity["sha256"] if training_identity else None,
        "manifest_sha256": file_digest(output / "manifest.json"),
        "task_seed_set_sha256": value_digest({"tasks": tasks, "seeds": args.seed}),
        "backend": manifest["backend"], "competition_ct_evaluated": False,
    }
    # All accelerator dependencies are intentionally inside the explicit CLI run.
    from .endpoint import PolicyEndpoint
    from .policy import Policy
    from .run import stop_process_tree

    deadline = started + args.walltime_seconds
    expired, processes, process_lock = threading.Event(), set(), threading.RLock()
    policy = endpoint = None
    rows, stop_reason = [], None

    def cancel(*_):
        expired.set()
        if policy is not None:
            policy.abort_generation.set()
        with process_lock:
            for process in list(processes):
                stop_process_tree(process, grace=0.5)

    timer = threading.Timer(max(0.0, deadline - time.monotonic()), cancel)
    timer.daemon = True
    previous_signals = {kind: signal.signal(kind, cancel) for kind in (signal.SIGTERM, signal.SIGINT)}
    timer.start()
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError("Evaluation allowance exhausted before model loading")
        policy = Policy(cfg["policy"])
        _load_exported_adapter(policy, adapter)
        if expired.is_set():
            raise TimeoutError("Evaluation allowance exhausted during model loading")
        endpoint = PolicyEndpoint(policy, scratch, generation_config=cfg["generation"])
        endpoint.start()
        uid_base = 100000 + random.SystemRandom().randrange(1000000)
        ordinal = 0
        for task in tasks:
            for seed in args.seed:
                if expired.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError("Evaluation allowance exhausted")
                row = {"task_id": task["id"], "repo": task["repo"], "base_commit": task["base_commit"],
                       "seed": seed, "settings_sha256": manifest["settings_sha256"],
                       "adapter_sha256": manifest["adapter"]["weights"]["sha256"]}
                # Alternate order deterministically to avoid always timing the base cold.
                order = ("base", "adapter") if len(rows) % 2 == 0 else ("adapter", "base")
                row["arm_order"] = list(order)
                rows.append(row)
                for arm in order:
                    if expired.is_set():
                        raise TimeoutError("Evaluation allowance exhausted between arms")
                    ordinal += 1
                    row[arm] = _run_arm(policy=policy, endpoint=endpoint, cfg=cfg, task=task, seed=seed,
                                        arm=arm, ordinal=ordinal, scratch=scratch, data_root=data_root,
                                        submission=submission, uid_base=uid_base, deadline=deadline,
                                        expired=expired, processes=processes, process_lock=process_lock)
                    atomic_json(output / f"pair-{len(rows):04d}.json", row)
                if row["base"]["valid"] and row["adapter"]["valid"]:
                    comparable = ("public_tasks_sha256", "harness_packages", "official_starter_source_sha256",
                                  "effective_evaluation", "context_management", "sampling", "session_clock")
                    if any(row["base"]["evidence"][key] != row["adapter"]["evidence"][key] for key in comparable):
                        row["pairing_error"] = "Official runtime or task-source evidence differs between arms"
                        atomic_json(output / f"pair-{len(rows):04d}.json", row)
                atomic_json(output / "paired-summary.json", {
                    **summarize_pairs(rows, planned_pairs=manifest["planned_pairs"], population=args.population),
                    **summary_identity})
                print(json.dumps({"event": "paired_evaluation", "task_id": task["id"], "seed": seed,
                                  "base": row["base"]["success"], "adapter": row["adapter"]["success"]}), flush=True)
    except Exception as error:
        stop_reason = f"{type(error).__name__}: {error}"
    finally:
        cancel()
        timer.cancel()
        if endpoint is not None:
            try:
                endpoint.close()
            except Exception as error:
                stop_reason = stop_reason or f"Endpoint cleanup failed: {type(error).__name__}: {error}"
        for kind, previous in previous_signals.items():
            signal.signal(kind, previous)
        summary = summarize_pairs(rows, planned_pairs=manifest["planned_pairs"], population=args.population)
        summary.update(**summary_identity, elapsed_seconds=time.monotonic() - started,
                       stop_reason=stop_reason)
        atomic_json(output / "paired-summary.json", summary)
    return 0 if summary["valid_pairs"] == manifest["planned_pairs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
