"""Single-stream SPO training launcher (the complete CPU/GPU orchestrator).

Wiring (one process, colocated 2xA800):
  TaskCatalog -> SPOSampler -> RolloutScheduler -> ExecutionManager
  (OfficialWorker + PolicyEndpoint/GPU + RequestManager) -> RolloutEngine
  -> RewardEngine -> ValueTracker -> TrajectoryBuffer -> ledger.reserve_update
  -> GpuLease.training + SPOTrainer.optimize_batch -> AdapterExporter
  -> AdapterManager/DoubleBuffer publish -> ledger.commit_update
  -> serving adapter resume.

Usage (server, authorized session only):
  python -m training.run_spo --config configs/spo_a800_2x80.toml \
    --data-root /root/gemma4/official/competition \
    --model-path /root/autodl-fs/models/gemma4-31b-qat \
    --run-dir /root/autodl-tmp/gemma4/runs/spo-next \
    --backend-url http://127.0.0.1:8000

The SGLang server itself is deployed separately; this launcher only speaks the
HTTP contract in training/rollout/backend.py (health/submit/drain/
release_memory_occupation/resume_memory_occupation/load_lora_adapter).
Exit codes: 0 finished all batches, 75 budget exhausted / continue with
--resume, 2 configuration/run failure.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import threading
import time
import tomllib
from pathlib import Path


REQUIRED_SECTIONS = ("policy", "run", "inference", "spo", "optimizer", "rewards", "adapter")
FORBIDDEN_KEYS = ("num_generations", "groups_per_batch", "max_candidate_groups_per_batch",
                  "num_iterations", "loss_type", "scale_rewards", "advantage_std_floor")


def load_config(path: Path) -> dict:
    with open(path, "rb") as handle:
        cfg = tomllib.load(handle)
    if cfg.get("algorithm") != "single_stream_spo":
        raise ValueError("algorithm must be single_stream_spo")
    for section in REQUIRED_SECTIONS:
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"missing config section: {section}")
    text = Path(path).read_text()
    for key in FORBIDDEN_KEYS:
        if key in text:
            raise ValueError(f"legacy GRPO key must not appear in SPO config: {key}")
    policy = cfg["policy"]
    if (policy.get("lora_rank"), policy.get("lora_alpha")) != (128, 256):
        raise ValueError("SPO requires lora_rank=128, lora_alpha=256")
    if float(policy.get("lora_dropout", 0.0)) != 0.0:
        raise ValueError("SPO requires lora_dropout=0")
    spo = cfg["spo"]
    if int(spo.get("target_action_tokens", 0)) <= 0 or int(spo.get("max_policy_lag", -1)) < 0:
        raise ValueError("invalid spo token budget or lag")
    if cfg["adapter"].get("allow_resident_pair") is not False:
        raise ValueError("allow_resident_pair must be false until VRAM is measured")
    if cfg["rewards"].get("objective_version") != "official_differential_spo_v1":
        raise ValueError("wrong rewards objective_version")
    return cfg


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_dir(model_path: Path, policy: dict) -> dict:
    """Torch-free preflight mirroring Policy's structural gates.

    Checks the local checkpoint directory (model_type gemma4, 60 text layers,
    no quantization_config, tokenizer present) without importing torch/CUDA.
    Returns the cheap model-identity record (config/tokenizer hashes only;
    full-weight scans stay disabled per project convention).
    """
    raw = json.loads((model_path / "config.json").read_text())
    text_config = raw.get("text_config", {})
    if raw.get("model_type") != "gemma4" or text_config.get("num_hidden_layers") != 60:
        raise ValueError("model directory is not the Gemma 4 31B checkpoint")
    if raw.get("quantization_config"):
        raise ValueError("use the official QAT-unquantized BF16 checkpoint, not a packed/NF4 base")
    tokenizer_files = [p for p in ("tokenizer.json", "tokenizer.model", "spiece.model")
                        if (model_path / p).is_file()]
    if not tokenizer_files:
        raise ValueError("model directory has no tokenizer file")
    if not str(policy.get("model_revision") or "").strip():
        raise ValueError("policy.model_revision is required for train/infer identity")
    if policy.get("response_parser", "official_vllm_gemma4_0_19_1") != "official_vllm_gemma4_0_19_1":
        raise ValueError("unsupported response parser")
    return {
        "model_path": str(model_path),
        "model_name": policy.get("model_name"),
        "model_revision": policy.get("model_revision"),
        "model_source_uri": policy.get("model_source_uri"),
        "config_json_sha256": _sha256_file(model_path / "config.json"),
        "tokenizer_files": {name: _sha256_file(model_path / name) for name in tokenizer_files},
        "lora": {"rank": policy.get("lora_rank"), "alpha": policy.get("lora_alpha"),
                 "dropout": policy.get("lora_dropout")},
    }


def build_task_keys(tasks_path: Path, fingerprint: str,
                    include: list[str], exclude: list[str]) -> list[object]:
    """Judge TaskKeys for catalog binding; heldout rows are never read."""
    from .reward.junit_bridge import task_key_for
    from .trajectory.schema import TaskKey
    keys: list[object] = []
    with open(tasks_path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("heldout") is True:
                raise ValueError("heldout rows cannot enter the public training catalog")
            task_id = row.get("instance_id")
            if not task_id or task_id in exclude or (include and task_id not in include):
                continue
            keys.append(TaskKey.from_dict(task_key_for(
                task_id, row.get("repo", ""), row.get("base_commit", ""),
                row.get("test_patch", "") or "", fingerprint,
            )))
    if include and len(keys) != len(set(include) - set(exclude)):
        raise ValueError("requested public tasks were not found")
    if not keys:
        raise ValueError("no public training tasks remain after selection")
    return keys


def _emit(metrics_path: Path, event: str, **fields: object) -> None:
    record = {"event": event, "unix": time.time(), **fields}
    with open(metrics_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def run_spo(args: argparse.Namespace) -> int:
    import torch  # noqa: F401  (GPU-only past this point, except --check-only)
    from .adapter.double_buffer import DoubleBuffer
    from .adapter.exporter import AdapterExporter
    from .adapter.manager import AdapterManager
    from .data.curriculum import TaskCatalog
    from .endpoint import PolicyEndpoint
    from .environment.execution_manager import ExecutionManager
    from .environment.worker import OfficialWorker, OfficialWorkerConfig
    from .environment.workspace_cache import WorkspaceCache
    from .policy import Policy
    from .reward.baseline import EmptyPatchBaselineStore
    from .reward.engine import RewardEngine
    from .rollout.backend import GpuLease, SGLangBackend, VLLMBackend
    from .rollout.engine import RolloutEngine
    from .rollout.request_manager import RequestManager
    from .rollout.scheduler import BarrierActive, RolloutScheduler
    from .spo.sampler import SPOSampler
    from .spo.session import VersionBarrier, commit_optimized_update
    from .spo.trainer import make_spo_trainer
    from .spo.value_tracker import ValueTracker
    from .trajectory.buffer import TrajectoryBuffer
    from .trajectory.ledger import TrajectoryLedger
    from .trajectory.schema import BehaviorAdapter

    cfg = load_config(Path(args.config))
    data_root = Path(args.data_root).expanduser().resolve(strict=True)
    resolved_model = args.model_path or cfg["policy"].get("model_path")
    model_path = Path(resolved_model).expanduser().resolve(strict=True)
    run_dir = Path(args.run_dir).expanduser()
    tasks_path = data_root / "tasks.jsonl"
    if not tasks_path.is_file() or not (model_path / "config.json").is_file():
        raise ValueError("Prepare the official tasks.jsonl and model directory on the server")
    run = cfg["run"]
    walltime = float(run.get("walltime_seconds", 7200)) + float(args.additional_walltime_seconds or 0)
    reserve = float(run.get("update_reserve_seconds", 900))
    if not walltime > reserve > 0:
        raise ValueError("walltime must exceed the positive update reserve")
    max_batches = int(args.max_batches or run.get("max_batches", 32))
    max_workers = int(run.get("max_cpu_workers", 8))
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    identity = validate_model_dir(model_path, cfg["policy"])
    with open(run_dir / "model_identity.json", "w", encoding="utf-8") as handle:
        json.dump(identity, handle, ensure_ascii=False, indent=2, sort_keys=True)
    _emit(metrics_path, "model_identity", **identity)

    fingerprint = hashlib.sha256(json.dumps({
        "config": {k: cfg[k] for k in ("policy", "spo", "optimizer", "rewards", "adapter")},
        "tasks_sha256": _sha256_file(tasks_path),
    }, sort_keys=True, default=str).encode()).hexdigest()
    catalog = TaskCatalog(tasks_path, include_ids=cfg["data"].get("task_ids") or None,
                          exclude_ids=cfg["data"].get("exclude_task_ids") or ())
    catalog.bind_task_keys(build_task_keys(
        tasks_path, fingerprint, cfg["data"].get("task_ids") or [], cfg["data"].get("exclude_task_ids") or []))

    ledger = TrajectoryLedger(run_dir / "spo_stream.db")
    tracker = ValueTracker(
        ledger, prior_value=float(cfg["spo"].get("value_prior", 0.5)),
        prior_success=float(cfg["spo"].get("success_prior", 0.5)),
        prior_strength=float(cfg["spo"].get("prior_strength", 1.0)),
        kl_half_life=float(cfg["spo"].get("kl_half_life", 0.05)))
    sampler = SPOSampler(tracker, epsilon=float(cfg["spo"].get("sampling_epsilon", 0.05)))
    requests = RequestManager(run_dir / "requests")
    baselines = EmptyPatchBaselineStore(run_dir / "baselines")
    reward_engine = RewardEngine(baselines)
    trajectory_buffer = TrajectoryBuffer(
        ledger, target_action_tokens=int(cfg["spo"]["target_action_tokens"]),
        max_policy_lag=int(cfg["spo"]["max_policy_lag"]),
        advantage_epsilon=float(cfg["spo"].get("advantage_epsilon", 1.0e-6)))
    rollout_engine = RolloutEngine(requests, reward_engine, ledger, tracker, trajectory_buffer)

    engine_name = str(cfg["inference"].get("engine", "sglang"))
    backend_url = args.backend_url or cfg["inference"].get("base_url")
    if not backend_url:
        raise ValueError("inference backend URL is required (--backend-url)")
    backend = SGLangBackend(backend_url) if engine_name == "sglang" else VLLMBackend(backend_url)
    try:
        handshake = backend.start()
    except (ValueError, RuntimeError, OSError) as error:
        raise RuntimeError(
            f"inference backend handshake failed at {backend_url}: "
            f"{type(error).__name__}: {error}") from error
    from .rollout.gateway import InferenceGateway
    try:
        deployment = InferenceGateway(backend).probe_deployment()
    except (ValueError, RuntimeError, OSError) as error:
        deployment = {"reachable": False, "error": f"{type(error).__name__}: {error}"}
    handshake = {**handshake, "deployment_probe": deployment}
    lease = GpuLease(backend)
    adapters = AdapterManager(run_dir / "adapters")
    double_buffer = DoubleBuffer(adapters, backend, gpu_lease=lease, allow_resident_pair=False)

    policy_cfg = dict(cfg["policy"])
    policy_cfg["model_path"] = str(model_path)
    policy_cfg.setdefault("gradient_checkpointing", True)
    policy_cfg.setdefault("expected_gpu_count", 2)
    policy = Policy(policy_cfg)
    endpoint = PolicyEndpoint(policy, run_dir / "endpoint", int(cfg["runtime"].get("port", 0)), {})
    endpoint.start()
    try:
        worker_cfg = OfficialWorkerConfig(
            data_root=data_root,
            submission_dir=Path(args.submission_dir or (data_root / "sample_submission")).expanduser(),
            endpoint_url=endpoint.url,
            sandbox_root=Path(cfg["runtime"].get("sandbox_root", "/root/autodl-tmp/gemma4/sandboxes")).expanduser(),
            baseline_python=Path(cfg["runtime"].get("sandbox_baseline_python", "/opt/gemma-sandbox-baseline/bin/python")),
            agent_uid_base=int(args.agent_uid_base or 61000),
            runtime_python=Path(cfg["runtime"].get("python", "/opt/gemma-runtime/bin/python")),
            session_clock=cfg["runtime"].get("session_clock", "queue_excluded"),
            event_summarizer=cfg["runtime"].get("event_summarizer", "official"))
        worker = OfficialWorker(worker_cfg, WorkspaceCache(
            run_dir / "workspaces", data_root=data_root))
        exec_manager = ExecutionManager(
            worker, endpoint, requests, max_workers=max_workers,
            max_attempts=int(run.get("max_worker_retries", 1)) + 1, emit=lambda n, **f: _emit(metrics_path, n, **f))
        scheduler = RolloutScheduler(
            catalog.sampler_tasks(), sampler, double_buffer.rollout,
            seed=int(policy_cfg.get("seed", 3407)), health=catalog.healthy)
        barrier = VersionBarrier(
            scheduler, exec_manager, double_buffer,
            drain_timeout_seconds=float(run.get("barrier_drain_timeout_seconds", 600)),
            emit=lambda n, **f: _emit(metrics_path, n, **f))
        exporter = AdapterExporter()
        trainer = make_spo_trainer(policy, ledger, {
            "spo": cfg["spo"],
            "optimizer": {**cfg["optimizer"], "total_optimizer_steps": max(1, max_batches)},
        })
        trainer_state = run_dir / "spo_trainer_state.pt"
        if args.resume and trainer_state.is_file():
            import torch as _torch
            trainer.restore_training_state(_torch.load(trainer_state, map_location="cpu", weights_only=False))
        if adapters.active is None:
            genesis_dir = run_dir / "adapters" / "staging-genesis"
            manifest = exporter.export(policy, genesis_dir, version=0, parent_version=None, optimizer_step=0)
            adapters.publish(manifest, genesis_dir, activate=True)
            _emit(metrics_path, "adapter_genesis", version=0)
        else:
            _emit(metrics_path, "adapter_resumed", version=adapters.active.version)

        start = time.monotonic()
        deadline = start + walltime
        stop_submit = start + walltime - reserve
        batches = 0
        in_flight: dict[str, tuple[object, object]] = {}
        lock = threading.RLock()

        def settle_done(future: concurrent.futures.Future, trajectory_id: str, assignment: object) -> None:
            try:
                try:
                    double_buffer.release_trajectory(trajectory_id)
                except (KeyError, ValueError):
                    pass
                error = future.exception()
                if error is not None:
                    _emit(metrics_path, "trajectory_failed", trajectory_id=trajectory_id,
                          error=f"{type(error).__name__}: {error}")
                    try:
                        catalog.record_outcome(assignment.task_key, resolved=None,
                                               infrastructure_error=f"{type(error).__name__}")
                    except (ValueError, KeyError, TypeError):
                        pass
                else:
                    try:
                        reward = ledger.get_reward(trajectory_id)
                        catalog.record_outcome(
                            assignment.task_key, resolved=reward.resolved,
                            infrastructure_error=None if reward.eligible else reward.exclusion_reason)
                    except (KeyError, ValueError, TypeError):
                        pass
            finally:
                with lock:
                    in_flight.pop(trajectory_id, None)

        def do_update(update_id: str) -> bool:
            rollout = double_buffer.rollout
            batch = trajectory_buffer.reserve(update_id, policy_before=rollout)
            if batch is None:
                return False
            with barrier.update_window():
                concurrent.futures.wait([f for f, _ in in_flight.values()])
                with lease.training():
                    stats = trainer.optimize_batch(batch)
                version = rollout.version + 1
                candidate_dir = run_dir / "adapters" / f"staging-v{version}"
                manifest = exporter.export(
                    policy, candidate_dir, version=version,
                    parent_version=rollout.version,
                    optimizer_step=trainer.completed_optimizer_steps)
                adapters.publish(manifest, candidate_dir)
                double_buffer.begin_training(manifest)
                import torch as _torch
                _torch.save(trainer.export_training_state(), trainer_state)
                behavior_before = BehaviorAdapter(
                    version=rollout.version, sha256=rollout.adapter_sha256)
                behavior_after = BehaviorAdapter(version=manifest.version, sha256=manifest.adapter_sha256)
                commit_optimized_update(trainer, behavior_after,
                                        version_chain=(behavior_before, behavior_after))
                barrier.publish_or_raise()
                _emit(metrics_path, "spo_update", update_id=update_id, version=version,
                      action_tokens=batch.total_action_tokens, published=True,
                      kl_value=stats.get("kl_value"), token_count=stats.get("token_count"))
                return True

        update_index = len(ledger.list_updates()) if args.resume else 0
        _emit(metrics_path, "spo_run_started", config=cfg.get("algorithm"),
              tasks=len(catalog), resume=bool(args.resume),
              inference={"engine": engine_name, "base_url": backend_url},
              backend_handshake=handshake,
              model_revision=identity["model_revision"], model_path=identity["model_path"])
        while batches < max_batches and time.monotonic() < stop_submit:
            with lock:
                room = max_workers - len(in_flight)
            for _ in range(room):
                if time.monotonic() >= stop_submit:
                    break
                try:
                    assignment = scheduler.next_assignment()
                except BarrierActive:
                    time.sleep(1.0)
                    break
                except (ValueError, RuntimeError) as error:
                    _emit(metrics_path, "scheduler_empty", error=str(error))
                    break
                try:
                    double_buffer.bind_trajectory(assignment.trajectory_id)
                except (ValueError, RuntimeError) as error:
                    scheduler.requeue(assignment)
                    _emit(metrics_path, "adapter_bind_failed", error=str(error))
                    break
                try:
                    future = exec_manager.submit(assignment, rollout_engine)
                except (ValueError, RuntimeError) as error:
                    try:
                        double_buffer.release_trajectory(assignment.trajectory_id)
                    except (KeyError, ValueError):
                        pass
                    scheduler.requeue(assignment)
                    _emit(metrics_path, "dispatch_failed", error=str(error))
                    break
                with lock:
                    in_flight[assignment.trajectory_id] = (future, assignment)
                future.add_done_callback(
                    lambda fut, tid=assignment.trajectory_id, asg=assignment: (
                        settle_done(fut, tid, asg), scheduler.complete(asg)))
            if not in_flight:
                time.sleep(1.0)
                continue
            concurrent.futures.wait([f for f, _ in in_flight.values()],
                                    timeout=5.0, return_when=concurrent.futures.FIRST_COMPLETED)
            rollout_now = double_buffer.rollout
            if rollout_now is not None and batches < max_batches:
                try:
                    if do_update(f"upd-{update_index:04d}"):
                        batches += 1
                        update_index += 1
                except (ValueError, RuntimeError) as error:
                    _emit(metrics_path, "spo_update_failed", error=str(error))
        with exec_manager.paused():
            if in_flight:
                concurrent.futures.wait([f for f, _ in in_flight.values()])
            final_done = False
            if batches < max_batches and time.monotonic() < deadline:
                try:
                    rollout_now = double_buffer.rollout
                    if rollout_now is not None and do_update(f"upd-{update_index:04d}"):
                        batches += 1
                        final_done = True
                except (ValueError, RuntimeError) as error:
                    _emit(metrics_path, "spo_update_failed", error=str(error))
            _emit(metrics_path, "spo_run_stopped", batches=batches, final_update=final_done,
                  buffer=ledger.buffer_state())
        return 0 if batches >= max_batches else 75
    finally:
        try:
            endpoint.close()
        except (RuntimeError, OSError, ValueError):
            pass
        ledger.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--model-path")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--backend-url")
    parser.add_argument("--submission-dir")
    parser.add_argument("--agent-uid-base", type=int, default=61000)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--additional-walltime-seconds", type=float, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if args.check_only:
        cfg = load_config(Path(args.config))
        resolved = args.model_path or cfg["policy"].get("model_path")
        if resolved is not None:
            try:
                identity = validate_model_dir(Path(resolved).expanduser(), cfg["policy"])
            except (ValueError, TypeError, OSError) as error:
                print(f"spo_config_invalid: {type(error).__name__}: {error}", file=sys.stderr)
                return 2
            print(f"spo_config_ok sections={sorted(cfg.keys())} "
                  f"model={identity['model_revision'][:12]} "
                  f"tokenizer={sorted(identity['tokenizer_files'])}")
        else:
            print(f"spo_config_ok sections={sorted(cfg.keys())} model=unresolved")
        backend_url = args.backend_url or cfg["inference"].get("base_url")
        print(f"inference engine={cfg['inference'].get('engine')} base_url={backend_url}")
        return 0
    for name in ("data_root",):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required for a training run")
    cfg = load_config(Path(args.config))
    if args.model_path is None and not cfg["policy"].get("model_path"):
        parser.error("--model-path is required for a training run")
    if args.backend_url is None and not cfg["inference"].get("base_url"):
        parser.error("--backend-url is required for a training run")
    try:
        return run_spo(args)
    except (ValueError, RuntimeError, TypeError, OSError) as error:
        print(f"spo_run_failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
