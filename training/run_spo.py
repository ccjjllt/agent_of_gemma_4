"""Single-stream SPO training launcher (the complete CPU/GPU orchestrator).

Wiring (one process, colocated 2xA800):
  TaskCatalog -> SPOSampler -> RolloutScheduler -> ExecutionManager
  (OfficialWorker + vLLM OpenAI gateway + RequestManager) -> RolloutEngine
  -> RewardEngine -> ValueTracker -> TrajectoryBuffer -> ledger.reserve_update
  -> GpuLease.training + SPOTrainer.optimize_batch -> AdapterExporter
  -> AdapterManager/DoubleBuffer publish -> ledger.commit_update
  -> barrier publish + endpoint version advance (vLLM serving resume).

Usage (server, authorized session only):
  python -m training.run_spo --config configs/spo_a800_2x80.toml \
    --data-root /root/gemma4/official/competition \
    --model-path /autodl-fs/data \
    --run-dir /root/autodl-tmp/gemma4/runs/spo-next \
    --backend-url http://127.0.0.1:8000

The vLLM OpenAI server itself is deployed separately; this launcher speaks the
native vLLM contract in training/rollout/vllm_backend.py
(OpenAI /v1/completions, dynamic LoRA, /pause + /resume, Level-2 sleep/wake).
Exit codes: 0 finished all batches, 75 budget exhausted / SIGTERM grace
expired or repeated (continue with --resume), 2 configuration/run failure.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import contextmanager
import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
import tomllib
from pathlib import Path
from typing import Any, Callable, Mapping

from .rollout.backend import SGLangBackend
from .rollout.vllm_backend import VLLMBackend


REQUIRED_SECTIONS = ("policy", "run", "inference", "spo", "optimizer", "rewards", "adapter")


def load_config(path: Path) -> dict:
    with open(path, "rb") as handle:
        cfg = tomllib.load(handle)
    if cfg.get("algorithm") != "single_stream_spo":
        raise ValueError("algorithm must be single_stream_spo")
    for section in REQUIRED_SECTIONS:
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"missing config section: {section}")
    for section in ("sft", "distillation", "opd", "teacher"):
        if section in cfg:
            raise ValueError(
                f"[{section}] is not implemented by this online-SPO launcher")
    policy = cfg["policy"]
    for name in ("lora_rank", "lora_alpha"):
        if type(policy.get(name)) is not int or policy[name] <= 0:
            raise ValueError(f"policy.{name} must be a positive integer")
    if float(policy.get("lora_dropout", 0.0)) != 0.0:
        raise ValueError("the current shared-prefix learner implements dropout=0")
    spo = cfg["spo"]
    if int(spo.get("target_action_tokens", 0)) <= 0 or int(spo.get("max_policy_lag", -1)) < 0:
        raise ValueError("invalid spo token budget or lag")
    # Old configs remain readable; this retired sampling gate no longer
    # discards valid negative trajectories or changes pending updates.
    spo.pop("require_positive_signal", None)
    if type(cfg["adapter"].get("allow_resident_pair", False)) is not bool:
        raise ValueError("adapter.allow_resident_pair must be bool")
    checkpoint_dir = cfg["run"].get("checkpoint_dir")
    if checkpoint_dir is not None:
        if not isinstance(checkpoint_dir, str) or not checkpoint_dir.strip():
            raise ValueError("run.checkpoint_dir must be a nonempty path string")
        resolved = Path(checkpoint_dir).expanduser()
        if not resolved.is_dir():
            raise ValueError(f"run.checkpoint_dir is not an existing directory: {resolved}")
        if os.access(resolved, os.W_OK) is not True:
            raise ValueError(f"run.checkpoint_dir is not writable: {resolved}")
    if type(cfg["run"].get("prune_finished_artifacts", False)) is not bool:
        raise ValueError("run.prune_finished_artifacts must be bool")
    init_adapter_dir = cfg["run"].get("init_adapter_dir")
    if init_adapter_dir is not None:
        if not isinstance(init_adapter_dir, str) or not init_adapter_dir.strip():
            raise ValueError("run.init_adapter_dir must be a nonempty path string")
    if type(cfg["run"].get("in_run_prune", False)) is not bool:
        raise ValueError("run.in_run_prune must be bool")
    keep_versions = cfg["run"].get("keep_adapter_versions", [])
    if not isinstance(keep_versions, list) \
            or any(type(value) is not int or value < 0 for value in keep_versions):
        raise ValueError(
            "run.keep_adapter_versions must be a list of non-negative integers")
    if len(set(keep_versions)) != len(keep_versions):
        raise ValueError("run.keep_adapter_versions must not repeat a version")
    keep_recent = cfg["run"].get("keep_recent_adapters", 1)
    if type(keep_recent) is not int or keep_recent < 0:
        raise ValueError("run.keep_recent_adapters must be a non-negative integer")
    from .reward.progress import SUPPORTED_OBJECTIVE_VERSIONS
    if cfg["rewards"].get("objective_version") not in SUPPORTED_OBJECTIVE_VERSIONS:
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


def _make_inference_backend(cfg: Mapping[str, Any], backend_url: str):
    """Construct the configured engine; no silent cross-engine fallback."""
    inference = cfg["inference"]
    engine = str(inference.get("engine", "vllm")).lower()
    if engine == "vllm":
        served = inference.get("served_model_name") or cfg["policy"].get("model_name")
        return VLLMBackend(
            backend_url,
            expected_base_model=served,
            expected_lora_rank=cfg["policy"]["lora_rank"],
            max_resident_adapters=int(inference.get("max_loras", 2)),
            lora_name_prefix=str(inference.get("lora_name_prefix", "spo")),
            require_sleep_mode=bool(inference.get("require_sleep_mode", True)),
            require_runtime_lora=bool(inference.get("require_runtime_lora", True)),
            sleep_level=int(inference.get("sleep_level", 2)),
            enable_prefix_cache=bool(inference.get("enable_prefix_cache", False)),
            control_timeout=float(inference.get("control_timeout_seconds", 600)),
        )
    if engine == "sglang":
        return SGLangBackend(backend_url)
    raise ValueError(f"unsupported inference engine: {engine!r}")


def _active_update_id(state: Any) -> Any:
    """Read the active update id from current and older checkpoint layouts."""
    if not isinstance(state, Mapping):
        return None
    active = state.get("active_batch")
    if not isinstance(active, Mapping):
        return None
    value = active.get("update_id")
    if value is None:
        update = active.get("update")
        if isinstance(update, Mapping):
            value = update.get("update_id")
    return value


def _checkpoint_update_cursor(state: Any, update_id: str) -> int | None:
    """Return the batch cursor only when the checkpoint owns ``update_id``.

    ``None`` means the checkpoint is the committed base or another update's
    state, or the layout is unknown. Every destructive rollback treats that as
    "not proven pre-step" and refuses to touch the checkpoint.
    """
    if not isinstance(state, Mapping) or _active_update_id(state) != update_id:
        return None
    active = state.get("active_batch")
    if not isinstance(active, Mapping):
        return None
    update = active.get("update")
    if not isinstance(update, Mapping):
        return None
    cursor = update.get("optimizer_cursor")
    if type(cursor) is not int or cursor not in (0, 1):
        return None
    return cursor


def _base_state_from_prestep_checkpoint(
    state: Mapping[str, Any], update_id: str
) -> dict[str, Any]:
    """Strip a pre-step active batch and keep the committed training tensors.

    A pre-step snapshot is the state *before* this update's AdamW: adapter,
    AdamW moments, scheduler and RNG all belong to the last committed update.
    Deleting it would reset training to step 0 while the ledger still reports
    the committed step count.
    """
    if _checkpoint_update_cursor(state, update_id) != 0:
        raise RuntimeError(
            f"checkpoint for {update_id} is not a pre-step (cursor 0) snapshot")
    active = state["active_batch"]
    offset = active.get("offset")
    completed = state.get("completed_optimizer_steps")
    if type(offset) is not int or offset < 0 or completed != offset:
        raise RuntimeError(
            "pre-step checkpoint offset and completed optimizer steps disagree")
    base = dict(state)
    base["active_batch"] = None
    base["checkpoint_phase"] = "discarded_unoptimized_batch"
    return base


def validate_pending_transaction(record: Any, checkpoint_state: Any,
                                 transaction: Any) -> None:
    """Prevent replay of a claimed optimizer step without its saved state.

    Recovery does not reconsider the batch's reward or positive-signal status.
    """
    if record.phase not in ("RESERVED", "LEARNING") or record.optimizer_cursor != 0:
        return None
    cursor = _checkpoint_update_cursor(checkpoint_state, record.update_id)
    if transaction is not None and transaction.get("update_id") == record.update_id:
        tx_cursor = transaction.get("optimizer_cursor")
        if type(tx_cursor) is not int or tx_cursor not in (0, 1):
            raise RuntimeError(
                f"update {record.update_id} has an invalid transaction cursor")
        if tx_cursor == 1:
            if cursor != 1:
                raise RuntimeError(
                    f"update {record.update_id} has a post-step transaction but "
                    "no matching cursor-1 checkpoint")
            return None


def discard_unoptimized_update(
    ledger: Any, update_id: str, reason: str, *,
    checkpoint_state: Mapping[str, Any] | None = None,
    transaction: Mapping[str, Any] | None = None,
    rebase_checkpoint: Any = None,
    drop_transaction: Any = None,
) -> dict[str, Any] | None:
    """Roll back an update that provably never ran AdamW.

    Every source is validated before the destructive rewrite, so a transaction
    or ledger mismatch cannot leave a rebased checkpoint behind. Returns the
    rebased base snapshot when the full checkpoint pointed at this update, so
    the caller can restore it into a fresh trainer. Otherwise the on-disk
    checkpoint already belongs to an earlier committed update and is left
    byte-for-byte untouched. No training state is ever deleted here.
    """
    base_state = None
    checkpoint_completed = 0
    cursor = None
    # Validate every source before the destructive rewrite. A transaction or
    # ledger mismatch must not leave a rebased checkpoint behind.
    batch = ledger.get_update(update_id)
    if batch.phase not in ("RESERVED", "LEARNING") or batch.optimizer_cursor != 0:
        raise RuntimeError(
            "only an unoptimized RESERVED/LEARNING update can be discarded")
    if checkpoint_state is not None:
        if not isinstance(checkpoint_state, Mapping):
            raise TypeError("checkpoint state must be a mapping")
        checkpoint_completed = checkpoint_state.get("completed_optimizer_steps")
        if type(checkpoint_completed) is not int or checkpoint_completed < 0:
            raise RuntimeError("checkpoint has an invalid completed optimizer count")
        cursor = _checkpoint_update_cursor(checkpoint_state, update_id)
        if cursor is None and _active_update_id(checkpoint_state) == update_id:
            # The checkpoint claims this update but its cursor cannot be read.
            # A destructive rollback must not guess that AdamW never ran.
            raise RuntimeError(
                f"refusing to discard {update_id}: its checkpoint cursor is unknown")
        if cursor == 1:
            raise RuntimeError(
                f"refusing to discard {update_id}: its optimizer step is durable")
    drop_transaction_pending = False
    if transaction is not None and transaction.get("update_id") == update_id:
        if transaction.get("optimizer_cursor") != 0:
            raise RuntimeError(
                f"refusing to discard {update_id}: its transaction records a post-step state")
        tx_completed = transaction.get("completed_optimizer_steps")
        if type(tx_completed) is not int or tx_completed < 0:
            raise RuntimeError("transaction has an invalid completed optimizer count")
        if tx_completed != checkpoint_completed:
            raise RuntimeError(
                "transaction and checkpoint disagree on completed optimizer steps")
        if drop_transaction is None:
            raise RuntimeError("a transaction cleanup callback is required")
        drop_transaction_pending = True
    if cursor == 0:
        base_state = _base_state_from_prestep_checkpoint(checkpoint_state, update_id)
        if rebase_checkpoint is None:
            raise RuntimeError("a rebase callback is required to preserve the base")
        rebase_checkpoint(base_state)
    if drop_transaction_pending:
        drop_transaction()
    ledger.discard_unoptimized_update(update_id, reason)
    return base_state


INITIAL_ADAPTER_RECORD = "initial_adapter.json"
INITIAL_ADAPTER_SCHEMA = "gemma.spo_initial_adapter.v1"


def _read_initial_adapter(run_dir: Path) -> dict[str, Any] | None:
    """Read the imported-base provenance of a continuation run, if any."""
    path = Path(run_dir) / INITIAL_ADAPTER_RECORD
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"initial adapter record is unreadable: {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != INITIAL_ADAPTER_SCHEMA:
        raise RuntimeError(f"initial adapter record has an invalid schema: {path}")
    version = document.get("version")
    sha = document.get("adapter_sha256")
    if type(version) is not int or version < 0 \
            or not isinstance(sha, str) or len(sha) != 64:
        raise RuntimeError(f"initial adapter record has invalid version/sha: {path}")
    return document


def _write_initial_adapter(run_dir: Path, record: Mapping[str, Any]) -> None:
    path = Path(run_dir) / INITIAL_ADAPTER_RECORD
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(dict(record), output, ensure_ascii=False, sort_keys=True, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _checkpoint_path(run_dir: Path, run_cfg: Mapping[str, Any]) -> Path:
    """Trainer checkpoint file; may live outside the run dir.

    The checkpoint is rewritten atomically (same-size temporary + rename), so
    the run dir needs 2x its size free. When the data disk cannot hold that, a
    config may point ``[run] checkpoint_dir`` at a larger volume.
    """
    directory = run_cfg.get("checkpoint_dir")
    base = Path(str(directory)).expanduser() if directory else Path(run_dir)
    return base / "spo_trainer_state.pt"


def _verify_resume_consistency(run_dir: Path, ledger: Any, *,
                               checkpoint_path: Path | None = None,
                               initial_version: int = 0) -> dict[str, Any]:
    """Fail closed when a --resume run cannot continue the previous run.

    Checks the serving registry, the trainer checkpoint and the ledger
    against each other before any GPU work:
    - serving v>0 requires a trainer checkpoint (else LoRA restarts under a
      trained policy);
    - checkpoint completed-steps must equal CONSUMED updates, or exceed them
      by exactly one when an update is still pending;
    - serving version equals CONSUMED updates, except one ACTIVATED pending
      update which legitimately serves consumed+1. That case must prove the
      serving adapter is the pending update's own product: registered
      manifest version/parent/SHA, matching checkpoint update id, and exactly
      one optimizer step ahead.
    Returns the verified resume summary.
    """
    import torch as _torch  # noqa: F401 (resume requires torch regardless)
    from .adapter.manager import AdapterManager
    if type(initial_version) is not int or initial_version < 0:
        raise ValueError("initial_version must be a non-negative integer")
    adapters = AdapterManager(run_dir / "adapters")
    active = adapters.active
    state_path = Path(checkpoint_path) if checkpoint_path else run_dir / "spo_trainer_state.pt"
    if active is not None and active.version > 0 and not state_path.is_file():
        imported_zero_step = (initial_version > 0
                              and active.version == initial_version
                              and not ledger.list_updates())
        if not imported_zero_step:
            raise RuntimeError(
                f"serving adapter is v{active.version} but no trainer checkpoint "
                "exists; resuming would restart LoRA from zero under a trained policy")
    completed = 0
    state: Any = None
    if state_path.is_file():
        try:
            state = _torch.load(state_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"trainer checkpoint is unreadable: {exc}") from exc
        if not isinstance(state, Mapping) or state.get("algorithm") != "single_stream_spo":
            raise RuntimeError("trainer checkpoint is not a single_stream_spo state")
        from .spo.trainer import TRAINER_STATE_VERSION
        if state.get("schema_version") != TRAINER_STATE_VERSION:
            raise RuntimeError(
                f"trainer checkpoint schema {state.get('schema_version')} is not "
                f"the current schema {TRAINER_STATE_VERSION}; start a new run")
        completed = state.get("completed_optimizer_steps", 0)
        if type(completed) is not int or completed < 0:
            raise RuntimeError("trainer checkpoint has an invalid optimizer-step count")
        if active is None and completed > 0:
            raise RuntimeError(
                "trainer checkpoint claims completed steps but no adapter is serving")
        if active is not None and active.version == initial_version and completed > 0:
            # Optimizer step finished but publish never completed (crash/OOM
            # between checkpoint write and ACTIVATED): serving is legitimately
            # still genesis.  Allow only a single post-step, pre-serve pending
            # update whose id matches the checkpoint; everything else still
            # fails closed.
            post_step = [b for b in ledger.list_updates(
                ("OPTIMIZED", "EXPORTED", "REGISTERED"))]
            if (len(post_step) != 1 or _active_update_id(state) != post_step[0].update_id
                    or completed != len(ledger.list_updates(("CONSUMED",))) + 1):
                raise RuntimeError(
                    "trainer checkpoint claims completed steps but serving is still the initial adapter")
    consumed = len(ledger.list_updates(("CONSUMED",)))
    pending = ledger.list_updates(
        ("RESERVED", "LEARNING", "OPTIMIZED", "EXPORTED", "REGISTERED", "ACTIVATED"))
    activated = [batch for batch in pending if batch.phase == "ACTIVATED"]
    if len([batch for batch in pending if batch.phase != "ACTIVATED"]) > 1:
        raise RuntimeError(
            "more than one non-ACTIVATED pending update; refusing ambiguous recovery")
    if active is None:
        if consumed > 0 or pending:
            raise RuntimeError(
                "ledger has consumed/pending updates but no adapter is serving")
    elif len(activated) > 1:
        raise RuntimeError(
            f"{len(activated)} ACTIVATED updates are pending; at most one serving "
            "advance may be uncommitted")
    elif len(activated) == 1:
        batch = activated[0]
        expected_version = initial_version + consumed + 1
        if active.version != expected_version:
            raise RuntimeError(
                f"serving adapter v{active.version} is not the product of pending "
                f"{batch.update_id} (expected v{expected_version})")
        try:
            registered = adapters.get(active.version)
        except (KeyError, ValueError) as exc:
            raise RuntimeError(f"serving adapter v{active.version} is not registered") from exc
        if registered.adapter_sha256 != active.adapter_sha256:
            raise RuntimeError("serving registry entry does not match the active adapter")
        if registered.parent_version != batch.policy_before.version:
            raise RuntimeError(
                f"serving v{active.version} is not the child of update "
                f"{batch.update_id}'s base v{batch.policy_before.version}")
        if state is not None:
            checkpoint_update = _active_update_id(state)
            if checkpoint_update != batch.update_id:
                raise RuntimeError(
                    f"trainer checkpoint belongs to {checkpoint_update}, not pending "
                    f"{batch.update_id}")
        if completed != consumed + 1:
            raise RuntimeError(
                f"ACTIVATED update {batch.update_id} requires exactly one finished "
                f"optimizer step, not {completed} completed vs {consumed} consumed")
    elif consumed > 0 and active.version != initial_version + consumed:
        raise RuntimeError(
            f"serving adapter v{active.version} disagrees with {consumed} consumed "
            f"updates after initial v{initial_version}")
    elif consumed == 0 and active.version != initial_version:
        raise RuntimeError(
            f"serving adapter v{active.version} exists with no consumed update "
            f"(expected initial v{initial_version})")
    if completed < consumed or completed > consumed + (1 if pending else 0):
        raise RuntimeError(
            f"trainer checkpoint ({completed} steps) disagrees with ledger "
            f"({consumed} consumed, {len(pending)} pending)")
    return {"serving_version": active.version if active is not None else None,
            "completed_optimizer_steps": completed, "consumed_updates": consumed,
            "pending_updates": len(pending),
            "activated_pending": [batch.update_id for batch in activated]}


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


def _read_budget(path: Path, allowance_seconds: float, *, resume: bool) -> dict[str, float]:
    """Read the durable budget; a resume never re-grants a fresh allowance."""
    default = {"allowance_seconds": allowance_seconds, "consumed_seconds": 0.0}
    try:
        raw = path.read_text()
    except FileNotFoundError as exc:
        if resume:
            raise RuntimeError(
                "resume requires the persisted budget record; refusing to "
                "re-grant a fresh allowance") from exc
        return default
    except OSError as exc:
        raise RuntimeError(f"budget record unreadable, refusing to run blind: {exc}") from exc
    try:
        document = json.loads(raw)
        allowance = float(document["allowance_seconds"])
        consumed = float(document.get("consumed_seconds", 0.0))
        if (not math.isfinite(allowance) or allowance < 0
                or not math.isfinite(consumed) or consumed < 0):
            raise ValueError("invalid budget record")
        return {"allowance_seconds": allowance, "consumed_seconds": consumed}
    except (ValueError, KeyError, TypeError) as exc:
        if resume:
            raise RuntimeError(
                "budget record is corrupt; refusing to reset consumed GPU "
                "time (restore budget.json manually)") from exc
        return default


def _write_budget(path: Path, allowance_seconds: float, consumed_seconds: float) -> None:
    """Atomically write one budget checkpoint (called only by _BudgetWriter).

    The writer serializes calls, so a slower older caller can no longer replace
    a newer record. A unique temporary name per write keeps the bytes private;
    os.replace makes the visible file always one complete JSON document.
    """
    record = {"allowance_seconds": allowance_seconds,
              "consumed_seconds": max(0.0, float(consumed_seconds)),
              "remaining_seconds": max(0.0, allowance_seconds - float(consumed_seconds)),
              "updated_unix": time.time()}
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.monotonic_ns()}")
    temporary.write_text(json.dumps(record) + "\n")
    os.replace(temporary, path)


class _BudgetWriter:
    """Own every budget.json write on one thread, keeping consumption monotonic.

    Callers submit the current consumed value; the writer persists
    max(last_written, submitted), so a delayed older submission can never roll
    the durable record back. Normal ``submit`` waits briefly on the writer
    lock (held only for micro-ops, never across disk IO); the ``from_signal``
    variant never takes the lock and is safe from the SIGTERM handler;
    ``flush`` waits (bounded) until every submitted value is on disk.
    """

    def __init__(self, path: Path, allowance_seconds: float) -> None:
        self._path = path
        self._allowance = float(allowance_seconds)
        self._condition = threading.Condition(threading.RLock())
        self._pending: float | None = None
        self._inflight: float | None = None
        # Signal-handler fallback: one atomic slot (seq, value). The tuple
        # reference is replaced in a single assignment and read in a single
        # load, so a reader can never pair a stale value with a newer seq.
        self._signal_slot: tuple[int, float | None] = (0, None)
        self._signal_seen = 0
        self._last_written: float | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="spo-budget-writer", daemon=True)
        self._thread.start()

    @staticmethod
    def _higher(current: float | None, value: float) -> float:
        return value if current is None else max(current, value)

    def _store_signal(self, value: float) -> None:
        """Record a value without taking the writer lock (never blocks)."""
        seq, current = self._signal_slot  # single atomic load
        if current is not None and value <= current:
            return  # slot already covers this submission
        merged = value if current is None else max(current, value)
        self._signal_slot = (seq + 1, merged)  # single atomic store

    def submit(self, consumed_seconds: float, *, from_signal: bool = False) -> None:
        """Offer a value; the writer keeps the largest one.

        Normal submits wait briefly on the writer lock (held only for
        micro-ops; disk IO happens outside it) and merge into ``_pending``.
        The ``from_signal`` variant never takes the lock and stays the sole
        producer of the signal slot, so a handler firing on the main thread
        while it is blocked in a normal submit cannot deadlock with it.
        """
        value = max(0.0, float(consumed_seconds))
        if from_signal:
            # Signal-handler context only: single atomic tuple store, no lock.
            self._store_signal(value)
            return
        # Cheap pre-check so a post-close() submit usually returns without
        # touching the lock; rechecked under the lock (close() joins the
        # writer outside the lock, so the acquire below always terminates).
        if self._closed:
            return
        with self._condition:
            if self._closed:
                return
            self._pending = self._higher(self._pending, value)
            self._condition.notify()

    def _drained(self) -> bool:
        return (self._closed
                or (self._pending is None and self._inflight is None
                    and self._signal_seen == self._signal_slot[0]))

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until all submitted values are on disk; never unbounded."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while not self._drained():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(min(0.02, remaining))
            return True

    def close(self, final_seconds: float | None = None,
              timeout: float = 5.0) -> bool:
        """Submit the final value, flush it, then stop the writer thread."""
        if final_seconds is not None:
            self.submit(final_seconds)
        written = self.flush(timeout)
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, timeout))
        return written and not self._thread.is_alive()

    def _run(self) -> None:
        while True:
            with self._condition:
                while (not self._closed and self._pending is None
                       and self._signal_seen == self._signal_slot[0]):
                    # Poll: a signal-slot store cannot notify (it never holds
                    # this lock), so wake periodically to observe it.
                    self._condition.wait(timeout=0.05)
                if self._closed:
                    return
                value = self._pending
                self._pending = None
                slot = self._signal_slot  # single atomic load
                seq_read, signal_value = slot
                if self._signal_seen != seq_read:
                    # Acknowledge exactly the generation that was read; a
                    # concurrent store leaves a newer seq pending for the next
                    # pass instead of being dropped with a stale value.
                    self._signal_seen = seq_read
                    if signal_value is not None:
                        value = self._higher(value, signal_value)
                if value is None:
                    continue
                if self._last_written is not None:
                    value = max(value, self._last_written)
                self._inflight = value
            try:
                _write_budget(self._path, self._allowance, value)
            except OSError:
                # Keep the value pending and retry; never lower last_written.
                with self._condition:
                    self._inflight = None
                    self._pending = self._higher(self._pending, value)
                    self._condition.wait(0.5)
                continue
            with self._condition:
                self._inflight = None
                self._last_written = value
                self._condition.notify_all()


def _drain_update_window(wait_for_workers: Callable[[float], int], metrics_path: Path, *,
                         update_id: str, phase: str, deadline: float,
                         update_min_seconds: float) -> bool:
    """Drain in-flight trajectories without spending the step's budget.

    True only when every worker settled before deadline - update_min_seconds.
    Otherwise the update is emitted as deferred and the caller must leave it
    RESERVED: an optimizer step started now would be killed mid-transaction by
    the external walltime cap (the 2026-11-16 failure).
    """
    remaining = max(0.0, deadline - time.monotonic())
    timeout = max(0.0, remaining - update_min_seconds)
    started = time.monotonic()
    pending = int(wait_for_workers(timeout))
    waited = time.monotonic() - started
    left = max(0.0, deadline - time.monotonic())
    # Settling early is not enough: a step still needs update_min_seconds of
    # walltime after the drain, whether or not anything was in flight.
    if pending <= 0 and left >= update_min_seconds:
        return True
    _emit(metrics_path, "spo_update_deferred", update_id=update_id, phase=phase,
          waited_seconds=round(waited, 3),
          remaining_seconds=round(left, 3),
          pending_workers=pending,
          update_min_seconds=round(update_min_seconds, 3))
    return False


def _cancel_stop_watchdog(stop_state: dict[str, object]) -> None:
    """Disarm the update-time SIGTERM grace watchdog, if armed."""
    watchdog = stop_state.get("watchdog")
    if isinstance(watchdog, threading.Timer):
        watchdog.cancel()
        stop_state["watchdog"] = None


def _finish_update_window(stop_state: dict[str, object]) -> None:
    """Leave in_update and disarm the watchdog as one TERM-atomic step.

    SIGTERM is blocked in this thread for the two stores (where supported), so
    the handler cannot arm a fresh watchdog after this step's watchdog was
    cancelled, nor observe in_update=True after it was cleared.
    """
    previous: set[int] | None = None
    try:
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    except (AttributeError, OSError, ValueError):
        previous = None
    try:
        stop_state["in_update"] = False
        _cancel_stop_watchdog(stop_state)
    finally:
        if previous is not None:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous)
            except (AttributeError, OSError, ValueError):
                pass


def _make_sigterm_handler(persist: Callable[[], object],
                          stop_state: dict[str, object], *,
                          grace_seconds: float = 60.0,
                          flush: Callable[[], object] | None = None,
                          exit_code: int = 75) -> Callable[[int, object], None]:
    """Persist the budget on SIGTERM, then unwind or enforce a bounded stop.

    The first idle TERM persists and raises KeyboardInterrupt at a safe
    boundary. The first TERM during an optimizer step persists, returns, and
    arms a watchdog that hard-exits with exit_code after grace_seconds so a
    running step cannot overrun the stop without a bound. A repeated TERM is
    an explicit hard stop: the budget is flushed and the process exits at once,
    even mid-step (a durable budget record matters more than the transaction).
    """
    def hard_exit(reason: str) -> None:
        try:
            if flush is not None:
                flush()
        finally:
            sys.stderr.write(f"spo_sigterm: {reason}; exit={exit_code}\n")
            sys.stderr.flush()
            os._exit(exit_code)

    def handler(signum: int, frame: object) -> None:
        stop_state["signals"] = int(stop_state.get("signals", 0)) + 1
        already_requested = bool(stop_state["requested"])
        stop_state["requested"] = True
        try:
            persist()
        except (OSError, RuntimeError, ValueError):
            # At minimum the periodic heartbeat has the last realistic value.
            pass
        if already_requested:
            hard_exit(f"repeated signal {signum} during an update or cleanup")
        if stop_state.get("in_update"):
            if not isinstance(stop_state.get("watchdog"), threading.Timer):
                watchdog = threading.Timer(
                    grace_seconds,
                    lambda: hard_exit(
                        f"optimizer step still running {grace_seconds:g}s "
                        f"after signal {signum}"))
                watchdog.daemon = True
                stop_state["watchdog"] = watchdog
                watchdog.start()
            # The step keeps running; give the first TERM's value a bounded
            # chance to reach disk without making the watchdog unbounded too.
            try:
                if flush is not None:
                    flush()
            except (OSError, RuntimeError, ValueError):
                pass
            return
        # Idle boundary: persist synchronously before unwinding.
        try:
            if flush is not None:
                flush()
        except (OSError, RuntimeError, ValueError):
            pass
        raise KeyboardInterrupt(f"received signal {signum}")

    return handler


@contextmanager
def _training_cuda_memory(torch: Any, metrics_path: Path, update_id: str):
    """Measure the learner allocator from model load through adapter export."""
    started = time.time()
    devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            row: dict[str, Any] = {"device": index, "peak_reset": False}
            try:
                torch.cuda.synchronize(index)
                torch.cuda.reset_peak_memory_stats(index)
                row["peak_reset"] = True
            except Exception as error:
                row["reset_error"] = f"{type(error).__name__}: {error}"
            devices.append(row)
    _emit(metrics_path, "spo_training_memory_started", update_id=update_id,
          devices=[row["device"] for row in devices], window_started_unix=started)
    try:
        yield
    finally:
        failure = sys.exc_info()[0]
        for row in devices:
            index = row["device"]
            try:
                torch.cuda.synchronize(index)
            except Exception as error:
                row["synchronize_error"] = f"{type(error).__name__}: {error}"
            for name in ("memory_allocated", "memory_reserved",
                         "max_memory_allocated", "max_memory_reserved"):
                if name.startswith("max_") and not row["peak_reset"]:
                    row[name + "_bytes"] = None
                    continue
                try:
                    row[name + "_bytes"] = int(getattr(torch.cuda, name)(index))
                except Exception as error:
                    row[name + "_error"] = f"{type(error).__name__}: {error}"
        _emit(metrics_path, "spo_training_cuda_memory", update_id=update_id,
              window_started_unix=started, elapsed_seconds=time.time() - started,
              outcome="exception" if failure is not None else "completed",
              error_type=failure.__name__ if failure is not None else None,
              scope="learner model load, restore, optimization, KL, checkpoint and adapter export; before learner release",
              measurement="per-device PyTorch allocator bytes in learner process; excludes vLLM and non-PyTorch allocations",
              devices=devices)


def run_spo(args: argparse.Namespace) -> int:
    import torch  # noqa: F401  (GPU-only past this point, except --check-only)
    from .adapter.double_buffer import DoubleBuffer
    from .adapter.exporter import AdapterExporter
    from .adapter.manager import AdapterManager
    from .checkpoint_io import CheckpointWriter, read_transaction, write_transaction
    from .data.curriculum import TaskCatalog
    from .endpoint import PolicyEndpoint
    from .environment.execution_manager import AttemptSequence, ExecutionManager
    from .environment.sandbox import UidAllocator
    from .reward.baseline_probe import EmptyPatchProbeRunner
    from .environment.worker import OfficialWorker, OfficialWorkerConfig
    from .environment.workspace_cache import WorkspaceCache
    from .policy import Policy
    from .reward.baseline import EmptyPatchBaselineStore
    from .reward.engine import RewardEngine
    from .rollout.backend import GpuLease
    from .rollout.engine import RolloutEngine
    from .rollout.request_manager import RequestManager
    from .rollout.scheduler import BarrierActive, NoDispatchableTask, RolloutScheduler
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
    init_adapter_dir = run.get("init_adapter_dir")
    in_run_prune = bool(run.get("in_run_prune", False))
    keep_adapter_versions = tuple(run.get("keep_adapter_versions", []))
    keep_recent_adapters = int(run.get("keep_recent_adapters", 1))
    walltime = float(run.get("walltime_seconds", 7200)) + float(args.additional_walltime_seconds or 0)
    reserve = float(run.get("update_reserve_seconds", 900))
    if not walltime > reserve > 0:
        raise ValueError("walltime must exceed the positive update reserve")
    # [run] update_min_seconds (default 1800): walltime that must remain after
    # the pre-update drain before a step may start. 1800 s is the conservative
    # 2xA800 31B update budget (learner load + one SPO step + export/commit).
    # update_reserve_seconds still only decides when new rollouts stop; this is
    # an additional admission bound, not a replacement for it. A smaller value
    # starts steps that the external 60-minute cap can kill mid-transaction.
    update_min = float(run.get("update_min_seconds", 1800))
    if not math.isfinite(update_min) or update_min < 0:
        raise ValueError("run.update_min_seconds must be a finite non-negative number")
    # [run] sigterm_grace_seconds (default 60): after the first SIGTERM lands
    # while an optimizer step is running, a watchdog hard-exits with code 75
    # once this many seconds pass and the step is still running (0 exits at
    # once). It is not a step admission gate like update_min_seconds: the step
    # may still start, but it cannot outlive the stop without a bound. A second
    # SIGTERM always hard-exits immediately after flushing the budget. 60 s is
    # one minute of overrun inside the external supervisor's kill window while
    # still short against a ~30-minute update.
    sigterm_grace = float(run.get("sigterm_grace_seconds", 60))
    if not math.isfinite(sigterm_grace) or sigterm_grace < 0:
        raise ValueError(
            "run.sigterm_grace_seconds must be a finite non-negative number")
    max_batches = int(args.max_batches or run.get("max_batches", 32))
    max_workers = int(run.get("max_cpu_workers", 8))
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    budget_path = run_dir / "budget.json"

    def read_budget() -> dict[str, float]:
        return _read_budget(budget_path, walltime, resume=bool(args.resume))

    # Resume never renews the allowance: only explicit additional seconds
    # extend the persisted remainder. On resume a missing/corrupt record is a
    # hard stop, never a fresh grant (see read_budget).
    saved_budget = read_budget() if args.resume else {
        "allowance_seconds": walltime, "consumed_seconds": 0.0}
    allowance = saved_budget["allowance_seconds"] + float(args.additional_walltime_seconds or 0)
    consumed = saved_budget["consumed_seconds"]
    # One writer thread owns budget.json: concurrent heartbeat, signal and main
    # path values can no longer race, and the file only moves forward.
    budget_writer = _BudgetWriter(budget_path, allowance)

    def persist_budget(consumed_seconds: float) -> None:
        budget_writer.submit(consumed_seconds)

    # The allowance covers the whole GPU session, including genesis load,
    # backend handshake and trainer construction: the clock starts here, not
    # at the first rollout.
    start = time.monotonic()
    stop_state: dict[str, object] = {
        "requested": False, "in_update": False, "signals": 0, "watchdog": None}
    budget_stop = threading.Event()

    def persist_consumed(now: float | None = None) -> float:
        """Persist wall-clock consumption, not only finished optimizer steps."""
        stamp = time.monotonic() if now is None else now
        value = max(0.0, consumed + (stamp - start))
        persist_budget(value)
        return value

    def budget_heartbeat() -> None:
        # A 30 s tick keeps budget.json realistic through a long drain or a
        # long optimizer step, where the main loop cannot reach a checkpoint.
        while not budget_stop.wait(30.0):
            try:
                persist_consumed()
            except (OSError, RuntimeError, ValueError):
                continue

    def persist_for_signal() -> float:
        """Signal-context submit: never waits on the writer's lock or disk."""
        value = max(0.0, consumed + (time.monotonic() - start))
        budget_writer.submit(value, from_signal=True)
        return value

    request_stop = _make_sigterm_handler(
        persist_for_signal, stop_state,
        grace_seconds=sigterm_grace,
        flush=lambda: budget_writer.flush(5.0))
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    heartbeat = threading.Thread(target=budget_heartbeat, name="spo-budget-heartbeat",
                                 daemon=True)
    heartbeat.start()
    # P0-2a: a reused run_dir without --resume would silently serve an old
    # adapter while training a fresh LoRA. Fail closed before touching GPUs.
    checkpoint_path = _checkpoint_path(run_dir, run)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _RUN_STATE_FILES = ("adapters/registry.json", "spo_stream.db",
                        "spo_trainer_tx.json", "budget.json", "judge_identity.json")
    if not args.resume and (checkpoint_path.exists()
                            or any((run_dir / name).exists() for name in _RUN_STATE_FILES)):
        raise RuntimeError(
            "run_dir already holds training state; start fresh in an empty "
            "directory or continue explicitly with --resume")
    from .judge_identity import read_runtime_identity, bind_run_identity
    extra_wheels = (Path(cfg["runtime"]["extra_wheels"]).expanduser().resolve()
                    if cfg["runtime"].get("extra_wheels") else None)
    judge_environment = read_runtime_identity(
        cfg["runtime"].get("python", "/opt/gemma-runtime/bin/python"),
        data_root=data_root, submission_dir=args.submission_dir or (data_root / "sample_submission"),
        baseline_python=cfg["runtime"].get("sandbox_baseline_python", "/opt/gemma-sandbox-baseline/bin/python"),
        sandbox=cfg["runtime"].get("sandbox", "subprocess"),
        evaluation_overrides=dict(cfg["runtime"].get("evaluation", {}) or {}),
        extra_wheels=str(extra_wheels) if extra_wheels else None)
    # Reject legacy/changed judge identities before touching model records,
    # the existing ledger, the inference backend, or any GPU allocation.
    fingerprint = bind_run_identity(run_dir, judge_environment, resume=args.resume)
    identity = validate_model_dir(model_path, cfg["policy"])
    with open(run_dir / "model_identity.json", "w", encoding="utf-8") as handle:
        json.dump(identity, handle, ensure_ascii=False, indent=2, sort_keys=True)
    _emit(metrics_path, "model_identity", **identity)

    catalog = TaskCatalog(tasks_path, include_ids=cfg["data"].get("task_ids") or None,
                          exclude_ids=cfg["data"].get("exclude_task_ids") or ())
    _emit(metrics_path, "public_data_identity", path=str(catalog.source),
          sha256=catalog.source_sha256,
          scope="official public tasks; provenance separate from judge environment identity")
    catalog.bind_task_keys(build_task_keys(
        tasks_path, fingerprint, cfg["data"].get("task_ids") or [], cfg["data"].get("exclude_task_ids") or []))

    ledger = TrajectoryLedger(run_dir / "spo_stream.db")
    tracker = ValueTracker(
        ledger, prior_value=float(cfg["spo"].get("value_prior", 0.5)),
        prior_success=float(cfg["spo"].get("success_prior", 0.5)),
        prior_strength=float(cfg["spo"].get("prior_strength", 1.0)),
        kl_half_life=float(cfg["spo"].get("kl_half_life", 0.05)))
    sampler = SPOSampler(tracker, epsilon=float(cfg["spo"].get("sampling_epsilon", 0.05)))
    # Startup recovery first: drop dead-process dispatch markers and fold
    # crash-window trajectories before any new dispatch observes them.
    purged_markers = tracker.purge_orphan_markers()
    recovered = tracker.recover_incomplete()
    _emit(metrics_path, "spo_startup_recovery",
          orphan_markers_purged=purged_markers, **recovered)
    requests = RequestManager(run_dir / "requests")
    uid_allocator = UidAllocator(int(args.agent_uid_base or 61000), 1000000)
    attempt_seq = AttemptSequence(run_dir / "attempt_counter.json")
    baselines = EmptyPatchBaselineStore(
        run_dir / "baselines",
        runner=EmptyPatchProbeRunner(
            data_root=data_root,
            submission_dir=args.submission_dir or (data_root / "sample_submission"),
            runtime_python=cfg["runtime"].get("python", "/opt/gemma-runtime/bin/python"),
            sandbox_root=cfg["runtime"].get("sandbox_root", "/root/autodl-tmp/gemma4/sandboxes"),
            baseline_python=cfg["runtime"].get("sandbox_baseline_python", "/opt/gemma-sandbox-baseline/bin/python"),
            uid_allocator=uid_allocator,
            attempt_seq=attempt_seq.take,
            workspace_root=run_dir / "probe_workspaces",
            evaluation_overrides=dict(cfg["runtime"].get("evaluation", {}) or {}),
            extra_wheels=extra_wheels,
            emit=lambda n, **f: _emit(metrics_path, n, **f)))
    reward_engine = RewardEngine(
        baselines, objective_version=cfg["rewards"]["objective_version"],
        efficiency_weight=float(cfg["rewards"].get("success_efficiency_weight", 0.05)))
    trajectory_buffer = TrajectoryBuffer(
        ledger, target_action_tokens=int(cfg["spo"]["target_action_tokens"]),
        max_policy_lag=int(cfg["spo"]["max_policy_lag"]),
        advantage_epsilon=float(cfg["spo"].get("advantage_epsilon", 1.0e-6)))
    rollout_engine = RolloutEngine(requests, reward_engine, ledger, tracker, trajectory_buffer)

    backend_url = args.backend_url or cfg["inference"].get("base_url")
    if not backend_url:
        raise ValueError("inference backend URL is required (--backend-url)")
    backend = _make_inference_backend(cfg, backend_url)
    try:
        handshake = backend.start()
    except (ValueError, RuntimeError, OSError) as error:
        raise RuntimeError(
            f"inference backend handshake failed at {backend_url}: "
            f"{type(error).__name__}: {error}") from error
    lease = GpuLease(backend)
    adapters = AdapterManager(run_dir / "adapters")
    initial_record: dict[str, Any] | None = None
    if args.resume:
        initial_record = _read_initial_adapter(run_dir)
    elif init_adapter_dir:
        from .rollout.policy_version import AdapterManifest as _InitialManifest
        source = Path(init_adapter_dir).expanduser().resolve(strict=True)
        if not source.is_dir():
            raise ValueError(f"run.init_adapter_dir is not a directory: {source}")
        manifest = _InitialManifest.read(source / "manifest.json")
        if manifest.rank != int(cfg["policy"]["lora_rank"]) \
                or manifest.alpha != int(cfg["policy"]["lora_alpha"]) \
                or manifest.dropout != float(cfg["policy"].get("lora_dropout", 0.0)) \
                or manifest.base_revision != cfg["policy"].get("model_revision"):
            raise ValueError(
                "run.init_adapter_dir manifest does not match the learner rank/alpha/"
                "dropout or model revision")
        adapters.import_adapter(manifest, source, activate=True)
        initial_record = {
            "schema": INITIAL_ADAPTER_SCHEMA,
            "version": manifest.version,
            "adapter_sha256": manifest.adapter_sha256,
            "source_dir": str(source),
            "source_parent_version": manifest.parent_version,
            "source_optimizer_step": manifest.optimizer_step,
            "imported_unix": time.time(),
            "note": ("optimizer state, AdamW moments and LR schedule start fresh; "
                     "only the LoRA weights are inherited"),
        }
        _write_initial_adapter(run_dir, initial_record)
        identity["initial_adapter"] = initial_record
        with (run_dir / "model_identity.json").open("w", encoding="utf-8") as handle:
            json.dump(identity, handle, ensure_ascii=False, indent=2, sort_keys=True)
        _emit(metrics_path, "spo_initial_adapter_imported", **initial_record)
    initial_version = int(initial_record["version"]) if initial_record else 0
    try:
        initial_adapter_manifest = adapters.get(initial_version) if initial_record else None
    except KeyError:
        initial_adapter_manifest = None
    if isinstance(backend, VLLMBackend):
        # The runtime validator checks the PEFT target layout against the
        # actual Gemma 4 config before vLLM is allowed to load the adapter.
        backend.adapter_validator = lambda manifest, directory: adapters.audit_lora_compatibility(
            manifest, model_config=model_path / "config.json", source=directory)
    if args.resume:
        resume_summary = _verify_resume_consistency(
            run_dir, ledger, checkpoint_path=checkpoint_path,
            initial_version=initial_version)
        _emit(metrics_path, "spo_resume_verified", **resume_summary)

    policy_cfg = dict(cfg["policy"])
    policy_cfg["model_path"] = str(model_path)
    # The 31B learner must not occupy GPU at startup: vLLM holds the colocated
    # GPUs until the first training lease. Tokenizer-only serving works unloaded.
    policy_cfg["defer_learner_load"] = True
    policy_cfg.setdefault("gradient_checkpointing", True)
    policy_cfg.setdefault("expected_gpu_count", 2)
    policy = Policy(policy_cfg)
    exporter = AdapterExporter()
    if adapters.active is None:
        # Genesis export needs the learner weights: load them inside the first
        # training lease (vLLM memory is released), then unload before the
        # inference engine resumes below.
        with lease.training():
            load_started = time.monotonic()
            policy.ensure_loaded()
            _emit(metrics_path, "spo_model_loaded", phase="genesis",
                  seconds=round(time.monotonic() - load_started, 3))
            try:
                genesis_dir = run_dir / "adapters" / "staging-genesis"
                manifest = exporter.export(policy, genesis_dir, version=0, parent_version=None, optimizer_step=0)
                adapters.publish(manifest, genesis_dir, activate=True)
                binding = adapters.serving_binding(
                    manifest, prefix=str(cfg["inference"].get("lora_name_prefix", "spo")))
            finally:
                policy.release_memory()
        _emit(metrics_path, "adapter_genesis", version=0,
              lora_name=binding["lora_name"])
    else:
        _emit(metrics_path,
              "adapter_initialized" if initial_record and not args.resume
              else "adapter_resumed",
              version=adapters.active.version,
              initial_version=initial_version)
    double_buffer = DoubleBuffer(
        adapters, backend, gpu_lease=lease,
        allow_resident_pair=cfg["adapter"].get("allow_resident_pair", False))
    # The inference server starts empty: make the single valid rollout version
    # resident before the scheduler or any trajectory can observe it.
    genesis_rollout = double_buffer.rollout
    if genesis_rollout is None:
        raise RuntimeError("no rollout adapter is available after genesis/resume")
    with lease.inference():
        backend.load_adapter(genesis_rollout, adapters.path(genesis_rollout))
    _emit(metrics_path, "adapter_rollout_loaded", version=genesis_rollout.version,
          sha256=genesis_rollout.adapter_sha256)
    endpoint = PolicyEndpoint(
        policy, run_dir / "endpoint", int(cfg["runtime"].get("port", 0)),
        {"max_inflight": int(cfg["inference"].get("max_inflight", 16)),
         "max_batch_size": int(cfg["inference"].get("max_inflight", 16)),
         "generation_timeout_seconds": float(
             cfg["inference"].get("generation_timeout_seconds", 1800)),
         "temperature": float(cfg["sampling"].get("temperature", 1.0)),
         "top_p": float(cfg["sampling"].get("top_p", 1.0)),
         "top_k": int(cfg["sampling"].get("top_k", 0)),
         "max_new_tokens": int(cfg["sampling"].get("max_new_tokens", 16384))},
        serving_backend=double_buffer)
    endpoint.start()
    # Resume on the actual serving version, not genesis 0: after a restart the
    # registry may already hold v1+. Workers register against the live rollout;
    # a version-0 collection would reject every registration.
    endpoint.begin([], genesis_rollout.version)
    exec_manager = None
    checkpoint_writer = None
    try:
        worker_cfg = OfficialWorkerConfig(
            data_root=data_root,
            submission_dir=Path(args.submission_dir or (data_root / "sample_submission")).expanduser(),
            endpoint_url=endpoint.url,
            sandbox_root=Path(cfg["runtime"].get("sandbox_root", "/root/autodl-tmp/gemma4/sandboxes")).expanduser(),
            baseline_python=Path(cfg["runtime"].get("sandbox_baseline_python", "/opt/gemma-sandbox-baseline/bin/python")),
            agent_uid_base=int(args.agent_uid_base or 61000),
            runtime_python=Path(cfg["runtime"].get("python", "/opt/gemma-runtime/bin/python")),
            extra_wheels=extra_wheels,
            session_clock=cfg["runtime"].get("session_clock", "queue_excluded"),
            event_summarizer=cfg["runtime"].get("event_summarizer", "official"),
            evaluation_overrides=dict(cfg["runtime"].get("evaluation", {}) or {}))
        worker = OfficialWorker(worker_cfg, WorkspaceCache(
            run_dir / "workspaces", data_root=data_root),
            uid_allocator=uid_allocator)
        exec_manager = ExecutionManager(
            worker, endpoint, requests, max_workers=max_workers,
            max_attempts=int(run.get("max_worker_retries", 1)) + 1, emit=lambda n, **f: _emit(metrics_path, n, **f),
            # A single task must not outlive the remaining run allowance.
            task_timeout_seconds=min(3600.0, max(60.0, allowance - consumed)),
            # Globally unique, persisted attempt numbers: retries and probes
            # never reuse another trajectory's UID pair or request namespace.
            attempt_seq=attempt_seq.take)
        scheduler = RolloutScheduler(
            catalog.sampler_tasks(), sampler,
            lambda: double_buffer.rollout,
            seed=int(policy_cfg.get("seed", 3407)), health=catalog.healthy,
            initial_stream_sequence=ledger.max_stream_sequence())
        barrier = VersionBarrier(
            scheduler, exec_manager, double_buffer,
            drain_timeout_seconds=float(run.get("barrier_drain_timeout_seconds", 600)),
            emit=lambda n, **f: _emit(metrics_path, n, **f))
        trainer_state = checkpoint_path
        trainer_tx_path = run_dir / "spo_trainer_tx.json"
        checkpoint_writer = CheckpointWriter(trainer_state, torch)

        def persist_trainer_state(state: Mapping[str, Any], *, sync: bool) -> None:
            """Queue the optimizer_step full snapshot; wait before KL.

            The trainer clones LoRA/Adam/scheduler/RNG to CPU before this call,
            so the background thread only owns filesystem I/O. This is the
            only full snapshot in the SPO update; learning_started/optimized
            use the lightweight transaction record.
            """
            checkpoint_writer.submit(state, sync=sync)
            _emit(metrics_path, "spo_checkpoint_queued",
                  phase=state.get("checkpoint_phase"), sync=bool(sync),
                  completed_optimizer_steps=state.get("completed_optimizer_steps"))

        def persist_transaction_state(transaction: Mapping[str, Any], *, adapter: Any = None) -> None:
            """Fsync the small phase/KL/link record; never the large tensors."""
            record = write_transaction(
                trainer_tx_path, transaction,
                checkpoint_path=trainer_state, adapter=adapter)
            _emit(metrics_path, "spo_transaction_persisted",
                  update_id=record.get("update_id"), phase=record.get("phase"),
                  kl_ready=record.get("kl_ready"),
                  adapter_sha256=(record.get("adapter") or {}).get("adapter_sha256"))

        def load_transaction_state() -> Mapping[str, Any] | None:
            return read_transaction(trainer_tx_path, checkpoint_path=trainer_state)

        trainer = make_spo_trainer(policy, ledger, {
            "spo": cfg["spo"],
            "optimizer": {**cfg["optimizer"], "total_optimizer_steps": max(1, max_batches)},
        }, checkpoint_callback=persist_trainer_state,
           transaction_callback=persist_transaction_state)
        # max_batches is the global cumulative update budget for the whole
        # training task, not a per-process allowance: the LR schedule total is
        # max(1, max_batches) and the main loop below is driven by the global
        # trainer.completed_optimizer_steps. A --resume run must reuse the
        # original max_batches; changing it would rewrite the schedule total
        # and break the restored AdamW/scheduler state (restore_training_state
        # already rejects a mismatched trainer_config/schedule).
        global_total = int(trainer.schedule_contract["total_steps"])
        if args.resume and trainer_state.is_file():
            import torch as _torch_check
            try:
                _saved = _torch_check.load(trainer_state, map_location="cpu",
                                           weights_only=False)
            except Exception as exc:
                raise RuntimeError(f"trainer checkpoint is unreadable: {exc}") from exc
            _saved_total = ((_saved.get("lr_schedule") or {}).get("total_steps")
                            if isinstance(_saved, Mapping) else None)
            if _saved_total != global_total:
                raise RuntimeError(
                    f"resume requested total_optimizer_steps={global_total} "
                    f"(max_batches={max_batches}) but the checkpoint holds "
                    f"total_steps={_saved_total}; max_batches is a global "
                    f"cumulative budget and must not change across --resume")
        # P0-1: safe while the learner is unloaded. Parameter capture, rank
        # validation and optimizer creation happen in bind_parameters()
        # inside the training lease; P0-2 reloads rebind without losing AdamW.
        # Resume is handled below by resume_pending_updates() under a training
        # lease (the learner must be resident to restore into). Never restore
        # here: the model is deliberately unloaded at startup.

        remaining = max(0.0, allowance - consumed)
        deadline = start + remaining
        stop_submit = start + max(0.0, remaining - reserve)
        _emit(metrics_path, "spo_budget_window",
              allowance_seconds=allowance, consumed_seconds=consumed,
              init_seconds=round(time.monotonic() - start, 3))
        persist_consumed()
        batches = 0
        in_flight: dict[str, tuple[object, object]] = {}
        lock = threading.RLock()

        def wait_for_workers(timeout: float | None = None, *,
                             return_when=concurrent.futures.ALL_COMPLETED) -> int:
            with lock:
                pending = {future for future, _ in in_flight.values()}
            until = None if timeout is None else time.monotonic() + max(0.0, timeout)
            while True:
                endpoint.raise_if_failed()
                if not pending:
                    return 0
                remaining = None if until is None else until - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return len(pending)
                done, pending = concurrent.futures.wait(
                    pending, timeout=1.0 if remaining is None else min(1.0, remaining),
                    return_when=return_when)
                endpoint.raise_if_failed()
                if done and return_when == concurrent.futures.FIRST_COMPLETED:
                    return len(pending)

        @contextmanager
        def update_transaction():
            """Mark the durable update; SIGTERM lets it finish before stopping."""
            stop_state["in_update"] = True
            try:
                yield
            finally:
                # The step finished: cancel a TERM tombstone watchdog so the
                # process returns to the normal boundary instead of hard-exiting.
                _finish_update_window(stop_state)

        deferred: dict[str, str | None] = {"update_id": None}

        def bounded_drain(update_id: str, phase: str) -> bool:
            return _drain_update_window(
                lambda timeout: wait_for_workers(timeout=timeout), metrics_path,
                update_id=update_id, phase=phase, deadline=deadline,
                update_min_seconds=update_min)

        def require_update_drain(update_id: str, phase: str) -> None:
            """Resume must not start a step that cannot finish; keep it pending."""
            if bounded_drain(update_id, phase):
                return
            raise RuntimeError(
                f"insufficient remaining budget to finish pending update {update_id} "
                f"({phase}); it stays pending for --resume with more walltime "
                f"(update_min_seconds={update_min:g})")

        def settle_done(future: concurrent.futures.Future, trajectory_id: str, assignment: object) -> None:
            try:
                try:
                    # Never auto-publish from a worker callback: publication is
                    # owned by the version barrier outside the training lease
                    # (auto-publish would re-enter an inference lease while a
                    # training lease is held).
                    double_buffer.release_trajectory(trajectory_id, auto_publish=False)
                except (KeyError, ValueError):
                    pass
                error = future.exception()
                if error is not None:
                    _emit(metrics_path, "trajectory_failed", trajectory_id=trajectory_id,
                          error=f"{type(error).__name__}: {error}")
                    try:
                        # A shared serving outage is not task-specific evidence.
                        if endpoint.fatal_error is None:
                            catalog.record_outcome(assignment.task_key, resolved=None,
                                                   infrastructure_error=f"{type(error).__name__}")
                    except (ValueError, KeyError, TypeError):
                        pass
                    # A trajectory sealed before the crash still blocks later
                    # dispatch sequences of its task. Finalize it now so the
                    # run continues without waiting for a restart.
                    try:
                        action = tracker.recover_trajectory(trajectory_id)
                        if action == "unknown":
                            # Sealed before crash would have returned a real
                            # action; unknown means no trajectory row exists,
                            # so drop the marker instead of blocking the task
                            # until it ages out.
                            try:
                                tracker.unmark_dispatch(trajectory_id)
                            except (ValueError, TypeError, RuntimeError, OSError):
                                pass
                        elif action not in ("no_reward_other_state",
                                            "already_staged"):
                            _emit(metrics_path, "trajectory_recovered",
                                  trajectory_id=trajectory_id, action=action)
                    except (ValueError, TypeError, RuntimeError, OSError):
                        pass
                else:
                    try:
                        reward = ledger.get_reward(trajectory_id)
                        if reward.eligible or endpoint.fatal_error is None:
                            catalog.record_outcome(
                                assignment.task_key, resolved=reward.resolved,
                                infrastructure_error=None if reward.eligible else reward.exclusion_reason)
                    except (KeyError, ValueError, TypeError):
                        pass
                    # A trajectory that will never observe (worker failed, or
                    # an ineligible reward) must not keep blocking later
                    # dispatch sequences of the same task. Applied observations
                    # already deleted their marker; only terminal cases remain.
                    try:
                        terminal = error is not None
                        if not terminal:
                            try:
                                terminal = not ledger.get_reward(trajectory_id).eligible
                            except (KeyError, ValueError, TypeError):
                                terminal = True
                        if terminal:
                            tracker.unmark_dispatch(trajectory_id)
                    except (ValueError, TypeError, RuntimeError, OSError):
                        pass
            finally:
                with lock:
                    in_flight.pop(trajectory_id, None)

        def restore_learner_weights() -> bool:
            """Load the committed, or imported-initial, LoRA into the learner.

            ensure_loaded() always builds a zero-initialized LoRA. A normal
            fresh run intentionally keeps that zero base; a continuation run
            loads the imported adapter once before its first optimizer step.
            After any committed step the trainer checkpoint is authoritative.
            """
            if trainer.completed_optimizer_steps <= 0:
                if initial_adapter_manifest is None:
                    if initial_record is not None:
                        raise RuntimeError(
                            f"initial adapter v{initial_version} is recorded but missing "
                            "from the registry; cannot start the first optimizer step")
                    return False
                policy.load_adapter_weights(
                    adapters.path(initial_adapter_manifest),
                    expected_manifest=initial_adapter_manifest)
                _emit(metrics_path, "spo_initial_adapter_loaded",
                      version=initial_adapter_manifest.version,
                      adapter_sha256=initial_adapter_manifest.adapter_sha256[:16],
                      completed_optimizer_steps=0)
                return True
            if not trainer_state.is_file():
                return False
            import torch as _torch
            state = _torch.load(trainer_state, map_location="cpu", weights_only=False)
            policy.restore_adapter_state(state["adapter"])
            return True

        def export_and_register(update_id: str, batch: object) -> Any:
            """Idempotent OPTIMIZED -> EXPORTED -> REGISTERED (no GPU needed)."""
            from .rollout.policy_version import AdapterManifest
            base = double_buffer.rollout
            if base is None:
                raise RuntimeError("no rollout adapter is available for update export")
            version = base.version + 1
            candidate_dir = run_dir / "adapters" / f"staging-v{version}"
            manifest_path = candidate_dir / "manifest.json"
            if manifest_path.is_file():
                manifest = AdapterManifest.read(manifest_path)
                if manifest.version != version or manifest.parent_version != base.version:
                    raise ValueError(f"stale candidate export in {candidate_dir}")
            else:
                manifest = exporter.export(
                    policy, candidate_dir, version=version,
                    parent_version=base.version,
                    optimizer_step=trainer.completed_optimizer_steps)
            ledger.set_update_phase(update_id, "EXPORTED", optimizer_cursor=1)
            try:
                adapters.publish(manifest, candidate_dir)
            except FileExistsError:
                registered = adapters.get(manifest.version)
                if registered != manifest:
                    raise ValueError(
                        f"adapter version {manifest.version} is registered with different content")
            pending = double_buffer.training
            if pending is None:
                double_buffer.begin_training(manifest)
            elif pending != manifest:
                raise ValueError("double buffer holds a different training candidate")
            ledger.set_update_phase(update_id, "REGISTERED", optimizer_cursor=1)
            return manifest

        def restore_training_candidate(update_id: str) -> Any:
            """Rebuild the DoubleBuffer N+1 slot for an EXPORTED/REGISTERED resume.

            A restart always constructs DoubleBuffer with training=None, so a
            pending update that already finished its optimizer step would hit
            ``publish()`` with no candidate. Rebuild from the staged export
            and the immutable registry without re-running AdamW:

            - EXPORTED: staging manifest must exist; publish it if the
              registry does not have it yet, then begin_training().
            - REGISTERED: registry must already hold the candidate; only
              begin_training() is missing.
            Verifies update id, parent version, adapter SHA256 (weights file
            hash), optimizer step and trainer checkpoint identity.
            """
            from .rollout.policy_version import AdapterManifest, sha256_file
            stored = ledger.get_update(update_id)
            if stored.phase not in ("EXPORTED", "REGISTERED"):
                raise RuntimeError(
                    f"restore_training_candidate only handles EXPORTED/REGISTERED, "
                    f"not {stored.phase} for {update_id}")
            expected_version = stored.policy_before.version + 1
            pending = double_buffer.training
            if pending is not None:
                if pending.version != expected_version \
                        or pending.parent_version != stored.policy_before.version:
                    raise ValueError(
                        f"double buffer holds v{pending.version}, not the candidate "
                        f"v{expected_version} for {update_id}")
                return pending
            if double_buffer.rollout is None \
                    or double_buffer.rollout.version != stored.policy_before.version:
                raise ValueError(
                    f"rollout v{double_buffer.rollout.version if double_buffer.rollout else None} "
                    f"is not the base v{stored.policy_before.version} of {update_id}")
            candidate_dir = run_dir / "adapters" / f"staging-v{expected_version}"
            manifest_path = candidate_dir / "manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"staged candidate export is missing for {update_id}: {manifest_path}")
            manifest = AdapterManifest.read(manifest_path)
            if manifest.version != expected_version \
                    or manifest.parent_version != stored.policy_before.version:
                raise ValueError(
                    f"staged candidate v{manifest.version}/parent {manifest.parent_version} "
                    f"does not match update {update_id} v{expected_version}/"
                    f"parent {stored.policy_before.version}")
            if manifest.optimizer_step != trainer.completed_optimizer_steps:
                raise ValueError(
                    f"staged candidate optimizer_step {manifest.optimizer_step} disagrees "
                    f"with trainer {trainer.completed_optimizer_steps} for {update_id}")
            weights_path = candidate_dir / "adapter_model.safetensors"
            if not weights_path.is_file() \
                    or sha256_file(weights_path) != manifest.adapter_sha256:
                raise ValueError(
                    f"staged weights for {update_id} are missing or do not match "
                    f"manifest SHA256")
            try:
                registered = adapters.get(expected_version)
            except KeyError:
                registered = None
            if registered is None:
                registered = adapters.publish(manifest, candidate_dir)
                _emit(metrics_path, "spo_candidate_republished",
                      update_id=update_id, version=registered.version)
            if registered != manifest:
                raise ValueError(
                    f"adapter version {expected_version} is registered with different "
                    f"content than the staged candidate for {update_id}")
            if adapters.active is not None \
                    and adapters.active.version >= expected_version:
                raise ValueError(
                    f"registry already serves v{adapters.active.version}; cannot "
                    f"restore candidate v{expected_version} for {update_id}")
            double_buffer.begin_training(registered)
            if stored.phase == "EXPORTED":
                ledger.set_update_phase(update_id, "REGISTERED", optimizer_cursor=1)
            _emit(metrics_path, "spo_candidate_restored",
                  update_id=update_id, version=registered.version,
                  sha256=registered.adapter_sha256[:16])
            return registered

        def pinned_adapter_versions() -> tuple[int, ...]:
            """Versions the buffer, training slot or trajectories still pin."""
            ownership = double_buffer.ownership()
            versions: set[int] = set()
            for version, count in (ownership.get("trajectory_versions") or {}).items():
                if type(version) is int and type(count) is int and count > 0:
                    versions.add(version)
            for slot in ("rollout", "training"):
                manifest = ownership.get(slot)
                if isinstance(manifest, Mapping) and type(manifest.get("version")) is int:
                    versions.add(int(manifest["version"]))
            return tuple(sorted(versions))

        def prune_in_run_after_commit(update_id: str, version: int) -> None:
            """Drop superseded adapter dirs after a committed, served update.

            Filesystem-only: never inside a training lease, and never allowed
            to turn a successful update into a failure. Any doubt is reported
            and skipped by retention itself.
            """
            if not in_run_prune:
                return
            try:
                from .adapter.retention import prune_in_run
                record = prune_in_run(
                    run_dir, keep_versions=keep_adapter_versions,
                    keep_recent=keep_recent_adapters,
                    pinned_versions=pinned_adapter_versions(),
                    expected_active=version)
            except Exception as error:  # retention must never fail the run
                try:
                    _emit(metrics_path, "spo_in_run_pruned", update_id=update_id,
                          outcome="skipped", active_version=version,
                          error_type=type(error).__name__, error=str(error))
                except OSError:
                    pass
                return
            try:
                _emit(metrics_path, "spo_in_run_pruned", update_id=update_id,
                      outcome="failed" if record["failed"] else "completed",
                      active_version=record["active_version"],
                      removed_count=len(record["removed_versions"]),
                      removed_bytes=record["removed_bytes"],
                      removed_file_bytes=record.get("removed_file_bytes", 0),
                      removed_versions=record["removed_versions"],
                      removed_staging=record["removed_staging"],
                      kept_versions=record["kept_versions"],
                      skipped=record["skipped"], failed=record["failed"])
            except OSError:
                pass

        def activate_and_commit(update_id: str, batch: object, stats: Mapping[str, Any],
                                behavior_before: BehaviorAdapter) -> int:
            """Idempotent REGISTERED -> ACTIVATED -> CONSUMED (inference lease only).

            Must run outside any training lease: double_buffer.publish() takes
            an inference lease, which waits for an exclusive training lease.
            """
            from .rollout.policy_version import AdapterManifest
            stored = ledger.get_update(update_id)
            base = double_buffer.rollout
            if base is None:
                raise RuntimeError("no rollout adapter is available for activation")
            version = stored.policy_before.version + 1
            if base.version == version:
                manifest = base
            else:
                if base.version != stored.policy_before.version:
                    raise ValueError("rollout moved past this update's base version")
                barrier.publish_or_raise()
                manifest = double_buffer.rollout
                if manifest is None or manifest.version != version:
                    raise RuntimeError("publish did not advance the rollout version")
            ledger.set_update_phase(update_id, "ACTIVATED", optimizer_cursor=1)
            try:
                endpoint.advance_version(manifest.version)
            except (ValueError, RuntimeError) as error:
                current = getattr(endpoint, "policy_version", None)
                if current != manifest.version:
                    raise
            behavior_after = BehaviorAdapter(version=manifest.version, sha256=manifest.adapter_sha256)
            if ledger.get_update(update_id).phase != "CONSUMED":
                commit_optimized_update(trainer, behavior_after,
                                        version_chain=(behavior_before, behavior_after))
            # Post-commit consistency: the ledger, trainer, registry, serving
            # buffer and endpoint must agree on the new version before the
            # barrier releases dispatch. Fail loudly, never drift silently.
            if ledger.get_update(update_id).phase != "CONSUMED":
                raise RuntimeError(f"update {update_id} did not reach CONSUMED")
            if trainer.pending_update_id is not None:
                raise RuntimeError(f"trainer still holds {trainer.pending_update_id} after commit")
            if double_buffer.rollout is None or double_buffer.rollout.version != manifest.version:
                raise RuntimeError("serving rollout does not match the committed adapter")
            if adapters.active is None or adapters.active.version != manifest.version:
                raise RuntimeError("adapter registry pointer does not match the committed adapter")
            if getattr(endpoint, "policy_version", None) != manifest.version:
                raise RuntimeError("endpoint policy version does not match the committed adapter")
            if not trainer_state.is_file():
                raise RuntimeError("trainer checkpoint is missing after commit")
            _emit(metrics_path, "spo_update", update_id=update_id, version=manifest.version,
                  action_tokens=batch.total_action_tokens, published=True,
                  kl_value=stats.get("kl_value"), token_count=stats.get("token_count"))
            # The update is committed, served and verified above. Only now may
            # superseded adapter directories be dropped; this is outside every
            # training lease and never changes ledger/registry rows.
            prune_in_run_after_commit(update_id, manifest.version)
            return manifest.version

        def finalize_update(update_id: str, batch: object, stats: Mapping[str, Any],
                            behavior_before: BehaviorAdapter) -> int:
            """Export, persist, commit and serve an OPTIMIZED batch.

            Split across leases: registry work runs inside the caller's
            training lease, serving activation and ledger commit run outside
            it (see activate_and_commit). Memory is released by the caller.
            The optimizer_step full checkpoint is already current; this only
            advances the lightweight transaction record.
            """
            manifest = export_and_register(update_id, batch)
            stored = ledger.get_update(update_id)
            persist_transaction_state(
                trainer.export_transaction_state(phase=stored.phase),
                adapter=manifest)
            return stored.policy_before.version + 1

        def do_update(update_id: str) -> bool:
            endpoint.raise_if_failed()
            deferred["update_id"] = None
            manifest = double_buffer.rollout
            if manifest is None:
                raise RuntimeError("no rollout adapter is available for update reservation")
            behavior_before = BehaviorAdapter(
                version=manifest.version, sha256=manifest.adapter_sha256)
            batch = trajectory_buffer.reserve(update_id, policy_before=behavior_before)
            if batch is None:
                return False
            with barrier.update_window():
                if stop_state["requested"]:
                    raise KeyboardInterrupt("stop requested before optimizer step")
                if not bounded_drain(update_id, "reserve"):
                    # The batch stays RESERVED for a later resume; starting
                    # AdamW now could not finish inside the external cap.
                    deferred["update_id"] = update_id
                    return False
                if stop_state["requested"]:
                    raise KeyboardInterrupt("stop requested before optimizer step")
                with update_transaction():
                    with lease.training():
                        try:
                            with _training_cuda_memory(torch, metrics_path, update_id):
                                load_started = time.monotonic()
                                policy.ensure_loaded()
                                _emit(metrics_path, "spo_model_loaded", phase="update",
                                      seconds=round(time.monotonic() - load_started, 3),
                                      completed_optimizer_steps=trainer.completed_optimizer_steps)
                                restore_learner_weights()
                                stats = trainer.optimize_batch(batch)
                                timing = dict(stats.get("timing") or {})
                                timing["checkpoint_writer"] = checkpoint_writer.stats()
                                _emit(metrics_path, "spo_update_timing", update_id=update_id,
                                      timing=timing)
                                finalize_update(update_id, batch, stats, behavior_before)
                        finally:
                            trainer.suspend_learner_binding()
                            policy.release_memory()
                    # Training lease is released (vLLM resumed) before any
                    # inference-lease work: publish would deadlock inside it.
                    version = activate_and_commit(update_id, batch, stats, behavior_before)
                    _emit(metrics_path, "spo_update_activated", update_id=update_id, version=version)
                    return True

        update_index = len(ledger.list_updates()) if args.resume else 0
        if args.resume:
            _emit(metrics_path, "spo_budget_resumed", allowance_seconds=allowance,
                  consumed_seconds=consumed)
        else:
            persist_consumed()

        def rebase_trainer_checkpoint(base_state: Mapping[str, Any]) -> None:
            """Durably replace a pre-step batch snapshot with the committed base.

            The optimizer_step tensors already describe the committed base; only
            the active-batch marker is removed. The atomic writer keeps the old
            file until the new one is fsynced, so a crash can only repeat the
            rebase, never lose the last committed adapter/AdamW.
            """
            checkpoint_writer.submit(base_state, sync=True)
            _emit(metrics_path, "spo_checkpoint_rebased",
                  completed_optimizer_steps=base_state.get("completed_optimizer_steps"),
                  checkpoint_phase=base_state.get("checkpoint_phase"))

        def drop_trainer_transaction() -> None:
            # Only a validated pre-step transaction reaches this point; any
            # post-step record refuses the discard before deletion.
            trainer_tx_path.unlink(missing_ok=True)
            _emit(metrics_path, "spo_transaction_discarded")

        def restore_discarded_base(base_state: Mapping[str, Any] | None) -> None:
            """Adopt the preserved base so the next update continues trained.

            A fresh trainer starts at completed_optimizer_steps == 0. Without
            this restore the next optimizer step would use zero LoRA and fresh
            AdamW moments while the ledger already reports committed updates.
            """
            if base_state is None:
                return
            completed = base_state.get("completed_optimizer_steps")
            if type(completed) is not int or completed <= 0:
                # A zero-step snapshot is equivalent to a fresh trainer;
                # restore_committed_base() deliberately skips it too.
                return
            if trainer._active is not None or trainer.completed_optimizer_steps \
                    or trainer.optimizer is not None:
                return
            with barrier.update_window():
                with lease.training():
                    load_started = time.monotonic()
                    policy.ensure_loaded()
                    _emit(metrics_path, "spo_model_loaded",
                          phase="resume_discarded_base",
                          seconds=round(time.monotonic() - load_started, 3))
                    try:
                        trainer.restore_base_state(base_state)
                        _emit(metrics_path, "spo_base_restored",
                              completed_optimizer_steps=trainer.completed_optimizer_steps,
                              source="discarded_unoptimized_update")
                    finally:
                        trainer.suspend_learner_binding()
                        policy.release_memory()

        def discard_prestep_update(update_id: str, reason: str, *,
                                   checkpoint_state: Mapping[str, Any] | None,
                                   transaction: Mapping[str, Any] | None) -> bool:
            """Discard without deleting trained state; True when rebased."""
            base_state = discard_unoptimized_update(
                ledger, update_id, reason,
                checkpoint_state=checkpoint_state,
                transaction=transaction,
                rebase_checkpoint=rebase_trainer_checkpoint,
                drop_transaction=drop_trainer_transaction)
            restore_discarded_base(base_state)
            return base_state is not None

        def resume_pending_updates() -> int:
            """Finish ledger-uncompleted updates oldest-first; never skip ahead.

            RESERVED batches optimize fresh through the normal finalize path.
            A single LEARNING/OPTIMIZED batch restores its exact checkpoint
            (update identity is verified) and resumes before any new reserve.
            EXPORTED/REGISTERED/ACTIVATED batches already finished the
            optimizer step: their remaining publish stages replay idempotently
            without another gradient step.

            Pending batches keep their frozen rewards and advantages. Only
            a legacy discard that already saved a rebased checkpoint is
            completed; no new batch is discarded for lacking a positive repair.
            """
            pending = list(ledger.list_updates(("RESERVED", "LEARNING", "OPTIMIZED",
                                                "EXPORTED", "REGISTERED", "ACTIVATED")))
            if not pending:
                return 0
            import torch as _torch
            transaction = load_transaction_state()
            finished = 0
            for record in pending:
                if stop_state["requested"]:
                    raise KeyboardInterrupt("stop requested during resume")
                update_id = record.update_id
                transaction_for_update = transaction \
                    if transaction is not None and transaction.get("update_id") == update_id else None
                if record.phase == "RESERVED":
                    # The ledger stays RESERVED while the async learning_started
                    # snapshot is in flight. If an optimizer_step checkpoint
                    # already exists (crash after the checkpoint but before the
                    # ledger cursor advanced), resume that post-step state
                    # instead of re-running AdamW from a stale base.
                    checkpoint_state = None
                    if trainer_state.is_file():
                        checkpoint_state = _torch.load(
                            trainer_state, map_location="cpu", weights_only=False)
                        if not isinstance(checkpoint_state, Mapping) \
                                or checkpoint_state.get("algorithm") != "single_stream_spo":
                            raise RuntimeError("trainer checkpoint is not a single_stream_spo state")
                        active = checkpoint_state.get("active_batch") or {}
                        if not isinstance(active, Mapping):
                            raise RuntimeError("trainer checkpoint has an invalid active batch")
                        active_update = active.get("update") or {}
                        if not isinstance(active_update, Mapping):
                            raise RuntimeError("trainer checkpoint has an invalid update payload")
                        if _active_update_id(checkpoint_state) == update_id \
                                and active_update.get("optimizer_cursor") not in (0, 1):
                            raise RuntimeError(
                                "current update checkpoint has a missing/invalid optimizer cursor")
                        if _active_update_id(checkpoint_state) == update_id \
                                and active_update.get("optimizer_cursor") == 1:
                            with barrier.update_window():
                                require_update_drain(update_id, "resume_post_step")
                                with update_transaction():
                                    with lease.training():
                                        load_started = time.monotonic()
                                        policy.ensure_loaded()
                                        _emit(metrics_path, "spo_model_loaded",
                                              phase="resume_post_step",
                                              seconds=round(time.monotonic() - load_started, 3))
                                        try:
                                            trainer.restore_training_state(
                                                checkpoint_state,
                                                transaction=transaction_for_update)
                                            stats = trainer.resume_batch()
                                            batch = ledger.get_update(update_id)
                                            finalize_update(update_id, batch, stats,
                                                            batch.policy_before)
                                            stats = dict(trainer._active["stats"]) \
                                                if trainer._active else {}
                                        finally:
                                            trainer.suspend_learner_binding()
                                            policy.release_memory()
                                    batch = ledger.get_update(update_id)
                                    if batch.phase in ("EXPORTED", "REGISTERED"):
                                        manifest = restore_training_candidate(update_id)
                                        batch = ledger.get_update(update_id)
                                        persist_transaction_state(
                                            trainer.export_transaction_state(phase=batch.phase),
                                            adapter=manifest)
                                    activate_and_commit(update_id, batch, stats,
                                                        batch.policy_before)
                            finished += 1
                            continue
                    validate_pending_transaction(
                        record, checkpoint_state, transaction_for_update)
                    behavior_before = record.policy_before
                    batch = ledger.get_update(update_id)
                    if batch.phase != "RESERVED":
                        raise RuntimeError(f"reserved update {update_id} changed phase during resume")
                    with barrier.update_window():
                        require_update_drain(update_id, "resume_reserved")
                        with update_transaction():
                            with lease.training():
                                load_started = time.monotonic()
                                policy.ensure_loaded()
                                _emit(metrics_path, "spo_model_loaded", phase="resume_reserved",
                                      seconds=round(time.monotonic() - load_started, 3))
                                try:
                                    trainer_is_fresh = (
                                        trainer._active is None
                                        and trainer.completed_optimizer_steps == 0
                                        and trainer.optimizer is None)
                                    if trainer_is_fresh and checkpoint_state is not None:
                                        checkpoint_active = _active_update_id(checkpoint_state)
                                        if checkpoint_active in (None, update_id):
                                            trainer.restore_base_state(checkpoint_state)
                                        else:
                                            try:
                                                previous_phase = ledger.get_update(checkpoint_active).phase
                                            except (KeyError, ValueError):
                                                previous_phase = None
                                            if previous_phase != "CONSUMED":
                                                raise RuntimeError(
                                                    f"trainer checkpoint belongs to non-consumed "
                                                    f"update {checkpoint_active}, not pending {update_id}")
                                            trainer.restore_base_state(checkpoint_state)
                                    else:
                                        restore_learner_weights()
                                    stats = trainer.optimize_batch(batch)
                                    timing = dict(stats.get("timing") or {})
                                    timing["checkpoint_writer"] = checkpoint_writer.stats()
                                    _emit(metrics_path, "spo_update_timing", update_id=update_id,
                                          timing=timing)
                                    finalize_update(update_id, batch, stats, behavior_before)
                                finally:
                                    trainer.suspend_learner_binding()
                                    policy.release_memory()
                            batch = ledger.get_update(update_id)
                            stored_stats = dict(trainer._active["stats"]) if trainer._active else {}
                            activate_and_commit(update_id, batch, stored_stats, behavior_before)
                    finished += 1
                    continue
                if not trainer_state.is_file():
                    raise RuntimeError(
                        f"interrupted update {update_id} has no trainer checkpoint; refusing to skip it")
                state = _torch.load(trainer_state, map_location="cpu", weights_only=False)
                active_id = _active_update_id(state)
                if active_id is None and record.phase == "LEARNING" \
                        and record.optimizer_cursor == 0 \
                        and state.get("checkpoint_phase") == "discarded_unoptimized_batch":
                    # Finish a discard already persisted by an older launcher.
                    # This is checkpoint recovery, not a new reward filter.
                    discard_prestep_update(
                        update_id, "finish_legacy_discard",
                        checkpoint_state=state,
                        transaction=transaction_for_update)
                    _emit(metrics_path, "spo_interrupted_discard_completed",
                          update_id=update_id)
                    continue
                if active_id != update_id:
                    raise RuntimeError(
                        f"trainer checkpoint belongs to {active_id}, not pending {update_id}")
                validate_pending_transaction(record, state, transaction_for_update)
                with barrier.update_window():
                    require_update_drain(update_id, "resume_learning")
                    with update_transaction():
                        with lease.training():
                            policy.ensure_loaded()
                            try:
                                trainer.restore_training_state(
                                    state, transaction=transaction_for_update)
                                fresh = ledger.get_update(update_id)
                                if fresh.phase in ("LEARNING", "OPTIMIZED", "RESERVED"):
                                    stats = trainer.resume_batch()
                                    batch = ledger.get_update(update_id)
                                    finalize_update(update_id, batch, stats, batch.policy_before)
                                    stats = dict(trainer._active["stats"]) if trainer._active else {}
                                else:
                                    # EXPORTED/REGISTERED/ACTIVATED already ran AdamW.
                                    # Spend only a KL probe if the transaction record
                                    # did not carry valid KL evidence.
                                    stats = trainer.ensure_kl_ready()
                                    batch = fresh
                            finally:
                                trainer.suspend_learner_binding()
                                policy.release_memory()
                        batch = ledger.get_update(update_id)
                        if batch.phase in ("EXPORTED", "REGISTERED"):
                            manifest = restore_training_candidate(update_id)
                            batch = ledger.get_update(update_id)
                            persist_transaction_state(
                                trainer.export_transaction_state(phase=batch.phase),
                                adapter=manifest)
                        activate_and_commit(update_id, batch, stats, batch.policy_before)
                finished += 1
            return finished

        def restore_committed_base() -> bool:
            """Restore the last committed adapter + AdamW/scheduler/RNG on --resume.

            A fresh process starts with completed_optimizer_steps == 0 even
            when the disk holds committed training. Without this step the next
            update would train from a zero LoRA with fresh moments. Runs
            inside a training lease; the learner is suspended/released after.
            Skipped when the checkpoint still owns an uncommitted batch
            (resume_pending_updates handles that path instead).
            """
            if not trainer_state.is_file():
                return False
            import torch as _torch
            state = _torch.load(trainer_state, map_location="cpu", weights_only=False)
            if not isinstance(state, Mapping) or int(state.get("completed_optimizer_steps", 0)) <= 0:
                return False
            active_id = _active_update_id(state)
            orphaned_prestep = False
            if active_id is not None:
                try:
                    pending_phase = ledger.get_update(active_id).phase
                except (KeyError, ValueError):
                    pending_phase = None
                if pending_phase is None:
                    # A pre-step rollback may have removed the ledger row but
                    # crashed before rebasing the checkpoint. Its cursor-0
                    # snapshot is the committed base, not a resumable batch.
                    cursor = _checkpoint_update_cursor(state, active_id)
                    active = state.get("active_batch") or {}
                    offset = active.get("offset")
                    completed = state.get("completed_optimizer_steps")
                    if cursor == 0 and type(completed) is int and completed == offset:
                        orphaned_prestep = True
                    else:
                        raise RuntimeError(
                            f"trainer checkpoint refers to unknown update {active_id}; "
                            "refusing to resume")
                elif pending_phase != "CONSUMED":
                    return False
            with barrier.update_window():
                with lease.training():
                    load_started = time.monotonic()
                    policy.ensure_loaded()
                    _emit(metrics_path, "spo_model_loaded", phase="resume_base",
                          seconds=round(time.monotonic() - load_started, 3))
                    try:
                        if orphaned_prestep:
                            trainer.restore_base_state(state)
                            _emit(metrics_path, "spo_base_restored",
                                  completed_optimizer_steps=trainer.completed_optimizer_steps,
                                  source="orphaned_prestep_owner", owner=active_id)
                        else:
                            trainer.restore_training_state(state)
                            _emit(metrics_path, "spo_base_restored",
                                  completed_optimizer_steps=trainer.completed_optimizer_steps)
                        return True
                    finally:
                        trainer.suspend_learner_binding()
                        policy.release_memory()

        if args.resume:
            restore_committed_base()
            # Fold any observations that were staged but unapplied at
            # interruption, in stream order. No GPU or learner is needed.
            drained = 0
            for pending_task in catalog.sampler_tasks():
                drained += tracker.drain_pending(pending_task)
            if drained:
                _emit(metrics_path, "spo_pending_drained", applied_observations=drained)
            resumed = resume_pending_updates()
            if resumed:
                _emit(metrics_path, "spo_resume_completed", finished_updates=resumed)
            update_index = max(
                [0] + [int(str(record.update_id).rsplit("-", 1)[-1])
                       for record in ledger.list_updates()
                       if str(record.update_id).rsplit("-", 1)[-1].isdigit()]) + 1
        else:
            update_index = 0
        _emit(metrics_path, "spo_run_started", config=cfg.get("algorithm"),
              tasks=len(catalog), resume=bool(args.resume),
              inference={"engine": backend.name, "base_url": backend_url},
              backend_handshake=handshake,
              model_revision=identity["model_revision"], model_path=identity["model_path"])
        # Report what this launcher actually runs. Other objectives need their
        # own execution path; no preceding GPU result is required to build it.
        _emit(metrics_path, "spo_training_phase", phase="online_spo_only",
              sft_wired=False, distillation_wired=False, opd_wired=False)
        while trainer.completed_optimizer_steps < global_total and time.monotonic() < stop_submit:
            if stop_state["requested"]:
                raise KeyboardInterrupt("stop requested")
            endpoint.raise_if_failed()
            with lock:
                room = max_workers - len(in_flight)
            for _ in range(room):
                if time.monotonic() >= stop_submit:
                    break
                # Registration happens in a worker; without this synchronous
                # check a dead generation thread produced hundreds of new
                # zero-call EXCLUDED trajectories before task health ran out.
                endpoint.raise_if_failed()
                try:
                    assignment = scheduler.next_assignment()
                except BarrierActive:
                    time.sleep(1.0)
                    break
                except NoDispatchableTask as idle:
                    # Normal wait (e.g. the only queued tasks are long and the
                    # long slot is occupied): back off, keep the run alive.
                    _emit(metrics_path, "scheduler_waiting", reason=str(idle))
                    time.sleep(min(5.0, max(0.5, stop_submit - time.monotonic())))
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
            endpoint.raise_if_failed()
            if not in_flight:
                time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))
                continue
            wait_for_workers(timeout=min(5.0, max(0.05, deadline - time.monotonic())),
                             return_when=concurrent.futures.FIRST_COMPLETED)
            rollout_now = double_buffer.rollout
            if rollout_now is not None and trainer.completed_optimizer_steps < global_total:
                try:
                    if do_update(f"upd-{update_index:04d}"):
                        batches += 1
                        update_index += 1
                    elif deferred["update_id"] is not None:
                        break
                except (ValueError, RuntimeError) as error:
                    endpoint.raise_if_failed()
                    _emit(metrics_path, "spo_update_failed", error=str(error))
        with exec_manager.paused():
            final_drained = deferred["update_id"] is None and bounded_drain(
                f"upd-{update_index:04d}", "final")
            final_done = False
            if final_drained and trainer.completed_optimizer_steps < global_total \
                    and time.monotonic() < deadline:
                try:
                    rollout_now = double_buffer.rollout
                    if rollout_now is not None and do_update(f"upd-{update_index:04d}"):
                        batches += 1
                        final_done = True
                except (ValueError, RuntimeError) as error:
                    endpoint.raise_if_failed()
                    _emit(metrics_path, "spo_update_failed", error=str(error))
            persist_consumed()
            _emit(metrics_path, "spo_run_stopped", batches=batches, final_update=final_done,
                  completed_optimizer_steps=trainer.completed_optimizer_steps,
                  global_total_steps=global_total,
                  buffer=ledger.buffer_state())
        return 0 if trainer.completed_optimizer_steps >= global_total else 75
    except Exception:
        if endpoint.fatal_error is not None:
            error = endpoint.fatal_error
            _emit(metrics_path, "spo_generation_failed",
                  error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            budget_stop.set()
            heartbeat.join(timeout=5.0)
        except (RuntimeError, ValueError):
            pass
        # Never let a stale TERM watchdog fire while the process is doing a
        # normal, bounded teardown (the update window already clears it).
        _cancel_stop_watchdog(stop_state)
        if stop_state["requested"]:
            try:
                _emit(metrics_path, "spo_stop_requested",
                      consumed_seconds=round(consumed + (time.monotonic() - start), 3),
                      in_update=stop_state["in_update"],
                      sigterm_count=stop_state.get("signals", 0),
                      sigterm_grace_seconds=sigterm_grace)
            except OSError:
                pass
        try:
            if exec_manager is not None:
                exec_manager.shutdown(cancel_pending=True)
        except (RuntimeError, OSError, ValueError):
            pass
        try:
            if checkpoint_writer is not None:
                checkpoint_writer.close()
        except (RuntimeError, OSError, ValueError) as error:
            try:
                _emit(metrics_path, "spo_checkpoint_writer_failed", error=str(error))
            except OSError:
                pass
        try:
            # Final submit + synchronous flush; the writer's max() rule makes
            # this unable to regress a newer value already on disk.
            budget_writer.close(max(0.0, consumed + (time.monotonic() - start)))
        except (RuntimeError, OSError, ValueError):
            pass
        try:
            signal.signal(signal.SIGTERM, previous_sigterm)
        except (ValueError, OSError, RuntimeError):
            pass
        try:
            endpoint.close()
        except (RuntimeError, OSError, ValueError):
            pass
        ledger.close()


def _prune_if_configured(args: argparse.Namespace) -> None:
    """Drop process artifacts of a finished run when the config asks for it.

    Only a run whose optimizer plan is complete and whose updates are all
    CONSUMED reaches this point; a budget/timeout exit returns 75 first.
    """
    try:
        cfg = load_config(Path(args.config))
    except (OSError, ValueError, TypeError):
        return
    if not cfg["run"].get("prune_finished_artifacts"):
        return
    run_dir = Path(args.run_dir).expanduser()
    checkpoint_path = _checkpoint_path(run_dir, cfg["run"])
    try:
        from .adapter.retention import prune_finished_run
        record = prune_finished_run(
            run_dir, keep=tuple(cfg["run"].get("keep_adapter_versions", [])),
            checkpoint_path=checkpoint_path)
    except Exception as error:  # never turn a finished run into a failure
        print(f"spo_prune_skipped: {type(error).__name__}: {error}", file=sys.stderr)
        return
    removed = record.get("total_bytes", 0)
    file_bytes = record.get("file_bytes", removed)
    suffix = ("" if file_bytes == removed else
              f" (file bytes {file_bytes / 1e9:.3f} GB; hardlinked duplicates count once)")
    print(f"spo_artifacts_pruned: {len(record.get('removed', []))} item(s), "
          f"{removed / 1e9:.3f} GB unique bytes reclaimed{suffix}")


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
        if not backend_url:
            print("spo_check_backend_missing: no inference base_url configured", file=sys.stderr)
            return 2
        # Read-only preflight: health, deployment metadata and metrics only.
        # start() never submits/drains/loads; it verifies the vLLM LoRA rank,
        # runtime-LoRA flag and Level-2 sleep endpoints before a GPU session.
        try:
            probe = _make_inference_backend(cfg, backend_url)
            report = probe.start()
            from .rollout.gateway import InferenceGateway
            deployment = InferenceGateway(probe).probe_deployment()
            print("inference_start=" + json.dumps(
                {"start": report, "deployment": deployment}, default=str)[:4000])
        except Exception as error:
            print(f"spo_check_backend_unreachable: {type(error).__name__}: {error}",
                  file=sys.stderr)
            return 2
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
        code = run_spo(args)
    except (ValueError, RuntimeError, TypeError, OSError) as error:
        print(f"spo_run_failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if code == 0:
        _prune_if_configured(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
