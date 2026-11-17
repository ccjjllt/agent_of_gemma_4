"""Independent evaluation of already-published adapters (rollout + judge only).

One invocation evaluates exactly one policy against the fixed public task subset
configured for evaluation:

- ``base``: the served base model, submitted without any LoRA adapter;
- ``v3`` / ``v5``: an existing adapter directory loaded from its original path
  and verified against its ``manifest.json`` before any GPU work.

The entry point deliberately reuses the training rollout and judge machinery
(:class:`OfficialWorker`, :class:`EmptyPatchBaselineStore`, :class:`RewardEngine`,
:class:`RolloutEngine`, :class:`TrajectoryLedger`) but never constructs a
trainer, writes a trainer checkpoint or reserves/consumes an update batch.  A
small eval-only buffer and value tracker keep ``RolloutEngine`` unchanged; they
are never reserved from.

What is measured: per official public task, ``resolved`` (official success),
``P`` (differential repair fraction when the empty-patch baseline is available),
and reward ``validity``.  Missing P stays missing in every report; it is never
coerced to zero.  The six-task pool is also used in training, so these are
in-sample diagnostics, not a generalization result, and three repeats are not a
significance test.

This module has no GPU-side substitute for the real vLLM server.  ``--plan-only``
is CPU-only and never contacts vLLM; actual rollouts require the configured
server and the two-GPU runtime identity used by training.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import stat
import sys
import threading
import time
import tomllib
import uuid
from typing import Any, Mapping, Sequence

from .data.curriculum import TaskCatalog
from .endpoint import PolicyEndpoint
from .environment.execution_manager import AttemptSequence, ExecutionManager
from .environment.sandbox import UidAllocator
from .environment.worker import OfficialWorker, OfficialWorkerConfig
from .environment.workspace_cache import WorkspaceCache
from .judge_identity import bind_run_identity, canonical_environment, read_runtime_identity
from .policy import Policy
from .reward.baseline import EmptyPatchBaselineStore, TaskKey as JudgeTaskKey, normalize_baseline
from .reward.baseline_probe import EmptyPatchProbeRunner
from .reward.engine import RewardEngine
from .reward.junit_bridge import task_key_for
from .reward.progress import SUPPORTED_OBJECTIVE_VERSIONS
from .rollout.engine import RolloutEngine
from .rollout.gateway import validate_sampling
from .rollout.policy_version import AdapterManifest, sha256_file
from .rollout.request_manager import RequestManager
from .rollout.scheduler import RolloutAssignment
from .rollout.vllm_backend import VLLMBackend
from .spo.value_tracker import ValueTracker
from .trajectory.buffer import TrajectoryBuffer
from .trajectory.ledger import TrajectoryLedger
from .trajectory.schema import BehaviorAdapter, RewardRecord, TaskKey


EVAL_SCHEMA = "gemma.adapter_eval.v1"
EVAL_EVENT = "eval_sample"
POLICY_LABEL = re.compile(r"v?(\d+)\Z")
BASE_BEHAVIOR_MARKER = "served_base_model_no_lora"
# A clearly marked behavior sentinel, not an adapter hash and not the deleted
# zero-initialized genesis adapter.  Version 0 plus this digest identifies the
# unadapted served base policy inside the normal Trajectory schema.
BASE_BEHAVIOR_SHA256 = hashlib.sha256(
    b"gemma4-eval-base-policy-v0:no-lora:served-base-model").hexdigest()
_TRAJECTORY = re.compile(
    r"^eval-(?P<policy>[a-z0-9]+)-(?P<task>[A-Za-z0-9_.-]+)"
    r"-r(?P<repeat>\d{2})-a(?P<attempt>\d{2})-(?P<nonce>[0-9a-f]{12})$")
_SAFE_LABEL = re.compile(r"[a-z0-9]+\Z")
_REQUIRED_CONFIG_SECTIONS = (
    "policy", "run", "inference", "spo", "optimizer", "rewards", "adapter",
    "data", "runtime", "sampling",
)
_EXIT_CONFIG = 2
_EXIT_INCOMPLETE = 4
_EXIT_DATABASE = 3
_EXIT_BUDGET = 75


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(dict(payload), output, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _emit(metrics_path: Path, event: str, **fields: Any) -> None:
    record = {"event": event, "unix": time.time(), **fields}
    with metrics_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                default=str) + "\n")
        output.flush()
        os.fsync(output.fileno())


# ---------------------------------------------------------------------------
# Configuration and model/adapters
# ---------------------------------------------------------------------------
def load_eval_config(path: Path) -> dict[str, Any]:
    """Load the shared training configuration and the eval-only additions.

    The policy, inference, reward, runtime and sampling sections are the same
    objects training uses.  Training-only fields (``checkpoint_dir``, update
    budgets) are ignored here instead of being silently reinterpreted.  The
    required gates below mirror the training launcher so an eval cannot use a
    config that a training run would reject.
    """
    with open(path, "rb") as handle:
        config = tomllib.load(handle)
    if config.get("algorithm") != "single_stream_spo":
        raise ValueError("algorithm must be single_stream_spo")
    for section in _REQUIRED_CONFIG_SECTIONS:
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"missing config section: {section}")
    for section in ("sft", "distillation", "opd", "teacher"):
        if section in config:
            raise ValueError(f"[{section}] is not implemented by this evaluation entry")
    policy = config["policy"]
    for name in ("lora_rank", "lora_alpha"):
        if type(policy.get(name)) is not int or policy[name] <= 0:
            raise ValueError(f"policy.{name} must be a positive integer")
    if float(policy.get("lora_dropout", 0.0)) != 0.0:
        raise ValueError("the current shared-prefix learner implements dropout=0")
    if not str(policy.get("model_revision") or "").strip():
        raise ValueError("policy.model_revision is required for train/infer identity")
    if policy.get("response_parser", "official_vllm_gemma4_0_19_1") != "official_vllm_gemma4_0_19_1":
        raise ValueError("unsupported response parser")
    if str(config["inference"].get("engine", "vllm")).lower() != "vllm":
        raise ValueError("this evaluation entry supports the vLLM backend only")
    if int(config["spo"].get("target_action_tokens", 0)) <= 0 \
            or int(config["spo"].get("max_policy_lag", -1)) < 0:
        raise ValueError("invalid spo token budget or lag")
    if type(config["adapter"].get("allow_resident_pair", False)) is not bool:
        raise ValueError("adapter.allow_resident_pair must be bool")
    if config["rewards"].get("objective_version") not in SUPPORTED_OBJECTIVE_VERSIONS:
        raise ValueError("wrong rewards objective_version")
    task_ids = config["data"].get("task_ids")
    if not isinstance(task_ids, list) or not task_ids \
            or any(not isinstance(item, str) or not item.strip() for item in task_ids):
        raise ValueError("data.task_ids must be a non-empty list of task ids")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("data.task_ids must be unique")
    eval_section = config.get("eval", {})
    if not isinstance(eval_section, dict):
        raise ValueError("[eval] must be a table")
    repeats = eval_section.get("repeats", 3)
    if type(repeats) is not int or repeats < 1:
        raise ValueError("eval.repeats must be a positive integer")
    return config


def validate_model_dir(model_path: Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Torch-free copy of the training model gate and cheap identity record."""
    raw = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_config = raw.get("text_config", {})
    if raw.get("model_type") != "gemma4" or text_config.get("num_hidden_layers") != 60:
        raise ValueError("model directory is not the Gemma 4 31B checkpoint")
    if raw.get("quantization_config"):
        raise ValueError("use the official QAT-unquantized BF16 checkpoint, not a packed/NF4 base")
    tokenizer_files = [name for name in ("tokenizer.json", "tokenizer.model", "spiece.model")
                       if (model_path / name).is_file()]
    if not tokenizer_files:
        raise ValueError("model directory has no tokenizer file")
    return {
        "model_path": str(model_path),
        "model_name": policy.get("model_name"),
        "model_revision": policy.get("model_revision"),
        "model_source_uri": policy.get("model_source_uri"),
        "config_json_sha256": sha256_file(model_path / "config.json"),
        "tokenizer_files": {name: sha256_file(model_path / name)
                            for name in tokenizer_files},
        "lora": {"rank": policy.get("lora_rank"), "alpha": policy.get("lora_alpha"),
                 "dropout": policy.get("lora_dropout")},
    }


@dataclass(frozen=True)
class AdapterEvidence:
    label: str
    directory: Path
    manifest_path: Path
    manifest_sha256: str
    weights_sha256: str
    weights_bytes: int
    manifest: AdapterManifest

    @property
    def version(self) -> int:
        return self.manifest.version

    @property
    def behavior(self) -> BehaviorAdapter:
        return BehaviorAdapter(self.manifest.version, self.manifest.adapter_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "directory": str(self.directory),
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "weights_sha256": self.weights_sha256,
            "weights_bytes": self.weights_bytes,
            "version": self.manifest.version,
            "parent_version": self.manifest.parent_version,
            "optimizer_step": self.manifest.optimizer_step,
            "rank": self.manifest.rank,
            "alpha": self.manifest.alpha,
            "dropout": self.manifest.dropout,
            "base_revision": self.manifest.base_revision,
            "config_hash": self.manifest.config_hash,
        }


def validate_adapter_dir(path: Path, policy: Mapping[str, Any], *,
                         expected_version: int | None = None) -> AdapterEvidence:
    """Read-only manifest/SHA/config validation; the source run is never opened.

    It intentionally does not construct ``AdapterManager``: a finished source
    run can have a registry that lists pruned intermediate versions, while the
    requested version directories remain valid.  The manifest and weights are
    the authority here, and ``VLLMBackend.load_adapter`` re-verifies the digest
    when the server loads the original path.  Any published version is allowed;
    ``expected_version`` only pins a user-supplied ``--policy vN`` label.
    """
    directory = path.expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError(f"adapter path is not a directory: {directory}")
    manifest_path = directory / "manifest.json"
    manifest = AdapterManifest.read(manifest_path)
    if expected_version is not None and manifest.version != expected_version:
        raise ValueError(
            f"--policy requests adapter version {expected_version}, "
            f"found {manifest.version}")
    if manifest.rank != int(policy["lora_rank"]) or manifest.alpha != int(policy["lora_alpha"]):
        raise ValueError("adapter rank/alpha differ from policy.lora_rank/lora_alpha")
    if manifest.dropout != float(policy.get("lora_dropout", 0.0)):
        raise ValueError("adapter dropout differs from policy.lora_dropout")
    if manifest.base_revision != policy.get("model_revision"):
        raise ValueError("adapter base_revision differs from policy.model_revision")
    weights = directory / "adapter_model.safetensors"
    if not weights.is_file():
        raise ValueError(f"adapter weights are missing: {weights}")
    weights_bytes = weights.stat().st_size
    weights_sha256 = sha256_file(weights)
    if weights_sha256 != manifest.adapter_sha256:
        raise ValueError("adapter weights SHA256 differs from manifest.adapter_sha256")
    config_path = directory / "adapter_config.json"
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"adapter_config.json is unreadable: {exc}") from exc
    if not isinstance(adapter_config, dict) or adapter_config.get("peft_type") != "LORA":
        raise ValueError("adapter_config.json is not a PEFT LORA adapter")
    if (adapter_config.get("r"), adapter_config.get("lora_alpha")) \
            != (manifest.rank, manifest.alpha):
        raise ValueError("adapter_config.json rank/alpha differ from the manifest")
    try:
        dropout = float(adapter_config.get("lora_dropout", -1))
    except (TypeError, ValueError):
        dropout = -1.0
    if isinstance(adapter_config.get("lora_dropout"), bool) or dropout != manifest.dropout:
        raise ValueError("adapter_config.json dropout differs from the manifest")
    if adapter_config.get("use_dora") or adapter_config.get("use_rslora") \
            or adapter_config.get("modules_to_save"):
        raise ValueError("adapter uses an unsupported PEFT variant")
    if adapter_config.get("bias") not in (None, "none"):
        raise ValueError("adapter must not train biases")
    return AdapterEvidence(
        label=f"v{manifest.version}",
        directory=directory,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        weights_sha256=weights_sha256,
        weights_bytes=weights_bytes,
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Official task keys and judge identity
# ---------------------------------------------------------------------------
def build_task_keys(tasks_path: Path, fingerprint: str,
                    include: Sequence[str], exclude: Sequence[str]) -> list[TaskKey]:
    """Judge TaskKeys for the eval catalog; heldout rows are never selected."""
    selected = set(include) - set(exclude)
    keys: dict[str, TaskKey] = {}
    with tasks_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("heldout") is True:
                raise ValueError("heldout rows cannot enter the public evaluation catalog")
            task_id = row.get("instance_id")
            if not task_id or task_id in exclude or (include and task_id not in selected):
                continue
            key = TaskKey.from_dict(task_key_for(
                task_id, row.get("repo", ""), row.get("base_commit", ""),
                row.get("test_patch", "") or "", fingerprint))
            keys[task_id] = key
    missing = [task_id for task_id in include if task_id not in keys]
    if missing:
        raise ValueError(f"requested public tasks were not found: {missing}")
    # Preserve the configured task order so the plan and tables are stable.
    return [keys[task_id] for task_id in include]


def _judge_settings(config: Mapping[str, Any], data_root: Path,
                    submission_dir: Path) -> dict[str, Any]:
    runtime = config["runtime"]
    extra_wheels = (Path(runtime["extra_wheels"]).expanduser().resolve()
                    if runtime.get("extra_wheels") else None)
    return {
        "data_root": data_root,
        "submission_dir": submission_dir,
        "baseline_python": runtime.get("sandbox_baseline_python",
                                       "/opt/gemma-sandbox-baseline/bin/python"),
        "sandbox": runtime.get("sandbox", "subprocess"),
        "evaluation_overrides": dict(config["runtime"].get("evaluation", {}) or {}),
        "extra_wheels": str(extra_wheels) if extra_wheels else None,
    }


def _judge_environment(config: Mapping[str, Any], data_root: Path,
                       submission_dir: Path) -> tuple[dict[str, Any], str]:
    settings = _judge_settings(config, data_root, submission_dir)
    environment = read_runtime_identity(
        config["runtime"].get("python", "/opt/gemma-runtime/bin/python"), **settings)
    canonical = canonical_environment(environment)
    return environment, _sha256_bytes(canonical.encode("utf-8"))


def _read_identity_hash(run_dir: Path) -> str | None:
    path = Path(run_dir) / "judge_identity.json"
    try:
        environment = json.loads(path.read_text(encoding="utf-8"))
        return _sha256_bytes(canonical_environment(environment).encode("utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _read_task_key_hashes(db_path: Path) -> dict[str, str]:
    """Read distinct task-key canonical JSON from a closed ledger (read-only)."""
    if not db_path.is_file():
        return {}
    uri = _read_only_uri(db_path)
    if uri is None:
        return {}
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        tables = {str(row[0]) for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "trajectories" not in tables:
            return {}
        rows = connection.execute(
            "SELECT DISTINCT task_key FROM trajectories").fetchall()
    finally:
        connection.close()
    result: dict[str, str] = {}
    for row in rows:
        try:
            key = TaskKey.from_dict(json.loads(row[0]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        result[key.task_id] = key.canonical_json()
    return result


def _reference_summary(reference_dir: Path | None, task_keys: Sequence[TaskKey],
                       identity_hash: str) -> dict[str, Any]:
    if reference_dir is None:
        return {"provided": False}
    reference = reference_dir.expanduser()
    reference_identity = _read_identity_hash(reference)
    reference_keys = _read_task_key_hashes(reference / "spo_stream.db")
    expected = {key.task_id: key.canonical_json() for key in task_keys}
    matched = {task_id: reference_keys.get(task_id) == value
               for task_id, value in expected.items()}
    return {
        "provided": True,
        "run_dir": str(reference),
        "judge_identity_sha256": reference_identity,
        "judge_identity_matches": reference_identity == identity_hash,
        "task_keys_expected": len(expected),
        "task_keys_matched": sum(matched.values()),
        "task_key_mismatches": sorted(task for task, ok in matched.items() if not ok),
    }


# ---------------------------------------------------------------------------
# Plan and preflight
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SampleSlot:
    sequence: int
    task_id: str
    repeat: int
    seed: int
    attempt: int = 0


@dataclass(frozen=True)
class Preflight:
    config: dict[str, Any]
    config_path: Path
    config_sha256: str
    data_root: Path
    model_path: Path
    submission_dir: Path
    run_dir: Path
    baseline_dir: Path
    policy_label: str
    policy_kind: str
    version: int
    behavior: BehaviorAdapter
    adapter: AdapterEvidence | None
    model_identity: dict[str, Any]
    task_keys: tuple[TaskKey, ...]
    judge_environment: dict[str, Any]
    judge_identity_sha256: str
    reference: dict[str, Any]
    repeats: int
    base_seed: int
    slots: tuple[SampleSlot, ...]
    baseline_status: dict[str, Any]


def _stable_seed(base_seed: int, task_id: str, repeat: int) -> int:
    """Pair the same seed across policies for one (task, repeat) sample.

    Different repeats get different seeds; base/v3/v5 use the same per-pair
    seed so a comparison is not accidentally confounded by a seed change.
    """
    digest = hashlib.sha256(
        f"{base_seed}:{task_id}:{repeat}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 2) + 1


def _slots(task_ids: Sequence[str], repeats: int, base_seed: int) -> tuple[SampleSlot, ...]:
    slots: list[SampleSlot] = []
    sequence = 0
    for repeat in range(repeats):
        for task_id in task_ids:
            sequence += 1
            slots.append(SampleSlot(
                sequence=sequence, task_id=task_id, repeat=repeat,
                seed=_stable_seed(base_seed, task_id, repeat)))
    return tuple(slots)


def _ensure_private_baseline_dir(directory: Path) -> dict[str, Any]:
    """Create/chmod the shared baseline root to the store's 0700 contract.

    ``EmptyPatchBaselineStore`` rejects an existing wrong-mode or foreign-owned
    root instead of silently sharing private judge evidence.  Do that check and
    the benign mode repair here before constructing the store.
    """
    directory = directory.expanduser()
    if directory.is_symlink():
        raise PermissionError("baseline directory must not be a symlink")
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = directory.stat()
    if not stat.S_ISDIR(info.st_mode):
        raise PermissionError("baseline root is not a directory")
    if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
        raise PermissionError("baseline root must be owned by the current user")
    mode = stat.S_IMODE(info.st_mode)
    if mode != 0o700:
        directory.chmod(0o700)
        mode = 0o700
    return {"directory": str(directory), "mode": f"{mode:04o}",
            "owner": info.st_uid}


def _baseline_status(directory: Path, keys: Sequence[JudgeTaskKey]) -> dict[str, Any]:
    """Validate the shared cache without creating it or running a probe."""
    status: dict[str, Any] = {
        "directory": str(directory), "exists": directory.is_dir(),
        "expected": len(keys), "cached": 0, "missing": [],
        "invalid": [],
    }
    if directory.is_symlink():
        status["invalid"].append(f"{directory}: baseline root is a symlink")
        status["missing"] = [key.task_id for key in keys]
        return status
    if not directory.is_dir():
        status["missing"] = [key.task_id for key in keys]
        return status
    info = directory.stat()
    status["mode"] = f"{stat.S_IMODE(info.st_mode):04o}"
    status["mode_ok"] = stat.S_IMODE(info.st_mode) == 0o700
    status["owner_ok"] = (not hasattr(os, "geteuid")) or info.st_uid == os.geteuid()
    for key in keys:
        path = directory / f"{key.cache_key}.json"
        if not path.is_file():
            status["missing"].append(key.task_id)
            continue
        try:
            if path.is_symlink():
                raise ValueError("cached baseline must not be a symlink")
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
                raise ValueError("cached baseline must be a regular mode 0600 file")
            if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
                raise ValueError("cached baseline must be owned by the current user")
            record = json.loads(path.read_text(encoding="utf-8"))
            normalized = normalize_baseline(key, record)
            if record.get("content_sha256") != _sha256_bytes(json.dumps(
                    {name: value for name, value in record.items()
                     if name != "content_sha256"},
                    sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")):
                raise ValueError("content_sha256 mismatch")
            if normalized.get("cache_key") != key.cache_key:
                raise ValueError("cache_key mismatch")
        except (OSError, ValueError, TypeError) as error:
            status["invalid"].append(f"{key.task_id}: {type(error).__name__}: {error}")
            continue
        status["cached"] += 1
    return status


def _run_dir_state(run_dir: Path) -> list[str]:
    names = ("judge_identity.json", "spo_stream.db", "eval_policy.json",
             "eval_summary.json", "metrics.jsonl")
    return [name for name in names if (run_dir / name).exists()]


def preflight(args: argparse.Namespace) -> Preflight:
    config_path = Path(args.config).expanduser().resolve(strict=True)
    config = load_eval_config(config_path)
    data_root = Path(args.data_root).expanduser().resolve(strict=True)
    tasks_path = data_root / "tasks.jsonl"
    if not tasks_path.is_file():
        raise ValueError(f"official public tasks missing: {tasks_path}")
    setup_path = data_root / "sandbox" / "setup.py"
    if not setup_path.is_file():
        raise ValueError(f"official sandbox setup missing: {setup_path}")
    submission_dir = Path(
        args.submission_dir or (data_root / "sample_submission")).expanduser().resolve()
    if not submission_dir.is_dir():
        raise ValueError(f"official sample submission directory missing: {submission_dir}")
    model_path = Path(args.model_path or config["policy"]["model_path"]).expanduser() \
        .resolve(strict=True)
    model_identity = validate_model_dir(model_path, config["policy"])
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        raise ValueError("--run-dir must be an absolute path")
    baseline_dir = Path(args.baseline_dir).expanduser() if args.baseline_dir \
        else run_dir.parent / "baselines"
    if not baseline_dir.is_absolute():
        raise ValueError("--baseline-dir must be an absolute path")

    requested = str(args.policy).strip()
    requested_version: int | None = None
    if requested != "base" and requested != "adapter":
        match = POLICY_LABEL.fullmatch(requested)
        if match is None:
            raise ValueError("--policy must be base, adapter, vN or N")
        requested_version = int(match.group(1))
    adapter: AdapterEvidence | None = None
    if requested == "base":
        if args.adapter_dir:
            raise ValueError("base evaluation must not receive --adapter-dir")
        label = "base"
        version = 0
        behavior = BehaviorAdapter(version, BASE_BEHAVIOR_SHA256)
        policy_kind = "base_served_model_no_lora"
    else:
        if not args.adapter_dir:
            raise ValueError(f"--policy {requested} requires --adapter-dir")
        adapter = validate_adapter_dir(
            Path(args.adapter_dir), config["policy"],
            expected_version=requested_version)
        version = adapter.version
        label = f"v{version}"
        behavior = adapter.behavior
        policy_kind = "adapter"

    environment, identity_hash = _judge_environment(
        config, data_root, submission_dir)
    task_keys = tuple(build_task_keys(
        tasks_path, canonical_environment(environment),
        list(config["data"]["task_ids"]),
        list(config["data"].get("exclude_task_ids") or ())))
    # TaskCatalog is used as the public-row gate; it never reads heldout rows.
    catalog = TaskCatalog(
        tasks_path, include_ids=config["data"]["task_ids"],
        exclude_ids=config["data"].get("exclude_task_ids") or ())
    key_by_id = {key.task_id: key for key in task_keys}
    if set(catalog.task_ids) != set(key_by_id):
        raise ValueError("public task catalog and judge TaskKeys differ")
    judge_keys = tuple(JudgeTaskKey.from_dict({
        "task_id": key.task_id, "repo": key.repo, "base_commit": key.base_commit,
        "test_patch_sha256": key.test_patch_sha256,
        "judge_environment_hash": key.judge_environment_hash,
    }) for key in task_keys)

    eval_section = config.get("eval", {})
    repeats = int(args.repeats if args.repeats is not None
                  else eval_section.get("repeats", 3))
    if repeats < 1:
        raise ValueError("--repeats must be positive")
    base_seed = int(args.seed if args.seed is not None
                    else eval_section.get("seed", config["policy"].get("seed", 3407)))
    slots = _slots([key.task_id for key in task_keys], repeats, base_seed)
    reference = _reference_summary(
        Path(args.reference_run_dir) if args.reference_run_dir else None,
        task_keys, identity_hash)
    baseline_status = _baseline_status(baseline_dir, judge_keys)
    if baseline_status["invalid"]:
        raise ValueError("shared baseline cache contains invalid evidence: "
                         + "; ".join(baseline_status["invalid"]))
    return Preflight(
        config=config,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        data_root=data_root,
        model_path=model_path,
        submission_dir=submission_dir,
        run_dir=run_dir,
        baseline_dir=baseline_dir,
        policy_label=label,
        policy_kind=policy_kind,
        version=version,
        behavior=behavior,
        adapter=adapter,
        model_identity=model_identity,
        task_keys=task_keys,
        judge_environment=environment,
        judge_identity_sha256=identity_hash,
        reference=reference,
        repeats=repeats,
        base_seed=base_seed,
        slots=slots,
        baseline_status=baseline_status,
    )


def _print_preflight(pre: Preflight) -> None:
    print("ADAPTER EVALUATION PLAN - read-only preflight")
    print(f"config: {pre.config_path} sha256={pre.config_sha256}")
    print(f"policy: {pre.policy_label} ({pre.policy_kind})")
    print(f"model: {pre.model_path} revision={pre.config['policy'].get('model_revision')}")
    print(f"model_config_sha256: {pre.model_identity['config_json_sha256']}")
    print(f"data_root: {pre.data_root}")
    print(f"submission_dir: {pre.submission_dir}")
    print(f"run_dir: {pre.run_dir}")
    print(f"baseline_dir: {pre.baseline_dir}")
    print(f"judge_identity_sha256: {pre.judge_identity_sha256}")
    print(f"judge packages: {json.dumps(pre.judge_environment.get('packages'), sort_keys=True)}")
    print(f"judge evaluation: {json.dumps(pre.judge_environment.get('evaluation'), sort_keys=True)}")
    print(f"judge setup_sha256: {pre.judge_environment.get('setup_sha256')}")
    print(f"behavior: version={pre.behavior.version} sha256={pre.behavior.sha256}")
    expected_base = pre.config["inference"].get("served_model_name") \
        or pre.config["policy"].get("model_name")
    print(f"base_model_id_source: VLLMBackend.start() reads /v1/models and must match "
          f"configured served_model_name={expected_base!r}")
    print("task_keys:")
    for key in pre.task_keys:
        print(f"  {key.task_id:<20} sha256="
              f"{_sha256_bytes(key.canonical_json().encode('utf-8'))}")
    if pre.adapter is not None:
        print(f"adapter_dir: {pre.adapter.directory}")
        print(f"adapter_manifest: {pre.adapter.manifest_path} sha256={pre.adapter.manifest_sha256}")
        print(f"adapter_weights: sha256={pre.adapter.weights_sha256} bytes={pre.adapter.weights_bytes}")
        print(f"adapter shape: rank={pre.adapter.manifest.rank} alpha={pre.adapter.manifest.alpha} "
              f"dropout={pre.adapter.manifest.dropout} optimizer_step="
              f"{pre.adapter.manifest.optimizer_step}")
    else:
        print("adapter: none (serve the base model directly)")
        print(f"base_behavior: version=0 sha256={BASE_BEHAVIOR_SHA256} "
              f"marker={BASE_BEHAVIOR_MARKER}")
    reference = pre.reference
    if reference.get("provided"):
        print(f"reference_run: {reference['run_dir']}")
        ref_hash = reference.get("judge_identity_sha256")
        if ref_hash is None:
            print("reference_judge_identity_sha256: MISSING (judge_identity.json unreadable)")
        else:
            print(f"reference_judge_identity_sha256: {ref_hash} "
                  f"{'MATCH' if reference['judge_identity_matches'] else 'DIFFERS'}")
        print(f"reference_task_keys: {reference['task_keys_matched']}/"
              f"{reference['task_keys_expected']} exact")
        if reference["task_key_mismatches"]:
            print("reference_task_key_mismatches: "
                  + ", ".join(reference["task_key_mismatches"]))
    else:
        print("reference_run: not supplied (judge identity not compared)")
    status = pre.baseline_status
    print(f"baseline_cache: {status['directory']} cached={status['cached']}/"
          f"{status['expected']} missing={len(status['missing'])}")
    if status.get("exists"):
        print(f"baseline_dir_private: mode={status.get('mode', '-')} "
              f"mode_ok={status.get('mode_ok')} owner_ok={status.get('owner_ok')}")
    if status["invalid"]:
        print("baseline_cache_invalid: " + "; ".join(status["invalid"]))
    print()
    print(f"planned trajectories: {len(pre.slots)} "
          f"({len(pre.task_keys)} tasks x {pre.repeats} repeats)")
    header = f"{'seq':>3}  {'task':<20} {'repeat':>6} {'seed':>10}  behavior"
    print(header)
    print("-" * len(header))
    for slot in pre.slots:
        print(f"{slot.sequence:>3}  {slot.task_id:<20} {slot.repeat:>6} {slot.seed:>10}  "
              f"v{pre.behavior.version}:{pre.behavior.sha256[:16]}")
    state = _run_dir_state(pre.run_dir)
    if state:
        print(f"run_dir_state: existing ({', '.join(state)}); actual eval requires --resume "
              f"or a fresh directory")
    else:
        print("run_dir_state: empty/absent")


# ---------------------------------------------------------------------------
# Eval-only serving and worker instrumentation
# ---------------------------------------------------------------------------
class FixedPolicyServer:
    """Single-policy serving gateway used as ``PolicyEndpoint.serving_backend``.

    ``RemoteServingEngine`` calls ``submit(trajectory_id, ...)`` and
    ``backend.cancel(request_id)``.  This wrapper selects the base or adapter
    path, overrides the sampling seed from the deterministic eval pair, and
    keeps the backend's sampling-evidence contract intact.
    """

    def __init__(self, backend: VLLMBackend, *, policy_label: str,
                 manifest: AdapterManifest | None, base_seed: int):
        self.backend = backend
        self.policy_label = policy_label
        self.manifest = manifest
        self.base_seed = int(base_seed)

    def _seed(self, trajectory_id: str) -> int:
        match = _TRAJECTORY.fullmatch(trajectory_id)
        if match is None:
            raise ValueError(f"unrecognized evaluation trajectory id: {trajectory_id}")
        return _stable_seed(self.base_seed, match.group("task"), int(match.group("repeat")))

    @staticmethod
    def _verify_sampling(result: Mapping[str, Any], submitted: Mapping[str, Any]) -> None:
        returned = result.get("sampling")
        if not isinstance(returned, Mapping):
            raise ValueError("backend dropped the sampling record")
        for key in ("temperature", "top_p", "top_k", "max_new_tokens", "seed"):
            if returned.get(key) != submitted.get(key):
                raise ValueError(f"backend sampling[{key}] does not match the submission")

    def submit(self, trajectory_id: str, input_ids: list[int], *,
               request_id: str | None = None,
               sampling: Mapping[str, Any] | None = None,
               cache_namespace: str | None = None,
               timeout: float | None = None) -> dict[str, Any]:
        record = validate_sampling(sampling)
        record["seed"] = self._seed(trajectory_id)
        if self.manifest is not None:
            from .rollout.gateway import InferenceGateway
            gateway = InferenceGateway(self.backend)
            return gateway.submit(
                input_ids, adapter=self.manifest, sampling=record,
                request_id=request_id, cache_namespace=None, timeout=timeout)
        result = self.backend.submit_base(
            input_ids, request_id=request_id, sampling=record, timeout=timeout)
        self._verify_sampling(result, record)
        return result


class _RecordingOfficialWorker(OfficialWorker):
    """OfficialWorker that exposes the winning raw verdict to the eval driver."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._outcomes: dict[str, dict[str, Any]] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()

    def run(self, assignment: Any, attempt: Any, **kwargs: Any) -> dict[str, Any]:
        trajectory_id = assignment.trajectory_id
        try:
            outcome = super().run(assignment, attempt, **kwargs)
        except BaseException as error:
            with self._lock:
                self._errors[trajectory_id] = f"{type(error).__name__}: {error}"
            raise
        with self._lock:
            self._outcomes[trajectory_id] = dict(outcome)
            self._errors.pop(trajectory_id, None)
        return outcome

    def outcome(self, trajectory_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._outcomes.get(trajectory_id)
            return dict(value) if value is not None else None

    def error(self, trajectory_id: str) -> str | None:
        with self._lock:
            return self._errors.get(trajectory_id)


# ---------------------------------------------------------------------------
# Sample accounting
# ---------------------------------------------------------------------------
def _outcome_parts(outcome: Mapping[str, Any] | None) -> tuple[dict[str, Any], str | None]:
    if not isinstance(outcome, Mapping):
        return {}, None
    result = outcome.get("result")
    raw = dict(result) if isinstance(result, Mapping) else {}
    worker_error = outcome.get("worker_error")
    if not isinstance(worker_error, str) or not worker_error.strip():
        worker_error = None
    return raw, worker_error


def _complete_sample(raw: Mapping[str, Any], reward: RewardRecord | None) -> tuple[bool, str | None]:
    """A repeated sample is complete only with a complete official verdict.

    ``reward.eligible`` is a training-batch property, not an evaluation sample
    property.  A candidate that makes tests uncollectable has a valid official
    verdict with ``validity=0`` and no P; it is still a real model sample and
    must count against the fixed three repeats.  Only infrastructure failures,
    worker errors, missing verdicts and identity mismatches are retried.
    """
    if raw.get("verification_complete") is not True:
        return False, "verdict_incomplete"
    if raw.get("verdict_valid") is not True:
        return False, "verdict_invalid"
    if type(raw.get("success")) is not bool:
        return False, "verdict_success_missing"
    resolved = reward.resolved if reward is not None else raw.get("success")
    if resolved is None:
        return False, "reward_resolved_missing"
    if reward is not None:
        reason = reward.exclusion_reason or ""
        if "task_key_invalid" in reason:
            return False, "task_key_invalid"
    if resolved != raw["success"]:
        return False, "reward_resolution_differs_from_verdict"
    return True, None


def _sample_payload(*, policy: str, slot: SampleSlot, trajectory_id: str,
                    stream_sequence: int, behavior: BehaviorAdapter,
                    raw: Mapping[str, Any], reward: RewardRecord | None,
                    status: str, action_tokens: int, worker_error: str | None,
                    elapsed_seconds: float | None, complete: bool,
                    incomplete_reason: str | None) -> dict[str, Any]:
    return {
        "schema": EVAL_SCHEMA,
        "policy": policy,
        "task_id": slot.task_id,
        "repeat": slot.repeat,
        "sample_attempt": slot.attempt,
        "trajectory_id": trajectory_id,
        "stream_sequence": stream_sequence,
        "behavior_version": behavior.version,
        "behavior_sha256": behavior.sha256,
        "seed": slot.seed,
        "complete": bool(complete),
        "incomplete_reason": incomplete_reason,
        "verification_complete": raw.get("verification_complete"),
        "verdict_valid": raw.get("verdict_valid"),
        "official_success": raw.get("success"),
        "verdict_kind": raw.get("verdict_kind"),
        "infrastructure_error": raw.get("infrastructure_error"),
        "worker_error": worker_error,
        "resolved": reward.resolved if reward is not None else raw.get("success"),
        "P": reward.P if reward is not None else None,
        "G": reward.G if reward is not None else None,
        "progress": reward.progress if reward is not None else None,
        "regression": reward.regression if reward is not None else None,
        "validity": reward.validity if reward is not None else raw.get("validity"),
        "total_reward": reward.total_reward if reward is not None else None,
        "reward_eligible": reward.eligible if reward is not None else False,
        "exclusion_reason": reward.exclusion_reason if reward is not None else None,
        "status": status,
        "action_tokens": action_tokens,
        "elapsed_seconds": elapsed_seconds,
    }


def _record_sample(ledger: TrajectoryLedger, metrics_path: Path,
                   payload: Mapping[str, Any]) -> None:
    with ledger.transaction() as connection:
        ledger._event(connection, EVAL_EVENT, str(payload["trajectory_id"]), payload)
    _emit(metrics_path, EVAL_EVENT, **payload)


def _event_rows(ledger: TrajectoryLedger) -> list[Mapping[str, Any]]:
    with ledger.transaction() as connection:
        rows = connection.execute(
            "SELECT payload FROM audit_events WHERE event_type=? ORDER BY event_id",
            (EVAL_EVENT,)).fetchall()
    result = []
    for row in rows:
        try:
            value = json.loads(row[0])
        except (TypeError, ValueError):
            continue
        if isinstance(value, Mapping):
            result.append(value)
    return result


def _trajectory_ids(ledger: TrajectoryLedger) -> set[str]:
    with ledger.transaction() as connection:
        return {str(row[0]) for row in connection.execute(
            "SELECT trajectory_id FROM trajectories").fetchall()}


# ---------------------------------------------------------------------------
# Actual evaluation
# ---------------------------------------------------------------------------
def _validate_run_dir(pre: Preflight, resume: bool) -> None:
    state = _run_dir_state(pre.run_dir)
    if resume:
        if not (pre.run_dir / "judge_identity.json").is_file() \
                or not (pre.run_dir / "eval_policy.json").is_file():
            raise RuntimeError("--resume requires judge_identity.json and eval_policy.json")
        saved = json.loads((pre.run_dir / "eval_policy.json").read_text(encoding="utf-8"))
        if saved.get("policy") != pre.policy_label:
            raise RuntimeError("--resume policy differs from the recorded eval policy")
        if saved.get("behavior") != pre.behavior.to_dict():
            raise RuntimeError("--resume behavior identity differs from the recorded eval")
        if saved.get("task_ids") != [key.task_id for key in pre.task_keys]:
            raise RuntimeError("--resume task ids differ from the recorded eval")
        if int(saved.get("repeats", -1)) != pre.repeats:
            raise RuntimeError("--resume repeats differ from the recorded eval")
        if saved.get("config_sha256") != pre.config_sha256:
            raise RuntimeError("--resume config file differs from the recorded eval")
        if int(saved.get("base_seed", -1)) != pre.base_seed:
            raise RuntimeError("--resume seed differs from the recorded eval")
        if saved.get("baseline_dir") != str(pre.baseline_dir):
            raise RuntimeError("--resume baseline directory differs from the recorded eval")
        return
    if state:
        raise RuntimeError(
            "run_dir already holds evaluation state; start fresh or use --resume: "
            + ", ".join(state))


def _write_eval_policy(pre: Preflight, *, resolved_base_model: str | None = None,
                       resume: bool) -> None:
    path = pre.run_dir / "eval_policy.json"
    if resume and path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if resolved_base_model and not saved.get("resolved_base_model_id"):
            saved["resolved_base_model_id"] = resolved_base_model
            _atomic_json(path, saved)
        return
    document = {
        "schema": EVAL_SCHEMA,
        "policy": pre.policy_label,
        "policy_kind": pre.policy_kind,
        "behavior": pre.behavior.to_dict(),
        "adapter": pre.adapter.to_dict() if pre.adapter is not None else None,
        "base_behavior_marker": BASE_BEHAVIOR_MARKER if pre.policy_kind != "adapter" else None,
        "resolved_base_model_id": resolved_base_model,
        "expected_served_model_name": pre.config["inference"].get("served_model_name")
        or pre.config["policy"].get("model_name"),
        "task_ids": [key.task_id for key in pre.task_keys],
        "task_keys": [{"task_id": key.task_id, "canonical_sha256":
                       _sha256_bytes(key.canonical_json().encode("utf-8"))}
                      for key in pre.task_keys],
        "repeats": pre.repeats,
        "base_seed": pre.base_seed,
        "slots": [slot.__dict__ for slot in pre.slots],
        "config_path": str(pre.config_path),
        "config_sha256": pre.config_sha256,
        "model_identity": pre.model_identity,
        "judge_identity_sha256": pre.judge_identity_sha256,
        "reference": pre.reference,
        "baseline_dir": str(pre.baseline_dir),
    }
    _atomic_json(path, document)


def _print_sample(payload: Mapping[str, Any]) -> None:
    p = payload.get("P")
    p_text = "-" if p is None else f"{float(p):.3f}"
    validity = payload.get("validity")
    validity_text = "-" if validity is None else f"{float(validity):.3f}"
    resolved = payload.get("resolved")
    resolved_text = "-" if resolved is None else str(bool(resolved))
    print(
        f"{payload['policy']} {payload['task_id']} r{int(payload['repeat']) + 1} "
        f"a{int(payload['sample_attempt']) + 1} "
        f"complete={payload['complete']} resolved={resolved_text} P={p_text} "
        f"validity={validity_text} status={payload['status']}"
        + (f" reason={payload['incomplete_reason']}"
           if payload.get("incomplete_reason") else "")
        + (f" error={payload['worker_error']}"
           if payload.get("worker_error") else ""),
        flush=True)


def run_eval(args: argparse.Namespace) -> int:
    started_monotonic = time.monotonic()
    started_unix = time.time()
    pre = preflight(args)
    if args.plan_only:
        _print_preflight(pre)
        return 0
    _print_preflight(pre)
    print()
    pre.run_dir.mkdir(parents=True, exist_ok=True)
    _validate_run_dir(pre, resume=bool(args.resume))
    environment = pre.judge_environment
    bind_run_identity(pre.run_dir, environment, resume=bool(args.resume))
    _write_eval_policy(pre, resume=bool(args.resume))
    metrics_path = pre.run_dir / "metrics.jsonl"
    _emit(metrics_path, "eval_preflight",
          schema=EVAL_SCHEMA, policy=pre.policy_label,
          policy_kind=pre.policy_kind, behavior=pre.behavior.to_dict(),
          adapter=pre.adapter.to_dict() if pre.adapter else None,
          judge_identity_sha256=pre.judge_identity_sha256,
          task_ids=[key.task_id for key in pre.task_keys],
          repeats=pre.repeats, base_seed=pre.base_seed,
          slots=[slot.__dict__ for slot in pre.slots],
          reference=pre.reference,
          baseline_dir=str(pre.baseline_dir))
    _atomic_json(pre.run_dir / "model_identity.json", pre.model_identity)

    walltime = float(args.walltime_seconds if args.walltime_seconds is not None
                     else pre.config.get("eval", {}).get("walltime_seconds", 14400))
    if not math.isfinite(walltime) or walltime <= 0:
        raise ValueError("--walltime-seconds must be finite and positive")
    max_workers = int(args.max_workers if args.max_workers is not None
                      else pre.config.get("eval", {}).get(
                          "max_workers", pre.config.get("run", {}).get("max_cpu_workers", 6)))
    if not 1 <= max_workers <= 64:
        raise ValueError("--max-workers must be between 1 and 64")
    max_attempts = int(args.max_attempts if args.max_attempts is not None
                       else pre.config.get("eval", {}).get("max_attempts", 2))
    max_sample_attempts = int(
        args.max_sample_attempts if args.max_sample_attempts is not None
        else pre.config.get("eval", {}).get("max_sample_attempts", 3))
    if max_attempts < 1 or max_sample_attempts < 1:
        raise ValueError("attempt caps must be positive")
    grace = float(args.shutdown_grace_seconds)
    if not math.isfinite(grace) or grace < 0:
        raise ValueError("--shutdown-grace-seconds must be finite and non-negative")

    deadline = started_monotonic + walltime
    stop = threading.Event()
    force = threading.Event()
    state = {"signals": 0, "reason": None}

    def handle_signal(signum: int, frame: object) -> None:
        state["signals"] += 1
        state["reason"] = "sigterm"
        stop.set()
        if state["signals"] > 1:
            force.set()

    previous = signal.signal(signal.SIGTERM, handle_signal)
    previous_int = signal.signal(signal.SIGINT, handle_signal)

    backend: VLLMBackend | None = None
    endpoint: PolicyEndpoint | None = None
    execution: ExecutionManager | None = None
    worker: _RecordingOfficialWorker | None = None
    ledger: TrajectoryLedger | None = None
    adapter_loaded = False
    exit_reason = "completed"
    exit_code = 0
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    attempts: Counter[tuple[str, int]] = Counter()
    pending: dict[concurrent.futures.Future, tuple[SampleSlot, str]] = {}
    unfinished_reasons: dict[tuple[str, int], str] = {}
    task_ids = [key.task_id for key in pre.task_keys]
    final_buffer_state: dict[str, int] | None = None
    final_updates: int | None = None
    try:
        # A resumed run skips pairs that already have a complete official sample.
        if args.resume:
            ledger = TrajectoryLedger(pre.run_dir / "spo_stream.db")
            for event in _event_rows(ledger):
                if event.get("task_id") in {key.task_id for key in pre.task_keys} \
                        and event.get("repeat") is not None:
                    pair = (str(event["task_id"]), int(event["repeat"]))
                    attempt = int(event.get("sample_attempt", -1))
                    if attempt >= 0:
                        attempts[pair] = max(attempts[pair], attempt + 1)
                    if event.get("complete") is True:
                        completed.setdefault(pair, dict(event))
            existing_ids = _trajectory_ids(ledger)
        else:
            ledger = TrajectoryLedger(pre.run_dir / "spo_stream.db")
            existing_ids = set()

        backend_url = args.backend_url or pre.config["inference"].get("base_url")
        if not backend_url:
            raise ValueError("inference backend URL is required (--backend-url)")
        backend = VLLMBackend(
            backend_url,
            expected_base_model=pre.config["inference"].get("served_model_name")
            or pre.config["policy"].get("model_name"),
            expected_lora_rank=int(pre.config["policy"]["lora_rank"]),
            max_resident_adapters=int(pre.config["inference"].get("max_loras", 2)),
            lora_name_prefix=str(pre.config["inference"].get("lora_name_prefix", "spo")),
            require_sleep_mode=bool(pre.config["inference"].get("require_sleep_mode", True)),
            require_runtime_lora=bool(pre.config["inference"].get("require_runtime_lora", True)),
            sleep_level=int(pre.config["inference"].get("sleep_level", 2)),
            enable_prefix_cache=bool(pre.config["inference"].get("enable_prefix_cache", False)),
            control_timeout=float(pre.config["inference"].get("control_timeout_seconds", 600)),
        )
        handshake = backend.start()
        _emit(metrics_path, "eval_backend_started", engine=handshake.get("engine"),
              base_model=backend.base_model, deployment=handshake.get("deployment"))
        print(f"vLLM: {backend_url} base_model={backend.base_model!r}", flush=True)
        if pre.adapter is not None:
            loaded = backend.load_adapter(pre.adapter.manifest, pre.adapter.directory)
            adapter_loaded = True
            _emit(metrics_path, "eval_adapter_loaded", **dict(loaded))
            print(f"adapter loaded: {loaded.get('lora_name')}", flush=True)
        else:
            if not backend.base_model:
                raise RuntimeError("vLLM reported no served base model")
            _write_eval_policy(pre, resolved_base_model=backend.base_model, resume=True)
            _emit(metrics_path, "eval_base_model_resolved",
                  base_model=backend.base_model, behavior=pre.behavior.to_dict())

        policy_config = dict(pre.config["policy"])
        policy_config["model_path"] = str(pre.model_path)
        policy_config["defer_learner_load"] = True
        policy_config.setdefault("gradient_checkpointing", True)
        policy_config.setdefault("expected_gpu_count", 2)
        policy = Policy(policy_config)
        serving = FixedPolicyServer(
            backend, policy_label=pre.policy_label,
            manifest=pre.adapter.manifest if pre.adapter else None,
            base_seed=pre.base_seed)
        endpoint = PolicyEndpoint(
            policy, pre.run_dir / "endpoint", int(pre.config["runtime"].get("port", 0)),
            {"max_inflight": int(pre.config["inference"].get("max_inflight", 16)),
             "max_batch_size": int(pre.config["inference"].get("max_inflight", 16)),
             "generation_timeout_seconds": float(
                 pre.config["inference"].get("generation_timeout_seconds", 1800)),
             "temperature": float(pre.config["sampling"].get("temperature", 1.0)),
             "top_p": float(pre.config["sampling"].get("top_p", 1.0)),
             "top_k": int(pre.config["sampling"].get("top_k", 0)),
             "max_new_tokens": int(pre.config["sampling"].get("max_new_tokens", 16384)),
             "seed": pre.base_seed},
            serving_backend=serving)
        endpoint.start()
        endpoint.begin([], pre.version)

        uid_allocator = UidAllocator(int(args.agent_uid_base), 1000000)
        attempt_seq = AttemptSequence(pre.run_dir / "attempt_counter.json")
        runtime = pre.config["runtime"]
        extra_wheels = (Path(runtime["extra_wheels"]).expanduser().resolve()
                        if runtime.get("extra_wheels") else None)
        if extra_wheels is not None and not extra_wheels.is_dir():
            raise ValueError(f"runtime.extra_wheels is missing: {extra_wheels}")
        sandbox_root = pre.run_dir / "sandboxes"
        sandbox_root.mkdir(parents=True, exist_ok=True)
        baseline_root = _ensure_private_baseline_dir(pre.baseline_dir)
        _emit(metrics_path, "eval_baseline_root", **baseline_root)
        baselines = EmptyPatchBaselineStore(
            pre.baseline_dir,
            runner=EmptyPatchProbeRunner(
                data_root=pre.data_root,
                submission_dir=pre.submission_dir,
                runtime_python=runtime.get("python", "/opt/gemma-runtime/bin/python"),
                sandbox_root=sandbox_root,
                baseline_python=runtime.get("sandbox_baseline_python",
                                            "/opt/gemma-sandbox-baseline/bin/python"),
                uid_allocator=uid_allocator,
                attempt_seq=attempt_seq.take,
                workspace_root=pre.run_dir / "probe_workspaces",
                evaluation_overrides=dict(runtime.get("evaluation", {}) or {}),
                extra_wheels=extra_wheels,
                timeout_seconds=min(1800.0, max(60.0, walltime)),
                emit=lambda name, **fields: _emit(metrics_path, name, **fields)))
        reward_engine = RewardEngine(
            baselines,
            objective_version=pre.config["rewards"]["objective_version"],
            efficiency_weight=float(
                pre.config["rewards"].get("success_efficiency_weight", 0.05)))
        tracker = ValueTracker(
            ledger,
            prior_value=float(pre.config["spo"].get("value_prior", 0.5)),
            prior_success=float(pre.config["spo"].get("success_prior", 0.5)),
            prior_strength=float(pre.config["spo"].get("prior_strength", 1.0)),
            kl_half_life=float(pre.config["spo"].get("kl_half_life", 0.05)))
        tracker.purge_orphan_markers()
        tracker.recover_incomplete()
        requests = RequestManager(pre.run_dir / "requests")
        buffer = TrajectoryBuffer(
            ledger,
            target_action_tokens=int(pre.config["spo"]["target_action_tokens"]),
            max_policy_lag=int(pre.config["spo"]["max_policy_lag"]),
            advantage_epsilon=float(pre.config["spo"].get("advantage_epsilon", 1.0e-6)))
        rollout_engine = RolloutEngine(requests, reward_engine, ledger, tracker, buffer)
        worker_config = OfficialWorkerConfig(
            data_root=pre.data_root,
            submission_dir=pre.submission_dir,
            endpoint_url=endpoint.url,
            sandbox_root=sandbox_root,
            baseline_python=Path(runtime.get(
                "sandbox_baseline_python", "/opt/gemma-sandbox-baseline/bin/python")),
            agent_uid_base=int(args.agent_uid_base),
            runtime_python=Path(runtime.get("python", "/opt/gemma-runtime/bin/python")),
            extra_wheels=extra_wheels,
            session_clock=runtime.get("session_clock", "queue_excluded"),
            event_summarizer=runtime.get("event_summarizer", "official"),
            evaluation_overrides=dict(runtime.get("evaluation", {}) or {}))
        worker = _RecordingOfficialWorker(
            worker_config, WorkspaceCache(pre.run_dir / "workspaces",
                                          data_root=pre.data_root),
            uid_allocator=uid_allocator)
        execution = ExecutionManager(
            worker, endpoint, requests, max_workers=max_workers,
            max_attempts=max_attempts,
            task_timeout_seconds=min(3600.0, max(60.0, deadline - time.monotonic())),
            emit=lambda name, **f: _emit(metrics_path, name, **f),
            attempt_seq=attempt_seq.take)

        queue: deque[SampleSlot] = deque()
        task_key_by_id = {key.task_id: key for key in pre.task_keys}
        for slot in pre.slots:
            if (slot.task_id, slot.repeat) not in completed:
                queue.append(slot)
        next_sequence = max([0, ledger.max_stream_sequence()]) + 1
        if args.resume and completed:
            print(f"resume: {len(completed)} complete sample(s) already recorded; "
                  f"{len(queue)} slot(s) remain", flush=True)

        def dispatch(slot: SampleSlot) -> tuple[SampleSlot, str, concurrent.futures.Future]:
            nonlocal next_sequence
            pair = (slot.task_id, slot.repeat)
            attempt = attempts[pair]
            attempts[pair] += 1
            slot = replace(slot, attempt=attempt)
            trajectory_id = (
                f"eval-{pre.policy_label}-{slot.task_id}-r{slot.repeat:02d}"
                f"-a{attempt:02d}-{uuid.uuid4().hex[:12]}")
            while trajectory_id in existing_ids:
                trajectory_id = (
                    f"eval-{pre.policy_label}-{slot.task_id}-r{slot.repeat:02d}"
                    f"-a{attempt:02d}-{uuid.uuid4().hex[:12]}")
            existing_ids.add(trajectory_id)
            sequence = next_sequence
            next_sequence += 1
            task_key = task_key_by_id[slot.task_id]
            # Use the same dispatch protocol as training: a real frozen
            # baseline snapshot and a durable dispatch marker.  The eval has no
            # optimizer, but fabricating a 0.5 baseline would make the ledger's
            # ordered value evidence dishonest and could stall later seals.
            frozen = tracker.freeze_dispatch(task_key, pre.version)
            assignment = RolloutAssignment(
                task_key=task_key,
                trajectory_id=trajectory_id,
                stream_sequence=sequence,
                behavior_adapter=pre.behavior,
                queue="short",
                attempt_id=f"eval-attempt-{uuid.uuid4().hex}",
                selected_at_unix=time.time(),
                provenance={
                    "eval_schema": EVAL_SCHEMA,
                    "eval_policy": pre.policy_label,
                    "eval_policy_kind": pre.policy_kind,
                    "eval_task_id": slot.task_id,
                    "eval_repeat": slot.repeat,
                    "eval_sample_attempt": attempt,
                    "eval_seed": slot.seed,
                    "eval_source_adapter": str(pre.adapter.directory)
                    if pre.adapter else None,
                    "eval_base_behavior_marker": BASE_BEHAVIOR_MARKER
                    if pre.adapter is None else None,
                    "frozen_V": frozen.V,
                    "frozen_success_p": frozen.success_p,
                    "frozen_effective_count": frozen.effective_count,
                    "frozen_kl_cursor": frozen.kl_cursor,
                    "dispatch_policy_version": frozen.policy_version,
                },
                frozen_V=frozen.V, frozen_success_p=frozen.success_p,
                frozen_effective_count=frozen.effective_count,
                frozen_kl_cursor=frozen.kl_cursor,
                dispatch_policy_version=frozen.policy_version,
            )
            try:
                tracker.note_dispatch(task_key, sequence, trajectory_id)
                future = execution.submit(assignment, rollout_engine)
            except BaseException:
                tracker.unmark_dispatch(trajectory_id)
                raise
            _emit(metrics_path, "eval_dispatch",
                  policy=pre.policy_label, task_id=slot.task_id,
                  repeat=slot.repeat, sample_attempt=attempt,
                  trajectory_id=trajectory_id, stream_sequence=sequence,
                  seed=slot.seed, frozen_V=frozen.V,
                  frozen_success_p=frozen.success_p,
                  frozen_effective_count=frozen.effective_count,
                  frozen_kl_cursor=frozen.kl_cursor)
            return slot, trajectory_id, future

        def finish_sample(future: concurrent.futures.Future, slot: SampleSlot,
                          trajectory_id: str, *, allow_requeue: bool) -> None:
            pair = (slot.task_id, slot.repeat)
            trajectory = None
            error_text = worker.error(trajectory_id) if worker is not None else None
            try:
                # A shutdown can cancel an executor task before it ever settles
                # its outer future; never block forever on such a future.
                trajectory = future.result(timeout=0)
            except concurrent.futures.TimeoutError:
                error_text = "cancelled_before_settle"
            except BaseException as error:
                error_text = f"{type(error).__name__}: {error}"
            if trajectory is not None:
                trajectory_id = trajectory.trajectory_id
            outcome = worker.outcome(trajectory_id) if worker is not None else None
            reward = None
            try:
                reward = ledger.get_reward(trajectory_id)
            except (KeyError, ValueError):
                reward = None
            if reward is None:
                # A worker exception can leave a sealed VERIFYING row with no
                # reward.  Exclude it before retrying the same task so it cannot
                # block a later sequence as an unobserved earlier trajectory.
                try:
                    tracker.recover_trajectory(trajectory_id)
                except (ValueError, TypeError, RuntimeError, OSError) as error:
                    print(f"WARNING: could not recover rewardless trajectory {trajectory_id}: "
                          f"{type(error).__name__}: {error}", flush=True)
                try:
                    reward = ledger.get_reward(trajectory_id)
                except (KeyError, ValueError):
                    reward = None
                if trajectory is not None:
                    try:
                        trajectory = ledger.get_trajectory(trajectory_id)
                    except (KeyError, ValueError):
                        pass
            raw, worker_error = _outcome_parts(outcome)
            if error_text and not worker_error:
                worker_error = error_text
            if trajectory is not None:
                complete, incomplete_reason = _complete_sample(raw, reward)
            else:
                complete, incomplete_reason = False, "trajectory_unavailable"
            elapsed = None
            if isinstance(outcome, Mapping) and isinstance(
                    outcome.get("elapsed_seconds"), (int, float)):
                elapsed = float(outcome["elapsed_seconds"])
            payload = _sample_payload(
                policy=pre.policy_label, slot=slot,
                trajectory_id=trajectory_id,
                stream_sequence=trajectory.stream_sequence if trajectory else -1,
                behavior=pre.behavior, raw=raw, reward=reward,
                status=trajectory.status if trajectory else "EXCLUDED",
                action_tokens=trajectory.action_token_count if trajectory else 0,
                worker_error=worker_error,
                elapsed_seconds=elapsed,
                complete=complete, incomplete_reason=incomplete_reason)
            _record_sample(ledger, metrics_path, payload)
            _print_sample(payload)
            if complete:
                if pair in completed:
                    print(f"WARNING: duplicate complete sample for {pair}", flush=True)
                else:
                    completed[pair] = payload
            else:
                unfinished_reasons[pair] = incomplete_reason or "incomplete"
                if allow_requeue and attempts[pair] < max_sample_attempts \
                        and not stop.is_set() and time.monotonic() < deadline:
                    queue.append(slot)
                elif attempts[pair] >= max_sample_attempts:
                    unfinished_reasons[pair] = (
                        incomplete_reason or "incomplete") + " (retry cap reached)"
            # A terminal attempt must not leave a dispatch marker behind: an
            # eligible observation already removed it, while an incomplete or
            # reward-ineligible attempt never observes and would otherwise
            # block a later sequence for the same task until it ages out.
            try:
                tracker.unmark_dispatch(trajectory_id)
            except (ValueError, TypeError, RuntimeError, OSError) as error:
                print(f"WARNING: could not unmark dispatch marker for {trajectory_id}: "
                      f"{type(error).__name__}: {error}", flush=True)

        while queue or pending:
            now = time.monotonic()
            if force.is_set():
                state["reason"] = "sigterm_forced"
                break
            if stop.is_set() or now >= deadline:
                if state["reason"] is None:
                    state["reason"] = "walltime"
                stop.set()
                break
            while queue and len(pending) < max_workers and not stop.is_set() \
                    and time.monotonic() < deadline:
                slot = queue.popleft()
                pair = (slot.task_id, slot.repeat)
                if pair in completed:
                    continue
                if attempts[pair] >= max_sample_attempts:
                    unfinished_reasons.setdefault(pair, "max_sample_attempts_exhausted")
                    continue
                slot, trajectory_id, future = dispatch(slot)
                pending[future] = (slot, trajectory_id)
            if not pending:
                if queue:
                    continue
                break
            timeout = max(0.05, min(5.0, deadline - time.monotonic()))
            done, _ = concurrent.futures.wait(
                pending, timeout=timeout,
                return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                slot, trajectory_id = pending.pop(future)
                finish_sample(future, slot, trajectory_id, allow_requeue=True)
    except BaseException as error:
        exit_code = _EXIT_CONFIG
        exit_reason = f"error: {type(error).__name__}: {error}"
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    finally:
        # Bounded shutdown: stop dispatch, cancel worker subprocesses, then close
        # the endpoint.  The worker tree is terminated by OfficialWorker.run's
        # finally block; ExecutionManager.shutdown prevents orphan executor
        # threads.  Every dispatched attempt is still recorded exactly once.
        finish = locals().get("finish_sample")
        if force.is_set() or stop.is_set():
            print(f"shutdown: {state['reason'] or 'stop'} requested; draining up to "
                  f"{grace:.1f}s", flush=True)
            drain_deadline = time.monotonic() + grace
            while pending and time.monotonic() < drain_deadline and not force.is_set():
                done, _ = concurrent.futures.wait(
                    list(pending), timeout=min(1.0, max(0.05, drain_deadline - time.monotonic())),
                    return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    slot, trajectory_id = pending.pop(future)
                    if callable(finish):
                        finish(future, slot, trajectory_id, allow_requeue=False)
                    else:
                        print(f"drained: {slot.task_id} r{slot.repeat + 1}",
                              flush=True)
            if pending and execution is not None:
                for _future, (slot, trajectory_id) in list(pending.items()):
                    try:
                        execution.cancel(trajectory_id, reason="eval_shutdown")
                    except BaseException:
                        pass
        if execution is not None:
            try:
                execution.shutdown(wait=True, cancel_pending=True)
            except BaseException as error:
                print(f"WARNING: execution shutdown failed: {type(error).__name__}: {error}",
                      flush=True)
        if pending and callable(finish):
            for future, (slot, trajectory_id) in list(pending.items()):
                pending.pop(future, None)
                finish(future, slot, trajectory_id, allow_requeue=False)
        if endpoint is not None:
            try:
                endpoint.close()
            except BaseException as error:
                print(f"WARNING: endpoint close failed: {type(error).__name__}: {error}",
                      flush=True)
        if backend is not None and adapter_loaded and pre.adapter is not None:
            try:
                backend.unload_adapter(pre.adapter.behavior)
            except BaseException as error:
                print(f"WARNING: adapter unload failed: {type(error).__name__}: {error}",
                      flush=True)
        if ledger is not None:
            # The eval never reserves an update.  Record the invariant before
            # closing so a reviewer can see that no training batch consumed it.
            try:
                buffer_state = ledger.buffer_state()
                updates = ledger.list_updates()
                final_buffer_state = dict(buffer_state)
                final_updates = len(updates)
                if updates:
                    print("WARNING: eval ledger contains optimizer updates; this is not "
                          "an evaluation-only run", file=sys.stderr, flush=True)
                _emit(metrics_path, "eval_shutdown",
                      policy=pre.policy_label, reason=state["reason"] or exit_reason,
                      buffer_state=buffer_state, updates=len(updates),
                      completed_samples=len(completed),
                      pending=max(0, len(pre.slots) - len(completed)))
            except BaseException:
                pass
            ledger.close()
        signal.signal(signal.SIGTERM, previous)
        signal.signal(signal.SIGINT, previous_int)

    # Build the summary from the durable events, not only the in-memory dict.
    if ledger is not None:
        try:
            if not ledger.path.exists():
                raise RuntimeError("ledger missing")
        except BaseException:
            pass
    policy_events = _read_events_from_db(pre.run_dir / "spo_stream.db")
    by_pair: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for event in policy_events:
        if event.get("policy") != pre.policy_label:
            continue
        by_pair[(str(event.get("task_id")), int(event.get("repeat", -1)))].append(event)
    complete_pairs = {}
    incomplete_attempts = 0
    for pair, rows in by_pair.items():
        complete_rows = [row for row in rows if row.get("complete") is True]
        incomplete_attempts += sum(1 for row in rows if row.get("complete") is not True)
        if complete_rows:
            complete_pairs[pair] = complete_rows[-1]
    summary = {
        "schema": EVAL_SCHEMA,
        "policy": pre.policy_label,
        "policy_kind": pre.policy_kind,
        "behavior": pre.behavior.to_dict(),
        "adapter": pre.adapter.to_dict() if pre.adapter else None,
        "judge_identity_sha256": pre.judge_identity_sha256,
        "task_ids": [key.task_id for key in pre.task_keys],
        "repeats": pre.repeats,
        "reference": pre.reference,
        "baseline_dir": str(pre.baseline_dir),
        "config_sha256": pre.config_sha256,
        "model_identity": pre.model_identity,
        "slots": len(pre.slots),
        "complete_samples": len(complete_pairs),
        "incomplete_attempts": incomplete_attempts,
        "buffer_state": final_buffer_state,
        "updates": final_updates,
        "unfilled": [
            {"task_id": task_id, "repeat": repeat,
             "reason": unfinished_reasons.get((task_id, repeat), "not_attempted")}
            for task_id in task_ids for repeat in range(pre.repeats)
            if (task_id, repeat) not in complete_pairs
        ],
        "tasks": {
            task_id: {
                "n": sum(1 for (task, _), _row in complete_pairs.items()
                         if task == task_id),
                "resolved": sum(1 for (task, _), row in complete_pairs.items()
                                if task == task_id and row.get("resolved") is True),
                "P_values": [row.get("P") for (task, _), row in complete_pairs.items()
                             if task == task_id and row.get("P") is not None],
                "validity_values": [row.get("validity")
                                    for (task, _), row in complete_pairs.items()
                                    if task == task_id and row.get("validity") is not None],
            }
            for task_id in task_ids
        },
        "started_unix": started_unix,
        "ended_unix": time.time(),
        "walltime_seconds": round(time.monotonic() - started_monotonic, 3),
        "walltime_budget_seconds": walltime,
        "exit_reason": state["reason"] or exit_reason,
    }
    _atomic_json(pre.run_dir / "eval_summary.json", summary)
    print()
    print(f"SUMMARY: {pre.policy_label} complete={len(complete_pairs)}/{len(pre.slots)} "
          f"incomplete_attempts={incomplete_attempts} reason={summary['exit_reason']}")
    for task_id in task_ids:
        n = summary["tasks"][task_id]["n"]
        missing = pre.repeats - n
        print(f"  {task_id:<20} n={n}/{pre.repeats}"
              + (f"  MISSING {missing}" if missing else ""))
    if state["reason"] in {"sigterm", "sigterm_forced", "walltime"}:
        return _EXIT_BUDGET
    if exit_code != 0:
        return exit_code
    if any(summary["tasks"][task_id]["n"] < pre.repeats for task_id in task_ids):
        return _EXIT_INCOMPLETE
    return exit_code


# ---------------------------------------------------------------------------
# Read-only report
# ---------------------------------------------------------------------------
def _is_wal_database(db_path: Path) -> bool:
    try:
        with open(db_path, "rb") as handle:
            header = handle.read(20)
    except OSError:
        return False
    return len(header) >= 20 and header[18] == 2 and header[19] == 2


def _wal_has_frames(db_path: Path) -> bool:
    wal_path = Path(str(db_path) + "-wal")
    try:
        return wal_path.is_file() and wal_path.stat().st_size > 0
    except OSError:
        return True


def _read_only_uri(db_path: Path) -> str | None:
    uri = f"{db_path.as_uri()}?mode=ro"
    if _is_wal_database(db_path):
        if _wal_has_frames(db_path):
            return None
        uri += "&immutable=1"
    return uri


def _read_events_from_db(db_path: Path) -> list[Mapping[str, Any]]:
    if not db_path.is_file():
        return []
    uri = _read_only_uri(db_path)
    if uri is None:
        raise RuntimeError(
            f"database is WAL mode with a non-empty -wal file: {db_path}; "
            "make a consistent copy before reporting")
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        tables = {str(row[0]) for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "audit_events" not in tables:
            return []
        rows = connection.execute(
            "SELECT subject_id,payload FROM audit_events WHERE event_type=? "
            "ORDER BY event_id", (EVAL_EVENT,)).fetchall()
    finally:
        connection.close()
    result: list[Mapping[str, Any]] = []
    for row in rows:
        try:
            value = json.loads(row["payload"])
        except (TypeError, ValueError):
            continue
        if isinstance(value, Mapping):
            result.append(dict(value, trajectory_id=row["subject_id"]))
    return result


@dataclass(frozen=True)
class PolicyReport:
    label: str
    display: str
    run_dir: Path
    identity_hash: str | None
    planned_tasks: tuple[str, ...]
    repeats: int
    samples: tuple[Mapping[str, Any], ...]
    attempts: tuple[Mapping[str, Any], ...]
    error: str | None = None


def _load_policy_report(run_dir: Path, label: str) -> PolicyReport:
    run_dir = run_dir.expanduser()
    display = "base(v0)" if label == "base" else label
    db_path = run_dir / "spo_stream.db"
    if not db_path.is_file():
        return PolicyReport(label, display, run_dir, None, (), 0, (), (),
                            f"ledger not found: {db_path}")
    try:
        events = _read_events_from_db(db_path)
    except (OSError, RuntimeError, sqlite3.Error) as error:
        return PolicyReport(label, display, run_dir, None, (), 0, (), (),
                            f"ledger unreadable: {type(error).__name__}: {error}")
    policy_doc: Mapping[str, Any] = {}
    try:
        policy_doc = json.loads((run_dir / "eval_policy.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        policy_doc = {}
    raw_tasks = policy_doc.get("task_ids", ())
    planned_tasks = tuple(
        str(item) for item in raw_tasks
        if isinstance(item, str)) if isinstance(raw_tasks, (list, tuple)) else ()
    try:
        repeats = int(policy_doc.get("repeats", 3)) if policy_doc else 3
    except (TypeError, ValueError):
        repeats = 3
    recorded_policy = policy_doc.get("policy")
    if isinstance(recorded_policy, str) and recorded_policy != label:
        return PolicyReport(
            label, display, run_dir, _read_identity_hash(run_dir), planned_tasks,
            repeats, (), (), f"recorded policy is {recorded_policy!r}, not {label!r}")
    identity_hash = _read_identity_hash(run_dir)
    if not events:
        # Resilience fallback for a ledger written before the eval event table
        # was introduced: only trajectories with eval provenance are used, and
        # P/resolution come from the durable reward row.
        events = _fallback_events(db_path)
    # Latest event per trajectory; keep every incomplete attempt for the
    # excluded/ineligible footer, and one complete sample per (task, repeat).
    latest: dict[str, Mapping[str, Any]] = {}
    for event in events:
        if event.get("policy") not in (None, label):
            continue
        trajectory_id = str(event.get("trajectory_id") or "")
        if trajectory_id:
            latest[trajectory_id] = event
    attempts = tuple(latest.values())
    complete = [event for event in attempts if event.get("complete") is True]
    by_pair: dict[tuple[str, int], Mapping[str, Any]] = {}
    for event in complete:
        try:
            pair = (str(event["task_id"]), int(event["repeat"]))
        except (KeyError, TypeError, ValueError):
            continue
        by_pair.setdefault(pair, event)
    samples = tuple(by_pair.values())
    if not planned_tasks:
        planned_tasks = tuple(sorted({str(event.get("task_id"))
                                      for event in attempts
                                      if event.get("task_id")}))
    return PolicyReport(label, display, run_dir, identity_hash, planned_tasks,
                        repeats, samples, attempts)


def _fallback_events(db_path: Path) -> list[Mapping[str, Any]]:
    uri = _read_only_uri(db_path)
    if uri is None:
        return []
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        tables = {str(row[0]) for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"trajectories", "rewards"}.issubset(tables):
            return []
        rows = connection.execute(
            "SELECT t.payload AS trajectory_payload, r.payload AS reward_payload, "
            "t.status AS status FROM trajectories t "
            "LEFT JOIN rewards r ON r.trajectory_id=t.trajectory_id").fetchall()
    finally:
        connection.close()
    result: list[Mapping[str, Any]] = []
    for row in rows:
        try:
            trajectory = json.loads(row["trajectory_payload"])
            provenance = trajectory.get("provenance") or {}
        except (TypeError, ValueError, AttributeError):
            continue
        if not isinstance(provenance, Mapping) \
                or provenance.get("eval_schema") != EVAL_SCHEMA:
            continue
        reward = None
        if row["reward_payload"] is not None:
            try:
                reward = RewardRecord.from_dict(json.loads(row["reward_payload"]))
            except (TypeError, ValueError):
                reward = None
        result.append({
            "policy": provenance.get("eval_policy"),
            "task_id": provenance.get("eval_task_id"),
            "repeat": provenance.get("eval_repeat"),
            "sample_attempt": provenance.get("eval_sample_attempt", 0),
            "trajectory_id": trajectory.get("trajectory_id"),
            "resolved": reward.resolved if reward is not None else None,
            "P": reward.P if reward is not None else None,
            "validity": reward.validity if reward is not None else None,
            "complete": reward is not None and reward.resolved is not None,
            "status": row["status"],
            "legacy_inferred": True,
        })
    return result


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rate(resolved: Sequence[bool]) -> float | None:
    return sum(resolved) / len(resolved) if resolved else None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _delta_text(value: float | None) -> str:
    return "-" if value is None else f"{value:+.3f}"


def _p_text(values: Sequence[float], n: int) -> str:
    if not values:
        return f"- [{0}/{n}]" if n else "-"
    return f"{_mean(values):.3f} [{len(values)}/{n}]"


def _print_report(reports: Sequence[PolicyReport], reference: Path | None) -> int:
    present = [report for report in reports if report.error is None]
    print("ADAPTER EVALUATION REPORT - complete official samples only")
    for report in reports:
        print(f"{report.display:<10} {report.run_dir}"
              + (f"  judge_identity={report.identity_hash}"
                 if report.identity_hash else "  judge_identity=-")
              + (f"  ERROR: {report.error}" if report.error else ""))
    identities = {report.identity_hash for report in reports if report.identity_hash}
    ident_ok = bool(identities) and len(identities) == 1 \
        and all(report.identity_hash for report in reports if report.error is None)
    if not ident_ok:
        print("WARNING: policy run judge identities are missing or differ; cross-policy "
              "deltas are suppressed because verdicts are not comparable across judges.")
    eval_key_sets: dict[str, dict[str, str]] = {}
    for report in present:
        eval_key_sets[report.label] = _read_task_key_hashes(
            report.run_dir / "spo_stream.db")
        keys = eval_key_sets[report.label]
        preview = ", ".join(
            f"{task}:{_sha256_bytes(value.encode('utf-8'))[:12]}"
            for task, value in sorted(keys.items()))
        print(f"task_keys {report.display}: {len(keys)} distinct exact key(s)"
              + (f" [{preview}]" if preview else ""))
    if reference is not None:
        reference_hash = _read_identity_hash(reference)
        reference_keys = _read_task_key_hashes(reference / "spo_stream.db")
        if reference_hash is None:
            print(f"reference training run: {reference} (judge_identity.json missing) "
                  "- not compared")
        else:
            print(f"reference training run: {reference} judge_identity={reference_hash} "
                  f"{'MATCH' if ident_ok and reference_hash in identities else 'DIFFERS'}")
            for report in present:
                keys = eval_key_sets.get(report.label, {})
                shared = sorted(set(keys) & set(reference_keys))
                exact = [task for task in shared if keys[task] == reference_keys[task]]
                print(f"reference task keys {report.display}: "
                      f"{len(exact)}/{len(keys)} exact, {len(shared)} shared task id(s)")
            print("NOTE: training outcomes are not merged into this table; this entry "
                  "evaluates published adapters independently.")
    print("NOTE: base(v0) is the unadapted served base model, not the deleted "
          "zero-initialized training v0 adapter.")
    tasks = sorted({task for report in reports for task in report.planned_tasks})
    if not tasks:
        tasks = sorted({str(event.get("task_id")) for report in present
                        for event in report.samples if event.get("task_id")})
    header = (f"{'task':<20}  {'policy':<10}  {'N':>3}  {'resolved':>8}  "
              f"{'rate':>7}  {'P mean [n/N]':>15}")
    print()
    print("TASK TABLE (N = complete official verdicts, not reward.eligible)")
    print(header)
    print("-" * len(header))
    stats: dict[tuple[str, str], tuple[int, list[bool], list[float], int]] = {}
    for task in tasks:
        for report in reports:
            rows = [event for event in report.samples
                    if event.get("task_id") == task]
            resolved = [bool(event["resolved"]) for event in rows
                        if event.get("resolved") is not None]
            p_values = [float(event["P"]) for event in rows
                        if event.get("P") is not None]
            n = len(rows)
            stats[(task, report.label)] = (n, resolved, p_values,
                                           sum(1 for event in rows
                                               if event.get("P") is None))
            rate = _rate(resolved)
            print(f"{task:<20}  {report.display:<10}  {n:>3}  "
                  f"{sum(resolved)}/{len(resolved) if resolved else 0:<5}  "
                  f"{'-' if rate is None else format(rate, '.3f'):>7}  "
                  f"{_p_text(p_values, n):>15}")
    print()
    reference_report = next((report for report in present if report.label == "base"),
                            present[0] if present else None)
    if ident_ok and reference_report is not None:
        print(f"DELTAS vs {reference_report.display} (right - left; "
              "'-' means missing, never zero)")
        delta_header = (f"{'task':<20}  {'arm':<10}  {'d_res':>8}  {'d_P':>8}")
        print(delta_header)
        print("-" * len(delta_header))
        for task in tasks:
            reference = stats.get((task, reference_report.label), (0, [], [], 0))
            ref_rate, ref_p = _rate(reference[1]), _mean(reference[2])
            for report in present:
                if report.label == reference_report.label:
                    continue
                current = stats.get((task, report.label), (0, [], [], 0))
                rate, p_value = _rate(current[1]), _mean(current[2])
                print(f"{task:<20}  {report.display:<10}  "
                      f"{_delta_text(_delta(rate, ref_rate)):>8}  "
                      f"{_delta_text(_delta(p_value, ref_p)):>8}")
        print()
    elif not ident_ok:
        print("DELTAS suppressed: judge identities are missing or differ.")
        print()
    for task in tasks:
        for report in reports:
            n, resolved, p_values, p_missing = stats.get(
                (task, report.label), (0, [], [], 0))
            expected = report.repeats
            if n == 0:
                print(f"WARNING: {task} {report.display} has no complete sample "
                      f"(N=0<2; expected {expected}).")
            elif n < 2:
                print(f"WARNING: {task} {report.display} has N={n} (<2); "
                      f"one sample is not a stable rate.")
            if n < expected:
                print(f"INCOMPLETE: {task} {report.display} has {n}/{expected} "
                      f"complete samples.")
            if p_missing:
                print(f"MISSING P: {task} {report.display} {p_missing}/{n} complete "
                      f"sample(s) have no P; the P mean uses the remaining values only.")
            if n and resolved and all(not value for value in resolved) \
                    and p_values and all(value == 0 for value in p_values):
                print(f"NOTE: {task} {report.display} has resolved=0 and P=0; "
                      f"P=0 is a measured value, not missing.")
    for report in reports:
        if report.error is not None:
            continue
        attempts = len(report.attempts)
        complete = sum(1 for event in report.attempts if event.get("complete") is True)
        excluded = attempts - complete
        print(f"ATTEMPTS {report.display}: total={attempts} complete={complete} "
              f"incomplete/excluded={excluded}")
    print()
    print("LIMITS: the evaluated public tasks are also used in training (in-sample); "
          "three repeats are not a significance test; base(v0) is the unadapted served "
          "model, not a training genesis adapter; P is unavailable when the empty-patch "
          "baseline is missing and is printed as '-'.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_adapters",
        description="Independent rollout+judge evaluation of published adapters.",
        epilog="Use --plan-only for a CPU-only validation; actual rollouts need vLLM.")
    parser.add_argument("--config", type=Path, help="training-compatible TOML config")
    parser.add_argument("--data-root", type=Path,
                        help="official/competition directory containing tasks.jsonl")
    parser.add_argument("--model-path", type=Path,
                        help="base model directory (defaults to policy.model_path)")
    parser.add_argument("--submission-dir", type=Path,
                        help="official sample_submission directory")
    parser.add_argument("--run-dir", type=Path, help="new eval run directory")
    parser.add_argument("--policy",
                        help="base, adapter, or an explicit vN/N (one policy per invocation)")
    parser.add_argument("--adapter-dir", type=Path,
                        help="existing published adapter version directory (read-only)")
    parser.add_argument("--baseline-dir", type=Path,
                        help="shared empty-patch baseline cache (default: <run parent>/baselines)")
    parser.add_argument("--backend-url", help="vLLM OpenAI base URL")
    parser.add_argument("--reference-run-dir", type=Path,
                        help="optional training run whose judge_identity/task keys are compared")
    parser.add_argument("--plan-only", action="store_true",
                        help="validate and print the plan without contacting vLLM")
    parser.add_argument("--resume", action="store_true",
                        help="resume an interrupted eval run (skips complete pairs)")
    parser.add_argument("--report", action="store_true",
                        help="print the per-task report from existing eval run dirs")
    parser.add_argument("--arm", action="append", default=[], metavar="LABEL=RUN_DIR",
                        help="report arm, repeatable (e.g. --arm v6=/path/to/eval-v6)")
    parser.add_argument("--base-run", type=Path, help="alias for --arm base=RUN_DIR")
    parser.add_argument("--v3-run", type=Path, help="alias for --arm v3=RUN_DIR")
    parser.add_argument("--v5-run", type=Path, help="alias for --arm v5=RUN_DIR")
    parser.add_argument("--repeats", type=int,
                        help="repeats per task (study default/configured: 3)")
    parser.add_argument("--max-workers", type=int,
                        help="max in-flight official workers (default: 6)")
    parser.add_argument("--walltime-seconds", type=float,
                        help="whole eval budget (default: 14400)")
    parser.add_argument("--shutdown-grace-seconds", type=float, default=60.0,
                        help="grace for in-flight workers after SIGTERM/walltime")
    parser.add_argument("--max-attempts", type=int,
                        help="OfficialWorker attempts inside one dispatch (default: 2)")
    parser.add_argument("--max-sample-attempts", type=int,
                        help="outer attempts per (task, repeat) (default: 3)")
    parser.add_argument("--agent-uid-base", type=int, default=61000,
                        help="first sandbox UID pair")
    parser.add_argument("--seed", type=int,
                        help="base seed for the fixed per-pair seed recipe")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.report:
        arms: list[tuple[str, Path]] = []
        for label, path in (("base", args.base_run), ("v3", args.v3_run),
                            ("v5", args.v5_run)):
            if path is not None:
                arms.append((label, path))
        for spec in args.arm:
            if "=" not in spec:
                parser.error(f"--arm must be LABEL=RUN_DIR, got {spec!r}")
            label, raw_path = spec.split("=", 1)
            label = label.strip()
            if not label or not _SAFE_LABEL.fullmatch(label):
                parser.error(f"unsafe report arm label: {label!r}")
            arms.append((label, Path(raw_path)))
        if not arms:
            parser.error("--report requires at least one --arm LABEL=RUN_DIR "
                         "(or the base/v3/v5 aliases)")
        labels = [label for label, _ in arms]
        if len(set(labels)) != len(labels):
            parser.error("duplicate report arm label")
        reports = [_load_policy_report(path, label) for label, path in arms]
        return _print_report(reports, args.reference_run_dir)
    for name in ("config", "data_root", "run_dir", "policy"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if str(args.policy).strip() == "base" and args.adapter_dir:
        parser.error("--adapter-dir is not valid for the base policy")
    if str(args.policy).strip() != "base" and not args.adapter_dir:
        parser.error("--adapter-dir is required for an adapter policy")
    try:
        return run_eval(args)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return _EXIT_CONFIG


if __name__ == "__main__":
    raise SystemExit(main())
