"""TRL-free, single-epoch SPO updates backed by :class:`TrajectoryLedger`.

The module deliberately does not import torch, NumPy, TRL, Transformers or
Unsloth at import time.  A constructed ``Policy`` remains the only owner of the
model.  Behavior logprobs are rollout evidence; learner forwards are used only
for current-policy scores and the bounded post-update diagnostic.
"""

from __future__ import annotations

import copy
from contextlib import ExitStack, nullcontext
from dataclasses import replace
import importlib
import math
import random
from typing import Any, Mapping, Sequence

from ..lr_schedule import make_scheduler, resolve_schedule
from ..prefix_tree import ATTENTION_LAYOUT_VERSION, plan_call_groups
from ..trajectory.ledger import StaleTrajectoryError, TrajectoryLedger
from ..trajectory.schema import Trajectory, UpdateBatch, validate_update_batch
from ..trajectory.token_alignment import validate_token_alignment
from .loss import spo_ppo_loss


TRAINER_STATE_VERSION = 1
SPO_KL_ESTIMATOR = "spo_behavior_old_to_new_action_token_k3_v1"
SPO_INTERVAL_KL_ESTIMATOR = "spo_pre_to_post_action_token_k3_v1"
_PHASE_ORDER = {"RESERVED": 0, "LEARNING": 1, "OPTIMIZED": 2, "CONSUMED": 3}


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError("SPOTrainer requires torch only for training/checkpoint operations") from exc


def capture_rng_state(torch_module: Any | None = None) -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU and all visible Torch CUDA RNGs."""
    torch = torch_module if torch_module is not None else _torch()
    np = importlib.import_module("numpy")
    state = np.random.get_state()
    return {
        "schema_version": 1,
        "python": random.getstate(),
        "numpy": {
            "algorithm": state[0], "keys": state[1].tolist(),
            "position": int(state[2]), "has_gauss": int(state[3]),
            "cached_gaussian": float(state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any], torch_module: Any | None = None) -> None:
    """Validate independent RNG probes before replacing process-global state."""
    torch = torch_module if torch_module is not None else _torch()
    np = importlib.import_module("numpy")
    if not isinstance(state, Mapping) or state.get("schema_version") != 1:
        raise ValueError("unsupported RNG checkpoint schema")
    cuda = state.get("torch_cuda")
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not isinstance(cuda, list) or len(cuda) != visible:
        raise ValueError("RNG checkpoint CUDA topology differs from this process")
    cpu = state.get("torch_cpu")
    if not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
        raise ValueError("invalid Torch CPU RNG state")
    for value in cuda:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError("invalid Torch CUDA RNG state")
    data = state["numpy"]
    python_state = tuple(state["python"])
    if isinstance(python_state[1], list):
        python_state = (python_state[0], tuple(python_state[1]), python_state[2])
    random.Random().setstate(python_state)
    numpy_state = (
        data["algorithm"], np.asarray(data["keys"], dtype=np.uint32),
        int(data["position"]), int(data["has_gauss"]), float(data["cached_gaussian"]),
    )
    np.random.RandomState().set_state(numpy_state)
    probe = torch.Generator(device="cpu")
    probe.set_state(cpu.cpu())
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu.cpu())
    if cuda:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda])


def _cpu_state(value: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_state(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_state(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(item, torch) for item in value)
    return copy.deepcopy(value)


def _clone_cpu_tensors(value: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone() if value.device.type == "cpu" else value
    if isinstance(value, dict):
        return {key: _clone_cpu_tensors(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu_tensors(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu_tensors(item, torch) for item in value)
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _same_frozen_batch(left: UpdateBatch, right: UpdateBatch) -> bool:
    fields = (
        "update_id", "policy_before", "trajectory_ids", "raw_advantages",
        "frozen_advantages", "token_mean", "token_std", "total_action_tokens",
        "max_behavior_lag",
    )
    return all(getattr(left, name) == getattr(right, name) for name in fields)


class SPOTrainer:
    """Apply one PPO epoch to a frozen, complete-trajectory ``UpdateBatch``.

    ``config`` accepts the new flat SPO options, or a mapping containing
    ``spo`` and ``optimizer`` sections.  The checkpoint callback is invoked at
    the pre-forward and post-optimizer boundaries, and again after KL/OPTIMIZED.
    It must durably persist the supplied state before returning if crash
    recovery is desired.  Adapter export and ``ledger.commit_update`` remain
    with the session because only the exported file supplies ``policy_after``
    its real SHA256.
    """

    def __init__(self, policy: Any, ledger: TrajectoryLedger, config: Mapping[str, Any] | None = None,
                 *, emit_callback: Any = None, checkpoint_callback: Any = None):
        if not isinstance(ledger, TrajectoryLedger):
            raise TypeError("ledger must be a TrajectoryLedger")
        declared = dict(config or {})
        spo = declared.get("spo", {})
        optimizer = declared.get("optimizer", {})
        if not isinstance(spo, Mapping) or not isinstance(optimizer, Mapping):
            raise ValueError("spo and optimizer config sections must be mappings")
        cfg = {**declared, **spo, **optimizer}
        self.policy = policy
        self.ledger = ledger
        self.emit_callback = emit_callback
        self.checkpoint_callback = checkpoint_callback
        self._declared_config = copy.deepcopy(declared)
        self.config = copy.deepcopy(cfg)

        policy_config = getattr(policy, "config", {})
        if not isinstance(policy_config, Mapping):
            raise TypeError("policy.config must be a mapping")
        if (policy_config.get("lora_rank", 128) != 128
                or policy_config.get("lora_alpha", 256) != 256
                or float(policy_config.get("lora_dropout", 0.0)) != 0.0):
            raise ValueError("SPO requires the fixed rank 128, alpha 256, dropout 0 adapter")
        if not cfg.get("gradient_checkpointing", True) or not bool(
                getattr(policy, "gradient_checkpointing", None)):
            raise ValueError("SPO requires Policy-owned gradient checkpointing")
        if cfg.get("ppo_epochs", 1) != 1:
            raise ValueError("SPOTrainer implements exactly one PPO epoch")
        self.max_policy_lag = _integer(cfg.get("max_policy_lag", 2), "max_policy_lag")
        self.clip_epsilon = float(cfg.get("clip_epsilon", 0.2))
        self.clip_epsilon_high = float(cfg.get("clip_epsilon_high", 0.28))
        if (not math.isfinite(self.clip_epsilon) or not 0.0 <= self.clip_epsilon < 1.0
                or not math.isfinite(self.clip_epsilon_high) or self.clip_epsilon_high < 0.0):
            raise ValueError("invalid PPO clipping configuration")
        self.max_grad_norm = float(optimizer.get("max_grad_norm", cfg.get("max_grad_norm", 1.0)))
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm != 1.0:
            raise ValueError("SPO uses global max_grad_norm=1.0")
        self.learning_rate = float(optimizer.get(
            "learning_rate", cfg.get("learning_rate", 5.0e-6)))
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("SPO requires a positive base learning rate")
        self.weight_decay = float(optimizer.get("weight_decay", cfg.get("weight_decay", 0.0)))
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        total_steps = _integer(
            optimizer.get("total_optimizer_steps", cfg.get("total_optimizer_steps", 1)),
            "total_optimizer_steps", 1,
        )
        self.schedule_contract = resolve_schedule(
            cfg, total_steps=total_steps, learning_rate=self.learning_rate)
        optimizer_type = optimizer.get("type", cfg.get("optimizer_type", "fused_adamw"))
        if optimizer_type != "fused_adamw":
            raise ValueError("SPO requires Fused AdamW")
        self.fused = optimizer.get("fused", cfg.get("fused", True))
        if type(self.fused) is not bool:
            raise ValueError("optimizer.fused must be bool")
        if cfg.get("formal_cuda", False) and not self.fused:
            raise ValueError("formal CUDA training requires fused=True")

        self.prefix_mode = cfg.get("training_prefix_reuse", "tree")
        if self.prefix_mode not in {"tree", "none"}:
            raise ValueError("training_prefix_reuse must be tree or none")
        if cfg.get("training_prefix_version", "exact_token_tree_v1") != "exact_token_tree_v1":
            raise ValueError("unsupported training prefix semantics")
        if cfg.get("training_attention_layout_version", ATTENTION_LAYOUT_VERSION) \
                != ATTENTION_LAYOUT_VERSION:
            raise ValueError("unsupported tree attention layout")
        self.post_update_kl_max_calls = _integer(
            cfg.get("post_update_kl_max_calls", 4), "post_update_kl_max_calls", 1)
        self.policy.training_tree_backend = self.prefix_mode == "tree"

        for name in ("trainable_parameters", "per_token_logprobs", "per_token_logprobs_group",
                     "score_call", "export_adapter_state", "restore_adapter_state", "_for_training"):
            if not callable(getattr(policy, name, None)):
                raise TypeError(f"policy must provide {name}()")
        self._parameters = list(policy.trainable_parameters())
        named = [(name, parameter) for name, parameter in policy.model.named_parameters()
                 if parameter.requires_grad]
        if not self._parameters or len(named) != len(self._parameters) \
                or {id(parameter) for _, parameter in named} != {id(p) for p in self._parameters}:
            raise ValueError("Policy.trainable_parameters must exactly match trainable model parameters")
        self._parameter_names: list[str] = []
        for name, parameter in named:
            if ".lora_" not in name or not parameter.is_floating_point():
                raise ValueError("only floating-point LoRA parameters may train")
            if parameter.device.type == "meta":
                raise ValueError("trainable LoRA parameters cannot be meta tensors")
            if parameter.device.type == "cuda" and not self.fused:
                raise ValueError("CUDA training requires fused=True")
            if ".lora_A." in name and parameter.ndim == 2 and parameter.shape[0] != 128:
                raise ValueError(f"LoRA A rank is not 128: {name}")
            if ".lora_B." in name and parameter.ndim == 2 and parameter.shape[1] != 128:
                raise ValueError(f"LoRA B rank is not 128: {name}")
            self._parameter_names.append(name)

        self.torch: Any | None = None
        self.optimizer: Any | None = None
        self.scheduler: Any | None = None
        self.completed_optimizer_steps = 0
        self._active: dict[str, Any] | None = None
        self._batch: UpdateBatch | None = None
        self._trajectories: tuple[Trajectory, ...] = ()
        self._flat: list[tuple[Trajectory, Any]] = []
        self._calls: list[dict[str, list[int]]] = []
        self._busy = False
        self._checkpoint_safe = True
        self._checkpoint_phase = "initialized"

    @property
    def has_pending_batch(self) -> bool:
        if self._active is None or self._batch is None:
            return False
        return self.ledger.get_update(self._batch.update_id).phase != "CONSUMED"

    @property
    def pending_update_id(self) -> str | None:
        return self._batch.update_id if self.has_pending_batch and self._batch else None

    def _require_torch(self) -> Any:
        if self.torch is None:
            self.torch = _torch()
        return self.torch

    def _initialize_optimizer(self) -> None:
        if self.optimizer is not None:
            return
        torch = self._require_torch()
        if any(parameter.dtype != torch.float32 for parameter in self._parameters):
            raise ValueError("SPO training requires FP32 LoRA parameters")
        # Never silently fall back to foreach/ordinary AdamW on formal runs.
        self.optimizer = torch.optim.AdamW(
            self._parameters, lr=self.learning_rate,
            weight_decay=self.weight_decay, fused=self.fused,
        )
        self.scheduler = make_scheduler(self.optimizer, self.schedule_contract)

    def _ensure_positive_lr(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("optimizer is not initialized")
        rates = [float(group["lr"]) for group in self.optimizer.param_groups]
        if not rates or any(not math.isfinite(rate) or rate <= 0.0 for rate in rates):
            raise RuntimeError("the next SPO update requires a positive scheduled LR")

    def _emit(self, event: str, **payload: Any) -> None:
        if self.emit_callback is not None:
            self.emit_callback(event, **payload)

    def _checkpoint(self, phase: str) -> None:
        self._checkpoint_phase = phase
        if self.checkpoint_callback is None:
            return
        state = self.export_training_state()
        rng = copy.deepcopy(state["rng"])
        try:
            self.checkpoint_callback(state)
        finally:
            # Checkpoint I/O and event serialization must not perturb training RNG.
            restore_rng_state(rng, self._require_torch())
        self._emit("spo_state_exported", phase=phase,
                   optimizer_steps=self.completed_optimizer_steps,
                   update_id=self._batch.update_id if self._batch else None)

    def _check_cancelled(self) -> None:
        event = getattr(self.policy, "abort_generation", None)
        if event is not None and event.is_set():
            raise RuntimeError("run cancelled during SPO update")

    def _autocast(self):
        torch = self._require_torch()
        if any(parameter.device.type == "cuda" for parameter in self._parameters):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _call_payload(call: Any) -> dict[str, list[int]]:
        return {"prompt_ids": list(call.prompt_ids), "completion_ids": list(call.completion_ids)}

    def _load_trajectories(self, batch: UpdateBatch,
                           current_policy_version: int) -> tuple[Trajectory, ...]:
        trajectories = self.ledger.get_trajectories(batch.trajectory_ids)
        validate_update_batch(batch, trajectories)
        for trajectory in trajectories:
            if self.ledger.get_update(batch.update_id).phase != "CONSUMED" \
                    and trajectory.status != "RESERVED":
                raise ValueError("every update trajectory must remain RESERVED until consumption")
            if not trajectory.calls:
                raise ValueError("an SPO trajectory must retain every original call")
            validate_token_alignment(trajectory)
            for expected_index, call in enumerate(trajectory.calls):
                validate_token_alignment(call)
                if call.call_index != expected_index or call.trajectory_id != trajectory.trajectory_id:
                    raise ValueError("call identity/order changed after reservation")
                if call.behavior_adapter != trajectory.behavior_adapter:
                    raise ValueError("a trajectory cannot switch behavior adapter")
                if not call.prompt_ids or not call.completion_ids:
                    raise ValueError("complete prompt and completion IDs are required")
                if any(type(token) is not int or token < 0
                       for token in (*call.prompt_ids, *call.completion_ids)):
                    raise ValueError("behavior token IDs must be nonnegative integers")
                if len(call.prompt_ids) + len(call.completion_ids) > self.policy.max_context:
                    raise ValueError("complete call exceeds context; truncation is forbidden")
                count = len(call.completion_ids)
                if not (count == len(call.behavior_logprobs) == len(call.action_mask)):
                    raise ValueError("behavior IDs, logprobs and action mask must align")
                if any(isinstance(value, bool) or not isinstance(value, (float, int))
                       or not math.isfinite(float(value)) or float(value) > 0.0
                       for value in call.behavior_logprobs):
                    raise ValueError("rollout behavior logprobs must be finite and <= 0")
                if any(mask not in (0, 1) for mask in call.action_mask):
                    raise ValueError("action mask must be binary")
                lag = current_policy_version - call.behavior_adapter.version
                if lag < 0:
                    raise ValueError("current policy predates a behavior adapter")
                if lag > self.max_policy_lag:
                    raise StaleTrajectoryError(batch.update_id, (trajectory.trajectory_id,))
        return trajectories

    def _install_active(self, active: Mapping[str, Any], trajectories: Sequence[Trajectory]) -> None:
        self._active = copy.deepcopy(dict(active))
        self._trajectories = tuple(trajectories)
        self._flat = [(trajectory, call) for trajectory in trajectories for call in trajectory.calls]
        self._calls = [self._call_payload(call) for _, call in self._flat]
        planned = self._plan_training_groups()
        if self._active.get("training_groups") != planned:
            raise ValueError("saved prefix plan differs from the complete ledger batch")
        selected = self._select_kl_calls()
        if self._active.get("kl_call_indices") != selected:
            raise ValueError("saved KL probe differs from the retained behavior-token subset")

    def _plan_training_groups(self) -> list[dict[str, Any]]:
        embeddings = self.policy.model.get_input_embeddings()
        groups = plan_call_groups(
            self._calls, max_context=self.policy.max_context,
            max_group_calls=int(self.config.get("training_prefix_max_calls", 4))
            if self.prefix_mode == "tree" else 1,
            max_tree_tokens=int(self.config.get(
                "training_prefix_max_tokens", self.policy.max_context)),
            max_mask_bytes=int(self.config.get("training_prefix_mask_bytes", 536870912)),
            mask_element_size=max(4, embeddings.weight.element_size()),
            min_shared_tokens=int(self.config.get("training_prefix_min_tokens", 256)),
            attention_query_chunk_size=int(self.config.get(
                "training_prefix_query_chunk_size", 256)),
        )
        if self.prefix_mode == "none":
            groups.sort(key=lambda group: group["call_indices"][0])
        indices = [index for group in groups for index in group["call_indices"]]
        if sorted(indices) != list(range(len(self._calls))):
            raise RuntimeError("prefix plan must retain every call exactly once")
        return groups

    def _select_kl_calls(self) -> list[int]:
        eligible = [index for index, (_, call) in enumerate(self._flat)
                    if call.action_token_count > 0]
        count = min(self.post_update_kl_max_calls, len(eligible))
        if count == 0:
            raise ValueError("post-update KL requires retained behavior action tokens")
        if count == len(eligible):
            return eligible
        if count == 1:
            return [eligible[len(eligible) // 2]]
        return [eligible[position * (len(eligible) - 1) // (count - 1)]
                for position in range(count)]

    def _new_active(self, batch: UpdateBatch, trajectories: Sequence[Trajectory],
                    current_policy_version: int) -> None:
        self._flat = [(trajectory, call) for trajectory in trajectories for call in trajectory.calls]
        self._calls = [self._call_payload(call) for _, call in self._flat]
        active = {
            "update": batch.to_dict(),
            "current_policy_version": current_policy_version,
            "offset": self.completed_optimizer_steps,
            "training_groups": self._plan_training_groups(),
            "kl_call_indices": self._select_kl_calls(),
            "stats": {
                "optimizer_steps": 0,
                "nonzero_lr_optimizer_steps": 0,
                "loss": None,
                "kl_value": None,
                "token_count": 0,
                "estimator": SPO_KL_ESTIMATOR,
                "post_update_kl": None,
            },
        }
        self._install_active(active, trajectories)
        self._batch = batch
        # Freeze the pre-update policy's scores before any optimizer step. The
        # registered drift is K3(pre || post) over exactly this update's
        # version interval; behavior→current stays a separate, explicitly
        # labeled off-policy diagnostic and must never stand in for it.
        self._active["pre_update_logprobs"] = self._capture_pre_update_logprobs()

    def _capture_pre_update_logprobs(self) -> dict[str, list[float]]:
        """Score the KL probe calls with the unstepped policy (training lease)."""
        torch = self._require_torch()
        if self._active is None or self._flat is None:
            raise RuntimeError("no active SPO batch to score")
        captured: dict[str, list[float]] = {}
        for index in self._active["kl_call_indices"]:
            _, call = self._flat[index]
            with torch.no_grad():
                scores = self.policy.score_call(self._call_payload(call), reference=False)
            values = (scores.detach().to(device="cpu", dtype=torch.float64).tolist()
                      if isinstance(scores, torch.Tensor)
                      else [float(value) for value in scores])
            if len(values) != len(call.behavior_logprobs) \
                    or not all(math.isfinite(value) for value in values):
                raise RuntimeError("pre-update scoring must return one finite logprob per behavior token")
            captured[str(index)] = values
        if not captured:
            raise RuntimeError("pre-update KL capture selected no calls")
        return captured

    def _adopt_batch(self, batch: UpdateBatch) -> None:
        self._batch = batch
        if self._active is None:
            raise RuntimeError("no active SPO batch")
        self._active["update"] = batch.to_dict()
        stats = self._active["stats"]
        stats["optimizer_cursor"] = batch.optimizer_cursor
        stats["phase"] = batch.phase

    def _set_phase(self, phase: str, cursor: int) -> None:
        if self._batch is None:
            raise RuntimeError("no active SPO batch")
        # Keep the in-memory checkpoint at the real model boundary even if the
        # separate ledger commit fails; restore reconciles a ledger behind it.
        self._adopt_batch(replace(self._batch, phase=phase, optimizer_cursor=cursor))
        self._checkpoint_safe = True
        persisted = self.ledger.set_update_phase(
            self._batch.update_id, phase, optimizer_cursor=cursor)
        self._adopt_batch(persisted)

    def optimize_batch(self, batch: UpdateBatch | Mapping[str, Any], *,
                       current_policy_version: int | None = None) -> dict[str, Any]:
        """Reserve no data itself: optimize exactly the ledger's frozen batch."""
        if self._busy or not self._checkpoint_safe:
            raise RuntimeError("restore a safe checkpoint before starting SPO")
        if self._active is not None and not self.has_pending_batch:
            self.acknowledge_commit(self._batch.update_id)
        if self._active is not None:
            raise RuntimeError("finish/commit the active update before starting another")
        if not isinstance(batch, UpdateBatch):
            batch = UpdateBatch.from_dict(batch)
        stored = self.ledger.get_update(batch.update_id)
        if stored != batch:
            raise ValueError("supplied UpdateBatch differs from the ledger")
        if stored.phase != "RESERVED" or stored.optimizer_cursor != 0:
            raise RuntimeError("LEARNING/OPTIMIZED updates require checkpoint restore and resume_batch")
        version = stored.policy_before.version if current_policy_version is None \
            else _integer(current_policy_version, "current_policy_version")
        self._busy = True
        try:
            trajectories = self._load_trajectories(stored, version)
            self._new_active(stored, trajectories, version)
            self._initialize_optimizer()
            self._ensure_positive_lr()
            self._set_phase("LEARNING", 0)
            self._checkpoint_safe = True
            self._checkpoint("learning_started")
            self.ledger.assert_update_lag(stored.update_id, version, self.max_policy_lag)
            return self._run_active()
        except BaseException:
            self.policy.model.zero_grad(set_to_none=True)
            raise
        finally:
            self._busy = False
            self.policy._mode = None

    def resume_batch(self) -> dict[str, Any]:
        """Resume LEARNING or redo only KL/commit work after OPTIMIZED."""
        if self._busy or not self._checkpoint_safe or self._active is None or self._batch is None:
            raise RuntimeError("no safe restored SPO batch is pending")
        self._busy = True
        try:
            self._reconcile_ledger()
            return self._run_active()
        except BaseException:
            self.policy.model.zero_grad(set_to_none=True)
            raise
        finally:
            self._busy = False
            self.policy._mode = None

    def _run_active(self) -> dict[str, Any]:
        if self._active is None or self._batch is None:
            raise RuntimeError("no active SPO batch")
        if self._batch.phase == "CONSUMED":
            return self._public_stats()
        if self._batch.phase == "OPTIMIZED" and self._kl_ready():
            return self._public_stats()
        if self._batch.phase == "RESERVED":
            if self._batch.optimizer_cursor != 0:
                raise ValueError("RESERVED batch has an optimizer cursor")
            self._set_phase("LEARNING", 0)
        if self._batch.optimizer_cursor == 0:
            self._ensure_positive_lr()
            self._apply_optimizer_step()
        elif self._batch.optimizer_cursor != 1:
            raise ValueError("one PPO epoch permits optimizer_cursor only 0 or 1")
        # Cursor 1 is durable proof that the optimizer already ran.  Recovery
        # enters only this diagnostic path and can never repeat Adam/scheduler.
        if not self._kl_ready():
            self._checkpoint_safe = False
            try:
                diagnostic = self._measure_post_update_kl()
            finally:
                self._checkpoint_safe = True
            self._active["stats"].update(
                kl_value=diagnostic["kl_value"], token_count=diagnostic["token_count"],
                estimator=diagnostic["estimator"], post_update_kl=diagnostic,
            )
            self._emit("spo_post_update_kl", update_id=self._batch.update_id, **diagnostic)
        if self._batch.phase != "OPTIMIZED":
            self._set_phase("OPTIMIZED", 1)
            self._checkpoint("optimized")
        return self._public_stats()

    def _apply_optimizer_step(self) -> None:
        if self._active is None or self._batch is None:
            raise RuntimeError("no active SPO batch")
        torch = self._require_torch()
        self.policy._for_training()
        self.optimizer.zero_grad(set_to_none=True)
        self._checkpoint_safe = False
        total_loss = 0.0
        groups = self._active["training_groups"]
        global_tokens = self._batch.total_action_tokens
        for group in groups:
            self._check_cancelled()
            members = [self._flat[index] for index in group["call_indices"]]
            calls = [self._calls[index] for index in group["call_indices"]]
            with ExitStack() as lifetime:
                with self._autocast():
                    if self.prefix_mode == "tree":
                        scores = lifetime.enter_context(self.policy.per_token_logprobs_group(
                            calls, compute_entropy=False,
                            attention_query_chunk_size=group.get(
                                "attention_query_chunk_size", 256),
                        ))
                    else:
                        scores = [self.policy.per_token_logprobs(
                            calls[0], compute_entropy=False)]
                    if len(scores) != len(members):
                        raise RuntimeError("Policy scorer dropped an original call occurrence")
                    losses = []
                    for (trajectory, call), scored in zip(members, scores):
                        current = scored[0] if isinstance(scored, (tuple, list)) else scored
                        advantage = self._batch.frozen_advantages[trajectory.trajectory_id]
                        # Every call contributes its local token sum divided by the
                        # SAME global denominator.  Sum group losses; never average
                        # microbatch means or divide by gradient accumulation steps.
                        losses.append(spo_ppo_loss(
                            current, call.behavior_logprobs, call.action_mask, advantage,
                            clip_epsilon=self.clip_epsilon,
                            clip_epsilon_high=self.clip_epsilon_high,
                            global_action_tokens=global_tokens,
                        ))
                    group_loss = torch.stack(losses).sum()
                if not bool(torch.isfinite(group_loss).item()):
                    raise FloatingPointError("non-finite SPO microbatch objective")
                total_loss += float(group_loss.detach().cpu())
                # The native tree/checkpoint context stays open through backward.
                group_loss.backward()
        self._check_cancelled()
        self.ledger.assert_update_lag(
            self._batch.update_id, self._active["current_policy_version"],
            self.max_policy_lag,
        )
        self._ensure_positive_lr()
        rates = [float(group["lr"]) for group in self.optimizer.param_groups]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._parameters, self.max_grad_norm, error_if_nonfinite=True)
        norm = float(grad_norm.detach().cpu())
        if not math.isfinite(norm):
            raise FloatingPointError("non-finite global SPO gradient norm")
        scheduler_before = self.scheduler.state_dict()["last_epoch"]
        self.optimizer.step()
        self.scheduler.step()
        self.completed_optimizer_steps += 1
        self.optimizer.zero_grad(set_to_none=True)
        self._active["stats"].update({
            "optimizer_steps": 1,
            "nonzero_lr_optimizer_steps": 1,
            "loss": total_loss,
            "global_grad_norm": norm,
            "nonzero_gradient": norm > 0.0,
            "learning_rates_used": rates,
            "learning_rate_used": rates[0] if len(set(rates)) == 1 else None,
            "scheduler_index_before": scheduler_before,
            "scheduler_index_after": self.scheduler.state_dict()["last_epoch"],
            "total_optimizer_steps": self.completed_optimizer_steps,
            "optimizer_step": self.completed_optimizer_steps,
            "nonzero_lr_step_semantics": "successful AdamW step at positive LR; not proof of weight change",
        })
        self._set_phase("LEARNING", 1)
        self._checkpoint_safe = True
        self._checkpoint("optimizer_step")

    def _kl_ready(self) -> bool:
        if self._active is None:
            return False
        stats = self._active["stats"]
        value = stats.get("kl_value")
        return (isinstance(value, (float, int)) and not isinstance(value, bool)
                and math.isfinite(float(value)) and value >= 0.0
                and type(stats.get("token_count")) is int and stats["token_count"] > 0)

    def _measure_post_update_kl(self) -> dict[str, Any]:
        """Separate the registered interval drift from the behavior diagnostic.

        The registered ``kl_value`` is K3(pre-update || post-update) over this
        update's exact version interval, using the frozen pre-step scores. The
        behavior→current K3 stays as an explicitly labeled off-policy
        diagnostic: with policy lag the behavior policy may be older than
        pre-update, so it must never stand in for the registered interval.
        Both are bounded sampled-token approximations, not full-vocabulary KL.
        """
        torch = self._require_torch()
        rng = capture_rng_state(torch)
        pre_captured = (self._active or {}).get("pre_update_logprobs") or {}
        total = 0.0
        interval_total = 0.0
        token_count = 0
        indices = list(self._active["kl_call_indices"])
        behavior_versions = set()
        try:
            for index in indices:
                self._check_cancelled()
                _, call = self._flat[index]
                with self._autocast(), torch.no_grad():
                    scores = self.policy.score_call(
                        self._call_payload(call), reference=False)
                if isinstance(scores, torch.Tensor):
                    current = scores.detach().to(device="cpu", dtype=torch.float64)
                else:
                    current = torch.tensor(scores, dtype=torch.float64)
                old = torch.tensor(call.behavior_logprobs, dtype=torch.float64)
                pre_values = pre_captured.get(str(index))
                if pre_values is None or len(pre_values) != len(call.behavior_logprobs):
                    raise RuntimeError(
                        "pre-update KL capture is missing for a probe call; refusing to mislabel drift")
                pre = torch.tensor(pre_values, dtype=torch.float64)
                mask = torch.tensor(call.action_mask, dtype=torch.bool)
                if current.ndim != 1 or current.shape != old.shape or current.shape != pre.shape \
                        or not bool(torch.isfinite(current).all().item()):
                    raise FloatingPointError("invalid post-update Policy logprobs")
                # Tokens were sampled from old.  With delta=log(new/old),
                # expm1(delta)-delta is the K3 estimator of KL(old || new).
                delta = (current - old)[mask]
                terms = torch.expm1(delta) - delta
                if not bool(torch.isfinite(terms).all().item()):
                    raise FloatingPointError("non-finite old-to-new K3 terms")
                values = terms.tolist()
                total = math.fsum((total, math.fsum(values)))
                # Interval drift uses the frozen pre-step scores, so the
                # registered quantity matches [policy_before, policy_after].
                interval_delta = (current - pre)[mask]
                interval_terms = torch.expm1(interval_delta) - interval_delta
                if not bool(torch.isfinite(interval_terms).all().item()):
                    raise FloatingPointError("non-finite pre-to-post K3 terms")
                interval_values = interval_terms.tolist()
                interval_total = math.fsum((interval_total, math.fsum(interval_values)))
                token_count += len(values)
                behavior_versions.add(call.behavior_adapter.version)
            if token_count <= 0:
                raise ValueError("KL probe selected no behavior action tokens")
            value = max(0.0, total / token_count)
            interval_value = max(0.0, interval_total / token_count)
            if not math.isfinite(value) or not math.isfinite(interval_value):
                raise FloatingPointError("non-finite K3 value")
            return {
                "kl_value": interval_value,
                "token_count": token_count,
                "estimator": SPO_INTERVAL_KL_ESTIMATOR,
                "formula": "mean(expm1(logp_post-logp_pre) - (logp_post-logp_pre))",
                "direction": "pre-update policy to post-update policy over this update interval",
                "call_indices": indices,
                "calls_scored": len(indices),
                "behavior_versions": sorted(behavior_versions),
                "aggregation": "uniform over selected action-mask=1 behavior token occurrences",
                "scope": "bounded retained behavior-token subset; not exact full-vocabulary KL",
                "behavior_diagnostic": {
                    "kl_value": value,
                    "estimator": SPO_KL_ESTIMATOR,
                    "formula": "mean(expm1(logp_new-logp_behavior) - (logp_new-logp_behavior))",
                    "direction": "old rollout behavior to optimized policy (off-policy when lagged)",
                    "behavior_versions": sorted(behavior_versions),
                },
            }
        finally:
            restore_rng_state(rng, torch)

    def _public_stats(self) -> dict[str, Any]:
        if self._active is None or self._batch is None:
            raise RuntimeError("no active SPO batch")
        result = copy.deepcopy(self._active["stats"])
        ready = self._kl_ready()
        result.update({
            "update_id": self._batch.update_id,
            "phase": self._batch.phase,
            "optimizer_cursor": self._batch.optimizer_cursor,
            "policy_before": self._batch.policy_before.to_dict(),
            "policy_version_after": self._batch.policy_before.version + 1,
            "optimizer_step": self.completed_optimizer_steps,
            "total_optimizer_steps": self.completed_optimizer_steps,
            "cohort_start_optimizer_step": self._active["offset"],
            "total_action_tokens": self._batch.total_action_tokens,
            "global_action_token_denominator": self._batch.total_action_tokens,
            "ppo_epochs": 1,
            "microbatch_count": len(self._active["training_groups"]),
            "lr_schedule": dict(self.schedule_contract),
            "commit_ready": self._batch.phase == "OPTIMIZED" and ready,
            "adapter_hash_change_required": True,
            "adapter_export": {
                "version": self._batch.policy_before.version + 1,
                "parent_version": self._batch.policy_before.version,
                "optimizer_step": self.completed_optimizer_steps,
            },
            "training_prefix_reuse": {
                "mode": self.prefix_mode,
                "version": "exact_token_tree_v1",
                "attention_layout_version": ATTENTION_LAYOUT_VERSION,
                "groups": copy.deepcopy(self._active["training_groups"]),
                "scope": "exact token paths and per-call loss occurrences; not a measured speedup",
            },
        })
        return result

    def export_training_state(self) -> dict[str, Any]:
        """Export a recoverable boundary, including the active frozen batch."""
        if not self._checkpoint_safe:
            raise RuntimeError("SPO state is not at a recoverable boundary")
        torch = self._require_torch()
        return {
            "schema_version": TRAINER_STATE_VERSION,
            "algorithm": "single_stream_spo",
            "checkpoint_phase": self._checkpoint_phase,
            "trainer_config": copy.deepcopy(self._declared_config),
            "lr_schedule": dict(self.schedule_contract),
            "completed_optimizer_steps": self.completed_optimizer_steps,
            "adapter": self.policy.export_adapter_state(),
            "optimizer_parameter_names": list(self._parameter_names),
            "optimizer": _cpu_state(self.optimizer.state_dict(), torch)
            if self.optimizer is not None else None,
            "scheduler": _cpu_state(self.scheduler.state_dict(), torch)
            if self.scheduler is not None else None,
            "active_batch": copy.deepcopy(self._active),
            "rng": capture_rng_state(torch),
        }

    def _reconcile_ledger(self) -> None:
        if self._active is None or self._batch is None:
            raise RuntimeError("no active SPO batch")
        saved = self._batch
        stored = self.ledger.get_update(saved.update_id)
        if not _same_frozen_batch(saved, stored):
            raise ValueError("ledger frozen advantages/IDs differ from the training checkpoint")
        if stored.phase == "CONSUMED":
            if saved.phase != "OPTIMIZED" or not self._kl_ready():
                raise ValueError("consumed ledger update needs its OPTIMIZED KL checkpoint")
            self._adopt_batch(stored)
            return
        if stored.optimizer_cursor > saved.optimizer_cursor:
            raise RuntimeError("ledger is ahead of the model checkpoint; refusing a duplicate/skipped step")
        if (stored.optimizer_cursor < saved.optimizer_cursor
                or _PHASE_ORDER[stored.phase] < _PHASE_ORDER[saved.phase]):
            if saved.phase == "OPTIMIZED" and not self._kl_ready():
                raise ValueError("OPTIMIZED checkpoint is missing its KL evidence")
            persisted = self.ledger.set_update_phase(
                saved.update_id, saved.phase, optimizer_cursor=saved.optimizer_cursor)
            self._adopt_batch(persisted)
        else:
            # Equal cursor with a later ledger phase is safe: the post-step model
            # is already restored, and missing KL can be measured without Adam.
            self._adopt_batch(stored)

    def restore_training_state(self, state: Mapping[str, Any]) -> None:
        """Restore model/Adam/schedule/RNG and the exact current batch once."""
        if self._busy or self._active is not None or self.optimizer is not None \
                or self.completed_optimizer_steps:
            raise RuntimeError("restore requires a fresh, inactive SPOTrainer")
        if not isinstance(state, Mapping) or state.get("schema_version") != TRAINER_STATE_VERSION \
                or state.get("algorithm") != "single_stream_spo":
            raise ValueError("unsupported SPO trainer checkpoint")
        if state.get("trainer_config") != self._declared_config:
            raise ValueError("checkpoint trainer configuration differs")
        if state.get("lr_schedule") != self.schedule_contract:
            raise ValueError("checkpoint LR schedule differs")
        if state.get("optimizer_parameter_names") != self._parameter_names:
            raise ValueError("checkpoint optimizer parameter ordering differs")
        completed = state.get("completed_optimizer_steps")
        if type(completed) is not int or completed < 0:
            raise ValueError("invalid completed optimizer count")
        torch = self._require_torch()
        self.policy.restore_adapter_state(state["adapter"])
        if state.get("optimizer") is not None:
            self._initialize_optimizer()
            self.optimizer.load_state_dict(state["optimizer"])
            for parameter, values in self.optimizer.state.items():
                self.optimizer.state[parameter] = _clone_cpu_tensors(values, torch)
            if state.get("scheduler") is None:
                raise ValueError("optimizer checkpoint requires scheduler state")
            self.scheduler.load_state_dict(state["scheduler"])
        elif state.get("scheduler") is not None or completed:
            raise ValueError("completed updates require Adam and scheduler state")
        self.completed_optimizer_steps = completed

        active = copy.deepcopy(state.get("active_batch"))
        if active is not None:
            if not isinstance(active, dict) or not isinstance(active.get("stats"), dict):
                raise ValueError("invalid active SPO batch checkpoint")
            batch = UpdateBatch.from_dict(active["update"])
            if batch.optimizer_cursor not in (0, 1):
                raise ValueError("one-epoch checkpoint has an invalid optimizer cursor")
            offset = active.get("offset")
            if type(offset) is not int or offset < 0 \
                    or completed != offset + batch.optimizer_cursor:
                raise ValueError("checkpoint global/batch optimizer cursors disagree")
            version = active.get("current_policy_version")
            if type(version) is not int or version < batch.policy_before.version:
                raise ValueError("invalid active policy version")
            self._batch = batch
            trajectories = self._load_trajectories(batch, version)
            self._install_active(active, trajectories)
            self._batch = batch
            self._reconcile_ledger()
        restore_rng_state(state["rng"], torch)
        self.policy.model.zero_grad(set_to_none=True)
        self.policy._mode = None
        self._checkpoint_phase = str(state.get("checkpoint_phase", "restored"))
        self._checkpoint_safe = True

    def acknowledge_commit(self, update_id: str) -> None:
        """Release a session-committed batch without another optimizer step."""
        if self._batch is None or self._batch.update_id != update_id:
            raise ValueError("active update does not match the committed update")
        if self.ledger.get_update(update_id).phase != "CONSUMED":
            raise RuntimeError("session must commit_update before releasing the batch")
        self._active = None
        self._batch = None
        self._trajectories = ()
        self._flat = []
        self._calls = []
        self._checkpoint_phase = "committed"
        self._checkpoint_safe = True

    optimize = optimize_batch
    resume = resume_batch
    export_state = export_training_state
    restore_state = restore_training_state


def make_spo_trainer(policy: Any, ledger: TrajectoryLedger,
                     config: Mapping[str, Any] | None = None, **callbacks: Any) -> SPOTrainer:
    return SPOTrainer(policy, ledger, config, **callbacks)
