"""TRL GRPOTrainer integration for complete, externally judged tool trajectories.

The official environment owns rollout generation. A Trainer data-loader item is
one frozen cohort, and each optimizer step accumulates its real model calls one
at a time. The token objective uses one explicit behavior correction measure;
TRL Trainer.train owns the optimizer, gradient clipping, scheduler and callbacks. This avoids flattening a
compacted multi-turn session into a token history the policy never actually saw.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import uuid

from .probability_cache import ReferenceProbabilityCache
from .probability_correction import CORRECTION_VERSION, corrected_token_terms, correction_metadata
from .lr_schedule import resolve_schedule, make_scheduler
from .prefix_tree import plan_call_groups


class _ReuseStopped(Exception):
    """A complete cohort needs fresh rollouts before another optimizer update."""


def capture_rng_state(torch_module=None):
    """Snapshot RNGs as primitives/CPU tensors suitable for weights_only load."""
    torch = torch_module
    if torch is None:
        import torch
    import numpy as np

    numpy_state = np.random.get_state()
    return {
        "schema_version": 1,
        "python": random.getstate(),
        "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
                      if torch.cuda.is_available() else [],
    }


def restore_rng_state(state, torch_module=None):
    """Restore a checkpoint's RNGs without reseeding an active random stream."""
    torch = torch_module
    if torch is None:
        import torch
    import numpy as np

    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("Unsupported RNG checkpoint schema")
    cuda = state.get("torch_cuda")
    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not isinstance(cuda, list) or len(cuda) != visible:
        raise ValueError("RNG checkpoint CUDA topology differs from this process")
    cpu = state.get("torch_cpu")
    if not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
        raise ValueError("Invalid CPU Torch RNG state")
    for value in cuda:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError("Invalid CUDA RNG state")
    data = state["numpy"]
    # Validate Python/NumPy using independent generators before changing globals.
    python_state = tuple(state["python"])
    if isinstance(python_state[1], list):
        python_state = (python_state[0], tuple(python_state[1]), python_state[2])
    random.Random().setstate(python_state)
    numpy_state = (data["algorithm"], np.asarray(data["keys"], dtype=np.uint32),
                   int(data["position"]), int(data["has_gauss"]), float(data["cached_gaussian"]))
    np.random.RandomState().set_state(numpy_state)
    probe = torch.Generator(device="cpu")
    probe.set_state(cpu.cpu())
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu.cpu())
    if cuda:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda])


def _trl_implementation():
    """Use the installed pinned TRL class, retaining Unsloth's model kernels.

    Unsloth replaces exported GRPOTrainer methods with generated fused versions.
    Those assume one flattened prompt/completion. Loading the installed official
    module under a private name keeps its original class and our full-prefix
    training lifecycle. The corrected token objective is owned by this adapter.
    """
    if importlib.metadata.version("trl") != "1.15.0":
        raise RuntimeError("This adapter targets the installed TRL 1.15.0 API")
    name = "trl.trainer._official_task_grpo"
    if name not in sys.modules:
        importlib.import_module("trl.trainer")
        path = importlib.metadata.distribution("trl").locate_file("trl/trainer/grpo_trainer.py")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the installed TRL GRPOTrainer implementation")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def make_grpo_trainer(policy, config: dict, output_dir: str | Path, emit_callback=None, checkpoint_callback=None):
    """Construct an actual TRL trainer; importing this module never loads CUDA.

    optimize_batch accepts trajectory dictionaries with calls, advantage,
    task_id and optional reward/reward_components. Each call contains the exact
    prompt_ids, completion_ids and behavior_logprobs collected by the endpoint.
    """
    import torch
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import TrainerCallback

    implementation = _trl_implementation()
    GRPOTrainer = implementation.GRPOTrainer
    GRPOConfig = implementation.GRPOConfig
    cfg = dict(config)
    if cfg.get("probability_correction_version", CORRECTION_VERSION) != CORRECTION_VERSION:
        raise ValueError("Unsupported probability correction semantics")
    cfg["probability_correction_version"] = CORRECTION_VERSION
    prefix_mode = cfg.get("training_prefix_reuse", "tree")
    if prefix_mode not in {"tree", "none"}:
        raise ValueError("training_prefix_reuse must be tree or none")
    if cfg.get("training_prefix_version", "exact_token_tree_v1") != "exact_token_tree_v1":
        raise ValueError("Unsupported training prefix semantics")
    cfg.update(training_prefix_reuse=prefix_mode, training_prefix_version="exact_token_tree_v1")
    policy.training_tree_backend = prefix_mode == "tree"
    if not cfg.get("use_bias_correction_kl", True):
        raise ValueError("Corrected KL requires differentiable current/old weighting")
    repetitions = int(cfg.get("num_iterations", 2))
    generations = int(cfg.get("num_generations", 8))
    beta = float(cfg.get("beta", 0.01))
    correction_cap = float(cfg.get("importance_sampling_cap", cfg.get("behavior_correction_cap", 2.0)))
    correction_enabled = bool(cfg.get("behavior_correction", True))
    correction_description = correction_metadata(correction_cap, correction_enabled)
    epsilon_low, epsilon_high = float(cfg.get("epsilon", 0.2)), float(cfg.get("epsilon_high", 0.28))
    target_kl = cfg.get("target_kl", 0.03)
    if (not math.isfinite(beta) or not 0 <= epsilon_low < 1 or not math.isfinite(epsilon_high)
            or epsilon_high < 0 or (target_kl is not None and not math.isfinite(float(target_kl)))):
        raise ValueError("Nonfinite or unsupported GRPO loss/stop parameters")
    fixed_length = int(cfg.get("max_completion_length", cfg.get("loss_normalizer", 32768)))
    total_optimizer_steps = int(cfg.get("total_optimizer_steps", repetitions))
    schedule_contract = resolve_schedule(cfg, total_steps=total_optimizer_steps,
                                         learning_rate=cfg.get("learning_rate", 5e-6))
    warmup_steps = schedule_contract["warmup_steps"]
    cfg.update(lr_schedule_version=schedule_contract["version"],
               lr_scheduler_type=schedule_contract["type"], warmup_steps=warmup_steps)
    post_update_kl_max_calls = int(cfg.get("post_update_kl_max_calls", 4))
    if repetitions < 1 or generations < 2 or beta < 0 or fixed_length <= 0 or correction_cap < 1:
        raise ValueError("Invalid GRPO repetition/group/KL/normalizer/correction configuration")
    if cfg.get("loss_type", "dr_grpo") != "dr_grpo":
        raise ValueError("This multi-call adapter uses dr_grpo with one fixed trajectory normalizer")
    if total_optimizer_steps < 1 or warmup_steps < 0:
        raise ValueError("GRPO needs positive total optimizer steps and nonnegative warmup")
    if post_update_kl_max_calls < 0:
        raise ValueError("Post-update KL call limit must be nonnegative")
    for name, parameter in policy.model.named_parameters():
        if ".lora_B." in name and bool(torch.count_nonzero(parameter.detach())):
            raise ValueError("The fixed disabled-adapter reference requires a fresh zero-B LoRA at trainer creation")

    def environment_rewards(completions=None, rewards=None, **kwargs):
        if rewards is None:
            raise RuntimeError("Official rewards must arrive with completed external trajectories")
        return rewards

    class OfficialTaskGRPOConfig(GRPOConfig):
        @property
        def place_model_on_device(self):
            # Transformers exposes this integration hook as a property, not
            # a constructor field. Preserve Policy's existing GPU placement.
            return False

    class CohortCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if owner._pending_rng_state is not None:
                restore_rng_state(owner._pending_rng_state, torch)
                owner._pending_rng_state = None
            owner._begin_reuse_step(owner._reuse_start + int(state.global_step))
            owner._synchronize_timing()
            owner._reuse_started_at = time.perf_counter()

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            owner._synchronize_timing()
            rates = [float(group["lr"]) for group in owner.optimizer.param_groups]
            owner._last_step_metrics.update(learning_rates_used=rates,
                                            learning_rate_used=rates[0] if len(set(rates)) == 1 else None,
                                            lr_schedule=schedule_contract,
                                            scheduler_index_before=owner.lr_scheduler.state_dict()["last_epoch"])
            owner._optimizer_started_at = time.perf_counter()

        def on_optimizer_step(self, args, state, control, **kwargs):
            owner._synchronize_timing()
            skipped = bool(owner.accelerator.optimizer_step_was_skipped)
            owner._last_step_metrics.update(
                optimizer_seconds=time.perf_counter() - owner._optimizer_started_at,
                optimizer_step_skipped=skipped,
                nonzero_lr_optimizer_step=not skipped and any(
                    rate != 0.0 for rate in owner._last_step_metrics["learning_rates_used"]),
                zero_lr_optimizer_step=not skipped and all(
                    rate == 0.0 for rate in owner._last_step_metrics["learning_rates_used"]),
            )

        def on_step_end(self, args, state, control, **kwargs):
            completed = owner._reuse_start + int(state.global_step)
            owner._last_step_metrics.update(
                step_wall_seconds=time.perf_counter() - owner._reuse_started_at,
                step_timing_scope="step_begin_to_step_end; includes pre-step checkpoint; excludes post-step checkpoint",
                scheduler_index_after=owner.lr_scheduler.state_dict()["last_epoch"],
            )
            row = {"event": "grpo_optimizer_step", "optimizer_step": owner._offset + completed,
                   "reuse_index": completed, **owner._last_step_metrics}
            owner._step_history.append(row)
            owner._attempt_history[-1].update(owner._last_step_metrics)
            owner._attempt_history[-1]["status"] = ("optimizer_skipped" if row["optimizer_step_skipped"] else
                "optimizer_zero_lr" if row["zero_lr_optimizer_step"] else "optimizer_applied")
            owner._optimizer_measurements["steps"] += 1
            owner._optimizer_measurements["nonzero_lr_steps"] += int(row["nonzero_lr_optimizer_step"])
            owner._optimizer_measurements["skipped_steps"] += int(row["optimizer_step_skipped"])
            owner.completed_optimizer_steps = owner._offset + completed
            owner._checkpoint_safe = True
            for trajectory in owner._cohort:
                for call in trajectory["calls"]:
                    call.pop("_first_gradient_rng", None)
            if emit_callback:
                emit_callback(row["event"], **{key: value for key, value in row.items() if key != "event"})
            owner._checkpoint("optimizer_step")

    class OfficialTaskGRPOTrainer(GRPOTrainer):
        def __init__(self):
            self.policy = policy
            self._cohort = None
            self._training_groups = []
            self._active_call = None
            self._active_scores = None
            self._offset = 0
            self.completed_optimizer_steps = 0
            self._optimizer_measurements = {"steps": 0, "nonzero_lr_steps": 0, "skipped_steps": 0}
            self._update_wall_seconds = 0.0
            self._update_timing_from_start = True
            self._update_started_at = self._reuse_started_at = self._optimizer_started_at = None
            self._step_history = []
            self._last_step_metrics = {}
            self._reuse_reason = None
            self._preupdate_diagnostics = {}
            self._attempt_history = []
            self._reuse_start = 0
            self._pending_rng_state = None
            self._restored_batch = False
            self._checkpoint_safe = True
            self._checkpoint_phase = "idle"
            self._optimizing = False
            self._reference_cache = ReferenceProbabilityCache(
                "disabled-adapter-" + uuid.uuid4().hex,
                max_entries=int(cfg.get("reference_cache_max_entries", 128)),
                max_tokens=int(cfg.get("reference_cache_max_tokens", 1000000)),
            )
            # beta=0 at construction prevents TRL from making a second reference
            # adapter. We supply frozen disabled-adapter probabilities ourselves.
            args = OfficialTaskGRPOConfig(
                output_dir=str(output_dir),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                generation_batch_size=generations,
                num_generations=generations,
                num_iterations=repetitions,
                max_steps=repetitions,
                max_completion_length=fixed_length,
                learning_rate=float(cfg.get("learning_rate", 5e-6)),
                max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
                weight_decay=float(cfg.get("weight_decay", 0.0)),
                lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
                warmup_steps=warmup_steps,
                epsilon=float(cfg.get("epsilon", 0.2)),
                epsilon_high=float(cfg.get("epsilon_high", 0.28)),
                beta=0.0,
                loss_type="dr_grpo",
                scale_rewards=cfg.get("scale_rewards", "group"),
                importance_sampling_level="token",
                use_bias_correction_kl=bool(cfg.get("use_bias_correction_kl", True)),
                use_vllm=False,
                use_liger_kernel=False,
                cast_lm_head_to_fp32=False,
                gradient_checkpointing=False,  # Policy owns the Unsloth checkpoint implementation.
                mask_truncated_completions=False,
                temperature=1.0, top_p=1.0, top_k=0,
                bf16=True, fp16=False,
                remove_unused_columns=False,
                dataloader_num_workers=0,
                report_to=[],
                save_strategy="no",
                eval_strategy="no",
                logging_strategy="steps",
                logging_steps=1,
                disable_tqdm=True,
                seed=int(cfg.get("seed", 3407)),
            )
            super().__init__(model=policy.model, args=args, reward_funcs=[environment_rewards],
                             train_dataset=Dataset.from_dict({"prompt": [""]}),
                             processing_class=policy.tokenizer)
            self.beta = beta
            self.args.beta = beta
            self.model_accepts_loss_kwargs = False
            if self.accelerator.num_processes != 1:
                raise RuntimeError("Use one process with the model split across GPUs, not DDP launch")
            self._optimizer_parameter_names = [name for name, parameter in self.policy.model.named_parameters()
                                               if parameter.requires_grad]
            if not self._optimizer_parameter_names:
                raise RuntimeError("Trainer construction requires the fresh trainable LoRA parameters")

        @property
        def has_pending_batch(self):
            # A checkpoint taken at the final step still needs the caller's
            # batch commit/export, even though no optimizer steps remain.
            return self._restored_batch and self._cohort is not None

        @staticmethod
        def _cpu_state(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {key: OfficialTaskGRPOTrainer._cpu_state(item) for key, item in value.items()}
            if isinstance(value, list):
                return [OfficialTaskGRPOTrainer._cpu_state(item) for item in value]
            if isinstance(value, tuple):
                return tuple(OfficialTaskGRPOTrainer._cpu_state(item) for item in value)
            return copy.deepcopy(value)

        @staticmethod
        def _clone_cpu_tensors(value):
            if isinstance(value, torch.Tensor):
                return value.detach().clone() if value.device.type == "cpu" else value
            if isinstance(value, dict):
                return {key: OfficialTaskGRPOTrainer._clone_cpu_tensors(item) for key, item in value.items()}
            if isinstance(value, list):
                return [OfficialTaskGRPOTrainer._clone_cpu_tensors(item) for item in value]
            if isinstance(value, tuple):
                return tuple(OfficialTaskGRPOTrainer._clone_cpu_tensors(item) for item in value)
            return value

        def _checkpoint(self, phase):
            self._checkpoint_phase = phase
            if checkpoint_callback is not None:
                started = time.perf_counter()
                state = self.export_training_state()
                export_seconds = time.perf_counter() - started
                rng = copy.deepcopy(state["rng"])
                try:
                    checkpoint_callback(state)
                    if emit_callback:
                        emit_callback("grpo_state_exported", phase=phase, state_export_seconds=export_seconds,
                                      optimizer_steps=state["completed_optimizer_steps"])
                finally:
                    # Checkpoint I/O/name allocation must not perturb the
                    # model's random stream relative to uninterrupted training.
                    restore_rng_state(rng, torch)

        def _synchronize_timing(self):
            if torch.cuda.is_available():
                for device in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(device)

        def _current_update_wall_seconds(self):
            return self._update_wall_seconds + (
                time.perf_counter() - self._update_started_at if self._update_started_at is not None else 0.0)

        def _finish_update_timing(self):
            self._update_wall_seconds = self._current_update_wall_seconds()
            self._update_started_at = None

        def _measure_post_update_kl(self):
            """Bounded diagnostic on frozen behavior calls; never an update gate.

            Frozen old probabilities describe the cohort before its first
            optimizer step. They supply the before measurement without another
            forward or a separate snapshot of the last reuse's gradients.
            """
            calls = [call for trajectory in self._cohort for call in trajectory["calls"]]
            count = min(post_update_kl_max_calls, len(calls))
            indices = ([len(calls) // 2] if count == 1 else
                       [index * (len(calls) - 1) // (count - 1) for index in range(count)] if count else [])
            result = {
                "status": "pending", "scope": "cohort_before_first_optimizer_to_after_last_applied_optimizer",
                "probe_scope": "deterministic_evenly_spaced_complete_calls; full_prompt; completion_tokens_only",
                "distribution": correction_description["base_measure"],
                "correction_version": CORRECTION_VERSION,
                "probability_correction": correction_description,
                "limitation": "capped token correction on fixed behavior prefixes; not exact KL, full-cohort or trajectory KL",
                "aggregation": "completion_token_weighted_over_selected_calls",
                "reference_K3_estimator": correction_description["reference_kl"],
                "old_policy_K3_estimator": correction_description["reuse_kl"],
                "call_indices": indices, "calls_available": len(calls), "calls_selected": count,
                "calls_scored": 0, "tokens_selected": sum(len(calls[index]["completion_ids"]) for index in indices),
                "tokens_scored": 0, "cohort_start_optimizer_step": self._offset,
                "final_optimizer_step": self._offset + len(self._step_history),
                "last_applied_optimizer_step": next((row["optimizer_step"] for row in reversed(self._step_history)
                                                      if row.get("optimizer_step_skipped") is False), None),
                "last_nonzero_lr_optimizer_step": next((row["optimizer_step"] for row in reversed(self._step_history)
                                                      if row.get("nonzero_lr_optimizer_step") is True), None),
                "reference_K3": {"before": None, "after": None, "delta": None},
                "old_policy_K3": {"before": None, "after": None, "delta": None},
            }
            if not count or not self._step_history:
                result.update(status="disabled" if not count else "no_optimizer_step", wall_seconds=0.0)
                return result
            if self.policy.abort_generation.is_set():
                raise RuntimeError("Run cancelled before post-update KL diagnostic")
            started = time.perf_counter()
            rng = None
            try:
                self._synchronize_timing()
                rng = capture_rng_state(torch)
                old_after = reference_before = reference_after = weight_sum = weight_sq_sum = clipped_count = 0.0
                reference_available = all("reference_logprobs" in calls[index] for index in indices)
                for index in indices:
                    if self.policy.abort_generation.is_set():
                        raise RuntimeError("Run cancelled during post-update KL diagnostic")
                    call = calls[index]
                    with self.accelerator.autocast(), self.compute_loss_context_manager(), torch.no_grad():
                        scores = self.policy.score_call(call, reference=False)
                    self._forward_invocations["post_update_probe"] = self._forward_invocations.get("post_update_probe", 0) + 1
                    current = torch.tensor(scores, dtype=torch.float64)
                    old = torch.tensor(call["old_learner_logprobs"], dtype=torch.float64)
                    if current.shape != old.shape or not torch.isfinite(current).all():
                        raise FloatingPointError("Post-update diagnostic returned invalid complete-call probabilities")

                    before_terms = self._call_terms(call, old, reference=reference_available)
                    after_terms = self._call_terms(call, current, reference=reference_available)
                    values = [after_terms["old_kl"].sum(), before_terms["weights"].sum(),
                              before_terms["weights"].square().sum(), before_terms["correction_clipped"].sum()]
                    if reference_available:
                        values.extend([before_terms["reference_kl"].sum(), after_terms["reference_kl"].sum()])
                    numbers = [float(value) for value in values]
                    if not all(map(math.isfinite, numbers)):
                        raise FloatingPointError("Corrected post-update diagnostic is nonfinite")
                    old_after += numbers[0]
                    weight_sum += numbers[1]
                    weight_sq_sum += numbers[2]
                    clipped_count += numbers[3]
                    if reference_available:
                        reference_before += numbers[4]
                        reference_after += numbers[5]
                    result["calls_scored"] += 1
                    result["tokens_scored"] += current.numel()
                self._synchronize_timing()
                if self.policy.abort_generation.is_set():
                    raise RuntimeError("Run cancelled during post-update KL diagnostic")
                tokens = result["tokens_scored"]
                result["probability_correction"] = {**correction_description,
                    "weight_mean": weight_sum / tokens, "clipped_token_fraction": clipped_count / tokens,
                    "ess_fraction": weight_sum**2 / max(weight_sq_sum * tokens, 1e-30)}
                result["old_policy_K3"] = {"before": 0.0, "after": old_after / tokens, "delta": old_after / tokens}
                if reference_available:
                    before, after = reference_before / tokens, reference_after / tokens
                    result["reference_K3"] = {"before": before, "after": after, "delta": after - before}
                result.update(status="measured", reference_status="measured" if reference_available else "reference_probabilities_not_recorded")
            except Exception as error:
                if self.policy.abort_generation.is_set():
                    raise
                # A diagnostic cannot discard an already completed optimizer
                # step or prevent adapter export. Partial scores stay unknown.
                result.update(status="unknown", error_type=type(error).__name__, error=str(error))
                if isinstance(error, torch.cuda.OutOfMemoryError) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                if rng is not None:
                    restore_rng_state(rng, torch)
                result["wall_seconds"] = time.perf_counter() - started
            if emit_callback:
                emit_callback("grpo_post_update_kl", **result)
            return result

        def export_training_state(self):
            """Export a consistent step boundary; the caller commits it atomically.

            The frozen-before-first-step boundary intentionally excludes live
            gradients. Stable group plans and group-start RNGs reproduce the
            shared graph with its already frozen old/reference probabilities.
            """
            if (not self._checkpoint_safe or self._active_call is not None
                    or self._active_scores is not None):
                raise RuntimeError("Training state is not at a recoverable checkpoint boundary")
            if torch.cuda.is_available():
                for device in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(device)
            active = None
            if self._cohort is not None:
                active = {
                    "cohort": self._cohort, "offset": self._offset,
                    "training_groups": copy.deepcopy(self._training_groups),
                    "completed_reuses": len(self._step_history),
                    "probabilities_frozen": self._probabilities_frozen,
                    "probability_file": self._probability_path.name,
                    "reference_matches_learner": self._reference_matches_learner,
                    "step_history": self._step_history,
                    "attempt_history": self._attempt_history,
                    "last_step_metrics": self._last_step_metrics,
                    "preupdate_diagnostics": self._preupdate_diagnostics,
                    "forward_invocations": dict(self._forward_invocations),
                    "probability_reuse": dict(self._probability_reuse),
                    "reuse_stop_reason": self._reuse_reason,
                    "update_wall_seconds": self._current_update_wall_seconds(),
                    "update_timing_from_start": self._update_timing_from_start,
                }
            completed = self._offset + len(self._step_history) if active is not None else self.completed_optimizer_steps
            return {
                "schema_version": 1, "checkpoint_phase": self._checkpoint_phase,
                "trainer_config": copy.deepcopy(cfg),
                "lr_schedule": dict(schedule_contract),
                "completed_optimizer_steps": completed,
                "optimizer_measurements": dict(self._optimizer_measurements),
                "adapter": self.policy.export_adapter_state(),
                "optimizer_parameter_names": list(self._optimizer_parameter_names),
                "optimizer": self._cpu_state(self.optimizer.state_dict()) if self.optimizer is not None else None,
                "scheduler": self._cpu_state(self.lr_scheduler.state_dict()) if self.lr_scheduler is not None else None,
                "active_batch": self._cpu_state(active),
                "rng": capture_rng_state(torch),
            }

        def restore_training_state(self, state):
            """Restore only an explicit checkpoint into a fresh zero-B trainer.

            The caller verifies the committed checkpoint's checksum and base
            fingerprints. Ordinary construction still rejects nonzero adapters.
            """
            if self._optimizing or self.optimizer is not None or self.completed_optimizer_steps or self._cohort is not None:
                raise RuntimeError("Restore requires a fresh, inactive trainer")
            if not isinstance(state, dict) or state.get("schema_version") != 1 or state.get("trainer_config") != cfg:
                raise ValueError("Checkpoint trainer configuration differs from this run")
            if state.get("lr_schedule") != schedule_contract:
                raise ValueError("Checkpoint LR schedule semantics differ from this run")
            completed = state.get("completed_optimizer_steps")
            if type(completed) is not int or completed < 0:
                raise ValueError("Invalid checkpoint optimizer-step count")
            # Older checkpoints did not record used LRs; keep that history
            # explicitly unmeasured rather than assuming every step changed weights.
            measurements = state.get("optimizer_measurements", {"steps": 0, "nonzero_lr_steps": 0, "skipped_steps": 0})
            if (not isinstance(measurements, dict) or
                    any(type(measurements.get(key)) is not int or measurements[key] < 0
                        for key in ("steps", "nonzero_lr_steps", "skipped_steps")) or
                    not measurements["nonzero_lr_steps"] + measurements["skipped_steps"] <= measurements["steps"] <= completed):
                raise ValueError("Invalid optimizer measurement counts")
            if state.get("optimizer_parameter_names") != self._optimizer_parameter_names:
                raise ValueError("Checkpoint optimizer parameter ordering differs")
            active = copy.deepcopy(state.get("active_batch"))
            if active is not None:
                if not isinstance(active, dict):
                    raise ValueError("Invalid active-cohort checkpoint")
                offset, done = active.get("offset"), active.get("completed_reuses")
                if (type(offset) is not int or offset < 0 or type(done) is not int or not 0 <= done <= repetitions
                        or completed != offset + done or len(active.get("step_history", [])) != done):
                    raise ValueError("Checkpoint cohort/update counters disagree")
                if not active.get("probabilities_frozen") or not active.get("cohort"):
                    raise ValueError("Active-cohort recovery requires fully frozen probabilities")
                groups = active.get("training_groups")
                count = sum(len(row["calls"]) for row in active["cohort"])
                if (not isinstance(groups, list) or not groups or
                        any(not isinstance(group, dict) or not group.get("call_indices") for group in groups)):
                    raise ValueError("Active recovery requires the original training group plan")
                indices = [index for group in groups for index in group["call_indices"]]
                if any(type(index) is not int for index in indices) or sorted(indices) != list(range(count)):
                    raise ValueError("Training groups must cover every original call exactly once")
                if groups != self._plan_training_groups(active["cohort"]):
                    raise ValueError("Saved training groups differ from the exact token/budget plan")
                filename = active.get("probability_file")
                if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".jsonl"):
                    raise ValueError("Invalid frozen-probability checkpoint filename")
                for trajectory in active["cohort"]:
                    if not math.isfinite(float(trajectory["advantage"])) or not trajectory.get("calls"):
                        raise ValueError("Invalid checkpoint trajectory")
                    for call in trajectory["calls"]:
                        n = len(call["completion_ids"])
                        if not call["prompt_ids"] or not n or len(call["prompt_ids"]) + n > self.policy.max_context:
                            raise ValueError("Invalid complete call in checkpoint")
                        for name in ("behavior_logprobs", "old_learner_logprobs", *( ["reference_logprobs"] if beta else [])):
                            values = call.get(name)
                            if not isinstance(values, list) or len(values) != n or not all(map(math.isfinite, values)):
                                raise ValueError(f"Invalid frozen checkpoint probabilities: {name}")
                        if done == 0 and not isinstance(call.get("_first_gradient_rng"), dict):
                            raise ValueError("Frozen zero-step recovery requires each call's gradient RNG")
            if completed and (state.get("optimizer") is None or state.get("scheduler") is None):
                raise ValueError("Completed updates require both Adam and scheduler state")
            self.policy.restore_adapter_state(state["adapter"])
            if state.get("optimizer") is not None:
                self.create_optimizer()
            if state.get("scheduler") is not None:
                if self.optimizer is None:
                    raise ValueError("Scheduler checkpoint requires an optimizer")
                self.create_scheduler(total_optimizer_steps)
            # Scheduler construction performs its initial step; load Adam's
            # saved param-group LR afterwards so it cannot reset resumed LR.
            if state.get("optimizer") is not None:
                self.optimizer.load_state_dict(state["optimizer"])
                # Adam retains CPU step tensors by identity even for CUDA
                # parameters. Break their mmap backing so a tiny counter does
                # not keep an entire pruned checkpoint file allocated. State
                # already transferred to a GPU needs no further copy.
                for parameter, values in self.optimizer.state.items():
                    self.optimizer.state[parameter] = self._clone_cpu_tensors(values)
            if state.get("scheduler") is not None:
                self.lr_scheduler.load_state_dict(state["scheduler"])
            self.completed_optimizer_steps = completed
            self._optimizer_measurements = dict(measurements)
            self._update_started_at = self._reuse_started_at = self._optimizer_started_at = None
            self._checkpoint_phase = state.get("checkpoint_phase", "restored")
            self._reference_cache.clear()
            if active is not None:
                self._cohort = active["cohort"]
                self._training_groups = active["training_groups"]
                self._offset, self._reuse_start = active["offset"], active["completed_reuses"]
                self._reuse_index = self._reuse_start
                self._probabilities_frozen = True
                self._probability_path = Path(output_dir) / "frozen_probabilities" / active["probability_file"]
                self._reference_matches_learner = active["reference_matches_learner"]
                self._step_history, self._attempt_history = active["step_history"], active["attempt_history"]
                if self._reuse_start == 0 and self._attempt_history:
                    self._attempt_history[-1]["status"] = "gradient_replay_required"
                self._last_step_metrics = active["last_step_metrics"]
                self._preupdate_diagnostics = active["preupdate_diagnostics"]
                self._preupdate_diagnostics["frozen_probabilities_source"] = "checkpoint"
                self._preupdate_diagnostics["frozen_probabilities_path"] = (
                    str(self._probability_path) if self._probability_path.is_file() else None)
                self._forward_invocations = active["forward_invocations"]
                self._probability_reuse = active["probability_reuse"]
                self._reuse_reason = active["reuse_stop_reason"]
                self._update_wall_seconds = float(active.get("update_wall_seconds", 0.0))
                self._update_timing_from_start = bool(active.get("update_timing_from_start", False))
                self._restored_batch = True
                self._pending_rng_state = copy.deepcopy(state["rng"])
            restore_rng_state(state["rng"], torch)
            self.policy.model.zero_grad(set_to_none=True)
            self.policy._mode = None
            self._checkpoint_safe = True

        def get_train_dataloader(self):
            if self._cohort is None:
                raise RuntimeError("Call optimize_batch with completed official trajectories")
            # One item contains one complete cohort. Trainer repeats this one-item
            # epoch mu times; native prompt repetition/generation is bypassed.
            # Dummy-loader iterator seeds must not perturb model/actor RNGs
            # differently after constructing a new Trainer during recovery.
            generator = torch.Generator(device="cpu").manual_seed(0)
            return DataLoader([0], batch_size=1, generator=generator,
                              collate_fn=lambda _: {"official_cohort": True})

        def _prepare_inputs(self, inputs):
            if not isinstance(inputs, dict) or inputs.get("official_cohort") is not True:
                raise ValueError("Expected the complete external official cohort")
            return inputs

        def create_optimizer(self):
            if self.optimizer is None:
                self.optimizer = torch.optim.AdamW(
                    self.policy.trainable_parameters(), lr=self.args.learning_rate,
                    weight_decay=self.args.weight_decay, foreach=False,
                )
            return self.optimizer

        def create_scheduler(self, num_training_steps, optimizer=None):
            if self.lr_scheduler is None:
                self.lr_scheduler = make_scheduler(optimizer or self.optimizer, schedule_contract)
            # Trainer otherwise recreates its scheduler on every train() call.
            self._created_lr_scheduler = False
            return self.lr_scheduler

        def _call_terms(self, call, current, advantage=0.0, *, reference=True):
            return corrected_token_terms(
                current, call["old_learner_logprobs"], call["behavior_logprobs"],
                call.get("reference_logprobs") if reference else None,
                advantage=advantage, cap=correction_cap, enabled=correction_enabled,
                epsilon_low=epsilon_low, epsilon_high=epsilon_high,
            )

        def _plan_training_groups(self, cohort):
            groups = plan_call_groups(
                [call for row in cohort for call in row["calls"]],
                max_context=self.policy.max_context,
                max_group_calls=int(cfg.get("training_prefix_max_calls", 4)) if prefix_mode == "tree" else 1,
                max_tree_tokens=int(cfg.get("training_prefix_max_tokens", self.policy.max_context)),
                max_mask_bytes=int(cfg.get("training_prefix_mask_bytes", 536870912)),
                # Reserve float32 even for BF16 masks: native attention may
                # promote query/mask arithmetic. This is a per-device bound.
                mask_element_size=max(4, self.policy.model.get_input_embeddings().weight.element_size()),
                min_shared_tokens=int(cfg.get("training_prefix_min_tokens", 256)),
            )
            if prefix_mode == "none":
                groups.sort(key=lambda group: group["call_indices"][0])
            return groups

        def _begin_reuse_step(self, local_step):
            if self.policy.abort_generation.is_set():
                raise RuntimeError("Run cancelled before GRPO update")
            if local_step and not self._probabilities_frozen:
                raise RuntimeError("Reuse requires the complete first-step probability snapshot")
            self._reuse_index = local_step

        def _reference_for_call(self, call):
            if not self.beta or self._reference_matches_learner:
                return
            cached = self._reference_cache.get(call["prompt_ids"], call["completion_ids"])
            if cached is not None:
                call["reference_logprobs"] = cached
                call["reference_probability_source"] = "exact_reference_cache"
                self._probability_reuse["reference_cache_hits"] += 1
                return
            # Finish disabled-adapter scoring before creating the current-policy
            # graph. Checkpoint recomputation must always see the active adapter.
            self._forward_invocations["reference"] += 1
            call["reference_logprobs"] = self.policy.score_call(call, reference=True)
            call["reference_probability_source"] = "disabled_adapter_forward"
            self._reference_cache.put(call["prompt_ids"], call["completion_ids"], call["reference_logprobs"])

        def _freeze_from_gradient(self, call, logp):
            # Only numerical denominators detach. The same live logp tensor is
            # used as the numerator and backpropagates through the full prefix.
            call["old_learner_logprobs"] = logp.detach().float().cpu().tolist()
            call["old_probability_source"] = "first_gradient_forward"
            self._probability_reuse["old_from_first_gradient"] += 1
            if self.beta and self._reference_matches_learner:
                call["reference_logprobs"] = list(call["old_learner_logprobs"])
                call["reference_probability_source"] = "zero_adapter_identity"
                self._probability_reuse["reference_from_zero_adapter"] += 1
                self._reference_cache.put(call["prompt_ids"], call["completion_ids"], call["reference_logprobs"])

        def _publish_frozen_probabilities(self):
            count = 0
            delta_sum = delta_abs_max = weight_sum = weight_sq_sum = clipped_count = 0.0
            for trajectory in self._cohort:
                for call in trajectory["calls"]:
                    old = torch.tensor(call["old_learner_logprobs"], dtype=torch.float64)
                    behavior = torch.tensor(call["behavior_logprobs"], dtype=torch.float64)
                    delta = old - behavior
                    terms = self._call_terms(call, old, reference=False)
                    weights = terms["weights"]
                    clipped_count += float(terms["correction_clipped"].sum())
                    count += delta.numel()
                    delta_sum += float(delta.sum())
                    delta_abs_max = max(delta_abs_max, float(delta.abs().max()))
                    weight_sum += float(weights.sum())
                    weight_sq_sum += float(weights.square().sum())
            path = self._probability_path
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                                 prefix=".probabilities-", delete=False) as output:
                    temporary = Path(output.name)
                    for trajectory in self._cohort:
                        for i, call in enumerate(trajectory["calls"]):
                            output.write(json.dumps({
                                "rollout_id": trajectory.get("rollout_id"), "task_id": trajectory.get("task_id"),
                                "call_index": call.get("call_index", i), "policy_version": call.get("policy_version"),
                                "old_learner_logprobs": call["old_learner_logprobs"],
                                "reference_logprobs": call.get("reference_logprobs"),
                                "old_probability_source": call["old_probability_source"],
                                "reference_probability_source": call.get("reference_probability_source"),
                            }) + "\n")
                # Atomic publication without overwriting another attempt's record.
                os.link(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            self._preupdate_diagnostics = {
                "old_vs_behavior_logprob_mean": delta_sum / count,
                "old_vs_behavior_logprob_abs_max": delta_abs_max,
                "behavior_correction_ess_fraction": weight_sum**2 / max(weight_sq_sum * count, 1e-30),
                "probability_correction": {**correction_description,
                    "weight_mean": weight_sum / count, "clipped_token_fraction": clipped_count / count,
                    "ess_fraction": weight_sum**2 / max(weight_sq_sum * count, 1e-30)},
                "generated_tokens": count, "frozen_probabilities_path": str(path),
                "old_probability_source": "first_gradient_forward",
                "reference_matches_learner": self._reference_matches_learner,
            }
            self._probabilities_frozen = True
            if emit_callback:
                emit_callback("grpo_probabilities_frozen", **self._preupdate_diagnostics)

        def _restore_attempt_metrics(self, lengths):
            for key in list(self._metrics["train"]):
                if key not in lengths:
                    del self._metrics["train"][key]
                else:
                    del self._metrics["train"][key][lengths[key]:]

        def training_step(self, model, inputs, num_items_in_batch=None):
            self._prepare_inputs(inputs)
            self.policy._for_training()
            self._synchronize_timing()
            gradient_started_at = time.perf_counter()
            self.current_gradient_accumulation_steps = 1
            local_step = self._reuse_index
            self._checkpoint_safe = False
            calibrating = local_step == 0 and not self._probabilities_frozen
            lengths = {key: len(value) for key, value in self._metrics["train"].items()}
            total_loss = kl_sum = reference_sum = weight_sum = weight_sq_sum = clipped_count = 0.0
            log_ratio_max = -math.inf
            tokens = calls_forward = calls_backward = 0
            model_forwards = backward_groups = processed_input_tokens = independent_input_tokens = 0
            cohort_tokens = sum(len(call["completion_ids"]) for row in self._cohort for call in row["calls"])
            metric_sums = {"entropy": 0.0, "clip_ratio/region_mean": 0.0,
                           "clip_ratio/low_mean": 0.0, "clip_ratio/high_mean": 0.0}
            attempt = {"reuse_index": local_step + 1, "status": "accumulating",
                       "correction_version": CORRECTION_VERSION,
                       "probability_correction": correction_description,
                       "cohort_completion_tokens": cohort_tokens,
                       "gradient_forwards": 0, "backwards": 0}
            self._attempt_history.append(attempt)
            try:
                flat = [(trajectory, call) for trajectory in self._cohort for call in trajectory["calls"]]
                for group in self._training_groups:
                    members = [flat[index] for index in group["call_indices"]]
                    group_calls = [call for _, call in members]
                    if self.policy.abort_generation.is_set():
                        raise RuntimeError("Run cancelled during GRPO gradient accumulation")
                    scores = group_loss = logp = entropy = loss = terms = None
                    try:
                        with ExitStack() as lifetime:
                            with self.accelerator.autocast(), self.compute_loss_context_manager():
                                if calibrating:
                                    # Reference must finish before any live shared
                                    # learner graph, including checkpoint replay.
                                    for call in group_calls:
                                        self._reference_for_call(call)
                                    group_rng = capture_rng_state(torch)
                                    for call in group_calls:
                                        call["_first_gradient_rng"] = group_rng
                                elif local_step == 0:
                                    restore_rng_state(group_calls[0]["_first_gradient_rng"], torch)
                                self._forward_invocations["gradient"] += 1
                                model_forwards += 1
                                calls_forward += len(group_calls)
                                processed_input_tokens += group["unique_tokens"]
                                independent_input_tokens += group["input_tokens"]
                                if prefix_mode == "tree":
                                    scores = lifetime.enter_context(self.policy.per_token_logprobs_group(
                                        group_calls, compute_entropy=True))
                                else:
                                    scores = [self.policy.per_token_logprobs(group_calls[0], compute_entropy=True)]
                                if len(scores) != len(group_calls):
                                    raise RuntimeError("Shared scorer must preserve every original call")
                                losses = []
                                for (trajectory, call), (logp, entropy) in zip(members, scores):
                                    if self.policy.abort_generation.is_set():
                                        raise RuntimeError("Run cancelled while scoring a training tree")
                                    if calibrating:
                                        self._freeze_from_gradient(call, logp)
                                    self._active_call, self._active_scores = call, (logp, entropy)
                                    terms = self._call_terms(call, logp, trajectory["advantage"])
                                    reference = terms["reference_kl"]
                                    per_token_loss = terms["policy_loss"]
                                    if self.beta:
                                        per_token_loss = per_token_loss + self.beta * reference
                                    # One fixed Dr.GRPO denominator per trajectory,
                                    # summed across its original calls. No padding-mask
                                    # hack, duplicated correction, or extra model pass.
                                    loss = per_token_loss.sum() / (fixed_length * len(self._cohort))
                                    values = torch.stack([
                                        terms["old_kl"].sum(),
                                        reference.sum() if reference is not None else logp.new_zeros(()),
                                        terms["weights"].sum(), terms["weights"].square().sum(),
                                        terms["correction_clipped"].sum(), terms["log_ratio"].max(),
                                        entropy.detach().double().sum(), terms["clipped"].sum(),
                                        (terms["ratio_lower"] & terms["clipped"]).sum(),
                                        (terms["ratio_upper"] & terms["clipped"]).sum(),
                                    ]).detach().cpu().tolist()
                                    if not all(map(math.isfinite, values)):
                                        if local_step:
                                            self._reuse_reason = "nonfinite_corrected_probability_terms"
                                            raise _ReuseStopped(self._reuse_reason)
                                        raise FloatingPointError("Nonfinite corrected GRPO probabilities")
                                    n = len(call["completion_ids"])
                                    tokens += n
                                    kl_sum += values[0]
                                    reference_sum += values[1]
                                    weight_sum += values[2]
                                    weight_sq_sum += values[3]
                                    clipped_count += values[4]
                                    log_ratio_max = max(log_ratio_max, values[5])
                                    for key, value in zip(metric_sums, values[6:]):
                                        metric_sums[key] += value
                                    # Each weighted K3 is nonnegative. This lower
                                    # bound uses the FINAL cohort denominator, so an
                                    # early stop cannot be caused by a biased partial
                                    # average. Discard all accumulated gradients.
                                    lower_bound = kl_sum / cohort_tokens
                                    attempt.update(diagnostic_tokens_scored=tokens,
                                                   old_policy_kl_lower_bound=lower_bound,
                                                   old_policy_kl_estimator=correction_description["reuse_kl"])
                                    if local_step and target_kl is not None and float(target_kl) > 0 and lower_bound > float(target_kl):
                                        self._reuse_reason = "target_kl"
                                        if tokens == cohort_tokens:
                                            attempt["old_policy_kl_pre_update"] = lower_bound
                                        if emit_callback:
                                            emit_callback("grpo_reuse_drift", reuse_index=local_step + 1,
                                                          correction_version=CORRECTION_VERSION,
                                                          old_policy_kl_lower_bound=lower_bound,
                                                          diagnostic_tokens_scored=tokens,
                                                          cohort_completion_tokens=cohort_tokens,
                                                          stopped_early=tokens < cohort_tokens)
                                        raise _ReuseStopped(self._reuse_reason)
                                    losses.append(loss)
                                group_loss = torch.stack(losses).sum()
                            if not torch.isfinite(group_loss):
                                raise FloatingPointError("Non-finite grouped GRPO objective")
                            # The native/checkpoint context must outlive backward.
                            # A shared hidden state gets the SUM of all original
                            # loss occurrences, including different advantages.
                            self.accelerator.backward(group_loss)
                            backward_groups += 1
                            calls_backward += len(group_calls)
                            total_loss += float(group_loss.detach().cpu())
                    finally:
                        self._active_call = self._active_scores = None
                        # No shared graph/cache crosses a group or optimizer step.
                        scores = group_loss = logp = entropy = loss = terms = None
                        losses = reference = per_token_loss = values = None
                if self.policy.abort_generation.is_set():
                    raise RuntimeError("Run cancelled before applying accumulated GRPO gradients")
                self._synchronize_timing()
                gradient_seconds = time.perf_counter() - gradient_started_at
                drift = kl_sum / tokens
                probability_correction = {**correction_description,
                    "weight_mean": weight_sum / tokens, "clipped_token_fraction": clipped_count / tokens,
                    "ess_fraction": weight_sum**2 / max(weight_sq_sum * tokens, 1e-30)}
                self._last_step_metrics = {
                    "loss": total_loss, "generated_tokens": tokens,
                    "gradient_forwards": model_forwards, "backwards": backward_groups,
                    "logical_calls_forward": calls_forward, "logical_calls_backward": calls_backward,
                    "training_input_positions": processed_input_tokens,
                    "independent_input_positions": independent_input_tokens,
                    "shared_input_positions_saved": independent_input_tokens - processed_input_tokens,
                    "prefix_reuse_mode": prefix_mode,
                    "prefix_reuse_scope": "One graph per bounded group; token-position counts exclude checkpoint recomputation and are not GPU speedups",
                    "gradient_seconds": gradient_seconds,
                    "gradient_timing_scope": "reference_scoring_and_gradient_accumulation; CUDA synchronized; excludes probability publication/checkpoints",
                    "pre_step_checkpoint_seconds": 0.0,
                    "correction_version": CORRECTION_VERSION,
                    "probability_correction": probability_correction,
                    "approx_old_policy_kl": drift, "old_policy_kl_pre_update": drift,
                    "reference_kl_pre_update": reference_sum / tokens if self.beta else None,
                    "kl_scope": "pre_optimizer; mean weighted token terms on collected behavior prefixes; not full-trajectory KL",
                    "old_policy_kl_estimator": correction_description["reuse_kl"],
                    "reference_kl_estimator": correction_description["reference_kl"] if self.beta else None,
                    "clip_statistics_scope": "unweighted behavior token counts; not importance-weighted fractions",
                    "log_ratio_max": log_ratio_max,
                    "post_update_kl": None, "post_update_kl_status": "not_measured_no_post_step_forward",
                    **{key: value / tokens for key, value in metric_sums.items()},
                }
                # TRL still logs the externally accumulated step; it no longer
                # recomputes a second loss with a different correction measure.
                for key in metric_sums:
                    self._metrics["train"][key].append(self._last_step_metrics[key])
                if self.beta:
                    self._metrics["train"]["kl"].append(reference_sum / tokens)
                attempt.update(self._last_step_metrics)
                if local_step and emit_callback:
                    emit_callback("grpo_reuse_drift", reuse_index=local_step + 1,
                                  approx_old_policy_kl=drift, log_ratio_max=log_ratio_max,
                                  correction_version=CORRECTION_VERSION,
                                  source="pre_optimizer_gradient_forward", generated_tokens=tokens)
                if calibrating:
                    checkpoint_started_at = time.perf_counter()
                    self._publish_frozen_probabilities()
                    attempt.update(status="gradients_frozen")
                    self._checkpoint_safe = True
                    try:
                        self._checkpoint("frozen_before_first_step")
                    finally:
                        self._checkpoint_safe = False
                    self._last_step_metrics["pre_step_checkpoint_seconds"] = time.perf_counter() - checkpoint_started_at
                    attempt.update(self._last_step_metrics)
                attempt["status"] = "gradients_ready"
                self._step += 1
                return torch.tensor(total_loss, device=self.accelerator.device)
            except BaseException as exc:
                model.zero_grad(set_to_none=True)
                self._restore_attempt_metrics(lengths)
                attempt.update(status="discarded", reason=f"{type(exc).__name__}: {exc}")
                if emit_callback:
                    emit_callback("grpo_gradient_attempt_discarded", reuse_index=local_step + 1,
                                  reason=self._reuse_reason or type(exc).__name__,
                                  gradient_forwards=model_forwards, backwards=backward_groups,
                                  logical_calls_forward=calls_forward, logical_calls_backward=calls_backward,
                                  generated_tokens=tokens)
                raise
            finally:
                attempt.update(gradient_forwards=model_forwards, backwards=backward_groups,
                               logical_calls_forward=calls_forward, logical_calls_backward=calls_backward,
                               training_input_positions=processed_input_tokens,
                               independent_input_positions=independent_input_tokens,
                               shared_input_positions_saved=independent_input_tokens - processed_input_tokens)
                if "gradient_seconds" not in attempt:
                    attempt["gradient_seconds"] = time.perf_counter() - gradient_started_at
                    attempt["gradient_timing_scope"] = "interrupted_gradient_attempt; host wall time without final CUDA synchronization"
                if attempt["status"] == "discarded" and self._reuse_started_at is not None:
                    attempt["step_wall_seconds"] = time.perf_counter() - self._reuse_started_at
                    attempt["step_timing_scope"] = "step_begin_to_gradient_discard; no optimizer call"
                self._active_call = self._active_scores = None

        def optimize_batch(self, trajectories):
            if self.has_pending_batch:
                raise RuntimeError("Resume the checkpoint's active cohort before collecting another policy batch")
            if self._optimizing or not self._checkpoint_safe:
                raise RuntimeError("Restore a safe checkpoint before starting another optimization")
            self._step_history = []
            self._offset = self.completed_optimizer_steps
            self._optimizing = True
            self._update_wall_seconds = 0.0
            self._update_timing_from_start = True
            self._update_started_at = time.perf_counter()
            try:
                return self._optimize_batch(trajectories)
            except BaseException:
                self.policy.model.zero_grad(set_to_none=True)
                self._checkpoint_safe = False
                raise
            finally:
                # Trainer can fail before replacing the preceding TrainerState.
                # Count only optimizer steps observed in THIS train() invocation.
                self.completed_optimizer_steps = self._offset + len(self._step_history)
                self._finish_update_timing()
                self.policy._mode = None
                self._active_call = self._active_scores = None
                self._cohort = None
                self._optimizing = False
                self._restored_batch = False
                self._pending_rng_state = None

        def resume_batch(self):
            """Finish the restored cohort, never recalibrating its denominators."""
            if not self.has_pending_batch or self._optimizing or not self._checkpoint_safe:
                raise RuntimeError("No safe restored cohort is pending")
            self._optimizing = True
            self._reuse_start = len(self._step_history)
            self._update_started_at = time.perf_counter()
            try:
                return self._run_active_batch()
            except BaseException:
                self.policy.model.zero_grad(set_to_none=True)
                self._checkpoint_safe = False
                raise
            finally:
                self.completed_optimizer_steps = self._offset + len(self._step_history)
                self._finish_update_timing()
                self.policy._mode = None
                self._active_call = self._active_scores = None
                self._cohort = None
                self._optimizing = False
                self._restored_batch = False
                self._pending_rng_state = None

        def _optimize_batch(self, trajectories):
            if not trajectories:
                raise ValueError("Cannot optimize an empty cohort")
            self._cohort = copy.deepcopy(trajectories)
            self._step_history, self._attempt_history = [], []
            self._reuse_reason = None
            self._offset = self.completed_optimizer_steps
            self._reuse_start = 0
            self._reuse_index = 0
            self._probabilities_frozen = False
            self._preupdate_diagnostics = {}
            self._forward_invocations = {"gradient": 0, "reference": 0, "standalone_old": 0, "standalone_drift": 0}
            self._probability_reuse = {"old_from_first_gradient": 0, "reference_cache_hits": 0,
                                       "reference_from_zero_adapter": 0}
            self.policy._for_training()
            self._reference_matches_learner = bool(
                beta and cfg.get("share_zero_reference", True)
                and getattr(self.policy, "reference_matches_current", lambda: False)()
            )
            for trajectory in self._cohort:
                if not math.isfinite(float(trajectory["advantage"])) or not trajectory.get("calls"):
                    raise ValueError("Every trajectory needs a finite frozen advantage and real calls")
                for call in trajectory["calls"]:
                    if self.policy.abort_generation.is_set():
                        raise RuntimeError("Run cancelled while preparing GRPO probabilities")
                    behavior = call["behavior_logprobs"]
                    if (not call.get("prompt_ids") or not call.get("completion_ids")
                            or len(behavior) != len(call["completion_ids"]) or not all(map(math.isfinite, behavior))):
                        raise ValueError("Missing complete call or finite true behavior probabilities")
                    if len(call["prompt_ids"]) + len(call["completion_ids"]) > self.policy.max_context:
                        raise ValueError("Complete call exceeds context; no truncation is permitted")
                    # Caller-supplied old/reference values are never trusted as
                    # calibration for this fresh on-policy cohort.
                    for key in ("old_learner_logprobs", "reference_logprobs",
                                "old_probability_source", "reference_probability_source"):
                        call.pop(key, None)
            self._probability_path = (Path(output_dir) / "frozen_probabilities" /
                                      f"optimizer-{self._offset:08d}-{uuid.uuid4().hex}.jsonl")
            self._training_groups = self._plan_training_groups(self._cohort)
            if emit_callback:
                emit_callback("training_prefix_plan", mode=prefix_mode,
                    logical_calls=sum(len(group["call_indices"]) for group in self._training_groups),
                    model_forward_groups=len(self._training_groups),
                    independent_input_positions=sum(group["input_tokens"] for group in self._training_groups),
                    planned_input_positions=sum(group["unique_tokens"] for group in self._training_groups),
                    planned_shared_positions=sum(group["saved_tokens"] for group in self._training_groups),
                    max_tree_mask_bytes=max(group["mask_bytes"] for group in self._training_groups),
                    scope="exact collected token identities; not a measured runtime speedup")
            return self._run_active_batch()

        def _run_active_batch(self):
            before = len(self._step_history)
            remaining = repetitions - before
            self._step = before
            if remaining and self._reuse_reason is None:
                self.args.max_steps = remaining
                try:
                    self.train()
                except _ReuseStopped:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.is_in_train = False
                    self._memory_tracker.stop_and_update_metrics({})
                    self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
                    self._checkpoint_safe = True
                    self._checkpoint("cohort_stopped")
            self._checkpoint_safe = True
            self._checkpoint_phase = "cohort_finished"
            post_update_kl = self._measure_post_update_kl()
            measured_steps = [row for row in self._step_history if "nonzero_lr_optimizer_step" in row]
            total_steps = self._offset + len(self._step_history)
            return {"optimizer_steps": len(self._step_history),
                    "lr_schedule": dict(schedule_contract),
                    "zero_lr_optimizer_steps": (sum(row["zero_lr_optimizer_step"] for row in self._step_history)
                        if all(type(row.get("zero_lr_optimizer_step")) is bool for row in self._step_history) else None),
                    "skipped_optimizer_steps": (sum(row["optimizer_step_skipped"] for row in self._step_history)
                        if all(type(row.get("optimizer_step_skipped")) is bool for row in self._step_history) else None),
                    "nonzero_lr_optimizer_steps": (sum(row["nonzero_lr_optimizer_step"] for row in measured_steps)
                                                   if len(measured_steps) == len(self._step_history) else None),
                    "optimizer_steps_with_lr_metrics": len(measured_steps),
                    "total_nonzero_lr_optimizer_steps": (self._optimizer_measurements["nonzero_lr_steps"]
                                                         if self._optimizer_measurements["steps"] == total_steps else None),
                    "optimizer_measurements": dict(self._optimizer_measurements),
                    "nonzero_lr_step_semantics": "optimizer not skipped and at least one used LR nonzero; not proof of adapter weight change",
                    "update_wall_seconds": self._current_update_wall_seconds() if self._update_timing_from_start else None,
                    "observed_update_wall_seconds": self._current_update_wall_seconds(),
                    "update_timing_from_start": self._update_timing_from_start,
                    "update_timing_scope": "cohort trainer invocation wall time including checkpoints; resumed work accumulates from durable snapshot time, excluding downtime and unrecorded work after that snapshot",
                    "optimizer_steps_this_invocation": len(self._step_history) - before,
                    "cohort_start_optimizer_step": self._offset,
                    "total_optimizer_steps": total_steps,
                    "reuse_stop_reason": self._reuse_reason,
                    "updates": list(self._step_history), "steps": list(self._step_history),
                    "gradient_attempts": list(self._attempt_history),
                    "kl_scope": "step KL is pre_optimizer_gradient_forward; post_update_kl is a separate bounded final-cohort probe",
                    "post_update_kl": post_update_kl,
                    "post_update_kl_seconds": post_update_kl["wall_seconds"],
                    "forward_invocations": dict(self._forward_invocations),
                    "probability_reuse": dict(self._probability_reuse),
                    "reference_cache": self._reference_cache.stats(),
                    "training_prefix_reuse": {
                        "mode": prefix_mode, "version": "exact_token_tree_v1",
                        "groups": copy.deepcopy(self._training_groups),
                        "scope": "Exact observed token identities; theoretical position/mask counts, not measured GPU speedup",
                    },
                    "checkpoint_recomputation_included_in_forward_counts": False,
                    "probability_correction": correction_description,
                    **self._preupdate_diagnostics}

    owner = OfficialTaskGRPOTrainer()
    owner.add_callback(CohortCallback())
    return owner
