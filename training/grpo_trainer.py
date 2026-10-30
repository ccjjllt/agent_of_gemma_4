"""TRL GRPOTrainer integration for complete, externally judged tool trajectories.

The official environment owns rollout generation. A Trainer data-loader item is
one frozen cohort, and each optimizer step accumulates its real model calls one
at a time. TRL supplies its clipped GRPO/KL objective and Trainer.train owns the
optimizer, gradient clipping, scheduler and callbacks. This avoids flattening a
compacted multi-turn session into a token history the policy never actually saw.
"""
from __future__ import annotations

import copy
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import uuid

from .probability_cache import ReferenceProbabilityCache


class _ReuseStopped(Exception):
    """A complete cohort needs fresh rollouts before another optimizer update."""


def _trl_implementation():
    """Use the installed pinned TRL class, retaining Unsloth's model kernels.

    Unsloth replaces exported GRPOTrainer methods with generated fused versions.
    Those assume one flattened prompt/completion. Loading the installed official
    module under a private name keeps its original class and our full-prefix
    scoring hook, without copying a loss implementation into this repository.
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


def make_grpo_trainer(policy, config: dict, output_dir: str | Path, emit_callback=None):
    """Construct an actual TRL trainer; importing this module never loads CUDA.

    optimize_batch accepts trajectory dictionaries with calls, advantage,
    task_id and optional reward/reward_components. Each call contains the exact
    prompt_ids, completion_ids and behavior_logprobs collected by the endpoint.
    """
    import torch
    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import TrainerCallback, get_scheduler

    implementation = _trl_implementation()
    GRPOTrainer = implementation.GRPOTrainer
    GRPOConfig = implementation.GRPOConfig
    cfg = dict(config)
    repetitions = int(cfg.get("num_iterations", 2))
    generations = int(cfg.get("num_generations", 8))
    beta = float(cfg.get("beta", 0.01))
    correction_cap = float(cfg.get("importance_sampling_cap", cfg.get("behavior_correction_cap", 2.0)))
    fixed_length = int(cfg.get("max_completion_length", cfg.get("loss_normalizer", 32768)))
    total_optimizer_steps = int(cfg.get("total_optimizer_steps", repetitions))
    warmup_steps = int(cfg.get("warmup_steps", math.ceil(total_optimizer_steps * float(cfg.get("warmup_ratio", 0.0)))))
    if repetitions < 1 or generations < 2 or beta < 0 or fixed_length <= 0 or correction_cap < 1:
        raise ValueError("Invalid GRPO repetition/group/KL/normalizer/correction configuration")
    if cfg.get("loss_type", "dr_grpo") != "dr_grpo":
        raise ValueError("This multi-call adapter uses dr_grpo with one fixed trajectory normalizer")
    if total_optimizer_steps < 1 or warmup_steps < 0:
        raise ValueError("GRPO needs positive total optimizer steps and nonnegative warmup")
    for name, parameter in policy.model.named_parameters():
        if ".lora_B." in name and bool(torch.count_nonzero(parameter.detach())):
            raise ValueError("The fixed disabled-adapter reference requires a fresh zero-B LoRA at trainer creation")

    def environment_rewards(completions=None, rewards=None, **kwargs):
        if rewards is None:
            raise RuntimeError("Official rewards must arrive with completed external trajectories")
        return rewards

    class CohortCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            owner._begin_reuse_step(int(state.global_step))

        def on_step_end(self, args, state, control, **kwargs):
            row = {"event": "grpo_optimizer_step", "optimizer_step": owner._offset + state.global_step,
                   "reuse_index": state.global_step, **owner._last_step_metrics}
            owner._step_history.append(row)
            owner._attempt_history[-1]["status"] = "optimizer_applied"
            if emit_callback:
                emit_callback(row["event"], **{key: value for key, value in row.items() if key != "event"})

    class OfficialTaskGRPOTrainer(GRPOTrainer):
        def __init__(self):
            self.policy = policy
            self._cohort = None
            self._active_call = None
            self._active_scores = None
            self._offset = 0
            self.completed_optimizer_steps = 0
            self._step_history = []
            self._last_step_metrics = {}
            self._reuse_reason = None
            self._preupdate_diagnostics = {}
            self._reference_cache = ReferenceProbabilityCache(
                "disabled-adapter-" + uuid.uuid4().hex,
                max_entries=int(cfg.get("reference_cache_max_entries", 128)),
                max_tokens=int(cfg.get("reference_cache_max_tokens", 1000000)),
            )
            # beta=0 at construction prevents TRL from making a second reference
            # adapter. We supply frozen disabled-adapter probabilities ourselves.
            args = GRPOConfig(
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
                place_model_on_device=False,
            )
            super().__init__(model=policy.model, args=args, reward_funcs=[environment_rewards],
                             train_dataset=Dataset.from_dict({"prompt": [""]}),
                             processing_class=policy.tokenizer)
            self.beta = beta
            self.args.beta = beta
            self.model_accepts_loss_kwargs = False
            if self.accelerator.num_processes != 1:
                raise RuntimeError("Use one process with the model split across GPUs, not DDP launch")

        def get_train_dataloader(self):
            if self._cohort is None:
                raise RuntimeError("Call optimize_batch with completed official trajectories")
            # One item contains one complete cohort. Trainer repeats this one-item
            # epoch mu times; native prompt repetition/generation is bypassed.
            return DataLoader([0], batch_size=1, collate_fn=lambda _: {"official_cohort": True})

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
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type, optimizer=optimizer or self.optimizer,
                    num_warmup_steps=self.args.warmup_steps,
                    num_training_steps=total_optimizer_steps,
                )
            # Trainer otherwise recreates its scheduler on every train() call.
            self._created_lr_scheduler = False
            return self.lr_scheduler

        def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask,
                                               logits_to_keep, **kwargs):
            if (self._active_call is None or self._active_scores is None
                    or logits_to_keep != len(self._active_call["completion_ids"])):
                raise RuntimeError("TRL loss requested an unexpected invocation")
            self._score_hook_calls += 1
            if self._score_hook_calls != 1:
                raise RuntimeError("Expected TRL to consume this invocation's scores exactly once")
            logp, entropy = self._active_scores
            return logp.unsqueeze(0), entropy.unsqueeze(0), None

        def _loss_inputs(self, call, advantage):
            device = self.policy.model.get_output_embeddings().weight.device
            prompt = torch.tensor([call["prompt_ids"]], dtype=torch.long, device=device)
            completion = torch.tensor([call["completion_ids"]], dtype=torch.long, device=device)
            old = torch.tensor([call["old_learner_logprobs"]], dtype=torch.float32, device=device)
            behavior = torch.tensor([call["behavior_logprobs"]], dtype=torch.float32, device=device)
            correction = ((old - behavior).clamp(max=math.log(correction_cap))).exp()
            if not cfg.get("behavior_correction", True):
                correction = torch.ones_like(old)
            # A positive fixed per-token correction commutes with PPO min/clip.
            # This activates the same policy TIS correction as TRL's vLLM path
            # without claiming that the actual generation backend is vLLM.
            inputs = {
                "prompt_ids": prompt, "prompt_mask": torch.ones_like(prompt),
                "completion_ids": completion, "completion_mask": torch.ones_like(completion),
                "advantages": correction * float(advantage),
                "old_per_token_logps": old,
                "sampling_per_token_logps": behavior,
            }
            if self.beta:
                inputs["ref_per_token_logps"] = torch.tensor(
                    [call["reference_logprobs"]], dtype=torch.float32, device=device)
            return inputs

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
            # passed to TRL as the numerator and backpropagates through all tokens.
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
            delta_sum = delta_abs_max = weight_sum = weight_sq_sum = 0.0
            for trajectory in self._cohort:
                for call in trajectory["calls"]:
                    old = torch.tensor(call["old_learner_logprobs"], dtype=torch.float64)
                    behavior = torch.tensor(call["behavior_logprobs"], dtype=torch.float64)
                    delta = old - behavior
                    weights = delta.clamp(max=math.log(correction_cap)).exp()
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
            self.current_gradient_accumulation_steps = 1
            local_step = self._reuse_index
            lengths = {key: len(value) for key, value in self._metrics["train"].items()}
            total_loss = kl_sum = ratio_max = 0.0
            tokens = calls_forward = calls_backward = 0
            metric_sums = {}
            attempt = {"reuse_index": local_step + 1, "status": "accumulating",
                       "gradient_forwards": 0, "backwards": 0}
            self._attempt_history.append(attempt)
            try:
                for trajectory in self._cohort:
                    for call in trajectory["calls"]:
                        if self.policy.abort_generation.is_set():
                            raise RuntimeError("Run cancelled during GRPO gradient accumulation")
                        if local_step == 0:
                            self._reference_for_call(call)
                        logp = entropy = loss = loss_inputs = None
                        try:
                            with self.compute_loss_context_manager():
                                self._forward_invocations["gradient"] += 1
                                calls_forward += 1
                                logp, entropy = self.policy.per_token_logprobs(call, compute_entropy=True)
                                if local_step == 0:
                                    self._freeze_from_gradient(call, logp)
                                else:
                                    # This complete-prefix gradient forward is
                                    # also the reuse diagnostic; no separate pass.
                                    delta = logp.detach().double().cpu() - torch.tensor(
                                        call["old_learner_logprobs"], dtype=torch.float64)
                                    values = torch.expm1(delta) - delta
                                    if not torch.isfinite(values).all():
                                        self._reuse_reason = "nonfinite_old_policy_drift"
                                        raise _ReuseStopped(self._reuse_reason)
                                    kl_sum += float(values.sum())
                                    ratio_max = max(ratio_max, float(delta.exp().max()))
                                self._active_call = call
                                self._active_scores = (logp, entropy)
                                self._score_hook_calls = 0
                                loss_inputs = self._loss_inputs(call, trajectory["advantage"])
                                # Real TRL clipped/KL objective, consuming the
                                # live scores already computed above exactly once.
                                loss = super()._compute_loss(model, loss_inputs) / len(self._cohort)
                                if self._score_hook_calls != 1:
                                    raise RuntimeError("TRL did not consume the current full-prefix scores")
                            if not torch.isfinite(loss):
                                raise FloatingPointError("Non-finite GRPO objective")
                            self.accelerator.backward(loss)
                            calls_backward += 1
                            total_loss += float(loss.detach().cpu())
                            n = len(call["completion_ids"])
                            tokens += n
                            for key in ("kl", "entropy", "clip_ratio/region_mean", "clip_ratio/low_mean", "clip_ratio/high_mean"):
                                values = self._metrics["train"].get(key)
                                if values:
                                    metric_sums[key] = metric_sums.get(key, 0.0) + float(values[-1]) * n
                        finally:
                            self._active_call = self._active_scores = None
                            del logp, entropy, loss, loss_inputs
                if self.policy.abort_generation.is_set():
                    raise RuntimeError("Run cancelled before applying accumulated GRPO gradients")
                drift = kl_sum / tokens if local_step else None
                if local_step:
                    attempt.update(approx_old_policy_kl=drift, ratio_max=ratio_max)
                    if emit_callback:
                        emit_callback("grpo_reuse_drift", reuse_index=local_step + 1,
                                      approx_old_policy_kl=drift, ratio_max=ratio_max,
                                      source="gradient_forward", generated_tokens=tokens)
                    target = cfg.get("target_kl", 0.03)
                    if target is not None and float(target) > 0 and drift > float(target):
                        self._reuse_reason = "target_kl"
                        raise _ReuseStopped(self._reuse_reason)
                else:
                    # Every denominator is now frozen under the pre-update
                    # weights. Publish only a complete snapshot, before step().
                    self._publish_frozen_probabilities()
                self._last_step_metrics = {
                    "loss": total_loss, "generated_tokens": tokens,
                    "gradient_forwards": calls_forward, "backwards": calls_backward,
                    "approx_old_policy_kl": drift,
                    **{key: value / tokens for key, value in metric_sums.items()},
                }
                attempt["status"] = "gradients_ready"
                self._step += 1
                return torch.tensor(total_loss, device=self.accelerator.device)
            except BaseException as exc:
                # All calls belong to one optimizer step. Reject the entire
                # accumulated gradient, including earlier calls in this attempt.
                model.zero_grad(set_to_none=True)
                self._restore_attempt_metrics(lengths)
                attempt.update(status="discarded", reason=f"{type(exc).__name__}: {exc}")
                if emit_callback:
                    emit_callback("grpo_gradient_attempt_discarded", reuse_index=local_step + 1,
                                  reason=self._reuse_reason or type(exc).__name__,
                                  gradient_forwards=calls_forward, backwards=calls_backward,
                                  generated_tokens=tokens)
                raise
            finally:
                attempt.update(gradient_forwards=calls_forward, backwards=calls_backward)
                self._active_call = self._active_scores = None

        def optimize_batch(self, trajectories):
            self._step_history = []
            self._offset = self.completed_optimizer_steps
            try:
                return self._optimize_batch(trajectories)
            except BaseException:
                self.policy.model.zero_grad(set_to_none=True)
                raise
            finally:
                # Trainer can fail before replacing the preceding TrainerState.
                # Count only optimizer steps observed in THIS train() invocation.
                self.completed_optimizer_steps = self._offset + len(self._step_history)
                self.policy._mode = None
                self._active_call = self._active_scores = None
                self._cohort = None

        def _optimize_batch(self, trajectories):
            if not trajectories:
                raise ValueError("Cannot optimize an empty cohort")
            self._cohort = copy.deepcopy(trajectories)
            self._step_history, self._attempt_history = [], []
            self._reuse_reason = None
            self._offset = self.completed_optimizer_steps
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
            self._probability_path = Path(output_dir) / "frozen_probabilities" / f"optimizer-{self._offset:08d}.jsonl"
            if self._probability_path.exists():
                raise FileExistsError("A frozen snapshot already exists for this optimizer offset")
            self.args.max_steps = repetitions
            self._step = 0
            try:
                self.train()
            except _ReuseStopped:
                self.optimizer.zero_grad(set_to_none=True)
                self.is_in_train = False
                self._memory_tracker.stop_and_update_metrics({})
                self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
            return {"optimizer_steps": len(self._step_history),
                    "total_optimizer_steps": self._offset + len(self._step_history),
                    "reuse_stop_reason": self._reuse_reason,
                    "updates": list(self._step_history), "gradient_attempts": list(self._attempt_history),
                    "forward_invocations": dict(self._forward_invocations),
                    "probability_reuse": dict(self._probability_reuse),
                    "reference_cache": self._reference_cache.stats(),
                    "checkpoint_recomputation_included_in_forward_counts": False,
                    **self._preupdate_diagnostics}

    owner = OfficialTaskGRPOTrainer()
    owner.add_callback(CohortCallback())
    return owner
