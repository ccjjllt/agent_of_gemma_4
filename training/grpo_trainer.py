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
from pathlib import Path
import sys


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
            owner._before_optimizer_step(int(state.global_step))

        def on_step_end(self, args, state, control, **kwargs):
            row = {"event": "grpo_optimizer_step", "optimizer_step": owner._offset + state.global_step,
                   "reuse_index": state.global_step, **owner._last_step_metrics}
            owner._step_history.append(row)
            if emit_callback:
                emit_callback(row["event"], **{key: value for key, value in row.items() if key != "event"})

    class OfficialTaskGRPOTrainer(GRPOTrainer):
        def __init__(self):
            self.policy = policy
            self._cohort = None
            self._active_call = None
            self._offset = 0
            self.completed_optimizer_steps = 0
            self._step_history = []
            self._last_step_metrics = {}
            self._reuse_reason = None
            self._preupdate_diagnostics = {}
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
            if self._active_call is None or logits_to_keep != len(self._active_call["completion_ids"]):
                raise RuntimeError("TRL loss requested an unexpected invocation")
            logp, entropy = self.policy.per_token_logprobs(self._active_call, compute_entropy=True)
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

        def _before_optimizer_step(self, local_step):
            if self.policy.abort_generation.is_set():
                raise RuntimeError("Run cancelled before GRPO update")
            target = cfg.get("target_kl", 0.03)
            if local_step == 0 or target is None or float(target) <= 0:
                return
            count, kl_sum, ratio_max = 0, 0.0, 0.0
            for trajectory in self._cohort:
                for call in trajectory["calls"]:
                    current = torch.tensor(self.policy.score_call(call), dtype=torch.float64)
                    old = torch.tensor(call["old_learner_logprobs"], dtype=torch.float64)
                    delta = current - old
                    values = torch.expm1(delta) - delta
                    if not torch.isfinite(values).all():
                        self._reuse_reason = "nonfinite_old_policy_drift"
                        raise _ReuseStopped(self._reuse_reason)
                    count += delta.numel()
                    kl_sum += float(values.sum())
                    ratio_max = max(ratio_max, float(delta.exp().max()))
            drift = kl_sum / max(count, 1)
            if emit_callback:
                emit_callback("grpo_reuse_drift", reuse_index=local_step + 1,
                              approx_old_policy_kl=drift, ratio_max=ratio_max)
            if drift > float(target):
                self._reuse_reason = "target_kl"
                raise _ReuseStopped(self._reuse_reason)

        def training_step(self, model, inputs, num_items_in_batch=None):
            self._prepare_inputs(inputs)
            self.policy._for_training()
            self.current_gradient_accumulation_steps = 1
            total_loss = 0.0
            tokens = 0
            metric_sums = {}
            for trajectory in self._cohort:
                for call in trajectory["calls"]:
                    if self.policy.abort_generation.is_set():
                        raise RuntimeError("Run cancelled during GRPO gradient accumulation")
                    self._active_call = call
                    loss_inputs = self._loss_inputs(call, trajectory["advantage"])
                    with self.compute_loss_context_manager():
                        # This is TRL 1.15.0's real clipped objective, not a local
                        # approximation. dr_grpo divides by the fixed length;
                        # we divide by trajectories, not by their number of calls.
                        loss = super()._compute_loss(model, loss_inputs) / len(self._cohort)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite GRPO objective")
                    self.accelerator.backward(loss)
                    total_loss += float(loss.detach().cpu())
                    n = len(call["completion_ids"])
                    tokens += n
                    for key in ("kl", "entropy", "clip_ratio/region_mean", "clip_ratio/low_mean", "clip_ratio/high_mean"):
                        values = self._metrics["train"].get(key)
                        if values:
                            metric_sums[key] = metric_sums.get(key, 0.0) + float(values[-1]) * n
                    del loss, loss_inputs
            self._active_call = None
            self._last_step_metrics = {"loss": total_loss, "generated_tokens": tokens,
                                       **{key: value / max(tokens, 1) for key, value in metric_sums.items()}}
            self._step += 1
            return torch.tensor(total_loss, device=self.accelerator.device)

        def optimize_batch(self, trajectories):
            if not trajectories:
                raise ValueError("Cannot optimize an empty cohort")
            self._cohort = copy.deepcopy(trajectories)
            self._step_history = []
            self._reuse_reason = None
            self._offset = self.completed_optimizer_steps
            self.policy._for_training()
            count, delta_sum, delta_abs_max, weight_sum, weight_sq_sum = 0, 0.0, 0.0, 0.0, 0.0
            for trajectory in self._cohort:
                if not math.isfinite(float(trajectory["advantage"])) or not trajectory.get("calls"):
                    raise ValueError("Every trajectory needs a finite frozen advantage and real calls")
                for call in trajectory["calls"]:
                    if self.policy.abort_generation.is_set():
                        raise RuntimeError("Run cancelled while freezing GRPO probabilities")
                    behavior = call["behavior_logprobs"]
                    if len(behavior) != len(call["completion_ids"]) or not all(map(math.isfinite, behavior)):
                        raise ValueError("Missing or non-finite true behavior probabilities")
                    call["old_learner_logprobs"] = self.policy.score_call(call)
                    if beta:
                        call["reference_logprobs"] = self.policy.score_call(call, reference=True)
                    delta = torch.tensor(call["old_learner_logprobs"], dtype=torch.float64) - torch.tensor(behavior, dtype=torch.float64)
                    weights = delta.clamp(max=math.log(correction_cap)).exp()
                    count += delta.numel()
                    delta_sum += float(delta.sum())
                    delta_abs_max = max(delta_abs_max, float(delta.abs().max()))
                    weight_sum += float(weights.sum())
                    weight_sq_sum += float(weights.square().sum())
            self._preupdate_diagnostics = {
                "old_vs_behavior_logprob_mean": delta_sum / max(count, 1),
                "old_vs_behavior_logprob_abs_max": delta_abs_max,
                "behavior_correction_ess_fraction": weight_sum**2 / max(weight_sq_sum * count, 1e-30),
                "generated_tokens": count,
            }
            probability_path = Path(output_dir) / "frozen_probabilities" / f"optimizer-{self._offset:08d}.jsonl"
            probability_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with probability_path.open("x", encoding="utf-8") as output:
                for trajectory in self._cohort:
                    for i, call in enumerate(trajectory["calls"]):
                        output.write(json.dumps({
                            "rollout_id": trajectory.get("rollout_id"),
                            "task_id": trajectory.get("task_id"),
                            "call_index": call.get("call_index", i),
                            "policy_version": call.get("policy_version"),
                            "old_learner_logprobs": call["old_learner_logprobs"],
                            "reference_logprobs": call.get("reference_logprobs"),
                        }) + "\n")
            self._preupdate_diagnostics["frozen_probabilities_path"] = str(probability_path)
            if emit_callback:
                emit_callback("grpo_probabilities_frozen", **self._preupdate_diagnostics)
            self.args.max_steps = repetitions
            self._step = 0
            try:
                self.train()
            except _ReuseStopped:
                self.optimizer.zero_grad(set_to_none=True)
                self.is_in_train = False
                self._memory_tracker.stop_and_update_metrics({})
                self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
            finally:
                self.completed_optimizer_steps = self._offset + int(self.state.global_step)
                self.policy._mode = None
                self._active_call = None
                self._cohort = None
            return {"optimizer_steps": len(self._step_history),
                    "total_optimizer_steps": self.completed_optimizer_steps,
                    "reuse_stop_reason": self._reuse_reason,
                    "updates": list(self._step_history), **self._preupdate_diagnostics}

    owner = OfficialTaskGRPOTrainer()
    owner.add_callback(CohortCallback())
    return owner
