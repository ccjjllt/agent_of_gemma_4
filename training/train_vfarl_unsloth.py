"""One real VFARL update using Unsloth backward and fused selected logprobs.

Run as one process over one or two visible GPUs. This keeps the existing
verified focal-action PPO objective, not an SFT or replacement GRPO dataset.
No model is loaded on import. GPU throughput and 16K backward capacity require
actual measurements; the collector's inference memory is not training evidence.
"""
import argparse
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback

from .common import TrainingError, atomic_json, sha256_file
from .fsdp_qlora import _write_adapter_state
from .train_ddp import trainable_hash
from .train_vfarl import (actor_learner_binding, importance_summary, prepare_rows,
                         probability_comparison, shard_snapshot, snapshot_hash)
from .vfarl_policy import digest, frozen_base_reference, identities, read_config


def focal_logprobs(policy, row, fused_linear_logprobs):
    """Project only selected predictors, retaining all prefix/control tokens."""
    torch, base = policy.torch, policy.model.get_base_model()
    prompt, completion = row['prompt_token_ids'], row['completion_token_ids']
    ids = torch.tensor([prompt + completion], dtype=torch.long, device=policy.device)
    selected = torch.tensor([[0] * len(prompt) + row['loss_mask']],
                            dtype=torch.bool, device=policy.device)[:, 1:]
    batch_indices, positions = selected.nonzero(as_tuple=True)
    targets = ids[:, 1:][selected]
    output = base.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                        use_cache=False, return_dict=True)
    hidden = output.last_hidden_state
    # The last decoder and tied head can be on different GPUs. The fused
    # projector moves selected hidden states to the head and preserves grads.
    values = hidden[batch_indices.to(hidden.device), positions.to(hidden.device)]
    cap = base.config.get_text_config().final_logit_softcapping
    return fused_linear_logprobs(values, base.lm_head.weight, targets,
        bias=getattr(base.lm_head, 'bias', None), temperature=1.0,
        logit_softcapping=cap or 0.0,
        max_rows_per_chunk=policy.rl['logprob_chunk_size'])


def memory(torch, devices):
    return [dict(device=str(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device)) for device in devices]


def run(args):
    cfg, rl = read_config(args.config)
    rows, evidence = prepare_rows(args.data, cfg['max_sequence_length'])
    if not rows:
        raise TrainingError('No verified mixed-outcome sampled focal actions for an update')
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise TrainingError('Refusing to overwrite an existing VFARL update')
    output.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(Path(args.data))
    step_taken = False
    started = time.monotonic()
    try:
        from .vfarl_unsloth import UnslothPolicy
        print(json.dumps(dict(phase='training_model_loading', backend='unsloth',
            policy_gpus=args.policy_gpus, context=cfg['max_sequence_length'])), flush=True)
        policy = UnslothPolicy(args.config, inference_device_count=args.policy_gpus,
                               training=True)
        torch, model = policy.torch, policy.model
        from unsloth_zoo.fused_losses.fused_linear import fused_linear_logprobs
        runtime = policy.info['probability_execution']
        runtime['identity']['implementation_sha256']['fused_linear_logprobs'] = sha256_file(
            Path(inspect.getfile(fused_linear_logprobs)))
        runtime['identity_sha256'] = digest(runtime['identity'])
        policy.identity = identities(model, policy.tokenizer, policy.info, policy.cfg,
                                     profile=policy.probability_profile)
        identity = policy.identity
        binding = actor_learner_binding(rows, identity, actor_backend=args.actor_backend,
                                       rollout_correction=rl['rollout_correction'])
        binding.update(learner_backend='unsloth-model-parallel',
            learner_execution_profile=policy.probability_profile,
            reference_interpretation='Actor old/reference remain immutable; KL uses a separate frozen-base Unsloth learner forward with zero live LoRA scaling.')
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        shapes = {name: list(value.shape) for name, value in model.named_parameters()
                  if value.requires_grad}
        before = shard_snapshot(model)
        optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=0.0)
        optimizer.zero_grad(set_to_none=True)
        for device in policy.cuda_devices:
            torch.cuda.reset_peak_memory_stats(device)
        print(json.dumps(dict(phase='training_model_ready', backend='unsloth',
            model_loading_seconds=time.monotonic() - started,
            selected_actions=len(rows), mixed_groups=evidence['mixed_group_count'],
            actual_max_sequence_tokens=max(len(r['prompt_token_ids']) +
                                          len(r['completion_token_ids']) for r in rows))), flush=True)
        metrics, before_logprobs = [], {}
        update_started = time.monotonic()
        for row in rows:
            row_started = time.monotonic()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                learner_old = focal_logprobs(policy, row, fused_linear_logprobs).detach()
                with frozen_base_reference(model):
                    reference = focal_logprobs(policy, row, fused_linear_logprobs).detach()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                current = focal_logprobs(policy, row, fused_linear_logprobs)
            old = current.new_tensor([value for value, keep in
                zip(row['behavior_logprobs'], row['loss_mask']) if keep])
            stored_reference = current.new_tensor([value for value, keep in
                zip(row['reference_logprobs'], row['loss_mask']) if keep])
            learner_old, reference = learner_old.to(current.device), reference.to(current.device)
            same_path_error = float((current.detach() - learner_old).abs().max())
            if not math.isfinite(same_path_error) or same_path_error > rl['max_old_logprob_error']:
                raise TrainingError('Unsloth gradient/no-gradient logprob mismatch: ' + str(same_path_error))
            log_is = float((learner_old.double() - old.double()).sum())
            if not math.isfinite(log_is):
                raise TrainingError('Nonfinite focal-action importance ratio')
            is_weight = math.exp(min(log_is, math.log(rl['rollout_is_max'])))
            importance = current.new_tensor(is_weight)
            if not bool(torch.isfinite(importance)) or not bool(importance > 0):
                raise TrainingError('Focal-action importance weight underflowed or is nonfinite')
            ratio = (current - learner_old).exp()
            advantage = row['advantage']
            policy_loss = -torch.minimum(ratio * advantage,
                ratio.clamp(1 - rl['clip_epsilon'], 1 + rl['clip_epsilon']) * advantage)
            log_ref_ratio = reference - current
            kl = log_ref_ratio.exp() - log_ref_ratio - 1
            loss = (policy_loss + rl['kl_beta'] * kl).mean() * importance
            if not bool(torch.isfinite(loss)):
                raise TrainingError('Nonfinite Unsloth PPO/KL loss')
            (loss / len(rows)).backward()
            before_logprobs[row['branch_id']] = current.detach().cpu().tolist()
            metrics.append(dict(branch_id=row['branch_id'], state_id=row['state_id'],
                reward=row['reward'], advantage=advantage, loss=float(loss.detach()),
                policy_loss=float(policy_loss.detach().mean()), kl=float(kl.detach().mean()),
                selected_tokens=old.numel(), sequence_tokens=len(row['prompt_token_ids']) +
                    len(row['completion_token_ids']),
                actor_behavior_logprobs=old.detach().cpu().tolist(),
                actor_reference_logprobs=stored_reference.detach().cpu().tolist(),
                frozen_learner_old_logprobs=learner_old.cpu().tolist(),
                actual_reference_logprobs=reference.cpu().tolist(),
                learner_grad_vs_nograd_max_abs_error=same_path_error,
                old_logprob_max_abs_error=float((current.detach() - old).abs().max()),
                ref_logprob_max_abs_error=float((reference - stored_reference).abs().max()),
                rollout_correction=rl['rollout_correction'],
                sequence_log_importance_weight=log_is,
                sequence_importance_weight_raw=math.exp(log_is) if log_is < 700 else None,
                sequence_importance_weight_used=is_weight,
                sequence_importance_weight_loss_dtype=float(importance.detach()),
                sequence_importance_weight_dtype=str(importance.dtype),
                sequence_importance_weight_capped=log_is > math.log(rl['rollout_is_max']),
                ratio_mean=float(ratio.detach().mean()),
                seconds=time.monotonic() - row_started))
            print(json.dumps(dict(phase='focal_backward_completed',
                branch_id=row['branch_id'], sequence_tokens=metrics[-1]['sequence_tokens'],
                selected_tokens=old.numel(), seconds=metrics[-1]['seconds'])), flush=True)
            del current, learner_old, reference, old, stored_reference, ratio, loss, kl, policy_loss
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['max_grad_norm'],
                                                  error_if_nonfinite=True)
        if not bool(grad_norm > 0):
            raise TrainingError('Verified rewards produced no nonzero gradient')
        optimizer.step()
        step_taken = True
        after = shard_snapshot(model)
        squared_delta = sum(float((after[name] - before[name]).double().square().sum())
                            for name in before)
        changed = sum(int(torch.count_nonzero(after[name] - before[name])) for name in before)
        if not math.isfinite(squared_delta) or not squared_delta or not changed:
            raise TrainingError('Optimizer did not produce a finite nonzero LoRA change')
        temporary_adapter = output / '.adapter-writing'
        _write_adapter_state(after, model, temporary_adapter, torch, shapes)
        temporary_adapter.replace(output / 'adapter')
        adapter_files = {name: dict(sha256=sha256_file(output / 'adapter' / name),
            bytes=(output / 'adapter' / name).stat().st_size)
            for name in ('adapter_config.json', 'adapter_model.safetensors')}
        committed = dict(status='optimizer-step-and-adapter-committed', optimizer_steps=1,
            learner_backend='unsloth-model-parallel',
            adapter_export_completed=True,
            post_update_diagnostics_completed=False, source_data_sha256=source_sha,
            config=cfg, rl=rl, evidence=evidence, policy=identity, model=policy.info,
            actor_learner_binding=binding, metrics=metrics,
            importance_weights=importance_summary(metrics),
            parameter_before_sha256=snapshot_hash(before),
            parameter_after_sha256=snapshot_hash(after),
            trainable_adapter_after_sha256=trainable_hash(model),
            global_parameter_delta_l2=math.sqrt(squared_delta),
            global_changed_parameter_elements=changed, grad_norm=float(grad_norm),
            adapter_files=adapter_files, memory_by_device=memory(torch, policy.cuda_devices),
            actual_max_sequence_tokens=max(m['sequence_tokens'] for m in metrics),
            wall_seconds_to_commit=time.monotonic() - started,
            update_seconds_to_commit=time.monotonic() - update_started,
            storage='Single-process model parallel; original NF4 bytes, no native probability overrides or FSDP conversion',
            recovery='One update for this exact behavior SHA. Reuse the committed adapter; never retrain the batch after a diagnostic failure.',
            limitation='This records one actual update; throughput, 16K backward capacity and independent task improvement require their own actual results.')
        atomic_json(output / 'optimizer-step.json', committed)
        atomic_json(output / 'training_manifest.json', committed)
        print(json.dumps(dict(phase='optimizer_step_committed', optimizer_steps=1,
            grad_norm=committed['grad_norm'],
            global_parameter_delta_l2=committed['global_parameter_delta_l2'],
            manifest=str(output / 'optimizer-step.json'))), flush=True)
        comparisons = []
        for row in rows:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                post = focal_logprobs(policy, row, fused_linear_logprobs).detach().cpu().tolist()
            comparison = probability_comparison(post, before_logprobs[row['branch_id']],
                                                clip_epsilon=rl['clip_epsilon'])
            comparisons.append(dict(branch_id=row['branch_id'], reward=row['reward'],
                advantage=row['advantage'], post_update_logprobs=post,
                before_update_logprobs=before_logprobs[row['branch_id']],
                comparison=comparison))
        final = dict(committed, status='one-verified-focal-PPO-update-completed',
            post_update_diagnostics_completed=True,
            post_update_comparisons=comparisons,
            mean_advantage_weighted_logprob_change=sum(r['advantage'] *
                r['comparison']['signed_mean'] for r in comparisons) / len(comparisons),
            wall_seconds=time.monotonic() - started)
        atomic_json(output / 'training_manifest.json', final)
        print(json.dumps(dict(phase='training_update_completed', optimizer_steps=1,
            post_update_diagnostics_completed=True,
            wall_seconds=final['wall_seconds'])), flush=True)
        return 0
    except Exception:
        (output / 'exception.txt').write_text(traceback.format_exc())
        # An export failure must not turn a real step into a retrainable batch.
        # A completed commit is immutable even if later diagnostics fail.
        if step_taken and not (output / 'optimizer-step.json').exists():
            partial = dict(status='optimizer-step-executed-before-adapter-commit',
                optimizer_steps=1, source_data_sha256=source_sha,
                learner_backend='unsloth-model-parallel', adapter_export_completed=False,
                post_update_diagnostics_completed=False,
                recovery='Preserve this batch and exception; do not perform its optimizer step again.')
            atomic_json(output / 'optimizer-step.json', partial)
            atomic_json(output / 'training_manifest.json', partial)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--actor-backend', choices=('unsloth',), default='unsloth')
    parser.add_argument('--learner-backend', choices=('unsloth',), default='unsloth')
    parser.add_argument('--policy-gpus', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Run this Unsloth learner as one process, not torchrun/FSDP')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
