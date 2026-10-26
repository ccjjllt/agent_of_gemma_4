"""One real FSDP PPO update on verified same-state sampled focal actions.

No rollout or reward is manufactured here. Only independently sampled focal
actions with complete terminal verification, restored declared state and mixed
rewards contribute. Original-action replays and all continuation tokens are
excluded. Launch with torchrun --standalone --nproc-per-node=4 -m training.train_vfarl.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from . import train_sft as single
from .common import TrainingError, atomic_json, sha256_file
from .fsdp_checkpoint import collective_call
from .fsdp_qlora import pack_floating_nf4_storage, wrap_fsdp, save_adapter
from .vfarl_policy import (bind_template, identities, install_eval_checkpointing,
                          install_logprob_forward, read_config, require_zero_lora,
                          frozen_base_reference)


def prepare_rows(path, context_limit):
    groups, excluded = {}, []
    seen = set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        identity = row.get('branch_id')
        if not identity or identity in seen:
            raise TrainingError('Missing/duplicate branch identity on line %d' % number)
        seen.add(identity)
        if (row.get('kind') != 'sampled_focal_action' or row.get('on_policy') is not True
                or row.get('verified') is not True or row.get('reward') not in (0, 1)
                or row.get('environment_control_verified') is not True
                or row.get('restoration_scope') != 'complete_declared_scope'):
            excluded.append(dict(branch_id=identity, reason='not-complete-verified-sampled-focal-action'))
            continue
        prompt, completion = row['prompt_token_ids'], row['completion_token_ids']
        mask, old, reference = row['loss_mask'], row['behavior_logprobs'], row['reference_logprobs']
        if (not prompt or not completion or len(prompt) + len(completion) > context_limit
                or not all(len(x) == len(completion) for x in (mask, old, reference))
                or any(x not in (0, 1) for x in mask) or not any(mask)
                or any(not isinstance(x, int) or x < 0 for x in prompt + completion)
                or any(not math.isfinite(x) or x > 1e-6 for x in old + reference)):
            raise TrainingError('Invalid exact token/probability record: ' + identity)
        if row.get('reference_forward_performed') is not True:
            raise TrainingError('Missing actual reference forward: ' + identity)
        if row.get('sampling') != {'temperature': 1.0, 'top_p': 1.0, 'top_k': 0}:
            raise TrainingError('This update requires full-support T1 sampling')
        if not row.get('state_id') or not row.get('policy_id') or not row.get('reference_id'):
            raise TrainingError('Missing state/policy identities: ' + identity)
        if row.get('finish_reason') not in ('tool_calls', 'stop'):
            excluded.append(dict(branch_id=identity, reason='incomplete-focal-response'))
            continue
        groups.setdefault(row['state_id'], []).append(row)
    rows, summary = [], []
    for state_id, members in groups.items():
        if len({(r['policy_id'], r['reference_id']) for r in members}) != 1:
            raise TrainingError('Different policies in same-state group ' + state_id)
        if len({tuple(r['prompt_token_ids']) for r in members}) != 1:
            raise TrainingError('Different exact model prefixes in same-state group ' + state_id)
        average = sum(r['reward'] for r in members) / len(members)
        mixed = len(members) >= 2 and 0 < average < 1
        summary.append(dict(state_id=state_id, count=len(members), reward_mean=average,
                            mixed_outcomes=mixed, branches=[r['branch_id'] for r in members]))
        if mixed:
            for row in members:
                row['advantage'] = row['reward'] - average
                rows.append(row)
    return rows, dict(groups=summary, excluded=excluded,
                      objective='same-state mean-centered terminal reward; focal action only',
                      valid_group_count=len(groups), mixed_group_count=sum(s['mixed_outcomes'] for s in summary))


def batch_for(row, torch, device):
    prompt, completion = row['prompt_token_ids'], row['completion_token_ids']
    ids = torch.tensor([prompt + completion], device=device, dtype=torch.long)
    mask = torch.tensor([[0] * len(prompt) + row['loss_mask']], device=device, dtype=torch.bool)
    old = torch.tensor([p for p, keep in zip(row['behavior_logprobs'], row['loss_mask']) if keep],
                       device=device, dtype=torch.float32)
    ref = torch.tensor([p for p, keep in zip(row['reference_logprobs'], row['loss_mask']) if keep],
                       device=device, dtype=torch.float32)
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), vfarl_selected_mask=mask), old, ref


def shard_snapshot(wrapped):
    return {name: p.detach().float().cpu().clone() for name, p in wrapped.named_parameters()
            if p.requires_grad}


def snapshot_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def run(args):
    cfg, rl = read_config(args.config)
    rows, evidence = prepare_rows(args.data, cfg['max_sequence_length'])
    if not rows:
        print(json.dumps(dict(optimizer_steps=0, reason='No fully verified mixed-outcome sampled state groups',
                              **evidence), ensure_ascii=False), flush=True)
        return 0
    stack = single.require_stack('nf4')
    torch, dist = stack['torch'], stack['torch'].distributed
    if int(os.environ.get('WORLD_SIZE', '1')) != 4:
        raise TrainingError('This L4 experiment requires torchrun with exactly four ranks')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    cfg['device'] = 'cuda:%d' % local_rank
    dist.init_process_group('nccl', timeout=timedelta(seconds=900))
    rank, world = dist.get_rank(), dist.get_world_size()
    output = Path(args.out).resolve()

    def prepare_output():
        if rank == 0:
            if output.exists() and any(output.iterdir()):
                raise TrainingError('Refusing to overwrite VFARL output')
            output.mkdir(parents=True, exist_ok=True)
    collective_call(torch, prepare_output)
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    model, tokenizer, device, info = single.load_model_and_tokenizer(
        cfg, stack, adapter_init_seed=cfg['seed'])
    bind_template(tokenizer, cfg)
    require_zero_lora(model, torch)
    identity = identities(model, tokenizer, info, cfg)

    def verify_identity():
        for row in rows:
            if row['policy_id'] != identity['policy_id'] or row['reference_id'] != identity['reference_id']:
                raise TrainingError('Sampled behavior/reference policy differs from actual initial trainer')
        if any(max(r['prompt_token_ids'] + r['completion_token_ids']) >=
               model.get_input_embeddings().num_embeddings for r in rows):
            raise TrainingError('Sample contains an out-of-vocabulary token')
    collective_call(torch, verify_identity)
    model.eval()
    checkpoint_layers = install_eval_checkpointing(model) if cfg['gradient_checkpointing'] else 0
    install_logprob_forward(model, rl['logprob_chunk_size'])
    shapes = {name: list(p.shape) for name, p in model.named_parameters() if p.requires_grad}
    storage = pack_floating_nf4_storage(model, torch, stack['bitsandbytes'], torch.bfloat16)
    # Gemma's generic no-split list also names absent audio/vision layers. The
    # experiment wraps the text decoder, as did the proven P015 NF4 path.
    os.environ.setdefault('FSDP_TRANSFORMER_CLS_TO_WRAP', 'Gemma4TextDecoderLayer')
    wrapped, units = wrap_fsdp(model, torch, device)
    wrapped.eval()
    optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],
                                 lr=cfg['learning_rate'], weight_decay=0.0)
    before = shard_snapshot(wrapped)
    initial_hash = snapshot_hash(before)
    optimizer.zero_grad(set_to_none=True)
    # Mean over sampled focal actions, then tokens within each action; long
    # thought streams do not receive more episode weight solely by length.
    denominator = len(rows)
    metrics, before_logprobs, reference_logprobs = [], {}, {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, len(rows), world):
        index = start + rank
        real = index < len(rows)
        row = rows[index] if real else rows[start]
        batch, old, stored_ref = batch_for(row, torch, device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16), frozen_base_reference(model):
            ref = wrapped(**batch).detach()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            current = wrapped(**batch)
        old_error = float((current.detach() - old).abs().max())
        ref_error = float((ref - stored_ref).abs().max())
        errors = torch.tensor([old_error, ref_error], device=device)
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        if float(errors.max()) > rl['max_old_logprob_error']:
            raise TrainingError('Captured vs update logprob mismatch: ' + str(errors.tolist()))
        ratio = (current - old).exp()
        advantage = row['advantage']
        policy_loss = -torch.minimum(ratio * advantage,
                                    ratio.clamp(1 - rl['clip_epsilon'], 1 + rl['clip_epsilon']) * advantage)
        log_ref_ratio = ref - current
        kl = log_ref_ratio.exp() - log_ref_ratio - 1
        per_token = policy_loss + rl['kl_beta'] * kl
        loss = per_token.mean()
        finite = torch.tensor(int(bool(torch.isfinite(loss).all())), device=device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not int(finite):
            raise TrainingError('Nonfinite PPO/KL loss')
        (loss * (world / denominator if real else 0.0)).backward()
        if real:
            before_logprobs[row['branch_id']] = current.detach().cpu().tolist()
            reference_logprobs[row['branch_id']] = ref.detach().cpu().tolist()
            metrics.append(dict(branch_id=row['branch_id'], state_id=row['state_id'],
                                reward=row['reward'], advantage=advantage, loss=float(loss.detach()),
                                policy_loss=float(policy_loss.detach().mean()), kl=float(kl.detach().mean()),
                                ratio_mean=float(ratio.detach().mean()),
                                clipped_fraction=float(((ratio.detach() - 1).abs() > rl['clip_epsilon']).float().mean()),
                                old_logprob_max_abs_error=old_error, ref_logprob_max_abs_error=ref_error,
                                sequence_tokens=len(row['prompt_token_ids']) + len(row['completion_token_ids']),
                                selected_tokens=old.numel()))
        del current, loss, per_token, ratio, kl, batch
    grad_norm = wrapped.clip_grad_norm_(cfg['max_grad_norm'])
    if not bool(torch.isfinite(grad_norm)) or float(grad_norm) == 0:
        raise TrainingError('Verified mixed outcomes produced no finite nonzero reward gradient')
    optimizer.step()
    after = shard_snapshot(wrapped)
    squared_delta = sum(float((after[name] - before[name]).double().square().sum()) for name in before)
    changed = sum(int(torch.count_nonzero(after[name] != before[name]).item()) for name in before)
    local_change = torch.tensor([squared_delta, changed], dtype=torch.float64, device=device)
    dist.all_reduce(local_change, op=dist.ReduceOp.SUM)
    if float(local_change[0]) == 0:
        raise TrainingError('Optimizer returned without changing any LoRA parameter')
    # Commit the actual updated adapter before optional post-update forwards.
    # An OOM during diagnostics must not destroy the only copy of a completed
    # update or cause the same behavior batch to be trained for a second time.
    save_adapter(wrapped, model, output / 'adapter', torch, shapes)
    committed_rank = dict(
        rank=rank, optimizer_steps=1, grad_norm=float(grad_norm),
        parameter_shard_before_sha256=initial_hash,
        parameter_shard_after_sha256=snapshot_hash(after),
        squared_delta_local=squared_delta, changed_parameters_local=changed,
        metrics=metrics, before_logprobs=before_logprobs,
        reference_logprobs=reference_logprobs,
        peak_allocated_bytes_at_commit=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes_at_commit=torch.cuda.max_memory_reserved(device))
    collective_call(torch, lambda: atomic_json(
        output / ('rank-%03d-update.json' % rank), committed_rank))

    def commit_update():
        if rank != 0:
            return
        adapter_files = {}
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            path = output / 'adapter' / name
            adapter_files[name] = dict(sha256=sha256_file(path), bytes=path.stat().st_size)
        committed = dict(
            status='optimizer-step-and-adapter-committed', optimizer_steps=1,
            post_update_diagnostics_completed=False,
            post_update_diagnostics='not yet executed at the commit boundary',
            source_data_sha256=sha256_file(Path(args.data)), config=cfg, rl=rl,
            evidence=evidence, policy=identity, model=info, storage=storage,
            checkpointed_decoder_layers=checkpoint_layers, rank0_wrap_units=units,
            global_parameter_delta_l2=float(local_change[0].sqrt()),
            global_changed_parameter_elements=int(local_change[1]),
            grad_norm=float(grad_norm), adapter_files=adapter_files,
            rank_update_files=[dict(path='rank-%03d-update.json' % r,
                                    sha256=sha256_file(output / ('rank-%03d-update.json' % r)))
                               for r in range(world)],
            wall_seconds_to_commit=time.monotonic() - started,
            frozen_reference='Actual separate zero-LoRA-scaling forwards; fixed base reference with unchanged FSDP traversal',
            recovery='The optimizer step is already complete. Reuse this adapter and the saved exact-token behavior data for separate diagnostics; do not repeat the training update.',
            limitation='Parameter changes are committed; post-update likelihood direction and re-rollout outcomes have not yet been measured.')
        atomic_json(output / 'optimizer-step.json', committed)
        # The carrier can distinguish an exported update from fully completed
        # diagnostics even if the process later fails. optimizer-step.json is
        # immutable evidence; the final manifest is replaced only on success.
        atomic_json(output / 'training_manifest.json', committed)

    collective_call(torch, commit_update)
    if rank == 0:
        print(json.dumps(dict(phase='optimizer_step_committed', optimizer_steps=1,
                              manifest=str(output / 'optimizer-step.json'),
                              post_update_diagnostics_completed=False)), flush=True)
    comparisons = []
    for start in range(0, len(rows), world):
        index = start + rank
        row = rows[index] if index < len(rows) else rows[start]
        batch, _, _ = batch_for(row, torch, device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            post = wrapped(**batch).detach().cpu()
        if index < len(rows):
            pre = torch.tensor(before_logprobs[row['branch_id']])
            reference = torch.tensor(reference_logprobs[row['branch_id']])
            post_reference_ratio = reference - post
            comparisons.append(dict(branch_id=row['branch_id'], reward=row['reward'], advantage=row['advantage'],
                                    baseline_source='Actual no-grad frozen-reference forward; equivalent to the required zero-B initial policy, not a substitute for behavior logprobs',
                                    before_logprobs=reference.tolist(), after_logprobs=post.tolist(),
                                    grad_forward_before_logprobs=pre.tolist(),
                                    reference_logprobs=reference.tolist(),
                                    initial_grad_vs_nograd_mean_delta=float((pre - reference).mean()),
                                    initial_grad_vs_nograd_max_abs_delta=float((pre - reference).abs().max()),
                                    post_reference_k3_on_behavior_tokens=float(
                                        (post_reference_ratio.exp() - post_reference_ratio - 1).mean()),
                                    mean_logprob_delta=float((post - reference).mean()),
                                    advantage_weighted_mean_delta=float((post - reference).mean()) * row['advantage']))
    torch.cuda.synchronize(device)
    local = dict(rank=rank, optimizer_steps=1, grad_norm=float(grad_norm),
                 parameter_shard_before_sha256=initial_hash,
                 parameter_shard_after_sha256=snapshot_hash(after),
                 squared_delta_local=squared_delta, changed_parameters_local=changed,
                 metrics=metrics, before_after=comparisons,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    atomic_json(output / ('rank-%03d-result.json' % rank), local)
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    if rank == 0:
        atomic_json(output / 'training_manifest.json', dict(
            status='one-verified-focal-PPO-update-completed', optimizer_steps=1,
            post_update_diagnostics_completed=True,
            optimizer_step_manifest='optimizer-step.json',
            optimizer_step_manifest_sha256=sha256_file(output / 'optimizer-step.json'),
            source_data_sha256=sha256_file(Path(args.data)), config=cfg, rl=rl,
            evidence=evidence, policy=identity, model=info, storage=storage,
            checkpointed_decoder_layers=checkpoint_layers, rank0_wrap_units=units,
            global_parameter_delta_l2=float(local_change[0].sqrt()),
            global_changed_parameter_elements=int(local_change[1]),
            wall_seconds=time.monotonic() - started, ranks=gathered,
            frozen_reference='Actual separate zero-LoRA-scaling forwards; fixed base reference with unchanged FSDP traversal',
            limitation='Training-state action likelihood changes are not heldout repair gains; re-rollout is separate.'))
    dist.barrier()
    dist.destroy_process_group()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception:
        import traceback
        traceback.print_exc()
        # Let torchrun kill peers that may be waiting in a failed collective.
        sys.stdout.flush()
        sys.stderr.flush()
        if 'WORLD_SIZE' in os.environ:
            os._exit(2)
        return 2


if __name__ == '__main__':
    sys.exit(main())
