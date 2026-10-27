"""One real FSDP PPO update on verified same-state sampled focal actions.

No rollout or reward is manufactured here. Only independently sampled focal
actions with complete terminal verification, restored declared state and mixed
rewards contribute. A live original action can be one sample when its pre-action
capture and independently restored partner are bound below. Forced original-action
replays and all continuation tokens are excluded. Launch with
torchrun --standalone --nproc-per-node=4 -m training.train_vfarl.
An Unsloth actor uses --actor-backend unsloth with the native FSDP learner and
sequence_tis; --calibrate fsdp-focal scores a fresh focal batch without updating.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

from . import train_sft as single
from .common import TrainingError, atomic_json, sha256_file
from .fsdp_checkpoint import collective_call
from .fsdp_qlora import pack_floating_nf4_storage, wrap_fsdp, save_adapter
from .vfarl_policy import (bind_template, identities, install_eval_checkpointing,
                          install_logprob_forward, read_config, require_zero_lora,
                          frozen_base_reference, install_probability_execution,
                          digest)


def live_focal_sample_sha256(row):
    """Bind actual sampled tokens/probabilities independently of reference scoring."""
    fields = ('prompt_token_ids', 'completion_token_ids', 'loss_mask',
              'behavior_logprobs', 'policy_id', 'reference_id')
    value = {key: row[key] for key in fields}
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _live_capture_binding(row):
    """Require provenance from the live pre-action capture, never a forced replay."""
    evidence = row.get('live_capture_evidence')
    if not isinstance(evidence, dict):
        raise TrainingError('Missing live-capture evidence: ' + row['branch_id'])
    prompt_hash = hashlib.sha256(json.dumps(row['prompt_token_ids'], ensure_ascii=False,
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    hashes = ('selection_evidence_sha256', 'snapshot_manifest_sha256', 'original_action_line_sha256',
              'original_episode_sha256', 'same_state_restore_sha256',
              'captured_ledger_sha256', 'restored_ledger_sha256')
    paths = ('selection_evidence_path', 'snapshot_manifest_path', 'original_actions_path',
             'original_episode_path', 'same_state_restore_evidence')
    valid = (
        evidence.get('format') == 'vfarl-live-capture-v1'
        and evidence.get('selected_by_prefix_only') is True
        and evidence.get('selection_rule') ==
            'first-real-tool-failure-with-recovery-budget-before-action'
        and row.get('original_action_replay') is False
        and evidence.get('original_action_replay') is False
        and evidence.get('original_behavior_preserved') is True
        and evidence.get('original_verified') is True
        and evidence.get('original_terminal_reward') == row['reward']
        and row.get('origin_episode_reward') == row['reward']
        and evidence.get('state_id') == row['state_id']
        and evidence.get('restored_state_id') == row['state_id']
        and evidence.get('restored_policy_id') == row['policy_id']
        and evidence.get('prompt_token_ids_sha256') == prompt_hash
        and evidence.get('restored_prompt_token_ids_sha256') == prompt_hash
        and evidence.get('original_action_sha256') == live_focal_sample_sha256(row)
        and type(evidence.get('original_action_index')) is int
        and evidence.get('original_action_index') == row['checkpoint_turn']
        and type(evidence.get('original_action_line')) is int
        and evidence.get('original_action_line') == row['checkpoint_turn'] + 1
        and evidence.get('captured_ledger_sha256') == evidence.get('restored_ledger_sha256')
        and evidence.get('same_state_restore_evidence') == row.get('restored_state_evidence')
        and isinstance(evidence.get('paired_branch_id'), str)
        and bool(evidence['paired_branch_id'])
        and evidence['paired_branch_id'] != row['branch_id']
        and all(isinstance(evidence.get(key), str) and bool(evidence[key]) for key in paths)
        and all(isinstance(evidence.get(key), str) and len(evidence[key]) == 64
                and all(char in '0123456789abcdef' for char in evidence[key]) for key in hashes)
    )
    if not valid:
        raise TrainingError('Unbound live-captured original action: ' + row['branch_id'])


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
        if (row.get('kind') not in ('sampled_focal_action', 'live_captured_original_action')
                or row.get('on_policy') is not True
                or row.get('verified') is not True or row.get('reward') not in (0, 1)
                or row.get('environment_control_verified') is not True
                or row.get('restoration_scope') != 'complete_declared_scope'):
            excluded.append(dict(branch_id=identity, reason='not-complete-verified-sampled-focal-action'))
            continue
        feedback = row.get('checkpoint_last_tool_feedback', {})
        checkpoint_turn = row.get('checkpoint_turn')
        observations = row.get('checkpoint_tool_observations')
        observed_failures = row.get('checkpoint_tool_failures_observed')
        if (type(checkpoint_turn) is not int or checkpoint_turn < 1
                or type(observations) is not int or observations < 1
                or type(observed_failures) is not int or not 1 <= observed_failures <= observations
                or feedback.get('observation_index') != observations
                or feedback.get('observed_after_turn') != checkpoint_turn
                or not feedback.get('response_sha256')):
            excluded.append(dict(branch_id=identity, reason='no-bound-real-failure-feedback-in-focal-prefix'))
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
        if row['kind'] == 'live_captured_original_action':
            _live_capture_binding(row)
        groups.setdefault(row['state_id'], []).append(row)
    rows, summary = [], []
    for state_id, members in groups.items():
        originals = [row for row in members if row['kind'] == 'live_captured_original_action']
        if len(originals) > 1:
            raise TrainingError('Multiple live originals in one state group ' + state_id)
        if originals:
            original = originals[0]
            evidence = original['live_capture_evidence']
            partner = next((row for row in members
                if row['branch_id'] == evidence['paired_branch_id']), None)
            if partner is None:
                excluded.append(dict(branch_id=original['branch_id'],
                    reason='missing-complete-independent-restored-partner'))
                members = [row for row in members if row is not original]
                if not members:
                    continue
            elif (partner['kind'] != 'sampled_focal_action'
                    or partner.get('original_action_replay') is True
                    or partner.get('restored_state_evidence') != evidence['same_state_restore_evidence']):
                raise TrainingError('Live original has no independently restored partner: ' + state_id)
        if len({(r['policy_id'], r['reference_id']) for r in members}) != 1:
            raise TrainingError('Different policies in same-state group ' + state_id)
        if len({tuple(r['prompt_token_ids']) for r in members}) != 1:
            raise TrainingError('Different exact model prefixes in same-state group ' + state_id)
        average = sum(r['reward'] for r in members) / len(members)
        distinct_actions = len({(tuple(r['completion_token_ids']),tuple(r['loss_mask'])) for r in members})
        mixed_rewards = len(members) >= 2 and 0 < average < 1
        mixed = mixed_rewards and distinct_actions >= 2
        summary.append(dict(state_id=state_id, count=len(members), reward_mean=average,
                            mixed_rewards=mixed_rewards, distinct_actions=distinct_actions,
                            mixed_outcomes=mixed, branches=[r['branch_id'] for r in members]))
        if mixed_rewards and distinct_actions < 2:
            excluded.extend(dict(branch_id=r['branch_id'],reason='identical-focal-action-different-terminal-outcomes')
                            for r in members)
        if mixed:
            for row in members:
                row['advantage'] = row['reward'] - average
                rows.append(row)
    return rows, dict(groups=summary, excluded=excluded,
                      objective='same-state mean-centered terminal reward; focal action only',
                      valid_group_count=len(summary), mixed_group_count=sum(s['mixed_outcomes'] for s in summary))


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


def probability_comparison(actual, baseline, *, clip_epsilon=0.2):
    """Describe actual execution differences without modifying either vector."""
    if not actual or len(actual) != len(baseline):
        raise TrainingError('Calibration probability vectors must be nonempty and equal length')
    differences = [float(a)-float(b) for a, b in zip(actual, baseline)]
    if not all(math.isfinite(value) for value in differences):
        raise TrainingError('Nonfinite calibration probability')
    absolute = sorted(abs(value) for value in differences)
    def quantile(fraction):
        position = (len(absolute)-1)*fraction
        lower = int(position)
        upper = min(lower+1, len(absolute)-1)
        return absolute[lower]+(absolute[upper]-absolute[lower])*(position-lower)
    ratios = [math.exp(value) for value in differences]
    extreme = max(range(len(differences)), key=lambda index: abs(differences[index]))
    return dict(selected_tokens=len(actual), signed_mean=statistics.mean(differences),
        mean_abs=statistics.mean(absolute), p95_abs=quantile(.95), p99_abs=quantile(.99),
        max_abs=absolute[-1], maximum_selected_index=extreme,
        maximum_actual_logprob=float(actual[extreme]), maximum_baseline_logprob=float(baseline[extreme]),
        action_mean_logratio=statistics.mean(differences), action_sum_logratio=sum(differences),
        ratio_min=min(ratios), ratio_max=max(ratios),
        clipped_fraction=sum(abs(ratio-1)>clip_epsilon for ratio in ratios)/len(ratios),
        sampled_k3=statistics.mean(math.expm1(value)-value for value in differences))


def importance_summary(metrics):
    """Report action-level concentration, including the uncapped diagnostic."""
    logs = [row['sequence_log_importance_weight'] for row in metrics]
    used = [row['sequence_importance_weight_loss_dtype'] for row in metrics]
    scale = max(logs)
    normalized = [math.exp(value-scale) for value in logs]
    def ess(values):
        squared = sum(value*value for value in values)
        return sum(values)**2 / squared if squared else 0.0
    return dict(actions=len(metrics), raw_sequence_ess=ess(normalized),
        raw_sequence_ess_fraction=ess(normalized)/len(metrics),
        used_sequence_ess=ess(used), used_sequence_ess_fraction=ess(used)/len(metrics),
        sequence_log_weight_min=min(logs), sequence_log_weight_max=max(logs),
        used_weight_min=min(used), used_weight_max=max(used),
        upper_capped_actions=sum(row['sequence_importance_weight_capped'] for row in metrics),
        unit='One complete independently sampled focal action, not one token',
        limitation='Describes this selected focal batch; does not establish full-trajectory coverage or an unbiased task-level objective.')


def calibration_rows(path, context_limit):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('format') != 'vfarl-probability-calibration-v1':
        raise TrainingError('Expected explicit real-source probability calibration manifest')
    rows = []
    for source in manifest['samples']:
        source_path = Path(source['source_path'])
        if source_path.is_absolute() or '..' in source_path.parts:
            raise TrainingError('Calibration source must be a frozen repository-relative path')
        if sha256_file(source_path) != source['source_sha256']:
            raise TrainingError('Calibration source file changed: '+str(source_path))
        lines = source_path.read_text().splitlines()
        line_number = source['line']
        if type(line_number) is not int or not 1 <= line_number <= len(lines):
            raise TrainingError('Invalid actual calibration source line')
        raw_line = lines[line_number-1]
        if hashlib.sha256(raw_line.encode()).hexdigest() != source['line_sha256']:
            raise TrainingError('Calibration source line changed')
        row = json.loads(raw_line)
        prompt, completion, mask = (row[key] for key in
            ('prompt_token_ids', 'completion_token_ids', 'loss_mask'))
        if (not prompt or not completion or len(prompt)+len(completion)>context_limit
                or any(type(token) is not int or token<0 for token in prompt+completion)
                or len(mask)!=len(completion) or not any(mask)
                or any(type(value) is not int or value not in (0,1) for value in mask)
                or any(len(row[key])!=len(completion) for key in ('behavior_logprobs','reference_logprobs'))):
            raise TrainingError('Invalid exact-token calibration source')
        row = dict(row, branch_id=str(source_path)+'#'+str(line_number), calibration_source=source)
        rows.append(row)
    if not rows:
        raise TrainingError('Empty calibration manifest')
    return rows, manifest


def require_calibration_weights(rows, identity):
    # Historical records intentionally have an older execution-profile ID.
    # Weight/tokenizer/schema identity must still agree field by field.
    keys = ('nf4_cache_manifest_sha256','tokenizer_sha256','template_sha256',
            'tool_schema_sha256','adapter_sha256')
    actual = identity['policy_identity']
    for row in rows:
        recorded = row.get('policy_identity', {})
        if any(recorded.get(key) != actual.get(key) or key not in actual for key in keys):
            raise TrainingError('Calibration tokens belong to different weights or protocol')


def actor_learner_binding(rows, identity, *, actor_backend, rollout_correction):
    """Bind identical weights/protocol while naming both execution paths.

    Native collection retains the original exact identity requirement. An
    explicitly selected Unsloth actor may differ only in probability execution;
    its real behavior probabilities remain the importance-sampling denominator.
    This does not make the native FSDP learner an Unsloth training backend.
    """
    if actor_backend == 'native':
        for row in rows:
            if (row['policy_id'] != identity['policy_id']
                    or row['reference_id'] != identity['reference_id']):
                raise TrainingError('Sampled behavior/reference policy differs from actual initial trainer')
        return dict(actor_backend='native', learner_backend='native-fsdp',
                    behavior_policy_id=identity['policy_id'],
                    behavior_reference_id=identity['reference_id'],
                    learner_policy_id=identity['policy_id'],
                    learner_reference_id=identity['reference_id'],
                    same_execution_identity=True)
    if actor_backend != 'unsloth' or rollout_correction != 'sequence_tis':
        raise TrainingError('Unsloth actor/native learner requires explicit sequence_tis correction')
    from .vfarl_unsloth import UNSLOTH_EXECUTION_PROFILE
    from .vfarl_batched import BATCHED_UNSLOTH_EXECUTION_PROFILE, SESSION_BATCHED_EXECUTION_PROFILE
    # Preserve the already sampled auto-r3 behavior path. This literal is
    # extracted from its frozen source bundle (vfarl_unsloth.py SHA256
    # 2855de90b8b9aa3b3e1f007204d391a9dedfd5497ed639b1f8815f4920c9e77a).
    # Historical dispatch labels remain historical metadata, not a claim that
    # those kernels were active. Never relabel these probabilities as v2.
    frozen_v1_profile = {
        'backend': 'unsloth',
        'version': 1,
        'prompt_prefill': 'complete-prefix-contiguous-256-token-KV-chunks',
        'completion': 'one-token-cached-autoregressive-forward',
        'nf4_linear': 'Unsloth-loader-native-NF4-kernels-no-VFARL-fallback-override',
        'head': 'Unsloth-native-tied-head-no-VFARL-FP32-head-override',
        'logprob': 'actual-returned-logits-to-float32-log-softmax',
        'reference': 'independent-same-autoregressive-path-with-zero-LoRA-scaling',
        'forced_controls': 'preserve-in-prefix-and-mask-from-loss',
        'compute_dtype': 'bfloat16',
        'fast_inference': False,
        'quantization': 'same-prequantized-official-QAT-derived-NF4-cache',
        'adapter_initialization': 'copy-exact-native-PEFT-CPU-shadow-initialization',
    }
    known_profiles = {
        digest(UNSLOTH_EXECUTION_PROFILE): UNSLOTH_EXECUTION_PROFILE,
        digest(BATCHED_UNSLOTH_EXECUTION_PROFILE): BATCHED_UNSLOTH_EXECUTION_PROFILE,
        digest(SESSION_BATCHED_EXECUTION_PROFILE): SESSION_BATCHED_EXECUTION_PROFILE,
        '1f84b034b728b36dfb47466e34b2b5c0a667e40a36bf33a4d5fbff7263971c5b': frozen_v1_profile,
    }
    execution_fields = {'probability_execution_sha256', 'probability_runtime_sha256'}
    without_execution = lambda value: {key: item for key, item in value.items()
                                       if key not in execution_fields}
    policies = set()
    runtime = None
    for row in rows:
        actor, reference = row.get('policy_identity'), row.get('reference_identity')
        if not isinstance(actor, dict) or not isinstance(reference, dict):
            raise TrainingError('Missing complete Unsloth actor/reference identity')
        if digest(actor) != row['policy_id'] or digest(reference) != row['reference_id']:
            raise TrainingError('Recorded actor/reference identity digest differs from its payload')
        profile_hash = actor.get('probability_execution_sha256')
        profile = known_profiles.get(profile_hash)
        if profile is None:
            raise TrainingError('Focal data uses an unknown Unsloth probability profile')
        if (without_execution(actor) != without_execution(identity['policy_identity'])
                or without_execution(reference) != without_execution(identity['reference_identity'])):
            raise TrainingError('Unsloth actor and native learner have different weights, adapter or protocol')
        if (actor.get('probability_execution_sha256') != profile_hash
                or reference.get('probability_execution_sha256') != profile_hash
                or row.get('probability_execution_sha256') != profile_hash):
            raise TrainingError('Focal data does not use the declared Unsloth probability profile')
        execution = row.get('execution', {})
        recorded = execution.get('probability_runtime', {})
        runtime_identity = recorded.get('identity')
        if (execution.get('backend') != 'unsloth'
                or execution.get('probability_execution_profile') != profile
                or not isinstance(runtime_identity, dict)
                or runtime_identity.get('profile') != profile
                or recorded.get('identity_sha256') != digest(runtime_identity)
                or actor.get('probability_runtime_sha256') != recorded['identity_sha256']
                or reference.get('probability_runtime_sha256') != recorded['identity_sha256']):
            raise TrainingError('Unsloth actor runtime provenance differs from its policy identity')
        policies.add((row['policy_id'], row['reference_id']))
        runtime = recorded
    if len(policies) != 1:
        raise TrainingError('One update requires a single frozen Unsloth behavior/reference policy')
    behavior, reference = next(iter(policies))
    return dict(actor_backend='unsloth', learner_backend='native-fsdp',
                actor_execution_profile_version=profile['version'],
                actor_execution_profile_sha256=profile_hash,
                behavior_policy_id=behavior, behavior_reference_id=reference,
                behavior_policy_identity=rows[0]['policy_identity'],
                behavior_reference_identity=rows[0]['reference_identity'],
                behavior_runtime=runtime,
                learner_policy_id=identity['policy_id'],
                learner_reference_id=identity['reference_id'],
                same_execution_identity=False, rollout_correction=rollout_correction,
                identity_difference='Probability execution profile/runtime only; all other fields match exactly',
                reference_interpretation='Actor reference probabilities are preserved as diagnostics; the KL target is a separate frozen native learner reference forward')


def calibrate_actor(args, cfg):
    from .vfarl_policy import NF4Policy
    policy_class = NF4Policy
    if args.actor_backend == 'unsloth':
        from .vfarl_unsloth import UnslothPolicy
        policy_class = UnslothPolicy
    rows, manifest = calibration_rows(args.data, cfg['max_sequence_length'])
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    policy = policy_class(args.config, inference_device_count=2)
    initial = trainable_actor_hash = policy.identity['policy_identity']['adapter_sha256']
    results = []
    try:
        require_calibration_weights(rows,policy.identity)
        for row in rows:
            tokens = (row['prompt_token_ids'], row['completion_token_ids'], row['loss_mask'])
            first = policy.score_teacher_forced(*tokens)
            repeated = policy.score_teacher_forced(*tokens)
            reference = policy.score_teacher_forced(*tokens, frozen_reference=True)
            blocked_reference = policy.score_cached_reference(*tokens)
            selected = lambda values: [value for value, keep in zip(values, row['loss_mask']) if keep]
            old = selected(row['behavior_logprobs'])
            current = selected(first['logprobs'])
            result = dict(source=row['calibration_source'], optimizer_steps=0,
                original_behavior_logprobs=row['behavior_logprobs'],
                original_reference_logprobs=row['reference_logprobs'],
                active_first=first, active_repeat=repeated, independent_reference=reference,
                blocked_reference=blocked_reference,
                comparisons=dict(replay_vs_original_behavior=probability_comparison(current, old),
                    repeat_vs_first=probability_comparison(selected(repeated['logprobs']), current),
                    reference_vs_active=probability_comparison(selected(reference['logprobs']), current),
                    blocked_vs_matched_reference=probability_comparison(
                        selected(blocked_reference['logprobs']), selected(reference['logprobs']))))
            atomic_json(output/('sample-%02d.json'%len(results)), result)
            results.append(result)
            print(json.dumps(dict(phase='actor_calibration_sample', source=row['calibration_source'],
                                  comparisons=result['comparisons'])), flush=True)
        from .train_ddp import trainable_hash
        final = trainable_hash(policy.model)
        if initial != final:
            raise TrainingError('No-update actor calibration changed LoRA parameters')
        atomic_json(output/'calibration_manifest.json', dict(mode='actor', optimizer_steps=0,
            source_manifest=manifest, source_manifest_sha256=sha256_file(Path(args.data)),
            policy=policy.identity, execution=policy.execution,
            adapter_sha256_before=trainable_actor_hash, adapter_sha256_after=final,
            comparisons=[row['comparisons'] for row in results],
            limitation='Real calibration of these exact shapes; not a reward update or a universal numerical bound.'))
    finally:
        policy.close()
    return 0


def calibrate_fsdp(args, cfg):
    focal = args.calibrate == 'fsdp-focal'
    if focal:
        rows, evidence = prepare_rows(args.data, cfg['max_sequence_length'])
        manifest = dict(format='vfarl-focal-probability-calibration-v1', evidence=evidence)
        if not rows:
            print(json.dumps(dict(optimizer_steps=0, reason='No verified mixed-outcome focal groups',
                                  **evidence)), flush=True)
            return 0
    else:
        rows, manifest = calibration_rows(args.data, cfg['max_sequence_length'])
    _, rl = read_config(args.config)
    stack = single.require_stack('nf4')
    torch, dist = stack['torch'], stack['torch'].distributed
    if int(os.environ.get('WORLD_SIZE','1')) != 4:
        raise TrainingError('FSDP calibration requires exactly four ranks')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    cfg['device'] = 'cuda:%d'%local_rank
    dist.init_process_group('nccl', timeout=timedelta(seconds=900))
    rank, world = dist.get_rank(), dist.get_world_size()
    output = Path(args.out)
    collective_call(torch, lambda: output.mkdir(parents=True, exist_ok=False) if rank==0 else None)
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    model, tokenizer, device, info = single.load_model_and_tokenizer(cfg, stack, adapter_init_seed=cfg['seed'])
    bind_template(tokenizer, cfg);require_zero_lora(model, torch)
    info['probability_execution'] = install_probability_execution(model,stack)
    identity = identities(model, tokenizer, info, cfg)
    binding = None
    def bind_sources():
        nonlocal binding
        if focal:
            binding = actor_learner_binding(rows, identity, actor_backend=args.actor_backend,
                                           rollout_correction=rl['rollout_correction'])
        else:
            require_calibration_weights(rows, identity)
    collective_call(torch, bind_sources)
    model.eval()
    checkpoint_layers = install_eval_checkpointing(model) if cfg['gradient_checkpointing'] else 0
    install_logprob_forward(model, rl['logprob_chunk_size'])
    storage = pack_floating_nf4_storage(model, torch, stack['bitsandbytes'], torch.bfloat16)
    os.environ.setdefault('FSDP_TRANSFORMER_CLS_TO_WRAP','Gemma4TextDecoderLayer')
    wrapped, units = wrap_fsdp(model, torch, device);wrapped.eval()
    before = snapshot_hash(shard_snapshot(wrapped))
    records=[]
    torch.cuda.reset_peak_memory_stats(device)
    for start in range(0,len(rows),world):
        index=start+rank;real=index<len(rows);row=rows[index] if real else rows[start]
        batch, old, stored_reference = batch_for(row,torch,device)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            first=wrapped(**batch).detach()
            repeated=wrapped(**batch).detach()
            with frozen_base_reference(model): reference=wrapped(**batch).detach()
        with torch.autocast('cuda',dtype=torch.bfloat16): current=wrapped(**batch)
        values=current.detach().cpu().tolist()
        # Complete the actual FSDP forward/backward lifecycle with zero loss.
        # There is no optimizer and no reward/advantage in this experiment.
        (current.sum()*0.0).backward()
        wrapped.zero_grad(set_to_none=True)
        if real:
            baseline=first.cpu().tolist()
            sequence_log_is = float((first.double()-old.double()).sum())
            used_is = math.exp(min(sequence_log_is, math.log(rl['rollout_is_max'])))
            records.append(dict(source=(dict(branch_id=row['branch_id'], state_id=row['state_id'],
                    instance_id=row.get('instance_id')) if focal else row['calibration_source']),
                original_behavior_logprobs=old.cpu().tolist(),
                original_reference_logprobs=stored_reference.cpu().tolist(),
                active_first=baseline, active_repeat=repeated.cpu().tolist(),
                frozen_reference=reference.cpu().tolist(), grad_enabled=values,
                sequence_log_importance_weight=sequence_log_is,
                sequence_importance_weight_raw=(math.exp(sequence_log_is) if sequence_log_is<700 else None),
                sequence_importance_weight_loss_dtype=used_is,
                sequence_importance_weight_capped=sequence_log_is>math.log(rl['rollout_is_max']),
                sequence_tokens=len(row['prompt_token_ids'])+len(row['completion_token_ids']),
                comparisons=dict(current_vs_original_behavior=probability_comparison(baseline,old.cpu().tolist()),
                    repeat_vs_first=probability_comparison(repeated.cpu().tolist(),baseline),
                    reference_vs_current=probability_comparison(reference.cpu().tolist(),baseline),
                    grad_vs_nograd=probability_comparison(values,baseline))))
        del batch,first,repeated,reference,current,old,stored_reference
    torch.cuda.synchronize(device)
    after=snapshot_hash(shard_snapshot(wrapped))
    if before!=after:raise TrainingError('Zero-update FSDP calibration changed LoRA parameters')
    local=dict(rank=rank,optimizer_steps=0,zero_loss_backward=True,
        parameter_shard_before_sha256=before,parameter_shard_after_sha256=after,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),records=records)
    atomic_json(output/('rank-%03d.json'%rank),local)
    gathered=[None]*world;dist.all_gather_object(gathered,local)
    if rank==0:atomic_json(output/'calibration_manifest.json',dict(mode=args.calibrate,optimizer_steps=0,
        source_manifest=manifest,source_manifest_sha256=sha256_file(Path(args.data)),
        policy=identity,actor_learner_binding=binding,model=info,storage=storage,
        importance_weights=importance_summary([record for result in gathered for record in result['records']]),
        rollout_correction=rl['rollout_correction'], rollout_is_max=rl['rollout_is_max'],
        importance_interpretation='Diagnostic complete-focal-action learner/actor ratios; no optimizer or weighted reward loss is executed in calibration',
        checkpointed_decoder_layers=checkpoint_layers,
        rank0_wrap_units=units,ranks=gathered,
        limitation='No reward, optimizer, or LoRA update. Measured differences apply only to the frozen source records.'))
    dist.barrier();dist.destroy_process_group()
    return 0


def run(args):
    cfg, rl = read_config(args.config)
    if args.calibrate == 'actor':
        return calibrate_actor(args,cfg)
    if args.calibrate in ('fsdp', 'fsdp-focal'):
        return calibrate_fsdp(args,cfg)
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
    info['probability_execution'] = install_probability_execution(model,stack)
    identity = identities(model, tokenizer, info, cfg)

    binding = None
    def verify_identity():
        nonlocal binding
        binding = actor_learner_binding(rows, identity, actor_backend=args.actor_backend,
                                       rollout_correction=rl['rollout_correction'])
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
    before_current_nograd, repeated_current_nograd = {}, {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, len(rows), world):
        index = start + rank
        real = index < len(rows)
        row = rows[index] if real else rows[start]
        batch, old, stored_ref = batch_for(row, torch, device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            current_nograd = wrapped(**batch).detach()
            repeated_nograd = wrapped(**batch).detach()
            with frozen_base_reference(model):
                ref = wrapped(**batch).detach()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            current = wrapped(**batch)
        old_error = float((current.detach() - old).abs().max())
        ref_error = float((ref - stored_ref).abs().max())
        corrected = rl['rollout_correction']=='sequence_tis'
        same_path_error = float((current.detach()-current_nograd).abs().max())
        repeat_error = float((repeated_nograd-current_nograd).abs().max())
        # Strict mode retains the original cross-engine check. An explicitly
        # selected correction uses an actual frozen learner baseline and keeps
        # the cross-engine discrepancy as data, never overwriting actor old.
        errors = torch.tensor([same_path_error,repeat_error] if corrected else
                              [old_error,ref_error],device=device)
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        if float(errors.max()) > rl['max_old_logprob_error']:
            atomic_json(output/('probability-mismatch-rank-%03d.json'%rank),dict(
                optimizer_steps=0, source_data_sha256=sha256_file(Path(args.data)),
                branch_id=row['branch_id'], state_id=row['state_id'], real_row=real,
                sequence_tokens=len(row['prompt_token_ids'])+len(row['completion_token_ids']),
                original_behavior_logprobs=old.detach().cpu().tolist(),
                stored_reference_logprobs=stored_ref.detach().cpu().tolist(),
                current_nograd_logprobs=current_nograd.detach().cpu().tolist(),
                current_grad_logprobs=current.detach().cpu().tolist(),
                actual_reference_logprobs=ref.detach().cpu().tolist(),
                current_vs_behavior=probability_comparison(current.detach().cpu().tolist(),old.cpu().tolist()),
                actual_vs_stored_reference=probability_comparison(ref.cpu().tolist(),stored_ref.cpu().tolist()),
                global_max_errors=errors.cpu().tolist(), policy=identity,
                meaning='Actual pre-optimizer numerical mismatch; no parameter update was attempted.'))
            raise TrainingError('Captured vs update logprob mismatch: ' + str(errors.tolist()))
        learner_old = current_nograd.detach()
        sequence_log_is = float((learner_old.double()-old.double()).sum())
        if corrected:
            is_weight = math.exp(min(sequence_log_is,math.log(rl['rollout_is_max'])))
            ratio = (current-learner_old).exp()
        else:
            is_weight = 1.0
            ratio = (current-old).exp()
        advantage = row['advantage']
        policy_loss = -torch.minimum(ratio * advantage,
                                    ratio.clamp(1 - rl['clip_epsilon'], 1 + rl['clip_epsilon']) * advantage)
        log_ref_ratio = ref - current
        kl = torch.expm1(log_ref_ratio) - log_ref_ratio
        per_token = policy_loss + rl['kl_beta'] * kl
        importance = per_token.new_tensor(is_weight)
        if corrected:
            valid_importance = torch.isfinite(importance).logical_and(importance>0).to(torch.int32)
            dist.all_reduce(valid_importance,op=dist.ReduceOp.MIN)
            if not int(valid_importance):
                raise TrainingError('The complete focal-action importance weight underflowed in loss dtype; preserve the pair without updating')
        loss = per_token.mean()*importance
        finite = torch.tensor(int(bool(torch.isfinite(loss).all())), device=device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not int(finite):
            raise TrainingError('Nonfinite PPO/KL loss')
        (loss * (world / denominator if real else 0.0)).backward()
        if real:
            before_logprobs[row['branch_id']] = current.detach().cpu().tolist()
            reference_logprobs[row['branch_id']] = ref.detach().cpu().tolist()
            before_current_nograd[row['branch_id']] = current_nograd.cpu().tolist()
            repeated_current_nograd[row['branch_id']] = repeated_nograd.cpu().tolist()
            metrics.append(dict(branch_id=row['branch_id'], state_id=row['state_id'],
                                reward=row['reward'], advantage=advantage, loss=float(loss.detach()),
                                policy_loss=float(policy_loss.detach().mean()), kl=float(kl.detach().mean()),
                                ratio_mean=float(ratio.detach().mean()),
                                clipped_fraction=float(((ratio.detach() - 1).abs() > rl['clip_epsilon']).float().mean()),
                                old_logprob_max_abs_error=old_error, ref_logprob_max_abs_error=ref_error,
                                rollout_correction=rl['rollout_correction'],
                                sequence_log_importance_weight=sequence_log_is,
                                sequence_importance_weight_raw=(math.exp(sequence_log_is)
                                    if sequence_log_is<700 else None),
                                sequence_importance_weight_used=is_weight,
                                sequence_importance_weight_loss_dtype=float(importance.detach()),
                                sequence_importance_weight_dtype=str(importance.dtype),
                                sequence_importance_weight_capped=(corrected and sequence_log_is>math.log(rl['rollout_is_max'])),
                                actor_behavior_logprobs=old.cpu().tolist(),
                                actor_reference_logprobs=stored_ref.cpu().tolist(),
                                frozen_learner_old_logprobs=learner_old.cpu().tolist(),
                                learner_grad_vs_nograd_max_abs_error=same_path_error,
                                learner_repeat_max_abs_error=repeat_error,
                                nograd_repeat=probability_comparison(repeated_nograd.cpu().tolist(),
                                    current_nograd.cpu().tolist()),
                                grad_vs_nograd=probability_comparison(current.detach().cpu().tolist(),
                                    current_nograd.cpu().tolist()),
                                sequence_tokens=len(row['prompt_token_ids']) + len(row['completion_token_ids']),
                                selected_tokens=old.numel()))
        del current, current_nograd, repeated_nograd, loss, per_token, ratio, kl, batch
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
        before_current_nograd_logprobs=before_current_nograd,
        repeated_current_nograd_logprobs=repeated_current_nograd,
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
        all_metrics = [row for r in range(world) for row in
            json.loads((output/('rank-%03d-update.json'%r)).read_text())['metrics']]
        committed = dict(
            status='optimizer-step-and-adapter-committed', optimizer_steps=1,
            post_update_diagnostics_completed=False,
            post_update_diagnostics='not yet executed at the commit boundary',
            source_data_sha256=sha256_file(Path(args.data)), config=cfg, rl=rl,
            evidence=evidence, policy=identity, actor_learner_binding=binding, model=info, storage=storage,
            checkpointed_decoder_layers=checkpoint_layers, rank0_wrap_units=units,
            global_parameter_delta_l2=float(local_change[0].sqrt()),
            global_changed_parameter_elements=int(local_change[1]),
            grad_norm=float(grad_norm), adapter_files=adapter_files,
            importance_weights=importance_summary(all_metrics),
            rank_update_files=[dict(path='rank-%03d-update.json' % r,
                                    sha256=sha256_file(output / ('rank-%03d-update.json' % r)))
                               for r in range(world)],
            wall_seconds_to_commit=time.monotonic() - started,
            frozen_reference='Actual separate zero-LoRA-scaling forwards; fixed base reference with unchanged FSDP traversal',
            recovery='The optimizer step is already complete. Reuse this adapter and the saved exact-token behavior data for separate diagnostics; do not repeat the training update.',
            limitation='Parameter changes are committed; post-update likelihood direction and re-rollout outcomes have not yet been measured.')
        committed['rollout_correction_interpretation'] = (
            'Actor old remains immutable. If sequence_tis is explicitly enabled, '
            'rho=stopgrad(exp(sum(mask*(learner_old-actor_old)))) capped above at rollout_is_max; '
            'PPO ratio=current_learner/frozen_learner_old. The same action weight scales the '
            'length-normalized policy and sampled k3 terms. This is a truncated, focal Q_actor '
            'surrogate, not an unbiased full-episode policy gradient or an exact current-policy KL.')
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
            baseline = torch.tensor(before_current_nograd[row['branch_id']])
            repeated = torch.tensor(repeated_current_nograd[row['branch_id']])
            post_reference_ratio = reference - post
            comparisons.append(dict(branch_id=row['branch_id'], reward=row['reward'], advantage=row['advantage'],
                                    baseline_source='Actual current-policy no-grad forward with matching post-update execution path; repeat measured before any parameter update',
                                    before_logprobs=baseline.tolist(), after_logprobs=post.tolist(),
                                    repeated_before_logprobs=repeated.tolist(),
                                    grad_forward_before_logprobs=pre.tolist(),
                                    reference_logprobs=reference.tolist(),
                                    initial_grad_vs_nograd_mean_delta=float((pre - baseline).mean()),
                                    initial_grad_vs_nograd_max_abs_delta=float((pre - baseline).abs().max()),
                                    initial_reference_vs_current_mean_delta=float((reference - baseline).mean()),
                                    initial_reference_vs_current_max_abs_delta=float((reference - baseline).abs().max()),
                                    repeat_mean_logprob_delta=float((repeated-baseline).mean()),
                                    repeat_mean_abs_delta=float((repeated-baseline).abs().mean()),
                                    post_reference_k3_on_behavior_tokens=float(
                                        (post_reference_ratio.exp() - post_reference_ratio - 1).mean()),
                                    mean_logprob_delta=float((post - baseline).mean()),
                                    advantage_weighted_mean_delta=float((post - baseline).mean()) * row['advantage']))
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
        all_metrics = [row for result in gathered for row in result['metrics']]
        atomic_json(output / 'training_manifest.json', dict(
            status='one-verified-focal-PPO-update-completed', optimizer_steps=1,
            post_update_diagnostics_completed=True,
            optimizer_step_manifest='optimizer-step.json',
            optimizer_step_manifest_sha256=sha256_file(output / 'optimizer-step.json'),
            source_data_sha256=sha256_file(Path(args.data)), config=cfg, rl=rl,
            evidence=evidence, policy=identity, actor_learner_binding=binding, model=info, storage=storage,
            checkpointed_decoder_layers=checkpoint_layers, rank0_wrap_units=units,
            global_parameter_delta_l2=float(local_change[0].sqrt()),
            global_changed_parameter_elements=int(local_change[1]),
            wall_seconds=time.monotonic() - started, ranks=gathered,
            rollout_correction=rl['rollout_correction'],
            importance_weights=importance_summary(all_metrics),
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
    parser.add_argument('--actor-backend', choices=('native', 'unsloth'), default='native',
                        help='Actual sampling backend; Unsloth/native-FSDP uses explicit execution identities and sequence_tis')
    parser.add_argument('--learner-backend', choices=('native',), default='native',
                        help='The memory-sharded native FSDP learner; Unsloth training kernels are not yet integrated')
    parser.add_argument('--calibrate', choices=('actor','fsdp','fsdp-focal'),
                        help='Score real source tokens without a reward or optimizer step')
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
