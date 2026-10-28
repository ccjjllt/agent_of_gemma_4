"""One real VFARL update using Unsloth backward and fused selected logprobs.

Run as one process over one or two visible GPUs. This keeps the existing
verified focal-action PPO objective, not an SFT or replacement GRPO dataset.
No model is loaded on import. GPU throughput and 16K backward capacity require
actual measurements; the collector's inference memory is not training evidence.
"""
import argparse
from contextlib import contextmanager
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback

from .common import TrainingError, atomic_json, sha256_file
from .fsdp_qlora import _write_adapter_state
from .train_vfarl import (actor_learner_binding, importance_summary, prepare_rows,
                         probability_comparison, shard_snapshot, snapshot_hash,
                         focal_reduction_denominator, focal_objective_description,
                         reduce_focal_terms, focal_action_denominator)
from .vfarl_policy import digest, frozen_base_reference, identities, read_config, plain_lora_scaling


def focal_predictors(policy, row):
    """Retain the complete context and select only sampled-token predictors."""
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
    return values, targets, cap


def focal_logprobs(policy, row, fused_linear_logprobs, *, return_predictors=False):
    """Project only selected predictors, retaining all prefix/control tokens."""
    values, targets, cap = focal_predictors(policy, row)
    base = policy.model.get_base_model()
    scores = fused_linear_logprobs(values, base.lm_head.weight, targets,
        bias=getattr(base.lm_head, 'bias', None), temperature=1.0,
        logit_softcapping=cap or 0.0,
        max_rows_per_chunk=policy.rl['logprob_chunk_size'])
    return (scores, values, targets, cap) if return_predictors else scores


@contextmanager
def exact_head_math(torch):
    """Use FP32 GEMM for the oracle, restoring the sampled runtime's math."""
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.set_float32_matmul_precision(precision)


def head_numerics(torch, hidden, weight, targets, fused, *, bias=None, cap=0.0,
                  coefficients=None):
    """Measure the real head's values and hidden gradients without changing it.

    The FP32 oracle streams vocabulary blocks, avoiding an FP32 copy of the
    entire tied 31B embedding. It uses an analytic log-softmax derivative.
    CUDA/Triton conclusions require actually running this with the real head.
    """
    hidden = hidden.detach().to(weight.device)
    targets = targets.detach().to(weight.device)
    if hidden.ndim != 2 or not 1 <= hidden.shape[0] <= 8:
        raise TrainingError('Head study is bounded to one through eight real predictor rows')
    if weight.requires_grad or (bias is not None and bias.requires_grad):
        raise TrainingError('Head study requires a frozen vocabulary projection')
    if coefficients is None:
        coefficients = torch.linspace(-0.5, 0.5, hidden.shape[0], device=weight.device)
        if hidden.shape[0] == 1:
            coefficients.fill_(1.0)
    else:
        coefficients = coefficients.detach().to(device=weight.device, dtype=torch.float32)
        if coefficients.shape != targets.shape or not bool(torch.isfinite(coefficients).all()):
            raise TrainingError('Head study coefficients must match the actual selected targets')
    measured = {}
    for name in ('fused', 'native_bf16'):
        value = hidden.clone().requires_grad_(True)
        with torch.autocast(weight.device.type, dtype=torch.bfloat16):
            if name == 'fused':
                scores = fused(value, weight, targets, bias=bias, temperature=1.0,
                    logit_softcapping=cap, max_rows_per_chunk=hidden.shape[0])
            else:
                logits = torch.nn.functional.linear(value, weight, bias)
                if cap:
                    logits = (logits / cap).tanh() * cap
                scores = torch.log_softmax(logits.float(), -1).gather(
                    1, targets[:, None]).squeeze(1)
        gradient, = torch.autograd.grad((scores * coefficients).sum(), value)
        measured[name] = (scores.detach().float(), gradient.detach().float())
        del value, scores, gradient
    with torch.no_grad(), torch.autocast(weight.device.type, enabled=False), exact_head_math(torch):
        h = hidden.float()
        lse = torch.full((len(h),), -math.inf, device=h.device)
        chosen = torch.empty_like(lse)
        block = 4096
        for start in range(0, weight.shape[0], block):
            end = min(start + block, weight.shape[0])
            logits = torch.nn.functional.linear(h, weight[start:end].float(),
                bias[start:end].float() if bias is not None else None)
            if cap:
                logits = (logits / cap).tanh() * cap
            lse = torch.logaddexp(lse, torch.logsumexp(logits, -1))
            selected = (targets >= start) & (targets < end)
            chosen[selected] = logits[selected, targets[selected] - start]
        exact_scores = chosen - lse
        exact_gradient = torch.zeros_like(h)
        for start in range(0, weight.shape[0], block):
            end = min(start + block, weight.shape[0])
            w = weight[start:end].float()
            raw = torch.nn.functional.linear(h, w,
                bias[start:end].float() if bias is not None else None)
            if cap:
                capped = (raw / cap).tanh()
                logits, derivative = capped * cap, 1 - capped.square()
            else:
                logits, derivative = raw, 1.0
            derivative_logits = -(logits - lse[:, None]).exp()
            selected = (targets >= start) & (targets < end)
            derivative_logits[selected, targets[selected] - start] += 1
            exact_gradient.add_((derivative_logits * derivative * coefficients[:, None]) @ w)
        measured['fp32_blocked'] = (exact_scores, exact_gradient)
    def comparison(actual, reference):
        av, ag = measured[actual]; rv, rg = measured[reference]
        denominator = float(rg.norm())
        return dict(logprob_max_abs=float((av-rv).abs().max()),
            hidden_gradient_cosine=float(torch.nn.functional.cosine_similarity(
                ag.reshape(1, -1), rg.reshape(1, -1)).item()),
            hidden_gradient_relative_l2=float((ag-rg).norm()) / max(denominator, 1e-30),
            hidden_gradient_max_abs=float((ag-rg).abs().max()))
    return dict(rows=len(hidden), vocabulary=weight.shape[0], hidden_size=hidden.shape[1],
        hidden_dtype=str(hidden.dtype), weight_dtype=str(weight.dtype), softcap=cap,
        oracle_math='FP32 blocked GEMM, highest precision, TF32 disabled; caller math restored',
        coefficients=coefficients.cpu().tolist(),
        logprobs={name: value[0].cpu().tolist() for name, value in measured.items()},
        fused_vs_native=comparison('fused', 'native_bf16'),
        fused_vs_fp32=comparison('fused', 'fp32_blocked'),
        native_vs_fp32=comparison('native_bf16', 'fp32_blocked'),
        limitation='Hidden states are detached for head isolation; this comparison alone does not validate the NF4 backbone backward or task learning.')


def focal_directional_numerics(policy, rows, anchors, fused, rl, action_denominator):
    """Measure central differences in a descent direction, then restore bytes.

    These are transient parameter probes, not optimizer updates. Both sides use
    frozen old/reference/IS values; no behavior probabilities are replaced.
    Every parameter is restored even if a forward raises. No probe is exported.
    """
    torch, model = policy.torch, policy.model
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    if not named or any(parameter.dtype != torch.float32 for _, parameter in named):
        raise TrainingError('Exact transient probes require the current FP32 trainable LoRA storage')
    before = shard_snapshot(model)
    # Sign descent is the new-Adam first-step direction apart from epsilon.
    # It lets probe RMS values have the same units as a declared Adam rate.
    derivative = -sum(float(parameter.grad.detach().double().abs().sum())
        for _, parameter in named if parameter.grad is not None)
    nonzero = sum(int(torch.count_nonzero(parameter.grad))
        for _, parameter in named if parameter.grad is not None)
    parameter_count = sum(parameter.numel() for _, parameter in named)

    def place(amount):
        with torch.no_grad():
            for name, parameter in named:
                parameter.copy_(before[name].to(parameter.device, dtype=parameter.dtype))
                if amount and parameter.grad is not None:
                    parameter.add_(parameter.grad.sign(), alpha=-amount)

    def evaluate():
        total = 0.0
        with torch.no_grad(), torch.autocast(policy.device.type, dtype=torch.bfloat16):
            for row in rows:
                # Match the gradient forward's kernels/checkpointing path.
                # Detaching immediately releases its graph without backward
                # or accumulation into the already measured parameter grads.
                with torch.enable_grad():
                    value = focal_logprobs(policy, row, fused).detach()
                anchor = anchors[row['branch_id']]
                old = value.new_tensor(anchor['old'])
                reference = value.new_tensor(anchor['reference'])
                ratio = (value-old).exp()
                advantage = row['advantage']
                pg = -torch.minimum(ratio*advantage,
                    ratio.clamp(1-rl['clip_epsilon'], 1+rl['clip_epsilon'])*advantage)
                gap = reference-value
                loss = reduce_focal_terms(pg+rl['kl_beta']*(torch.expm1(gap)-gap), rl)
                total += float((loss * value.new_tensor(anchor['importance'])) / action_denominator)
        if not math.isfinite(total):
            raise TrainingError('Nonfinite directional-probe objective')
        return total

    probes = []
    failure = None
    try:
        baseline = evaluate()
        for epsilon in (1e-6, 5e-6):
            place(epsilon)
            plus = evaluate()
            place(-epsilon)
            minus = evaluate()
            probes.append(dict(epsilon=epsilon, objective_plus_descent=plus,
                objective_minus_descent=minus, measured_descent_change=plus-baseline,
                first_order_descent_change=epsilon*derivative,
                requested_parameter_step_rms=epsilon*math.sqrt(nonzero/parameter_count),
                central_difference_derivative=(plus-minus)/(2*epsilon),
                predicted_derivative=derivative))
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            place(0.0)
            if snapshot_hash(shard_snapshot(model)) != snapshot_hash(before):
                raise TrainingError('Transient numerical probes did not restore exact adapter bytes')
        except Exception as restoration_error:
            raise TrainingError('Transient probe restoration failed; original forward error: '
                + repr(failure)) from restoration_error
    return dict(direction='negative sign of the measured, unclipped focal objective gradient',
        forward_mode='grad-enabled forward, immediately detached; no probe backward',
        nonzero_direction_elements=nonzero, trainable_parameter_count=parameter_count,
        objective_at_anchor=baseline, probes=probes,
        adapter_bytes_restored=True, optimizer_steps=0,
        limitation='Finite differences in BF16 can be nonsmooth. Sign descent is not a resumed-Adam direction; no step is accepted, selected, exported or used for training.')


def head_probe_positions(tokenizer, row):
    """Choose at most four existing selected rows using source-token structure.

    This is only a diagnostic slice. Selection never reads rewards, probabilities
    or gradients, and never adds a forced/masked target to the objective.
    """
    tokens, mask = row['completion_token_ids'], row['loss_mask']
    selected = [index for index, keep in enumerate(mask) if keep]
    if len(tokens) != len(mask) or not selected:
        raise TrainingError('Head diagnostics require actual selected completion tokens')

    def control(piece):
        ids = tokenizer.encode(piece, add_special_tokens=False)
        if (len(ids) == 1 and tokenizer.decode(ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False) == piece):
            return ids[0]
        return None

    opening, closing, tool_open, tool_close, quote, turn_close = [control(piece)
        for piece in ('<|channel>', '<channel|>', '<|tool_call>', '<tool_call|>', '<|"|>', '<turn|>')]
    controls = {value for value in (opening, closing, tool_open, tool_close, quote, turn_close)
                if value is not None}
    regions = ['other'] * len(tokens)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if opening is not None and closing is not None and token == opening:
            end = next((i for i in range(index + 1, len(tokens)) if tokens[i] == closing), len(tokens))
            regions[index] = 'protocol_control'
            # Decode only the observed lexical header, without re-tokenizing
            # the response. Alternate source-token splits can still spell it.
            header_end = None
            expected = 'thought\n'
            for stop in range(index + 2, min(end, index + 17) + 1):
                text = tokenizer.decode(tokens[index + 1:stop], skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
                if text == expected:
                    header_end = stop
                    break
                if not expected.startswith(text):
                    break
            if header_end is not None:
                regions[index + 1:header_end] = ['channel_header'] * (header_end - index - 1)
                regions[header_end:end] = ['thought_body'] * (end - header_end)
            if end < len(tokens):
                regions[end] = 'protocol_control'
            # Tool markers inside a channel body are not emitted tool calls.
            index = end + 1
        elif tool_open is not None and tool_close is not None and token == tool_open:
            end = next((i for i in range(index + 1, len(tokens)) if tokens[i] == tool_close), len(tokens))
            regions[index] = 'protocol_control'
            for i in range(index + 1, end):
                regions[i] = 'protocol_control' if tokens[i] in controls else 'tool_body'
            if end < len(tokens):
                regions[end] = 'protocol_control'
            index = end + 1
        else:
            if token in controls:
                regions[index] = 'protocol_control'
            index += 1

    thought = [i for i in selected if regions[i] == 'thought_body']
    tool = [i for i in selected if regions[i] == 'tool_body']
    candidates = []
    if thought:
        candidates.append((thought[(len(thought) - 1) // 2], 'thought_lower_median'))
    if tool:
        candidates.append((tool[(len(tool) - 1) // 2], 'tool_lower_median'))
    if thought:
        candidates.append((thought[0], 'thought_first'))
    candidates.append((selected[-1], 'response_last_selected'))
    candidates.extend((selected[(len(selected) - 1) * numerator // 3], 'selected_quantile_fallback')
                      for numerator in (0, 1, 2, 3))
    chosen = {}
    for position, slot in candidates:
        chosen.setdefault(position, slot)
        if len(chosen) == min(4, len(selected)):
            break
    selected_indices = {position: index for index, position in enumerate(selected)}
    return [dict(completion_position_0=position, selected_index_0=selected_indices[position],
                 token_id=tokens[position], region=regions[position], slot=slot,
                 full_action_selected_tokens=len(selected))
            for position, slot in sorted(chosen.items())]


def numerics_study(args):
    """A bounded real-model experiment with no optimizer, updates or rollouts."""
    from .vfarl_unsloth import UnslothPolicy
    from .train_ddp import trainable_hash
    cfg, rl = read_config(args.config)
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise TrainingError('Preserve the existing numerical study')
    output.mkdir(parents=True, exist_ok=True)
    raw = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line.strip()]
    if not 1 <= len(raw) <= 2:
        raise TrainingError('Short numerical study permits at most one two-action state group')
    started = time.monotonic()
    policy = None
    try:
        policy = UnslothPolicy(args.config, inference_device_count=args.policy_gpus,
            adapter_run=args.adapter_run, training=True)
        from unsloth_zoo.fused_losses.fused_linear import fused_linear_logprobs
        torch, model = policy.torch, policy.model
        before_hash = trainable_hash(model)
        data_path = Path(args.data)
        reference_result = None
        if args.complete_pending_reference:
            from competition.vfarl.l4x4.recovered import complete_reference
            data_path = output / 'focal-reference-completed.jsonl'
            reference_result = complete_reference(policy, args.data, data_path)
        rows, evidence = prepare_rows(data_path, cfg['max_sequence_length'], focal_unit=rl['focal_unit'])
        if len(rows) != len(raw) or evidence['mixed_group_count'] != 1:
            raise TrainingError('Numerical study requires one complete source-bound mixed state group')
        action_denominator = focal_action_denominator(rows, evidence, rl)
        policy.switch_execution_mode('training')
        runtime = policy.info['probability_execution']
        runtime['identity']['implementation_sha256']['fused_linear_logprobs'] = sha256_file(
            Path(inspect.getfile(fused_linear_logprobs)))
        runtime['identity_sha256'] = digest(runtime['identity'])
        policy.identity = identities(model, policy.tokenizer, policy.info, policy.cfg,
                                     profile=policy.probability_profile)
        binding = actor_learner_binding(rows, policy.identity, actor_backend='unsloth',
                                       rollout_correction=rl['rollout_correction'],
                                       learner_lora_scaling=plain_lora_scaling(model),
                                       learner_backend='unsloth')
        model.zero_grad(set_to_none=True)
        for device in policy.cuda_devices:
            torch.cuda.reset_peak_memory_stats(device)
        reports, head_parts, head_rows, anchors = [], [], [], {}
        head_cap = None
        for action_index, row in enumerate(rows):
            probe_rows = head_probe_positions(policy.tokenizer, row)
            probe_indices = [item['selected_index_0'] for item in probe_rows]
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                baseline = focal_logprobs(policy, row, fused_linear_logprobs).detach()
                with frozen_base_reference(model):
                    reference = focal_logprobs(policy, row, fused_linear_logprobs).detach()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                current, hidden, targets, cap = focal_logprobs(
                    policy, row, fused_linear_logprobs, return_predictors=True)
                # Keep only this action's <=4 detached rows from its actual
                # gradient forward, never its complete activation graph.
                head_hidden = hidden[probe_indices].detach().clone()
                head_targets = targets[probe_indices].detach().clone()
                if head_cap is not None and head_cap != float(cap or 0.0):
                    raise TrainingError('Head softcap changed between diagnostic actions')
                head_cap = float(cap or 0.0)
                del hidden, targets
            behavior = current.new_tensor([p for p, keep in
                zip(row['behavior_logprobs'], row['loss_mask']) if keep])
            baseline, reference = baseline.to(current.device), reference.to(current.device)
            log_is = float((baseline.double()-behavior.double()).sum())
            if not math.isfinite(log_is):
                raise TrainingError('Nonfinite importance ratio in the zero-update study')
            importance = math.exp(min(log_is, math.log(rl['rollout_is_max'])))
            importance_tensor = current.new_tensor(importance)
            if not bool(torch.isfinite(importance_tensor)) or not bool(importance_tensor > 0):
                raise TrainingError('Invalid importance weight in the zero-update study')
            ratio = (current-baseline).exp()
            advantage = row['advantage']
            pg = -torch.minimum(ratio*advantage,
                ratio.clamp(1-rl['clip_epsilon'], 1+rl['clip_epsilon'])*advantage)
            gap = reference-current
            kl = torch.expm1(gap)-gap
            objective = reduce_focal_terms(pg+rl['kl_beta']*kl, rl)*importance_tensor
            if not bool(torch.isfinite(objective)):
                raise TrainingError('Nonfinite zero-update objective')
            scaled = objective / action_denominator
            head_coefficients, = torch.autograd.grad(scaled, current, retain_graph=True)
            scaled.backward()
            anchors[row['branch_id']] = dict(old=baseline.cpu().tolist(),
                reference=reference.cpu().tolist(), importance=float(importance_tensor))
            reports.append(dict(branch_id=row['branch_id'], reward=row['reward'],
                selected_tokens=len(behavior), sequence_tokens=len(row['prompt_token_ids'])+len(row['completion_token_ids']),
                log_importance_ratio=log_is, importance_used=float(importance_tensor),
                importance_before_loss_dtype_rounding=importance,
                grad_vs_nograd=probability_comparison(current.detach().cpu().tolist(), baseline.cpu().tolist()),
                learner_vs_behavior=probability_comparison(baseline.cpu().tolist(), behavior.cpu().tolist()),
                objective=float(objective.detach())))
            del current, baseline, reference, behavior, ratio, pg, gap, kl, objective, importance_tensor
            head_parts.append((head_hidden, head_targets,
                               head_coefficients[probe_indices].detach().clone()))
            head_rows.extend(dict(item, action_index_0=action_index, branch_id=row['branch_id'],
                                  reward=row['reward']) for item in probe_rows)
            del head_hidden, head_targets
            del head_coefficients, scaled
        head = model.get_base_model().lm_head
        head_hidden, head_targets, head_coefficients = [torch.cat(
            [part[index].to(head.weight.device) for part in head_parts], dim=0) for index in range(3)]
        head_report = head_numerics(torch, head_hidden, head.weight, head_targets,
            fused_linear_logprobs, bias=getattr(head, 'bias', None), cap=head_cap,
            coefficients=head_coefficients)
        head_report.update(selection_rule='source-token-thought-tool-quantiles-v1',
            selection_index_base=0, selected_source_rows=head_rows,
            actual_learner_projection_chunk_size=rl['logprob_chunk_size'],
            scope='At most four source-structured detached predictors per action, at most two actions/eight rows; actual RL upstream coefficients use identical selected indices. Reward is attribution metadata only; probe shape differs from the complete focal head shape')
        del head_parts, head_hidden, head_targets, head_coefficients
        norm_squared = sum(float(p.grad.detach().double().square().sum())
            for p in model.parameters() if p.requires_grad and p.grad is not None)
        if not math.isfinite(norm_squared):
            raise TrainingError('Zero-update backward produced nonfinite gradients')
        directional = focal_directional_numerics(policy, rows, anchors,
            fused_linear_logprobs, rl, action_denominator) if norm_squared > 0 else None
        if directional is not None:
            directional['objective_grad_anchor'] = sum(item['objective'] for item in reports) / action_denominator
            directional['probe_anchor_minus_backward_anchor'] = (
                directional['objective_at_anchor']-directional['objective_grad_anchor'])
        after_hash = trainable_hash(model)
        if before_hash != after_hash:
            raise TrainingError('Numerical study unexpectedly changed adapter weights')
        result = dict(status='zero-optimizer-numerics-study-completed', optimizer_steps=0,
            new_rollouts=0, adapter_exported=False, source_data_sha256=sha256_file(Path(args.data)),
            scored_data_sha256=sha256_file(data_path), reference_completion=reference_result,
            before_adapter_sha256=before_hash, after_adapter_sha256=after_hash,
            adapter_unchanged=True, gradient_norm=math.sqrt(norm_squared),
            gradient_nonzero=norm_squared > 0,
            objective=focal_objective_description(rl), focal_groups=evidence,
            action_reduction_denominator=action_denominator,
            actor_learner_binding=binding, actions=reports, head=head_report,
            directional_numerics=directional,
            policy=policy.identity, memory_by_device=memory(torch, policy.cuda_devices),
            wall_seconds=time.monotonic()-started,
            limitation='No optimizer was constructed or called. Gradient/head measurements do not establish update direction, 16K backward capacity or independent task gains. Behavior data remains unconsumed by training.')
        atomic_json(output / 'numerics-study.json', result)
        print(json.dumps(dict(phase=result['status'], optimizer_steps=0,
            gradient_norm=result['gradient_norm'], adapter_unchanged=True,
            wall_seconds=result['wall_seconds'])), flush=True)
        model.zero_grad(set_to_none=True)
        return 0
    except Exception:
        (output / 'exception.txt').write_text(traceback.format_exc())
        raise
    finally:
        if policy is not None:
            policy.close()


def memory(torch, devices):
    return [dict(device=str(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device)) for device in devices]


def optimizer_updates(optimizer):
    """Read Adam's existing counter without replacing its moment tensors."""
    return max((int(state['step']) for state in optimizer.state.values()
                if 'step' in state), default=0)


def objective_values(logprobs, learner_old, reference, *, advantage, importance,
                     clip_epsilon, kl_beta, objective_normalization='action_mean',
                     action_token_scale=128.0):
    """Recompute the saved PPO/KL objective; this does not run the model."""
    policy, kl = [], []
    for value, old, ref in zip(logprobs, learner_old, reference):
        ratio = math.exp(value - old)
        clipped = min(max(ratio, 1 - clip_epsilon), 1 + clip_epsilon)
        policy.append(-min(ratio * advantage, clipped * advantage))
        log_ref_ratio = ref - value
        kl.append(math.expm1(log_ref_ratio) - log_ref_ratio)
    policy_mean, kl_mean = sum(policy) / len(policy), sum(kl) / len(kl)
    denominator = focal_reduction_denominator(dict(
        objective_normalization=objective_normalization, action_token_scale=action_token_scale),
        len(policy))
    policy_reduced, kl_reduced = sum(policy) / denominator, sum(kl) / denominator
    return dict(policy_loss=policy_mean, kl=kl_mean,
                token_reduction_denominator=denominator,
                policy_loss_reduced=policy_reduced, kl_reduced=kl_reduced,
                weighted_objective=(policy_reduced + kl_beta * kl_reduced) * importance)


def run(args, policy=None, optimizer=None):
    cfg, rl = read_config(args.config)
    rows, evidence = prepare_rows(args.data, cfg['max_sequence_length'], focal_unit=rl['focal_unit'])
    if not rows:
        raise TrainingError('No verified mixed-outcome sampled focal actions for an update')
    action_denominator = focal_action_denominator(rows, evidence, rl)
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise TrainingError('Refusing to overwrite an existing VFARL update')
    output.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(Path(args.data))
    step_taken = False
    resident_model_reused = policy is not None
    optimizer_object_reused = optimizer is not None
    started = time.monotonic()
    try:
        if policy is None:
            from .vfarl_unsloth import UnslothPolicy
            print(json.dumps(dict(phase='training_model_loading', backend='unsloth',
                policy_gpus=args.policy_gpus, context=cfg['max_sequence_length'],
                adapter_run=str(args.adapter_run) if getattr(args, 'adapter_run', None) else None)), flush=True)
            policy = UnslothPolicy(args.config, inference_device_count=args.policy_gpus,
                                   adapter_run=getattr(args, 'adapter_run', None), training=True)
        else:
            if policy.cfg['max_sequence_length'] != cfg['max_sequence_length']:
                raise TrainingError('Resident policy context differs from the training configuration')
            policy.switch_execution_mode('training')
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
                                       rollout_correction=rl['rollout_correction'],
                                       learner_lora_scaling=plain_lora_scaling(model),
                                       learner_backend='unsloth')
        binding.update(learner_backend='unsloth-model-parallel',
            learner_execution_profile=policy.probability_profile,
            reference_interpretation='Actor old/reference remain immutable; KL uses a separate frozen-base Unsloth learner forward with zero live LoRA scaling.')
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        shapes = {name: list(value.shape) for name, value in model.named_parameters()
                  if value.requires_grad}
        before = shard_snapshot(model)
        if optimizer is None:
            optimizer = torch.optim.AdamW(parameters, lr=cfg['learning_rate'], weight_decay=0.0)
            state_path = getattr(args, 'optimizer_state', None)
            if state_path:
                from .vfarl_optimizer import load_bound_optimizer
                load_bound_optimizer(state_path, policy, optimizer,
                                     learning_rate=cfg['learning_rate'])
        elif {id(value) for group in optimizer.param_groups for value in group['params']} != {id(value) for value in parameters}:
            raise TrainingError('Resident optimizer does not own the active LoRA parameters')
        loaded_learning_rates = [float(group['lr']) for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            # load_state_dict restores the old param_groups too. Preserve Adam
            # moments, but honor this run's explicit learning-rate declaration.
            group['lr'] = cfg['learning_rate']
        optimizer_updates_before = optimizer_updates(optimizer)
        optimizer_state_reused = bool(optimizer.state) and optimizer_updates_before > 0
        optimizer.zero_grad(set_to_none=True)
        for device in policy.cuda_devices:
            torch.cuda.reset_peak_memory_stats(device)
        print(json.dumps(dict(phase='training_model_ready', backend='unsloth',
            model_loading_seconds=0.0 if resident_model_reused else time.monotonic() - started,
            execution_preparation_seconds=time.monotonic() - started,
            resident_model_reused=resident_model_reused,
            optimizer_object_reused=optimizer_object_reused,
            optimizer_state_reused=optimizer_state_reused,
            optimizer_updates_before=optimizer_updates_before,
            loaded_learning_rates=loaded_learning_rates,
            actual_learning_rates=[float(group['lr']) for group in optimizer.param_groups],
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
            # The reference is usually very close to the actor. exp(x)-1
            # loses the small k3 value around zero in FP32; expm1 keeps the
            # same objective without that avoidable cancellation.
            kl = torch.expm1(log_ref_ratio) - log_ref_ratio
            token_denominator = focal_reduction_denominator(rl, old.numel())
            loss = reduce_focal_terms(policy_loss + rl['kl_beta'] * kl, rl) * importance
            if not bool(torch.isfinite(loss)):
                raise TrainingError('Nonfinite Unsloth PPO/KL loss')
            (loss / action_denominator).backward()
            before_logprobs[row['branch_id']] = current.detach().cpu().tolist()
            metrics.append(dict(branch_id=row['branch_id'], state_id=row['state_id'],
                reward=row['reward'], advantage=advantage, loss=float(loss.detach()),
                policy_loss=float(policy_loss.detach().mean()), kl=float(kl.detach().mean()),
                token_reduction_denominator=token_denominator,
                policy_loss_reduced=float(reduce_focal_terms(policy_loss.detach(), rl)),
                kl_reduced=float(reduce_focal_terms(kl.detach(), rl)),
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
        gradient_clipping_scale = min(1.0, cfg['max_grad_norm'] / (float(grad_norm) + 1e-6))
        optimizer.step()
        step_taken = True
        after = shard_snapshot(model)
        squared_delta = sum(float((after[name] - before[name]).double().square().sum())
                            for name in before)
        changed = sum(int(torch.count_nonzero(after[name] - before[name])) for name in before)
        parameter_count = sum(value.numel() for value in before.values())
        parameter_step_rms = math.sqrt(squared_delta / parameter_count)
        # This is the observed clipped loss gradient dotted with the actual
        # parameter step. It adds tensor transfers, never a forward/backward.
        gradient_dot_parameter_delta = sum(float((
            value.grad.detach().float().cpu().double() *
            (after[name] - before[name]).double()).sum())
            for name, value in model.named_parameters()
            if value.requires_grad and value.grad is not None)
        if not math.isfinite(squared_delta) or not squared_delta or not changed:
            raise TrainingError('Optimizer did not produce a finite nonzero LoRA change')
        temporary_adapter = output / '.adapter-writing'
        _write_adapter_state(after, model, temporary_adapter, torch, shapes)
        temporary_adapter.replace(output / 'adapter')
        adapter_files = {name: dict(sha256=sha256_file(output / 'adapter' / name),
            bytes=(output / 'adapter' / name).stat().st_size)
            for name in ('adapter_config.json', 'adapter_model.safetensors')}
        policy.identity = identities(model, policy.tokenizer, policy.info, policy.cfg,
                                     profile=policy.probability_profile)
        committed = dict(status='optimizer-step-and-adapter-committed', optimizer_steps=1,
            learner_backend='unsloth-model-parallel',
            adapter_export_completed=True,
            post_update_diagnostics_completed=False, source_data_sha256=source_sha,
            config=cfg, rl=rl, objective=focal_objective_description(rl),
            action_reduction_denominator=action_denominator,
            evidence=evidence, policy=identity, model=policy.info,
            policy_after_update=policy.identity,
            actor_learner_binding=binding, metrics=metrics,
            resident_model_reused=resident_model_reused,
            optimizer_object_reused=optimizer_object_reused,
            optimizer_state_reused=optimizer_state_reused,
            optimizer_updates_before=optimizer_updates_before,
            optimizer_updates_after=optimizer_updates(optimizer),
            starting_adapter_run=str(args.adapter_run) if not resident_model_reused and getattr(args, 'adapter_run', None) else None,
            optimizer_hyperparameters=[{key: group[key] for key in
                ('lr', 'betas', 'eps', 'weight_decay') if key in group}
                for group in optimizer.param_groups],
            loaded_learning_rates=loaded_learning_rates,
            learning_rate_policy='Current run declaration; Adam moments and step counters preserved',
            importance_weights=importance_summary(metrics),
            parameter_before_sha256=snapshot_hash(before),
            parameter_after_sha256=snapshot_hash(after),
            trainable_adapter_after_sha256=policy.identity['policy_identity']['adapter_sha256'],
            global_parameter_delta_l2=math.sqrt(squared_delta),
            trainable_parameter_count=parameter_count,
            parameter_step_rms=parameter_step_rms,
            gradient_dot_parameter_delta=gradient_dot_parameter_delta,
            gradient_clipping_scale=gradient_clipping_scale,
            unclipped_gradient_dot_parameter_delta_estimate=(
                gradient_dot_parameter_delta / gradient_clipping_scale),
            gradient_step_interpretation='Observed clipped objective gradient dot the actual parameter change; negative predicts first-order descent but is not proof of measured objective improvement.',
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
            optimizer_updates_after=committed['optimizer_updates_after'],
            grad_norm=committed['grad_norm'],
            global_parameter_delta_l2=committed['global_parameter_delta_l2'],
            gradient_dot_parameter_delta=gradient_dot_parameter_delta,
            manifest=str(output / 'optimizer-step.json'))), flush=True)
        comparisons = []
        for row in rows:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                post = focal_logprobs(policy, row, fused_linear_logprobs).detach().cpu().tolist()
            comparison = probability_comparison(post, before_logprobs[row['branch_id']],
                                                clip_epsilon=rl['clip_epsilon'])
            metric = next(item for item in metrics if item['branch_id'] == row['branch_id'])
            matched_comparison = probability_comparison(post, metric['frozen_learner_old_logprobs'],
                                                        clip_epsilon=rl['clip_epsilon'])
            objective_args = dict(advantage=row['advantage'],
                importance=metric['sequence_importance_weight_loss_dtype'],
                clip_epsilon=rl['clip_epsilon'], kl_beta=rl['kl_beta'],
                objective_normalization=rl['objective_normalization'],
                action_token_scale=rl['action_token_scale'])
            objective_before = objective_values(metric['frozen_learner_old_logprobs'],
                metric['frozen_learner_old_logprobs'], metric['actual_reference_logprobs'],
                **objective_args)
            objective_after = objective_values(post, metric['frozen_learner_old_logprobs'],
                metric['actual_reference_logprobs'], **objective_args)
            comparisons.append(dict(branch_id=row['branch_id'], reward=row['reward'],
                advantage=row['advantage'], post_update_logprobs=post,
                before_update_logprobs=before_logprobs[row['branch_id']],
                comparison=comparison, matched_nograd_comparison=matched_comparison,
                optimized_objective_before=objective_before,
                optimized_objective_after=objective_after,
                importance_advantage_weighted_reduced_logprob_change=(
                    objective_args['importance'] * row['advantage'] *
                    sum(value - old for value, old in zip(post, metric['frozen_learner_old_logprobs'])) /
                    metric['token_reduction_denominator']),
                importance_advantage_weighted_logprob_change=
                    objective_args['importance'] * row['advantage'] * matched_comparison['signed_mean']))
        final = dict(committed, status='one-verified-focal-PPO-update-completed',
            post_update_diagnostics_completed=True,
            post_update_comparisons=comparisons,
            mean_advantage_weighted_logprob_change=sum(r['advantage'] *
                r['comparison']['signed_mean'] for r in comparisons) / len(comparisons),
            mean_importance_advantage_weighted_logprob_change=sum(
                item['importance_advantage_weighted_logprob_change'] for item in comparisons) / len(comparisons),
            mean_importance_advantage_weighted_reduced_logprob_change=sum(
                item['importance_advantage_weighted_reduced_logprob_change']
                for item in comparisons) / action_denominator,
            mean_optimized_objective_before=sum(item['optimized_objective_before']['weighted_objective']
                for item in comparisons) / action_denominator,
            mean_optimized_objective_after=sum(item['optimized_objective_after']['weighted_objective']
                for item in comparisons) / action_denominator,
            wall_seconds=time.monotonic() - started)
        final['optimized_objective_change'] = (final['mean_optimized_objective_after']
                                               - final['mean_optimized_objective_before'])
        final['first_order_predicted_objective_change'] = (
            committed['unclipped_gradient_dot_parameter_delta_estimate'])
        final['measured_minus_first_order_change'] = (final['optimized_objective_change']
            - final['first_order_predicted_objective_change'])
        atomic_json(output / 'training_manifest.json', final)
        print(json.dumps(dict(phase='training_update_completed', optimizer_steps=1,
            post_update_diagnostics_completed=True,
            wall_seconds=final['wall_seconds'])), flush=True)
        optimizer.zero_grad(set_to_none=True)
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
    parser.add_argument('--adapter-run', type=Path,
                        help='Continue from an exported adapter without changing the frozen NF4 base')
    parser.add_argument('--optimizer-state', type=Path,
                        help='Optional saved Adam state; resident callers normally keep the same optimizer object')
    parser.add_argument('--numerics-only', action='store_true',
                        help='Short zero-optimizer head/backward study; no adapter export or policy rollout')
    parser.add_argument('--complete-pending-reference', action='store_true',
                        help='Numerical study only: complete a recovered pair reference without changing its behavior')
    parser.add_argument('--mode-study-order', choices=('a-first', 'b-first'),
                        help='Opt-in zero-backward forward-mode observation; requires --numerics-only')
    parser.add_argument('--backward-mode-study', action='store_true',
                        help='Opt-in one-backward mode observation, zero optimizer updates; requires --numerics-only')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Run this Unsloth learner as one process, not torchrun/FSDP')
    if args.complete_pending_reference and not args.numerics_only:
        parser.error('Pending reference completion is restricted to the explicit zero-update study')
    if args.mode_study_order and not args.numerics_only:
        parser.error('Forward mode observation is restricted to the explicit zero-update study')
    if args.backward_mode_study and (not args.numerics_only or args.mode_study_order):
        parser.error('One-backward observation requires --numerics-only and no --mode-study-order')
    if args.numerics_only:
        if args.optimizer_state:
            parser.error('The zero-update study does not load or construct an optimizer')
        if args.backward_mode_study:
            from .vfarl_backward_mode_study import backward_mode_study
            return backward_mode_study(args)
        if args.mode_study_order:
            from .vfarl_mode_study import mode_order_study
            return mode_order_study(args)
        return numerics_study(args)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
