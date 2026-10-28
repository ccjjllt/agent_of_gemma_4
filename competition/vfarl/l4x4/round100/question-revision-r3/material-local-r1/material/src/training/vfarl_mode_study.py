"""Opt-in, zero-backward observation of the real focal forward execution modes.

The cold N0/R0/G0 sequence retains the existing diagnostic's autocast spans.
Every observation is persisted after its real forward; partial work is useful
when the externally imposed process budget expires. No optimizer is constructed
or stepped.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import time
import traceback

from .common import TrainingError, atomic_json, sha256_file


FIRST_ACTION_SEQUENCE = ['N0', 'R0', 'G0', 'G1', 'N1', 'R1', 'N2', 'G2']
SECOND_ACTION_SEQUENCE = ['N0', 'R0', 'G0', 'G1', 'N1']


def mode_order_study(args):
    from .vfarl_unsloth import UnslothPolicy
    from .train_ddp import trainable_hash
    from .train_vfarl import prepare_rows
    from .train_vfarl_unsloth import focal_logprobs, memory
    from .vfarl_policy import frozen_base_reference, read_config

    cfg, rl = read_config(args.config)
    source = Path(args.data)
    raw = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if len(raw) != 2 or sum(bool(row.get('reference_pending')) for row in raw) != 1:
        raise TrainingError('Mode study preserves the original two-action pending-reference warmup')
    if not args.complete_pending_reference or args.optimizer_state:
        raise TrainingError('Mode study requires the original reference completion and no optimizer state')
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise TrainingError('Preserve an existing mode-study output directory')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    policy = None
    observations = []

    def event(phase, **values):
        row = dict(phase=phase, monotonic=time.monotonic(),
                   elapsed_seconds=time.monotonic() - started, **values)
        with (output / 'phases.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
        print(json.dumps(row, ensure_ascii=False), flush=True)

    @contextmanager
    def phase(name):
        event(name + '_start')
        try:
            yield
        finally:
            event(name + '_end')

    try:
        atomic_json(output / 'declared-protocol.json', dict(
            format='vfarl-zero-backward-mode-order-v1', order=args.mode_study_order,
            first_action_sequence=FIRST_ACTION_SEQUENCE,
            second_action_sequence=SECOND_ACTION_SEQUENCE,
            source_data_sha256=sha256_file(source), seed=cfg['seed'],
            config_sha256=sha256_file(Path(args.config)), optimizer_steps=0,
            backward_calls=0, new_rollouts=0, parameter_perturbations=0,
            cold_autocast_spans=[['N0', 'R0'], ['G0'], ['G1']],
            warm_autocast_spans=[['N1', 'R1', 'N2'], ['G2']],
            warmup='Original complete_reference on the original pending input; only missing A reference is scored before training-order swap',
            limitation='A/B-first varies training forward order, not whether each input was ever seen; cold reference insertion and first grad forward remain confounded'))
        with phase('policy_construction'):
            policy = UnslothPolicy(args.config, inference_device_count=args.policy_gpus,
                adapter_run=args.adapter_run, training=True,
                diagnostic_phase=lambda name: event('policy_' + name))
        from unsloth_zoo.fused_losses.fused_linear import fused_linear_logprobs
        from peft.tuners.tuners_utils import BaseTunerLayer
        from unsloth.kernels.utils import get_lora_parameters
        torch, model = policy.torch, policy.model
        with phase('adapter_before_hash'):
            before_hash = trainable_hash(model)
        atomic_json(output / 'loaded-identity.json', dict(
            policy=policy.identity, execution=policy.execution,
            before_adapter_sha256=before_hash,
            fused_source_sha256=sha256_file(Path(inspect.getfile(fused_linear_logprobs))),
            lora_accessor_source_sha256=sha256_file(Path(inspect.getfile(get_lora_parameters)))))
        from competition.vfarl.l4x4.recovered import complete_reference
        scored = output / 'focal-reference-completed.jsonl'
        with phase('original_pending_reference_warmup'):
            reference_result = complete_reference(policy, source, scored)
        rows, evidence = prepare_rows(scored, cfg['max_sequence_length'], focal_unit=rl['focal_unit'])
        if len(rows) != 2 or evidence['mixed_group_count'] != 1:
            raise TrainingError('Mode study needs the original complete source-bound state group')
        # Exactly the original explicit call after complete_reference's own
        # restoration, including its cache clearing and for_training call.
        with phase('explicit_switch_to_training'):
            policy.switch_execution_mode('training')
        ordering = [0, 1] if args.mode_study_order == 'a-first' else [1, 0]
        model.zero_grad(set_to_none=True)
        tuner_layers = [(name, module) for name, module in model.named_modules()
                        if isinstance(module, BaseTunerLayer)]
        hidden_history, lp_history = {}, {}

        def tensor_metadata(value):
            if not isinstance(value, torch.Tensor):
                return {'type': type(value).__name__}
            try:
                version = value._version
            except RuntimeError:
                version = 'unavailable-for-inference-tensor'
            return dict(shape=list(value.shape), dtype=str(value.dtype),
                        device=str(value.device), data_ptr=value.data_ptr(),
                        version=version, requires_grad=bool(value.requires_grad))

        def state():
            # Called only after a completed real forward. Do not invoke the
            # fast accessor merely to inspect it: its generic TP/fake-quant/
            # FP8 branches can mutate state and would contaminate the cold
            # N0/R0/G0 sequence. Its exact source is pinned separately.
            layers = []
            for name, module in tuner_layers:
                base_layer = module._modules.get('base_layer', module)
                layers.append(dict(name=name, training=bool(module.training),
                    scaling={str(k): float(v) for k, v in module.scaling.items()},
                    active_adapters=list(module.active_adapters),
                    lora_A={str(k): tensor_metadata(v.weight) for k, v in module.lora_A.items()},
                    lora_B={str(k): tensor_metadata(v.weight) for k, v in module.lora_B.items()},
                    disable_adapters=bool(getattr(module, 'disable_adapters', False)),
                    merged=bool(getattr(module, 'merged', False)),
                    base_type=type(base_layer).__module__ + '.' + type(base_layer).__qualname__,
                    base_weight=tensor_metadata(getattr(base_layer, 'weight', None)),
                    accessor_branch_attributes=dict(
                        tp_local_present=module.__dict__.get('_unsloth_tp_local') is not None,
                        base_fake_quantizer_present=getattr(base_layer, 'weight_fake_quantizer', None) is not None,
                        lora_fake_quantizer_present=any(getattr(v, 'weight_fake_quantizer', None) is not None
                            for table in (module.lora_A, module.lora_B) for v in table.values()),
                        quant_method=str(getattr(base_layer, 'quant_method', None)),
                        packed_mxfp4=bool(getattr(type(base_layer), '_unsloth_mxfp4_packed_linear', False))),
                    cache_attributes={k: tensor_metadata(v) for k, v in module.__dict__.items()
                                      if 'cache' in k.lower() or 'fast_lora' in k.lower()}))
            checkpointing = []
            for name, module in model.named_modules():
                if hasattr(module, 'gradient_checkpointing'):
                    value = getattr(module, 'gradient_checkpointing')
                    checkpointing.append(dict(name=name, training=bool(module.training),
                        gradient_checkpointing=value if isinstance(value, (bool, int, str)) else str(value)))
            return dict(grad_enabled=torch.is_grad_enabled(),
                inference_mode=torch.is_inference_mode_enabled(),
                autocast_enabled=torch.is_autocast_enabled('cuda'),
                autocast_cache_enabled=torch.is_autocast_cache_enabled(),
                autocast_dtype=str(torch.get_autocast_dtype('cuda')),
                top_training=bool(model.training),
                module_training_counts=dict(training=sum(m.training for m in model.modules()),
                                            evaluation=sum(not m.training for m in model.modules())),
                gradient_checkpointing=checkpointing, lora_layers=layers)

        def comparison(current, previous):
            delta = current.double() - previous.double()
            base_norm = previous.double().square().sum().sqrt()
            return dict(max_abs=float(delta.abs().max()),
                relative_l2=float(delta.square().sum().sqrt() / base_norm) if base_norm > 0 else None,
                row_max_abs=delta.abs().amax(dim=-1).tolist() if delta.ndim == 2 else None)

        def observe(action_index, label, span):
            row = rows[action_index]
            key = f'{action_index}-{label}'
            event('forward_start', action_index=action_index, label=label, autocast_span=span,
                  grad_enabled=torch.is_grad_enabled(),
                  sequence_tokens=len(row['prompt_token_ids']) + len(row['completion_token_ids']))
            for device in policy.cuda_devices:
                torch.cuda.synchronize(device)
            begin = time.monotonic()
            scores, hidden, targets, cap = focal_logprobs(
                policy, row, fused_linear_logprobs, return_predictors=True)
            grad_fields = dict(scores_requires_grad=bool(scores.requires_grad),
                               hidden_requires_grad=bool(hidden.requires_grad))
            detached_scores, detached_hidden = scores.detach(), hidden.detach()
            del scores, hidden
            for device in policy.cuda_devices:
                torch.cuda.synchronize(device)
            forward_seconds = time.monotonic() - begin
            actual_state = state()
            cpu_scores = detached_scores.float().cpu().contiguous()
            cpu_hidden = detached_hidden.cpu().contiguous()
            cpu_targets = targets.detach().cpu().tolist()
            del detached_scores, detached_hidden, targets
            comparisons = {}
            for previous_key, previous_hidden in hidden_history.items():
                if previous_key.startswith(str(action_index) + '-'):
                    comparisons[previous_key] = dict(
                        selected_hidden=comparison(cpu_hidden, previous_hidden),
                        selected_logprobs=comparison(cpu_scores, lp_history[previous_key]))
            hidden_history[key], lp_history[key] = cpu_hidden, cpu_scores
            record = dict(format='vfarl-real-mode-forward-observation-v1',
                action_index=action_index, label=label, autocast_span=span,
                declared_order=args.mode_study_order, branch_id=row['branch_id'],
                sequence_tokens=len(row['prompt_token_ids']) + len(row['completion_token_ids']),
                selected_tokens=len(cpu_targets), target_token_ids=cpu_targets,
                selected_logprobs=cpu_scores.tolist(),
                selected_hidden_shape=list(cpu_hidden.shape),
                selected_hidden_dtype=str(cpu_hidden.dtype),
                selected_hidden_sha256=hashlib.sha256(cpu_hidden.view(torch.uint8).numpy().tobytes()).hexdigest(),
                comparisons=comparisons, execution_state=actual_state,
                graph=grad_fields, forward_seconds=forward_seconds,
                elapsed_seconds=time.monotonic() - started, softcap=float(cap or 0.0),
                parameter_grad_tensors=sum(p.grad is not None for p in model.parameters()),
                optimizer_steps=0, backward_calls=0)
            atomic_json(output / f'forward-{len(observations):02d}-{key}.json', record)
            observations.append(dict(action_index=action_index, label=label,
                artifact=f'forward-{len(observations):02d}-{key}.json',
                forward_seconds=forward_seconds, elapsed_seconds=record['elapsed_seconds']))
            event('forward_complete', action_index=action_index, label=label,
                  forward_seconds=forward_seconds, selected_tokens=len(cpu_targets))

        for ordinal, action_index in enumerate(ordering):
            # Original first actor/reference pair shares one no-grad autocast
            # context; no additional actor computation occurs before G0.
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                observe(action_index, 'N0', 'cold-N0-R0')
                with frozen_base_reference(model):
                    observe(action_index, 'R0', 'cold-N0-R0')
            with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                observe(action_index, 'G0', 'cold-G0')
            with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                observe(action_index, 'G1', 'repeat-G1')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                observe(action_index, 'N1', 'warm-N1-R1-N2' if ordinal == 0 else 'post-N1')
                if ordinal == 0:
                    with frozen_base_reference(model):
                        observe(action_index, 'R1', 'warm-N1-R1-N2')
                    observe(action_index, 'N2', 'warm-N1-R1-N2')
            if ordinal == 0:
                with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    observe(action_index, 'G2', 'warm-G2')
        with phase('adapter_after_hash'):
            after_hash = trainable_hash(model)
        result = dict(format='vfarl-zero-backward-mode-order-v1',
            status='mode-order-observation-completed', order=args.mode_study_order,
            actual_action_order=ordering, observations=observations,
            source_data_sha256=sha256_file(source), scored_data_sha256=sha256_file(scored),
            reference_completion=reference_result,
            before_adapter_sha256=before_hash, after_adapter_sha256=after_hash,
            adapter_unchanged=before_hash == after_hash,
            optimizer_steps=0, backward_calls=0, new_rollouts=0, parameter_perturbations=0,
            adapter_exported=False,
            parameter_grad_tensors=sum(p.grad is not None for p in model.parameters()),
            memory_by_device=memory(torch, policy.cuda_devices),
            wall_seconds=time.monotonic() - started,
            limitation='Forward observation only; cold reference insertion and first grad invocation remain confounded. No loss, anchor, learning-rate, optimizer or sampling change is inferred.')
        atomic_json(output / 'mode-order-study.json', result)
        event('mode_order_observation_completed', order=args.mode_study_order,
              observations=len(observations), adapter_unchanged=result['adapter_unchanged'])
        return 0
    except Exception:
        (output / 'exception.txt').write_text(traceback.format_exc())
        raise
    finally:
        if policy is not None:
            policy.close()
