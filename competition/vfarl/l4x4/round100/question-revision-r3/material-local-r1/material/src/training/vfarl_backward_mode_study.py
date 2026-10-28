"""Opt-in one-backward mode observation with no optimizer or parameter step.

Only the original first-action focal scalar is differentiated. Checkpoint
metadata is inspected without unpacking saved tensors or modifying libraries.
Completed real observations survive an externally imposed process deadline.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
import sys
import functools
from pathlib import Path
import time
import traceback

from .common import TrainingError, atomic_json, sha256_file


PRE_BACKWARD_SEQUENCE = ['A:N0', 'A:R0', 'A:G0', 'A:backward']
POST_BACKWARD_SEQUENCE = ['B:N1', 'B:R1', 'B:G1', 'B:G2', 'B:N2',
                          'A:N1', 'A:R1', 'A:G1', 'A:G2', 'A:N2']

GC_SCALARS = ('FIRST_PASS', 'BACKWARD_PASS', 'LAST_GC_INDEX', 'CURRENT_GC_INDEX',
              'CPU_INDEX', 'OFFLOAD_GENERATION', 'FP32_OFFLOAD_EXACT',
              'USE_DOUBLE_BUFFER', 'GC_PREFETCH', 'MINIMUM_SIZE', '_CPU_SLOT_TOP',
              'PINNED_MEMORY_AVAILABLE', 'USE_UNSLOTH_GC')


def checkpoint_graph_metadata(scores, maximum_nodes=100000):
    """Read only graph edges and explicit Python ctx fields, never tensor saves."""
    pending = [scores.grad_fn] if scores.grad_fn is not None else []
    seen, kept_nodes, counts, records = set(), [], {}, []
    while pending and len(seen) < maximum_nodes:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        # Keep wrappers alive until traversal ends, so ids cannot be reused.
        kept_nodes.append(node)
        name = type(node).__name__
        counts[name] = counts.get(name, 0) + 1
        if name == 'UnslothCheckpointFunctionBackward':
            attrs = getattr(node, '__dict__', {})
            metadata = attrs.get('_saved_metadata')
            valid_metadata = isinstance(metadata, tuple) and len(metadata) == 7
            offloaded = valid_metadata and metadata[0] is not None
            item = dict(node_index=len(seen) - 1,
                python_attribute_keys_present=[key for key in (
                    '_requires_gradient', '_saved_dtype', '_store_dtype',
                    '_saved_metadata', '_offload_generation') if key in attrs],
                requires_gradient=attrs.get('_requires_gradient'),
                metadata_available=valid_metadata, offloaded=offloaded,
                saved_dtype=str(attrs['_saved_dtype']) if '_saved_dtype' in attrs else None,
                store_dtype=str(attrs['_store_dtype']) if '_store_dtype' in attrs else None,
                offload_generation=attrs.get('_offload_generation'))
            if valid_metadata:
                item['metadata'] = dict(new_size_bytes=metadata[0],
                    shape=list(metadata[1]) if metadata[1] is not None else None,
                    cpu_slot_index=metadata[2], device_index=metadata[3],
                    main_stream_present=metadata[4] is not None,
                    extra_stream_present=metadata[5] is not None,
                    buffer_slot=metadata[6])
            records.append(item)
        pending.extend(next_node for next_node, _ in node.next_functions if next_node is not None)
    truncated = bool(pending)
    return dict(format='vfarl-passive-checkpoint-ctx-v1',
        traversed_nodes=len(seen), traversal_truncated=truncated, node_type_counts=counts,
        checkpoint_nodes=records, checkpoint_node_count=len(records),
        offloaded_node_count=sum(item['offloaded'] for item in records),
        offloaded_nodes_with_both_dtypes=sum(item['offloaded'] and item['saved_dtype'] is not None
            and item['store_dtype'] is not None for item in records),
        fp32_to_bf16_offloaded_nodes=sum(item['offloaded'] and item['saved_dtype'] == 'torch.float32'
            and item['store_dtype'] == 'torch.bfloat16' for item in records),
        exposure='checkpoint nodes exposed' if records else 'GC nodes not exposed in this graph',
        saved_tensor_access_performed=False,
        boundary='Only next_functions and explicit Python ctx.__dict__ metadata; no saved_tensors or automatic saved attributes. Missing nodes/metadata do not prove absence of compression.')


def backward_mode_study(args):
    from .vfarl_unsloth import UnslothPolicy
    from .train_ddp import trainable_hash
    from .train_vfarl import (prepare_rows, actor_learner_binding, focal_action_denominator,
                              reduce_focal_terms, focal_objective_description)
    from .train_vfarl_unsloth import focal_logprobs, memory
    from .vfarl_policy import frozen_base_reference, read_config, identities, digest, plain_lora_scaling

    cfg, rl = read_config(args.config)
    source = Path(args.data)
    raw = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if len(raw) != 2 or sum(bool(row.get('reference_pending')) for row in raw) != 1:
        raise TrainingError('Backward mode study preserves the original two-action pending-reference warmup')
    if not args.complete_pending_reference or args.optimizer_state:
        raise TrainingError('Backward mode study requires the original reference completion and no optimizer state')
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise TrainingError('Preserve an existing backward-mode-study output directory')
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
            format='vfarl-one-backward-mode-v1',
            pre_backward_sequence=PRE_BACKWARD_SEQUENCE,
            post_backward_sequence=POST_BACKWARD_SEQUENCE,
            source_data_sha256=sha256_file(source), seed=cfg['seed'],
            config_sha256=sha256_file(Path(args.config)), optimizer_steps=0,
            planned_backward_calls=1, new_rollouts=0, parameter_perturbations=0,
            cold_autocast_spans=[['A:N0', 'A:R0'], ['A:G0']],
            post_autocast_spans=[['B:N1', 'B:R1'], ['B:G1'], ['B:G2'], ['B:N2'],
                                 ['A:N1', 'A:R1'], ['A:G1'], ['A:G2'], ['A:N2']],
            warmup='Original complete_reference on original pending data, then original explicit training switch',
            limitation='One backward changes multiple execution states together; passive metadata identifies associations, not a unique causal mechanism. No exact-offload intervention.'))
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
            raise TrainingError('Backward mode study needs the original complete source-bound state group')
        # Exactly the original explicit call after complete_reference's own
        # restoration, including its cache clearing and for_training call.
        with phase('explicit_switch_to_training'):
            policy.switch_execution_mode('training')
        runtime = policy.info['probability_execution']
        runtime['identity']['implementation_sha256']['fused_linear_logprobs'] = sha256_file(
            Path(inspect.getfile(fused_linear_logprobs)))
        runtime['identity_sha256'] = digest(runtime['identity'])
        policy.identity = identities(model, policy.tokenizer, policy.info, policy.cfg,
                                     profile=policy.probability_profile)
        binding = actor_learner_binding(rows, policy.identity, actor_backend='unsloth',
            rollout_correction=rl['rollout_correction'], learner_lora_scaling=plain_lora_scaling(model),
            learner_backend='unsloth')
        action_denominator = focal_action_denominator(rows, evidence, rl)
        model.zero_grad(set_to_none=True)
        backward_calls = 0
        tuner_layers = [(name, module) for name, module in model.named_modules()
                        if isinstance(module, BaseTunerLayer)]
        hidden_history, lp_history = {}, {}
        gc_module = sys.modules.get('unsloth_zoo.gradient_checkpointing')

        def scalar_metadata(value):
            return value if value is None or type(value) in (bool, int, float, str) else {'type': type(value).__name__}

        source_hashes = {}

        def callable_metadata(function):
            result = {'type': type(function).__name__}
            if isinstance(function, functools.partial):
                result['partial_keywords'] = {str(k): scalar_metadata(v)
                                              for k, v in (function.keywords or {}).items()}
                result['partial_positional_argument_count'] = len(function.args)
                function = function.func
            result.update(module=getattr(function, '__module__', None),
                          qualified_name=getattr(function, '__qualname__', None))
            try:
                path = Path(inspect.getfile(function))
                if path.is_file():
                    if str(path) not in source_hashes:
                        source_hashes[str(path)] = sha256_file(path)
                    result.update(source_file=str(path), source_sha256=source_hashes[str(path)])
            except (TypeError, OSError):
                result['source_file_available'] = False
            return result

        def gc_state():
            if gc_module is None:
                return {'available': False}
            attrs = vars(gc_module)
            result = dict(available=True, scalars={key: scalar_metadata(attrs[key])
                for key in GC_SCALARS if key in attrs}, missing=[key for key in GC_SCALARS if key not in attrs])
            slots = attrs.get('NEXT_BUFFER_SLOT')
            result['next_buffer_slots'] = list(slots) if isinstance(slots, (list, tuple)) else scalar_metadata(slots)
            free_slots = attrs.get('_FREE_CPU_SLOTS')
            result['free_cpu_slot_count'] = len(free_slots) if isinstance(free_slots, (list, tuple, set, dict)) else None
            result['gpu_buffer_dtype'] = str(attrs.get('GPU_BUFFER_DTYPE'))
            return result

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
            # Passive snapshots before/after execution. Do not invoke the
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
            checkpointing, compiled_blocks = [], []
            for name, module in model.named_modules():
                if hasattr(module, 'gradient_checkpointing'):
                    value = getattr(module, 'gradient_checkpointing')
                    checkpointing.append(dict(name=name, training=bool(module.training),
                        gradient_checkpointing=value if isinstance(value, (bool, int, str)) else str(value),
                        checkpoint_callable=callable_metadata(getattr(module, '_gradient_checkpointing_func', None))))
                attrs = vars(module)
                if '_unsloth_block_checked' in attrs or '_unsloth_block_mode' in attrs:
                    compiled_blocks.append(dict(name=name, **{key: scalar_metadata(attrs[key]) for key in
                        ('_unsloth_block_checked', '_unsloth_block_mode') if key in attrs}))
            return dict(grad_enabled=torch.is_grad_enabled(),
                inference_mode=torch.is_inference_mode_enabled(),
                autocast_enabled=torch.is_autocast_enabled('cuda'),
                autocast_cache_enabled=torch.is_autocast_cache_enabled(),
                autocast_dtype=str(torch.get_autocast_dtype('cuda')),
                top_training=bool(model.training),
                module_training_counts=dict(training=sum(m.training for m in model.modules()),
                                            evaluation=sum(not m.training for m in model.modules())),
                gradient_checkpointing=checkpointing, lora_layers=layers,
                compiled_blocks=compiled_blocks, gc_globals=gc_state(),
                math=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                    allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                    allow_bf16_reduced_precision_reduction=bool(
                        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)),
                environment={key: os.environ.get(key) for key in
                    ('UNSLOTH_OFFLOAD_FP32', 'UNSLOTH_GC_PREFETCH')})

        def comparison(current, previous):
            delta = current.double() - previous.double()
            base_norm = previous.double().square().sum().sqrt()
            return dict(max_abs=float(delta.abs().max()),
                relative_l2=float(delta.square().sum().sqrt() / base_norm) if base_norm > 0 else None,
                row_max_abs=delta.abs().amax(dim=-1).tolist() if delta.ndim == 2 else None)

        def observe(action_index, label, span, *, retain_scores=False, inspect_checkpoint_graph=False):
            row = rows[action_index]
            key = f'{action_index}-{label}'
            before_state = state()
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
            checkpoint_graph = checkpoint_graph_metadata(scores) if inspect_checkpoint_graph else None
            retained = scores if retain_scores else None
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
                declared_order='A backward then B repeats then A repeats', branch_id=row['branch_id'],
                sequence_tokens=len(row['prompt_token_ids']) + len(row['completion_token_ids']),
                selected_tokens=len(cpu_targets), target_token_ids=cpu_targets,
                selected_logprobs=cpu_scores.tolist(),
                selected_hidden_shape=list(cpu_hidden.shape),
                selected_hidden_dtype=str(cpu_hidden.dtype),
                selected_hidden_sha256=hashlib.sha256(cpu_hidden.view(torch.uint8).numpy().tobytes()).hexdigest(),
                comparisons=comparisons, execution_state_before=before_state,
                execution_state=actual_state, checkpoint_graph=checkpoint_graph,
                graph=grad_fields, forward_seconds=forward_seconds,
                elapsed_seconds=time.monotonic() - started, softcap=float(cap or 0.0),
                parameter_grad_tensors=sum(p.grad is not None for p in model.parameters()),
                optimizer_steps=0, backward_calls=backward_calls)
            atomic_json(output / f'forward-{len(observations):02d}-{key}.json', record)
            observations.append(dict(action_index=action_index, label=label,
                artifact=f'forward-{len(observations):02d}-{key}.json',
                forward_seconds=forward_seconds, elapsed_seconds=record['elapsed_seconds']))
            event('forward_complete', action_index=action_index, label=label,
                  forward_seconds=forward_seconds, selected_tokens=len(cpu_targets))
            return retained

        active_sources = {}
        for name in ('unsloth_zoo.gradient_checkpointing', 'unsloth.models._block_compile'):
            module = sys.modules.get(name)
            path = Path(module.__file__) if module is not None and getattr(module, '__file__', None) else None
            active_sources[name] = dict(loaded=module is not None,
                source_file=str(path) if path else None,
                source_sha256=sha256_file(path) if path and path.is_file() else None)
        checkpoint_module = sys.modules.get('torch.utils.checkpoint')
        atomic_json(output / 'active-checkpoint-sources.json', dict(
            modules=active_sources,
            torch_checkpoint=callable_metadata(getattr(checkpoint_module, 'checkpoint', None)),
            torch_checkpoint_function=callable_metadata(getattr(checkpoint_module, 'CheckpointFunction', None)),
            state_before_first_actor=state(), actor_learner_binding=binding))

        # Exactly one G before the only backward: detached G repetitions would
        # themselves advance the library's FIRST_PASS/LAST_GC_INDEX state.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            baseline = observe(0, 'N0', 'cold-N0-R0', retain_scores=True)
            with frozen_base_reference(model):
                reference = observe(0, 'R0', 'cold-N0-R0', retain_scores=True)
        with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            current = observe(0, 'G0', 'cold-G0', retain_scores=True,
                              inspect_checkpoint_graph=True)
        row = rows[0]
        behavior = current.new_tensor([p for p, keep in zip(row['behavior_logprobs'], row['loss_mask']) if keep])
        baseline, reference = baseline.to(current.device), reference.to(current.device)
        log_is = float((baseline.double() - behavior.double()).sum())
        if not math.isfinite(log_is):
            raise TrainingError('Nonfinite importance ratio in one-backward observation')
        importance = math.exp(min(log_is, math.log(rl['rollout_is_max'])))
        importance_tensor = current.new_tensor(importance)
        if not bool(torch.isfinite(importance_tensor)) or not bool(importance_tensor > 0):
            raise TrainingError('Invalid importance weight in one-backward observation')
        ratio = (current - baseline).exp()
        advantage = row['advantage']
        pg = -torch.minimum(ratio * advantage,
            ratio.clamp(1 - rl['clip_epsilon'], 1 + rl['clip_epsilon']) * advantage)
        gap = reference - current
        kl = torch.expm1(gap) - gap
        objective = reduce_focal_terms(pg + rl['kl_beta'] * kl, rl) * importance_tensor
        if not bool(torch.isfinite(objective)):
            raise TrainingError('Nonfinite original focal scalar before diagnostic backward')
        scaled = objective / action_denominator
        backward_record = dict(action_index=0, branch_id=row['branch_id'],
            log_importance_ratio=log_is, importance_used=float(importance_tensor),
            action_reduction_denominator=action_denominator, advantage=advantage,
            objective=float(objective.detach()), scaled_objective=float(scaled.detach()),
            selected_actor_anchor=baseline.cpu().tolist(), selected_reference=reference.cpu().tolist(),
            before_state=state(), optimizer_steps=0, backward_calls=0,
            scope='Original first-action scalar only; no B backward, optimizer, clipping or parameter step')
        atomic_json(output / 'backward-before.json', backward_record)
        event('single_backward_start', action_index=0, scaled_objective=backward_record['scaled_objective'])
        begin = time.monotonic()
        scaled.backward()
        for device in policy.cuda_devices:
            torch.cuda.synchronize(device)
        backward_seconds = time.monotonic() - begin
        backward_calls = 1
        del current, baseline, reference, behavior, ratio, pg, gap, kl, objective, importance_tensor, scaled
        backward_record.update(after_state=state(), backward_calls=1,
            backward_seconds=backward_seconds,
            gradient_norm_measurement='Deferred until all post forwards; gradients remain intact',
            parameter_grad_tensors=sum(p.grad is not None for p in model.parameters()))
        atomic_json(output / 'backward-after.json', backward_record)
        event('single_backward_complete', backward_calls=1,
              backward_seconds=backward_seconds)

        # B immediately follows A's backward, matching the old study's next
        # action more closely than inserting additional A grad forwards first.
        # Keep the accumulated A gradients throughout all post observations.
        for action_index in (1, 0):
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                observe(action_index, 'N1', 'post-N1-R1')
                with frozen_base_reference(model):
                    observe(action_index, 'R1', 'post-N1-R1')
            with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                observe(action_index, 'G1', 'post-G1', inspect_checkpoint_graph=True)
            with torch.enable_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                observe(action_index, 'G2', 'post-repeat-G2')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                observe(action_index, 'N2', 'post-N2')
        # No norm-reduction CUDA work or extra allocations intervene between
        # the one backward and the immediately following B actor/reference.
        norm_squared = sum(float(p.grad.detach().double().square().sum())
            for p in model.parameters() if p.requires_grad and p.grad is not None)
        if not math.isfinite(norm_squared):
            raise TrainingError('One diagnostic backward produced nonfinite gradients')
        gradient_measurement = dict(gradient_norm=math.sqrt(norm_squared),
            parameter_grad_tensors=sum(p.grad is not None for p in model.parameters()),
            measured_after_all_post_forwards=True, backward_calls=backward_calls)
        atomic_json(output / 'retained-gradient-measurement.json', gradient_measurement)
        with phase('adapter_after_hash'):
            after_hash = trainable_hash(model)
        result = dict(format='vfarl-one-backward-mode-v1',
            status='one-backward-mode-observation-completed',
            actual_action_order=[0, 1, 0], observations=observations,
            source_data_sha256=sha256_file(source), scored_data_sha256=sha256_file(scored),
            reference_completion=reference_result,
            before_adapter_sha256=before_hash, after_adapter_sha256=after_hash,
            adapter_unchanged=before_hash == after_hash,
            optimizer_steps=0, backward_calls=backward_calls, new_rollouts=0, parameter_perturbations=0,
            adapter_exported=False,
            parameter_grad_tensors=sum(p.grad is not None for p in model.parameters()),
            memory_by_device=memory(torch, policy.cuda_devices),
            wall_seconds=time.monotonic() - started,
            backward_observation=backward_record, objective=focal_objective_description(rl),
            gradient_measurement=gradient_measurement,
            actor_learner_binding=binding, final_execution_state=state(),
            limitation='One backward changes several library states together; observed association is not a unique causal attribution. No exact-offload intervention, parameter step, anchor change or learning-rate choice.')
        atomic_json(output / 'backward-mode-study.json', result)
        event('backward_mode_observation_completed', backward_calls=backward_calls,
              observations=len(observations), adapter_unchanged=result['adapter_unchanged'])
        model.zero_grad(set_to_none=True)
        return 0
    except Exception:
        (output / 'exception.txt').write_text(traceback.format_exc())
        raise
    finally:
        if policy is not None:
            policy.close()
