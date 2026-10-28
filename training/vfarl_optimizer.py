"""Bind resumable Adam moments to actual VFARL tensors and optimizer slots.

PyTorch optimizer state uses positional parameter IDs, not model parameter names.
The binding prevents a same-shaped, differently ordered adapter from silently
receiving those moments. Inference/training execution profiles are deliberately
excluded: switching execution mode does not change the saved weight identity.
"""
import hashlib
import json
import math
from pathlib import Path

from .common import TrainingError
from .train_ddp import trainable_hash


FORMAT = 'vfarl-bound-adam-v1'
CORE_FIELDS = ('nf4_cache_manifest_sha256', 'tokenizer_sha256', 'template_sha256',
               'tool_schema_sha256', 'reference')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def optimizer_binding(policy, optimizer):
    named = {id(parameter): (name, parameter)
             for name, parameter in policy.model.named_parameters()}
    wanted = {identifier for identifier, (_, parameter) in named.items()
              if parameter.requires_grad}
    groups, owned = [], []
    for group in optimizer.param_groups:
        records = []
        for parameter in group['params']:
            item = named.get(id(parameter))
            if item is None or not parameter.requires_grad:
                raise TrainingError('VFARL optimizer owns an unknown or frozen parameter')
            name, _ = item
            records.append(dict(name=name, shape=list(parameter.shape), dtype=str(parameter.dtype)))
            owned.append(id(parameter))
        groups.append(records)
    if not wanted or set(owned) != wanted or len(set(owned)) != len(owned):
        raise TrainingError('VFARL optimizer must own every trainable adapter parameter exactly once')
    reference = policy.identity.get('reference_identity', {})
    core = {key: reference.get(key) for key in CORE_FIELDS}
    if any(not isinstance(value, str) or not value for value in core.values()):
        raise TrainingError('Cannot bind Adam state without the frozen base/tokenizer/tool identity')
    # Use actual live scaling, rather than a backend mode label or an exported
    # inference_mode flag. Plain LoRA scaling must also survive mode changes.
    scaling = []
    for name, module in policy.model.named_modules():
        values = getattr(module, 'scaling', None)
        if isinstance(values, dict) and values:
            record = {key: float(value) for key, value in values.items()}
            if any(not math.isfinite(value) for value in record.values()):
                raise TrainingError('Cannot bind Adam state to nonfinite LoRA scaling')
            scaling.append(dict(name=name, scaling=record))
    return dict(optimizer_class=type(optimizer).__module__+'.'+type(optimizer).__qualname__,
        ordered_parameter_groups=groups, adapter_sha256=trainable_hash(policy.model),
        frozen_core=core, lora_scaling=scaling)


def save_bound_optimizer(path, policy, optimizer):
    path = Path(path)
    binding = optimizer_binding(policy, optimizer)
    payload = dict(format=FORMAT, binding=binding, binding_sha256=_digest(binding),
                   optimizer_state=optimizer.state_dict())
    temporary = path.with_name(path.name+'.part')
    policy.torch.save(payload, temporary)
    temporary.replace(path)
    return dict(format=FORMAT, binding_sha256=payload['binding_sha256'],
                adapter_sha256=binding['adapter_sha256'])


def load_bound_optimizer(path, policy, optimizer, *, learning_rate):
    if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise TrainingError('Resuming Adam requires a positive declared learning rate')
    payload = policy.torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != FORMAT:
        raise TrainingError('Unbound legacy optimizer state cannot be resumed safely; '
                            'keep the adapter and explicitly start a fresh Adam optimizer')
    binding = payload.get('binding')
    if not isinstance(binding, dict) or payload.get('binding_sha256') != _digest(binding):
        raise TrainingError('VFARL optimizer binding is missing or changed')
    current = optimizer_binding(policy, optimizer)
    if binding != current:
        differences = [key for key in current if binding.get(key) != current[key]]
        raise TrainingError('VFARL optimizer does not match the actual adapter/slots: '+', '.join(differences))
    state = payload.get('optimizer_state')
    if not isinstance(state, dict) or not isinstance(state.get('param_groups'), list):
        raise TrainingError('Bound VFARL optimizer has no valid parameter groups')
    if (len(state['param_groups']) != len(optimizer.param_groups)
            or any(len(saved.get('params', [])) != len(actual['params'])
                   for saved, actual in zip(state['param_groups'], optimizer.param_groups))):
        raise TrainingError('Bound VFARL optimizer slot counts differ')
    loaded_rates = [float(group['lr']) for group in state['param_groups']]
    optimizer.load_state_dict(state)
    for group in optimizer.param_groups:
        group['lr'] = float(learning_rate)
    return dict(format=FORMAT, binding_sha256=payload['binding_sha256'],
        adapter_sha256=binding['adapter_sha256'], loaded_learning_rates=loaded_rates,
        actual_learning_rates=[float(group['lr']) for group in optimizer.param_groups],
        learning_rate_policy='Declared current learning rate; bound Adam moments and steps retained')
