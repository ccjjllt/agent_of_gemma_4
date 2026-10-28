"""FSDP1 helpers for the existing immutable QAT-derived NF4 cache.

Storage conversion is a byte reinterpretation, never a numerical dtype cast.
No model/GPU work happens on import. All ranks must join adapter export.
"""
from functools import partial
import copy
from pathlib import Path

from .common import TrainingError


def pack_floating_nf4_storage(model, torch, bnb, dtype):
    if dtype != torch.bfloat16:
        raise TrainingError('The P015 FSDP recipe requires BF16 packed storage')
    report = []
    seen = set()
    for name, module in model.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        weight = module.weight
        if not isinstance(weight, bnb.nn.Params4bit) or weight.requires_grad:
            raise TrainingError('FSDP expects frozen Params4bit: ' + name)
        state = weight.quant_state
        if state is None or not weight.bnb_quantized or state.quant_type != 'nf4':
            raise TrainingError('Uninitialized/non-NF4 packed parameter: ' + name)
        if getattr(state, 'packing_format_for_cpu', False):
            raise TrainingError('CPU-specific repacking is unsupported: ' + name)
        if id(weight) not in seen:
            seen.add(id(weight))
            old_dtype = weight.dtype
            if old_dtype != torch.uint8:
                raise TrainingError('Expected original uint8 NF4 cache, got ' + str(old_dtype))
            if not weight.is_contiguous() or weight.numel() % torch.empty((), dtype=dtype).element_size():
                raise TrainingError('NF4 bytes cannot be reinterpreted exactly: ' + name)
            raw = weight.data.reshape(-1)
            pointer, nbytes = raw.data_ptr(), raw.numel()
            # Flatten before dtype-view: the old [packed_bytes,1] last dimension
            # is too short for view(BF16). Restore [packed_elements,1] for bnb's
            # FSDP quant-state reconstruction (its shape[1] == 1 contract).
            packed = raw.view(dtype).reshape(-1, 1)
            if packed.data_ptr() != pointer or packed.numel() * packed.element_size() != nbytes:
                raise TrainingError('NF4 storage view changed byte identity: ' + name)
            weight.data = packed
            weight.quant_storage = dtype
            report.append({'name': name, 'bytes': nbytes, 'from': str(old_dtype),
                           'to': str(dtype), 'shape': list(packed.shape),
                           'same_storage_pointer': True})
        module.quant_storage = dtype
        module.quant_state = state
        weight.module = module
    if not report:
        raise TrainingError('No packed NF4 linear weights found')
    return {'operation': 'uint8 bytes viewed as BF16; no numerical cast or requantization',
            'quantization_metadata': 'original QuantState objects retained',
            'tensors': report, 'packed_bytes': sum(x['bytes'] for x in report)}


def wrap_fsdp(model, torch, device):
    """PEFT trainable-leaf policy plus separate FP32 frozen norm units.

    BF16 packed bytes must never be cast by FSDP mixed precision. LoRA stays
    FP32; autocast handles its arithmetic. Frozen FP32 norms are isolated from
    each decoder's BF16 flat parameter. Tied embedding/head remain at the root.
    """
    from peft.utils.other import fsdp_auto_wrap_policy
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy

    peft_policy = fsdp_auto_wrap_policy(model)
    def fp32_leaf(module):
        parameters = list(module.parameters(recurse=False))
        return (not list(module.children()) and bool(parameters)
                and all(not p.requires_grad and p.dtype == torch.float32 for p in parameters))
    policy = partial(_or_policy, policies=[peft_policy,
        partial(lambda_auto_wrap_policy, lambda_fn=fp32_leaf)])
    # sync_module_states performs tensor broadcasts. BF16 packed containers may
    # encode NaNs; broadcast copies bytes, while arithmetic dtype conversion or
    # averaging could corrupt them. Every rank loaded the same immutable cache.
    wrapped = FSDP(model, auto_wrap_policy=policy, use_orig_params=False,
        sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=device,
        sync_module_states=False, limit_all_gathers=True,
        forward_prefetch=False, backward_prefetch=None,
        mixed_precision=MixedPrecision(param_dtype=None, reduce_dtype=torch.float32,
                                       buffer_dtype=None))
    units = []
    for name, module in wrapped.named_modules():
        if not isinstance(module, FSDP):
            continue
        parameter = module._flat_param
        if parameter is not None:
            units.append({'name': name, 'dtype': str(parameter.dtype),
                          'requires_grad': parameter.requires_grad,
                          'local_numel': parameter.numel()})
    return wrapped, units


def clean_fsdp_name(name):
    return '.'.join(part for part in name.split('.') if part != '_fsdp_wrapped_module')


def gather_adapter_state(wrapped, torch, expected_shapes):
    """Gather each trainable leaf in identical order on every rank.

    This deliberately avoids full-model state_dict: the frozen NF4 base is
    never assembled, and bnb quantization metadata is not treated as a model
    parameter. Only rank zero retains CPU copies. Shapes/keys must match the
    original unwrapped adapter exactly before PEFT serialization.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    if wrapped._is_root is not True:
        raise TrainingError('Gather adapters after the root FSDP model has completed a forward')
    state = {}
    rank = torch.distributed.get_rank()
    for prefix, unit in wrapped.named_modules():
        if not isinstance(unit, FSDP) or unit._flat_param is None or not unit._flat_param.requires_grad:
            continue
        with FSDP.summon_full_params(unit, recurse=False, writeback=False,
                                    rank0_only=False, offload_to_cpu=False):
            for local_name, parameter in unit.module.named_parameters():
                if not parameter.requires_grad:
                    continue
                name = clean_fsdp_name(prefix + '.' + local_name)
                if name not in expected_shapes or tuple(parameter.shape) != tuple(expected_shapes[name]):
                    raise TrainingError('Unexpected gathered adapter parameter: ' + name)
                if name in state:
                    raise TrainingError('Duplicate gathered adapter parameter: ' + name)
                # All ranks record keys, but only rank0 allocates CPU tensors.
                state[name] = parameter.detach().cpu().clone() if rank == 0 else None
    if set(state) != set(expected_shapes):
        raise TrainingError('Incomplete adapter gather; missing: ' + str(sorted(set(expected_shapes)-set(state))))
    return state if rank == 0 else {}


def save_adapter(wrapped, peft_model, destination, torch, expected_shapes):
    state = gather_adapter_state(wrapped, torch, expected_shapes)
    error = None
    if torch.distributed.get_rank() == 0:
        try:
            _write_adapter_state(state, peft_model, destination, torch, expected_shapes)
        except Exception as exc:
            error = '%s: %s' % (type(exc).__name__, exc)
    errors = [error]
    torch.distributed.broadcast_object_list(errors, src=0)
    if errors[0]:
        raise TrainingError('FSDP adapter export failed: ' + errors[0])


def _write_adapter_state(state, peft_model, destination, torch, expected_shapes):
    # PEFT 0.21.2 structurally filters against named_modules(), but strips
    # only a leading FSDP wrapper. Nested decoder wrappers then silently
    # produce an EMPTY adapter even when supplied a complete clean state.
    # This recipe only supports plain default LoRA, so serialize its normal
    # PEFT key format explicitly and save the independent PEFT config.
    from safetensors.torch import save_file
    configs = peft_model.peft_config
    if set(configs) != {'default'}:
        raise TrainingError('FSDP export supports one default LoRA adapter')
    config = copy.deepcopy(configs['default'])
    kind = getattr(config.peft_type, 'value', config.peft_type)
    if (kind != 'LORA' or config.bias != 'none' or config.use_dora
            or config.modules_to_save or getattr(config, 'lora_bias', False)):
        raise TrainingError('FSDP export requires plain bias-free LoRA')
    adapter = {}
    for name, tensor in state.items():
        if not any(name.endswith(suffix) for suffix in
                   ('.lora_A.default.weight', '.lora_B.default.weight')):
            raise TrainingError('Unexpected LoRA export key: ' + name)
        if not bool(torch.isfinite(tensor).all()):
            raise TrainingError('Non-finite LoRA export tensor: ' + name)
        key = name.replace('.default.weight', '.weight')
        adapter[key] = tensor.contiguous()
    if not adapter or len(adapter) != len(expected_shapes):
        raise TrainingError('FSDP refuses empty/incomplete PEFT export')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    save_file(adapter, str(destination/'adapter_model.safetensors'), metadata={'format': 'pt'})
    config.inference_mode = True
    config.save_pretrained(destination)
