"""Scoped native Gemma4 forwards for exact, differentiable prefix trees.

Only methods and checkpoint settings are temporarily rebound. Projection
modules, LoRA parameters and Accelerate device hooks remain the live learner's.
Keep this context open through backward so checkpoint replay uses the same
attention implementation and tree masks as the original forward.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import importlib
import importlib.metadata
import importlib.util
import sys
from types import MethodType


def _load_source(name, relative_path):
    if name in sys.modules:
        return sys.modules[name]
    path = importlib.metadata.distribution("transformers").locate_file(relative_path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load installed Transformers source: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class _AttributeView:
    """Local namespace override; never monkeypatch the shared torch module."""

    def __init__(self, target, **overrides):
        self._target, self._overrides = target, overrides

    def __getattr__(self, name):
        return self._overrides[name] if name in self._overrides else getattr(self._target, name)


class _DeviceMasks:
    """One tensor per original mask/device, shared by all layers and replays."""

    def __init__(self):
        self.copies = {}
        self.origins = {}

    def place(self, mask, device):
        import torch

        if mask is None:
            return None
        if not isinstance(mask, torch.Tensor) or mask.ndim != 4 or mask.requires_grad:
            raise ValueError("Native tree attention requires a non-differentiable 4D tensor mask")
        alias = self.origins.get(id(mask))
        source = alias[1] if alias is not None and alias[0] is mask else mask
        key = (id(source), torch.device(device))
        if key not in self.copies:
            # Float32 additive masks are supported by native SDPA with fp16,
            # bf16, fp32 and fp64 queries. Do not recast them at every layer.
            dtype = torch.bool if source.dtype == torch.bool else torch.float32
            placed = source.to(device=device, dtype=dtype)
            self.copies[key] = placed
            self.origins[id(source)] = (source, source)
            self.origins[id(placed)] = (placed, source)
        return self.copies[key]

    def clear(self):
        self.copies.clear()
        self.origins.clear()


def _layer_execution_device(layer):
    # This backend keeps the project's existing no-offload placement. A split
    # within the norm/Q-projection boundary needs an explicit execution design;
    # silently guessing the hook's destination can allocate per-layer masks.
    def devices(module):
        return {parameter.device for parameter in module.parameters() if parameter.numel()}

    norm = devices(layer.input_layernorm)
    projection = devices(layer.self_attn.q_proj)
    if (len(norm) != 1 or norm != projection or norm != devices(layer.self_attn)
            or next(iter(norm)).type == "meta"):
        raise ValueError("Tree attention requires the input norm and attention parameters on one materialized execution device")
    device = next(iter(norm))

    def check_hook(hook):
        import torch

        if getattr(hook, "offload", False):
            raise ValueError("Prefix-tree mask reuse does not support parameter-offload hooks")
        execution = getattr(hook, "execution_device", None)
        if execution is not None:
            execution = torch.device("cuda", execution) if type(execution) is int else torch.device(execution)
            if execution.type != device.type or (execution.index is not None and execution.index != device.index):
                raise ValueError("Attention hook execution device disagrees with materialized parameter placement")
        for child in getattr(hook, "hooks", ()):
            check_hook(child)

    for module in (layer, layer.input_layernorm, *layer.self_attn.modules()):
        if hasattr(module, "_hf_hook"):
            check_hook(module._hf_hook)
    return device


@lru_cache(maxsize=1)
def _native_gemma4():
    import torch

    if importlib.metadata.version("transformers") != "5.17.0":
        raise RuntimeError("The native prefix-tree backend targets installed Transformers 5.17.0")
    importlib.import_module("transformers.models.gemma4")
    sdpa = _load_source("transformers.integrations._gemma4_tree_sdpa",
                        "transformers/integrations/sdpa_attention.py")
    # Some training packages also wrap torch.nn.functional SDPA. The official
    # integration keeps its mask/GQA handling, calling PyTorch's native kernel.
    sdpa.torch = _AttributeView(torch, nn=_AttributeView(
        torch.nn, functional=_AttributeView(torch.nn.functional,
            scaled_dot_product_attention=torch._C._nn.scaled_dot_product_attention)))
    masks = _load_source("transformers._gemma4_tree_masking", "transformers/masking_utils.py")
    native = _load_source("transformers.models.gemma4._gemma4_tree_native",
                          "transformers/models/gemma4/modeling_gemma4.py")
    native.create_causal_mask = masks.create_causal_mask
    native.create_sliding_window_causal_mask = masks.create_sliding_window_causal_mask

    def attention(module, query, key, value, attention_mask, **kwargs):
        if attention_mask is not None:
            # Also handles modules with no Accelerate hook, without creating a
            # second dtype/device copy during each checkpoint replay.
            attention_mask = module._gemma4_tree_device_masks.place(attention_mask, query.device)
        return sdpa.sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    class NativeAttentionRegistry:
        @staticmethod
        def get_interface(implementation, default):
            if implementation != "sdpa":
                raise ValueError("Prefix-tree native attention requires SDPA")
            return attention

    # This is the private source module's registry, not the process-wide HF or
    # Unsloth registry. Other forwards and generation keep their implementations.
    native.ALL_ATTENTION_FUNCTIONS = NativeAttentionRegistry()
    return native


def _validate_decoder(decoder):
    import torch

    config = decoder.config
    if (getattr(config, "model_type", None) != "gemma4_text"
            or getattr(config, "num_kv_shared_layers", 0) != 0
            or getattr(config, "enable_moe_block", False)
            or getattr(config, "hidden_size_per_layer_input", 0) != 0
            or getattr(config, "use_bidirectional_attention", None) not in (None, "vision")
            or getattr(config, "is_causal", True) is False):
        raise ValueError("Tree sharing supports causal dense Gemma4 text without cross-layer KV sharing or PLE")
    layers = getattr(decoder, "layers", ())
    if not layers or len(layers) != config.num_hidden_layers:
        raise ValueError("Unexpected Gemma4 decoder layer layout")
    if float(getattr(config, "attention_dropout", 0.0)) != 0.0:
        raise ValueError("Prefix sharing is not equivalent with attention dropout")
    for module in decoder.modules():
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.p != 0.0:
            raise ValueError("Prefix sharing requires zero dropout, including LoRA dropout")
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if (attention is None or getattr(attention, "is_kv_shared_layer", True)
                or getattr(layer, "enable_moe_block", True)
                or getattr(layer, "hidden_size_per_layer_input", None) != 0
                or getattr(attention, "layer_type", None) != config.layer_types[index]
                or config.layer_types[index] not in ("full_attention", "sliding_attention")
                or float(getattr(attention, "attention_dropout", 0.0)) != 0.0):
            raise ValueError("Unsupported Gemma4 attention/layer layout for tree sharing")
        for name in ("q_proj", "k_proj", "o_proj", "q_norm", "k_norm", "v_norm"):
            if not hasattr(attention, name):
                raise ValueError(f"Gemma4 attention is missing {name}")


@contextmanager
def native_tree_forward(decoder, *, gradient_checkpointing):
    """Bind mask-aware native forwards until all associated backward work ends."""
    import torch
    from torch.utils.checkpoint import checkpoint

    _validate_decoder(decoder)
    native = _native_gemma4()
    missing = object()
    saved = []
    device_masks = _DeviceMasks()
    layer_devices = [_layer_execution_device(layer) for layer in decoder.layers]

    def replace(instance, name, value):
        old = instance.__dict__.get(name, missing)
        saved.append((instance, name, old))
        setattr(instance, name, value)

    def bind(instance, function):
        if hasattr(instance, "_hf_hook"):
            if not hasattr(instance, "_old_forward"):
                raise RuntimeError("Accelerate hook has no original forward to rebind")
            replace(instance, "_old_forward", MethodType(function, instance))
        else:
            replace(instance, "forward", MethodType(function, instance))
        if getattr(instance, "_compiled_call_impl", None) is not None:
            replace(instance, "_compiled_call_impl", None)

    def before_device_hook(instance, device):
        if not hasattr(instance, "_hf_hook"):
            return
        wrapped_forward = instance.forward

        def forward(module, *args, **kwargs):
            # Native model/layer calls pass attention_mask by keyword. Place
            # it BEFORE the original Accelerate pre_forward calls .to(device),
            # so every layer captures the same already-placed tensor.
            if kwargs.get("attention_mask") is not None:
                kwargs["attention_mask"] = device_masks.place(kwargs["attention_mask"], device)
            return wrapped_forward(*args, **kwargs)

        replace(instance, "forward", MethodType(forward, instance))

    def checkpointed_layer(instance, *args, **kwargs):
        forward = MethodType(native.Gemma4TextDecoderLayer.forward, instance)
        if gradient_checkpointing and instance.training and torch.is_grad_enabled():
            return checkpoint(forward, *args, use_reentrant=False, **kwargs)
        return forward(*args, **kwargs)

    configs = {id(decoder.config): decoder.config}
    for layer in decoder.layers:
        configs[id(layer.config)] = layer.config
        configs[id(layer.self_attn.config)] = layer.self_attn.config
    attention_settings = [(config, config._attn_implementation) for config in configs.values()]
    try:
        for config, _ in attention_settings:
            config._attn_implementation = "sdpa"
        bind(decoder, native.Gemma4TextModel.forward)
        bind(decoder.rotary_emb, native.Gemma4TextRotaryEmbedding.forward)
        for layer, device in zip(decoder.layers, layer_devices):
            # Bypass inherited Unsloth/reentrant layer checkpoint wrappers;
            # the scoped forward owns one explicit non-reentrant checkpoint.
            replace(layer, "gradient_checkpointing", False)
            bind(layer, checkpointed_layer)
            before_device_hook(layer, device)
            replace(layer.self_attn, "_gemma4_tree_device_masks", device_masks)
            bind(layer.self_attn, native.Gemma4TextAttention.forward)
            before_device_hook(layer.self_attn, device)
        yield
    finally:
        for instance, name, old in reversed(saved):
            if old is missing:
                delattr(instance, name)
            else:
                setattr(instance, name, old)
        for config, implementation in attention_settings:
            config._attn_implementation = implementation
        device_masks.clear()
