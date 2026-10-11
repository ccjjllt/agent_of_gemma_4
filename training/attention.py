"""Opt-in grouped SDPA for Gemma4-31B's 32Q/4KV, head_dim=512 layers.

Nothing is registered or changed on import. The regular trainer does not call
this module. An experiment must call ``enable_grouped_global_sdpa(model,
enabled=True)`` after model/LoRA loading and before DDP wrapping.

This preserves attention over every token. Each of the four KV heads serves
its original eight consecutive query heads, using stride-zero KV views rather
than materializing all 32 KV heads first. CUDA peak-memory/throughput benefit
has NOT been established. Backend-internal copies and backward temporaries can
reduce or eliminate the benefit; no SDPA backend is forced here.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F


_INTERFACE = "p005_grouped_global_sdpa"


def grouped_global_sdpa(query, key, value, *, attention_mask=None,
                        is_causal=False, scaling=1.0, dropout=0.0):
    """Compute four independent head groups; return [B, 32, L, 512].

    Supported: dense Gemma4 global attention, [B,32,L,512] query and
    [B,4,S,512] key/value, identical floating dtype/device, zero dropout.
    Masks are boolean (True means allowed) or same-dtype additive masks with
    shape [L,S] or [B|1,32|1,L|1,S|1]. Full-head masks are sliced with the query
    heads. Non-square causal alignment follows PyTorch's upper-left convention.

    Dropout is rejected: splitting calls changes random-number consumption and
    cannot promise the baseline mask for a given seed. Packed/ragged input,
    multimodal masks, attention scores and custom tensor subclasses are outside
    this small candidate's intended scope. Gemma4 uses scaling=1.0, not 1/sqrt(D).
    """
    if dropout != 0.0:
        raise ValueError("grouped global SDPA requires zero attention dropout")
    if not isinstance(is_causal, bool):
        raise ValueError("is_causal must be an explicit bool")
    if any(not isinstance(x, torch.Tensor) or x.ndim != 4 for x in (query, key, value)):
        raise ValueError("query/key/value must be dense four-dimensional tensors")
    batch, heads, length, dim = query.shape
    if heads != 32 or dim != 512 or batch < 1 or length < 1:
        raise ValueError("expected Gemma4 global query shape [B,32,L,512]")
    if key.shape != value.shape or key.shape[:2] != (batch, 4) or key.shape[-1] != 512 or key.shape[-2] < 1:
        raise ValueError("expected matching key/value shape [B,4,S,512]")
    if not query.is_floating_point() or any(x.dtype != query.dtype or x.device != query.device for x in (key, value)):
        raise ValueError("query/key/value must share floating dtype and device")
    if attention_mask is not None:
        if is_causal:
            raise ValueError("pass the complete mask or is_causal, not both")
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.device != query.device:
            raise ValueError("attention mask must be a tensor on the query device")
        if attention_mask.dtype not in (torch.bool, query.dtype):
            raise ValueError("attention mask must be boolean or match the query dtype")
        source_length = key.shape[-2]
        shape = attention_mask.shape
        if attention_mask.ndim == 2:
            valid = shape[0] in (1, length) and shape[1] in (1, source_length)
        elif attention_mask.ndim == 4:
            valid = (shape[0] in (1, batch) and shape[1] in (1, 32)
                     and shape[2] in (1, length) and shape[3] in (1, source_length))
        else:
            valid = False
        if not valid:
            raise ValueError("unsupported mask layout; expected 2D or broadcastable 4D mask")

    outputs = []
    for group in range(4):
        start, stop = group * 8, (group + 1) * 8
        mask = attention_mask
        if mask is not None and mask.ndim == 4 and mask.shape[1] == 32:
            mask = mask[:, start:stop]
        # The head stride is zero. Do not add contiguous()/repeat_interleave()
        # here: materializing these views would defeat the intended mechanism.
        group_key = key[:, group:group + 1].expand(-1, 8, -1, -1)
        group_value = value[:, group:group + 1].expand(-1, 8, -1, -1)
        outputs.append(F.scaled_dot_product_attention(
            query[:, start:stop], group_key, group_value,
            attn_mask=mask, dropout_p=0.0, is_causal=is_causal,
            scale=scaling, enable_gqa=False))
    # This concatenation temporarily coexists with the four output tensors.
    # Grouping does not remove output storage or guarantee a lower peak.
    return torch.cat(outputs, dim=1)


def gemma4_grouped_global_attention_forward(module, query, key, value,
                                          attention_mask, dropout=0.0,
                                          scaling=None, is_causal=None, **kwargs):
    """HF 5.13.1 attention-interface adapter, global layers only."""
    if getattr(module, "is_sliding", True) or getattr(module, "head_dim", None) != 512:
        raise ValueError("grouped interface is only for Gemma4 global head_dim=512")
    if kwargs.get("output_attentions"):
        raise ValueError("grouped interface returns no attention score matrix")
    if kwargs.get("sliding_window") is not None:
        raise ValueError("global grouped interface does not implement sliding windows")
    for name in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"):
        if kwargs.get(name) is not None:
            raise ValueError("packed attention is unsupported")
    causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
    # Match HF sdpa_attention_forward's mask/causal dispatch exactly.
    causal = bool(query.shape[2] > 1 and attention_mask is None and causal)
    scale = module.scaling if scaling is None else scaling
    output = grouped_global_sdpa(
        query, key, value, attention_mask=attention_mask, is_causal=causal,
        scaling=scale, dropout=dropout)
    return output.transpose(1, 2).contiguous(), None


def enable_grouped_global_sdpa(model, *, enabled=False):
    """Explicit runtime opt-in; returns changed global-layer names.

    Default False is a no-op. Call before DDP, using an SDPA-loaded Gemma4 model.
    Only per-attention config copies change: the language model retains its
    original SDPA mask builder and sliding layers retain ordinary SDPA.
    The runtime setting is not serialized in a PEFT adapter. Reapply explicitly
    when loading an experimental training model; use a fresh model for baseline.
    """
    if enabled is False:
        return []
    if enabled is not True:
        raise ValueError("enabled must be an explicit bool")
    from transformers.modeling_utils import AttentionInterface
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, Gemma4TextAttention) or module.is_sliding:
            continue
        config = module.config
        if (module.head_dim != 512 or config.num_attention_heads != 32
                or config.num_global_key_value_heads != 4 or module.num_key_value_groups != 8
                or module.attention_dropout != 0.0 or not module.is_causal
                or config.num_kv_shared_layers != 0):
            raise ValueError("unsupported Gemma4 global-attention configuration: " + name)
        if config._attn_implementation not in ("sdpa", _INTERFACE):
            raise ValueError("grouped candidate requires the ordinary SDPA mask path: " + name)
        candidates.append((name, module))
    if not candidates:
        raise ValueError("no supported Gemma4 global attention layers found")
    AttentionInterface.register(_INTERFACE, gemma4_grouped_global_attention_forward)
    for _, module in candidates:
        config = copy.copy(module.config)
        config._attn_implementation = _INTERFACE
        module.config = config
    return [name for name, _ in candidates]
