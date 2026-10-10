"""Memory-bounded, text-only Gemma4 LoRA training helpers.

Implementation checked against Transformers 5.13.1 ``modeling_gemma4.py`` /
``loss_utils.py`` and PEFT 0.21.2 ``prepare_model_for_kbit_training``. No model
weights are downloaded here. Import this module only in the real ML path.

The preparation helper must run BEFORE injecting LoRA. It deliberately differs
from PEFT's blanket FP32 cast: frozen embedding/head and other large matrices
retain their loaded dtype. Gemma4RMSNorm already computes its norm in FP32 and
returns the input dtype, so keeping only its small scale in FP32 is sufficient
to preserve that implementation's normalization arithmetic.

Install the loss helper AFTER PEFT wrapping, BEFORE DDP wrapping. The normal
outer PEFT/DDP forward is still called. Labelled text forwards return loss and
``logits=None``; unlabelled inference is delegated to the original forward.
Full backbone context is retained. Only the vocabulary projection / CE is
restricted to supervised next-token positions, with checkpointed chunks so
autograd does not retain every chunk's vocabulary-sized intermediates.
"""
from __future__ import annotations

import functools
import inspect
import types

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4CausalLMOutputWithPast,
    Gemma4ForConditionalGeneration,
    Gemma4RMSNorm,
)


def _is_4bit_parameter(parameter):
    # Do not import bitsandbytes just to identify its parameter subclass; this
    # also preserves subclasses of Params4bit, not merely the exact class.
    return any(cls.__name__ == "Params4bit" for cls in type(parameter).__mro__)


def _gemma4_base(model):
    if isinstance(model, Gemma4ForConditionalGeneration):
        return model
    configs = getattr(model, "peft_config", None)
    if not configs or not callable(getattr(model, "get_base_model", None)):
        raise TypeError("Only Gemma4ForConditionalGeneration with optional plain PEFT LoRA is supported")
    for config in configs.values():
        kind = getattr(config.peft_type, "value", config.peft_type)
        if kind != "LORA" or config.is_prompt_learning or getattr(config, "alora_invocation_tokens", None):
            raise TypeError("Selected-token loss supports plain LoRA, not prompt learning or aLoRA")
    base = model.get_base_model()
    if not isinstance(base, Gemma4ForConditionalGeneration):
        raise TypeError("Selected-token loss requires a Gemma4ForConditionalGeneration base")
    return base


def prepare_model_for_memory_efficient_kbit_training(model, *, use_gradient_checkpointing=True):
    """Freeze the base, cast small norms only, and enable non-reentrant GC.

    Returns the same model. ``model._p005_memory_prepare_report`` records the
    exact changed parameter names and dtypes for run metadata. Params4bit data
    is never reassigned. This does not make compressed W4A16 weights trainable;
    it prepares an already loaded BF16 / bitsandbytes NF4 model for LoRA.
    """
    if getattr(model, "peft_config", None) or any("lora_" in n for n, _ in model.named_parameters()):
        raise ValueError("Prepare the frozen base before LoRA injection; refusing to freeze existing adapters")
    _gemma4_base(model)
    changed = []
    quantized = 0
    parameters = list(model.named_parameters())
    for _, parameter in parameters:
        parameter.requires_grad_(False)
        quantized += int(_is_4bit_parameter(parameter))
    seen = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, (Gemma4RMSNorm, torch.nn.LayerNorm)):
            continue
        for local_name, parameter in module.named_parameters(recurse=False):
            if id(parameter) in seen or _is_4bit_parameter(parameter):
                continue
            seen.add(id(parameter))
            # Never turn an unexpectedly large/matrix-valued "norm" into a
            # hidden allocation. The actual Gemma4 norms have vector scales.
            if parameter.ndim > 1 or parameter.numel() > 65536:
                raise ValueError(f"Unexpected large norm parameter: {module_name}.{local_name}")
            if parameter.dtype in (torch.float16, torch.bfloat16):
                previous = str(parameter.dtype)
                parameter.data = parameter.data.to(torch.float32)
                changed.append({"name": f"{module_name}.{local_name}", "numel": parameter.numel(),
                                "from_dtype": previous, "to_dtype": str(parameter.dtype)})
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    if use_gradient_checkpointing:
        # Non-reentrant checkpointing supports frozen embedding outputs and
        # nested LoRA gradients without the input-requires-grad memory hack.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    elif model.is_gradient_checkpointing:
        model.gradient_checkpointing_disable()
    model._p005_memory_prepare_report = {
        "policy": "norms_fp32", "frozen_parameter_tensors": len(parameters),
        "unchanged_params4bit_tensors": quantized, "norm_casts": changed,
        "gradient_checkpointing": bool(use_gradient_checkpointing),
        "use_reentrant": False, "use_cache": False,
    }
    return model


def _text_inputs(base, values):
    """Fail closed on unsupported labelled calls, before running the backbone."""
    allowed = {"input_ids", "labels", "attention_mask", "position_ids", "num_items_in_batch",
               "output_attentions", "output_hidden_states", "return_dict", "use_cache"}
    for key, value in values.items():
        if key == "logits_to_keep" and isinstance(value, int) and value == 0:
            continue
        if key not in allowed and value is not None:
            raise ValueError(f"Selected-token text loss does not support {key}")
    if values.get("use_cache"):
        raise ValueError("Selected-token training requires use_cache=False")
    ids, labels = values.get("input_ids"), values.get("labels")
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2:
        raise ValueError("Selected-token loss requires 2D text input_ids, not inputs_embeds")
    if not isinstance(labels, torch.Tensor) or labels.shape != ids.shape or labels.dtype != torch.long:
        raise ValueError("labels must be int64 and have the same [batch, sequence] shape as input_ids")
    if ids.shape[1] < 2:
        raise ValueError("At least two context tokens are required for next-token loss")
    mask = values.get("attention_mask")
    if mask is not None and (not isinstance(mask, torch.Tensor) or mask.shape != ids.shape):
        raise ValueError("Only a 2D text attention_mask is supported")
    token_fields = ("image_token_id", "video_token_id", "audio_token_id", "boi_token_id",
                    "eoi_token_id", "boa_token_id", "eoa_token_index")
    mm_ids = [getattr(base.config, name, None) for name in token_fields]
    mm_ids = [value for value in mm_ids if isinstance(value, int) and value >= 0]
    if mm_ids and torch.isin(ids, torch.tensor(mm_ids, device=ids.device, dtype=ids.dtype)).any().item():
        raise ValueError("Multimodal token IDs are unsupported by selected-token text loss")
    selected = labels[:, 1:] != -100
    if not selected.any().item():
        raise ValueError("No supervised next-token labels")
    targets = labels[:, 1:][selected]
    vocab_size = base.config.get_text_config().vocab_size
    if ((targets < 0) | (targets >= vocab_size)).any().item():
        raise ValueError("Supervised label lies outside the Gemma4 vocabulary")
    return selected


def selected_token_forward(base, *, chunk_size=32, **values):
    """Gemma4 loss-only forward with HF's shift, softcap and mean CE semantics.

    ``num_items_in_batch`` has the same optional sum/divisor semantics as HF's
    ForCausalLMLoss. It is the caller's responsibility to apply any additional
    distributed normalization. Every chunk is recomputed during backward,
    including the LM head and CE; full-vocabulary logits never leave this API.
    Prefer ``install_selected_token_loss`` to keep PEFT/DDP forward hooks.
    """
    if not isinstance(base, Gemma4ForConditionalGeneration):
        raise TypeError("selected_token_forward requires the unwrapped Gemma4 base")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    selected = _text_inputs(base, values)
    labels = values["labels"]
    batch_indices, prediction_indices = selected.nonzero(as_tuple=True)
    targets = labels[:, 1:][selected]
    outputs = base.model(
        input_ids=values["input_ids"], attention_mask=values.get("attention_mask"),
        position_ids=values.get("position_ids"), use_cache=False, return_dict=True,
        output_attentions=values.get("output_attentions"),
        output_hidden_states=values.get("output_hidden_states"),
    )
    hidden = outputs.last_hidden_state
    batch_indices = batch_indices.to(hidden.device)
    prediction_indices = prediction_indices.to(hidden.device)
    cap = base.config.get_text_config().final_logit_softcapping

    def project_and_sum_ce(chunk_hidden, chunk_targets):
        logits = base.lm_head(chunk_hidden)
        # Match Gemma4ForConditionalGeneration.forward exactly: softcap in
        # projection dtype, THEN the loss function's FP32 upcast.
        if cap is not None:
            logits = logits / cap
            logits = torch.tanh(logits)
            logits = logits * cap
        return F.cross_entropy(logits.float(), chunk_targets.to(logits.device), reduction="sum")

    loss_sum = None
    for start in range(0, targets.numel(), chunk_size):
        stop = start + chunk_size
        chunk_hidden = hidden[batch_indices[start:stop], prediction_indices[start:stop]]
        chunk_targets = targets[start:stop]
        # A plain loop of CE sums retains ALL logits for backward. This
        # checkpoint stores only hidden states/targets and recomputes the CE.
        chunk_loss = checkpoint(project_and_sum_ce, chunk_hidden, chunk_targets,
                                use_reentrant=False)
        loss_sum = chunk_loss if loss_sum is None else loss_sum + chunk_loss
    denominator = values.get("num_items_in_batch")
    if denominator is None:
        denominator = targets.numel()
    elif isinstance(denominator, torch.Tensor):
        if denominator.numel() != 1 or denominator.detach().item() <= 0:
            raise ValueError("num_items_in_batch must be a positive scalar")
        denominator = denominator.to(loss_sum.device)
    elif not isinstance(denominator, (int, float)) or denominator <= 0:
        raise ValueError("num_items_in_batch must be a positive scalar")
    output = Gemma4CausalLMOutputWithPast(
        loss=loss_sum / denominator, logits=None,
        past_key_values=outputs.past_key_values,
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
        shared_kv_states=getattr(outputs, "shared_kv_states", None),
    )
    return_dict = values.get("return_dict")
    if return_dict is None:
        return_dict = base.config.return_dict
    return output if return_dict else output.to_tuple()


def install_selected_token_loss(model, *, chunk_size=32):
    """Bind a loss-only Gemma4 forward underneath PEFT (and future DDP).

    Calling this twice is idempotent and updates the chunk size. Save/load the
    PEFT adapter normally; this runtime optimization is intentionally not
    serialized into the adapter and must be installed again when training.
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    base = _gemma4_base(model)
    base._p005_loss_chunk_size = chunk_size
    if hasattr(base, "_p005_original_forward"):
        return model
    original = base.forward
    signature = inspect.signature(original)
    base._p005_original_forward = original

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        values = dict(bound.arguments)
        values.update(values.pop("kwargs", {}))
        if values.get("labels") is None:
            return original(*args, **kwargs)
        return selected_token_forward(self, chunk_size=self._p005_loss_chunk_size, **values)

    # ``original`` is already bound. Add self to the wrapper's unbound
    # signature so inspection after MethodType binding still exposes input_ids
    # correctly (Transformers generation inspects this signature).
    forward.__signature__ = signature.replace(parameters=[
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        *signature.parameters.values(),
    ])
    base.forward = types.MethodType(forward, base)
    return model
