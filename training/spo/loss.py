"""Masked PPO-Clip with a global action-token denominator.

Torch is imported lazily and tensors remain on their input device; no CUDA
availability or device is required.  There is deliberately no old/behavior TIS
ratio: behavior logprobs are the original sampling evidence.
"""

from __future__ import annotations

import importlib
import math
from typing import Any


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError("spo_ppo_loss requires torch only when computing the loss") from exc


def microbatch_weight(local_action_tokens: int, global_action_tokens: int) -> float:
    """Return local/global token mass for logging or external accumulators.

    ``spo_ppo_loss`` already divides by the global denominator, so callers
    should sum its microbatch losses rather than applying this weight twice.
    """
    for value, name in ((local_action_tokens, "local_action_tokens"),
                        (global_action_tokens, "global_action_tokens")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if global_action_tokens <= 0 or local_action_tokens > global_action_tokens:
        raise ValueError("require 0 <= local_action_tokens <= global_action_tokens and global > 0")
    return local_action_tokens / global_action_tokens


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def spo_ppo_loss(current_logprobs: Any, behavior_logprobs: Any, action_mask: Any,
                 advantages: Any, *, clip_epsilon: float = 0.2,
                 clip_epsilon_high: float | None = 0.28,
                 global_action_tokens: int | None = None,
                 reference_logprobs: Any | None = None,
                 reference_kl_coefficient: float = 0.0) -> Any:
    """Compute one microbatch's contribution to the global SPO actor loss.

    Shapes of current/behavior/mask must match.  Advantages may be scalar,
    token-shaped, or one value per leading trajectory/call row.  For a whole
    batch, omit ``global_action_tokens`` to use its mask sum.  For gradient
    accumulation, pass the same whole-update mask sum to every microbatch and
    sum the returned scalar losses; do not independently average microbatches.
    """
    torch = _torch()
    low = _finite(clip_epsilon, "clip_epsilon")
    if not 0.0 <= low < 1.0:
        raise ValueError("clip_epsilon must satisfy 0 <= epsilon < 1")
    high = low if clip_epsilon_high is None else _finite(
        clip_epsilon_high, "clip_epsilon_high"
    )
    if high < 0:
        raise ValueError("clip_epsilon_high must be non-negative")
    kl_coefficient = _finite(reference_kl_coefficient, "reference_kl_coefficient")
    if kl_coefficient < 0:
        raise ValueError("reference_kl_coefficient must be non-negative")
    current = torch.as_tensor(current_logprobs)
    if not current.is_floating_point():
        raise ValueError("current_logprobs must be floating point")
    behavior = torch.as_tensor(
        behavior_logprobs, dtype=current.dtype, device=current.device
    ).detach()
    mask = torch.as_tensor(action_mask, dtype=current.dtype, device=current.device)
    if current.shape != behavior.shape or current.shape != mask.shape:
        raise ValueError("current logprobs, behavior logprobs and mask shapes must match")
    if bool(((mask != 0) & (mask != 1)).any().item()):
        raise ValueError("action_mask must be binary")
    local_tokens = int(mask.sum().item())
    if global_action_tokens is None:
        global_action_tokens = local_tokens
    microbatch_weight(local_tokens, global_action_tokens)
    selected = mask != 0
    if local_tokens:
        if not bool(torch.isfinite(current[selected]).all().item()):
            raise ValueError("current logprobs must be finite at action tokens")
        if not bool(torch.isfinite(behavior[selected]).all().item()):
            raise ValueError("behavior logprobs must be finite at action tokens")
    advantage = torch.as_tensor(advantages, dtype=current.dtype, device=current.device)
    if advantage.ndim == 0:
        advantage = advantage.expand_as(current)
    elif advantage.shape == current.shape:
        pass
    elif current.ndim >= 2 and advantage.ndim == 1 \
            and advantage.shape[0] == current.shape[0]:
        advantage = advantage.reshape(
            (current.shape[0],) + (1,) * (current.ndim - 1)
        ).expand_as(current)
    else:
        raise ValueError(
            "advantages must be scalar, token-shaped, or one value per leading row"
        )
    if local_tokens and not bool(torch.isfinite(advantage[selected]).all().item()):
        raise ValueError("advantages must be finite at action tokens")
    denominator = float(global_action_tokens)
    if local_tokens == 0:
        # Preserve a differentiable zero for an empty microbatch in a nonempty update.
        return current.sum() * 0.0
    ratio = torch.exp(current - behavior)
    if not bool(torch.isfinite(ratio[selected]).all().item()):
        raise ValueError("PPO ratio overflowed; check current/behavior logprob evidence")
    clipped = torch.clamp(ratio, 1.0 - low, 1.0 + high)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    loss = -((surrogate * mask).sum() / denominator)
    if kl_coefficient:
        if reference_logprobs is None:
            raise ValueError("reference_logprobs are required when reference KL is enabled")
        reference = torch.as_tensor(
            reference_logprobs, dtype=current.dtype, device=current.device
        )
        if reference.shape != current.shape:
            raise ValueError("reference logprobs must match current logprobs")
        if not bool(torch.isfinite(reference[selected]).all().item()):
            raise ValueError("reference logprobs must be finite at action tokens")
        log_difference = reference - current
        token_kl = torch.exp(log_difference) - log_difference - 1.0
        if not bool(torch.isfinite(token_kl[selected]).all().item()):
            raise ValueError("reference KL overflowed")
        loss = loss + kl_coefficient * (token_kl * mask).sum() / denominator
    return loss


class SPOLoss:
    """Reusable trainer configuration; call arguments match ``spo_ppo_loss``."""

    def __init__(self, *, clip_epsilon: float = 0.2,
                 clip_epsilon_high: float | None = 0.28,
                 reference_kl_coefficient: float = 0.0):
        self.clip_epsilon = clip_epsilon
        self.clip_epsilon_high = clip_epsilon_high
        self.reference_kl_coefficient = reference_kl_coefficient

    def __call__(self, current_logprobs: Any, behavior_logprobs: Any, action_mask: Any,
                 advantages: Any, *, global_action_tokens: int | None = None,
                 reference_logprobs: Any | None = None) -> Any:
        return spo_ppo_loss(
            current_logprobs, behavior_logprobs, action_mask, advantages,
            clip_epsilon=self.clip_epsilon,
            clip_epsilon_high=self.clip_epsilon_high,
            global_action_tokens=global_action_tokens,
            reference_logprobs=reference_logprobs,
            reference_kl_coefficient=self.reference_kl_coefficient,
        )


masked_ppo_clip_loss = spo_ppo_loss
