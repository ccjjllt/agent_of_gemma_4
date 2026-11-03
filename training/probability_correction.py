"""One decoupled token-importance measure for GRPO and its KL diagnostics.

Only the completion-token vectors are promoted to float64. Full-prefix model
scoring and the caller's loss reduction remain outside this module.
"""

from __future__ import annotations

import math


CORRECTION_VERSION = "decoupled_token_tis_v1"


def correction_metadata(cap, enabled=True):
    """Describe the actual estimator, including its distributional limitations."""
    cap = float(cap)
    if not math.isfinite(cap) or cap < 1:
        raise ValueError("Token importance cap must be finite and at least one")
    return {
        "version": CORRECTION_VERSION,
        "enabled": bool(enabled),
        "cap": cap,
        "sampling_distribution": "behavior actions on fixed collected behavior prefixes",
        "base_measure": "min(old, cap * behavior)" if enabled else "behavior (correction disabled)",
        "weight": "min(old/behavior, cap), frozen" if enabled else "1",
        "policy_loss": "-w * min((current/old)*A, clip(current/old)*A)",
        "reference_kl": "w * (current/old) * K3(log(reference/current)); current/old remains differentiable",
        "reuse_kl": "w * K3(log(current/old))",
        "k3": "K3(d) = exp(d) - 1 - d",
        "conditional_kl_directions": "reference: current||reference; reuse: old||current",
        "conditional_kl_identity": "Requires untruncated old/behavior correction, or behavior=old; common support assumed",
        "truncation_bias": "Capped weights define an unnormalized measure; weighted K3 is a biased diagnostic, not exact KL",
        "aggregation": "No self-normalization; caller reduces token terms with its declared loss/diagnostic denominator",
        "scope": "Token correction does not reweight prefix occupancy or complete trajectories; not full-trajectory KL",
    }


def _weighted_k3(log_weight, delta):
    """Compute exp(log_weight) * K3(delta) without overflowing factors first.

    Evaluate branches separately: torch.where would still evaluate an unsafe
    unused exponential and can contaminate its backward pass with NaNs.
    """
    import torch

    result = (log_weight + delta) * 0.0
    positive = delta > 1.0
    small = (delta.abs() < 1e-3) & (delta != 0)
    remaining = ~(positive | small) & (delta != 0)
    # Empty selected tensors are valid too; avoid a device-to-host .any() sync
    # for every branch on each GPU call.
    d = delta[positive]
    # K3(d) = exp(d) * (1 - (1+d)*exp(-d)). Combine exponents
    # before exp, e.g. exp(-998)*K3(999) is finite.
    residual = torch.log1p(-(1.0 + d) * torch.exp(-d))
    result[positive] = torch.exp(log_weight[positive] + d + residual)
    d = delta[small]
    # K3(d) = d^2/2 * (1 + d/3 + d^2/12 + ...). The series
    # avoids subtracting nearly equal values at zero. At this cutoff the
    # omitted term is below float64 precision; log form handles large w.
    remainder = d * (1 / 3 + d * (1 / 12 + d * (1 / 60 + d * (1 / 360 + d / 2520))))
    log_k3 = 2 * torch.log(d.abs()) - math.log(2.0) + torch.log1p(remainder)
    result[small] = torch.exp(log_weight[small] + log_k3)
    d = delta[remaining]
    # Includes d < -1, where expm1(d)-d has no cancellation. Nonfinite
    # inputs remain nonfinite for the trainer to reject.
    result[remaining] = torch.exp(log_weight[remaining] + torch.log(torch.expm1(d) - d))
    return result


def corrected_token_terms(current, old, behavior, reference=None, *, advantage,
                          cap, enabled=True, epsilon_low=0.2, epsilon_high=0.28):
    """Return unreduced token loss and diagnostics for one complete call.

    ``old``, ``behavior``, ``reference`` and ``advantage`` are frozen numerical
    evidence. ``current`` remains live, including both occurrences in the
    reference term: detaching current/old would give the wrong KL gradient.
    No log-ratio clipping is used beyond the explicit TIS cap and PPO clip.
    """
    import torch

    cap = float(cap)
    low, high = float(epsilon_low), float(epsilon_high)
    if not math.isfinite(cap) or cap < 1:
        raise ValueError("Token importance cap must be finite and at least one")
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < 1 and high >= 0):
        raise ValueError("PPO clipping needs 0 <= epsilon_low < 1 and finite epsilon_high >= 0")
    current = torch.as_tensor(current).to(dtype=torch.float64)
    if current.ndim != 1 or not current.numel():
        raise ValueError("Current probabilities must be a nonempty complete-call token vector")

    def frozen(value, name, *, scalar=False):
        value = torch.as_tensor(value, dtype=torch.float64, device=current.device).detach()
        if scalar and value.ndim == 0:
            return value.expand_as(current)
        if value.shape != current.shape:
            raise ValueError(f"{name} must match the complete current token vector")
        return value

    old = frozen(old, "old")
    behavior = frozen(behavior, "behavior")
    advantage = frozen(advantage, "advantage", scalar=True)
    reference = None if reference is None else frozen(reference, "reference")
    mismatch = old - behavior
    log_cap = math.log(cap)
    log_weight = mismatch.clamp(max=log_cap) if enabled else torch.zeros_like(old)
    weights = log_weight.exp()
    log_ratio = current - old
    log_lower, log_upper = math.log1p(-low), math.log1p(high)
    ratio_lower, ratio_upper = log_ratio < log_lower, log_ratio > log_upper
    clipped = ((advantage > 0) & ratio_upper) | ((advantage < 0) & ratio_lower)

    # Positive advantages clip only the upper ratio; negative ones clip only
    # the lower ratio. Form w*rho in log space, without materializing rho.
    policy_loss = log_ratio * 0.0 + log_weight * 0.0
    nonzero = advantage != 0
    ratio = log_ratio[nonzero]
    a = advantage[nonzero]
    selected = torch.where(a > 0, ratio.clamp(max=log_upper), ratio.clamp(min=log_lower))
    policy_loss[nonzero] = -a * torch.exp(log_weight[nonzero] + selected)
    # PPO saturation/zero advantage must not hide invalid probability evidence.
    policy_loss = policy_loss + (current - current) + (old - old) + (behavior - behavior) + (advantage - advantage)

    return {
        "policy_loss": policy_loss,
        "reference_kl": (None if reference is None else
                         _weighted_k3(log_weight + log_ratio, reference - current)),
        "old_kl": _weighted_k3(log_weight, log_ratio),
        "weights": weights,
        "log_ratio": log_ratio,
        "clipped": clipped,
        "correction_clipped": ((mismatch > log_cap) if enabled else torch.zeros_like(old, dtype=torch.bool)),
        "ratio_lower": ratio_lower,
        "ratio_upper": ratio_upper,
    }
