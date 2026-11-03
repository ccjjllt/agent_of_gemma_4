"""A positive first warmup update with one explicit linear LR contract.

The schedule advances after applied optimizer steps, not after attempted or
skipped steps. Importing its numerical contract does not import torch.
"""

from __future__ import annotations

from functools import partial
import math


SCHEDULE_VERSION = "positive_warmup_linear_v1"


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def lr_multiplier(completed_steps, total_steps, warmup_steps):
    """Multiplier prepared before the next optimizer update.

    k=0 is the first real update. The warmup branch samples 1/W through
    W/W, while the original linear decay branch is unchanged. Thus 0<W<T
    has two consecutive peak-LR updates. A horizon fully inside warmup keeps
    the requested W; it does not silently shorten warmup to T.
    """
    k = _integer(completed_steps, "completed_steps")
    total = _integer(total_steps, "total_steps", minimum=1)
    warmup = _integer(warmup_steps, "warmup_steps")
    if k >= total:
        return 0.0
    if warmup > 0 and k < warmup:
        return (k + 1) / warmup
    return (total - k) / max(1, total - warmup)


def resolve_schedule(grpo, *, total_steps, learning_rate):
    """Resolve an effective JSON-safe schedule shared by comparison arms."""
    if grpo.get("lr_schedule_version", SCHEDULE_VERSION) != SCHEDULE_VERSION:
        raise ValueError("Unsupported LR schedule version")
    if grpo.get("lr_scheduler_type", "linear") != "linear":
        raise ValueError("positive_warmup_linear_v1 supports only the linear schedule")
    total = _integer(total_steps, "total_steps", minimum=1)
    if isinstance(learning_rate, bool):
        raise ValueError("learning_rate must be a finite nonnegative number")
    base_lr = float(learning_rate)
    if not math.isfinite(base_lr) or base_lr < 0:
        raise ValueError("learning_rate must be a finite nonnegative number")
    if "warmup_steps" in grpo:
        warmup = _integer(grpo["warmup_steps"], "warmup_steps")
    else:
        ratio = grpo.get("warmup_ratio", 0.0)
        if isinstance(ratio, bool) or not math.isfinite(float(ratio)) or float(ratio) < 0:
            raise ValueError("warmup_ratio must be finite and nonnegative")
        warmup = math.ceil(total * float(ratio))
    first_lr = base_lr * lr_multiplier(0, total, warmup)
    if base_lr > 0 and first_lr == 0:
        raise ValueError("Positive learning_rate underflows at the first scheduled update")
    return {
        "version": SCHEDULE_VERSION,
        "type": "linear",
        "total_steps": total,
        "warmup_steps": warmup,
        "base_lr": base_lr,
        "first_update_lr": first_lr,
        "index_semantics": "k = completed non-skipped optimizer steps before the next update; initially zero",
        "advance_semantics": "Prepare f(0) at construction; advance once after each applied optimizer step; do not advance on skipped attempts",
        "boundary_semantics": "k>=T:0; k<W:(k+1)/W; otherwise:(T-k)/max(1,T-W). If 0<W<T the peak repeats; W>=T is retained without shortening",
    }


def make_scheduler(optimizer, contract):
    """Create a real LambdaLR without a dummy update or an extra pre-step.

    For recovery, construct this scheduler first, then load optimizer state,
    then scheduler state. Their saved next LR and applied-step index are the
    authority; optimizer-attempt counters may include skipped steps.
    """
    import torch

    resolved = resolve_schedule(
        {"lr_schedule_version": contract.get("version"), "lr_scheduler_type": contract.get("type"),
         "warmup_steps": contract.get("warmup_steps")},
        total_steps=contract.get("total_steps"), learning_rate=contract.get("base_lr"),
    )
    if resolved != contract:
        raise ValueError("LR schedule contract does not match its declared semantics")
    for group in optimizer.param_groups:
        # LambdaLR's base_lrs must match the declared base, including when a
        # caller has already populated initial_lr. Never silently overwrite an
        # optimizer that has advanced; load its state after construction.
        if (float(group["lr"]) != contract["base_lr"]
                or float(group.get("initial_lr", group["lr"])) != contract["base_lr"]):
            raise ValueError("Construct the LR scheduler on a fresh optimizer at the declared base_lr")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        partial(lr_multiplier, total_steps=contract["total_steps"], warmup_steps=contract["warmup_steps"]),
    )
    scheduler.schedule_contract = dict(contract)
    return scheduler
