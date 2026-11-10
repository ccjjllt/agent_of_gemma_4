"""SPO++ action-token-measure advantage normalization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..trajectory.schema import BehaviorAdapter, Trajectory, UpdateBatch


@dataclass(frozen=True)
class TokenStatistics:
    mean: float
    std: float
    total_action_tokens: int


def action_token_statistics(raw_advantages: Mapping[str, float],
                            action_token_counts: Mapping[str, int]) -> TokenStatistics:
    """Compute population mean/variance with each trajectory weighted by L_i."""
    if not raw_advantages or set(raw_advantages) != set(action_token_counts):
        raise ValueError("advantage and action-token keys must be identical and non-empty")
    total = 0
    weighted: list[tuple[int, float]] = []
    for trajectory_id, raw in raw_advantages.items():
        count = action_token_counts[trajectory_id]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("every normalized trajectory must have action tokens")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) \
                or not math.isfinite(float(raw)):
            raise ValueError("raw advantages must be finite")
        total += count
        weighted.append((count, float(raw)))
    mean = math.fsum(count * raw for count, raw in weighted) / total
    variance = math.fsum(count * (raw - mean) ** 2 for count, raw in weighted) / total
    return TokenStatistics(mean=mean, std=math.sqrt(max(0.0, variance)),
                           total_action_tokens=total)


def freeze_advantages(trajectories: Sequence[Trajectory], *, update_id: str,
                      policy_before: BehaviorAdapter, max_policy_lag: int | None = None,
                      epsilon: float = 1.0e-6) -> UpdateBatch:
    """Freeze raw A, token statistics and normalized A for crash-safe recovery.

    This pure function never updates V.  ValueTracker must already have stored
    each raw advantage using its pre-observation value.
    """
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) \
            or not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("advantage epsilon must be finite and positive")
    if max_policy_lag is not None and (
            isinstance(max_policy_lag, bool) or not isinstance(max_policy_lag, int)
            or max_policy_lag < 0):
        raise ValueError("max_policy_lag must be non-negative")
    if not trajectories:
        raise ValueError("cannot freeze an empty update")
    ids = tuple(item.trajectory_id for item in trajectories)
    if len(set(ids)) != len(ids):
        raise ValueError("an update cannot contain a trajectory twice")
    raw: dict[str, float] = {}
    counts: dict[str, int] = {}
    maximum_lag = 0
    for trajectory in trajectories:
        if not isinstance(trajectory, Trajectory) or trajectory.raw_advantage is None:
            raise ValueError("all trajectories need an observed raw advantage")
        if trajectory.action_token_count <= 0:
            raise ValueError("zero-action trajectories are NOOP, not PPO updates")
        raw[trajectory.trajectory_id] = trajectory.raw_advantage
        counts[trajectory.trajectory_id] = trajectory.action_token_count
        for call in trajectory.calls:
            lag = policy_before.version - call.behavior_adapter.version
            if lag < 0:
                raise ValueError("policy_before predates a behavior adapter")
            if max_policy_lag is not None and lag > max_policy_lag:
                raise ValueError("trajectory exceeds max policy lag")
            maximum_lag = max(maximum_lag, lag)
    statistics = action_token_statistics(raw, counts)
    denominator = statistics.std + float(epsilon)
    frozen = {
        trajectory_id: (value - statistics.mean) / denominator
        for trajectory_id, value in raw.items()
    }
    return UpdateBatch(
        update_id=update_id,
        policy_before=policy_before,
        trajectory_ids=ids,
        raw_advantages=raw,
        frozen_advantages=frozen,
        token_mean=statistics.mean,
        token_std=statistics.std,
        total_action_tokens=statistics.total_action_tokens,
        max_behavior_lag=maximum_lag,
        optimizer_cursor=0,
        phase="RESERVED",
    )
