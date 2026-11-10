"""Complete-trajectory FIFO buffer backed by :class:`TrajectoryLedger`."""

from __future__ import annotations

from typing import Sequence

from .ledger import TrajectoryLedger
from .schema import BehaviorAdapter, UpdateBatch


class TrajectoryBuffer:
    """Session/trainer interface; it owns no second database or in-memory queue."""

    def __init__(self, ledger: TrajectoryLedger, *, target_action_tokens: int = 32768,
                 max_policy_lag: int = 2, advantage_epsilon: float = 1.0e-6):
        if isinstance(target_action_tokens, bool) or not isinstance(target_action_tokens, int) \
                or target_action_tokens <= 0:
            raise ValueError("target_action_tokens must be positive")
        if isinstance(max_policy_lag, bool) or not isinstance(max_policy_lag, int) \
                or max_policy_lag < 0:
            raise ValueError("max_policy_lag must be non-negative")
        if not isinstance(advantage_epsilon, (int, float)) or advantage_epsilon <= 0:
            raise ValueError("advantage_epsilon must be positive")
        self.ledger = ledger
        self.target_action_tokens = target_action_tokens
        self.max_policy_lag = max_policy_lag
        self.advantage_epsilon = float(advantage_epsilon)

    def offer(self, trajectory_id: str) -> str:
        return self.ledger.offer_trajectory(trajectory_id)

    def reserve(self, update_id: str, *, policy_before: BehaviorAdapter,
                force: bool = False) -> UpdateBatch | None:
        return self.ledger.reserve_update(
            update_id, policy_before=policy_before,
            target_action_tokens=self.target_action_tokens,
            max_policy_lag=self.max_policy_lag,
            advantage_epsilon=self.advantage_epsilon, force=force,
        )

    def consume(self, update_id: str, *, policy_after: BehaviorAdapter,
                kl_value: float, token_count: int, estimator: str,
                version_chain: Sequence[BehaviorAdapter] | None = None) -> UpdateBatch:
        """Atomically commit KL and consume; safe to retry after OPTIMIZED."""
        return self.ledger.commit_update(
            update_id, policy_after=policy_after, kl_value=kl_value,
            token_count=token_count, estimator=estimator, version_chain=version_chain,
        )

    def stale(self, trajectory_id: str, reason: str) -> None:
        self.ledger.mark_stale(trajectory_id, reason)

    mark_stale = stale

    def state(self) -> dict[str, int]:
        return self.ledger.buffer_state()
