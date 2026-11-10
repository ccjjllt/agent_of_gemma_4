"""Single-stream SPO session commit helper (CPU-safe, no GPU required).

Order: ledger.reserve_update (freezes SPO++ advantages) -> SPOTrainer.optimize_batch
(LEARNING -> OPTIMIZED + KL diagnostic) -> export candidate adapter -> this helper
(ledger.commit_update + trainer.acknowledge_commit) -> publish serving adapter.

Only real adapter SHA256 changes publish a new version; KL registry writes happen
exactly once inside ledger.commit_update.  Stale trajectories are excluded at
reserve time; a StaleTrajectoryError from the trainer aborts the batch for
operator handling (no silent zero rewards).
"""

from __future__ import annotations

from typing import Any, Sequence

from ..trajectory.ledger import TrajectoryLedger
from ..trajectory.schema import BehaviorAdapter, UpdateBatch


def commit_optimized_update(trainer: Any, policy_after: BehaviorAdapter,
                            version_chain: Sequence[BehaviorAdapter] | None = None) -> UpdateBatch:
    """Commit an OPTIMIZED trainer batch and release it. Returns CONSUMED batch."""
    batch = getattr(trainer, "_batch", None)
    active = getattr(trainer, "_active", None)
    if batch is None or active is None:
        raise RuntimeError("trainer has no active SPO batch")
    if batch.phase != "OPTIMIZED":
        raise RuntimeError("only an OPTIMIZED update can be committed")
    stats = active.get("stats", {})
    kl_value = stats.get("kl_value")
    token_count = stats.get("token_count")
    estimator = stats.get("estimator")
    if not isinstance(kl_value, (int, float)) or not isinstance(token_count, int):
        raise RuntimeError("OPTIMIZED batch is missing its KL evidence")
    ledger = trainer.ledger
    if not isinstance(ledger, TrajectoryLedger):
        raise TypeError("trainer.ledger must be a TrajectoryLedger")
    if not isinstance(policy_after, BehaviorAdapter):
        raise TypeError("policy_after must be a BehaviorAdapter")
    consumed = ledger.commit_update(
        batch.update_id, policy_after=policy_after,
        kl_value=float(kl_value), token_count=token_count,
        estimator=str(estimator), version_chain=version_chain,
    )
    trainer.acknowledge_commit(batch.update_id)
    return consumed
