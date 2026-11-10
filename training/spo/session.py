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

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

from ..trajectory.ledger import TrajectoryLedger
from ..trajectory.schema import BehaviorAdapter, UpdateBatch


class BarrierBlocked(RuntimeError):
    """Publish refused: old-policy trajectories still pin the rollout adapter."""


def commit_optimized_update(trainer: Any, policy_after: BehaviorAdapter,
                            version_chain: Sequence[BehaviorAdapter] | None = None) -> UpdateBatch:
    """Commit a post-optimizer batch and release it. Returns CONSUMED batch."""
    batch = getattr(trainer, "_batch", None)
    active = getattr(trainer, "_active", None)
    if batch is None or active is None:
        raise RuntimeError("trainer has no active SPO batch")
    if batch.phase not in ("OPTIMIZED", "EXPORTED", "REGISTERED", "ACTIVATED"):
        raise RuntimeError("only a post-optimizer update can be committed")
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


class VersionBarrier:
    """Explicit policy-version barrier: hold, drain, publish, resume.

    1. Hold the scheduler: no new task is dispatched on the old policy.
    2. Pause the execution manager and drain in-flight trajectories.
    3. Wait until no trajectory pins the rollout adapter (refcount zero).
    4. Publish the new adapter; old references are already gone, so the
       double buffer cannot refuse and starvation is impossible.
    5. Release the hold: sampling resumes on the new policy.

    The caller performs training/export/commit inside :meth:`update_window`;
    the barrier owns dispatch gating and the publish precondition.
    """

    def __init__(self, scheduler: Any, execution: Any, buffers: Any,
                 *, drain_timeout_seconds: float = 600.0,
                 emit: Callable[..., None] | None = None):
        self.scheduler = scheduler
        self.execution = execution
        self.buffers = buffers
        if isinstance(drain_timeout_seconds, bool) \
                or not isinstance(drain_timeout_seconds, (int, float)) \
                or drain_timeout_seconds <= 0:
            raise ValueError("drain_timeout_seconds must be positive")
        self.drain_timeout_seconds = float(drain_timeout_seconds)
        self.emit = emit

    def _event(self, name: str, **fields: Any) -> None:
        if self.emit is not None:
            self.emit(name, **fields)

    @contextmanager
    def update_window(self) -> Iterator[None]:
        """Hold dispatches and pause workers; release on exit."""
        self.scheduler.hold_dispatch()
        self._event("version_barrier_hold")
        paused = self.execution.paused()
        paused.__enter__()
        try:
            yield
        finally:
            try:
                paused.__exit__(None, None, None)
            finally:
                self.scheduler.release_dispatch()
                self._event("version_barrier_released")

    def publish_or_raise(self, timeout: float | None = None) -> bool:
        """Wait for rollout references to clear, then publish or raise."""
        limit = self.drain_timeout_seconds if timeout is None else float(timeout)
        if not self.buffers.wait_drained(timeout=limit):
            raise BarrierBlocked(
                f"rollout adapter still pinned by {self.buffers.rollout_refcount} trajectories")
        published = self.buffers.publish()
        if not published:
            raise BarrierBlocked(
                f"publish refused with {self.buffers.rollout_refcount} rollout references")
        self._event("version_barrier_published")
        return True
