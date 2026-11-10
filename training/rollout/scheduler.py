"""Independent-task rollout scheduling with SPO sampling and fair queues.

The scheduler owns selection only.  :class:`ExecutionManager` owns worker
threads, so selecting a trajectory never forms a same-task cohort or waits for
another rollout.  A behavior adapter is captured when an assignment is dispatched
and remains immutable for that trajectory.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import math
import random
import threading
import time
from typing import Any, Callable, Iterable, Mapping
import uuid

from ..spo.sampler import SPOSampler
from ..trajectory.schema import BehaviorAdapter, TaskKey
from .policy_version import AdapterManifest, PolicyVersion


AdapterSource = (
    BehaviorAdapter | PolicyVersion | AdapterManifest |
    Callable[[], BehaviorAdapter | PolicyVersion | AdapterManifest]
)


def behavior_adapter(value: BehaviorAdapter | PolicyVersion | AdapterManifest) -> BehaviorAdapter:
    if isinstance(value, BehaviorAdapter):
        return value
    if isinstance(value, (PolicyVersion, AdapterManifest)):
        return BehaviorAdapter(version=value.version, sha256=value.adapter_sha256)
    raise TypeError("behavior adapter must be BehaviorAdapter, PolicyVersion or AdapterManifest")


@dataclass(frozen=True)
class RolloutAssignment:
    task_key: TaskKey
    trajectory_id: str
    stream_sequence: int
    behavior_adapter: BehaviorAdapter
    queue: str
    attempt_id: str
    selected_at_unix: float
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _QueuedTask:
    task: TaskKey
    queue: str
    sequence: int
    queued_at: float


@dataclass
class _ActiveTask:
    assignment: RolloutAssignment
    started_at: float


class RolloutScheduler:
    """Select one independent task and maintain short/long aging queues.

    Unknown runtimes enter the short queue.  Repository multipliers preserve the
    sampler's configured values while reducing recent overexposure; no mastered
    or persistently failing task is hard-deleted.  Long tasks have a separate
    active cap and both queues acquire priority with actual waiting time.
    """

    def __init__(
        self,
        tasks: Iterable[TaskKey],
        sampler: SPOSampler,
        adapter_source: AdapterSource,
        *,
        seed: int = 3407,
        task_types: Mapping[str, str] | None = None,
        health: Callable[[TaskKey], bool] | None = None,
        long_threshold_seconds: float = 300.0,
        ewma_alpha: float = 0.2,
        aging_seconds: float = 60.0,
        max_long_active: int = 1,
        max_queued: int = 128,
        exposure_window_seconds: float = 3600.0,
    ):
        self.tasks = tuple(tasks)
        if not self.tasks or any(not isinstance(item, TaskKey) for item in self.tasks):
            raise ValueError("tasks must be a non-empty iterable of TaskKey records")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("duplicate TaskKey in rollout catalog")
        if not isinstance(sampler, SPOSampler):
            raise TypeError("sampler must be SPOSampler")
        self.sampler = sampler
        self.adapter_source = adapter_source
        if not callable(adapter_source):
            behavior_adapter(adapter_source)  # Validate before any worker starts.
        self.task_types = dict(task_types or {})
        self.health = health
        for name, value, positive in (
            ("long_threshold_seconds", long_threshold_seconds, True),
            ("ewma_alpha", ewma_alpha, True),
            ("aging_seconds", aging_seconds, True),
            ("exposure_window_seconds", exposure_window_seconds, True),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < float(ewma_alpha) <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        for name, value in (("max_long_active", max_long_active), ("max_queued", max_queued)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.long_threshold_seconds = float(long_threshold_seconds)
        self.ewma_alpha = float(ewma_alpha)
        self.aging_seconds = float(aging_seconds)
        self.max_long_active = max_long_active
        self.max_queued = max_queued
        self.exposure_window_seconds = float(exposure_window_seconds)
        self._rng = random.Random(seed)
        self._lock = threading.RLock()
        self._queues = {"short": deque(), "long": deque()}
        self._active: dict[str, _ActiveTask] = {}
        self._duration_ewma: dict[str, float] = {}
        self._exposures: deque[tuple[float, str, str]] = deque()
        self._recent_tasks: Counter[str] = Counter()
        self._recent_repos: Counter[str] = Counter()
        self._sequence = 0
        self._stream_sequence = 0

    def _queue_for(self, task: TaskKey) -> str:
        # Unknown tasks deliberately enter short with their epsilon exploration mass.
        return "long" if self._duration_ewma.get(task.task_id, 0.0) > self.long_threshold_seconds else "short"

    def _trim_exposures(self, now: float) -> None:
        cutoff = now - self.exposure_window_seconds
        while self._exposures and self._exposures[0][0] < cutoff:
            _, task_id, repo = self._exposures.popleft()
            self._recent_tasks[task_id] -= 1
            self._recent_repos[repo] -= 1
            if not self._recent_tasks[task_id]:
                del self._recent_tasks[task_id]
            if not self._recent_repos[repo]:
                del self._recent_repos[repo]

    def _sample_one(self, now: float) -> None:
        if sum(len(queue) for queue in self._queues.values()) >= self.max_queued:
            return
        excluded = {item.assignment.task_key.task_id for item in self._active.values()}
        excluded.update(item.task.task_id for queue in self._queues.values() for item in queue)
        available = tuple(item for item in self.tasks if item.task_id not in excluded)
        if not available:
            return
        self._trim_exposures(now)
        repo_overrides: dict[str, float] = {}
        for repo in {item.repo for item in available}:
            # This is a soft coverage multiplier, not an eligibility filter.
            coverage = 1.0 / (1.0 + self._recent_repos.get(repo, 0))
            repo_overrides[repo] = self.sampler.repo_overrides.get(repo, 1.0) * coverage
        selected = self.sampler.sample(
            available,
            rng=self._rng,
            repo_overrides=repo_overrides,
            task_types=self.task_types,
            recent_exposure=dict(self._recent_tasks),
            health=self.health,
        )
        self._sequence += 1
        queue_name = self._queue_for(selected)
        self._queues[queue_name].append(_QueuedTask(selected, queue_name, self._sequence, now))

    def _select_queued(self, now: float) -> _QueuedTask:
        long_active = sum(item.assignment.queue == "long" for item in self._active.values())
        candidates: list[tuple[float, float, int, _QueuedTask]] = []
        for queue in self._queues.values():
            for item in queue:
                if item.queue == "long" and long_active >= self.max_long_active:
                    continue
                age = max(0.0, now - item.queued_at) / self.aging_seconds
                uncovered = 1.0 if not self._recent_repos.get(item.task.repo) else 0.0
                # Aging eventually dominates repository tie-breaking and queue identity.
                priority = age + uncovered
                candidates.append((priority, -item.queued_at, -item.sequence, item))
        if not candidates:
            raise IndexError("no dispatchable queued task (long-worker cap may be reached)")
        selected = max(candidates, key=lambda row: row[:3])[3]
        self._queues[selected.queue].remove(selected)
        return selected

    def next_assignment(self) -> RolloutAssignment:
        """Sample/enqueue at most one task, then dispatch the highest-priority one."""
        with self._lock:
            now = time.monotonic()
            self._sample_one(now)
            selected = self._select_queued(now)
            adapter = behavior_adapter(
                self.adapter_source() if callable(self.adapter_source) else self.adapter_source
            )
            self._stream_sequence += 1
            trajectory_id = f"traj-{self._stream_sequence:012d}-{uuid.uuid4().hex[:12]}"
            assignment = RolloutAssignment(
                task_key=selected.task,
                trajectory_id=trajectory_id,
                stream_sequence=self._stream_sequence,
                behavior_adapter=adapter,
                queue=selected.queue,
                attempt_id=f"attempt-{uuid.uuid4().hex}",
                selected_at_unix=time.time(),
                provenance={
                    "selection": "single_stream_spo_v1",
                    "queue_sequence": selected.sequence,
                    "queue_wait_seconds": max(0.0, now - selected.queued_at),
                    "recent_task_exposure": self._recent_tasks.get(selected.task.task_id, 0),
                    "recent_repo_exposure": self._recent_repos.get(selected.task.repo, 0),
                },
            )
            self._active[trajectory_id] = _ActiveTask(assignment, now)
            self._exposures.append((now, selected.task.task_id, selected.task.repo))
            self._recent_tasks[selected.task.task_id] += 1
            self._recent_repos[selected.task.repo] += 1
            return assignment

    def submit_next(self, execution_manager: Any = None, rollout_engine: Any = None) -> Any:
        """Return an assignment, or atomically hand it to an ExecutionManager.

        The optional form is ``submit_next(manager, engine)`` and returns the
        manager's Future.  Adapter capture and active registration happen before
        submission, so a submission failure can safely requeue the exact task.
        """
        if execution_manager is None:
            if rollout_engine is not None:
                raise ValueError("rollout_engine requires execution_manager")
            return self.next_assignment()
        if rollout_engine is None:
            raise ValueError("submit_next(manager, engine) requires a RolloutEngine")
        assignment = self.next_assignment()
        try:
            future = execution_manager.submit(assignment, rollout_engine)
        except BaseException:
            self.requeue(assignment)
            raise
        future.add_done_callback(lambda _future: self.complete(assignment))
        return future

    def requeue(self, assignment: RolloutAssignment) -> None:
        """Return an unsubmitted/cancelled selection without recording task health."""
        with self._lock:
            active = self._active.pop(assignment.trajectory_id, None)
            if active is None:
                return
            self._sequence += 1
            self._queues[assignment.queue].appendleft(_QueuedTask(
                assignment.task_key, assignment.queue, self._sequence, time.monotonic(),
            ))

    def complete(self, assignment: RolloutAssignment, elapsed_seconds: float | None = None) -> None:
        """Release capacity and update only execution-time EWMA, never success_p."""
        with self._lock:
            active = self._active.pop(assignment.trajectory_id, None)
            if active is None:
                return
            elapsed = time.monotonic() - active.started_at if elapsed_seconds is None else elapsed_seconds
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) \
                    or not math.isfinite(float(elapsed)) or elapsed < 0:
                raise ValueError("elapsed_seconds must be finite and non-negative")
            task_id = assignment.task_key.task_id
            previous = self._duration_ewma.get(task_id)
            self._duration_ewma[task_id] = (
                float(elapsed) if previous is None else
                self.ewma_alpha * float(elapsed) + (1.0 - self.ewma_alpha) * previous
            )

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queued": {name: len(queue) for name, queue in self._queues.items()},
                "active": len(self._active),
                "active_long": sum(item.assignment.queue == "long" for item in self._active.values()),
                "duration_ewma_seconds": dict(self._duration_ewma),
                "recent_repo_exposure": dict(self._recent_repos),
                "next_stream_sequence": self._stream_sequence + 1,
            }
