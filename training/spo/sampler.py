"""Uncertainty-prioritized independent-task sampling with epsilon coverage."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence

from ..trajectory.schema import TaskKey
from .value_tracker import ValueTracker


class SPOSampler:
    """Sample ``sqrt(p(1-p)) + epsilon`` without encoding health in success_p.

    Repository/type multipliers are explicit coverage overrides.  Recent task
    exposure applies a ``1/(1+count)`` penalty.  Only infrastructure health
    filters hard-exclude tasks; mastered and persistently failing healthy tasks
    retain their epsilon probability.
    """

    def __init__(self, value_tracker: ValueTracker, *, epsilon: float = 0.05,
                 repo_overrides: Mapping[str, float] | None = None,
                 type_overrides: Mapping[str, float] | None = None):
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) \
                or not math.isfinite(float(epsilon)) or epsilon <= 0:
            raise ValueError("sampling epsilon must be finite and positive")
        self.value_tracker = value_tracker
        self.epsilon = float(epsilon)
        self.repo_overrides = self._multipliers(repo_overrides or {}, "repo_overrides")
        self.type_overrides = self._multipliers(type_overrides or {}, "type_overrides")

    @staticmethod
    def _multipliers(values: Mapping[str, float], name: str) -> dict[str, float]:
        if not isinstance(values, Mapping):
            raise ValueError(f"{name} must be a mapping")
        result: dict[str, float] = {}
        for key, value in values.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{name} keys must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} multipliers must be finite and positive")
            result[key] = float(value)
        return result

    def eligible_tasks(self, tasks: Iterable[TaskKey], *,
                       excluded_repos: Iterable[str] = (),
                       excluded_task_ids: Iterable[str] = (),
                       health: Callable[[TaskKey], bool] | None = None) -> tuple[TaskKey, ...]:
        repos = set(excluded_repos)
        task_ids = set(excluded_task_ids)
        eligible: list[TaskKey] = []
        seen: set[TaskKey] = set()
        for task in tasks:
            if not isinstance(task, TaskKey):
                raise TypeError("tasks must contain TaskKey records")
            if task in seen:
                raise ValueError("duplicate TaskKey in sampling catalog")
            seen.add(task)
            if task.repo in repos or task.task_id in task_ids:
                continue
            if health is not None and not health(task):
                continue
            eligible.append(task)
        return tuple(eligible)

    def weights(self, tasks: Iterable[TaskKey], *,
                repo_overrides: Mapping[str, float] | None = None,
                type_overrides: Mapping[str, float] | None = None,
                task_types: Mapping[str, str] | None = None,
                recent_exposure: Mapping[str, int] | None = None,
                excluded_repos: Iterable[str] = (),
                excluded_task_ids: Iterable[str] = (),
                health: Callable[[TaskKey], bool] | None = None) -> dict[TaskKey, float]:
        eligible = self.eligible_tasks(
            tasks, excluded_repos=excluded_repos, excluded_task_ids=excluded_task_ids,
            health=health,
        )
        repo_weights = self.repo_overrides if repo_overrides is None else self._multipliers(
            repo_overrides, "repo_overrides"
        )
        type_weights = self.type_overrides if type_overrides is None else self._multipliers(
            type_overrides, "type_overrides"
        )
        types = task_types or {}
        exposure = recent_exposure or {}
        result: dict[TaskKey, float] = {}
        for task in eligible:
            count = exposure.get(task.task_id, 0)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("recent_exposure counts must be non-negative integers")
            task_type = types.get(task.task_id)
            record = self.value_tracker.snapshot(task)
            uncertainty = math.sqrt(record.success_p * (1.0 - record.success_p))
            weight = uncertainty + self.epsilon
            weight *= repo_weights.get(task.repo, 1.0)
            if task_type is not None:
                weight *= type_weights.get(task_type, 1.0)
            weight /= 1.0 + count
            if not math.isfinite(weight) or weight <= 0:
                raise RuntimeError("invalid SPO sampling weight")
            result[task] = weight
        return result

    def probabilities(self, tasks: Iterable[TaskKey], **kwargs: object) -> dict[TaskKey, float]:
        weights = self.weights(tasks, **kwargs)
        total = math.fsum(weights.values())
        if total <= 0:
            return {}
        return {task: weight / total for task, weight in weights.items()}

    def sample(self, tasks: Sequence[TaskKey], *, rng: random.Random | None = None,
               **kwargs: object) -> TaskKey:
        probabilities = self.probabilities(tasks, **kwargs)
        if not probabilities:
            raise IndexError("no healthy tasks are available for sampling")
        generator = rng or random
        keys = list(probabilities)
        return generator.choices(keys, weights=[probabilities[key] for key in keys], k=1)[0]

    def sample_many(self, tasks: Sequence[TaskKey], count: int, *,
                    rng: random.Random | None = None, **kwargs: object) -> tuple[TaskKey, ...]:
        """Weighted sampling without replacement, preserving all remaining epsilon mass."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        generator = rng or random
        remaining = list(self.eligible_tasks(
            tasks,
            excluded_repos=kwargs.get("excluded_repos", ()),
            excluded_task_ids=kwargs.get("excluded_task_ids", ()),
            health=kwargs.get("health"),
        ))
        if count > len(remaining):
            raise ValueError("cannot sample more unique tasks than are healthy")
        selected: list[TaskKey] = []
        # Recompute probabilities after removal; static coverage/health inputs remain explicit.
        for _ in range(count):
            probabilities = self.probabilities(remaining, **kwargs)
            keys = list(probabilities)
            choice = generator.choices(
                keys, weights=[probabilities[key] for key in keys], k=1
            )[0]
            selected.append(choice)
            remaining.remove(choice)
        return tuple(selected)
