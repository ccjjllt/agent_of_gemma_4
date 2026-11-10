"""Public task metadata and online capability state for :class:`SPOSampler`.

Only three public fields are retained from an explicitly supplied official
``tasks.jsonl``.  Judge-private hashes are bound later by the caller; this module
never invents them and never discovers or reads heldout files.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable, Iterator, Mapping

from ..trajectory.schema import TaskKey


_FORBIDDEN = re.compile(r"(^|[/_.-])(heldout|holdout|reserved|secret|hidden)($|[/_.-])", re.I)
_TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _public_path(path: str | Path, *, filename: str | None = None) -> Path:
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("an explicit local public data path is required")
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or _FORBIDDEN.search(str(source)):
        raise ValueError("only an explicit non-heldout public JSONL file is accepted")
    if filename is not None and source.name != filename:
        raise ValueError(f"the official public task file must be named {filename}")
    return source


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    repo: str
    base_commit: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or _TASK_ID.fullmatch(self.task_id) is None \
                or ".." in self.task_id.split("/"):
            raise ValueError("invalid public task identity")
        if not isinstance(self.repo, str) or not self.repo.strip():
            raise ValueError("public task repo must be non-empty")
        if not isinstance(self.base_commit, str) or not self.base_commit.strip():
            raise ValueError("public task base_commit must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"task_id": self.task_id, "repo": self.repo,
                "base_commit": self.base_commit}


@dataclass(frozen=True)
class CapabilityState:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    timeouts: int = 0
    mastered: bool = False
    long_term_failure: bool = False
    failure_reasons: Mapping[str, int] | None = None
    tool_errors: Mapping[str, int] | None = None
    infrastructure_failures: int = 0
    consecutive_infrastructure_failures: int = 0
    retry_after_unix: float = 0.0
    last_outcome: str | None = None
    last_failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts, "successes": self.successes,
            "failures": self.failures,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "timeouts": self.timeouts, "mastered": self.mastered,
            "long_term_failure": self.long_term_failure,
            "failure_reasons": dict(self.failure_reasons or {}),
            "tool_errors": dict(self.tool_errors or {}),
            "infrastructure_failures": self.infrastructure_failures,
            "consecutive_infrastructure_failures": self.consecutive_infrastructure_failures,
            "retry_after_unix": self.retry_after_unix,
            "last_outcome": self.last_outcome,
            "last_failure_reason": self.last_failure_reason,
        }


class TaskCatalog:
    """Read public tasks and maintain capability, not reward/value, state.

    ``healthy`` is directly suitable for ``SPOSampler(..., health=catalog.healthy)``.
    Mastered and persistently failing tasks remain healthy and therefore retain
    the sampler's epsilon coverage.  Only timeouts/infrastructure failures back off.
    """

    def __init__(self, tasks_jsonl: str | Path, *, include_ids: Iterable[str] | None = None,
                 exclude_ids: Iterable[str] | None = None, mastered_after: int = 3,
                 long_term_failure_after: int = 8,
                 infrastructure_backoff_seconds: float = 60.0):
        self.source = _public_path(tasks_jsonl, filename="tasks.jsonl")
        self.source_sha256 = _digest(self.source)
        if isinstance(include_ids, (str, bytes)) or isinstance(exclude_ids, (str, bytes)):
            raise TypeError("include_ids/exclude_ids must be task ID collections")
        included = None if include_ids is None else set(include_ids)
        excluded = set(exclude_ids or ())
        if included is not None and included & excluded:
            raise ValueError("a task cannot be both included and excluded")
        for name, value, minimum in (
            ("mastered_after", mastered_after, 1),
            ("long_term_failure_after", long_term_failure_after, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if isinstance(infrastructure_backoff_seconds, bool) or not isinstance(
                infrastructure_backoff_seconds, (int, float)) or not math.isfinite(
                float(infrastructure_backoff_seconds)) or infrastructure_backoff_seconds <= 0:
            raise ValueError("infrastructure_backoff_seconds must be positive and finite")
        self.mastered_after = mastered_after
        self.long_term_failure_after = long_term_failure_after
        self.infrastructure_backoff_seconds = float(infrastructure_backoff_seconds)

        tasks: list[PublicTask] = []
        seen: set[str] = set()
        with self.source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed public task row {line_number}") from exc
                if not isinstance(row, Mapping):
                    raise ValueError(f"public task row {line_number} is not an object")
                split = row.get("split")
                if row.get("heldout") is True or (isinstance(split, str) and _FORBIDDEN.search(split)):
                    raise ValueError("heldout rows cannot enter the public training catalog")
                identifier = row.get("instance_id")
                task = PublicTask(identifier, row.get("repo"), row.get("base_commit"))
                if task.task_id in seen:
                    raise ValueError(f"duplicate public instance_id: {task.task_id}")
                seen.add(task.task_id)
                if task.task_id in excluded or (included is not None and task.task_id not in included):
                    continue
                tasks.append(task)
        if included is not None and included - seen:
            raise ValueError(f"requested public tasks were not found: {sorted(included - seen)}")
        if not tasks:
            raise ValueError("no public training tasks remain after selection")
        self._tasks = tuple(tasks)
        self._by_id = {task.task_id: task for task in self._tasks}
        self._bindings: dict[str, TaskKey] = {}
        self._failures = {task.task_id: Counter() for task in self._tasks}
        self._tool_errors = {task.task_id: Counter() for task in self._tasks}
        self._states = {task.task_id: CapabilityState() for task in self._tasks}
        self._events: dict[str, tuple[str, str]] = {}

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[PublicTask]:
        return iter(self._tasks)

    @property
    def tasks(self) -> tuple[PublicTask, ...]:
        return self._tasks

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    def get(self, task_id: str) -> PublicTask:
        try:
            return self._by_id[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown public training task: {task_id}") from exc

    def bind_task_keys(self, task_keys: Iterable[TaskKey]) -> None:
        """Bind judge-created keys without exposing their hashes in public rows."""
        bindings: dict[str, TaskKey] = {}
        for key in task_keys:
            if not isinstance(key, TaskKey):
                raise TypeError("sampler bindings must contain TaskKey records")
            task = self.get(key.task_id)
            if (key.repo, key.base_commit) != (task.repo, task.base_commit):
                raise ValueError("TaskKey identity differs from the official public catalog")
            if key.task_id in bindings and bindings[key.task_id] != key:
                raise ValueError("conflicting TaskKey binding")
            bindings[key.task_id] = key
        self._bindings.update(bindings)

    def sampler_tasks(self) -> tuple[TaskKey, ...]:
        missing = [task_id for task_id in self._by_id if task_id not in self._bindings]
        if missing:
            raise ValueError(f"TaskKey bindings are missing for: {missing}")
        return tuple(self._bindings[task_id] for task_id in self._by_id)

    def _task_id(self, value: str | TaskKey | PublicTask) -> str:
        if isinstance(value, TaskKey):
            task = self.get(value.task_id)
            if (value.repo, value.base_commit) != (task.repo, task.base_commit):
                raise ValueError("TaskKey does not match the public catalog")
            bound = self._bindings.get(value.task_id)
            if bound is not None and bound != value:
                raise ValueError("TaskKey differs from the catalog's judge binding")
            return value.task_id
        if isinstance(value, PublicTask):
            return self.get(value.task_id).task_id
        return self.get(value).task_id

    def healthy(self, task: str | TaskKey | PublicTask, *, now: float | None = None) -> bool:
        """Health callback for SPOSampler; capability labels never hard-exclude."""
        state = self._states[self._task_id(task)]
        current = time.time() if now is None else float(now)
        return state.consecutive_infrastructure_failures == 0 or state.retry_after_unix <= current

    spo_health = healthy

    @staticmethod
    def _error_counts(values: Iterable[str] | Mapping[str, int] | None) -> Counter:
        result: Counter = Counter()
        if values is None:
            return result
        if isinstance(values, Mapping):
            items = values.items()
            for name, count in items:
                if not isinstance(name, str) or not name.strip() or isinstance(count, bool) \
                        or not isinstance(count, int) or count <= 0:
                    raise ValueError("tool error counts require named positive integers")
                result[name] += count
            return result
        if isinstance(values, (str, bytes)):
            values = (values,)
        for name in values:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("tool errors must be non-empty names")
            result[name] += 1
        return result

    def record_outcome(self, task: str | TaskKey | PublicTask, *,
                       resolved: bool | None, failure_reason: str | None = None,
                       tool_errors: Iterable[str] | Mapping[str, int] | None = None,
                       timed_out: bool = False, infrastructure_error: str | None = None,
                       runtime_healthy: bool | None = None,
                       event_id: str | None = None) -> dict[str, Any]:
        """Record one online observation; unresolved infrastructure is not failure=0."""
        task_id = self._task_id(task)
        if resolved is not None and type(resolved) is not bool:
            raise ValueError("resolved must be an official bool verdict or None")
        if type(timed_out) is not bool:
            raise ValueError("timed_out must be bool")
        if infrastructure_error is not None and (not isinstance(infrastructure_error, str)
                                                  or not infrastructure_error.strip()):
            raise ValueError("infrastructure_error must be a non-empty string or None")
        if failure_reason is not None and (not isinstance(failure_reason, str)
                                           or not failure_reason.strip()):
            raise ValueError("failure_reason must be a non-empty string or None")
        if runtime_healthy is not None and type(runtime_healthy) is not bool:
            raise ValueError("runtime_healthy must be bool or None")
        errors = self._error_counts(tool_errors)
        event_fingerprint = hashlib.sha256(json.dumps({
            "task_id": task_id, "resolved": resolved, "failure_reason": failure_reason,
            "tool_errors": dict(errors), "timed_out": timed_out,
            "infrastructure_error": infrastructure_error,
            "runtime_healthy": runtime_healthy,
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if event_id is not None:
            if not isinstance(event_id, str) or not event_id.strip():
                raise ValueError("event_id must be a stable non-empty string")
            previous = self._events.get(event_id)
            if previous is not None:
                if previous != (task_id, event_fingerprint):
                    raise ValueError("event_id replay has different outcome evidence")
                return self.capability(task_id)

        previous = self._states[task_id]
        reasons = self._failures[task_id]
        tools = self._tool_errors[task_id]
        tools.update(errors)
        unhealthy = timed_out or infrastructure_error is not None
        healthy = (not unhealthy and resolved is not None) if runtime_healthy is None else runtime_healthy
        if unhealthy and healthy:
            raise ValueError("a timeout/infrastructure failure cannot be runtime healthy")
        if resolved is not None:
            attempts = previous.attempts + 1
            successes = previous.successes + int(resolved)
            failures = previous.failures + int(not resolved)
            consecutive_successes = previous.consecutive_successes + 1 if resolved else 0
            consecutive_failures = 0 if resolved else previous.consecutive_failures + 1
            mastered = consecutive_successes >= self.mastered_after
            long_term_failure = consecutive_failures >= self.long_term_failure_after
            reason = None if resolved else (failure_reason or "official_unresolved")
            if not resolved:
                reasons[reason] += 1
            outcome = "success" if resolved else "official_failure"
        else:
            attempts, successes, failures = previous.attempts, previous.successes, previous.failures
            consecutive_successes = previous.consecutive_successes
            consecutive_failures = previous.consecutive_failures
            mastered, long_term_failure = previous.mastered, previous.long_term_failure
            reason = failure_reason or infrastructure_error or ("timeout" if timed_out else None)
            if reason:
                reasons[reason] += 1
            outcome = "timeout" if timed_out else "infrastructure_failure" if unhealthy else "unresolved"
        infrastructure_failures = previous.infrastructure_failures + int(unhealthy)
        consecutive_infrastructure = (previous.consecutive_infrastructure_failures + 1 if unhealthy
                                      else 0 if healthy else previous.consecutive_infrastructure_failures)
        retry_after = previous.retry_after_unix
        if unhealthy:
            retry_after = time.time() + self.infrastructure_backoff_seconds * 2 ** min(
                consecutive_infrastructure - 1, 10)
        elif healthy:
            retry_after = 0.0
        state = CapabilityState(
            attempts=attempts, successes=successes, failures=failures,
            consecutive_successes=consecutive_successes,
            consecutive_failures=consecutive_failures,
            timeouts=previous.timeouts + int(timed_out), mastered=mastered,
            long_term_failure=long_term_failure, failure_reasons=reasons,
            tool_errors=tools, infrastructure_failures=infrastructure_failures,
            consecutive_infrastructure_failures=consecutive_infrastructure,
            retry_after_unix=retry_after, last_outcome=outcome,
            last_failure_reason=reason,
        )
        self._states[task_id] = state
        if event_id is not None:
            self._events[event_id] = (task_id, event_fingerprint)
        return state.to_dict()

    observe = record_outcome

    def capability(self, task: str | TaskKey | PublicTask) -> dict[str, Any]:
        return self._states[self._task_id(task)].to_dict()

    def capability_snapshot(self) -> dict[str, Any]:
        return {
            "public_source_sha256": self.source_sha256,
            "mastered_after": self.mastered_after,
            "long_term_failure_after": self.long_term_failure_after,
            "health_scope": "timeouts_and_infrastructure_only",
            "tasks": {task_id: self.capability(task_id) for task_id in self._by_id},
        }
