"""Asynchronous official-worker execution without global training stops."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Protocol, runtime_checkable

from ..rollout.request_manager import RequestManager
from .worker import OfficialWorker


@runtime_checkable
class EndpointBridge(Protocol):
    def register(self, identities: list[str], version: int) -> Any: ...
    def rollout_token(self, identity: str) -> str: ...
    @property
    def url(self) -> str: ...
    def recover_full_rollout_calls(self, identity: str, version: int) -> list[Mapping[str, Any]]: ...
    def release_rollouts(self, identities: list[str]) -> Any: ...
    def cancel_rollouts(self, identities: list[str], *, reason: str = "rollout_cancelled") -> Any: ...


def _behavior(assignment: Any) -> tuple[int, str]:
    adapter = getattr(assignment, "behavior_adapter", None)
    version = getattr(adapter, "version", None)
    digest = getattr(adapter, "sha256", None)
    if type(version) is not int or version < 0 or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("assignment must carry a versioned behavior adapter digest")
    return version, digest


def _request_hash(trajectory_id: str, request_id: str, payload: Mapping[str, Any], adapter: Mapping[str, Any]) -> str:
    canonical = json.dumps({
        "trajectory_id": trajectory_id,
        "request_id": request_id,
        "prompt_ids": list(payload.get("prompt_ids", payload.get("input_ids", []))),
        "completion_ids": list(payload.get("completion_ids", payload.get("output_ids", []))),
        "behavior_adapter": dict(adapter),
    }, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ExecutionManager:
    def __init__(
        self,
        worker: OfficialWorker,
        endpoint: EndpointBridge,
        request_manager: RequestManager,
        *,
        max_workers: int = 8,
        max_attempts: int = 2,
        task_timeout_seconds: float | None = 3600.0,
        emit: Callable[..., None] | None = None,
    ):
        if not isinstance(worker, OfficialWorker):
            raise TypeError("worker must be OfficialWorker")
        if not isinstance(endpoint, EndpointBridge):
            raise TypeError("endpoint must expose trajectory-scoped registration and full call recovery")
        if not isinstance(request_manager, RequestManager):
            raise TypeError("request_manager must be RequestManager")
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("max_workers must be positive")
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if task_timeout_seconds is not None and (
            isinstance(task_timeout_seconds, bool) or not isinstance(task_timeout_seconds, (int, float))
            or not task_timeout_seconds > 0
        ):
            raise ValueError("task timeout must be positive")
        self.worker = worker
        self.endpoint = endpoint
        self.requests = request_manager
        self.max_attempts = max_attempts
        self.task_timeout = None if task_timeout_seconds is None else float(task_timeout_seconds)
        self.emit = emit
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="spo-worker")
        self._lock = threading.RLock()
        self._paused = False
        self._attempts = 0
        self._active: dict[str, dict[str, Any]] = {}
        self._shutdown = False

    def _event(self, name: str, **fields: Any) -> None:
        if self.emit is not None:
            self.emit(name, **fields)

    @contextmanager
    def paused(self) -> Iterator[None]:
        with self._lock:
            self._paused = True
        try:
            yield
        finally:
            with self._lock:
                self._paused = False

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {"paused": self._paused, "active": len(self._active), "shutdown": self._shutdown}

    def submit(self, assignment: Any, rollout_engine: Any) -> Future:
        trajectory_id = getattr(assignment, "trajectory_id", None)
        task_key = getattr(assignment, "task_key", None)
        task_id = getattr(task_key, "task_id", None)
        if not isinstance(trajectory_id, str) or not trajectory_id or not isinstance(task_id, str):
            raise ValueError("assignment must carry trajectory_id and task_key.task_id")
        if not hasattr(rollout_engine, "run"):
            raise TypeError("rollout_engine must expose run(assignment, worker_result)")
        version, _digest = _behavior(assignment)
        with self._lock:
            if self._shutdown:
                raise RuntimeError("execution manager is shut down")
            if self._paused:
                raise RuntimeError("dispatch is paused for a training update")
            if trajectory_id in self._active:
                raise ValueError("trajectory is already executing")
            outer: Future = Future()
            state = {"cancel": threading.Event(), "executor_future": None, "outcome": outer}
            self._active[trajectory_id] = state
        executor_future = self._pool.submit(self._run_and_settle, assignment, rollout_engine, version, outer, state)
        with self._lock:
            state["executor_future"] = executor_future
        return outer

    def cancel(self, trajectory_id: str, *, reason: str = "cancelled") -> None:
        with self._lock:
            state = self._active.get(trajectory_id)
            executor_future = state["executor_future"] if state else None
        if state is None:
            return
        state["cancel"].set()
        try:
            self.endpoint.cancel_rollouts([trajectory_id], reason=reason)
        except Exception:
            pass
        if executor_future is not None:
            executor_future.cancel()

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = True) -> None:
        with self._lock:
            self._shutdown = True
            self._paused = True
            states = list(self._active.values())
        for state in states:
            state["cancel"].set()
        if cancel_pending:
            for state in states:
                future = state.get("executor_future")
                if future is not None:
                    future.cancel()
        self._pool.shutdown(wait=wait, cancel_futures=cancel_pending)

    def _run_and_settle(self, assignment: Any, rollout_engine: Any, version: int, outer: Future, state: dict[str, Any]) -> None:
        trajectory_id = assignment.trajectory_id
        try:
            if state["cancel"].is_set():
                raise TimeoutError("trajectory was cancelled before execution")
            trajectory = self._execute(assignment, rollout_engine, version, state["cancel"])
            if not outer.done():
                outer.set_result(trajectory)
        except BaseException as error:
            if not outer.done():
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    outer.set_exception(error)
                elif state["cancel"].is_set() and not isinstance(error, TimeoutError):
                    outer.set_exception(TimeoutError(f"trajectory cancelled: {trajectory_id}"))
                else:
                    outer.set_exception(error)
        finally:
            with self._lock:
                self._active.pop(trajectory_id, None)
            self._event("worker_settled", trajectory_id=trajectory_id)

    def _execute(self, assignment: Any, rollout_engine: Any, version: int, cancel_event: threading.Event) -> Any:
        trajectory_id = assignment.trajectory_id
        digest = _behavior(assignment)[1]
        adapter = {"version": version, "sha256": digest}
        attempt_index = self._next_attempt()
        try:
            self.endpoint.register([trajectory_id], version)
            token = self.endpoint.rollout_token(trajectory_id)
        except Exception as error:
            return rollout_engine.exclude(assignment, f"endpoint_registration_{type(error).__name__}: {error}")
        outcome: Mapping[str, Any] | None = None
        try:
            for attempt_number in range(self.max_attempts):
                if cancel_event.is_set():
                    return rollout_engine.exclude(assignment, "worker_cancelled_before_attempt")
                attempt = self.worker.prepare_attempt(assignment, attempt_index + attempt_number)
                deadline = None if self.task_timeout is None else time.time() + self.task_timeout
                try:
                    outcome = self.worker.run(
                        assignment, attempt,
                        endpoint_token_path=attempt.paths.credential_path,
                        endpoint_token=token,
                        cancel_event=cancel_event,
                        deadline_unix=deadline,
                    )
                except TimeoutError as error:
                    if cancel_event.is_set():
                        return rollout_engine.exclude(assignment, "worker_cancelled")
                    if attempt_number + 1 >= self.max_attempts:
                        return rollout_engine.exclude(assignment, f"worker_timeout: {error}")
                    continue
                if self._retry_worker(outcome) and attempt_number + 1 < self.max_attempts:
                    self._event("worker_attempt_retry", trajectory_id=trajectory_id, attempt=attempt.attempt_id)
                    continue
                break
            assert outcome is not None
            self._bridge_endpoint_calls(trajectory_id, adapter)
            return rollout_engine.run(assignment, dict(outcome))
        finally:
            try:
                self.endpoint.release_rollouts([trajectory_id])
            except Exception:
                pass

    def _next_attempt(self) -> int:
        with self._lock:
            value = self._attempts
            self._attempts += 1
            return value

    @staticmethod
    def _retry_worker(outcome: Mapping[str, Any] | None) -> bool:
        if not isinstance(outcome, Mapping):
            return True
        if outcome.get("worker_error"):
            return True
        result = outcome.get("result")
        if not isinstance(result, dict):
            return True
        if result.get("infrastructure_error") and not result.get("verdict_valid"):
            return True
        return False

    def _bridge_endpoint_calls(self, trajectory_id: str, adapter: Mapping[str, Any]) -> int:
        rows = self.endpoint.recover_full_rollout_calls(trajectory_id, int(adapter["version"]))
        bridged = 0
        for row in sorted(rows, key=lambda item: int(item.get("call_index", 0))):
            request_id = row.get("request_id")
            result = row.get("result")
            if not isinstance(request_id, str) or not isinstance(result, Mapping):
                continue
            key = f"endpoint-{request_id}"
            payload = dict(result)
            sampling = dict(payload.get("sampling") or {})
            sampling.setdefault("seed", -1)
            payload["sampling"] = sampling
            payload["behavior_adapter"] = dict(adapter)
            payload["adapter_sha256"] = adapter["sha256"]
            request_hash = _request_hash(trajectory_id, request_id, payload, adapter)
            stored, _created = self.requests.create_call(
                trajectory_id, adapter, key, request_hash=request_hash,
                metadata={"behavior_adapter": dict(adapter), "endpoint_request_id": request_id},
            )
            self.requests.commit_generated(
                stored["request_id"], payload, trajectory_id=trajectory_id, behavior=adapter
            )
            self.requests.acknowledge(stored["request_id"], trajectory_id, behavior=adapter)
            bridged += 1
        return bridged
