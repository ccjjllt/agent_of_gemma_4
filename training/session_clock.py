"""Training session time: exclude only witnessed idle GPU-queue intervals.

The official runner and verifier files stay unchanged. A private function clone
replaces only the agent-loop timeout factory and its code-executor constructor.
The original wall clock is always retained; this is not official wall-time
evaluation. Unknown timing never earns credit or an unlimited session.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import functools
import hashlib
import inspect
import math
import re
import threading
import textwrap
import time
import types
from urllib.parse import urlsplit


class SessionTimingError(RuntimeError):
    """Unconfirmed training-time accounting, never a model-quality failure."""


_SOURCE_HASHES = {
    "runner": "1a5672a86e495b3c9a86bdea4b7fc44ebc571798a69f5ba7c754fa5df3b3f079",
    "evaluator": "5dbccdfa2001826994997dcbc16b99262b36ecdfe8749ca2ea36653f1d9287c8",
    "elapsed": "72c2c92e17487128bc635f8829573706f20c25c0b13ec11bc2eed599422dc6e0",
    "remaining": "1e1ee79a657e1a058d2bca77e659dbf2f8f1aa56ebb32ed6d41b6ca2783ea2f4",
    "budget": "940e4202b6f664e7bf11fecd4fb71b23b989c285908635bb6e7b9ecbb87812bb",
    "executor": "0bcb9730702592f60ba40431a9eff961f01c19cdd2e65f1399e30580ea79a262",
}


def _guard(function, name):
    source = textwrap.dedent(inspect.getsource(function))
    if hashlib.sha256(source.encode()).hexdigest() != _SOURCE_HASHES[name]:
        raise SessionTimingError(f"The pinned official {name} timing boundary changed")


def validate_official_timing_sources():
    """CPU preflight for the installed harness; reads code, never task assets.

    The caller configures the runtime's ordinary offline SDK environment before
    these imports. No evaluator, sandbox, model, or session is instantiated.
    """
    from swegemma.context import SwegemmaContext
    from swegemma.evaluate import Evaluator
    from swegemma.harness.agent_runner import run_agent_sandbox
    from swegemma.sandbox import AdkSandboxCodeExecutor

    for function, name in (
        (run_agent_sandbox, "runner"), (Evaluator._run_agent_sandbox, "evaluator"),
        (SwegemmaContext.agent_elapsed_seconds.fget, "elapsed"),
        (SwegemmaContext.remaining_time_seconds.fget, "remaining"),
        (SwegemmaContext.check_budget, "budget"), (AdkSandboxCodeExecutor.execute_code, "executor"),
    ):
        _guard(function, name)
    return {"protocol": "gemma.training_session_clock.v1", "source_sha256": dict(_SOURCE_HASHES)}


def _clone(function, replacements):
    clone = types.FunctionType(function.__code__, {**function.__globals__, **replacements},
                               function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    functools.update_wrapper(clone, function)
    return clone


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _intersect(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


class SessionClock:
    def __init__(self, endpoint, rollout_id, *, mode="queue_excluded", poll_seconds=0.2, api_key=None):
        if mode not in {"queue_excluded", "wall"}:
            raise ValueError("session_clock must be queue_excluded or wall")
        if not math.isfinite(poll_seconds) or not 0.02 <= poll_seconds <= 1.0:
            raise ValueError("Queue timing poll interval must be finite and between 0.02 and 1 second")
        if urlsplit(endpoint).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Queue accounting requires the same-host loopback endpoint and monotonic clock")
        if (not isinstance(api_key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", api_key)
                or api_key == rollout_id):
            raise ValueError("Session timing requires a private rollout API key")
        self.endpoint, self.rollout_id = endpoint.rstrip("/"), rollout_id
        self.api_key = api_key
        self.mode, self.poll_seconds = mode, float(poll_seconds)
        self.started = self.stopped = None
        self._lock = threading.RLock()
        self._waiting, self._working = {}, {}
        self._wait_intervals, self._work_intervals = [], []
        self._server_intervals = []
        self._snapshot = None
        self._epoch = self._version = self._sequence = None
        self._client = None
        self._refresh_lock = None
        self.error = None
        self.polls = self.poll_failures = 0
        self.active_timeout_reached = False
        self.timeout_at = self.timeout_seconds = None

    def start(self, at):
        with self._lock:
            if self.started is not None:
                raise SessionTimingError("A training session clock cannot be restarted")
            self.started = float(at)

    def stop(self, at):
        with self._lock:
            if self.stopped is None:
                self.stopped = float(at)

    @contextmanager
    def _interval(self, active, completed):
        token = object()
        with self._lock:
            active[token] = time.perf_counter()
        try:
            yield
        finally:
            with self._lock:
                completed.append((active.pop(token), time.perf_counter()))

    def model_wait(self):
        return self._interval(self._waiting, self._wait_intervals)

    def tool_work(self):
        return self._interval(self._working, self._work_intervals)

    def queue_credit(self, at=None):
        with self._lock:
            if self.started is None or self._snapshot is None:
                return 0.0
            end = min(time.perf_counter() if at is None else at, self._snapshot)
            if self.stopped is not None:
                end = min(end, self.stopped)
            window = [(self.started, end)]
            waits = _union(self._wait_intervals + [(start, end) for start in self._waiting.values()])
            work = _union(self._work_intervals + [(start, end) for start in self._working.values()])
            eligible = _intersect(_intersect(self._server_intervals, waits), window)
            blocked = _intersect(eligible, work)
            return max(0.0, math.fsum(end - start for start, end in eligible)
                       - math.fsum(end - start for start, end in blocked))

    def elapsed(self, at=None, *, adjusted=True):
        with self._lock:
            if self.started is None:
                return 0.0
            end = time.perf_counter() if at is None else at
            if self.stopped is not None:
                end = min(end, self.stopped)
            wall = max(0.0, end - self.started)
        return max(0.0, wall - self.queue_credit(end)) if adjusted else wall

    def observe(self, payload, requested_at, received_at):
        """Accept only increasing same-host snapshots with explicit intervals."""
        if not isinstance(payload, dict) or payload.get("protocol") != "gemma.queue_timing.v1":
            raise SessionTimingError("The endpoint has no supported queue timing receipt")
        if payload.get("rollout_id") != self.rollout_id:
            raise SessionTimingError("Queue timing belongs to another rollout")
        sample, total = payload.get("snapshot_monotonic"), payload.get("queue_excluded_seconds")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in (sample, total)):
            raise SessionTimingError("Queue timing contains invalid durations")
        if sample < requested_at - 0.01 or sample > received_at + 0.01 or total < 0:
            raise SessionTimingError("Queue timing does not share this worker's monotonic clock")
        epoch, version, sequence = payload.get("server_epoch"), payload.get("policy_version"), payload.get("snapshot_seq")
        if (not isinstance(epoch, str) or not epoch or type(version) is not int or version < 0
                or type(sequence) is not int or sequence < 0):
            raise SessionTimingError("Queue timing has invalid scheduler identity")
        intervals = payload.get("eligible_intervals")
        if not isinstance(intervals, list):
            raise SessionTimingError("Queue timing omits its interval evidence")
        parsed = []
        for item in intervals:
            if not isinstance(item, dict):
                raise SessionTimingError("Invalid queue interval evidence")
            start, end = item.get("start"), item.get("end")
            if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (start, end))
                    or start < 0 or end < start or end > sample + 1e-8):
                raise SessionTimingError("Invalid queue interval boundaries")
            parsed.append((float(start), float(end)))
        merged = _union(parsed)
        if not math.isclose(math.fsum(end - start for start, end in merged), total,
                            rel_tol=1e-9, abs_tol=1e-6):
            raise SessionTimingError("Queue intervals do not match their cumulative duration")
        with self._lock:
            if self._epoch is not None and (epoch != self._epoch or version != self._version):
                raise SessionTimingError("The scheduler changed during this agent session")
            if self._sequence is not None and (sequence <= self._sequence or sample < self._snapshot):
                raise SessionTimingError("Queue timing receipt is stale")
            # A later snapshot may extend an open interval, never retract any
            # previously witnessed queue time, including after cancellation.
            if self._server_intervals and _intersect(self._server_intervals, merged) != self._server_intervals:
                raise SessionTimingError("The scheduler retracted previously witnessed queue intervals")
            self._server_intervals, self._snapshot = merged, float(sample)
            self._epoch, self._version, self._sequence = epoch, version, sequence
            self.polls += 1

    async def refresh(self):
        import httpx

        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()
        async with self._refresh_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=1.0, trust_env=False)
            requested = time.perf_counter()
            try:
                response = await self._client.get(f"{self.endpoint}/rollouts/{self.rollout_id}/timing",
                                                  headers={"Authorization": "Bearer " + self.api_key})
                response.raise_for_status()
                self.observe(response.json(), requested, time.perf_counter())
            except (httpx.HTTPError, ValueError, SessionTimingError) as error:
                self.poll_failures += 1
                self.error = f"Queue timing could not be confirmed ({type(error).__name__})"
                raise SessionTimingError(self.error) from error

    def timeout(self, seconds):
        return _SessionTimeout(self, seconds)

    def assert_valid(self):
        if (self.mode == "queue_excluded" and self.timeout_at is not None
                and self.elapsed(self.timeout_at) + 1e-6 < self.timeout_seconds):
            self.error = self.error or "Session cancellation preceded its active allowance; queue evidence arrived too late"
        if self.timeout_at is not None and self.timeout_seconds is not None:
            self.active_timeout_reached = self.elapsed(self.timeout_at) + 1e-6 >= self.timeout_seconds
        if self.mode == "queue_excluded" and self.error is not None:
            raise SessionTimingError(self.error)

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def summary(self):
        return {"mode": self.mode, "wall_seconds": self.elapsed(adjusted=False),
                "queue_excluded_seconds": self.queue_credit(), "active_seconds": self.elapsed(),
                "charged_seconds": self.elapsed(adjusted=self.mode == "queue_excluded"),
                "active_timeout_reached": self.active_timeout_reached,
                "queue_receipt_polls": self.polls, "queue_receipt_failures": self.poll_failures,
                "timing_error": self.error,
                "semantics": "server queued-without-service intersected with client model-await, excluding concurrent tool work",
                "official_wallclock_evaluation": self.mode == "wall"}


class _SessionTimeout:
    def __init__(self, clock, seconds):
        self.clock, self.seconds = clock, seconds
        self.native = self.watcher = None
        self.deadline_perf = None

    async def __aenter__(self):
        # Start with the unchanged finite allowance. A later receipt extends
        # this deadline by confirmed credit only; unknown waiting never does.
        self.native = asyncio.timeout(self.seconds)
        if self.native.when() is not None:
            self.deadline_perf = time.perf_counter() + self.native.when() - asyncio.get_running_loop().time()
        await self.native.__aenter__()
        self.watcher = asyncio.create_task(self._watch(), name="training-session-clock")
        return self

    def _reschedule(self, remaining):
        if self.native.expired():
            return False
        self.deadline_perf = time.perf_counter() + max(0.0, remaining)
        self.native.reschedule(asyncio.get_running_loop().time() + max(0.0, remaining))
        return True

    async def _watch(self):
        try:
            while True:
                if self.native.expired():
                    return  # The cancelled Runner may still be unwinding.
                # Always obtain live evidence while an LLM is awaited. Only
                # already witnessed queue intervals extend the native timeout.
                with self.clock._lock:
                    waiting = bool(self.clock._waiting)
                if waiting or self.clock._snapshot is None:
                    await self.clock.refresh()
                if self.native.expired():
                    return
                remaining = None if self.seconds is None else self.seconds - self.clock.elapsed()
                if self.clock.mode == "queue_excluded" and remaining is not None:
                    self._reschedule(remaining)
                    if remaining <= 0:
                        return
                delay = self.clock.poll_seconds if remaining is None else min(self.clock.poll_seconds, max(0.001, remaining / 2))
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self.clock.error = self.clock.error or f"Queue timing monitor failed ({type(error).__name__})"
            if self.clock.mode == "queue_excluded":
                # The official runner still performs its normal cancellation
                # and final acceptance. The worker subsequently rejects this
                # unconfirmed timing as infrastructure, never a zero reward.
                self._reschedule(0.0)

    async def __aexit__(self, exc_type, exc, traceback):
        if self.native.expired():
            # Use the deadline that triggered cancellation, not the end of a
            # possibly slow async-generator finally/cleanup afterwards.
            self.clock.timeout_at = self.deadline_perf
            self.clock.timeout_seconds = self.seconds
        self.watcher.cancel()
        try:
            await self.watcher
        except asyncio.CancelledError:
            pass
        except BaseException as error:
            self.clock.error = self.clock.error or f"Queue timing monitor cleanup failed ({type(error).__name__})"
        return await self.native.__aexit__(exc_type, exc, traceback)


def timed_context_class(base, clock):
    for function, name in ((base.agent_elapsed_seconds.fget, "elapsed"),
                           (base.remaining_time_seconds.fget, "remaining"), (base.check_budget, "budget")):
        _guard(function, name)

    class TrainingContext(base):
        def start_agent_session(self):
            super().start_agent_session()
            clock.start(self.agent_start_time)

        def stop_agent_session(self):
            super().stop_agent_session()
            if self.agent_end_time is not None:
                clock.stop(self.agent_end_time)

        @property
        def agent_elapsed_seconds(self):
            wall = super().agent_elapsed_seconds
            return max(0.0, wall - clock.queue_credit()) if clock.mode == "queue_excluded" else wall

        def create_tools(self):
            result = super().create_tools()
            for name, tool in result.items():
                if inspect.iscoroutinefunction(tool):
                    @functools.wraps(tool)
                    async def wrapped(*args, _tool=tool, **kwargs):
                        with clock.tool_work():
                            return await _tool(*args, **kwargs)
                else:
                    @functools.wraps(tool)
                    def wrapped(*args, _tool=tool, **kwargs):
                        with clock.tool_work():
                            return _tool(*args, **kwargs)
                result[name] = wrapped
            return result

    return TrainingContext


def bind_timed_runner(evaluator, context, clock):
    """Bind private clones to one evaluator; no imported module is patched."""
    original_method = evaluator._run_agent_sandbox.__func__
    runner = original_method.__globals__["run_agent_sandbox"]
    executor = runner.__globals__["AdkSandboxCodeExecutor"]
    _guard(original_method, "evaluator")
    _guard(runner, "runner")
    _guard(executor.execute_code, "executor")

    class TrainingExecutor(executor):
        def execute_code(self, invocation_context, code_execution_input):
            with clock.tool_work():
                # Keep the original executor's command cap and budget checks.
                # A private model copy avoids mutating a shared concurrent
                # executor's start_time, or changing the official context start.
                instance = self
                if clock.mode == "queue_excluded" and context.agent_start_time is not None:
                    instance = self.model_copy(update={"start_time": context.agent_start_time + clock.queue_credit()})
                return executor.execute_code(instance, invocation_context, code_execution_input)

    class AsyncioScope:
        timeout = staticmethod(clock.timeout)

        def __getattr__(self, name):
            return getattr(asyncio, name)

    def wrap_skills(agent, _visited=None):
        runner.__globals__["_wrap_skill_toolset_tools"](agent, _visited)
        visited = set()

        def walk(current):
            if current is None or id(current) in visited:
                return
            visited.add(id(current))
            for item in getattr(current, "tools", []) or []:
                walk(getattr(item, "agent", None))
                for tool in getattr(item, "_tools", []) or []:
                    if getattr(tool, "name", None) not in {"load_skill", "load_skill_resource", "run_skill_script"}:
                        continue
                    original = tool.run_async

                    @functools.wraps(original)
                    async def run(*args, _original=original, **kwargs):
                        with clock.tool_work():
                            return await _original(*args, **kwargs)

                    object.__setattr__(tool, "run_async", run)
            for child in getattr(current, "sub_agents", []) or []:
                walk(child)

        walk(agent)

    private_runner = _clone(runner, {"asyncio": AsyncioScope(), "AdkSandboxCodeExecutor": TrainingExecutor,
                                     "_wrap_skill_toolset_tools": wrap_skills})
    private_method = _clone(original_method, {"run_agent_sandbox": private_runner})
    evaluator._run_agent_sandbox = types.MethodType(private_method, evaluator)

