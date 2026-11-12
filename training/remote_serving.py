"""Whole-request serving through the lease-guarded inference backend.

This engine exposes the same loop-facing interface as
:class:`training.generation.GenerationEngine` (capacity/can_admit/
active_count/cancel/step/close plus a ``prefixes`` shim) so
:class:`training.endpoint.PolicyEndpoint` keeps its OpenAI protocol, ACK,
persistence and TITO records unchanged.  Token production, batching and
prefix caching live in the remote engine; every request is submitted with
its trajectory-pinned adapter through
:meth:`training.adapter.double_buffer.DoubleBuffer.submit`, which holds the
inference side of :class:`training.rollout.backend.GpuLease`.

Differences from the local continuous engine are explicit, not hidden:

- one pool thread per in-flight request; the server batches natively;
- no client-side KV, prefill chunking or prefix reuse accounting;
- ``first_token_seconds`` is unknown from the whole-completion response
  and stays ``None``; per-request wall times overlap and must not be summed
  as batch throughput (same rule as the local engine's metrics);
- response-parse, backend submission and lease errors become per-request
  failures surfaced through the endpoint (inspect `generation_summary`); a
  failed server-side abort is logged instead of silently dropped.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping


# The execution manager registers one request namespace per attempt
# (``<trajectory_id>-aNNNNNN``) and issues that namespace its own bearer token,
# while DoubleBuffer pins exactly one adapter per trajectory.  Serving resolves
# the adapter, so it must translate the attempt namespace back to its parent.
_ATTEMPT_NAMESPACE = re.compile(r"^(?P<trajectory>.+)-a\d{6}$")


class _NoClientPrefixes:
    """Prefix caching lives server-side; the loop only needs drop_owner."""

    def drop_owner(self, owner: str) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {"location": "inference_server", "client_entries": 0}


class RemoteServingEngine:
    """Serve endpoint requests via DoubleBuffer and whole-completion backend."""

    def __init__(self, policy: Any, double_buffer: Any, config: Mapping[str, Any] | None,
                 run_dir: str | Path, on_request_started: Any = None):
        if policy is None or double_buffer is None:
            raise ValueError("remote serving requires a policy and a double buffer")
        self.policy = policy
        self.buffers = double_buffer
        self.config = dict(config or {})
        self.run_dir = Path(run_dir)
        self.on_request_started = on_request_started
        capacity = int(self.config.get("max_inflight",
                                       self.config.get("sglang_max_inflight",
                                                      self.config.get("max_batch_size", 16))))
        if capacity < 1:
            raise ValueError("max_inflight must be a positive integer")
        self._capacity = capacity
        self.generation_timeout = float(self.config.get("generation_timeout_seconds", 1800))
        if not math.isfinite(self.generation_timeout) or self.generation_timeout <= 0:
            raise ValueError("generation_timeout_seconds must be finite and positive")
        self.max_new_tokens = int(self.config.get("max_new_tokens", 16384))
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        self.policy_version = int(self.config.get("policy_version", 0))
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=capacity, thread_name_prefix="remote-serve")
        self._lock = threading.Lock()
        self._pending: dict[str, concurrent.futures.Future] = {}
        self._cancelled: set[str] = set()
        self._started_at: dict[str, float] = {}
        self._queued_at: dict[str, float] = {}
        self.prefixes = _NoClientPrefixes()
        self.closed = False
        self.started_at = time.perf_counter()
        self.completed = 0
        self.failed = 0
        self.cancelled_requests = 0
        self.sampled_tokens = 0
        self.max_observed_inflight = 0
        self.latency_sum = 0.0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def _check_abort(self) -> None:
        if getattr(self.policy, "abort_generation", None) is not None \
                and self.policy.abort_generation.is_set():
            raise RuntimeError("Generation cancelled by the enclosing run budget")

    def can_admit(self, prepared: Mapping[str, Any], incoming: Any = ()) -> bool:
        try:
            prompt = prepared["prompt_ids"]
            effective = min(int(prepared["effective_max_tokens"]), self.max_new_tokens)
        except (KeyError, TypeError, ValueError):
            return False
        if not prompt or effective < 1:
            return False
        if len(prompt) + effective > int(self.policy.max_context):
            return False
        return self.active_count + len(tuple(incoming)) < self._capacity

    def _serve(self, request_id: str, prepared: Mapping[str, Any], rollout_id: str,
               queued_at: float) -> tuple[str, Any]:
        started = time.perf_counter()
        prompt_ids = list(prepared["prompt_ids"])
        sampling = {
            "temperature": float(self.config.get("temperature", 1.0)),
            "top_p": float(self.config.get("top_p", 1.0)),
            "top_k": int(self.config.get("top_k", 0)),
            "max_new_tokens": min(int(prepared["effective_max_tokens"]), self.max_new_tokens),
            "seed": int(self.config.get("seed", -1)),
        }
        try:
            trajectory_id = rollout_id
            match = _ATTEMPT_NAMESPACE.match(rollout_id)
            if match is not None:
                trajectory_id = match.group("trajectory")
            response = self.buffers.submit(
                trajectory_id, prompt_ids,
                request_id=request_id, sampling=sampling,
                timeout=self.generation_timeout)
        except (ValueError, KeyError, TypeError) as error:
            return request_id, error
        completion = list(response.get("output_ids") or [])
        behavior = [float(value) for value in (response.get("logprobs") or [])]
        try:
            result = self.policy.finish_generation(
                dict(prepared), completion, behavior,
                generation_elapsed_seconds=time.perf_counter() - started,
                first_token_seconds=None,
                queue_wait_seconds=started - queued_at,
                reused_prefix_tokens=0,
                profiled_tokens=0,
                generation_backend="remote_gateway",
                timing_semantics="per-request wall time overlaps other requests; use generation_summary for throughput",
            )
        except (ValueError, KeyError, TypeError) as error:
            return request_id, error
        return request_id, result

    def step(self, incoming: Any) -> list[tuple[str, Any]]:
        """Submit new requests, then return whichever completions are ready."""
        if self.closed:
            raise RuntimeError("remote serving engine is closed")
        self._check_abort()
        incoming = list(incoming or [])
        for item in incoming:
            if not self.can_admit(item["prepared"]):
                raise ValueError("Scheduler exceeded the reserved serving capacity")
        with self._lock:
            for item in incoming:
                request_id = item["request_id"]
                if request_id in self._pending:
                    raise RuntimeError("Generation returned a duplicate identity")
                future = self._pool.submit(
                    self._serve, request_id, dict(item["prepared"]),
                    item["rollout_id"], item["queued_at"])
                self._pending[request_id] = future
                self._started_at[request_id] = time.perf_counter()
                self._queued_at[request_id] = item["queued_at"]
                self.max_observed_inflight = max(
                    self.max_observed_inflight, len(self._pending))
                if self.on_request_started is not None:
                    self.on_request_started(request_id, self._started_at[request_id])
            done_ids = [request_id for request_id, future in self._pending.items()
                        if future.done()]
            outputs = []
            for request_id in done_ids:
                future = self._pending.pop(request_id)
                self._started_at.pop(request_id, None)
                queued_at = self._queued_at.pop(request_id, None)
                if request_id in self._cancelled:
                    self._cancelled.discard(request_id)
                    continue
                try:
                    _, result = future.result()
                except BaseException as error:
                    outputs.append((request_id, error))
                    self.failed += 1
                    continue
                if isinstance(result, Exception):
                    self.failed += 1
                else:
                    self.completed += 1
                    completion = result.get("completion_ids") or []
                    self.sampled_tokens += len(completion)
                    if queued_at is not None:
                        self.latency_sum += max(
                            0.0, time.perf_counter() - queued_at)
                outputs.append((request_id, result))
            return outputs

    def cancel(self, request_ids: Any) -> list[str]:
        """Drop in-flight requests; late completions are swallowed in step()."""
        identities = set(request_ids or [])
        removed = []
        with self._lock:
            for request_id in identities:
                future = self._pending.get(request_id)
                if future is None:
                    continue
                removed.append(request_id)
                self._cancelled.add(request_id)
                future.cancel()
                try:
                    self.buffers.backend.cancel(request_id)
                except Exception as error:
                    # Do not hide a failed server-side abort: the client
                    # future is dropped anyway, so this event is the only
                    # evidence that the server may still be decoding.
                    self.failed += 1
                    print(json.dumps({
                        "event": "remote_cancel_failed",
                        "request_id": request_id,
                        "error": f"{type(error).__name__}: {error}",
                    }), flush=True)
                self.cancelled_requests += 1
        return removed

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
            self._cancelled.clear()
        for future in pending:
            future.cancel()
        self._pool.shutdown(wait=True, cancel_futures=True)
        summary = {
            "event": "generation_summary",
            "generation_backend": "remote_gateway",
            "policy_version": self.policy_version,
            "completed_requests": self.completed,
            "failed_requests": self.failed,
            "cancelled_requests": self.cancelled_requests,
            "sampled_tokens": self.sampled_tokens,
            "max_observed_inflight": self.max_observed_inflight,
            "per_request_latency_sum_seconds": self.latency_sum,
            "latency_scope": "sum of overlapping per-request walls; not batch throughput",
            "prefix_cache": self.prefixes.stats(),
            "collection_wall_seconds": time.perf_counter() - self.started_at,
        }
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            with (self.run_dir / "generation_metrics.jsonl").open(
                    "a", encoding="utf-8") as output:
                output.write(json.dumps(summary) + "\n")
        except OSError:
            pass
        print(json.dumps(summary), flush=True)
