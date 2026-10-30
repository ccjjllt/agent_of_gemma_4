"""Durable request lifecycle around one continuously batched GPU worker.

Only an explicit consumer ACK admits a generated call to training. HTTP flush,
request retries and GPU completion are separate durable states. The GPU-state
condition is never held during database, file or network I/O.
"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import Future, InvalidStateError, TimeoutError as FutureTimeout
import json
import math
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .request_store import RequestConflict, RequestStore, RequestUnavailable


class RequestCancelled(RequestUnavailable):
    pass


class PolicyEndpoint:
    def __init__(self, policy, run_dir: Path, port: int = 0, generation_config=None):
        self.policy, self.run_dir = policy, Path(run_dir)
        self.generation_config = dict(generation_config or {})
        self.generation_config.setdefault("max_batch_size", 4)
        self.batch_wait_seconds = float(self.generation_config.get("batch_wait_ms", 5)) / 1000
        if not math.isfinite(self.batch_wait_seconds) or not 0 <= self.batch_wait_seconds <= 1:
            raise ValueError("batch_wait_ms must be finite and between 0 and 1000")
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.calls, self.errors = {}, {}
        self.active_ids, self._registered_ids, self._cancelled_rollouts = set(), set(), set()
        self._auth_tokens, self._auth_identities = {}, {}
        self.collecting, self.policy_version = False, 0
        self._pending, self._barriers = deque(), deque()
        self._requests, self._record_locks = {}, {}
        self._cancel_pending = set()
        self._cancelled_keys = set()
        self._busy_rollouts = Counter()
        self._handler_count = 0
        self._generation_thread = None
        self._generation_ready = threading.Event()
        self._fatal_error = None
        self._storage_thread = None
        self._storage_queue = deque()
        self._storage_stop = False
        self._pending_results = 0
        self._storage_seconds = 0.0
        self._storage_actions = Counter()
        self._max_pending_results_observed = 0
        self.max_pending_results = int(self.generation_config.get("max_pending_results", 2 * self.generation_config["max_batch_size"]))
        if self.max_pending_results < 1:
            raise ValueError("max_pending_results must be positive")
        self._server_epoch = uuid.uuid4().hex
        self._timings = {}
        self._timing_sequence = 0
        self._closed = self._http_started = False
        self.store = RequestStore(self.run_dir)
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def respond(self, status, payload, headers=None):
                data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()

            def token(self):
                value = self.headers.get("Authorization", "")
                return value[7:] if value.startswith("Bearer ") else ""

            def read_json(self, *, required=True):
                self.connection.settimeout(30)
                size = int(self.headers.get("Content-Length", "0"))
                valid = 0 < size <= 16 * 1024**2 if required else 0 <= size <= 4096
                if not valid:
                    raise ValueError("Invalid request size")
                value = json.loads(self.rfile.read(size)) if size else {}
                if not isinstance(value, dict):
                    raise ValueError("The request must be a JSON object")
                return value

            def do_GET(self):
                if self.path == "/health":
                    with parent.condition:
                        failed = parent._fatal_error is not None
                        payload = {"status": "failed" if failed else "ready", "policy_version": parent.policy_version,
                                   "collecting": parent.collecting, "queued_requests": len(parent._pending)}
                    self.respond(503 if failed else 200, payload)
                    return
                if self.path == "/v1/models":
                    self.respond(200, {"object": "list", "data": [{"id": "gemma-4-31b-it-qat-w4a16-ct",
                                  "object": "model", "owned_by": "local-qat-bf16-learner"}]})
                    return
                timing = re.fullmatch(r"/v1/rollouts/([A-Za-z0-9][A-Za-z0-9_.-]{0,127})/timing", self.path)
                if timing:
                    claimed = False
                    identity = None
                    try:
                        identity = parent._claim_authenticated(self.token(), generate=False)
                        claimed = True
                        if timing[1] != identity:
                            raise PermissionError("Timing belongs to another rollout")
                        self.respond(200, parent.rollout_timing(identity))
                    except Exception as error:
                        self.error_response(error)
                    finally:
                        if claimed:
                            parent._release_handler(identity)
                    return
                match = re.fullmatch(r"/v1/requests/([a-f0-9]{32})", self.path)
                if not match:
                    self.respond(404, {"error": "Unknown endpoint"})
                    return
                claimed = False
                identity = None
                try:
                    identity = parent._claim_authenticated(self.token(), generate=False)
                    claimed = True
                    row = parent.store.status(match[1], identity, parent.policy_version)
                    self.respond(200, {"request_id": row["request_id"], "state": row["state"],
                                       "call_index": row["call_index"], "policy_version": row["policy_version"]})
                except Exception as error:
                    self.error_response(error)
                finally:
                    if claimed:
                        parent._release_handler(identity)

            def error_response(self, error):
                status = (401 if isinstance(error, PermissionError) else 404 if isinstance(error, KeyError)
                          else 409 if isinstance(error, RequestConflict) else 410 if isinstance(error, RequestCancelled)
                          else 400 if isinstance(error, (ValueError, TypeError)) else 503 if parent.fatal_error else 500)
                try:
                    self.respond(status, {"error": {"message": f"{type(error).__name__}: {error}",
                                                    "type": "local_policy_error"}})
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass

            def do_POST(self):
                identity, claimed = None, False
                match = re.fullmatch(r"/v1/requests/([a-f0-9]{32})/(ack|cancel)", self.path)
                key_cancel = re.fullmatch(r"/v1/requests/by-key/([A-Za-z0-9_.:-]{1,200})/cancel", self.path)
                if self.path != "/v1/chat/completions" and not match and not key_cancel:
                    self.respond(404, {"error": "Unknown endpoint"})
                    return
                try:
                    identity = parent._claim_authenticated(self.token(), generate=match is None and key_cancel is None)
                    claimed = True
                    request = self.read_json(required=match is None and key_cancel is None)
                    if key_cancel:
                        self.respond(200, parent.cancel_by_key(identity, key_cancel[1]))
                        return
                    if match:
                        if match[2] == "ack":
                            row = parent.acknowledge(match[1], identity)
                        else:
                            row = parent.cancel_request(match[1], identity, reason="client_cancelled")
                        self.respond(200, {"request_id": row["request_id"], "state": row["state"]})
                        return
                    started = time.perf_counter()
                    prepared = parent.prepare(request)
                    item = parent.enqueue(identity, prepared, request=request,
                                          idempotency_key=self.headers.get("Idempotency-Key"),
                                          preparation_elapsed_seconds=time.perf_counter() - started)
                    while True:
                        try:
                            item["future"].result(timeout=0.1)
                            break
                        except FutureTimeout:
                            if parent.policy.abort_generation.is_set():
                                parent._fail(RuntimeError("Run cancelled while waiting for generation"))
                    row = parent.store.get(item["request_id"], identity, parent.policy_version)
                    if row["state"] in {"cancelled", "failed"}:
                        raise parent._request_exception(row)
                    try:
                        parent.send_response(self, request, row["result"],
                                             request_id=row["request_id"], created_at=row["created_at"])
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        parent.store.delivered(row["request_id"], success=False)
                        return  # The durable response remains replayable, not consumed.
                    parent.store.delivered(row["request_id"], success=True)
                except Exception as error:
                    self.error_response(error)
                finally:
                    if claimed:
                        parent._release_handler(identity)

        try:
            self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except BaseException:
            self.store.close()
            raise
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    @property
    def fatal_error(self):
        with self.condition:
            return self._fatal_error

    def _authenticate(self, token):
        """Resolve a private credential; public rollout names are never keys."""
        with self.condition:
            identity = self._auth_identities.get(token)
            if identity is None or self._closed:
                raise PermissionError("Unknown or expired rollout credential")
            return identity

    def rollout_token(self, identity):
        """Trusted collector access only; never serialize this into run logs."""
        with self.condition:
            if identity not in self.active_ids or identity in self._cancelled_rollouts:
                raise PermissionError("Rollout no longer has an active credential")
            return self._auth_tokens[identity]

    def _claim_authenticated(self, token, *, generate):
        # Resolve and register the handler under one lock, so release/end
        # cannot revoke a credential between authentication and accounting.
        with self.condition:
            identity = self._authenticate(token)
            self._claim(identity, generate=generate)
            return identity

    def _revoke_token_locked(self, identity):
        token = self._auth_tokens.pop(identity, None)
        if token is not None:
            self._auth_identities.pop(token, None)

    def _timing_change_locked(self, item, phase, now=None):
        """Track queue-only wall intervals, never credit overlapping service."""
        timing = self._timings[item["rollout_id"]]
        now = max(time.perf_counter() if now is None else float(now), timing["last_clock"])
        timing["last_clock"] = now
        request_id = item["request_id"]
        timing["queued"].discard(request_id)
        timing["serving"].discard(request_id)
        if phase == "queued":
            timing["queued"].add(request_id)
        elif phase == "serving":
            timing["serving"].add(request_id)
        eligible = bool(timing["queued"] and not timing["serving"])
        if eligible and timing["open_start"] is None:
            timing["open_start"] = now
        elif not eligible and timing["open_start"] is not None:
            start = timing["open_start"]
            if now > start:
                timing["intervals"].append({"start": start, "end": now})
                timing["closed_seconds"] += now - start
            timing["open_start"] = None
        item["timing_phase"] = phase

    def rollout_timing(self, identity):
        """Small authenticated protocol; no token scans, database locks or I/O."""
        with self.condition:
            if identity not in self._registered_ids:
                raise PermissionError("Unknown rollout identity")
            timing = self._timings[identity]
            now = max(time.perf_counter(), timing["last_clock"])
            timing["last_clock"] = now
            intervals = [dict(item) for item in timing["intervals"]]
            total = timing["closed_seconds"]
            if timing["open_start"] is not None:
                intervals.append({"start": timing["open_start"], "end": now})
                total += now - timing["open_start"]
            self._timing_sequence += 1
            return {"protocol": "gemma.queue_timing.v1", "rollout_id": identity,
                    "policy_version": self.policy_version, "server_epoch": self._server_epoch,
                    "snapshot_seq": self._timing_sequence, "snapshot_monotonic": now,
                    "queue_excluded_seconds": total, "eligible_intervals": intervals,
                    "queued_requests": len(timing["queued"]), "serving_requests": len(timing["serving"]),
                    "semantics": "union of this rollout's queued intervals minus its serving intervals"}

    def _request_started(self, request_id, started_at):
        """Generation callback: only in-memory work on the GPU owner thread."""
        with self.condition:
            item = self._requests[request_id]
            if item.get("timing_phase") == "finished":
                return  # Cancellation can race a prefill step boundary.
            self._timing_change_locked(item, "serving", started_at)
            waiting = max(0.0, float(started_at) - item["queued_at"])
            self._storage_queue.append(("started", (item, waiting)))
            self.condition.notify_all()

    def _queue_storage(self, action, *arguments):
        with self.condition:
            if self._storage_stop:
                raise RequestUnavailable("Request persistence worker is no longer accepting work")
            self._storage_queue.append((action, arguments))
            if action in {"generated", "failed"}:
                self._pending_results += 1
                self._max_pending_results_observed = max(self._max_pending_results_observed, self._pending_results)
            self.condition.notify_all()

    def _storage_loop(self):
        """Keep fsync, JSON encoding and stdout outside the GPU owner thread.

        A completed response is made visible only after its FULL SQLite commit.
        The bounded admission watermark includes active and pending responses.
        """
        while True:
            with self.condition:
                while not self._storage_queue and not self._storage_stop:
                    self.condition.wait(timeout=0.1)
                if not self._storage_queue:
                    return
                action, arguments = self._storage_queue.popleft()
            item = arguments[0]
            started = time.perf_counter()
            try:
                if action == "started":
                    row = self.store.generating(item["request_id"], {"scheduler_queue_wait_seconds": arguments[1]})
                    if row["state"] == "generating":
                        print(json.dumps({"event": "model_call_started", "rollout_id": item["rollout_id"],
                                          "request_id": item["request_id"], "call_index": item["call_index"]}), flush=True)
                elif action == "generated":
                    result = arguments[1]
                    row = self.store.generated(item["request_id"], result)
                    if row["state"] in {"cancelled", "failed"}:
                        self._resolve(item["future"], error=self._request_exception(row))
                    elif self.fatal_error is not None:
                        self._resolve(item["future"], error=self.fatal_error)
                    else:
                        self._resolve(item["future"], result=result)
                    print(json.dumps({"event": "model_call_generated", "request_id": item["request_id"],
                                      "rollout_id": item["rollout_id"], "call_index": item["call_index"],
                                      "completion_tokens": len(result["completion_ids"]), "delivery_state": row["state"]}), flush=True)
                elif action == "failed":
                    error = arguments[1]
                    row = self.store.fail(item["request_id"], f"{type(error).__name__}: {error}")
                    with self.condition:
                        self.errors.setdefault(item["rollout_id"], []).append(row["error"])
                    self._resolve(item["future"], error=error)
                else:
                    raise RuntimeError("Unknown persistence action")
            except BaseException as error:
                # Disk failure still aborts collection, but the GPU owner never
                # waits for another failed disk operation to signal callers.
                self._fail(error, persist=False)
                with self.condition:
                    self._storage_queue.clear()
                    self._storage_stop = True
                    self._pending_results = 0
                    self.condition.notify_all()
                return
            finally:
                with self.condition:
                    self._storage_seconds += max(0.0, time.perf_counter() - started)
                    self._storage_actions[action] += 1
                    if action in {"generated", "failed"}:
                        self._pending_results = max(0, self._pending_results - 1)
                    self.condition.notify_all()

    def _claim(self, identity, *, generate):
        with self.condition:
            allowed = self.active_ids if generate else self._registered_ids
            if identity not in allowed:
                raise PermissionError("Unknown rollout identity")
            if self._closed or (generate and (not self.collecting or self._fatal_error
                                              or identity in self._cancelled_rollouts)):
                raise RequestUnavailable("Policy is not accepting model calls for this rollout")
            self._busy_rollouts[identity] += 1
            self._handler_count += 1

    def _release_handler(self, identity):
        with self.condition:
            self._busy_rollouts[identity] -= 1
            if not self._busy_rollouts[identity]:
                del self._busy_rollouts[identity]
            self._handler_count -= 1
            self.condition.notify_all()

    def start(self):
        with self.condition:
            if self._closed or self._http_started:
                raise RuntimeError("Endpoint cannot be started again")
            self._http_started = True
        self.thread.start()

    def close(self):
        with self.condition:
            if self._closed:
                return
            outstanding = bool(self._handler_count or self._pending)
        if outstanding:
            self._fail(RuntimeError("Generation endpoint closed while requests were active"))
        self.end(raise_errors=False)
        with self.condition:
            self._closed = True
            self._auth_tokens.clear()
            self._auth_identities.clear()
        if self._http_started:
            self.server.shutdown()
            self.thread.join()
        self.server.server_close()
        self.store.close()

    def begin(self, identities, version):
        with self.condition:
            if self._closed or self._generation_thread is not None or self._storage_thread is not None or self._handler_count:
                raise RuntimeError("The preceding collection must be drained first")
            if self.policy.abort_generation.is_set():
                raise RuntimeError("The enclosing run has cancelled generation")
            self.calls, self.errors = {}, {}
            self.active_ids, self._registered_ids, self._cancelled_rollouts = set(), set(), set()
            self._auth_tokens.clear()
            self._auth_identities.clear()
            self._pending.clear()
            self._requests.clear()
            self._barriers.clear()
            self._cancel_pending.clear()
            self._cancelled_keys.clear()
            self._busy_rollouts.clear()
            self._record_locks.clear()
            self._timings.clear()
            self._storage_queue.clear()
            self._storage_stop = False
            self._pending_results = 0
            self._storage_seconds = 0.0
            self._storage_actions.clear()
            self._max_pending_results_observed = 0
            self._fatal_error = None
            self.policy_version = int(version)
            self.collecting = True
            self._generation_ready.clear()
        self.register(identities, version)
        with self.condition:
            storage = threading.Thread(target=self._storage_loop, name="gemma-request-persistence", daemon=True)
            self._storage_thread = storage
            worker = threading.Thread(target=self._generation_loop, name="gemma-generation", daemon=True)
            self._generation_thread = worker
        storage.start()
        worker.start()
        self._generation_ready.wait()
        if self.fatal_error:
            raise RuntimeError(f"Generation scheduler initialization failed: {self.fatal_error}") from self.fatal_error

    def register(self, identities, version):
        identities = list(identities)
        with self.condition:
            if not self.collecting or self._fatal_error or int(version) != self.policy_version:
                raise RuntimeError("Rollouts can only join their unchanged policy collection")
            if len(set(identities)) != len(identities) or self._registered_ids.intersection(identities):
                raise ValueError("Rollout identities must be unique within a collection")
            if any(not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identity)
                   for identity in identities):
                raise ValueError("Unsafe rollout identity")
            self._registered_ids.update(identities)
            for identity in identities:
                token = secrets.token_urlsafe(32)
                while token == identity or token in self._auth_identities:
                    token = secrets.token_urlsafe(32)
                self._auth_tokens[identity] = token
                self._auth_identities[token] = identity
                self.calls[identity], self.errors[identity] = [], []
                self._record_locks[identity] = threading.Lock()
                self._timings[identity] = {"queued": set(), "serving": set(), "intervals": [],
                                          "open_start": None, "closed_seconds": 0.0,
                                          "last_clock": time.perf_counter()}
        # Loading durable receipts is disk work, outside the GPU-state lock.
        for identity in identities:
            self.recover_rollout_calls(identity, int(version))
        with self.condition:
            self.active_ids.update(identities)
            self.condition.notify_all()

    def release_rollouts(self, identities):
        identities = list(identities)
        self.cancel_rollouts(identities, reason="rollout_released_unconsumed")
        with self.condition:
            while any(self._busy_rollouts.get(identity) for identity in identities):
                self.condition.wait(timeout=0.05)
        # Cancellation is nonblocking, but memory release waits for the owner
        # to drop any abandoned active row before deleting its bookkeeping.
        if self._generation_thread is not None and self.fatal_error is None:
            self.generation_barrier(lambda: None)
        with self.condition:
            for identity in identities:
                self.active_ids.discard(identity)
                self.calls.pop(identity, None)
                self.errors.pop(identity, None)
                self._record_locks.pop(identity, None)
            if self._fatal_error is None:
                self._requests = {key: item for key, item in self._requests.items()
                                  if item["rollout_id"] not in identities}

    def end(self, *, raise_errors=True):
        with self.condition:
            self.collecting = False
            worker = self._generation_thread
            self.condition.notify_all()
        if worker is None:
            return
        if worker is threading.current_thread():
            raise RuntimeError("A generation worker cannot join itself")
        worker.join()
        with self.condition:
            if self._generation_thread is worker:
                self._generation_thread = None
            self.active_ids.clear()
            self._auth_tokens.clear()
            self._auth_identities.clear()
            self._requests.clear()
            self._pending.clear()
            self._storage_thread = None
            error = self._fatal_error
        if raise_errors and error is not None and not self.policy.abort_generation.is_set():
            raise RuntimeError(f"Generation scheduler failed: {error}") from error

    def generation_barrier(self, callback):
        """Run a short snapshot callback between GPU steps on the owning thread."""
        with self.condition:
            if self._fatal_error:
                raise RuntimeError("Cannot snapshot a failed generation worker") from self._fatal_error
            worker = self._generation_thread
            if worker is None:
                direct = True
            elif worker is threading.current_thread():
                direct = True
            else:
                direct = False
                future = Future()
                self._barriers.append((callback, future))
                self.condition.notify_all()
        if direct:
            return callback()
        while True:
            try:
                return future.result(timeout=0.1)
            except FutureTimeout:
                if self.policy.abort_generation.is_set():
                    raise RuntimeError("Run cancelled before generation barrier")
                if not worker.is_alive():
                    raise RuntimeError("Generation worker stopped before barrier completion") from self.fatal_error

    def prepare(self, request):
        if request.get("n", 1) != 1:
            raise ValueError("Each official model call must request n=1")
        extra = request.get("extra_body") or {}
        if not isinstance(extra, dict):
            raise ValueError("extra_body must be an object")
        for key, expected in (("temperature", 1.0), ("top_p", 1.0), ("top_k", 0),
                              ("presence_penalty", 0.0), ("frequency_penalty", 0.0)):
            if request.get(key, expected) != expected or extra.get(key, expected) != expected:
                raise ValueError(f"This baseline requires {key}={expected}")
        if request.get("stop") or request.get("logit_bias"):
            raise ValueError("Custom stop strings/logit_bias are not supported by this baseline")
        if request.get("tool_choice", "auto") not in (None, "auto"):
            raise ValueError("This baseline requires automatic tool choice")
        maximum = request.get("max_completion_tokens", request.get("max_tokens"))
        if maximum is None or int(maximum) <= 0:
            raise ValueError("Official harness must supply a positive output-token budget")
        if request.get("thinking_token_budget", extra.get("thinking_token_budget")) is not None:
            raise ValueError("Remove thinking_token_budget in the explicit training agent configuration; "
                             "this baseline does not emulate the vLLM thinking-budget processor")
        return self.policy.prepare_generation(
            request["messages"], tools=request.get("tools"), max_tokens=int(maximum),
            enable_thinking=request.get("enable_thinking", extra.get("enable_thinking", True)),
        )

    @staticmethod
    def _request_exception(row):
        error = row.get("error") or "Request is unavailable"
        return RequestCancelled(error) if row["state"] == "cancelled" else RequestUnavailable(error)

    @staticmethod
    def _resolve(future, *, result=None, error=None):
        try:
            future.set_exception(error) if error is not None else future.set_result(result)
        except InvalidStateError:
            pass

    def enqueue(self, identity, prepared, *, request=None, idempotency_key=None, preparation_elapsed_seconds=None):
        with self.condition:
            if (self._fatal_error or self._closed or identity not in self._busy_rollouts
                    or identity in self._cancelled_rollouts):
                raise RequestUnavailable("The collection no longer accepts this request")
            version = self.policy_version
        request_hash = self.store.request_hash(request if request is not None else prepared)
        row, _created = self.store.create(identity, version, idempotency_key, request_hash,
                                         {"preparation_elapsed_seconds": preparation_elapsed_seconds})
        with self.condition:
            cancelled = (identity in self._cancelled_rollouts or self._fatal_error or self._closed
                         or (identity, version, idempotency_key) in self._cancelled_keys)
            existing = self._requests.get(row["request_id"])
            if existing is not None:
                return existing
            item = {"request_id": row["request_id"], "rollout_id": identity, "policy_version": version,
                    "call_index": row["call_index"], "prepared": prepared,
                    "idempotency_key": row["idempotency_key"],
                    "queued_at": time.perf_counter(), "future": Future()}
            self._requests[item["request_id"]] = item
            if not cancelled and row["state"] == "queued":
                self._timing_change_locked(item, "queued", item["queued_at"])
                self._pending.append(item)
                self.condition.notify_all()
        if cancelled:
            row = self.store.cancel(row["request_id"], identity, "rollout_cancelled_during_preparation")
        if row["state"] in {"generated", "delivered", "consumed"}:
            self._resolve(item["future"], result=row["result"])
        elif row["state"] != "queued":
            self._resolve(item["future"], error=self._request_exception(row))
        return item

    def cancel_request(self, request_id, identity=None, *, reason="cancelled"):
        with self.condition:
            known = self._requests.get(request_id)
            if known is not None and identity is not None and known["rollout_id"] != identity:
                raise KeyError("Unknown request identity")
        if known is None and identity is not None:
            self.store.status(request_id, identity, self.policy_version)
        # Signal the owner before doing durable I/O. A cancelled request cannot
        # keep consuming GPU steps merely because another fsync is slow.
        with self.condition:
            item = self._requests.get(request_id)
            self._pending = deque(candidate for candidate in self._pending if candidate["request_id"] != request_id)
            self._cancel_pending.add(request_id)
            if item is not None:
                self._timing_change_locked(item, "finished")
            self.condition.notify_all()
        try:
            row = self.store.cancel(request_id, identity, reason)
        except BaseException as error:
            # The owner may already have removed this request. Do not strand
            # its HTTP waiter if durable cancellation cannot be committed.
            self._fail(error, persist=False)
            raise
        if row["state"] == "cancelled":
            with self.condition:
                self._pending = deque(item for item in self._pending if item["request_id"] != request_id)
                self._cancel_pending.add(request_id)
                item = self._requests.get(request_id)
                self.condition.notify_all()
            if item is not None:
                self._resolve(item["future"], error=RequestCancelled(reason))
        return row

    def cancel_by_key(self, identity, key, *, reason="client_cancelled_by_key"):
        """Cancel immediately by the client's logical identity, before an ID exists."""
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", key):
            raise ValueError("Invalid idempotency key")
        with self.condition:
            if identity not in self._registered_ids:
                raise PermissionError("Unknown rollout identity")
            version = self.policy_version
            scope = (identity, version, key)
            # The in-memory tombstone closes the race with an HTTP handler
            # currently preparing or durably creating this same request.
            self._cancelled_keys.add(scope)
            matches = [item for item in self._requests.values()
                       if (item["rollout_id"], item["policy_version"], item.get("idempotency_key")) == scope]
            request_ids = {item["request_id"] for item in matches}
            self._pending = deque(item for item in self._pending if item["request_id"] not in request_ids)
            self._cancel_pending.update(request_ids)
            for item in matches:
                self._timing_change_locked(item, "finished")
            self.condition.notify_all()
        # No database or fsync lies before the owner cancellation signal.
        try:
            receipt = self.store.cancel_by_key(identity, version, key, reason)
        except BaseException as error:
            self._fail(error, persist=False)
            raise
        with self.condition:
            if receipt["state"] == "consumed":
                self._cancelled_keys.discard(scope)
            # A request may have finished create() while the durable tombstone
            # was being committed. Its own enqueue sees the in-memory key too.
            item = self._requests.get(receipt["request_id"])
        if item is not None and receipt["state"] in {"cancelled", "failed"}:
            error = RequestCancelled(reason) if receipt["state"] == "cancelled" else RequestUnavailable("Request already failed")
            self._resolve(item["future"], error=error)
        return receipt

    def cancel_rollouts(self, identities, *, reason="rollout_cancelled"):
        identities = list(identities)
        with self.condition:
            self._cancelled_rollouts.update(identities)
            self.active_ids.difference_update(identities)
            for identity in identities:
                self._revoke_token_locked(identity)
            version = self.policy_version
            selected = [item for item in self._requests.values() if item["rollout_id"] in identities]
            selected_ids = {item["request_id"] for item in selected}
            self._pending = deque(item for item in self._pending if item["request_id"] not in selected_ids)
            self._cancel_pending.update(selected_ids)
            for item in selected:
                self._timing_change_locked(item, "finished")
            self.condition.notify_all()
        for identity in identities:
            for row in self.store.rollout_states(identity, version):
                if row["state"] not in {"consumed", "failed", "cancelled"}:
                    self.cancel_request(row["request_id"], identity, reason=reason)

    def rollout_delivery_status(self, identity, version=None):
        version = self.policy_version if version is None else version
        rows = self.store.rollout_states(identity, version)
        return {"rollout_id": identity, "policy_version": version,
                "states": dict(Counter(row["state"] for row in rows)),
                "consumed_request_ids": [row["request_id"] for row in rows if row["state"] == "consumed"],
                "unconsumed_generated_requests": sum(row["has_result"] and row["state"] != "consumed" for row in rows)}

    def _write_calls_snapshot(self, identity, records):
        path = self.run_dir / identity / "model_calls.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=".consumed-calls-", delete=False) as output:
                temporary = Path(output.name)
                for record in records:
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def recover_rollout_calls(self, identity, version):
        with self.condition:
            record_lock = self._record_locks.get(identity, threading.Lock())
        with record_lock:
            records = self.store.consumed_calls(identity, version)
            if records or (self.run_dir / identity / "model_calls.jsonl").exists():
                self._write_calls_snapshot(identity, records)
            with self.condition:
                if identity in self.calls:
                    self.calls[identity] = records
        return records

    def acknowledge(self, request_id, identity):
        with self.condition:
            record_lock = self._record_locks.get(identity)
            version = self.policy_version
        if record_lock is None:
            raise PermissionError("Rollout no longer accepts acknowledgements")
        with record_lock:
            row, changed = self.store.acknowledge(request_id, identity, version)
            record = self.store.call_record(row)
            if changed:
                path = self.run_dir / identity / "model_calls.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                with self.condition:
                    records = list(self.calls.get(identity, []))
                records.append(record)
                records.sort(key=lambda item: item["call_index"])
            else:
                # Repairs a crash/write failure between the durable ACK and its
                # derived JSONL export. Idempotent ACKs cannot duplicate calls.
                records = self.store.consumed_calls(identity, version)
                self._write_calls_snapshot(identity, records)
            with self.condition:
                if identity in self.calls:
                    self.calls[identity] = records
        print(json.dumps({"event": "model_call_consumed", "rollout_id": identity,
                          "request_id": request_id, "call_index": row["call_index"],
                          "new_acknowledgement": changed}), flush=True)
        return row

    def _fail(self, error, *, persist=True):
        with self.condition:
            first = self._fatal_error is None
            if self._fatal_error is None:
                self._fatal_error = error
            failure = self._fatal_error
            self.collecting = False
            requests = list(self._requests.values())
            barriers = list(self._barriers)
            self._barriers.clear()
            self._pending.clear()
            for item in requests:
                self._timing_change_locked(item, "finished")
                if first and persist and self._storage_thread is not None and not self._storage_stop:
                    self._storage_queue.append(("failed", (item, failure)))
                    self._pending_results += 1
            self.condition.notify_all()
        for item in requests:
            # Disk failure cannot strand waiters or block the owner loop.
            self._resolve(item["future"], error=failure)
        for _callback, future in barriers:
            self._resolve(future, error=failure)

    def _generation_loop(self):
        engine = None
        steps = completed = admitted = 0
        inflight = set()
        try:
            from .generation import GenerationEngine
            engine = GenerationEngine(self.policy, {**self.generation_config, "policy_version": self.policy_version},
                                      self.run_dir, on_request_started=self._request_started)
            self._generation_ready.set()
            while True:
                if self.policy.abort_generation.is_set():
                    raise RuntimeError("Generation cancelled by the enclosing run budget")
                with self.condition:
                    if self._fatal_error:
                        raise self._fatal_error
                    cancelled = set(self._cancel_pending)
                    self._cancel_pending.clear()
                    barriers = list(self._barriers)
                    self._barriers.clear()
                if cancelled:
                    removed = engine.cancel(cancelled)
                    inflight.difference_update(removed)
                for callback, future in barriers:
                    try:
                        self._resolve(future, result=callback())
                    except BaseException as error:
                        self._resolve(future, error=error)
                with self.condition:
                    if not self._pending and not engine.active_count:
                        if not self.collecting and not self._handler_count:
                            break
                        self.condition.wait(timeout=0.05)
                        continue
                    if not engine.active_count and self.collecting and len(self._pending) < engine.capacity:
                        remaining = self._pending[0]["queued_at"] + self.batch_wait_seconds - time.perf_counter()
                        if remaining > 0:
                            self.condition.wait(timeout=remaining)
                            continue
                    queued = list(self._pending)
                    persistence_capacity = self.max_pending_results - self._pending_results - len(inflight)
                    if persistence_capacity <= 0 and not engine.active_count:
                        self.condition.wait(timeout=0.05)
                        continue
                incoming, selected = [], []
                for item in queued:
                    if len(incoming) >= min(engine.capacity, persistence_capacity):
                        break
                    if engine.can_admit(item["prepared"], incoming=incoming):
                        incoming.append({key: item[key] for key in ("request_id", "prepared", "queued_at")})
                        selected.append(item)
                with self.condition:
                    # A cancellation can arrive while capacity is being checked.
                    selected = [item for item in selected if item.get("timing_phase") == "queued"
                                and item["rollout_id"] not in self._cancelled_rollouts]
                    incoming = [{key: item[key] for key in ("request_id", "prepared", "queued_at")} for item in selected]
                    selected_ids = {item["request_id"] for item in selected}
                    self._pending = deque(item for item in self._pending if item["request_id"] not in selected_ids)
                    inflight.update(selected_ids)
                if not incoming and not engine.active_count:
                    if queued:
                        for item in queued:
                            with self.condition:
                                if item.get("timing_phase") != "queued":
                                    continue
                                self._timing_change_locked(item, "finished")
                            self._queue_storage("failed", item, RequestUnavailable("request_cannot_fit_generation_budget"))
                        with self.condition:
                            rejected = {item["request_id"] for item in queued}
                            self._pending = deque(item for item in self._pending if item["request_id"] not in rejected)
                    continue
                outputs = engine.step(incoming)
                steps += 1
                admitted += len(incoming)
                for request_id, result in outputs:
                    if request_id not in inflight:
                        raise RuntimeError("Generation returned a duplicate or unadmitted identity")
                    inflight.remove(request_id)
                    with self.condition:
                        item = self._requests[request_id]
                        self._timing_change_locked(item, "finished")
                    if isinstance(result, Exception):
                        self._queue_storage("failed", item, result)
                    else:
                        self._queue_storage("generated", item, result)
                        completed += 1
                if engine.active_count != len(inflight):
                    raise RuntimeError("Generation engine lost an admitted request")
        except BaseException as error:
            self._fail(error)
        finally:
            self._generation_ready.set()
            if engine is not None:
                try:
                    engine.close()
                except BaseException as error:
                    self._fail(error)
            with self.condition:
                self._storage_stop = True
                self.condition.notify_all()
                storage = self._storage_thread
            if storage is not None and storage is not threading.current_thread():
                storage.join()
            with self.condition:
                barriers = list(self._barriers)
                self._barriers.clear()
            for _callback, future in barriers:
                self._resolve(future, error=RuntimeError("Generation worker ended before snapshot"))
            with self.condition:
                while self._handler_count:
                    self.condition.wait(timeout=0.05)
            print(json.dumps({"event": "generation_scheduler_summary", "policy_version": self.policy_version,
                              "steps": steps, "admitted_requests": admitted, "generated_requests": completed,
                              "persistence_seconds": self._storage_seconds,
                              "persistence_actions": dict(self._storage_actions),
                              "max_pending_results": self._max_pending_results_observed,
                              "failed": self._fatal_error is not None}), flush=True)

    @staticmethod
    def send_response(handler, request, result, *, request_id, created_at):
        message = result["assistant_message"]
        base = {"id": "chatcmpl-" + request_id, "created": int(created_at), "model": request.get("model", "local-policy")}
        usage = {"prompt_tokens": len(result["prompt_ids"]), "completion_tokens": len(result["completion_ids"])}
        usage["total_tokens"] = sum(usage.values())
        if not request.get("stream", False):
            handler.respond(200, {**base, "object": "chat.completion", "usage": usage,
                                  "choices": [{"index": 0, "message": message,
                                               "finish_reason": result["finish_reason"]}]},
                            headers={"X-Request-ID": request_id})
            return
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("X-Request-ID", request_id)
        handler.end_headers()
        delta = dict(message)
        if "tool_calls" in delta:
            delta["tool_calls"] = [dict(item, index=i) for i, item in enumerate(delta["tool_calls"])]
        for choices, usage_value in (([{"index": 0, "delta": delta, "finish_reason": None}], None),
                                     ([{"index": 0, "delta": {}, "finish_reason": result["finish_reason"]}], usage)):
            chunk = {**base, "object": "chat.completion.chunk", "choices": choices}
            if usage_value is not None:
                chunk["usage"] = usage_value
            handler.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode())
            handler.wfile.flush()
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()
