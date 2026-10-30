"""Local OpenAI transport with one continuously batched GPU generation worker.

HTTP handlers prepare requests and wait on futures. The condition lock protects
only lifecycle and queue state; model execution, file writes and HTTP waits run
outside it. A collection drains fully before the learner may update its policy.
"""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import Future, InvalidStateError, TimeoutError as FutureTimeout
import json
import math
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class PolicyEndpoint:
    def __init__(self, policy, run_dir: Path, port: int = 0, generation_config=None):
        self.policy = policy
        self.run_dir = Path(run_dir)
        self.generation_config = dict(generation_config or {})
        self.generation_config.setdefault("max_batch_size", 4)
        self.batch_wait_seconds = float(self.generation_config.get("batch_wait_ms", 5)) / 1000
        if not math.isfinite(self.batch_wait_seconds) or not 0 <= self.batch_wait_seconds <= 1:
            raise ValueError("batch_wait_ms must be finite and between 0 and 1000")
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.calls: dict[str, list[dict]] = {}
        self.errors: dict[str, list[str]] = {}
        self.active_ids: set[str] = set()
        self.collecting = False
        self.policy_version = 0
        self._pending = deque()
        self._requests = {}
        self._busy_rollouts = Counter()
        self._next_call_index = {}
        self._record_locks = {}
        self._registered_ids = set()
        self._handler_count = 0
        self._generation_thread = None
        self._generation_ready = threading.Event()
        self._fatal_error = None
        self._closed = False
        self._http_started = False
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                # Never print Authorization headers or official request bodies.
                return

            def respond(self, status, payload):
                data = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    with parent.condition:
                        status = "failed" if parent._fatal_error else "ready"
                        payload = {"status": status, "policy_version": parent.policy_version,
                                   "collecting": parent.collecting, "queued_requests": len(parent._pending)}
                    self.respond(503 if status == "failed" else 200, payload)
                elif self.path == "/v1/models":
                    self.respond(200, {"object": "list", "data": [{
                        "id": "gemma-4-31b-it-qat-w4a16-ct", "object": "model",
                        "owned_by": "local-qat-bf16-learner",
                    }]})
                else:
                    self.respond(404, {"error": "Unknown endpoint"})

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.respond(404, {"error": "Unknown endpoint"})
                    return
                token = self.headers.get("Authorization", "").removeprefix("Bearer ")
                claimed, request_id = False, None
                try:
                    with parent.condition:
                        if token not in parent.active_ids:
                            rejection = (401, "Unknown rollout identity")
                        elif not parent.collecting or parent._fatal_error or parent._closed:
                            rejection = (503, "Policy is not accepting model calls")
                        else:
                            rejection = None
                            parent._busy_rollouts[token] += 1
                            parent._handler_count += 1
                            claimed = True
                    if rejection:
                        self.respond(rejection[0], {"error": rejection[1]})
                        return
                    # A stalled local HTTP client must not hold collection drain
                    # forever. This is a socket I/O limit, not a generation cap.
                    self.connection.settimeout(30)
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 16 * 1024 * 1024:
                        raise ValueError("Invalid request size")
                    request = json.loads(self.rfile.read(size))
                    if not isinstance(request, dict):
                        raise ValueError("The model request must be a JSON object")
                    prepared_at = time.perf_counter()
                    prepared = parent.prepare(request)
                    item = parent.enqueue(token, prepared)
                    item["preparation_elapsed_seconds"] = time.perf_counter() - prepared_at
                    request_id = item["request_id"]
                    while True:
                        try:
                            result = item["future"].result(timeout=0.1)
                            break
                        except FutureTimeout:
                            if parent.policy.abort_generation.is_set():
                                parent._fail(RuntimeError("Generation cancelled by the enclosing run budget"))
                    parent.record_result(item, result)
                    parent.send_response(self, request, result)
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    if claimed:
                        with parent.condition:
                            parent.errors.setdefault(token, []).append(message)
                    try:
                        self.respond(500, {"error": {"message": message, "type": "local_policy_error"}})
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        pass
                finally:
                    if claimed:
                        with parent.condition:
                            if request_id is not None:
                                parent._requests.pop(request_id, None)
                            parent._busy_rollouts[token] -= 1
                            if not parent._busy_rollouts[token]:
                                del parent._busy_rollouts[token]
                            parent._handler_count -= 1
                            parent.condition.notify_all()

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def start(self):
        with self.condition:
            if self._closed or self._http_started:
                raise RuntimeError("Endpoint cannot be started again")
            self._http_started = True
        self.thread.start()

    def close(self):
        with self.condition:
            self._closed = True
            outstanding = bool(self._handler_count or self._pending)
        if outstanding:
            self._fail(RuntimeError("Generation endpoint closed while requests were active"))
        self.end(raise_errors=False)
        if self._http_started:
            self.server.shutdown()
        self.server.server_close()
        if self._http_started:
            self.thread.join()

    def begin(self, identities, version):
        with self.condition:
            if self._closed or self._generation_thread is not None or self._handler_count:
                raise RuntimeError("The preceding collection must be drained before beginning another")
            if self.policy.abort_generation.is_set():
                raise RuntimeError("The enclosing run has already cancelled generation")
            self.calls, self.errors = {}, {}
            self.active_ids, self._registered_ids = set(), set()
            self._pending.clear()
            self._requests.clear()
            self._busy_rollouts.clear()
            self._next_call_index.clear()
            self._record_locks.clear()
            self._fatal_error = None
            self.policy_version = int(version)
            self.collecting = True
            self._generation_ready.clear()
            self._register_locked(identities, version)
            self._generation_thread = threading.Thread(
                target=self._generation_loop, name="gemma-generation", daemon=True,
            )
            worker = self._generation_thread
        worker.start()
        self._generation_ready.wait()
        with self.condition:
            error = self._fatal_error
        if error:
            raise RuntimeError(f"Generation scheduler initialization failed: {error}") from error

    def _register_locked(self, identities, version):
        identities = list(identities)
        if not self.collecting or self._fatal_error or int(version) != self.policy_version:
            raise RuntimeError("Rollouts can only join their active, unchanged policy collection")
        if len(set(identities)) != len(identities) or self._registered_ids.intersection(identities):
            raise ValueError("Rollout identities must be unique within a collection")
        if any(not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identity)
               for identity in identities):
            raise ValueError("Unsafe rollout identity")
        self._registered_ids.update(identities)
        self.active_ids.update(identities)
        for identity in identities:
            self.calls[identity] = []
            self.errors[identity] = []
            self._next_call_index[identity] = 0
            self._record_locks[identity] = threading.Lock()

    def register(self, identities, version):
        with self.condition:
            self._register_locked(identities, version)
            self.condition.notify_all()

    def release_rollouts(self, identities):
        identities = list(identities)
        with self.condition:
            # HTTP response delivery can precede a handler's final bookkeeping
            # by a few instructions. Wait only for these finished rollout IDs.
            while any(self._busy_rollouts.get(identity) for identity in identities):
                self.condition.wait(timeout=0.05)
            for identity in identities:
                self.active_ids.discard(identity)
                self.calls.pop(identity, None)
                self.errors.pop(identity, None)
                self._next_call_index.pop(identity, None)
                self._record_locks.pop(identity, None)

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
            error = self._fatal_error
        if raise_errors and error is not None and not self.policy.abort_generation.is_set():
            raise RuntimeError(f"Generation scheduler failed: {error}") from error

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

    def enqueue(self, identity, prepared):
        queued_at = time.perf_counter()
        with self.condition:
            # Previously claimed handlers may finish preparation during end().
            # Their handler count keeps the worker alive until they have drained.
            if self._fatal_error or self._closed or identity not in self._busy_rollouts:
                raise RuntimeError("The collection no longer accepts this prepared request") from self._fatal_error
            item = {"request_id": uuid.uuid4().hex, "rollout_id": identity,
                    "prepared": prepared, "queued_at": queued_at, "future": Future(),
                    "policy_version": self.policy_version, "call_index": self._next_call_index[identity]}
            self._next_call_index[identity] += 1
            self._requests[item["request_id"]] = item
            self._pending.append(item)
            self.condition.notify_all()
            return item

    @staticmethod
    def _resolve(future, *, result=None, error=None):
        try:
            if error is None:
                future.set_result(result)
            else:
                future.set_exception(error)
        except InvalidStateError:
            # shutdown/abort may have delivered an error while a step finished.
            pass

    def _fail(self, error):
        with self.condition:
            if self._fatal_error is None:
                self._fatal_error = error
            failure = self._fatal_error
            self.collecting = False
            requests = list(self._requests.values())
            self._pending.clear()
            self.condition.notify_all()
        for item in requests:
            self._resolve(item["future"], error=failure)

    def _generation_loop(self):
        engine = None
        steps = completed = admitted = 0
        inflight = set()
        batch_histogram = Counter()
        step_elapsed = 0.0
        try:
            from .generation import GenerationEngine
            engine = GenerationEngine(
                self.policy, {**self.generation_config, "policy_version": self.policy_version}, self.run_dir,
            )
            self._generation_ready.set()
            while True:
                if self.policy.abort_generation.is_set():
                    raise RuntimeError("Generation cancelled by the enclosing run budget")
                with self.condition:
                    if self._fatal_error:
                        raise self._fatal_error
                    if self._closed:
                        raise RuntimeError("Generation endpoint closed")
                    if not self._pending and not engine.active_count:
                        if not self.collecting and not self._handler_count:
                            break
                        self.condition.wait(timeout=0.05)
                        continue
                    # Only an idle engine pays the short initial coalescing wait;
                    # live decoding never waits for another HTTP request to arrive.
                    if (not engine.active_count and self.collecting
                            and len(self._pending) < engine.capacity):
                        remaining = self._pending[0]["queued_at"] + self.batch_wait_seconds - time.perf_counter()
                        if remaining > 0:
                            self.condition.wait(timeout=remaining)
                            continue
                    queued = list(self._pending)
                incoming = []
                selected = []
                for item in queued:
                    if len(incoming) >= engine.capacity:
                        break
                    if engine.can_admit(item["prepared"], incoming=incoming):
                        incoming.append({key: item[key] for key in ("request_id", "prepared", "queued_at")})
                        selected.append(item)
                if not incoming and not engine.active_count and queued:
                    raise RuntimeError("No queued request fits the configured generation admission budget")
                now = time.perf_counter()
                with self.condition:
                    if self._fatal_error:
                        raise self._fatal_error
                    selected_ids = {item["request_id"] for item in selected}
                    self._pending = deque(item for item in self._pending if item["request_id"] not in selected_ids)
                    for item in selected:
                        item["queue_wait_seconds"] = now - item["queued_at"]
                        inflight.add(item["request_id"])
                for item in selected:
                    print(json.dumps({"event": "model_call_started", "rollout_id": item["rollout_id"],
                                      "call_index": item["call_index"],
                                      "stage": "scheduler_admitted", "scheduler_queue_wait_seconds": item["queue_wait_seconds"]}), flush=True)
                batch_width = engine.active_count + len(incoming)
                before = time.perf_counter()
                outputs = engine.step(incoming)
                step_elapsed += time.perf_counter() - before
                steps += 1
                admitted += len(incoming)
                batch_histogram[batch_width] += 1
                for request_id, result in outputs:
                    if request_id not in inflight:
                        raise RuntimeError("Generation returned a duplicate or unadmitted request identity")
                    inflight.remove(request_id)
                    with self.condition:
                        item = self._requests.get(request_id)
                    if item is None:
                        if self._fatal_error:
                            continue
                        raise RuntimeError("Generation returned an unknown request identity")
                    self._resolve(item["future"], result=result)
                    completed += 1
                if engine.active_count != len(inflight):
                    raise RuntimeError("Generation engine lost track of an admitted request")
        except BaseException as exc:
            self._fail(exc)
        finally:
            self._generation_ready.set()
            if engine is not None:
                try:
                    engine.close()
                except BaseException as exc:
                    self._fail(exc)
            # Futures are already resolved before this wait; no GPU work remains.
            # Keep end() from returning while handlers still record a result/error.
            with self.condition:
                while self._handler_count:
                    self.condition.wait(timeout=0.05)
            print(json.dumps({"event": "generation_scheduler_summary", "policy_version": self.policy_version,
                              "steps": steps, "admitted_requests": admitted, "completed_requests": completed,
                              "batch_size_steps": dict(batch_histogram),
                              "host_step_elapsed_seconds": step_elapsed,
                              "failed": self._fatal_error is not None}), flush=True)

    def record_result(self, item, result):
        record = {key: result[key] for key in (
            "prompt_ids", "completion_ids", "behavior_logprobs", "finish_reason")}
        for key in ("sampling", "requested_max_tokens", "effective_max_tokens", "context_limited",
                    "generation_elapsed_seconds", "first_scores_seconds", "first_token_seconds",
                    "reused_prefix_tokens", "profiled_tokens", "generation_backend", "timing_semantics"):
            if key in result:
                record[key] = result[key]
        record.update(rollout_id=item["rollout_id"], policy_version=item["policy_version"],
                      call_index=item["call_index"],
                      queue_wait_seconds=result.get("queue_wait_seconds", item["queue_wait_seconds"]),
                      scheduler_queue_wait_seconds=item["queue_wait_seconds"],
                      preparation_elapsed_seconds=item.get("preparation_elapsed_seconds"))
        with self.condition:
            record_lock = self._record_locks[item["rollout_id"]]
        # Independent rollout files can be written concurrently. Even when two
        # subagents share one rollout, complete JSONL records never interleave.
        with record_lock:
            path = self.run_dir / item["rollout_id"] / "model_calls.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            with self.condition:
                self.calls[item["rollout_id"]].append(record)
                self.calls[item["rollout_id"]].sort(key=lambda row: row["call_index"])
        print(json.dumps({"event": "model_call_finished", "rollout_id": item["rollout_id"],
                          "call_index": item["call_index"], "prompt_tokens": len(record["prompt_ids"]),
                          "completion_tokens": len(record["completion_ids"]),
                          "generation_elapsed_seconds": result.get("generation_elapsed_seconds"),
                          "first_scores_seconds": result.get("first_scores_seconds"),
                          "first_token_seconds": result.get("first_token_seconds"),
                          "generation_backend": result.get("generation_backend"),
                          "timing_semantics": result.get("timing_semantics"),
                          "queue_wait_seconds": record["queue_wait_seconds"],
                          "scheduler_queue_wait_seconds": item["queue_wait_seconds"]}), flush=True)

    @staticmethod
    def send_response(handler, request, result):
        message = result["assistant_message"]
        identifier = "chatcmpl-" + uuid.uuid4().hex
        base = {"id": identifier, "created": int(time.time()), "model": request.get("model", "local-policy")}
        usage = {"prompt_tokens": len(result["prompt_ids"]),
                 "completion_tokens": len(result["completion_ids"])}
        usage["total_tokens"] = sum(usage.values())
        if not request.get("stream", False):
            handler.respond(200, {**base, "object": "chat.completion", "usage": usage,
                                  "choices": [{"index": 0, "message": message,
                                               "finish_reason": result["finish_reason"]}]})
            return
        # LiteLLM/ADK accept a complete delta; the model call remains non-streaming.
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
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
