"""Local OpenAI transport for the official CPU harness and an Unsloth policy.

Generation uses the policy's Transformers generate implementation. This initial
transport serializes GPU calls; it does not claim continuous batching or vLLM
throughput. Tool/verification workers can run independently between calls.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class PolicyEndpoint:
    def __init__(self, policy, run_dir: Path, port: int = 0):
        self.policy = policy
        self.run_dir = Path(run_dir)
        self.lock = threading.RLock()
        self.calls: dict[str, list[dict]] = {}
        self.errors: dict[str, list[str]] = {}
        self.active_ids: set[str] = set()
        self.collecting = False
        self.policy_version = 0
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                # Do not log Authorization headers or official request bodies.
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
                    self.respond(200, {"status": "ready", "policy_version": parent.policy_version})
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
                if token not in parent.active_ids:
                    self.respond(401, {"error": "Unknown rollout identity"})
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 16 * 1024 * 1024:
                        raise ValueError("Invalid request size")
                    request = json.loads(self.rfile.read(size))
                    if request.get("n", 1) != 1:
                        raise ValueError("Each official model call must request n=1")
                    for key, expected in (("temperature", 1.0), ("top_p", 1.0), ("top_k", 0),
                                          ("presence_penalty", 0.0), ("frequency_penalty", 0.0)):
                        if (request.get(key, expected) != expected or
                                (request.get("extra_body") or {}).get(key, expected) != expected):
                            raise ValueError(f"This baseline requires {key}={expected}")
                    if request.get("stop") or request.get("logit_bias"):
                        raise ValueError("Custom stop strings/logit_bias are not supported by this baseline")
                    tool_choice = request.get("tool_choice", "auto")
                    if tool_choice not in (None, "auto"):
                        raise ValueError("This baseline requires automatic tool choice")
                    queued_at = time.perf_counter()
                    with parent.lock:
                        if not parent.collecting:
                            raise RuntimeError("Policy is not collecting trajectories")
                        queue_wait = time.perf_counter() - queued_at
                        print(json.dumps({"event": "model_call_started", "rollout_id": token,
                                          "call_index": len(parent.calls[token]),
                                          "queue_wait_seconds": queue_wait}), flush=True)
                        result = parent.generate(request)
                        record = {key: result[key] for key in (
                            "prompt_ids", "completion_ids", "behavior_logprobs", "finish_reason")}
                        for key in ("sampling", "requested_max_tokens", "effective_max_tokens", "context_limited",
                                    "generation_elapsed_seconds", "first_scores_seconds"):
                            if key in result:
                                record[key] = result[key]
                        record.update(rollout_id=token, policy_version=parent.policy_version,
                                      call_index=len(parent.calls[token]), queue_wait_seconds=queue_wait)
                        parent.calls[token].append(record)
                        path = parent.run_dir / token / "model_calls.jsonl"
                        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                        with path.open("a", encoding="utf-8") as out:
                            out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        print(json.dumps({"event": "model_call_finished", "rollout_id": token,
                                          "call_index": record["call_index"],
                                          "prompt_tokens": len(record["prompt_ids"]),
                                          "completion_tokens": len(record["completion_ids"]),
                                          "generation_elapsed_seconds": result.get("generation_elapsed_seconds"),
                                          "first_scores_seconds": result.get("first_scores_seconds")}), flush=True)
                    parent.send_response(self, request, result)
                except Exception as exc:
                    # A transport/parsing/capacity failure invalidates this cohort.
                    message = f"{type(exc).__name__}: {exc}"
                    with parent.lock:
                        parent.errors.setdefault(token, []).append(message)
                    try:
                        self.respond(500, {"error": {"message": message, "type": "local_policy_error"}})
                    except (BrokenPipeError, ConnectionResetError):
                        pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def start(self):
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def begin(self, identities, version):
        with self.lock:
            self.active_ids = set(identities)
            self.calls = {identity: [] for identity in identities}
            self.errors = {identity: [] for identity in identities}
            self.policy_version = version
            self.collecting = True

    def end(self):
        with self.lock:
            self.collecting = False

    def generate(self, request):
        maximum = request.get("max_completion_tokens", request.get("max_tokens"))
        if maximum is None or int(maximum) <= 0:
            raise ValueError("Official harness must supply a positive output-token budget")
        extra = request.get("extra_body") or {}
        thinking_budget = request.get("thinking_token_budget", extra.get("thinking_token_budget"))
        if thinking_budget is not None:
            raise ValueError("Remove thinking_token_budget in the explicit training agent configuration; "
                             "this baseline does not emulate the vLLM thinking-budget processor")
        return self.policy.generate(
            request["messages"], tools=request.get("tools"), max_tokens=int(maximum),
            enable_thinking=request.get("enable_thinking", extra.get("enable_thinking", True)),
        )

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
