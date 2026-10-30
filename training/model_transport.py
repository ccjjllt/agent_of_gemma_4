"""Idempotent model transport and explicit ADK consumption acknowledgements.

An HTTP write or a parsed LiteLLM response is not agent consumption. The ACK is
sent only after ADK has parsed a final response, yielded it to its caller, and
that caller has resumed the generator through normal exhaustion. Stream calls
also require an explicit finish frame and normal SDK stream exhaustion.

This hook targets the pinned ADK/LiteLLM runtime. A logical invocation gets a
fresh UUID; SDK HTTP retries keep that same header. A distinct ADK invocation,
even with identical text, is a distinct request. Receipts contain no prompts,
token IDs, tool arguments, authorization values or response text.
"""
from __future__ import annotations

import asyncio
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
import functools
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
import uuid


class ConsumptionConfirmationError(RuntimeError):
    """Non-retryable at the model layer: retry ACK, never re-execute a response."""


@dataclass
class _Invocation:
    key: str
    streaming: bool
    request_id: str | None = None
    transport_calls: int = 0
    terminal_seen: bool = False
    raw_exhausted: bool = False
    wire_exhausted: bool = False
    final_yielded: int = 0
    final_resumed: int = 0
    acknowledged: bool = False
    completed: bool = False
    failed: bool = False
    cancellation_attempted: bool = False
    cancellation_confirmed: bool = False
    stream: object = None
    raw_stream: object = None


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


class _TrackedStream:
    def __init__(self, source, tracker, invocation, *, raw=False):
        self._source = source
        self._iterator = source.__aiter__()
        self._tracker, self._invocation = tracker, invocation
        self._raw = raw

    def __getattr__(self, name):
        return getattr(self._source, name)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            with self._tracker.model_wait():
                part = await self._iterator.__anext__()
        except StopAsyncIteration:
            await self._tracker.refresh_after_wait()
            if self._raw:
                self._invocation.raw_exhausted = True
            else:
                self._invocation.wire_exhausted = True
            raise
        self._tracker._observe_wire(self._invocation, part, terminal=self._raw)
        return part

    async def aclose(self):
        close = getattr(self._source, "aclose", None) or getattr(self._source, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result


class DeliveryTracker:
    def __init__(self, endpoint, rollout_id, receipt_path, *, ack_attempts=3, ack_timeout=5.0,
                 session_clock=None, api_key=None):
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in {"http", "https"} or not parsed.netloc
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path.rstrip("/") != "/v1"):
            raise ValueError("Delivery tracking requires the local OpenAI endpoint ending in /v1")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", rollout_id):
            raise ValueError("Invalid rollout identity")
        if (not isinstance(api_key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", api_key)
                or api_key == rollout_id):
            raise ValueError("Delivery tracking requires a private rollout API key")
        if (type(ack_attempts) is not int or ack_attempts < 1
                or not math.isfinite(ack_timeout) or ack_timeout <= 0):
            raise ValueError("ACK attempts and timeout must be positive")
        self.endpoint = endpoint.rstrip("/")
        self.rollout_id = rollout_id
        self.api_key = api_key
        self.receipt_path = Path(receipt_path)
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.ack_attempts, self.ack_timeout = ack_attempts, float(ack_timeout)
        self.session_clock = session_clock
        self.invocations = []
        self.protocol_errors = 0
        self._context = ContextVar("gemma_delivery_invocation", default=None)
        self._restore = None

    def model_wait(self):
        return self.session_clock.model_wait() if self.session_clock is not None else nullcontext()

    async def refresh_after_wait(self):
        if self.session_clock is None:
            return
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            return  # Final runtime accounting checks this cancellation boundary.
        try:
            await self.session_clock.refresh()
        except Exception:
            if self.session_clock.mode == "queue_excluded":
                raise

    def _receipt(self, event, invocation):
        row = {"event": event, "time_unix": time.time(), "idempotency_key": invocation.key,
               "request_id": invocation.request_id, "streaming": invocation.streaming,
               "final_yielded": invocation.final_yielded, "final_resumed": invocation.final_resumed}
        # A consumer-received receipt is durable before the network ACK. An
        # interrupted ACK never becomes an invented consumed server state.
        try:
            descriptor = os.open(self.receipt_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(json.dumps(row) + "\n")
                output.flush()
                os.fsync(output.fileno())
        except OSError as error:
            # Official model retry treats OSError as a reason to regenerate.
            # A receipt failure after final yield must never trigger that retry.
            raise self._protocol_error("The local consumption receipt could not be persisted") from error

    def _protocol_error(self, message):
        self.protocol_errors += 1
        # Do not include provider status strings that the official retry plugin
        # could mistake for permission to generate an already consumed call again.
        return ConsumptionConfirmationError(message)

    def _observe_wire(self, invocation, response, *, terminal=True):
        identifier = _field(response, "id")
        if not isinstance(identifier, str) or not re.fullmatch(r"chatcmpl-[a-f0-9]{32}", identifier):
            raise self._protocol_error("Model response has no verifiable local request identity")
        request_id = identifier.removeprefix("chatcmpl-")
        if invocation.request_id is not None and invocation.request_id != request_id:
            raise self._protocol_error("A logical model call changed its response identity")
        invocation.request_id = request_id
        for choice in _field(response, "choices", []) or []:
            delta = _field(choice, "delta")
            # The local endpoint emits a separate empty-delta finish frame.
            # LiteLLM can synthesize stop on content chunks; that is not proof
            # that the endpoint's terminal frame was consumed.
            empty_delta = delta is not None and not any(
                _field(delta, name) for name in
                ("content", "tool_calls", "function_call", "reasoning_content", "reasoning")
            )
            if terminal and _field(choice, "finish_reason") is not None and (not invocation.streaming or empty_delta):
                invocation.terminal_seen = True

    async def _post(self, request_id, action, *, attempts, report_error=True):
        import httpx

        def error(message):
            return self._protocol_error(message) if report_error else ConsumptionConfirmationError(message)

        for attempt in range(attempts):
            retry = False
            try:
                async with httpx.AsyncClient(timeout=self.ack_timeout, trust_env=False) as client:
                    response = await client.post(
                        f"{self.endpoint}/requests/{request_id}/{action}",
                        headers={"Authorization": "Bearer " + self.api_key}, json={},
                    )
                if response.status_code == 200:
                    body = response.json()
                    if isinstance(body, dict) and body.get("request_id") == request_id and body.get("state") == (
                            "consumed" if action == "ack" else "cancelled"):
                        return
                    raise error("The endpoint returned an invalid delivery acknowledgement")
                retry = response.status_code in {408, 429, 500, 502, 503, 504}
            except (httpx.RequestError, ValueError):
                retry = True
            if not retry or attempt + 1 == attempts:
                break
            await asyncio.sleep(0.1 * (2**attempt))
        raise error("Consumption confirmation could not be established" if action == "ack"
                    else "Request cancellation could not be established")

    async def _acknowledge(self, invocation):
        self._receipt("consumer_received", invocation)
        await self._post(invocation.request_id, "ack", attempts=self.ack_attempts)
        invocation.acknowledged = True
        self._receipt("consumption_acknowledged", invocation)

    async def _cancel_unconsumed(self, invocation):
        # Never relabel an actually yielded final response as unconsumed just
        # because its ACK was lost. Its whole rollout remains unconfirmed.
        if invocation.final_yielded or invocation.acknowledged:
            return
        import httpx

        invocation.cancellation_attempted = True
        timeout = min(self.ack_timeout, 0.25)
        try:
            # A model call can expire before receiving HTTP headers or any
            # response ID. Its logical key already exists locally; the server
            # records a tombstone even if enqueue/create is still in progress.
            # Attempt cleanup even while unwinding task cancellation, with a
            # separate short cap so the official verifier is not held up.
            async with asyncio.timeout(timeout):
                async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
                    response = await client.post(
                        f"{self.endpoint}/requests/by-key/{invocation.key}/cancel",
                        headers={"Authorization": "Bearer " + self.api_key}, json={},
                    )
                if response.status_code == 200:
                    body = response.json()
                    invocation.cancellation_confirmed = bool(
                        isinstance(body, dict) and body.get("idempotency_key") == invocation.key
                        and body.get("state") in {"cancelled", "consumed", "failed"}
                    )
        except (Exception, asyncio.CancelledError):
            # Best effort cleanup is not evidence of consumption. The coordinator
            # also cancels every unconsumed request when this attempt is released.
            pass

    def install(self):
        if self._restore is not None:
            raise RuntimeError("Delivery tracker is already installed")
        versions = {name: importlib.metadata.version(name) for name in ("google-adk", "litellm", "openai")}
        if versions != {"google-adk": "1.36.1", "litellm": "1.83.14", "openai": "2.24.0"}:
            raise RuntimeError("Consumption tracking targets the pinned ADK, LiteLLM and OpenAI SDK versions")
        from google.adk.models import lite_llm

        original_generate = lite_llm.LiteLlm.generate_content_async
        original_completion = lite_llm.LiteLLMClient.acompletion
        tracker = self

        @functools.wraps(original_completion)
        async def completion(llm_client, *args, **kwargs):
            invocation = tracker._context.get()
            if invocation is None:
                raise tracker._protocol_error("Local model transport bypassed the ADK consumption boundary")
            invocation.transport_calls += 1
            if invocation.transport_calls != 1:
                raise tracker._protocol_error("A logical ADK invocation attempted multiple distinct completions")
            actual_base = kwargs.get("api_base") or kwargs.get("base_url") or os.environ.get("OPENAI_BASE_URL")
            if not isinstance(actual_base, str) or actual_base.rstrip("/") != tracker.endpoint:
                raise tracker._protocol_error("Model completion is not bound to this rollout endpoint")
            headers = dict(kwargs.get("extra_headers") or {})
            for source in (headers, kwargs.get("headers") or {}):
                for key in source:
                    if key.lower() == "idempotency-key":
                        raise tracker._protocol_error("Idempotency identity must belong to one logical invocation")
            headers["Idempotency-Key"] = invocation.key
            kwargs["extra_headers"] = headers
            try:
                with tracker.model_wait():
                    response = await original_completion(llm_client, *args, **kwargs)
            finally:
                # Flush the final <=poll-period queue interval before ADK can
                # yield a tool call whose budget check uses this session clock.
                await tracker.refresh_after_wait()
            if invocation.streaming:
                # LiteLLM fabricates a final stop chunk on raw EOF. Inspect its
                # underlying OpenAI stream before that normalization; checking
                # only LiteLLM's output would mistake a disconnect for completion.
                from litellm import CustomStreamWrapper
                from openai import AsyncStream

                if not isinstance(response, CustomStreamWrapper) or not isinstance(response.completion_stream, AsyncStream):
                    raise tracker._protocol_error("Streaming model response has no supported raw consumption boundary")
                invocation.raw_stream = _TrackedStream(response.completion_stream, tracker, invocation, raw=True)
                response.completion_stream = invocation.raw_stream
                invocation.stream = _TrackedStream(response, tracker, invocation)
                return invocation.stream
            tracker._observe_wire(invocation, response)
            invocation.wire_exhausted = True
            return response

        @functools.wraps(original_generate)
        async def generate(model, llm_request, stream=False):
            invocation = _Invocation(key=uuid.uuid4().hex, streaming=bool(stream))
            tracker.invocations.append(invocation)
            tracker._receipt("logical_call_started", invocation)
            token = tracker._context.set(invocation)
            source = original_generate(model, llm_request, stream=stream)
            try:
                async for response in source:
                    final = not bool(getattr(response, "partial", False))
                    if final:
                        if not (invocation.request_id and invocation.terminal_seen and invocation.wire_exhausted
                                and (not invocation.streaming or invocation.raw_exhausted)):
                            raise tracker._protocol_error("A complete model response lacks a verified finish frame")
                        invocation.final_yielded += 1
                    yield response  # ADK parsing has succeeded, but caller has not consumed it yet.
                    if final:
                        invocation.final_resumed += 1
                if not invocation.final_yielded:
                    raise tracker._protocol_error("The model stream ended without a complete parsed response")
                await tracker._acknowledge(invocation)
                invocation.completed = True
            except BaseException:
                invocation.failed = True
                raise
            finally:
                try:
                    if not invocation.completed:
                        await tracker._cancel_unconsumed(invocation)
                    for stream_to_close in (source, invocation.stream, invocation.raw_stream):
                        if stream_to_close is not None:
                            try:
                                await stream_to_close.aclose()
                            except Exception:
                                pass  # Preserve the original delivery/model failure.
                finally:
                    # Async-generator finalization can run in a copied context.
                    # Never leak/reset another call's active consumption identity.
                    try:
                        tracker._context.reset(token)
                    except ValueError:
                        if tracker._context.get() is invocation:
                            tracker._context.set(None)

        lite_llm.LiteLLMClient.acompletion = completion
        lite_llm.LiteLlm.generate_content_async = generate

        def restore():
            if lite_llm.LiteLLMClient.acompletion is completion:
                lite_llm.LiteLLMClient.acompletion = original_completion
            if lite_llm.LiteLlm.generate_content_async is generate:
                lite_llm.LiteLlm.generate_content_async = original_generate

        self._restore = restore
        return self

    def assert_delivery_complete(self):
        # Ordinary failed HTTP calls that never yielded a complete response may
        # be retried by the official agent. They never become consumed calls.
        unconfirmed = [item for item in self.invocations if item.final_yielded and not item.acknowledged]
        if self.protocol_errors or unconfirmed:
            raise ConsumptionConfirmationError("This rollout has unconfirmed model-response consumption")

    def summary(self):
        return {"protocol": "adk_final_response_ack_v1", "logical_invocations": len(self.invocations),
                "acknowledged": sum(item.acknowledged for item in self.invocations),
                "unconfirmed_final_responses": sum(bool(item.final_yielded) and not item.acknowledged
                                                   for item in self.invocations),
                "failed_before_final_response": sum(item.failed and not item.final_yielded for item in self.invocations),
                "cancellation_attempted": sum(item.cancellation_attempted for item in self.invocations),
                "cancellation_confirmed": sum(item.cancellation_confirmed for item in self.invocations),
                "protocol_errors": self.protocol_errors,
                "receipt_path": str(self.receipt_path),
                "consumption_boundary": "final parsed ADK responses yielded and caller resumed through normal exhaustion"}

    def restore(self):
        if self._restore is not None:
            self._restore()
            self._restore = None


def install_delivery_tracking(endpoint, rollout_id, receipt_path, *, session_clock=None, api_key=None):
    return DeliveryTracker(endpoint, rollout_id, receipt_path, session_clock=session_clock,
                           api_key=api_key).install()
