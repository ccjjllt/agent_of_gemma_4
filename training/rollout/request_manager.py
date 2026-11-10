"""Trajectory-scoped TITO requests on top of the durable RequestStore.

RequestStore remains the owner of idempotency, cancellation and explicit consumer
ACKs.  This adapter removes policy version from the public read API: callers use
an immutable trajectory ID or request ID, while the fixed behavior adapter is
validated from durable per-request metadata.  No text is ever retokenized.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import os
import re
from typing import Any, Mapping, Sequence

from ..request_store import RequestConflict, RequestStore, RequestUnavailable
from ..trajectory.schema import ActionCall, BehaviorAdapter, SamplingConfig, TaskKey, Trajectory
from .policy_version import AdapterManifest, PolicyVersion


_SAFE_TRAJECTORY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_TIMING_FIELDS = (
    "requested_max_tokens", "effective_max_tokens", "context_limited",
    "output_budget_source", "client_requested_max_tokens", "generation_elapsed_seconds",
    "first_scores_seconds", "first_token_seconds", "reused_prefix_tokens", "profiled_tokens",
    "generation_backend", "timing_semantics", "queue_wait_seconds",
)


def _adapter(value: BehaviorAdapter | PolicyVersion | AdapterManifest | Mapping[str, Any]) -> BehaviorAdapter:
    if isinstance(value, BehaviorAdapter):
        return value
    if isinstance(value, (PolicyVersion, AdapterManifest)):
        return BehaviorAdapter(value.version, value.adapter_sha256)
    if isinstance(value, Mapping):
        if "adapter_sha256" in value:
            return BehaviorAdapter(value["version"], value["adapter_sha256"])
        return BehaviorAdapter.from_dict(value)
    raise TypeError("a fixed BehaviorAdapter is required")


def _ids(value: Any, name: str, *, nonempty: bool) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RequestUnavailable(f"{name} is required for TITO")
    if any(type(item) is not int or item < 0 for item in value):
        raise RequestUnavailable(f"{name} contains an invalid token ID")
    result = tuple(value)
    if nonempty and not result:
        raise RequestUnavailable(f"{name} must not be empty")
    return result


def _logprobs(value: Any) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RequestUnavailable("behavior_logprobs are required for TITO")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) \
                or not math.isfinite(float(item)) or item > 0:
            raise RequestUnavailable("behavior logprobs must be finite and <= 0")
        result.append(float(item))
    return tuple(result)


def _mask(value: Any, completion: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        # Existing Policy.finish_generation returns only actually sampled model
        # tokens in completion_ids; host messages live exclusively in prompt_ids.
        # This is a provenance-preserving mask, not a text/span reconstruction.
        return (1,) * len(completion)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RequestUnavailable("action_mask must be a token-aligned sequence")
    if any(type(item) is not int or item not in (0, 1) for item in value):
        raise RequestUnavailable("action_mask must contain only integer 0/1")
    return tuple(value)


def _tool_ids(result: Mapping[str, Any]) -> tuple[str, ...]:
    message = result.get("assistant_message")
    calls = message.get("tool_calls", []) if isinstance(message, Mapping) else []
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        raise RequestUnavailable("assistant tool_calls must be a sequence")
    message_ids: list[str] = []
    for call in calls:
        if not isinstance(call, Mapping) or not isinstance(call.get("id"), str) or not call["id"].strip():
            raise RequestUnavailable("every raw tool call must retain its original stable ID")
        message_ids.append(call["id"])
    supplied = result.get("raw_tool_call_ids")
    if supplied is not None:
        if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence) \
                or any(not isinstance(item, str) or not item.strip() for item in supplied):
            raise RequestUnavailable("raw_tool_call_ids must be a sequence of stable strings")
        supplied_ids = list(supplied)
        if message_ids and supplied_ids != message_ids:
            raise RequestConflict("raw tool IDs differ from the delivered assistant message")
        message_ids = supplied_ids
    if len(set(message_ids)) != len(message_ids):
        raise RequestUnavailable("raw tool call IDs must be unique within a completion")
    return tuple(message_ids)


def _timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RequestUnavailable("consumed_at has an invalid timestamp")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise RequestUnavailable("consumed_at has an invalid timestamp")
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
    if isinstance(value, str) and value.strip():
        return value
    raise RequestUnavailable("consumed_at has an invalid timestamp")


class RequestManager:
    """Strict ActionCall construction and complete-trajectory sealing."""

    def __init__(self, store: RequestStore | str | os.PathLike[str]):
        self.store = store if isinstance(store, RequestStore) else RequestStore(store)
        self._owns_store = not isinstance(store, RequestStore)

    @staticmethod
    def _trajectory_id(value: str) -> str:
        if not isinstance(value, str) or _SAFE_TRAJECTORY.fullmatch(value) is None:
            raise ValueError("trajectory_id must be a safe public identifier")
        return value

    @staticmethod
    def _metadata_adapter(row: Mapping[str, Any]) -> BehaviorAdapter | None:
        metadata = row.get("metadata") or {}
        value = metadata.get("behavior_adapter") if isinstance(metadata, Mapping) else None
        return _adapter(value) if value is not None else None

    def _rows(self, trajectory_id: str) -> list[dict[str, Any]]:
        """Read by immutable trajectory ID, not by a public policy-version argument."""
        trajectory_id = self._trajectory_id(trajectory_id)
        with self.store._lock:
            rows = [self.store._decode(row) for row in self.store.db.execute(
                "SELECT * FROM requests WHERE rollout_id=? ORDER BY call_index", (trajectory_id,)
            ).fetchall()]
        versions = {row["policy_version"] for row in rows}
        if len(versions) > 1:
            raise RequestConflict("a trajectory cannot span behavior policy versions")
        return rows

    def _resolve_adapter(
        self,
        row: Mapping[str, Any],
        explicit: BehaviorAdapter | PolicyVersion | AdapterManifest | Mapping[str, Any] | None = None,
    ) -> BehaviorAdapter:
        stored = self._metadata_adapter(row)
        if explicit is None:
            if stored is None:
                # Legacy rows still carry their immutable version, but a hash is
                # required; callers recovering them must supply the run manifest.
                raise RequestUnavailable("request is missing its behavior adapter hash")
            return stored
        adapter = _adapter(explicit)
        if stored is not None and stored != adapter:
            raise RequestConflict("request behavior adapter differs from its trajectory binding")
        if row["policy_version"] != adapter.version:
            raise RequestConflict("request policy version differs from its behavior adapter")
        return adapter

    def create_call(
        self,
        trajectory_id: str,
        behavior: BehaviorAdapter | PolicyVersion | AdapterManifest | Mapping[str, Any],
        idempotency_key: str | None = None,
        request: Mapping[str, Any] | None = None,
        *,
        request_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        trajectory_id = self._trajectory_id(trajectory_id)
        adapter = _adapter(behavior)
        if request_hash is None:
            if request is None:
                raise ValueError("create_call requires request or request_hash")
            request_hash = self.store.request_hash(request)
        if not isinstance(request_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", request_hash):
            raise ValueError("request_hash must be a SHA256 digest")
        for row in self._rows(trajectory_id):
            self._resolve_adapter(row, adapter)
            if row["idempotency_key"] == idempotency_key and idempotency_key is not None:
                # RequestStore performs the authoritative request-body comparison.
                break
        details = dict(metadata or {})
        supplied_adapter = details.get("behavior_adapter")
        if supplied_adapter is not None and _adapter(supplied_adapter) != adapter:
            raise RequestConflict("metadata behavior_adapter conflicts with create_call")
        details["behavior_adapter"] = adapter.to_dict()
        row, created = self.store.create(
            trajectory_id, adapter.version, idempotency_key, request_hash, details,
        )
        self._resolve_adapter(row, adapter)
        return row, created

    def commit_generated(
        self,
        request_id: str,
        result: Mapping[str, Any],
        *,
        trajectory_id: str | None = None,
        behavior: BehaviorAdapter | PolicyVersion | AdapterManifest | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise TypeError("generated result must be a mapping")
        row = self.store.get(request_id, trajectory_id)
        adapter = self._resolve_adapter(row, behavior)
        # Validate all TITO evidence before making a deliverable durable result.
        prompt = _ids(result.get("prompt_ids", result.get("input_ids")), "prompt_ids", nonempty=True)
        completion = _ids(
            result.get("completion_ids", result.get("output_ids")), "completion_ids", nonempty=True,
        )
        logprobs = _logprobs(result.get("behavior_logprobs", result.get("logprobs")))
        mask = _mask(result.get("action_mask"), completion)
        if not len(completion) == len(logprobs) == len(mask):
            raise RequestUnavailable("TITO completion, logprob and action_mask lengths differ")
        raw_tool_ids = _tool_ids(result)
        returned_hash = result.get("adapter_sha256")
        if returned_hash is not None and returned_hash != adapter.sha256:
            raise RequestConflict("generated adapter hash differs from its fixed trajectory adapter")
        supplied_behavior = result.get("behavior_adapter")
        if supplied_behavior is not None and _adapter(supplied_behavior) != adapter:
            raise RequestConflict("generated behavior adapter differs from its request")
        payload = dict(result)
        payload.update({
            "request_id": request_id,
            "prompt_ids": list(prompt),
            "completion_ids": list(completion),
            "behavior_logprobs": list(logprobs),
            "action_mask": list(mask),
            "raw_tool_call_ids": list(raw_tool_ids),
            "behavior_adapter": adapter.to_dict(),
            "adapter_sha256": adapter.sha256,
        })
        payload.pop("input_ids", None)
        payload.pop("output_ids", None)
        payload.pop("logprobs", None)
        if row["result"] is not None:
            if row["result"] != payload:
                raise RequestConflict("A request result may only be committed once")
            return row
        return self.store.generated(request_id, payload)

    def to_action_call(
        self,
        row: Mapping[str, Any],
        *,
        behavior: BehaviorAdapter | PolicyVersion | AdapterManifest | Mapping[str, Any] | None = None,
    ) -> ActionCall:
        if row.get("state") != "consumed" or row.get("result") is None:
            raise RequestUnavailable("only explicitly ACKed calls can enter a trajectory")
        result = row["result"]
        if result.get("parse_error") is not None:
            raise RequestUnavailable("a parse-failed response cannot enter training")
        adapter = self._resolve_adapter(row, behavior)
        prompt = _ids(result.get("prompt_ids"), "prompt_ids", nonempty=True)
        completion = _ids(result.get("completion_ids"), "completion_ids", nonempty=True)
        logprobs = _logprobs(result.get("behavior_logprobs"))
        mask = _mask(result.get("action_mask"), completion)
        raw_tool_ids = _tool_ids(result)
        message = result.get("assistant_message")
        parser_valid = result.get("parser_valid")
        if parser_valid is None:
            parser_valid = isinstance(message, Mapping)
        if type(parser_valid) is not bool:
            raise RequestUnavailable("parser_valid must be explicit boolean evidence")
        sampling = dict(result.get("sampling") or {})
        # seed=-1 records engine-owned RNG when no per-call seed was supplied;
        # it never pretends that a deterministic per-request seed existed.
        sampling.setdefault("seed", -1)
        sampling_config = SamplingConfig.from_dict({
            key: sampling[key] for key in ("temperature", "top_p", "top_k", "seed")
            if key in sampling
        })
        finish_reason = result.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise RequestUnavailable("finish_reason is required")
        metadata = row.get("metadata") or {}
        timing = {key: result[key] for key in _TIMING_FIELDS if key in result}
        for key in ("scheduler_queue_wait_seconds", "preparation_elapsed_seconds"):
            if key in metadata and metadata[key] is not None:
                timing[key] = metadata[key]
        return ActionCall(
            trajectory_id=row["rollout_id"],
            call_index=row["call_index"],
            request_id=row["request_id"],
            prompt_ids=prompt,
            completion_ids=completion,
            behavior_logprobs=logprobs,
            action_mask=mask,
            behavior_adapter=adapter,
            sampling=sampling_config,
            parser_valid=parser_valid,
            finish_reason=finish_reason,
            raw_tool_call_ids=raw_tool_ids,
            delivery_state="CONSUMED",
            consumed_at=_timestamp(row.get("consumed_at")),
            token_timing=timing,
        )

    def acknowledge(
        self,
        request_id: str,
        trajectory_id: str | None = None,
        *,
        behavior: BehaviorAdapter | PolicyVersion | AdapterManifest | Mapping[str, Any] | None = None,
    ) -> ActionCall:
        current = self.store.get(request_id, trajectory_id)
        adapter = self._resolve_adapter(current, behavior)
        row, _changed = self.store.acknowledge(request_id, current["rollout_id"], adapter.version)
        return self.to_action_call(row, behavior=adapter)

    def cancel(self, request_id: str, trajectory_id: str | None = None, *, reason: str = "cancelled") -> dict[str, Any]:
        return self.store.cancel(request_id, trajectory_id, reason)

    def cancel_by_key(
        self, trajectory_id: str, behavior: BehaviorAdapter | PolicyVersion | AdapterManifest,
        idempotency_key: str, *, reason: str = "client_cancelled_by_key",
    ) -> dict[str, Any]:
        return self.store.cancel_by_key(
            self._trajectory_id(trajectory_id), _adapter(behavior).version, idempotency_key, reason,
        )

    def seal_trajectory(
        self,
        trajectory_id: str,
        task_key: TaskKey,
        stream_sequence: int,
        behavior: BehaviorAdapter | PolicyVersion | AdapterManifest | Mapping[str, Any] | None = None,
        *,
        status: str = "VERIFYING",
        verification_ref: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> Trajectory:
        rows = self._rows(trajectory_id)
        if not rows:
            raise RequestUnavailable("a complete trajectory requires at least one model call")
        unresolved = [row["request_id"] for row in rows if row["state"] != "consumed"]
        if unresolved:
            raise RequestUnavailable(f"trajectory has unacknowledged model calls: {unresolved}")
        adapter = self._resolve_adapter(rows[0], behavior)
        calls = tuple(self.to_action_call(row, behavior=adapter) for row in rows)
        return Trajectory(
            schema_version=1,
            trajectory_id=trajectory_id,
            task_key=task_key,
            stream_sequence=stream_sequence,
            behavior_adapter=adapter,
            calls=calls,
            action_token_count=sum(call.action_token_count for call in calls),
            status=status,
            verification_ref=verification_ref,
            provenance={**(dict(provenance or {})), "ack_complete": True},
        )

    def excluded_trajectory(
        self,
        trajectory_id: str,
        task_key: TaskKey,
        stream_sequence: int,
        behavior: BehaviorAdapter | PolicyVersion | AdapterManifest | Mapping[str, Any],
        *,
        reason: str,
        verification_ref: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> Trajectory:
        """Preserve the valid consumed prefix without admitting incomplete ACK/TITO data."""
        adapter = _adapter(behavior)
        calls: list[ActionCall] = []
        dropped = 0
        for row in self._rows(trajectory_id):
            if row["state"] != "consumed" or row["call_index"] != len(calls):
                dropped += 1
                continue
            try:
                calls.append(self.to_action_call(row, behavior=adapter))
            except (RequestUnavailable, RequestConflict, ValueError, TypeError):
                dropped += 1
        return Trajectory(
            schema_version=1,
            trajectory_id=trajectory_id,
            task_key=task_key,
            stream_sequence=stream_sequence,
            behavior_adapter=adapter,
            calls=tuple(calls),
            action_token_count=sum(call.action_token_count for call in calls),
            status="EXCLUDED",
            verification_ref=verification_ref,
            provenance={
                **dict(provenance or {}), "ack_complete": False,
                "exclusion_reason": reason, "unusable_request_count": dropped,
            },
        )

    def close(self) -> None:
        if self._owns_store:
            self.store.close()


# Re-export the store's explicit failure types at the new integration boundary.
behavior_adapter = _adapter

__all__ = [
    "RequestManager", "RequestConflict", "RequestUnavailable", "behavior_adapter",
]
