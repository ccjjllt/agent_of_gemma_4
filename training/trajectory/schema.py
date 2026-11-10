"""Immutable, strictly validated records shared by rollout, SPO and recovery.

The ``from_dict`` constructors reject unknown and missing required fields.  The
 dataclass constructors perform the same validation, so persisted JSON and
in-memory objects cannot bypass TITO or adapter invariants.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, ClassVar, Mapping, Sequence


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value: Any, name: str, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc
    if not isinstance(decoded, dict) or any(not isinstance(k, str) for k in decoded):
        raise ValueError(f"{name} must have string keys")
    return decoded


def _token_ids(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of token IDs")
    return tuple(_integer(item, f"{name}[{index}]")
                 for index, item in enumerate(value))


def _from_dict(cls: type[Any], value: Mapping[str, Any]) -> Any:
    if not isinstance(value, Mapping):
        raise ValueError(f"{cls.__name__} must be constructed from an object")
    definitions = fields(cls)
    allowed = {item.name for item in definitions}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{cls.__name__} has unknown fields: {sorted(unknown)}")
    required = {
        item.name for item in definitions
        if item.default is dataclasses.MISSING
        and item.default_factory is dataclasses.MISSING
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"{cls.__name__} is missing fields: {sorted(missing)}")
    return cls(**dict(value))


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class TaskKey:
    task_id: str
    repo: str
    base_commit: str
    test_patch_sha256: str
    judge_environment_hash: str

    def __post_init__(self) -> None:
        for name in ("task_id", "repo", "base_commit"):
            _text(getattr(self, name), name)
        _sha256(self.test_patch_sha256, "test_patch_sha256")
        _sha256(self.judge_environment_hash, "judge_environment_hash")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskKey":
        return _from_dict(cls, value)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                          allow_nan=False)


@dataclass(frozen=True)
class BehaviorAdapter:
    version: int
    sha256: str

    def __post_init__(self) -> None:
        _integer(self.version, "behavior_adapter.version")
        _sha256(self.sha256, "behavior_adapter.sha256")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BehaviorAdapter":
        return _from_dict(cls, value)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    top_k: int
    seed: int

    def __post_init__(self) -> None:
        _number(self.temperature, "sampling.temperature", 0.0)
        _number(self.top_p, "sampling.top_p", 0.0, 1.0)
        if self.top_p <= 0:
            raise ValueError("sampling.top_p must be > 0")
        _integer(self.top_k, "sampling.top_k")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("sampling.seed must be an integer")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SamplingConfig":
        return _from_dict(cls, value)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class ActionCall:
    trajectory_id: str
    call_index: int
    request_id: str
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    action_mask: tuple[int, ...]
    behavior_adapter: BehaviorAdapter
    sampling: SamplingConfig
    parser_valid: bool
    finish_reason: str
    raw_tool_call_ids: tuple[str, ...]
    delivery_state: str
    consumed_at: str | None
    token_timing: dict[str, Any]

    def __post_init__(self) -> None:
        _text(self.trajectory_id, "trajectory_id")
        _integer(self.call_index, "call_index")
        _text(self.request_id, "request_id")
        object.__setattr__(self, "prompt_ids", _token_ids(self.prompt_ids, "prompt_ids"))
        object.__setattr__(self, "completion_ids",
                           _token_ids(self.completion_ids, "completion_ids"))
        if isinstance(self.behavior_logprobs, (str, bytes)) or not isinstance(
                self.behavior_logprobs, Sequence):
            raise ValueError("behavior_logprobs must be a sequence")
        logprobs = tuple(_number(value, f"behavior_logprobs[{index}]", maximum=0.0)
                         for index, value in enumerate(self.behavior_logprobs))
        object.__setattr__(self, "behavior_logprobs", logprobs)
        if isinstance(self.action_mask, (str, bytes)) or not isinstance(
                self.action_mask, Sequence):
            raise ValueError("action_mask must be a sequence")
        mask: list[int] = []
        for index, value in enumerate(self.action_mask):
            if isinstance(value, bool):
                value = int(value)
            if not isinstance(value, int) or value not in (0, 1):
                raise ValueError(f"action_mask[{index}] must be 0 or 1")
            mask.append(value)
        object.__setattr__(self, "action_mask", tuple(mask))
        if not (len(self.completion_ids) == len(logprobs) == len(mask)):
            raise ValueError(
                "TITO requires len(completion_ids) == len(behavior_logprobs) "
                "== len(action_mask)"
            )
        if isinstance(self.behavior_adapter, Mapping):
            object.__setattr__(self, "behavior_adapter",
                               BehaviorAdapter.from_dict(self.behavior_adapter))
        if not isinstance(self.behavior_adapter, BehaviorAdapter):
            raise ValueError("behavior_adapter must be a BehaviorAdapter or object")
        if isinstance(self.sampling, Mapping):
            object.__setattr__(self, "sampling", SamplingConfig.from_dict(self.sampling))
        if not isinstance(self.sampling, SamplingConfig):
            raise ValueError("sampling must be a SamplingConfig or object")
        _boolean(self.parser_valid, "parser_valid")
        _text(self.finish_reason, "finish_reason")
        _text(self.delivery_state, "delivery_state")
        if self.consumed_at is not None:
            _text(self.consumed_at, "consumed_at")
        object.__setattr__(self, "token_timing",
                           _json_object(self.token_timing, "token_timing"))
        if isinstance(self.raw_tool_call_ids, (str, bytes)) or not isinstance(
                self.raw_tool_call_ids, Sequence):
            raise ValueError("raw_tool_call_ids must be a sequence")
        raw_ids = tuple(_text(value, f"raw_tool_call_ids[{index}]")
                        for index, value in enumerate(self.raw_tool_call_ids))
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError("raw_tool_call_ids must be unique within a call")
        object.__setattr__(self, "raw_tool_call_ids", raw_ids)

    @property
    def action_token_count(self) -> int:
        return sum(self.action_mask)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionCall":
        return _from_dict(cls, value)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class Trajectory:
    schema_version: int
    trajectory_id: str
    task_key: TaskKey
    stream_sequence: int
    behavior_adapter: BehaviorAdapter
    calls: tuple[ActionCall, ...]
    action_token_count: int
    status: str
    verification_ref: str | None = None
    reward_ref: str | None = None
    raw_advantage: float | None = None
    provenance: dict[str, Any] = dataclasses.field(default_factory=dict)

    STATUSES: ClassVar[frozenset[str]] = frozenset({
        "ASSIGNED", "RUNNING", "WAITING_GPU", "VERIFYING", "REWARDED",
        "BUFFERED", "EXCLUDED", "RESERVED", "CONSUMED", "STALE_EXCLUDED",
        "NOOP_CONSUMED",
    })

    def __post_init__(self) -> None:
        _integer(self.schema_version, "schema_version", 1)
        _text(self.trajectory_id, "trajectory_id")
        if isinstance(self.task_key, Mapping):
            object.__setattr__(self, "task_key", TaskKey.from_dict(self.task_key))
        if not isinstance(self.task_key, TaskKey):
            raise ValueError("task_key must be a TaskKey or object")
        _integer(self.stream_sequence, "stream_sequence")
        if isinstance(self.behavior_adapter, Mapping):
            object.__setattr__(self, "behavior_adapter",
                               BehaviorAdapter.from_dict(self.behavior_adapter))
        if not isinstance(self.behavior_adapter, BehaviorAdapter):
            raise ValueError("behavior_adapter must be a BehaviorAdapter or object")
        if isinstance(self.calls, (str, bytes)) or not isinstance(self.calls, Sequence):
            raise ValueError("calls must be a sequence")
        calls = tuple(ActionCall.from_dict(item) if isinstance(item, Mapping) else item
                      for item in self.calls)
        if any(not isinstance(item, ActionCall) for item in calls):
            raise ValueError("every call must be an ActionCall or object")
        object.__setattr__(self, "calls", calls)
        for index, call in enumerate(calls):
            if call.trajectory_id != self.trajectory_id:
                raise ValueError("every call must belong to its trajectory")
            if call.call_index != index:
                raise ValueError("call_index values must be contiguous from zero")
            if call.behavior_adapter != self.behavior_adapter:
                raise ValueError("a trajectory cannot switch behavior adapter")
        _integer(self.action_token_count, "action_token_count")
        actual = sum(call.action_token_count for call in calls)
        if self.action_token_count != actual:
            raise ValueError(
                f"action_token_count={self.action_token_count} does not match mask sum {actual}"
            )
        if self.status not in self.STATUSES:
            raise ValueError(f"unknown trajectory status: {self.status}")
        for name in ("verification_ref", "reward_ref"):
            value = getattr(self, name)
            if value is not None:
                _text(value, name)
        if self.raw_advantage is not None:
            object.__setattr__(self, "raw_advantage",
                               _number(self.raw_advantage, "raw_advantage"))
        object.__setattr__(self, "provenance", _json_object(self.provenance, "provenance"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Trajectory":
        return _from_dict(cls, value)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class RewardRecord:
    resolved: bool | None
    P: float | None
    G: float | None
    progress: float | None
    regression: float | None
    validity: float | None
    base_reward: float | None
    efficiency: float | None
    efficiency_weight: float
    total_reward: float | None
    eligible: bool
    exclusion_reason: str | None
    objective_version: str

    def __post_init__(self) -> None:
        if self.resolved is not None:
            _boolean(self.resolved, "resolved")
        for name in ("P", "G", "progress", "validity", "base_reward",
                     "efficiency", "total_reward"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _number(value, name, 0.0, 1.0))
        if self.regression is not None:
            object.__setattr__(self, "regression",
                               _number(self.regression, "regression", -1.0, 0.0))
        object.__setattr__(self, "efficiency_weight",
                           _number(self.efficiency_weight, "efficiency_weight", 0.0, 1.0))
        _boolean(self.eligible, "eligible")
        if self.eligible:
            if type(self.resolved) is not bool:
                raise ValueError("eligible rewards require an official resolved verdict")
            for name in ("P", "G", "progress", "regression", "validity",
                         "base_reward", "total_reward"):
                if getattr(self, name) is None:
                    raise ValueError(f"eligible rewards require {name}")
            if self.exclusion_reason is not None:
                raise ValueError("eligible rewards cannot have an exclusion_reason")
        else:
            if self.total_reward is not None:
                raise ValueError("ineligible rewards cannot have a training reward")
            if not isinstance(self.exclusion_reason, str) or not self.exclusion_reason.strip():
                raise ValueError("ineligible rewards require an exclusion_reason")
        _text(self.objective_version, "objective_version")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RewardRecord":
        return _from_dict(cls, value)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass(frozen=True)
class UpdateBatch:
    update_id: str
    policy_before: BehaviorAdapter
    trajectory_ids: tuple[str, ...]
    raw_advantages: dict[str, float]
    frozen_advantages: dict[str, float]
    token_mean: float
    token_std: float
    total_action_tokens: int
    max_behavior_lag: int
    optimizer_cursor: int
    phase: str

    PHASES: ClassVar[frozenset[str]] = frozenset({
        "RESERVED", "LEARNING", "OPTIMIZED", "EXPORTED", "REGISTERED",
        "ACTIVATED", "CONSUMED",
    })

    def __post_init__(self) -> None:
        _text(self.update_id, "update_id")
        if isinstance(self.policy_before, Mapping):
            object.__setattr__(self, "policy_before",
                               BehaviorAdapter.from_dict(self.policy_before))
        if not isinstance(self.policy_before, BehaviorAdapter):
            raise ValueError("policy_before must be a BehaviorAdapter or object")
        if isinstance(self.trajectory_ids, (str, bytes)) or not isinstance(
                self.trajectory_ids, Sequence):
            raise ValueError("trajectory_ids must be a sequence")
        ids = tuple(_text(value, "trajectory_id") for value in self.trajectory_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("trajectory_ids must be non-empty and unique")
        object.__setattr__(self, "trajectory_ids", ids)
        for name in ("raw_advantages", "frozen_advantages"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or set(value) != set(ids):
                raise ValueError(f"{name} keys must exactly match trajectory_ids")
            converted = {
                key: _number(value[key], f"{name}[{key}]") for key in ids
            }
            object.__setattr__(self, name, converted)
        object.__setattr__(self, "token_mean", _number(self.token_mean, "token_mean"))
        object.__setattr__(self, "token_std",
                           _number(self.token_std, "token_std", 0.0))
        _integer(self.total_action_tokens, "total_action_tokens", 1)
        _integer(self.max_behavior_lag, "max_behavior_lag")
        _integer(self.optimizer_cursor, "optimizer_cursor")
        if self.phase not in self.PHASES:
            raise ValueError(f"unknown update phase: {self.phase}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UpdateBatch":
        return _from_dict(cls, value)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


def validate_update_batch(batch: UpdateBatch, trajectories: Sequence[Trajectory]) -> None:
    """Check frozen IDs, advantages, adapter lag and action-token accounting."""
    if tuple(item.trajectory_id for item in trajectories) != batch.trajectory_ids:
        raise ValueError("update trajectory order does not match its frozen batch")
    total = 0
    maximum_lag = 0
    for trajectory in trajectories:
        trajectory_id = trajectory.trajectory_id
        if batch.raw_advantages[trajectory_id] != trajectory.raw_advantage:
            raise ValueError("raw advantage changed after batch freezing")
        total += trajectory.action_token_count
        for call in trajectory.calls:
            lag = batch.policy_before.version - call.behavior_adapter.version
            if lag < 0:
                raise ValueError("behavior adapter is newer than policy_before")
            maximum_lag = max(maximum_lag, lag)
    if total != batch.total_action_tokens or maximum_lag != batch.max_behavior_lag:
        raise ValueError("frozen action-token count or behavior lag is inconsistent")
