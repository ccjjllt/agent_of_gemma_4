"""Token-in/token-out (TITO) alignment helpers.

No function in this module retokenizes text.  Every emitted token retains its
original call identity, position, behavior logprob and adapter evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .schema import ActionCall, BehaviorAdapter, Trajectory


@dataclass(frozen=True)
class ActionToken:
    trajectory_id: str
    call_index: int
    request_id: str
    completion_index: int
    token_id: int
    behavior_logprob: float
    behavior_adapter: BehaviorAdapter
    raw_tool_call_ids: tuple[str, ...]


def validate_token_alignment(value: ActionCall | Trajectory) -> int:
    """Validate TITO lengths and return the exact action-token count.

    Schema construction already enforces these invariants; this explicit entry
    point is intended for trainer boundaries and data recovered from storage.
    """
    calls = value.calls if isinstance(value, Trajectory) else (value,)
    if not isinstance(value, (ActionCall, Trajectory)):
        raise TypeError("value must be an ActionCall or Trajectory")
    total = 0
    for call in calls:
        if not (len(call.completion_ids) == len(call.behavior_logprobs)
                == len(call.action_mask)):
            raise ValueError("TITO completion, behavior logprob and mask lengths differ")
        if any(mask not in (0, 1) for mask in call.action_mask):
            raise ValueError("action masks are binary")
        total += sum(call.action_mask)
    if isinstance(value, Trajectory):
        if value.action_token_count != total:
            raise ValueError("trajectory action_token_count is not its mask sum")
        for call in calls:
            if call.behavior_adapter != value.behavior_adapter:
                raise ValueError("trajectory contains multiple behavior adapters")
    return total


def count_action_tokens(value: ActionCall | Trajectory) -> int:
    return validate_token_alignment(value)


def iter_action_tokens(trajectory: Trajectory) -> Iterator[ActionToken]:
    """Yield only mask=1 tokens in stable trajectory/call/completion order."""
    validate_token_alignment(trajectory)
    for call in trajectory.calls:
        for index, token_id in enumerate(call.completion_ids):
            if call.action_mask[index]:
                yield ActionToken(
                    trajectory_id=trajectory.trajectory_id,
                    call_index=call.call_index,
                    request_id=call.request_id,
                    completion_index=index,
                    token_id=token_id,
                    behavior_logprob=call.behavior_logprobs[index],
                    behavior_adapter=call.behavior_adapter,
                    raw_tool_call_ids=call.raw_tool_call_ids,
                )


def flatten_action_tokens(trajectory: Trajectory) -> tuple[ActionToken, ...]:
    return tuple(iter_action_tokens(trajectory))


def aligned_call_payload(call: ActionCall) -> dict[str, object]:
    """Return trainer-ready original IDs without dropping non-action positions."""
    validate_token_alignment(call)
    return {
        "trajectory_id": call.trajectory_id,
        "call_index": call.call_index,
        "request_id": call.request_id,
        "prompt_ids": call.prompt_ids,
        "completion_ids": call.completion_ids,
        "behavior_logprobs": call.behavior_logprobs,
        "action_mask": call.action_mask,
        "behavior_adapter": call.behavior_adapter,
        "raw_tool_call_ids": call.raw_tool_call_ids,
    }
