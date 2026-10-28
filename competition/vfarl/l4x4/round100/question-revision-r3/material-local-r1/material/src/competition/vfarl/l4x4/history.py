# SPDX-License-Identifier: Apache-2.0
"""Prospective, opt-in ``canonical-ct-v1`` history construction.

This module owns no model and executes no tool. It never receives, changes, or
relabels sampled token IDs/logprobs. Returned IDs are NEW conditioning, obtained
by rendering structured history; they are not a replacement behavior sample.
Nothing here selects a rollout, assigns reward, or enables this profile.

``new_history`` returns JSON data suitable for a Snapshot ledger.
``render_prefix``, ``append_tool_result``, and ``append_user_nudge`` return
``{history, prompt_token_ids, metadata}``, without mutating their inputs.
The caller must keep every original sample and decide its existing fallback for
UnsupportedHistory; this exception is not an instruction to drop a sample.
HistoryBindingError instead identifies invalid caller/configuration/persisted
state contracts; it is deliberately not an UnsupportedHistory subclass.

Scope: one complete thought channel (or none), and one call already accepted by
the caller's strict NativeArguments parser. This is not vLLM's tolerant parser
and does not accept/recover the eight historical malformed calls. Visible text
before a call is preserved as content. Non-whitespace after a call, multiple or
unclosed channels/calls are unsupported. Mismatched parsed names are fatal
binding errors.

Reasoning split/label semantics and pre-call content semantics are adapted from
vLLM, Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0):
  vllm-project/vllm@b1388b1fbf5aaef47937fabe98931211684666a6
  vllm/reasoning/basic_parsers.py: BaseThinkingReasoningParser.extract_reasoning
  vllm/reasoning/gemma4_reasoning_parser.py: extract_reasoning
  vllm/reasoning/gemma4_utils.py: _strip_thought_label
  vllm/tool_parsers/gemma4_tool_parser.py: extract_tool_calls (content only)
Gemma4 utility copyright: Copyright 2025 Google Inc. HuggingFace Inc. team.
Only the complete-channel subset is used; ambiguity is rejected, not repaired.
These source semantics plus a pinned CT template do not establish equivalence
to an actual official HTTP/ADK/vLLM deployment or any model benefit.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any


PROFILE = 'canonical-ct-v1'
SCHEMA = 'vfarl-canonical-history-v1'
CT_TEMPLATE_SHA256 = '94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413'
SOURCE_COMMIT = 'b1388b1fbf5aaef47937fabe98931211684666a6'
_CHANNEL_OPEN, _CHANNEL_CLOSE = '<|channel>', '<channel|>'
_CALL_OPEN, _CALL_CLOSE = '<|tool_call>', '<tool_call|>'
_TURN_ENDS = ('<turn|>', '<|end|>')


class UnsupportedHistory(ValueError):
    """Outside this profile's scope; caller retains the original sampled data."""


class HistoryBindingError(ValueError):
    """Invalid caller/configuration/state binding; not a sample-shape fallback."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_text(value: str) -> str:
    return _sha_bytes(value.encode('utf-8'))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _sha_json(value: Any) -> str:
    return _sha_bytes(_json_bytes(value))


def _copy_json(value: Any, *, binding: bool = False) -> Any:
    # Round-trip ordinary JSON, rejecting nonfinite/unserializable values.
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        error = HistoryBindingError if binding else UnsupportedHistory
        raise error('Inputs must be JSON data') from exc


def _history_copy(history: dict[str, Any]) -> dict[str, Any]:
    result = _copy_json(history, binding=True)
    if (not isinstance(result, dict) or result.get('schema_version') != SCHEMA
            or result.get('profile') != PROFILE
            or result.get('template_sha256') != CT_TEMPLATE_SHA256
            or not isinstance(result.get('messages'), list)
            or not result['messages']
            or not isinstance(result.get('transitions'), list)
            or any(not isinstance(t, dict) for t in result['transitions'])):
        raise HistoryBindingError('Expected a canonical-ct-v1 history')
    if any(not isinstance(m, dict) or m.get('role') not in
           {'system', 'developer', 'user', 'assistant', 'tool'}
           for m in result['messages']):
        raise HistoryBindingError('Invalid persisted chat messages')
    for message in result['messages']:
        calls = message.get('tool_calls')
        if calls is not None and (not isinstance(calls, list)
                                  or any(not isinstance(call, dict) for call in calls)):
            raise HistoryBindingError('Invalid persisted tool-call records')
    return result


def new_history(initial_messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Copy actual initial messages; do not invent instructions or a user turn."""
    messages = _copy_json(initial_messages, binding=True)
    if (not isinstance(messages, list) or not messages
            or any(not isinstance(m, dict) or m.get('role') not in
                   {'system', 'developer', 'user', 'assistant', 'tool'}
                   for m in messages)):
        raise HistoryBindingError('Expected nonempty actual chat messages')
    return dict(schema_version=SCHEMA, profile=PROFILE,
                template_sha256=CT_TEMPLATE_SHA256, tools_sha256=None,
                messages=messages, transitions=[])


def _split_complete_channel(raw_text: str) -> tuple[str | None, str, dict[str, Any]]:
    if not isinstance(raw_text, str):
        raise UnsupportedHistory('raw_text must be the actual decoded sample')
    opens, closes = raw_text.count(_CHANNEL_OPEN), raw_text.count(_CHANNEL_CLOSE)
    if opens == closes == 0:
        return None, raw_text, dict(channel_present=False)
    if opens != 1 or closes != 1:
        raise UnsupportedHistory('Only one complete native thought channel is supported')
    start = raw_text.index(_CHANNEL_OPEN)
    end = raw_text.index(_CHANNEL_CLOSE)
    if end < start or raw_text[:start].strip():
        raise UnsupportedHistory('Ambiguous thought/channel content boundary')
    reasoning = raw_text[start + len(_CHANNEL_OPEN):end]
    if reasoning.startswith('thought\n'):
        reasoning = reasoning[len('thought\n'):]
    stop = end + len(_CHANNEL_CLOSE)
    return reasoning, raw_text[stop:], dict(
        channel_present=True, raw_channel_span=[start, stop],
        leading_whitespace_sha256=_sha_text(raw_text[:start]))


def _assistant_fields(reasoning: str | None, content: str | None) -> dict[str, Any]:
    result: dict[str, Any] = dict(role='assistant', content=content)
    if reasoning:
        # ADK LiteLlm emits reasoning_content; adk_submission mirrors reasoning.
        result.update(reasoning_content=reasoning, reasoning=reasoning)
    return result


def _render(history: dict[str, Any], tokenizer: Any,
            tools: list[dict[str, Any]]) -> tuple[list[int], dict[str, Any]]:
    tools_copy = _copy_json(tools, binding=True)
    if not isinstance(tools_copy, list):
        raise HistoryBindingError('Expected the actual tool declaration list')
    tools_hash = _sha_json(tools_copy)
    if history.get('tools_sha256') not in (None, tools_hash):
        raise HistoryBindingError('Tools changed within the persisted history')
    try:
        template = tokenizer.get_chat_template(tools=tools_copy)
    except Exception as exc:
        raise HistoryBindingError('Unable to resolve the bound CT template') from exc
    if not isinstance(template, str) or _sha_text(template) != CT_TEMPLATE_SHA256:
        raise HistoryBindingError('canonical-ct-v1 requires the pinned CT template')
    try:
        rendered = tokenizer.apply_chat_template(
            _copy_json(history['messages'], binding=True), tools=tools_copy,
            chat_template=template, tokenize=False, add_generation_prompt=True,
            enable_thinking=True)
        if not isinstance(rendered, str):
            raise HistoryBindingError('Expected a rendered CT prefix string')
        # Encoding the new prefix does not re-encode or replace behavior tokens.
        ids = tokenizer.encode(rendered, add_special_tokens=False)
    except HistoryBindingError:
        raise
    except Exception as exc:
        raise HistoryBindingError('Unable to render/encode the bound CT history') from exc
    if (not isinstance(ids, list) or not ids
            or any(type(token) is not int or token < 0 for token in ids)):
        raise HistoryBindingError('Expected flat canonical prefix token IDs')
    return ids, dict(profile=PROFILE, template_sha256=CT_TEMPLATE_SHA256,
        tools_sha256=tools_hash, rendered_prefix_sha256=_sha_text(rendered),
        prompt_token_ids_sha256=_sha_json(ids), prompt_tokens=len(ids),
        template_kwargs=dict(add_generation_prompt=True, enable_thinking=True),
        behavior_tokens_or_logprobs_modified=False,
        prefix_is_new_conditioning_not_behavior=True)


def render_prefix(history: dict[str, Any], tokenizer: Any,
                  tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Render/bind a copied history, including the actual initial generation prefix."""
    result = _history_copy(history)
    before_hash = _sha_json(result)
    ids, metadata = _render(result, tokenizer, tools)
    result['tools_sha256'] = metadata['tools_sha256']
    metadata['history_before_sha256'] = before_hash
    metadata['history_after_sha256'] = _sha_json(result)
    return dict(history=result, prompt_token_ids=ids, metadata=metadata)


def _advance(history: dict[str, Any], additions: list[dict[str, Any]],
             transition: dict[str, Any], tokenizer: Any,
             tools: list[dict[str, Any]]) -> dict[str, Any]:
    result = _history_copy(history)
    before_hash = _sha_json(result)
    before_messages_hash = _sha_json(result['messages'])
    result['messages'].extend(_copy_json(additions, binding=True))
    ids, rendered = _render(result, tokenizer, tools)
    result['tools_sha256'] = rendered['tools_sha256']
    record = dict(transition, transition_index=len(result['transitions']),
        history_before_sha256=before_hash,
        messages_before_sha256=before_messages_hash,
        messages_after_sha256=_sha_json(result['messages']),
        appended_messages_sha256=_sha_json(additions), **rendered)
    record['transition_sha256'] = _sha_json(record)
    result['transitions'].append(record)
    # The after-history digest is outside its own persisted record, avoiding a
    # circular hash. The record binds after-messages and the rendered prefix.
    metadata = dict(record, history_after_sha256=_sha_json(result))
    return dict(history=result, prompt_token_ids=ids, metadata=metadata)


def append_tool_result(history: dict[str, Any], *, raw_text: str,
                       parsed_call: tuple[str, dict[str, Any]], response: Any,
                       call_id: str, tokenizer: Any,
                       tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Append one caller-strict-parsed call and its actual observed response.

    The caller must supply its strict NativeArguments result for this SAME raw
    sample. The helper checks the raw matched name and records the caller's args
    hash; it trusts that argument binding and is not a second argument parser.
    Parsed arguments do not establish successful tool execution.
    """
    current = _history_copy(history)
    if (not isinstance(parsed_call, (tuple, list)) or len(parsed_call) != 2
            or not isinstance(parsed_call[0], str)
            or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', parsed_call[0])
            or not isinstance(parsed_call[1], dict)):
        raise HistoryBindingError('Expected the strict single-call (name, arguments) result')
    if not isinstance(call_id, str) or not call_id:
        raise HistoryBindingError('An actual collector call ID is required')
    if any(call.get('id') == call_id for message in current['messages']
           for call in message.get('tool_calls') or []):
        raise HistoryBindingError('Duplicate history tool-call ID')
    name, arguments = parsed_call[0], _copy_json(parsed_call[1], binding=True)
    reasoning, visible, channel = _split_complete_channel(raw_text)
    if visible.count(_CALL_OPEN) != 1 or visible.count(_CALL_CLOSE) != 1:
        raise UnsupportedHistory('Only one complete native tool call is supported')
    start = visible.index(_CALL_OPEN)
    end = visible.index(_CALL_CLOSE)
    if end < start or visible[end + len(_CALL_CLOSE):].strip():
        raise UnsupportedHistory('Ambiguous or unclosed native tool-call boundary')
    body = visible[start + len(_CALL_OPEN):end]
    matched = re.match(r'call:([A-Za-z_][A-Za-z_0-9]*)\s*\{', body)
    if not matched or not body.rstrip().endswith('}'):
        raise UnsupportedHistory('Unsupported native tool-call shape')
    if matched.group(1) != name:
        raise HistoryBindingError('Raw matched name differs from caller strict name')
    content = visible[:start].strip() or None
    assistant = _assistant_fields(reasoning, content)
    assistant['tool_calls'] = [dict(id=call_id, type='function',
        function=dict(name=name, arguments=arguments))]
    actual_response = _copy_json(response)
    # Same boundary as ADK's response serializer: strings stay strings.
    response_text = (actual_response if isinstance(actual_response, str)
                     else json.dumps(actual_response, ensure_ascii=False, allow_nan=False))
    tool_message = dict(role='tool', tool_call_id=call_id, name=name, content=response_text)
    transition = dict(kind='actual_single_tool_result', raw_text_sha256=_sha_text(raw_text),
        raw_text_utf8_bytes=len(raw_text.encode('utf-8')), parsed_tool_name=name,
        parsed_arguments_sha256=_sha_json(arguments), call_id=call_id,
        actual_response_sha256=_sha_json(actual_response),
        response_message_content_sha256=_sha_text(response_text),
        response_serialization='verbatim-string' if isinstance(actual_response, str)
                               else 'json.dumps ensure_ascii=False',
        reasoning_sha256=_sha_text(reasoning) if reasoning is not None else None,
        visible_content_sha256=_sha_text(content) if content is not None else None,
        native_call_span_in_visible=[start, end + len(_CALL_CLOSE)],
        split=dict(channel, semantics='complete-channel/pre-call-content subset',
                   upstream_commit=SOURCE_COMMIT),
        parsed_arguments_source='caller strict NativeArguments; not tolerant recovery')
    return _advance(current, [assistant, tool_message], transition, tokenizer, tools)


def append_user_nudge(history: dict[str, Any], *, text: str, tokenizer: Any,
                      tools: list[dict[str, Any]],
                      preceding_assistant_raw_text: str | None = None) -> dict[str, Any]:
    """Append an actual user nudge, optionally recording its actual no-call reply.

    Supply preceding_assistant_raw_text whenever that response is not already in
    history. It must contain no call and no incomplete thought channel. Omit it
    only when no assistant output is being omitted by the caller. Structural
    completeness does not infer a generation finish reason or available budget.
    """
    if not isinstance(text, str) or not text:
        raise HistoryBindingError('The actual nonempty user nudge text is required')
    additions = []
    detail: dict[str, Any] = {}
    if preceding_assistant_raw_text is not None:
        reasoning, visible, channel = _split_complete_channel(preceding_assistant_raw_text)
        if _CALL_OPEN in visible or _CALL_CLOSE in visible:
            raise UnsupportedHistory('A no-call nudge path cannot repair/omit native calls')
        terminal = None
        for marker in _TURN_ENDS:
            if visible.endswith(marker):
                visible, terminal = visible[:-len(marker)], marker
                break
        if any(marker in visible for marker in _TURN_ENDS) or not visible.strip():
            raise UnsupportedHistory('Expected one actual visible no-call assistant reply')
        additions.append(_assistant_fields(reasoning, visible))
        detail = dict(preceding_assistant_raw_text_sha256=_sha_text(preceding_assistant_raw_text),
            preceding_assistant_content_sha256=_sha_text(visible),
            preceding_assistant_reasoning_sha256=_sha_text(reasoning) if reasoning is not None else None,
            removed_terminal_control=terminal, preceding_split=channel)
    additions.append(dict(role='user', content=text))
    transition = dict(kind='actual_user_nudge', actual_user_text_sha256=_sha_text(text),
        preceding_assistant_recorded=preceding_assistant_raw_text is not None, **detail)
    return _advance(history, additions, transition, tokenizer, tools)
