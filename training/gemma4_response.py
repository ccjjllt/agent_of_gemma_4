# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, non-streaming adapter for the official wheelhouse 31 Gemma4 parser.

Derived from vLLM v0.19.1; exact source URLs/SHA256s and modification boundaries
are in third_party/vllm_0_19_1/SOURCES.json, alongside the Apache-2.0 license.
This implements the public official auto-tool/reasoning path, not streaming,
forced tool choice, JSON repair, schema validation, or tool-name inference.

Execution-limit failures deliberately escape the upstream tool-content fallback
so Policy can retain the sampled IDs/logprobs as failed raw generation evidence.
"""

from __future__ import annotations

import json
import logging

import regex as re

logger = logging.getLogger(__name__)

TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'
_REASONING_START = "<|channel>"
_REASONING_END = "<channel|>"
_THOUGHT_PREFIX = "thought\n"
_TOOL_CALL_REGEX = re.compile(
    r"<\|tool_call>call:([\w\-\.]+)\{(.*?)\}<tool_call\|>", re.DOTALL
)

# Explicit execution envelope; exceeding it is a failure, never parsed content.
_MAX_TOKEN_IDS = 32768
_MAX_TEXT_CHARS = 1_048_576
_MAX_NESTING = 128
_MAX_SCAN_STEPS = 4_194_304
_REGEX_TIMEOUT_SECONDS = 2.0


class Gemma4ParseExecutionError(ValueError):
    """The parser could not safely finish; keep the original sampled output."""


class _ParseBudget:
    def __init__(self):
        self.remaining = _MAX_SCAN_STEPS

    def enter(self, depth):
        if depth > _MAX_NESTING:
            raise Gemma4ParseExecutionError("Gemma4 parser nesting limit exceeded")
        self.tick()

    def tick(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise Gemma4ParseExecutionError("Gemma4 parser scan budget exceeded")


def _parse_gemma4_value(value_str: str) -> object:
    """Parse a single Gemma4 value (after key:) into a Python object."""
    value_str = value_str.strip()
    if not value_str:
        return value_str

    # Boolean
    if value_str == "true":
        return True
    if value_str == "false":
        return False

    # Null
    if value_str.lower() in ("null", "none", "nil"):
        return None

    # Number (int or float)
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    # Bare string (no <|"|> delimiters — shouldn't happen but be safe)
    return value_str


def _parse_gemma4_args(args_str: str, *, partial: bool = False, _budget=None, _depth: int = 0) -> dict:
    """Parse Gemma4's custom key:value format into a Python dict.

    Format examples::

        location:<|"|>Tokyo<|"|>
        location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>
        count:42,flag:true
        nested:{inner_key:<|"|>val<|"|>}
        items:[<|"|>a<|"|>,<|"|>b<|"|>]

    Args:
        args_str: The raw Gemma4 argument string.
        partial: When True (streaming), bare values at end of string are
            omitted because they may be incomplete and type-unstable
            (e.g. partial boolean parsed as bare string).

    Returns a dict ready for ``json.dumps()``.
    """
    _budget = _budget or _ParseBudget()
    _budget.enter(_depth)
    if not args_str or not args_str.strip():
        return {}

    result: dict = {}
    i = 0
    n = len(args_str)

    while i < n:
        _budget.tick()
        # Skip whitespace and commas
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            _budget.tick()
            i += 1
        if i >= n:
            break

        # Parse key (unquoted, ends at ':')
        key_start = i
        while i < n and args_str[i] != ":":
            _budget.tick()
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1  # skip ':'

        # Parse value
        if i >= n:
            if not partial:
                result[key] = ""
            break

        # Skip whitespace after ':'
        while i < n and args_str[i] in (" ", "\n", "\t"):
            _budget.tick()
            i += 1
        if i >= n:
            if not partial:
                result[key] = ""
            break

        # String value: <|"|>...<|"|>
        if args_str.startswith(STRING_DELIM, i):
            i += len(STRING_DELIM)
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                # Unterminated string — take rest
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(STRING_DELIM)

        # Nested object: {...}
        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                _budget.tick()
                if args_str.startswith(STRING_DELIM, i):
                    # Skip over string contents to avoid counting { inside strings
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                # Incomplete nested object — use i (not i-1) to avoid
                # dropping the last char, and recurse as partial.
                result[key] = _parse_gemma4_args(args_str[obj_start:i], partial=True, _budget=_budget, _depth=_depth + 1)
            else:
                result[key] = _parse_gemma4_args(args_str[obj_start : i - 1], _budget=_budget, _depth=_depth + 1)

        # Array: [...]
        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                _budget.tick()
                if args_str.startswith(STRING_DELIM, i):
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_array(args_str[arr_start:i], partial=True, _budget=_budget, _depth=_depth + 1)
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start : i - 1], _budget=_budget, _depth=_depth + 1)

        # Bare value (number, boolean, etc.)
        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                _budget.tick()
                i += 1
            if partial and i >= n:
                # Value may be incomplete (e.g. partial boolean) —
                # withhold to avoid type instability during streaming.
                break
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False, _budget=None, _depth: int = 0) -> list:
    """Parse a Gemma4 array content string into a Python list."""
    _budget = _budget or _ParseBudget()
    _budget.enter(_depth)
    items: list = []
    i = 0
    n = len(arr_str)

    while i < n:
        _budget.tick()
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            _budget.tick()
            i += 1
        if i >= n:
            break

        # String element
        if arr_str.startswith(STRING_DELIM, i):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(STRING_DELIM)

        # Nested object
        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                _budget.tick()
                if arr_str.startswith(STRING_DELIM, i):
                    i += len(STRING_DELIM)
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_args(arr_str[obj_start:i], partial=True, _budget=_budget, _depth=_depth + 1))
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start : i - 1], _budget=_budget, _depth=_depth + 1))

        # Nested array
        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                _budget.tick()
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_array(arr_str[sub_start:i], partial=True, _budget=_budget, _depth=_depth + 1))
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start : i - 1], _budget=_budget, _depth=_depth + 1))

        # Bare value
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                _budget.tick()
                i += 1
            if partial and i >= n:
                break
            if i == val_start:
                raise Gemma4ParseExecutionError(
                    "vLLM 0.19.1 array parser would not advance at a closing bracket"
                )
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items


def _extract_reasoning(model_output):
    # Gemma4ReasoningParser.extract_reasoning, followed by its base method.
    if _REASONING_START not in model_output and _REASONING_END not in model_output:
        return None, model_output
    parts = model_output.partition(_REASONING_START)
    model_output = parts[2] if parts[1] else parts[0]
    if _REASONING_END not in model_output:
        reasoning, content = model_output, None
    else:
        reasoning, _, content = model_output.partition(_REASONING_END)
        content = content or None
    if reasoning.startswith(_THOUGHT_PREFIX):
        reasoning = reasoning[len(_THOUGHT_PREFIX):]
    return reasoning, content


def _extract_tool_calls(model_output):
    # Gemma4ToolParser.extract_tool_calls: preserve all native fallback behavior.
    if TOOL_CALL_START not in model_output:
        return [], model_output
    try:
        matches = _TOOL_CALL_REGEX.findall(model_output, timeout=_REGEX_TIMEOUT_SECONDS)
        if not matches:
            return [], model_output
        budget = _ParseBudget()
        tool_calls = []
        for func_name, args_str in matches:
            arguments = _parse_gemma4_args(args_str, _budget=budget)
            tool_calls.append({"type": "function", "function": {
                "name": func_name, "arguments": json.dumps(arguments, ensure_ascii=False),
            }})
        content_end = model_output.find(TOOL_CALL_START)
        content = model_output[:content_end].strip() if content_end > 0 else None
        return tool_calls, content if content else None
    except Gemma4ParseExecutionError:
        raise
    except (TimeoutError, RecursionError, MemoryError) as error:
        raise Gemma4ParseExecutionError(
            f"Gemma4 parser execution failed: {type(error).__name__}"
        ) from error
    except Exception:
        logger.exception("Error extracting tool calls from Gemma4 response")
        return [], model_output


def _parse_model_output(model_output, tools):
    if len(model_output) > _MAX_TEXT_CHARS:
        raise Gemma4ParseExecutionError("Gemma4 decoded output exceeds character limit")
    # Serving extracts reasoning first, then tools exclusively from content.
    thinking, content = _extract_reasoning(model_output)
    calls = []
    if tools:
        parsed_calls, parsed_content = _extract_tool_calls(content if content is not None else "")
        if parsed_calls:
            calls, content = parsed_calls, parsed_content
    return {"content": content, "thinking": thinking, "tool_calls": calls}


def _decode_completion(tokenizer, completion_ids, prompt_ids, stop_token_ids):
    """Mirror vLLM's fast detokenizer with its default output settings.

    Prompt IDs prime DecodeStream; they are never fed into the reasoning parser.
    Only the final actually sampled EOS/stop ID is excluded from display text.
    Boundary/control tokens inside the completion remain intact.
    """
    from tokenizers.decoders import DecodeStream

    if len(completion_ids) > _MAX_TOKEN_IDS or len(prompt_ids) > _MAX_TOKEN_IDS:
        raise Gemma4ParseExecutionError("Gemma4 token input exceeds the 32768-token limit")
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None:
        raise Gemma4ParseExecutionError("Gemma4 adapter requires the fast tokenizer backend")
    stops = stop_token_ids
    if stops is None:
        stops = getattr(tokenizer, "eos_token_id", None)
    if isinstance(stops, int):
        stops = (stops,)
    stops = set(stops or ())
    ids = list(completion_ids)
    if ids and ids[-1] in stops:
        ids = ids[:-1]
    stream = DecodeStream(ids=list(prompt_ids), skip_special_tokens=False)
    parts = []
    size = 0
    for token_id in ids:
        try:
            token = stream.step(backend, token_id)
        except (OverflowError, TypeError):
            # Same narrowly defined native decoder fallback as vLLM.
            logger.exception("Encountered invalid token id: %r", token_id)
            token = None
        except Exception as error:
            if not str(error).startswith("Invalid prefix encountered"):
                raise
            stream = DecodeStream(skip_special_tokens=False)
            token = stream.step(backend, token_id)
        if token:
            size += len(token)
            if size > _MAX_TEXT_CHARS:
                raise Gemma4ParseExecutionError("Gemma4 decoded output exceeds character limit")
            parts.append(token)
    return "".join(parts)


def parse_response(tokenizer, completion_ids, prompt_ids=None, tools=None, *,
                   stop_token_ids=None, prefix=None):
    """Return Policy's content/thinking/tool_calls dict for auto tool choice.

    Pass generation_config.eos_token_id as stop_token_ids: tokenizer.eos_token_id
    alone may omit Gemma4's additional end-of-turn IDs. No sampled IDs, logprobs,
    tool names, string values, or reward inputs are modified. Tool arguments are
    the exact upstream json.dumps(..., ensure_ascii=False) strings. ``prefix``
    is an alias for prompt_ids; supplying both is an error.

    tools selects the normal OpenAI default: auto when present, none otherwise.
    Tool definitions do not validate/filter/rewrite the model's arguments.
    """
    if prefix is not None:
        if prompt_ids is not None:
            raise ValueError("Provide prompt_ids or prefix, not both")
        prompt_ids = prefix
    if prompt_ids is None:
        prompt_ids = []
    vocab = tokenizer.get_vocab()
    required = [_REASONING_START, _REASONING_END, "<|turn>", TOOL_CALL_START, "<|tool_response>"]
    if any(token not in vocab for token in required):
        raise Gemma4ParseExecutionError("Required Gemma4 parser boundary tokens are missing")
    try:
        model_output = _decode_completion(tokenizer, completion_ids, prompt_ids, stop_token_ids)
        return _parse_model_output(model_output, tools)
    except Gemma4ParseExecutionError:
        raise
    except Exception as error:
        raise Gemma4ParseExecutionError(
            f"Gemma4 response decoding/parsing failed: {type(error).__name__}: {error}"
        ) from error
