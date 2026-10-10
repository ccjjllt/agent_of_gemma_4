#!/usr/bin/env python3
"""Normalize native/ATIF/OpenAI traces without renaming tools or losing observations."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

PROTOCOL_ID = "p005-harness-tools-v1"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "harness_tools.json"


def load_tools(path=SCHEMA_PATH):
    tools = json.loads(Path(path).read_text())
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools must be a nonempty OpenAI function list")
    return tools


def tool_schema_hash(tools=None):
    return hashlib.sha256(json.dumps(load_tools() if tools is None else tools,
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def map_tool_name(raw_name):
    """No aliases: read_file is not inspect; old-string edit is not span apply."""
    return raw_name, False, raw_name in {t["function"]["name"] for t in load_tools()}


def arguments_object(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be an object")
    return value


def compatible_call(name, args, schemas):
    schema = schemas.get(name)
    if schema is None:
        return False
    properties = schema.get("properties", {})
    if any(k not in args for k in schema.get("required", [])) or any(k not in properties for k in args):
        return False
    for k, value in args.items():
        kind = properties[k].get("type")
        if isinstance(kind, list):
            if value is None and "null" in kind:
                continue
            kind = next((t for t in kind if t != "null"), None)
        expected = {"string": str, "integer": int, "boolean": bool, "object": dict, "array": list}.get(kind)
        if expected and (not isinstance(value, expected) or (kind == "integer" and isinstance(value, bool))):
            return False
    return True


def normalize_messages(turns, tools):
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}
    messages, pending, incompatible, call_ids = [], {}, [], set()
    for i, turn in enumerate(turns):
        role = turn.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"message {i}: missing/unsupported role {role!r}")
        if pending and role != "tool":
            raise ValueError(f"message {i}: missing tool replies for {list(pending)}")
        msg = {"role": role, "content": turn.get("content") or ""}
        if not isinstance(msg["content"], str):
            raise ValueError("text-only training requires string content")
        for key in ("reasoning_content", "source_step_id"):
            if key in turn:
                msg[key] = turn[key]
        if role == "assistant":
            raw_calls = turn.get("tool_calls") or []
            if turn.get("tool_name"):
                raw_calls = [{"id": turn.get("tool_call_id"), "function": {
                    "name": turn["tool_name"], "arguments": turn.get("arguments", turn.get("params", {}))}}]
            calls = []
            for j, raw in enumerate(raw_calls):
                function = raw.get("function", raw)
                name = function.get("name") or raw.get("function_name")
                cid = raw.get("id") or raw.get("tool_call_id") or f"call_{i}_{j}"
                if not name or cid in call_ids:
                    raise ValueError(f"missing name or duplicate call id {cid!r}")
                args = arguments_object(function.get("arguments", raw.get("arguments", {})))
                if not compatible_call(name, args, schemas):
                    incompatible.append({"tool_call_id": cid, "name": name,
                        "requires_replay": name not in schemas,
                        "reason": "unknown tool/incompatible arguments; replay required"})
                call_ids.add(cid)
                pending[cid] = name
                calls.append({"id": cid, "type": "function", "function": {"name": name, "arguments": args}})
            if calls:
                msg["tool_calls"] = calls
        elif role == "tool":
            cid = turn.get("tool_call_id")
            if cid is None and len(pending) == 1:
                cid = next(iter(pending))
            if cid not in pending:
                raise ValueError(f"orphan/ambiguous tool response at {i}")
            name = pending.pop(cid)
            supplied = turn.get("name") or turn.get("tool_name")
            if supplied and supplied != name:
                raise ValueError(f"tool response name mismatch for {cid}")
            response = turn.get("tool_response", turn.get("content", ""))
            msg.update(tool_call_id=cid, name=name, status=turn.get("status", "unknown"),
                content=response if isinstance(response, str) else json.dumps(response, ensure_ascii=False))
        messages.append(msg)
    if pending:
        raise ValueError(f"trace ends without tool replies: {list(pending)}")
    if not any(m["role"] == "user" for m in messages):
        raise ValueError("trace lacks visible task prefix")
    return messages, incompatible


def atif_messages(trace):
    turns = []
    for step in trace.get("steps", []):
        role = {"agent": "assistant"}.get(step.get("source"), step.get("source"))
        msg = {"role": role, "content": step.get("message") or "", "source_step_id": step.get("step_id")}
        calls = step.get("tool_calls", [])
        if role == "assistant" and step.get("extra", {}).get("event_type") == "thinking":
            msg["reasoning_content"], msg["content"] = msg["content"], ""
        if calls:
            msg["tool_calls"] = [{"id": c.get("tool_call_id"), "function": {
                "name": c.get("function_name"), "arguments": c.get("arguments", {})}} for c in calls]
        turns.append(msg)
        observation = step.get("observation")
        if calls:
            if not observation or len(calls) != 1:
                raise ValueError("ATIF requires one explicitly correlated observation per call")
            content, status = observation.get("content", ""), "unknown"
            try:
                parsed = json.loads(content) if isinstance(content, str) else content
                if isinstance(parsed, dict):
                    status = parsed.get("status", parsed.get("error_type", "unknown"))
            except (ValueError, TypeError):
                pass
            turns.append({"role": "tool", "tool_call_id": calls[0]["tool_call_id"],
                "tool_name": calls[0]["function_name"], "content": content, "status": status,
                "source_step_id": step.get("step_id")})
    return turns


def canonicalize_trace(trace, tools=None):
    tools = load_tools() if tools is None else tools
    raw = trace.get("trajectory", trace)
    turns = atif_messages(raw) if "steps" in raw else trace.get("raw_turns", trace.get("messages", []))
    messages, incompatible = normalize_messages(turns, tools)
    group = trace.get("group", {})
    if not all(group.get(k) for k in ("issue", "repo", "commit")):
        raise ValueError("group requires issue/repo/commit")
    count = sum(m["role"] == "assistant" for m in messages)
    selected = trace.get("target_assistant_steps", list(range(count)))
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= count for i in selected):
        raise ValueError("targets must be zero-based assistant ordinals")
    invalid_ids = {c["tool_call_id"] for c in incompatible}
    # Mechanical read/edit failures are observations for recovery, not positive
    # action targets. A failing shell repro can be intentional, so do not infer
    # that every nonzero command was a bad choice.
    for message in messages:
        if message["role"] != "tool" or message.get("name") not in ("read_file", "edit_file", "write_file"):
            continue
        try:
            response = json.loads(message["content"])
        except (ValueError, TypeError):
            response = {}
        if isinstance(response, dict) and (response.get("error_type") or
                response.get("status") in ("error", "failed", "exception")):
            invalid_ids.add(message["tool_call_id"])
    ordinal = -1
    invalid_ordinals = set()
    for message in messages:
        if message["role"] == "assistant":
            ordinal += 1
            if any(c["id"] in invalid_ids for c in message.get("tool_calls", [])):
                invalid_ordinals.add(ordinal)
    selected = [i for i in selected if i not in invalid_ordinals]
    return {"sample_id": trace.get("trace_id", trace.get("sample_id", "")),
        "origin_kind": trace.get("origin_kind", "student"), "group": group,
        "source": trace.get("source", {}), "synthetic": bool(trace.get("synthetic", False)),
        "protocol_id": trace.get("protocol_id", PROTOCOL_ID), "tool_schema_hash": tool_schema_hash(tools),
        "messages": messages, "target_assistant_steps": selected,
        "behavior": trace.get("behavior", "local_edit"),
        "observed_outcome": trace.get("observed_outcome", {}),
        "offline_labels": trace.get("offline_labels", {}),
        "protocol_incompatible_calls": incompatible,
        "masked_invalid_assistant_steps": sorted(invalid_ordinals),
        "needs_replay": any(c["requires_replay"] for c in incompatible)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd in ("protocol", "convert"):
        s = sub.add_parser(cmd)
        s.add_argument("--tools", default=str(SCHEMA_PATH))
        if cmd == "convert":
            s.add_argument("--input", required=True)
            s.add_argument("--output", required=True)
            s.add_argument("--compatible-only", action="store_true")
    args = p.parse_args(argv)
    tools = load_tools(args.tools)
    if args.cmd == "protocol":
        print(json.dumps({"protocol_id": PROTOCOL_ID, "tool_schema_hash": tool_schema_hash(tools),
                          "tools": tools, "aliases": {}}, indent=2))
        return 0
    written = skipped = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as fin, open(args.output, "w") as fout:
        for number, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                sample = canonicalize_trace(json.loads(line), tools)
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(f"line {number}: {e}") from e
            if args.compatible_only and sample["needs_replay"]:
                skipped += 1
                continue
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
    print(json.dumps({"wrote": written, "skipped_incompatible": skipped}))
    return 0 if written else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as e:
        print(f"canonicalize: {e}", file=sys.stderr)
        sys.exit(2)
