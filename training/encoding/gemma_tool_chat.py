#!/usr/bin/env python3
"""Complete-transaction Gemma 4 tool-chat encoding; no model-weight downloads.

Real mode requires AutoTokenizer and the supplied deployment chat template.
--synthetic is an explicit byte-level smoke fixture, never training eligible.
Masking probes the SAME WHOLE conversation template using unique sentinels, then
verifies unchanged context and maps character spans with fast-tokenizer offsets.
No isolated-message rendering, clipping JSON, or splitting dependency context.

Official sources inspected 2026-10-10:
https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
https://huggingface.co/google/gemma-4-31B-it/blob/340d7b5d066f598ed877c2feebadcd4ba5242956/chat_template.jinja
These document the format; encoding always uses the caller's deployed revision.
"""
import argparse
from collections.abc import Mapping
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

LOSS_IGNORE = -100
ENCODING_VERSION = "p005-full-transaction-v2"
CALL_OPEN, CALL_CLOSE = "<|tool_call>", "<tool_call|>"


class EncodingError(ValueError):
    pass


class OversizeTransaction(EncodingError):
    """Whole-transaction reject. token_length is the untruncated template length."""

    def __init__(self, token_length, max_seq_len):
        self.token_length = int(token_length)
        self.max_seq_len = int(max_seq_len)
        super().__init__("%d tokens exceeds max_seq_len=%d; whole transaction rejected" %
                         (self.token_length, self.max_seq_len))


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _provenance_fields(value, allowed):
    """Keep identifiers only; never copy trace bodies, code or offline labels."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise EncodingError("group/source provenance must be an object")
    result = {}
    for key in allowed:
        if key in value:
            item = value[key]
            if not (item is None or isinstance(item, (str, int, float, bool))):
                raise EncodingError("provenance %s must be a scalar identifier" % key)
            result[key] = item
    return result


def tokenizer_fingerprint(tokenizer):
    """Shared trainer hash: ignores paths and mutable padding/truncation state."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        state = json.loads(backend.to_str())
        state.pop("padding", None)
        state.pop("truncation", None)
    else:
        state = {"vocab": tokenizer.get_vocab()}
    special = {k: [str(x) for x in v] if isinstance(v, list) else str(v)
               for k, v in getattr(tokenizer, "special_tokens_map", {}).items()}
    return sha256_json({"class": type(tokenizer).__name__, "backend": state,
                        "special_tokens_map": special})


def normalize_tools(raw):
    protocol_id = raw.get("protocol_id") if isinstance(raw, dict) else None
    tools = raw.get("tools") if isinstance(raw, dict) else raw
    if not isinstance(tools, list):
        raise EncodingError("tool schema must be an array or {tools: [...]} manifest")
    seen = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise EncodingError("tools require OpenAI {type: function, function: ...} schema")
        fun = tool.get("function", {})
        name = fun.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise EncodingError("tool names must be nonempty and unique")
        if not isinstance(fun.get("parameters"), dict):
            raise EncodingError("tool parameters must be a JSON Schema object: " + name)
        seen.add(name)
    return copy.deepcopy(tools), protocol_id


def _content(value, where):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(
            isinstance(p, dict) and p.get("type") == "text"
            and isinstance(p.get("text"), str) for p in value):
        return copy.deepcopy(value)
    raise EncodingError(where + ": text only; no silent multimodal drop")


def _arguments(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise EncodingError("tool arguments are not complete JSON: " + str(e))
    if not isinstance(value, dict):
        raise EncodingError("tool arguments must decode to an object")
    return copy.deepcopy(value)


def normalize_messages(raw, tools, content_format="preserve"):
    """Preserve tool semantics. Return messages and original assistant ordinals."""
    if not isinstance(raw, list) or not raw:
        raise EncodingError("messages must be a nonempty list")
    allowed = {t["function"]["name"] for t in tools}
    messages, ordinals, pending, used_ids = [], {}, {}, set()
    ordinal = 0
    for index, src in enumerate(raw):
        if not isinstance(src, dict):
            raise EncodingError("every message must be an object")
        role = src.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise EncodingError("unsupported role: " + str(role))
        if pending and role != "tool":
            raise EncodingError("missing tool result before message %d" % index)
        msg = {"role": role}
        if role != "tool":
            msg["content"] = _content(src.get("content", ""), role)
        if role == "assistant":
            ordinals[len(messages)] = ordinal
            ordinal += 1
            reasoning = src.get("reasoning_content", src.get("reasoning"))
            if src.get("reasoning") is not None and src.get("reasoning_content") is not None:
                if src["reasoning"] != src["reasoning_content"]:
                    raise EncodingError("conflicting reasoning fields")
            if reasoning is not None:
                if not isinstance(reasoning, str):
                    raise EncodingError("reasoning_content must be a string")
                msg["reasoning_content"] = reasoning
            calls = src.get("tool_calls")
            if calls and src.get("tool_name"):
                raise EncodingError("both flat and standard tool calls supplied")
            if src.get("tool_name"):
                calls = [{"id": src.get("tool_call_id"), "type": "function",
                          "function": {"name": src["tool_name"],
                                       "arguments": src.get("arguments")}}]
            if calls:
                if not isinstance(calls, list):
                    raise EncodingError("tool_calls must be an array")
                msg["tool_calls"] = []
                for ci, call in enumerate(calls):
                    fun = call.get("function", {})
                    name = fun.get("name")
                    if name not in allowed:
                        raise EncodingError("tool absent from supplied schema: " + str(name))
                    call_id = call.get("id") or "p005-%d-%d" % (index, ci)
                    if not isinstance(call_id, str) or call_id in used_ids:
                        raise EncodingError("tool call IDs must be unique strings")
                    used_ids.add(call_id)
                    pending[call_id] = name
                    msg["tool_calls"].append({"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": _arguments(fun.get("arguments"))}})
            if reasoning and not calls:
                raise EncodingError("reasoning-only/non-tool reasoning: join original thinking "
                                    "event to its tool-call turn; Gemma template would drop it")
            if src.get("tool_responses"):
                raise EncodingError("embedded tool_responses require explicit role:tool messages")
        elif role == "tool":
            call_id = src.get("tool_call_id")
            name = src.get("name") or src.get("tool_name")
            if call_id is None:
                matches = [k for k, v in pending.items() if not name or name == v]
                if len(matches) != 1:
                    raise EncodingError("ambiguous tool response")
                call_id = matches[0]
            if call_id not in pending:
                raise EncodingError("orphan or duplicate tool response: " + str(call_id))
            expected = pending.pop(call_id)
            if name and name != expected:
                raise EncodingError("tool response name disagrees with its call")
            msg.update(name=expected, tool_call_id=call_id)
            if "tool_response" in src:
                response = src["tool_response"]
                if src.get("content") not in (None, "", response):
                    raise EncodingError("conflicting content and tool_response")
                msg["content"] = copy.deepcopy(response)
            else:
                msg["content"] = _content(src.get("content", ""), "tool")
        if content_format == "text-parts" and isinstance(msg.get("content"), str):
            msg["content"] = [{"type": "text", "text": msg["content"]}]
        messages.append(msg)
    if pending:
        raise EncodingError("incomplete transaction: missing tool responses")
    if not any(m["role"] == "user" for m in messages):
        raise EncodingError("transaction needs user context, not an orphan fragment")
    return messages, ordinals


class SyntheticTokenizer:
    """Deterministic byte smoke tokenizer; never eligible for real training."""
    name_or_path = "synthetic-byte-tokenizer-NOT-FOR-TRAINING"
    special_tokens_map = {}
    chat_template = "p005-synthetic-whole-chat-v2"
    is_fast = True

    def get_vocab(self):
        return {"byte-%03d" % i: i + 1 for i in range(256)}

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        ids, offsets = [], []
        for i, char in enumerate(text):
            for byte in char.encode("utf-8"):
                ids.append(byte + 1)
                offsets.append((i, i + 1))
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    def apply_chat_template(self, messages, tools=None, tokenize=False,
                            add_generation_prompt=False, **kwargs):
        parts = ["[SYNTHETIC NOT FOR TRAINING]\n", canonical_json(tools or [])]
        for msg in messages:
            parts.append("\n[" + msg["role"].upper() + "]")
            if msg.get("reasoning_content"):
                parts.append("<|channel>thought\n" + msg["reasoning_content"] + "<channel|>")
            for call in msg.get("tool_calls", []):
                fun = call["function"]
                parts.append(CALL_OPEN + "call:" + fun["name"] +
                             canonical_json(fun["arguments"]) + CALL_CLOSE)
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "".join(p["text"] for p in content)
            elif not isinstance(content, str):
                content = canonical_json(content)
            parts.extend((content, "[END]"))
        if add_generation_prompt:
            parts.append("\n[ASSISTANT]")
        text = "".join(parts)
        return self(text)["input_ids"] if tokenize else text


def _ids(value):
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or any(type(i) is not int for i in value):
        raise EncodingError("expected one unbatched integer input_ids list")
    return value


def _span_from_probe(full, probe, marker, action=False):
    if probe.count(marker) != 1:
        raise EncodingError("template dropped or duplicated assistant field")
    pos = probe.index(marker)
    start, end = pos, pos + len(marker)
    if action:
        start = probe.rfind(CALL_OPEN, 0, pos)
        end = probe.find(CALL_CLOSE, pos)
        if start < 0 or end < 0:
            raise EncodingError("template lacks Gemma 4 action delimiters")
        end += len(CALL_CLOSE)
    prefix, suffix = probe[:start], probe[end:]
    if not full.startswith(prefix) or not full.endswith(suffix):
        raise EncodingError("template changed context during probe; ambiguous label boundary")
    stop = len(full) - len(suffix) if suffix else len(full)
    if stop < len(prefix):
        raise EncodingError("overlapping probe context")
    return len(prefix), stop


def _field_probes(messages, selected, ordinals, thought_loss):
    for mi, ordinal in ordinals.items():
        msg = messages[mi]
        if msg.get("reasoning_content"):
            # Masked thoughts also get a preservation check.
            yield mi, ordinal, "reasoning_content", None, (
                ordinal in selected and thought_loss == "include")
        if ordinal not in selected:
            continue
        for ci, _ in enumerate(msg.get("tool_calls", [])):
            yield mi, ordinal, "tool_calls", ci, True
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            yield mi, ordinal, "content", None, True
        elif isinstance(content, list):
            for ci, part in enumerate(content):
                if part["text"].strip():
                    yield mi, ordinal, "content", ci, True


def _render_transaction(tx, tokenizer, tools, *, content_format="preserve",
                        enable_thinking=True, preserve_thinking=True,
                        chat_template=None, tool_protocol_id=None):
    """Normalize and render the full chat template. No clipping."""
    messages, ordinals = normalize_messages(tx.get("messages"), tools, content_format)
    protocol_id = tx.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise EncodingError("transaction protocol_id is required")
    if tool_protocol_id and protocol_id != tool_protocol_id:
        raise EncodingError("transaction and tools protocol_id differ")
    template = chat_template
    if template is None:
        template = tokenizer.get_chat_template(tools=tools) if hasattr(
            tokenizer, "get_chat_template") else getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise EncodingError("no unambiguous deployment chat template")
    kwargs = dict(tools=tools, chat_template=template,
                  enable_thinking=bool(enable_thinking),
                  preserve_thinking=bool(preserve_thinking), add_generation_prompt=False)

    def render(ms, tokenize=False):
        return tokenizer.apply_chat_template(ms, tokenize=tokenize, **kwargs)

    full = render(messages)
    if not isinstance(full, str):
        raise EncodingError("template must render text")
    template_ids = _ids(render(messages, True))
    return dict(messages=messages, ordinals=ordinals, protocol_id=protocol_id,
                template=template, render=render, full=full, template_ids=template_ids)


def count_template_tokens(tx, tokenizer, tools, **kwargs):
    """Raw full-template token count before any max_seq_len reject."""
    return len(_render_transaction(tx, tokenizer, tools, **kwargs)["template_ids"])


def encode_transaction(tx, tokenizer, tools, max_seq_len=4096, *,
                       enable_thinking=True, preserve_thinking=True,
                       thought_loss="mask", content_format="preserve",
                       tokenizer_revision=None, synthetic=False,
                       chat_template=None, tool_protocol_id=None,
                       tokenizer_hash=None):
    """One record per entire transaction; any unsafe boundary is an error."""
    if max_seq_len < 2 or thought_loss not in ("mask", "include"):
        raise EncodingError("invalid sequence limit or thought-loss policy")
    rendered = _render_transaction(
        tx, tokenizer, tools, content_format=content_format,
        enable_thinking=enable_thinking, preserve_thinking=preserve_thinking,
        chat_template=chat_template, tool_protocol_id=tool_protocol_id)
    messages, ordinals = rendered["messages"], rendered["ordinals"]
    targets = tx.get("target_assistant_steps", list(range(len(ordinals))))
    if not isinstance(targets, list) or any(
            type(n) is not int or n < 0 or n >= len(ordinals) for n in targets):
        raise EncodingError("target_assistant_steps needs zero-based assistant ordinals")
    selected = set(targets)
    if not selected:
        raise EncodingError("no selected assistant steps")
    protocol_id = rendered["protocol_id"]
    template = rendered["template"]
    render = rendered["render"]
    full = rendered["full"]
    template_ids = rendered["template_ids"]
    if len(template_ids) > max_seq_len:
        raise OversizeTransaction(len(template_ids), max_seq_len)
    tokenized = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = _ids(tokenized)
    if input_ids != template_ids:
        raise EncodingError("full-render tokens differ from apply_chat_template tokens")
    offsets = tokenized.get("offset_mapping")
    if offsets is None or len(offsets) != len(input_ids):
        raise EncodingError("fast tokenizer with exact offsets required")
    if any(len(o) != 2 or not 0 <= o[0] <= o[1] <= len(full) for o in offsets):
        raise EncodingError("invalid token offsets")
    spans = []
    for pi, (mi, ordinal, field, ci, supervise) in enumerate(
            _field_probes(messages, selected, ordinals, thought_loss)):
        marker = "P005_SENTINEL_" + sha256_json([full, mi, field, ci, pi])
        if marker in full:
            raise EncodingError("sentinel collision")
        changed = copy.deepcopy(messages)
        if field == "tool_calls":
            changed[mi][field][ci]["function"]["arguments"] = {"p005_probe": marker}
        elif ci is not None:
            changed[mi][field][ci]["text"] = marker
        else:
            changed[mi][field] = marker
        begin, end = _span_from_probe(full, render(changed), marker, field == "tool_calls")
        if field == "reasoning_content" and full[begin:end] != messages[mi][field]:
            raise EncodingError("template transformed original reasoning_content")
        if supervise and end > begin:
            spans.append(dict(start=begin, end=end, assistant_ordinal=ordinal, field=field))
    labels = [LOSS_IGNORE] * len(input_ids)
    for span in spans:
        count = 0
        for ti, (start, end) in enumerate(offsets):
            # Never train a token partly containing a header, observation or prefix.
            if ti > 0 and end > start and start >= span["start"] and end <= span["end"]:
                labels[ti] = input_ids[ti]
                count += 1
        if not count:
            raise EncodingError("assistant field has no safely aligned target tokens")
    supervised = sum(x != LOSS_IGNORE for x in labels[1:])
    if not supervised:
        raise EncodingError("no supervised causal-LM target tokens")
    data_synthetic = bool(tx.get("synthetic") or tx.get("origin_kind") == "synthetic"
                          or str((tx.get("source") or {}).get("uri", "")).startswith("synthetic:"))
    is_synthetic = bool(synthetic or isinstance(tokenizer, SyntheticTokenizer) or data_synthetic)
    tools_hash = sha256_json(tools)
    thought_config = dict(enable_thinking=bool(enable_thinking),
                          preserve_thinking=bool(preserve_thinking), thought_loss=thought_loss)
    protocol = dict(encoding_version=ENCODING_VERSION, protocol_id=protocol_id,
                    tools_hash=tools_hash, thought_config=thought_config,
                    content_format=content_format,
                    mask_policy="selected-content-and-tool-actions; no-role-prefix; no-tool-results")
    return dict(input_ids=input_ids, attention_mask=[1] * len(input_ids), labels=labels,
                meta=dict(transaction_id=tx.get("transaction_id", tx.get("sample_id")),
                    sample_id=tx.get("sample_id"), origin_kind=tx.get("origin_kind"),
                    group=_provenance_fields(tx.get("group"), ("issue", "repo", "commit", "split")),
                    source=_provenance_fields(tx.get("source"), (
                        "uri", "sha256", "hash", "license", "revision", "format", "source_id",
                        "trace_id", "trace_sha256", "raw_sha256", "fingerprint", "dataset", "dataset_revision")),
                    behavior=tx.get("behavior"), encoding_version=ENCODING_VERSION,
                    synthetic=is_synthetic, training_eligible=not is_synthetic,
                    tokenizer_name=getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
                    tokenizer_revision=tokenizer_revision,
                    tokenizer_hash=tokenizer_hash or tokenizer_fingerprint(tokenizer),
                    chat_template_hash=hashlib.sha256(template.encode("utf-8")).hexdigest(),
                    tools_hash=tools_hash, tool_schema_hash=tools_hash,
                    protocol_id=protocol_id, protocol_hash=sha256_json(protocol),
                    thought_config=thought_config, content_format=content_format,
                    max_seq_len=max_seq_len, sequence_length=len(input_ids),
                    raw_token_length=len(input_ids),
                    supervised_tokens=supervised, target_assistant_steps=sorted(selected),
                    truncated=False, alignment="whole-template-differential-spans-and-token-offsets",
                    supervised_spans=spans))


def load_tokenizer(name, revision=None, allow_download=False):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise EncodingError("real mode requires transformers; no synthetic fallback") from e
    try:
        tok = AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True,
            local_files_only=not allow_download, trust_remote_code=False)
    except Exception as e:
        raise EncodingError("real tokenizer load failed; no placeholder fallback: " + str(e)) from e
    if not getattr(tok, "is_fast", False):
        raise EncodingError("fast tokenizer required for verified offsets")
    return tok


def cmd_encode(args):
    if args.synthetic and args.tokenizer:
        raise EncodingError("--synthetic and --tokenizer are mutually exclusive")
    if not args.synthetic and not args.tokenizer:
        raise EncodingError("--tokenizer required; smoke needs explicit --synthetic")
    if args.allow_download and not args.revision:
        raise EncodingError("--allow-download requires --revision")
    output = Path(args.output)
    report_path = Path(args.report or (str(output) + ".report.json"))
    paths = [Path(args.input).resolve(), output.resolve(), report_path.resolve()]
    if len(set(paths)) != 3:
        raise EncodingError("input, output and report paths must differ")
    tools, tool_protocol = normalize_tools(json.loads(Path(args.tools).read_text(encoding="utf-8")))
    tok = SyntheticTokenizer() if args.synthetic else load_tokenizer(
        args.tokenizer, args.revision, args.allow_download)
    template = Path(args.chat_template).read_text(encoding="utf-8") if args.chat_template else None
    tok_hash = tokenizer_fingerprint(tok)
    report = dict(encoding_version=ENCODING_VERSION, transactions=0, written=0,
                  skipped_oversize=[], synthetic=bool(args.synthetic), supervised_tokens=0,
                  tokenizer_hash=tok_hash, ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=str(output.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fout, open(args.input, encoding="utf-8") as fin:
            for line_number, line in enumerate(fin, 1):
                if not line.strip():
                    continue
                report["transactions"] += 1
                tx = json.loads(line)
                try:
                    record = encode_transaction(tx, tok, tools, args.max_seq_len,
                        enable_thinking=args.thinking == "on",
                        preserve_thinking=args.preserve_thinking == "true",
                        thought_loss=args.thought_loss, content_format=args.content_format,
                        tokenizer_revision=args.revision, synthetic=args.synthetic,
                        chat_template=template, tool_protocol_id=tool_protocol,
                        tokenizer_hash=tok_hash)
                except OversizeTransaction as e:
                    if args.on_oversize != "skip":
                        raise
                    report["skipped_oversize"].append(dict(
                        line=line_number,
                        transaction_id=tx.get("transaction_id", tx.get("sample_id")),
                        reason=str(e), raw_token_length=e.token_length,
                        truncated=False, clipped_json=False))
                    continue
                fout.write(canonical_json(record) + "\n")
                report["written"] += 1
                report["supervised_tokens"] += record["meta"]["supervised_tokens"]
                report["synthetic"] = report["synthetic"] or record["meta"]["synthetic"]
        if not report["written"]:
            raise EncodingError("no records encoded; refusing empty output")
        os.replace(temporary, output)
        report["ok"] = True
    except Exception as e:
        report["error"] = str(e)
        raise
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(canonical_json(report))
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("encode")
    e.add_argument("--input", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--tokenizer", help="deployment tokenizer local path or HF model ID")
    e.add_argument("--revision", help="deployment tokenizer revision; commit preferred")
    e.add_argument("--tools", required=True, help="OpenAI tools JSON array or protocol manifest")
    e.add_argument("--chat-template", help="exact deployed Jinja override")
    e.add_argument("--thinking", choices=("on", "off"), default="on")
    e.add_argument("--preserve-thinking", choices=("true", "false"), default="true")
    e.add_argument("--thought-loss", choices=("mask", "include"), default="mask")
    e.add_argument("--content-format", choices=("preserve", "text-parts"), default="preserve")
    e.add_argument("--max-seq-len", type=int, default=4096)
    e.add_argument("--on-oversize", choices=("error", "skip"), default="error")
    e.add_argument("--report", help="defaults to OUTPUT.report.json")
    e.add_argument("--allow-download", action="store_true", help="tokenizer assets only; default local")
    e.add_argument("--synthetic", action="store_true", help="smoke placeholder; rejected for real training")
    e.set_defaults(func=cmd_encode)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (EncodingError, OSError, ValueError, TypeError) as e:
        print(canonical_json(dict(ok=False, error=str(e))), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
