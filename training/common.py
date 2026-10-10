"""Shared, dependency-free validation for the training CLI.

The shipped YAML intentionally uses only top-level scalars and string lists.
Full JSON objects are also accepted. Unknown/nested YAML is rejected, not ignored.
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


class TrainingError(ValueError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _uncomment(line):
    quote = None
    escaped = False
    for i, char in enumerate(line):
        if escaped:
            escaped = False
        elif char == "\\" and quote == '"':
            escaped = True
        elif quote:
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        elif char == "#":
            return line[:i]
    return line


def _scalar(value):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() in ("null", "~"):
        return None
    if value.startswith(('"', "'")):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise TrainingError("invalid quoted scalar: " + value) from exc
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except ValueError as exc:
            raise TrainingError("inline containers must use JSON syntax") from exc
    try:
        return int(value)
    except ValueError:
        try:
            result = float(value)
            if not math.isfinite(result):
                raise TrainingError("non-finite config number")
            return result
        except ValueError:
            return value


def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("{"):
        result = json.loads(text)
        if not isinstance(result, dict):
            raise TrainingError("config must be an object")
        return result
    result, current = {}, None
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = _uncomment(raw).rstrip()
        if not line.strip():
            continue
        if line[0].isspace():
            item = line.strip()
            if current is None or not item.startswith("- "):
                raise TrainingError("unsupported YAML indentation at line %d" % line_number)
            result[current].append(_scalar(item[2:].strip()))
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*):\s*(.*)", line)
        if not match:
            raise TrainingError("unsupported YAML at line %d" % line_number)
        key, value = match.groups()
        if key in result:
            raise TrainingError("duplicate config key: " + key)
        result[key] = _scalar(value) if value else []
        current = key if not value else None
    return result


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise TrainingError("%s must be an integer >= %d" % (name, minimum))
    return value


def require_bool(value, name):
    if type(value) is not bool:
        raise TrainingError(name + " must be true or false, not a string")
    return value


PROVENANCE_KEYS = (
    "tokenizer_hash", "chat_template_hash", "tools_hash", "protocol_id", "protocol_hash"
)


def validate_encoded_record(record, max_length, require_real=True):
    if not isinstance(record, dict):
        raise TrainingError("encoded row must be an object")
    fields = [record.get(key) for key in ("input_ids", "attention_mask", "labels")]
    if any(not isinstance(value, list) for value in fields):
        raise TrainingError("input_ids, attention_mask and labels must be integer lists")
    ids, mask, labels = fields
    if len(ids) < 2 or len(ids) > max_length or len(mask) != len(ids) or len(labels) != len(ids):
        raise TrainingError("encoded row has invalid/equal-length/maximum sequence length")
    if any(type(token) is not int or token < 0 for token in ids):
        raise TrainingError("input_ids must contain nonnegative integers")
    # Encoder emits complete, unpadded transactions. Padding is owned by collator.
    if any(type(bit) is not int or bit != 1 for bit in mask):
        raise TrainingError("encoded attention_mask must contain only 1; pre-padding is unsupported")
    if any(type(label) is not int or label not in (-100, token) for label, token in zip(labels, ids)):
        raise TrainingError("each label must be -100 or the corresponding input token")
    if not any(label != -100 for label in labels[1:]):
        raise TrainingError("row has no supervised next-token labels after causal shift")
    meta = record.get("meta")
    if not isinstance(meta, dict):
        raise TrainingError("encoded row requires meta provenance")
    if require_real:
        if meta.get("synthetic") is not False or meta.get("training_eligible") is not True:
            raise TrainingError("real training rejects synthetic/unapproved encoder output")
        group = meta.get("group") or {}
        if group.get("split") in ("select", "heldout", "heldout-excluded", "heldout_excluded"):
            raise TrainingError("selection/heldout records must not enter training")
        for key in PROVENANCE_KEYS:
            value = meta.get(key)
            if not isinstance(value, str) or not value or "placeholder" in value.lower():
                raise TrainingError("missing deployment provenance: meta." + key)
        for key in ("tokenizer_hash", "chat_template_hash", "tools_hash", "protocol_hash"):
            if not re.fullmatch(r"[0-9a-f]{64}", meta[key]):
                raise TrainingError("meta.%s must be SHA256" % key)
    if meta.get("truncated_tool_block"):
        raise TrainingError("truncated tool block cannot enter training")
    return sum(label != -100 for label in labels[1:])


def load_encoded_jsonl(path, max_length, require_real=True):
    rows, provenance, tokens = [], None, 0
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                tokens += validate_encoded_record(row, max_length, require_real)
                identity = {key: row["meta"].get(key) for key in PROVENANCE_KEYS}
                if provenance is not None and identity != provenance:
                    raise TrainingError("mixed tokenizer/template/tool protocol fingerprints")
                provenance = identity
                rows.append(row)
            except (ValueError, TypeError) as exc:
                raise TrainingError("%s:%d: %s" % (path, line_number, exc)) from exc
    if not rows:
        raise TrainingError("encoded dataset is empty")
    return rows, {"rows": len(rows), "input_tokens": sum(len(row["input_ids"]) for row in rows),
                  "supervised_tokens_after_shift": tokens, "sha256": sha256_file(path),
                  "provenance": provenance}


def pad_encoded_batch(rows, pad_token_id):
    """Right padding; observations and pad tokens never acquire loss labels."""
    integer(pad_token_id, "pad_token_id", minimum=0)
    if not rows:
        raise TrainingError("cannot collate empty batch")
    length = max(len(row["input_ids"]) for row in rows)
    result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for row in rows:
        count = length - len(row["input_ids"])
        result["input_ids"].append(list(row["input_ids"]) + [pad_token_id] * count)
        result["attention_mask"].append(list(row["attention_mask"]) + [0] * count)
        result["labels"].append(list(row["labels"]) + [-100] * count)
    return result
