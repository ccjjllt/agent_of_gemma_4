#!/usr/bin/env python3
"""Select short assistant targets while preserving their entire observed prefix."""
import argparse
import json
import sys
from pathlib import Path


def assistant_indices(messages):
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def calls_of(message):
    if message.get("tool_calls"):
        return [(c["id"], c["function"]["name"]) for c in message["tool_calls"]]
    if message.get("tool_name"):
        return [(message["tool_call_id"], message["tool_name"])]
    return []


def reply_status(reply):
    value = reply.get("content", reply.get("tool_response", ""))
    try:
        value = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        value = {}
    if not isinstance(value, dict):
        value = {}
    status = value.get("status", reply.get("status", "unknown"))
    if value.get("error_type") or status in ("error", "failed", "exception", "FileReadError", "FileEditError"):
        return "error"
    return "ok" if status in ("ok", "success", "applied") else "unknown"


def slice_transactions(sample, max_actions=6):
    selected = set(sample.get("target_assistant_steps", range(len(assistant_indices(sample["messages"])))))
    chunks, current = [], []
    for ordinal, position in enumerate(assistant_indices(sample["messages"])):
        if ordinal in selected:
            current.append(position)
        if len(current) >= max_actions:
            chunks.append((current, False))
            current = []
    warnings = []
    if len(current) >= 2:
        chunks.append((current, False))
    elif current:
        warnings.append("one-action tail omitted; earlier history never removed")
    return chunks, warnings


def build_transaction(sample, tx_num, step_ids, err_seen=False):
    messages = sample["messages"]
    if sample.get("needs_replay"):
        raise ValueError("incompatible tools require replay before training")
    if not step_ids or not any(m["role"] == "user" for m in messages[:step_ids[0]]):
        raise ValueError("selected actions lack original task prefix")
    end = step_ids[-1] + 1
    while end < len(messages) and messages[end]["role"] == "tool":
        end += 1
    prefix = messages[:end]
    positions = assistant_indices(prefix)
    targets = [positions.index(p) for p in step_ids]
    failed, recovered = {}, []
    for i, message in enumerate(prefix):
        for cid, name in calls_of(message):
            replies = [m for m in prefix[i+1:] if m.get("role") == "tool" and m.get("tool_call_id") == cid]
            if len(replies) != 1:
                raise ValueError(f"missing/ambiguous response to {cid}")
            status = reply_status(replies[0])
            # Command error->success may be an unrelated check, not a recovery.
            if name not in ("read_file", "edit_file", "write_file"):
                continue
            if status == "error":
                failed[name] = cid
            elif status == "ok" and name in failed and i in step_ids:
                recovered.append({"failed_call": failed.pop(name), "successful_call": cid})
    behavior = "tool_recovery" if recovered else sample.get("behavior", "local_edit")
    if behavior == "tool_recovery" and not recovered:
        behavior = "local_edit"
    tx = {k: sample[k] for k in ("sample_id", "origin_kind", "group", "source",
        "protocol_id", "tool_schema_hash", "synthetic", "observed_outcome") if k in sample}
    tx.update(transaction_id=f"{sample['sample_id']}-tx{tx_num:02d}",
        messages=prefix, target_assistant_steps=targets, behavior=behavior,
        missing_observation=False, recovery_evidence=recovered,
        prefix_assistant_count=targets[0], selected_action_count=len(targets))
    sidecar = {"transaction_id": tx["transaction_id"],
               "offline_labels": sample.get("offline_labels", {}),
               "note": "Offline only; never rendered in model input."}
    return tx, sidecar


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--labels-out", required=True)
    s.add_argument("--max-actions", type=int, default=6, choices=range(2, 7))
    args = p.parse_args(argv)
    for path in (args.output, args.labels_out):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = skipped = 0
    with open(args.input) as fin, open(args.output, "w") as fout, open(args.labels_out, "w") as labels:
        for line in fin:
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("needs_replay"):
                skipped += 1
                continue
            groups, _ = slice_transactions(sample, args.max_actions)
            for number, (positions, _) in enumerate(groups, 1):
                tx, sidecar = build_transaction(sample, number, positions)
                fout.write(json.dumps(tx, ensure_ascii=False) + "\n")
                labels.write(json.dumps(sidecar, ensure_ascii=False) + "\n")
                n += 1
    print(json.dumps({"transactions": n, "skipped_incompatible_traces": skipped}))
    return 0 if n else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as e:
        print(f"transactions: {e}", file=sys.stderr)
        sys.exit(2)
