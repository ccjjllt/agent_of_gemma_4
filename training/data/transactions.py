#!/usr/bin/env python3
"""Select short assistant targets while preserving their entire observed prefix."""
import argparse
import json
import sys
from pathlib import Path

from sample_behaviors import (
    BEHAVIORS, attach_event_ids, classify_messages, calls_of as detailed_calls,
    reply_status as labeled_reply_status,
)


def assistant_indices(messages):
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def calls_of(message):
    return [(cid, name) for cid, name, _args in detailed_calls(message)]


def reply_status(reply):
    return labeled_reply_status(reply)


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
        labeled = classify_messages(sample["messages"], current)["behavior"]
        if labeled in BEHAVIORS:
            chunks.append((current, False))
        else:
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
    labeled = classify_messages(prefix, step_ids)
    evidence, event_ids = attach_event_ids(sample.get("sample_id", ""), labeled["recovery_evidence"])
    outcome = dict(sample.get("observed_outcome") or {})
    outcome["action_succeeded"] = labeled["action_succeeded"]
    if "task_resolved" not in outcome:
        outcome["task_resolved"] = None
    tx = {k: sample[k] for k in ("sample_id", "origin_kind", "group", "source",
        "protocol_id", "tool_schema_hash", "synthetic") if k in sample}
    tx.update(transaction_id=f"{sample['sample_id']}-tx{tx_num:02d}",
        messages=prefix, target_assistant_steps=targets, behavior=labeled["behavior"],
        behaviors_present=labeled["behaviors_present"],
        observed_outcome=outcome, missing_observation=False,
        recovery_evidence=evidence, recovery_event_ids=event_ids,
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
