#!/usr/bin/env python3
"""Label repair transactions from tool actions and sample across traces.

Failed file-tool calls stay in the prefix as context. A later success on the
same tool+file closes that failure so the same recovery is not supervised twice.
Behavior names: local_edit, tool_recovery, verify_close. Missing classes are
reported, never invented. Pairing-reserved issues and the same repo+commit are
excluded from train/select.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections import defaultdict

BEHAVIORS = ("local_edit", "tool_recovery", "verify_close")
COVER_ORDER = ("tool_recovery", "local_edit", "verify_close")
DEFAULT_SEED = 20261010
FILE_TOOLS = ("read_file", "edit_file", "write_file")
EDIT_TOOLS = ("edit_file", "write_file")
CLOSE_TOOLS = ("submit_patch", "get_status")
PAIRING_RESERVED_ISSUES = (
    "fastapi_13713",
    "requests_6592",
    "requests_6644",
    "requests_7309",
    "requests_7505",
    "rich_3130",
    "rich_3296",
    "rich_3469",
)
# Model-jammed extra fields inside filepath (not extra JSON keys).
_FP_JUNK = re.compile(
    r"(?:\\+[\"']|`|,?\s*start_line|,?\s*end_line).*",
    re.IGNORECASE,
)
# Invoked-program names only. Do not search the full command string: a heredoc
# that writes `import pytest` is not a test runner.
_HEREDOC_OPEN = re.compile(r"""<<[-]?\s*(['\"]?)(\w+)\1""")
_TEST_BIN = re.compile(r"^(?:pytest|py\.test|nosetests|tox)(?:-\d+(?:\.\d+)*)?$")
_PYTHON_BIN = re.compile(r"^(?:python|pypy)(?:\d+(?:\.\d+)*)?$")
_TEST_MODULES = frozenset({"pytest", "unittest", "nose", "tox"})


def rank_key(seed, *parts):
    payload = ":".join(str(p) for p in (seed,) + parts)
    return hashlib.sha256(payload.encode()).hexdigest()


def content_digest(tx):
    payload = {"messages": tx.get("messages", []), "targets": tx.get("target_assistant_steps", [])}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def canonical_filepath(value):
    if not isinstance(value, str):
        return None
    text = value.strip().strip("\"'")
    text = _FP_JUNK.sub("", text)
    text = text.strip().strip("\"'`").rstrip("\\").strip()
    return text or None


def arguments_of(call):
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    value = function.get("arguments", call.get("arguments", call.get("params", {})))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            value = {}
    return value if isinstance(value, dict) else {}


def calls_of(message):
    rows = []
    if message.get("tool_calls"):
        for call in message["tool_calls"]:
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = function.get("name") or call.get("name") or call.get("function_name")
            cid = call.get("id") or call.get("tool_call_id")
            rows.append((cid, name, arguments_of(call)))
        return rows
    if message.get("tool_name"):
        args = message.get("arguments", message.get("params", {}))
        if not isinstance(args, dict):
            args = arguments_of({"arguments": args})
        rows.append((message.get("tool_call_id"), message.get("tool_name"), args))
    return rows


def parse_payload(value):
    raw = value
    if isinstance(value, dict) and "content" in value and "status" in value:
        raw = value.get("content", "")
    elif isinstance(value, dict) and "content" in value and "tool_call_id" in value:
        raw = value.get("content", "")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def reply_status(reply):
    payload = parse_payload(reply.get("content", reply.get("tool_response", reply)))
    status = payload.get("status", reply.get("status", "unknown") if isinstance(reply, dict) else "unknown")
    if payload.get("error_type") or status in (
            "error", "failed", "exception", "FileReadError", "FileEditError"):
        return "error"
    return "ok" if status in ("ok", "success", "applied") else "unknown"


def operation_key(name, args):
    if name not in FILE_TOOLS:
        return None
    path = canonical_filepath((args or {}).get("filepath"))
    if not path:
        return None
    return name, path


def _without_heredocs(command):
    """Drop heredoc payloads so body text cannot look like an invoked program."""
    out, pos = [], 0
    for match in _HEREDOC_OPEN.finditer(command):
        out.append(command[pos:match.start()])
        delim = match.group(2)
        rest = command[match.end():]
        newline = rest.find("\n")
        if newline < 0:
            return "".join(out)
        body = rest[newline + 1:]
        end = re.search(r"(?m)^" + re.escape(delim) + r"\s*$", body)
        if not end:
            return "".join(out)
        pos = match.end() + newline + 1 + end.end()
    out.append(command[pos:])
    return "".join(out)


def _drop_env(argv):
    i = 0
    if i < len(argv) and _prog_name(argv[0]) == "env":
        i = 1
    while i < len(argv) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[i]):
        i += 1
    return argv[i:]


def _prog_name(token):
    return token.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()


def _iter_invoked_argv(command):
    text = _without_heredocs(command)
    for chunk in re.split(r"[;\n]|&&|\|\|", text):
        chunk = chunk.strip()
        if not chunk or chunk.startswith("#"):
            continue
        try:
            tokens = shlex.split(chunk, posix=True)
        except ValueError:
            tokens = chunk.split()
        argv = []
        for token in tokens:
            if token == "|":
                if argv:
                    yield _drop_env(argv)
                argv = []
                continue
            argv.append(token)
        if argv:
            yield _drop_env(argv)


def is_verify_command(args):
    """True only if the invoked program is a test runner, not a substring match."""
    command = str((args or {}).get("command") or "")
    for argv in _iter_invoked_argv(command):
        if not argv:
            continue
        name = _prog_name(argv[0])
        if _TEST_BIN.match(name):
            return True
        if _PYTHON_BIN.match(name) and "-m" in argv:
            module = argv[argv.index("-m") + 1] if argv.index("-m") + 1 < len(argv) else ""
            if module.split(".")[0] in _TEST_MODULES:
                return True
    return False


def pairing_reserved(splits_records=None, extra_ids=None):
    issues = set(PAIRING_RESERVED_ISSUES)
    if extra_ids:
        issues.update(extra_ids)
    snaps = set()
    for row in splits_records or []:
        if row.get("instance_id") in issues:
            snaps.add((row.get("repo"), row.get("base_commit") or row.get("commit")))
    return issues, snaps


def is_reserved_group(issue, repo, commit, reserved_issues, reserved_snapshots):
    if issue in reserved_issues:
        return True
    return (repo, commit) in (reserved_snapshots or set())


def recovery_event_id(sample_id, failed_call):
    return "%s:%s" % (sample_id, failed_call)


def recoveries_in_prefix(messages, step_ids=None):
    """Close each (tool, file) failure on the next same-op success.

    Success always ends the open failure. Only successes inside step_ids are
    supervised recoveries. Command error→ok is not treated as recovery.
    """
    selected = set(step_ids or [])
    open_fail = {}
    events = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for cid, name, args in calls_of(message):
            replies = [m for m in messages[index + 1:]
                       if m.get("role") == "tool" and m.get("tool_call_id") == cid]
            if len(replies) != 1:
                raise ValueError("missing/ambiguous response to %s" % cid)
            status = reply_status(replies[0])
            key = operation_key(name, args)
            if key is None:
                continue
            if status == "error":
                open_fail[key] = {
                    "failed_call": cid,
                    "failed_observation": replies[0].get("content", ""),
                    "failed_index": index,
                    "tool_name": name,
                    "filepath": key[1],
                }
            elif status == "ok" and key in open_fail:
                failed = open_fail.pop(key)
                events.append({
                    "failed_call": failed["failed_call"],
                    "successful_call": cid,
                    "tool_name": name,
                    "filepath": key[1],
                    "failed_observation": failed["failed_observation"],
                    "success_observation": replies[0].get("content", ""),
                    "failed_index": failed["failed_index"],
                    "success_index": index,
                    "supervised": index in selected,
                })
    return events


def classify_messages(messages, step_ids):
    selected = list(step_ids)
    selected_set = set(selected)
    events = recoveries_in_prefix(messages, selected)
    supervised = [e for e in events if e["supervised"]]
    prefix_edit_ok = False
    selected_edit_ok = False
    selected_verify = False
    selected_statuses = []
    selected_names = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for cid, name, args in calls_of(message):
            replies = [m for m in messages[index + 1:]
                       if m.get("role") == "tool" and m.get("tool_call_id") == cid]
            if len(replies) != 1:
                raise ValueError("missing/ambiguous response to %s" % cid)
            status = reply_status(replies[0])
            if name in EDIT_TOOLS and status == "ok":
                prefix_edit_ok = True
            if index not in selected_set:
                continue
            selected_statuses.append(status)
            selected_names.append(name)
            if name in EDIT_TOOLS and status == "ok":
                selected_edit_ok = True
            if name in CLOSE_TOOLS:
                selected_verify = True
            elif name == "run_command" and is_verify_command(args) and prefix_edit_ok:
                selected_verify = True
    present = []
    if supervised:
        present.append("tool_recovery")
    if selected_edit_ok:
        present.append("local_edit")
    if selected_verify:
        present.append("verify_close")
    if "tool_recovery" in present:
        behavior = "tool_recovery"
    elif "local_edit" in present:
        behavior = "local_edit"
    elif "verify_close" in present:
        behavior = "verify_close"
    else:
        behavior = "other"
    if not selected_statuses:
        action_succeeded = None
    else:
        action_succeeded = all(status == "ok" for status in selected_statuses)
    return {
        "behavior": behavior,
        "behaviors_present": present,
        "recovery_evidence": [{k: e[k] for k in (
            "failed_call", "successful_call", "tool_name", "filepath",
            "failed_observation", "success_observation")} for e in supervised],
        "closed_recovery_events": events,
        "action_succeeded": action_succeeded,
        "selected_tool_names": selected_names,
    }


def attach_event_ids(sample_id, evidence):
    rows = []
    ids = []
    for item in evidence:
        eid = recovery_event_id(sample_id, item["failed_call"])
        row = dict(item)
        row["event_id"] = eid
        rows.append(row)
        ids.append(eid)
    return rows, ids


def _group_fields(tx):
    group = tx.get("group") or {}
    return group.get("issue"), group.get("repo"), group.get("commit")


def exclude_reserved(transactions, reserved_issues, reserved_snapshots):
    kept, dropped = [], []
    for tx in transactions:
        issue, repo, commit = _group_fields(tx)
        if is_reserved_group(issue, repo, commit, reserved_issues, reserved_snapshots):
            dropped.append(tx.get("transaction_id"))
            continue
        kept.append(tx)
    return kept, dropped


def dedup_content(transactions, seed):
    best = {}
    for tx in transactions:
        digest = content_digest(tx)
        tid = tx.get("transaction_id") or ""
        score = rank_key(seed, "content", tid)
        prev = best.get(digest)
        if prev is None or score < prev[0]:
            best[digest] = (score, tx, digest)
    return [item[1] for item in best.values()]


def _dedup_recovery_events(transactions):
    """Keep one transaction per closed failure; prefer the shortest prefix."""
    by_event = defaultdict(list)
    others = []
    for tx in transactions:
        evidence = tx.get("recovery_evidence") or []
        sample_id = tx.get("sample_id") or ""
        if tx.get("behavior") == "tool_recovery" and evidence:
            for item in evidence:
                eid = item.get("event_id") or recovery_event_id(sample_id, item.get("failed_call"))
                by_event[eid].append(tx)
        else:
            others.append(tx)
    chosen = {}
    event_owner = {}
    for eid, group in by_event.items():
        group.sort(key=lambda tx: (len(tx.get("messages") or []), tx.get("transaction_id") or ""))
        winner = group[0]
        event_owner[eid] = winner.get("transaction_id")
        chosen[winner.get("transaction_id")] = winner
    return list(chosen.values()) + others, event_owner


def select_issue_transactions(transactions, max_per_issue=3, seed=DEFAULT_SEED):
    if max_per_issue < 1:
        raise ValueError("max-per-issue must be positive")
    if not transactions:
        return [], {"missing_behaviors": list(BEHAVIORS), "reasons": {},
                    "recovery_events_before": 0, "recovery_events_kept": 0}
    issue = _group_fields(transactions[0])[0]
    recovered, event_owner = _dedup_recovery_events(transactions)
    pools = {name: [] for name in BEHAVIORS}
    event_ids_before = set()
    for tx in transactions:
        for item in tx.get("recovery_evidence") or []:
            event_ids_before.add(item.get("event_id") or recovery_event_id(
                tx.get("sample_id"), item.get("failed_call")))
    for tx in recovered:
        behavior = tx.get("behavior")
        if behavior in pools:
            pools[behavior].append(tx)
    selected, reasons, used = [], {}, set()
    missing = []
    for behavior in COVER_ORDER:
        pool = [tx for tx in pools[behavior] if tx.get("transaction_id") not in used]
        if not pool:
            missing.append(behavior)
            continue
        pool.sort(key=lambda tx: rank_key(seed, issue, behavior, tx.get("transaction_id")))
        pick = pool[0]
        tid = pick["transaction_id"]
        used.add(tid)
        reasons[tid] = "cover %s; rank 0 of %d with seed=%s after recovery-event dedup" % (
            behavior, len(pool), seed)
        selected.append(pick)
        if len(selected) >= max_per_issue:
            break
    if len(selected) < max_per_issue:
        leftover = []
        for behavior in COVER_ORDER:
            leftover.extend(tx for tx in pools[behavior] if tx.get("transaction_id") not in used)
        traces = {tx.get("sample_id") for tx in selected}
        leftover.sort(key=lambda tx: (
            0 if tx.get("sample_id") not in traces else 1,
            rank_key(seed, issue, "fill", tx.get("transaction_id")),
        ))
        for tx in leftover:
            if len(selected) >= max_per_issue:
                break
            tid = tx["transaction_id"]
            used.add(tid)
            other = "other-trace" if tx.get("sample_id") not in traces else "same-trace"
            reasons[tid] = "fill remaining slot from %s (%s) seed=%s" % (
                tx.get("behavior"), other, seed)
            selected.append(tx)
            traces.add(tx.get("sample_id"))
    kept_events = set()
    for tx in selected:
        for item in tx.get("recovery_evidence") or []:
            kept_events.add(item.get("event_id") or recovery_event_id(
                tx.get("sample_id"), item.get("failed_call")))
    return selected, {
        "missing_behaviors": missing,
        "reasons": reasons,
        "recovery_events_before": len(event_ids_before),
        "recovery_events_kept": len(kept_events),
        "event_owner": event_owner,
        "pool_counts": {k: len(v) for k, v in pools.items()},
    }


def select_transactions(transactions, max_per_issue=3, seed=DEFAULT_SEED,
                        reserved_issues=None, reserved_snapshots=None):
    reserved_issues = set(reserved_issues or ()) | set(PAIRING_RESERVED_ISSUES)
    reserved_snapshots = set(tuple(x) for x in (reserved_snapshots or ()))
    usable, dropped_reserved = exclude_reserved(
        transactions, reserved_issues, reserved_snapshots)
    unique = dedup_content(usable, seed)
    by_issue = defaultdict(list)
    for tx in unique:
        issue = _group_fields(tx)[0]
        if not issue:
            raise ValueError("every transaction needs group.issue")
        by_issue[issue].append(tx)
    selected, audits = [], {}
    missing_by_issue = {}
    for issue in sorted(by_issue):
        rows, audit = select_issue_transactions(by_issue[issue], max_per_issue, seed)
        for tx in rows:
            out = dict(tx)
            reason = audit["reasons"].get(tx["transaction_id"])
            out["sampling"] = {"seed": seed, "reason": reason, "max_per_issue": max_per_issue}
            selected.append(out)
        audits[issue] = audit
        missing_by_issue[issue] = audit["missing_behaviors"]
    return selected, {
        "seed": seed,
        "max_per_issue": max_per_issue,
        "reserved_issues": sorted(reserved_issues),
        "dropped_reserved": dropped_reserved,
        "before": len(transactions),
        "after_reserved_filter": len(usable),
        "after_content_dedup": len(unique),
        "selected": len(selected),
        "issues": sorted(by_issue),
        "missing_behaviors_by_issue": missing_by_issue,
        "issue_audits": {k: {kk: vv for kk, vv in audit.items() if kk != "event_owner"}
                         for k, audit in audits.items()},
    }
