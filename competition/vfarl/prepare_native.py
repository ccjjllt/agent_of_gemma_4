#!/usr/bin/env python3
"""Prepare provenance-preserving candidates; never execute source commands.

This is data production, not target-tool replay or verified-pair generation.
Usage: python3 competition/vfarl/prepare_native.py --raw RAW.jsonl ... --out DIR
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


VERSION = "native-candidates-v1"
SUPPORTED = {
    "nebius/SWE-rebench-openhands-trajectories":
        ("35455389ab51bf5e2306bfd436ef72d0f98bf882", "train"),
    "SWE-bench/SWE-smith-trajectories":
        ("08e109b4a59eaeebf80e4675cd125d42e7ac99a4", "tool"),
}
EXCLUDED = re.compile(
    r"(?:^|[/_])(?:fastapi|rich|requests|httpx)(?:$|[._/-])", re.I
)
ERROR_CUES = re.compile(
    r"No match found|No replacement was performed|multiple occurrences|"
    r"old_string.*(?:not found|not match)|SyntaxError:|IndentationError:|"
    r"Traceback \(most recent call last\)|command not found|"
    r"timed out|TimeoutExceeded|FileNotFoundError:|exit code[: ]+[1-9]", re.I
)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def as_text(value):
    return value if isinstance(value, str) else encoded(value).decode("utf-8")


def calls(message):
    """Preserve source arguments verbatim; indices link back to raw messages."""
    for i, call in enumerate(message.get("tool_calls") or []):
        fn = call.get("function") or {}
        yield i, call.get("id"), fn.get("name"), fn.get("arguments")


def target_proposal(name, arguments):
    """A replay work item, never a claim that tools are interchangeable."""
    if name in {"run_command", "execute_bash", "bash"}:
        return {"possible_tools": ["run_command"],
                "work": "Re-execute; resolve cwd/env/session/poll and truncation semantics."}
    if name in {"str_replace_editor", "editor", "file_editor"}:
        args = arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                args = {}
        command = args.get("command") if isinstance(args, dict) else None
        targets = {"view": ["read_file"], "str_replace": ["edit_file"],
                   "create": ["write_file"]}.get(command, [])
        return {"possible_tools": targets, "source_editor_command": command,
                "work": "Re-execute with actual six-tool path, anchor and error semantics."}
    if name in {"finish", "submit", "submit_patch"}:
        return {"possible_tools": ["submit_patch"],
                "work": "Capture actual protected-path-filtered git diff; finishing is not validation."}
    if name in {"read_file", "write_file", "edit_file", "get_status"}:
        return {"possible_tools": [name],
                "work": "Matching name alone is insufficient; execute official implementation."}
    return {"possible_tools": [], "work": "Manual source-tool review required."}


def error_evidence(content):
    parsed = content
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except (ValueError, TypeError):
            pass
    evidence = []
    if isinstance(parsed, dict):
        if parsed.get("status") == "error" or parsed.get("is_error") is True:
            evidence.append({"kind": "source_structured_error"})
        code = parsed.get("exit_code")
        if isinstance(code, int) and code != 0:
            evidence.append({"kind": "source_nonzero_exit", "exit_code": code})
    match = ERROR_CUES.search(as_text(content))
    if match:
        evidence.append({"kind": "text_cue_requires_review", "matched": match.group(0)})
    return evidence


def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(encoded(row).decode("utf-8") + "\n")
    return {"rows": len(rows), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare(raw_paths, out):
    out.mkdir(parents=True, exist_ok=True)
    episodes, cold, recovery, seen = [], [], [], set()
    tasks, tool_stats, roles = {}, Counter(), Counter()
    input_files, exclusions = [], []
    observations_without_id = 0
    for path in raw_paths:
        payload = path.read_bytes()
        input_files.append({"path": str(path.resolve()), "bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest()})
        for lineno, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            wrapper = json.loads(line)
            source = wrapper.get("source") or wrapper.get("dataset")
            expected = SUPPORTED.get(source)
            if expected != (wrapper.get("revision"), wrapper.get("split")):
                raise ValueError("Unapproved dataset/revision/split at %s:%s" % (path, lineno))
            row = wrapper["row"]
            instance = row["instance_id"]
            repo = row.get("repo")
            # Conservative name/known-alias exclusion; unknown renamed forks still need review.
            if EXCLUDED.search((repo or "") + "/" + instance):
                exclusions.append({"source": source, "instance_id": instance,
                                   "reason": "competition_repository_name_or_alias"})
                continue
            trajectory = row.get("trajectory_id") or row.get("traj_id")
            if not trajectory:
                raise ValueError("Missing stable trajectory ID")
            identity = (source, wrapper["revision"], trajectory)
            if identity in seen:
                continue
            seen.add(identity)
            messages = row.get("trajectory")
            if messages is None:
                messages = json.loads(row["messages"])
            if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
                raise ValueError("Native messages must be a list of objects")
            resolved = row.get("resolved")
            if resolved not in (0, 1, False, True):
                raise ValueError("Unexpected source resolved value")
            episode_id = digest(list(identity))
            task_key = digest([source, wrapper["revision"], instance])
            provenance = {k: v for k, v in wrapper.items() if k != "row"}
            record = {
                "schema_version": VERSION, "episode_id": episode_id,
                "source_task_key": task_key, "instance_id": instance,
                "source_repo": repo, "source_resolved": bool(resolved),
                "source_model": row.get("model"),
                "source_exit_status": row.get("exit_status"),
                "source_provenance": provenance, "source_row_sha256": digest(row),
                "raw_ref": {"path": str(path.resolve()), "line_1_based": lineno},
                "native_messages_sha256": digest(messages), "messages": messages,
                "source_tools": row.get("tools"),
                "execution_evidence": "publisher_native_only",
                "target_six_tool_replayed": False,
                "behavior_logprobs": None,
                "source_patch_training_admitted": False,
                "source_patch_audit": (
                    "patch column quarantined after confirmed cross-repository mismatches in seed"
                    if source == "SWE-bench/SWE-smith-trajectories" else
                    "model_patch requires production-change and auxiliary-artifact review"
                ),
            }
            episodes.append(record)
            task = tasks.setdefault(task_key, {
                "source_task_key": task_key, "source": source,
                "revision": wrapper["revision"], "instance_id": instance,
                "source_repo": repo, "episode_ids": [], "source_outcomes": [],
                "primary_adapter": None, "submechanism": None,
                "mechanism_evidence": None,
                "canonical_repo_and_repair_family_review": "required",
            })
            task["episode_ids"].append(episode_id)
            task["source_outcomes"].append(bool(resolved))
            if resolved:
                cold.append({"episode_id": episode_id, "source_task_key": task_key,
                             "evidence": "publisher_success_candidate",
                             "six_tool_sft_ready": False,
                             "required_work": ["task/environment identity join",
                                               "mechanism and leakage review",
                                               "actual six-tool replay and task verification",
                                               "actual Gemma template/tokenizer/loss mask"]})
            actions, result_by_id, observation_groups = {}, defaultdict(list), {}
            for mi, message in enumerate(messages):
                roles[message.get("role", "missing")] += 1
                for ci, call_id, name, args in calls(message):
                    tool_stats[name or "missing"] += 1
                    if call_id:
                        if call_id in actions:
                            raise ValueError("Duplicate native tool_call ID within episode")
                        actions[call_id] = (mi, ci, name, args)
                if message.get("role") == "tool":
                    ids = message.get("tool_call_ids")
                    if ids is None:
                        ids = [message["tool_call_id"]] if message.get("tool_call_id") else []
                    if isinstance(ids, str):
                        ids = [ids]
                    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                        raise ValueError("Malformed native tool-call result IDs")
                    if ids:
                        observation_groups[mi] = ids
                        for call_id in ids:
                            result_by_id[call_id].append(mi)
                    else:
                        observations_without_id += 1
            previously_flagged = {}
            for call_id, (mi, ci, name, args) in actions.items():
                results = [ri for ri in result_by_id.get(call_id, []) if ri > mi]
                evidence = []
                for ri in results:
                    evidence.extend(dict(item, observation_message_index=ri)
                                    for item in error_evidence(messages[ri].get("content")))
                action_hash = digest([name, args])
                previous = previously_flagged.get(action_hash)
                if previous and any(ri < mi for ri in previous[1]):
                    evidence.append({"kind": "same_action_after_error_cue",
                                     "earlier_action_message_index": previous[0]})
                if not evidence:
                    continue
                observed_error_indices = [e["observation_message_index"] for e in evidence
                                          if "observation_message_index" in e]
                if observed_error_indices:
                    previously_flagged[action_hash] = (mi, observed_error_indices)
                recovery.append({
                    "candidate_id": digest([episode_id, mi, ci]),
                    "episode_id": episode_id, "source_task_key": task_key,
                    "action_message_index": mi, "tool_call_index": ci,
                    "source_tool_call_id": call_id, "source_tool_name": name,
                    "observation_message_indices": results, "candidate_evidence": evidence,
                    "observations_aggregate_multiple_calls": any(
                        len(observation_groups[ri]) > 1 for ri in results),
                    "native_message_prefix_sha256": digest(messages[:mi]),
                    "prefix_boundary": "source_assistant_turn_start",
                    "assistant_turn_tool_call_count": len(messages[mi].get("tool_calls") or []),
                    "prefix_hash_is_environment_state_id": False,
                    "source_task_resolved": bool(resolved),
                    "target_replay_proposal": target_proposal(name, args),
                    "training_pair_created": False,
                })
    counts = {}
    for filename, rows in [("native-episodes.jsonl", episodes),
                           ("cold-start-candidates.jsonl", cold),
                           ("recovery-candidates.jsonl", recovery),
                           ("mechanism-review-queue.jsonl", list(tasks.values()))]:
        counts[filename] = write_rows(out / filename, rows)
    report = {
        "schema_version": VERSION, "inputs": input_files, "outputs": counts,
        "source_qualified_task_ids": len(tasks),
        "cross_source_repair_family_dedup_completed": False,
        "source_successful_episodes": sum(e["source_resolved"] for e in episodes),
        "source_unsuccessful_episodes": sum(not e["source_resolved"] for e in episodes),
        "source_tool_call_counts": dict(tool_stats), "message_role_counts": dict(roles),
        "tool_observations_without_id_not_linked": observations_without_id,
        "patch_columns_exported_as_training_targets": False,
        "excluded_rows": exclusions,
        "six_tool_replayed_episodes": 0, "verified_critical_pairs": 0,
        "verified_recovery_segments": 0, "current_policy_rl_rollouts": 0,
        "source_commands_executed": 0,
        "limitations": ["Error text cues are candidates, not verified critical mistakes.",
                        "Later recovery does not prove a same-state counterfactual.",
                        "Native results are never relabeled as official-tool observations.",
                        "No behavior probabilities, time or state snapshots are fabricated."],
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.raw, args.out)
    print(json.dumps({k: v for k, v in report.items() if k not in ("inputs", "limitations")},
                     ensure_ascii=False, indent=2))
