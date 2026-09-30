#!/usr/bin/env python3
"""Read recovered P003 panel artifacts; save bounded observations, not new labels.

This does not execute model calls, tool calls, repository tests, or experiments.
Missing files remain explicit. Re-run after recovery to complete the inventory.
"""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "results/003/recovered-20260930/results/003/run-003-joint-panel-20260929T151450Z-32b18764"
OUT = Path(__file__).with_name("trace_review.json")


def distribution(values):
    values = sorted(values)
    def percentile(p):
        x = (len(values) - 1) * p
        return values[math.floor(x)] + (values[math.ceil(x)]-values[math.floor(x)]) * (x % 1) if values else None
    return {"n": len(values), "sum": sum(values), "max": max(values) if values else None, "p95": percentile(.95), "p99": percentile(.99), "ge4096": sum(v >= 4096 for v in values), "ge8192": sum(v >= 8192 for v in values), "ge16384": sum(v >= 16384 for v in values)}


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def production(filepath, repo):
    if not isinstance(filepath, str):
        return False
    prefix = {"fastapi/fastapi": "fastapi/", "Textualize/rich": "rich/", "psf/requests": "src/requests/"}[repo]
    return filepath.startswith(prefix) or "/" + prefix in filepath


def main():
    rows = list(csv.DictReader((ROOT / "research/003/analysis_003_panel.csv").open()))
    attempts, provenance, missing, completion_events = [], [], [], []
    for row in rows:
        directory = RUN / row["variant"] / "r1/attempts" / row["attempt_id"]
        result_path = directory / "result.json"
        trace_path = directory / "traces" / ("trace_" + row["task_id"] + ".json")
        item = {"attempt_id": row["attempt_id"], "variant": row["variant"], "task_id": row["task_id"], "csv_category": row["notes"].split(";", 1)[0].removeprefix("primary:"), "csv_first_source_edit_at": float(row["first_source_edit_at"]) if row["first_source_edit_at"] else None, "csv_production_source_changed": row["production_source_changed"] == "true", "result_present": result_path.exists(), "trace_present": trace_path.exists()}
        if result_path.exists():
            result, prov = load(result_path)
            provenance.append(prov)
            item["original_result"] = {k: result.get(k) for k in ("resolved", "test_exit_code", "agent_patch_size", "tool_calls", "total_llm_calls", "error")}
            item["csv_result_matches_original"] = all((str(result[k]).lower() == row[k]) for k in ("resolved", "test_exit_code", "tool_calls", "total_llm_calls"))
        else:
            missing.append(str(result_path.relative_to(ROOT)))
        if not trace_path.exists():
            missing.append(str(trace_path.relative_to(ROOT)))
            attempts.append(item)
            continue
        trace, prov = load(trace_path)
        provenance.append(prov)
        events, errors, unmatched, command_failures = [], [], [], []
        source_edits = []
        previous_signature, streak, longest_streak, duplicate_actions = None, 0, 0, 0
        local_completion = []
        for step in trace.get("steps", []):
            calls = step.get("tool_calls", [])
            completion = step.get("metrics", {}).get("completion_tokens")
            if isinstance(completion, (int, float)):
                local_completion.append(completion)
                completion_events.append({"attempt_id": row["attempt_id"], "variant": row["variant"], "resolved": item.get("original_result", {}).get("resolved"), "step_id": step.get("step_id"), "source": step.get("source"), "llm_call_count": step.get("llm_call_count"), "completion_tokens": completion, "tool_names": [c.get("function_name") for c in calls]})
            for call in calls:
                signature = json.dumps([call.get("function_name"), call.get("arguments")], sort_keys=True)
                if signature == previous_signature:
                    streak += 1
                    duplicate_actions += 1
                else:
                    streak = 1
                previous_signature = signature
                longest_streak = max(longest_streak, streak)
            observation = step.get("observation", {})
            if not calls:
                continue
            content = observation.get("content")
            try:
                decoded = json.loads(content) if isinstance(content, str) else content
            except (TypeError, json.JSONDecodeError):
                decoded = None
            if len(calls) != 1 or not isinstance(decoded, dict):
                unmatched.append({"step_id": step.get("step_id"), "tool_call_count": len(calls), "decoded_observation_type": type(decoded).__name__})
                continue
            call = calls[0]
            name = call.get("function_name")
            args = call.get("arguments") or {}
            if not isinstance(args, dict):
                unmatched.append({"step_id": step.get("step_id"), "arguments_type": type(args).__name__})
                continue
            event = {"step_id": step.get("step_id"), "tool": name, "filepath": args.get("filepath"), "call_elapsed_s": call.get("extra", {}).get("elapsed_s"), "observation_elapsed_s": observation.get("extra", {}).get("elapsed_s"), "status": decoded.get("status"), "error_type": decoded.get("error_type"), "error_message": decoded.get("error_message"), "exit_code": decoded.get("exit_code")}
            if decoded.get("status") is None and decoded.get("error"):
                event.update({"status": "error", "error_type": "InvocationErrorField_derived", "error_message": decoded["error"], "original_status": None})
            if name == "edit_file":
                old = args.get("old_string", "")
                event["old_string_literal_backslash_quote_count"] = old.count('\\"') if isinstance(old, str) else None
                event["old_string_literal_backslash_n_count"] = old.count("\\n") if isinstance(old, str) else None
            if name in ("edit_file", "write_file") and production(args.get("filepath"), row["repo"]):
                source_edits.append(event)
            if event["status"] == "error":
                errors.append(event)
            if name == "run_command" and isinstance(decoded.get("exit_code"), int) and decoded["exit_code"] != 0:
                command_failures.append(event)
            events.append(event)
        successful = [e for e in source_edits if e["status"] == "ok"]
        item.update({"trace_steps": len(trace.get("steps", [])), "attributable_tool_observations": len(events), "unmatched_observations": unmatched, "tool_error_count": len(errors), "tool_error_types": dict(Counter(e["error_type"] or "unspecified" for e in errors)), "tool_errors": errors, "nonzero_run_command_exits": command_failures, "source_edit_attempts": source_edits, "source_edit_attempt_count": len(source_edits), "source_edit_success_count": len(successful), "first_successful_source_edit_tool_call_at": successful[0]["call_elapsed_s"] if successful else None, "max_recorded_prompt_tokens": max((s.get("metrics", {}).get("prompt_tokens", 0) or 0 for s in trace.get("steps", [])), default=None), "completion_tokens": distribution(local_completion), "completion_sum_matches_trace_final_metrics": sum(local_completion) == trace.get("final_metrics", {}).get("total_completion_tokens"), "consecutive_identical_tool_call_duplicate_actions": duplicate_actions, "longest_identical_tool_call_streak": longest_streak})
        attempts.append(item)
    variants = {}
    for variant in ("a0", "b"):
        valid = [r for r in attempts if r["variant"] == variant and r["trace_present"]]
        errors = [e for r in valid for e in r["tool_errors"]]
        prod = [e for r in valid for e in r["source_edit_attempts"]]
        variants[variant] = {"traces_available": len(valid), "attempts_with_tool_errors": sum(bool(r["tool_errors"]) for r in valid), "tool_error_count": len(errors), "tool_error_types": dict(Counter(e["error_type"] or "unspecified" for e in errors)), "source_edit_tool_attempts": len(prod), "source_edit_tool_successes": sum(e["status"] == "ok" for e in prod), "source_edit_tool_errors": sum(e["status"] == "error" for e in prod), "source_edit_other_statuses": sum(e["status"] not in ("ok", "error") for e in prod), "attempts_with_successful_source_edit_tool": sum(r["source_edit_success_count"] > 0 for r in valid), "attempts_with_source_edit_tool_but_none_successful": sum(r["source_edit_attempt_count"] > 0 and r["source_edit_success_count"] == 0 for r in valid), "completion_tokens": distribution(e["completion_tokens"] for e in completion_events if e["variant"] == variant), "identical_consecutive_tool_call_duplicate_actions": sum(r["consecutive_identical_tool_call_duplicate_actions"] for r in valid)}
    findings = {
        "csv_first_edit_without_successful_tool_edit": [{k: r[k] for k in ("attempt_id", "csv_category", "csv_first_source_edit_at", "source_edit_attempts")} for r in attempts if r["trace_present"] and r["csv_first_source_edit_at"] is not None and not r["source_edit_success_count"]],
        "original_errors_other_than_session_timeout": [{"attempt_id": r["attempt_id"], "csv_category": r["csv_category"], "error": r["original_result"]["error"]} for r in attempts if r.get("original_result", {}).get("error") and "session timeout" not in r["original_result"]["error"].lower()],
        "csv_original_result_mismatches": [r["attempt_id"] for r in attempts if r.get("csv_result_matches_original") is False],
    }
    paired_attempts = defaultdict(dict)
    for row in attempts:
        paired_attempts[row["task_id"]][row["variant"]] = row
    confirmed_pairs = []
    for task, variants_for_task in paired_attempts.items():
        a = variants_for_task["a0"].get("first_successful_source_edit_tool_call_at")
        b = variants_for_task["b"].get("first_successful_source_edit_tool_call_at")
        if a is not None and b is not None:
            confirmed_pairs.append({"task_id": task, "a0_seconds": a, "b_seconds": b, "b_minus_a0_seconds": b-a})
    deltas = [r["b_minus_a0_seconds"] for r in confirmed_pairs]
    findings["confirmed_source_edit_timing"] = {"eligible_pairs": len(deltas), "excluded_pairs_without_two_confirmed_tool_edits": len(paired_attempts)-len(deltas), "mean_b_minus_a0_seconds": statistics.mean(deltas) if deltas else None, "median_b_minus_a0_seconds": statistics.median(deltas) if deltas else None, "pairs": confirmed_pairs, "caveat": "Subset selected on successful tool edit in both arms, not an intention-to-treat or causal comparison. Tool status=ok does not establish correct code or final patch retention."}
    output = {"review_date": "2026-09-30", "scope": "P003 original panel only; exploration traces not scanned.", "method": "Only one-tool-call steps with dict observation after one JSON decode are attributed. Production-file prefix is a path proxy. Successful edit_file/write_file is status=ok, independent of final diff; run_command can edit files and is not captured by this particular counter. Dict observations with only an error field are counted separately under InvocationErrorField_derived. No attempt is reclassified automatically. Read and edit escaping flags alone do not prove harness fault. Identical consecutive tool calls are a new exact-signature measure, not the CSV no-progress heuristic; repeating a valid call need not imply no progress.", "completeness": {"expected_attempts": 48, "results_available": sum(r["result_present"] for r in attempts), "traces_available": sum(r["trace_present"] for r in attempts), "missing": missing}, "completion_analysis": {"all_recorded_llm_steps": distribution(e["completion_tokens"] for e in completion_events), "successful_attempts_recorded_steps": distribution(e["completion_tokens"] for e in completion_events if e["resolved"]), "step_source_counts": dict(Counter(e["source"] for e in completion_events)), "llm_call_count_values": dict(Counter(str(e["llm_call_count"]) for e in completion_events)), "all_final_metric_sums_match": all(r.get("completion_sum_matches_trace_final_metrics", False) for r in attempts), "largest_20_steps": sorted(completion_events, key=lambda e:e["completion_tokens"], reverse=True)[:20], "limitation": "The trace has no independent compaction-call identifier. Distribution covers recorded agent steps only and does not establish coverage of any unrecorded compaction/inference calls or the failed overflowing call. Zero observed calls >=8192 supports a lower-cap diagnostic, not a guarantee against future truncation or changed trajectories. Quantiles use linear interpolation at (n-1)*p. No model thought text is exported."}, "by_variant": variants, "findings": findings, "attempts": attempts, "provenance": provenance}
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: output[k] for k in ("completeness", "by_variant", "findings")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
