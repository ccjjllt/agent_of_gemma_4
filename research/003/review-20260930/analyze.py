#!/usr/bin/env python3
"""Reproduce descriptive P003 CSV review; no external dependencies or experiments.

Run from any directory: python3 /absolute/path/to/this/analyze.py
Writes review.json beside this script. Original inputs are never modified.
All study grain is attempt within batch; paired grain is task_id, not attempts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "review.json"
BOOL_FIELDS = ("resolved", "production_source_changed", "explicit_submit")
INT_FIELDS = ("test_exit_code", "tool_calls", "total_llm_calls", "repeated_no_progress_actions")
FLOAT_FIELDS = ("task_duration_seconds", "agent_duration_seconds", "first_source_edit_at")
METRICS = INT_FIELDS[1:] + FLOAT_FIELDS
SMOKE = {"fastapi_11194", "requests_6592", "rich_2725"}


def load_csv(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for line, row in enumerate(reader, 2):
            for field in BOOL_FIELDS:
                assert row[field] in ("true", "false"), (line, field, row[field])
                row[field] = row[field] == "true"
            for field in INT_FIELDS:
                row[field] = int(row[field]) if row[field] else None
            for field in FLOAT_FIELDS:
                row[field] = float(row[field]) if row[field] else None
            row["source"] = relative
            row["source_line"] = line
            row["category"] = row["notes"].split(";", 1)[0].removeprefix("primary:") if row["notes"].startswith("primary:") else None
            row["smoke_task"] = row["task_id"] in SMOKE
            rows.append(row)
    assert len({r["attempt_id"] for r in rows}) == len(rows), "Duplicate attempt_id"
    assert len({(r["variant"], r["task_id"]) for r in rows}) == len(rows), "Duplicate variant-task grain"
    return rows, {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "rows": len(rows), "columns": reader.fieldnames}


def quantile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * p
    low, high = math.floor(index), math.ceil(index)
    return values[low] + (values[high] - values[low]) * (index - low)


def numeric(values):
    values = list(values)
    valid = [v for v in values if v is not None]
    return {
        "n": len(valid), "missing": len(values) - len(valid),
        "sum": sum(valid) if valid else None,
        "mean": statistics.mean(valid) if valid else None,
        "median": statistics.median(valid) if valid else None,
        "min": min(valid) if valid else None, "max": max(valid) if valid else None,
        "q25": quantile(valid, 0.25), "q75": quantile(valid, 0.75),
    }


def counts(rows, field):
    return dict(sorted(Counter(str(r[field]).lower() for r in rows).items()))


def compact(row):
    return {k: row[k] for k in ("attempt_id", "task_id", "variant", "repo", "category", "resolved", "production_source_changed", "explicit_submit", "agent_end_reason", "verification_status", "test_exit_code", "agent_duration_seconds", "first_source_edit_at", "tool_calls", "total_llm_calls", "repeated_no_progress_actions", "source", "source_line", "notes")}


def describe(rows):
    first = [r for r in rows if r["first_source_edit_at"] is not None]
    return {
        "attempts": len(rows), "unique_tasks": len({r["task_id"] for r in rows}),
        "resolved": sum(r["resolved"] for r in rows),
        "resolved_rate": sum(r["resolved"] for r in rows) / len(rows) if rows else None,
        "explicit_submit": sum(r["explicit_submit"] for r in rows),
        "production_source_changed": sum(r["production_source_changed"] for r in rows),
        "first_edit_observed": len(first),
        "agent_end_reason": counts(rows, "agent_end_reason"),
        "verification_status": counts(rows, "verification_status"),
        "test_exit_code": counts(rows, "test_exit_code"),
        "category": counts(rows, "category"),
        "metrics": {field: numeric(r[field] for r in rows) for field in METRICS},
        "descriptive_flags": {
            "agent_duration_ge_295s": sum(r["agent_duration_seconds"] >= 295 for r in rows),
            "agent_duration_ge_300s": sum(r["agent_duration_seconds"] >= 300 for r in rows),
            "tool_calls_ge_50": sum(r["tool_calls"] >= 50 for r in rows),
            "first_source_edit_ge_240s": sum(r["first_source_edit_at"] >= 240 for r in first),
            "first_source_edit_ge_270s": sum(r["first_source_edit_at"] >= 270 for r in first),
            "first_source_edit_ge_280s": sum(r["first_source_edit_at"] >= 280 for r in first),
            "first_source_edit_observed_but_no_final_source_change": sum(not r["production_source_changed"] for r in first),
            "final_source_change_but_no_first_edit_observed": sum(r["production_source_changed"] and r["first_source_edit_at"] is None for r in rows),
        },
        "first_edit_to_agent_end_seconds": numeric(r["agent_duration_seconds"] - r["first_source_edit_at"] for r in first),
        "cross_submit_source_resolved": [
            {"explicit_submit": s, "production_source_changed": p, "resolved": y,
             "n": sum(r["explicit_submit"] == s and r["production_source_changed"] == p and r["resolved"] == y for r in rows)}
            for s in (False, True) for p in (False, True) for y in (False, True)
        ],
    }


def group_describe(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[field]).lower()].append(row)
    return {key: describe(value) for key, value in sorted(groups.items())}


def paired(rows, baseline, candidate):
    a = {r["task_id"]: r for r in rows if r["variant"] == baseline}
    b = {r["task_id"]: r for r in rows if r["variant"] == candidate}
    assert a.keys() == b.keys(), "Incomplete task pairing"
    ids = sorted(a)
    win = [t for t in ids if b[t]["resolved"] and not a[t]["resolved"]]
    loss = [t for t in ids if a[t]["resolved"] and not b[t]["resolved"]]
    both = [t for t in ids if a[t]["resolved"] and b[t]["resolved"]]
    neither = [t for t in ids if not a[t]["resolved"] and not b[t]["resolved"]]
    n = len(win) + len(loss)
    exact_p = min(1, 2 * sum(math.comb(n, k) for k in range(min(len(win), len(loss)) + 1)) / 2**n) if n else 1.0
    metrics = {}
    for field in METRICS:
        matched = [t for t in ids if a[t][field] is not None and b[t][field] is not None]
        differences = [b[t][field] - a[t][field] for t in matched]
        metrics[field] = {"candidate_minus_baseline": numeric(differences), "eligible_pairs": len(matched), "excluded_missing_pairs": len(ids)-len(matched), "candidate_lower": sum(d < 0 for d in differences), "equal": sum(d == 0 for d in differences), "candidate_higher": sum(d > 0 for d in differences)}
    boolean_pairs = {}
    for field in BOOL_FIELDS:
        boolean_pairs[field] = {
            "both_true": sum(a[t][field] and b[t][field] for t in ids),
            "baseline_only": sum(a[t][field] and not b[t][field] for t in ids),
            "candidate_only": sum(b[t][field] and not a[t][field] for t in ids),
            "both_false": sum(not a[t][field] and not b[t][field] for t in ids),
        }
    transitions = Counter((a[t]["category"], b[t]["category"]) for t in ids)
    return {"baseline": baseline, "candidate": candidate, "tasks": len(ids), "baseline_resolved": sum(r["resolved"] for r in a.values()), "candidate_resolved": sum(r["resolved"] for r in b.values()), "wins": win, "losses": loss, "both_resolved": both, "neither_resolved": neither, "net_resolved": len(win)-len(loss), "exact_two_sided_discordant_binomial_p": exact_p, "exact_p_caveat": "Descriptive conditional paired test only; no multiplicity correction, no power/equivalence claim. Task independence is not guaranteed across snapshot groups; small n.", "boolean_pairs": boolean_pairs, "metrics": metrics, "category_transitions": [{"baseline": x, "candidate": y, "n": v} for (x,y),v in transitions.items()], "task_pairs": [{"task_id": t, "baseline_line": a[t]["source_line"], "candidate_line": b[t]["source_line"], "baseline_resolved": a[t]["resolved"], "candidate_resolved": b[t]["resolved"], "baseline_category": a[t]["category"], "candidate_category": b[t]["category"], "first_variant_in_csv": baseline if a[t]["source_line"] < b[t]["source_line"] else candidate, "metric_deltas": {f: b[t][f]-a[t][f] if a[t][f] is not None and b[t][f] is not None else None for f in METRICS}} for t in ids]}


def batch(rows, variants):
    return {"summary": describe(rows), "by_variant": group_describe(rows, "variant"), "by_variant_repo": {v: group_describe([r for r in rows if r["variant"] == v], "repo") for v in variants}, "by_variant_resolved": {v: group_describe([r for r in rows if r["variant"] == v], "resolved") for v in variants}, "by_variant_submit": {v: group_describe([r for r in rows if r["variant"] == v], "explicit_submit") for v in variants}}


def main():
    panel, panel_prov = load_csv("research/003/analysis_003_panel.csv")
    explore_file, explore_prov = load_csv("research/003/analysis_003_explore.csv")
    explore = [r for r in explore_file if r["variant"] in ("a0", "b", "c")]
    smoke = [r for r in explore_file if r["variant"] == "a0-smoke"]
    assert len(panel) == 48 and len(explore) == 18 and len(smoke) == 3
    assert set(r["variant"] for r in panel) == {"a0", "b"}
    split_path = ROOT / "research/task_splits.json"
    split_data = json.loads(split_path.read_text())
    split_map = {r["instance_id"]: r for r in split_data["records"]}
    used = {r["task_id"] for r in panel + explore_file}
    heldout = [r for r in split_data["records"] if r["split"] == "heldout"]
    assert not used.intersection(r["instance_id"] for r in heldout)
    assert set(r["task_id"] for r in panel) == set(split_data["dev_panel_ids"])
    loc = [r for r in panel if r["category"] == "localization"]
    budget = [r for r in panel if r["category"] == "budget"]
    group_pairs = [("all", panel), ("smoke_tasks", [r for r in panel if r["smoke_task"]]), ("other_21_tasks", [r for r in panel if not r["smoke_task"]])]
    group_pairs += [(repo, [r for r in panel if r["repo"] == repo]) for repo in sorted({r["repo"] for r in panel})]
    first_groups = defaultdict(list)
    for t in sorted({r["task_id"] for r in panel}):
        task_rows = [r for r in panel if r["task_id"] == t]
        first_groups[task_rows[0]["variant"]].extend(task_rows)
    panel_pairs = paired(panel, "a0", "b")
    time_pairs = sorted(panel_pairs["task_pairs"], key=lambda r: r["metric_deltas"]["agent_duration_seconds"])
    net_agent_seconds = -panel_pairs["metrics"]["agent_duration_seconds"]["candidate_minus_baseline"]["sum"]
    leading_two_reductions = -sum(r["metric_deltas"]["agent_duration_seconds"] for r in time_pairs[:2])
    result = {
        "schema_version": 1,
        "method": {"review_date": "2026-09-30", "source_authority": "Versioned research analysis CSVs; derived annotations, not recovered original traces.", "grain": "One attempt per task_id and variant within batch; cross-batch smoke stays separate.", "denominators": "48 panel attempts = 24 paired tasks; 18 exploration attempts = 6 tasks x 3 variants; extra 3 smoke attempts excluded from exploration.", "statistics": "Arithmetic mean and median. Quantiles use linear interpolation at (n-1)*p. Missing remains null and is never zero-filled. All numeric pair deltas are candidate minus baseline.", "time_note": "Agent duration is context_metrics for exploration/panel; smoke uses trace_max_elapsed according to CSV notes. task_duration is not GPU wall clock and does not include shared startup. First edit is trace-derived proxy; final patch production change is another variable. Neither establishes correct localization.", "cutoff_note": "295s/300s durations, 240/270/280s edits and 50 tools are descriptive flags, not a causal classification. Timing beyond nominal limits is retained.", "repeat_note": "repeated_no_progress_actions is an existing heuristic annotation; explore notes define adj_plus_run_read_run_any. Counts do not prove true no-progress or establish repeat reduction causally.", "paired_note": "All paired comparisons retain original tasks including invalid-verification results. No post-hoc task deletion to improve scores. Repo/smoke/order cuts are descriptive.", "causal_note": "Changes are prompt interventions but one attempt per variant, stochastic sampling, shared server/order/caches, task selection and annotation uncertainty preclude strong causal or generalization claims."},
        "sources": [panel_prov, explore_prov, {"path": "research/task_splits.json", "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest()}],
        "integrity": {"unique_attempt_ids_per_file": True, "unique_variant_task_per_file": True, "panel_complete_pairs": True, "explore_extra_smoke_rows": len(smoke), "unique_used_tasks": len(used), "used_task_splits": dict(Counter(split_map[t]["split"] for t in used)), "heldout_overlap": [], "heldout_tasks": len(heldout), "heldout_snapshot_groups": len({(r["repo"], r["base_commit"]) for r in heldout}), "panel_snapshot_groups": len({(split_map[t]["repo"], split_map[t]["base_commit"]) for t in {r["task_id"] for r in panel}}), "missing_by_file": {"panel": {f: sum(r[f] is None for r in panel) for f in METRICS}, "explore": {f: sum(r[f] is None for r in explore) for f in METRICS}}},
        "panel": batch(panel, ["a0", "b"]),
        "panel_pairings": {key: paired(rows, "a0", "b") for key, rows in group_pairs},
        "panel_order_groups": {f"{first}_first": paired(rows, "a0", "b") for first, rows in first_groups.items()},
        "panel_time_decomposition": {"net_lower_agent_seconds_for_b": net_agent_seconds, "two_largest_reductions": time_pairs[:2], "two_largest_reductions_seconds": leading_two_reductions, "two_largest_share_of_net_difference": leading_two_reductions/net_agent_seconds if net_agent_seconds else None, "pairs_with_agent_delta_abs_le_1s": sum(abs(r["metric_deltas"]["agent_duration_seconds"]) <= 1 for r in time_pairs), "caveat": "Net time difference is arithmetic, not productive efficiency. One leading pair has candidate error at 88.84s; the other submits no final production change. Do not delete them from the headline comparison or infer that faster means better."},
        "explore": batch(explore, ["a0", "b", "c"]),
        "explore_pairings": {"b_vs_a0": paired(explore, "a0", "b"), "c_vs_b": paired(explore, "b", "c")},
        "smoke_batch": describe(smoke),
        "category_analysis": {"by_category": group_describe(panel, "category"), "by_variant_category": {v: group_describe([r for r in panel if r["variant"] == v], "category") for v in ("a0", "b")}, "localization": {"attempts": len(loc), "unique_tasks": sorted({r["task_id"] for r in loc}), "rows": [compact(r) for r in loc]}, "budget": {"attempts": len(budget), "unique_tasks": sorted({r["task_id"] for r in budget}), "rows": [compact(r) for r in budget]}, "localization_budget_union": describe(loc + budget), "shared_task_ids_between_localization_and_budget": sorted({r["task_id"] for r in loc} & {r["task_id"] for r in budget})},
        "review_flags": {
            "first_edit_observed_but_no_final_source_change": [compact(r) for r in panel + explore if r["first_source_edit_at"] is not None and not r["production_source_changed"]],
            "budget_not_agent_end_timeout_or_below_295s": [compact(r) for r in budget if r["agent_end_reason"] != "timeout" or r["agent_duration_seconds"] < 295],
            "non_budget_with_agent_duration_ge_295s": [compact(r) for r in panel if r["category"] != "budget" and r["agent_duration_seconds"] >= 295],
            "explicit_submit_with_no_final_source_change": [compact(r) for r in panel + explore if r["explicit_submit"] and not r["production_source_changed"]],
            "tool_calls_ge_50": [compact(r) for r in panel + explore if r["tool_calls"] >= 50],
            "resolved_without_explicit_submit": [compact(r) for r in panel + explore if r["resolved"] and not r["explicit_submit"]],
            "notes_with_explicit_edit_or_import_error_evidence": [compact(r) for r in panel if any(s in r["notes"] for s in ("agent_broke_import", "wrong_locus", "late_markdown_edits"))],
        },
        "cross_batch_observations": {task: [compact(r) for r in smoke + explore + panel if r["task_id"] == task] for task in sorted(SMOKE)},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "panel": {k: {f: v[f] for f in ("tasks", "baseline_resolved", "candidate_resolved", "wins", "losses")} for k,v in result["panel_pairings"].items()}, "explore": {v: result["explore"]["by_variant"][v]["resolved"] for v in ("a0", "b", "c")}, "localization": [len(loc), len({r["task_id"] for r in loc})], "budget": [len(budget), len({r["task_id"] for r in budget})]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
