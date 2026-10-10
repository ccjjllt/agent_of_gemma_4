#!/usr/bin/env python3
"""CPU scan of exposed traces for naturally shorter complete transactions.

Re-prepares with current worktree code. Measures every complete prefix in the
eligible pool (full causal history, no truncation). Does not overwrite v2
census files. Does not invent issues. No GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "training" / "data"
sys.path.insert(0, str(DATA))

import length_census as lc  # noqa: E402
from prepare_local import prepare  # noqa: E402
from sample_behaviors import PAIRING_RESERVED_ISSUES, pairing_reserved  # noqa: E402

WINDOWS = (2048, 4096, 8192, 16384)
V2_CENSUS = (
    ROOT / "research/005/length-census-v2-20261010/length_report.json",
    ROOT / "research/005/length-census-v2-20261010/sample_list.json",
    ROOT / "research/005/length-census-v2-20261010/train_issues.json",
)
MAIN_RESULTS = Path("/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/results")
KNOWN_6032 = ("fastapi_11194-3f51fb716edd-tx01", 6032)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_v2():
    rows = []
    for path in V2_CENSUS:
        rows.append({"path": lc.rel_to_root(path), "sha256": sha256_file(path),
                     "bytes": path.stat().st_size})
    return rows


def default_results() -> Path:
    local = ROOT / "results"
    if local.exists():
        return local
    if MAIN_RESULTS.exists():
        return MAIN_RESULTS
    raise FileNotFoundError("results dir missing in worktree and main checkout")


def classify_trace(trace_path: Path, results: Path, allowed, heldout, eligible,
                   reserved_issues, reserved_snaps, eligible_by_id):
    task_id = trace_path.stem.removeprefix("trace_")
    result_path = trace_path.parent.parent / "result.json"
    rec = allowed.get(task_id) or {}
    repo, commit = rec.get("repo"), rec.get("base_commit") or rec.get("commit")
    reasons = []
    if task_id in heldout:
        reasons.append("heldout")
    if task_id not in allowed:
        reasons.append("not_in_splits")
    if task_id not in eligible:
        reasons.append("not_eligible")
    if task_id in reserved_issues:
        reasons.append("pairing_reserved")
    if (repo, commit) in reserved_snaps:
        reasons.append("reserved_repo_commit")
    if not result_path.exists():
        reasons.append("missing_result.json")
    usable = not reasons
    return {
        "task_id": task_id,
        "path": results_relative_uri(str(trace_path), results),
        "split": rec.get("split"),
        "repo": repo,
        "commit": commit,
        "eligibility": eligible_by_id.get(task_id, "unlisted"),
        "would_prepare": usable,
        "exclude_reasons": reasons,
        "has_result_json": result_path.exists(),
    }


def inventory_traces(results: Path, splits_path: Path, eligibility_path: Path):
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    records = splits.get("records") or []
    allowed = {r["instance_id"]: r for r in records if r.get("split") != "heldout"}
    heldout = {r["instance_id"] for r in records if r.get("split") == "heldout"}
    reserved_issues, reserved_snaps = pairing_reserved(records)
    elig_data = json.loads(eligibility_path.read_text(encoding="utf-8"))
    tasks = elig_data["tasks"]
    if isinstance(tasks, dict):
        tasks = [dict(v, task_id=k) for k, v in tasks.items()]
    eligible = set()
    eligible_by_id = {}
    for row in tasks:
        tid = row.get("task_id", row.get("instance_id"))
        status = row.get("status") or row.get("eligibility")
        eligible_by_id[tid] = status
        if status == "eligible":
            eligible.add(tid)
    traces = []
    for path in sorted(results.rglob("trace_*.json")):
        traces.append(classify_trace(
            path, results, allowed, heldout, eligible, reserved_issues,
            reserved_snaps, eligible_by_id))
    by_issue = {}
    for row in traces:
        item = by_issue.setdefault(row["task_id"], {
            "task_id": row["task_id"], "n_traces": 0, "would_prepare": 0,
            "eligibility": row["eligibility"], "split": row["split"],
            "repo": row["repo"], "commit": row["commit"],
            "exclude_reasons": sorted(set(row["exclude_reasons"])),
        })
        item["n_traces"] += 1
        if row["would_prepare"]:
            item["would_prepare"] += 1
        for reason in row["exclude_reasons"]:
            if reason not in item["exclude_reasons"]:
                item["exclude_reasons"].append(reason)
    return {
        "results": "results",
        "n_trace_files": len(traces),
        "unique_issues_with_traces": sorted(by_issue),
        "eligible_issues": sorted(eligible),
        "reserved_pairing_issues": sorted(reserved_issues),
        "issues": [by_issue[k] for k in sorted(by_issue)],
        "would_prepare_issues": sorted(k for k, v in by_issue.items() if v["would_prepare"]),
        "excluded_issue_groups": {
            "pairing_reserved_with_traces": sorted(
                k for k, v in by_issue.items() if "pairing_reserved" in v["exclude_reasons"]),
            "not_eligible_with_traces": sorted(
                k for k, v in by_issue.items() if "not_eligible" in v["exclude_reasons"]),
            "heldout_with_traces": sorted(
                k for k, v in by_issue.items() if "heldout" in v["exclude_reasons"]),
        },
    }


def pack_rows(rows):
    lengths = [r["raw_token_length"] for r in rows]
    events = sorted({eid for r in rows for eid in (r.get("recovery_event_ids") or [])})
    return {
        "n": len(rows),
        "independent_issues": sorted({r["issue"] for r in rows}),
        "independent_recovery_events": len(events),
        "recovery_event_ids": events,
        "behavior_counts": lc.behavior_counts(rows),
        "quantiles": lc.length_quantiles(lengths) if lengths else None,
        "histogram": lc.length_histogram(lengths),
        "windows": {str(w): lc.window_coverage(rows, w) for w in WINDOWS},
    }


def results_relative_uri(uri, results: Path):
    if not uri:
        return uri
    try:
        return str(Path("results") / Path(uri).resolve().relative_to(results.resolve()))
    except Exception:
        return lc.repo_relative_uri(uri)


def compact_row(row, v2_ids, results: Path):
    n = row["raw_token_length"]
    return {
        "transaction_id": row["transaction_id"],
        "sample_id": row.get("sample_id"),
        "issue": row.get("issue"),
        "behavior": row.get("behavior"),
        "raw_token_length": n,
        "n_messages": row.get("n_messages"),
        "recovery_event_ids": row.get("recovery_event_ids") or [],
        "action_succeeded": row.get("action_succeeded"),
        "task_resolved": row.get("task_resolved"),
        "source_uri": results_relative_uri(row.get("source_uri"), results),
        "in_v2_capped_sample": row["transaction_id"] in v2_ids,
        "fits": {str(w): n <= w for w in WINDOWS},
    }


def run(args):
    v2_before = snapshot_v2()
    results = Path(args.results)
    splits_path = ROOT / "research/task_splits.json"
    eligibility_path = ROOT / "research/002/verification_eligibility.json"
    tools_path = ROOT / "training/configs/harness_tools.json"
    tokenizer_dir = args.tokenizer or lc.default_tokenizer_dir()
    if not tokenizer_dir:
        raise FileNotFoundError("pass --tokenizer or set P005_GEMMA_TOKENIZER_DIR")
    tok_fp = lc.verify_ct_v2_tokenizer(tokenizer_dir)
    inv = inventory_traces(results, splits_path, eligibility_path)

    with tempfile.TemporaryDirectory(prefix="p005-data-expansion-") as tmp:
        out = Path(tmp)
        prepare_report = prepare(
            results, splits_path, eligibility_path, out, tools_path,
            include_failures=True)
        all_tx = lc.read_jsonl(out / "transactions.jsonl")
        enc = lc._load_encoding()
        tokenizer = enc.load_tokenizer(str(tokenizer_dir), allow_download=False)
        tok_hash = enc.tokenizer_fingerprint(tokenizer)
        tools, _protocol = enc.normalize_tools(
            json.loads(tools_path.read_text(encoding="utf-8")))
        rows, errors = [], []
        total = len(all_tx)
        for i, tx in enumerate(all_tx, 1):
            batch, err = lc.measure_rows([tx], tokenizer, tools, enc, encode_supervised=False)
            rows.extend(batch)
            errors.extend(err)
            if i == 1 or i % 20 == 0 or i == total:
                last = batch[-1] if batch else (err[-1] if err else {})
                print(json.dumps({"measured": i, "of": total,
                                  "transaction_id": last.get("transaction_id"),
                                  "raw_token_length": last.get("raw_token_length"),
                                  "error": last.get("error")}, ensure_ascii=False),
                      flush=True)

    v2_sample = json.loads(
        (ROOT / "research/005/length-census-v2-20261010/sample_list.json").read_text())
    v2_ids = {s["transaction_id"] for s in v2_sample.get("samples") or []}
    v2_issues = json.loads(
        (ROOT / "research/005/length-census-v2-20261010/train_issues.json").read_text())
    known = next((r for r in rows if r["transaction_id"] == KNOWN_6032[0]), None)
    if not known or known["raw_token_length"] != KNOWN_6032[1]:
        raise RuntimeError("tokenizer/template mismatch: expected %s=%d, got %s" % (
            KNOWN_6032[0], KNOWN_6032[1],
            None if known is None else known["raw_token_length"]))

    labeled = [r for r in rows if r.get("behavior") in ("local_edit", "tool_recovery", "verify_close")]
    short = [r for r in rows if r["raw_token_length"] <= 8192]
    short_labeled = [r for r in short if r.get("behavior") in (
        "local_edit", "tool_recovery", "verify_close")]
    new_short = [r for r in short if r["transaction_id"] not in v2_ids]
    new_short_labeled = [r for r in short_labeled if r["transaction_id"] not in v2_ids]
    new_issues = sorted(set(inv["would_prepare_issues"]) - set(
        prepare_report.get("tasks") or []) - set(PAIRING_RESERVED_ISSUES))

    v2_after = snapshot_v2()
    if v2_before != v2_after:
        raise RuntimeError("v2 census files changed during scan")

    list_rows = [compact_row(r, v2_ids, results) for r in sorted(
        rows, key=lambda x: (x["raw_token_length"], x["transaction_id"]))]
    short_labeled_list = [compact_row(r, v2_ids, results) for r in sorted(
        short_labeled, key=lambda x: (x["raw_token_length"], x["transaction_id"]))]
    report = {
        "generated_at": "2026-10-11",
        "schema": "p005-data-expansion-cpu-20261010",
        "gpu_launched": False,
        "did_not_overwrite_v2_census": True,
        "did_not_invent_issues": True,
        "did_not_truncate": True,
        "json_clipped": False,
        "tokenizer": tok_fp,
        "tokenizer_hash": tok_hash,
        "protocol_id": "p005-harness-tools-v1",
        "v2_census_sha256_unchanged": v2_after,
        "trace_inventory": {
            "n_trace_files": inv["n_trace_files"],
            "unique_issues_with_traces": inv["unique_issues_with_traces"],
            "eligible_issues": inv["eligible_issues"],
            "would_prepare_issues": inv["would_prepare_issues"],
            "excluded_issue_groups": inv["excluded_issue_groups"],
            "issues": inv["issues"],
        },
        "prepare": {
            "traces": len(prepare_report.get("traces") or []),
            "rejected": [
                {"path": results_relative_uri(row.get("path"), results),
                 "reason": row.get("reason")}
                for row in (prepare_report.get("rejected") or [])
            ],
            "transactions_before_cap": prepare_report.get("transactions"),
            "tasks": prepare_report.get("tasks"),
            "independent_task_count": prepare_report.get("independent_task_count"),
        },
        "all_complete_prefixes": pack_rows(rows),
        "labeled_behaviors_only": pack_rows(labeled),
        "naturally_short_complete_le_8192": pack_rows(short),
        "naturally_short_labeled_le_8192": pack_rows(short_labeled),
        "naturally_short_labeled_rows": short_labeled_list,
        "new_vs_v2_capped": {
            "short_complete_not_in_v2_sample": pack_rows(new_short),
            "short_labeled_not_in_v2_sample": pack_rows(new_short_labeled),
            "note": "These are earlier complete prefixes of the same 9 eligible issues, "
                    "not new independent issues. Do not treat unlabeled/other prefixes "
                    "as the repair distribution.",
        },
        "independent_issue_growth": {
            "v2_prepare_issues": prepare_report.get("tasks"),
            "v2_train_issues": v2_issues.get("train_issues"),
            "new_eligible_issues_with_traces": new_issues,
            "corpus_cannot_grow_without_new_rollouts": not new_issues,
            "reason": "Only 9 frozen-eligible non-heldout issues have traces. Extra traces "
                      "are reserved pairing IDs, ineligible panel/dev_future tasks, or "
                      "missing result.json. Reserved eight and same repo+commit stay out. "
                      "Do not invent ineligible issues as train data.",
        },
        "known_length_check": {
            "transaction_id": KNOWN_6032[0],
            "expected": KNOWN_6032[1],
            "measured": known["raw_token_length"],
        },
        "measure_errors": errors,
        "reserved_pairing_issues": list(PAIRING_RESERVED_ISSUES),
        "reserved_in_measured": sorted({
            r["issue"] for r in rows if r["issue"] in PAIRING_RESERVED_ISSUES}),
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "trace_inventory.json").write_text(
        json.dumps(inv, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "complete_tx_lengths.json").write_text(
        json.dumps({"generated_at": "2026-10-11", "n": len(list_rows),
                    "transactions": list_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (HERE / "short_labeled.json").write_text(
        json.dumps({"generated_at": "2026-10-11",
                    "note": "Naturally shorter complete prefixes with labeled repair behavior; "
                            "full causal history, no truncation. Not new independent issues.",
                    "n": len(short_labeled_list),
                    "transactions": short_labeled_list}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (HERE / "window_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "n_complete": len(rows),
        "issues": report["all_complete_prefixes"]["independent_issues"],
        "windows": {w: report["all_complete_prefixes"]["windows"][w]["transactions"]
                    for w in ("2048", "4096", "8192", "16384")},
        "short_labeled_le_8192": report["naturally_short_labeled_le_8192"]["n"],
        "new_eligible_issues": new_issues,
        "cannot_grow_without_new_rollouts": not new_issues,
        "v2_untouched": True,
        "report": str(HERE / "window_report.json"),
    }, ensure_ascii=False))
    return 0 if not errors else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(default_results()))
    p.add_argument("--tokenizer", default=os.environ.get("P005_GEMMA_TOKENIZER_DIR")
                   or lc.default_tokenizer_dir())
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, RuntimeError, FileNotFoundError) as err:
        print(json.dumps({"ok": False, "error": str(err)}), file=sys.stderr)
        sys.exit(2)
