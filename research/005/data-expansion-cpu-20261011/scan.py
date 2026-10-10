#!/usr/bin/env python3
"""CPU inventory of local traces for new independent issues and recoveries.

Growth is versus the frozen v2 issue set, not this run's own prepare list.
A file is new only if its sha256 is absent from the 20261010 inventory.
Does not overwrite v2 census files. Does not invent issues. No GPU. No new
rollouts. If no new eligible traces exist, do not re-prepare the old 214
prefixes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "training" / "data"
sys.path.insert(0, str(DATA))

import length_census as lc  # noqa: E402
from prepare_local import prepare  # noqa: E402
from sample_behaviors import PAIRING_RESERVED_ISSUES, pairing_reserved, exclude_reserved  # noqa: E402

WINDOWS = (2048, 4096, 8192, 16384)
V2_CENSUS = (
    ROOT / "research/005/length-census-v2-20261010/length_report.json",
    ROOT / "research/005/length-census-v2-20261010/sample_list.json",
    ROOT / "research/005/length-census-v2-20261010/train_issues.json",
)
EXPECTED_V2_SHA256 = {
    "research/005/length-census-v2-20261010/length_report.json":
        "2a219102b6779f5ea9aa37e696c1089e67cc0283b77c5fe23f73720c246eef76",
    "research/005/length-census-v2-20261010/sample_list.json":
        "05cbf49bcd7b67cf8fc8d98937a35031d6192b183a3ac2dcdc8adcd138ee79c1",
    "research/005/length-census-v2-20261010/train_issues.json":
        "b25fe2c97599d4d15d5e0c32c69b1ef6af4a7f6a95a662109b446e865a9159b1",
}
PREV_INVENTORY = ROOT / "research/005/data-expansion-cpu-20261010/trace_inventory.json"
MAIN_RESULTS = Path("/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/results")
PROTOCOL_ID = "p005-harness-tools-v1"
SCHEMA = "p005-data-expansion-cpu-20261011"
CT_V2_TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
CT_V2_TOKENIZER_BYTES = 32169626


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_v2():
    rows = []
    for path in V2_CENSUS:
        rel = lc.rel_to_root(path)
        digest = sha256_file(path)
        expected = EXPECTED_V2_SHA256.get(rel)
        if expected and digest != expected:
            raise RuntimeError("v2 census hash changed: %s" % rel)
        rows.append({"path": rel, "sha256": digest, "bytes": path.stat().st_size})
    return rows


def default_results() -> Path:
    local = ROOT / "results"
    if local.exists():
        return local
    if MAIN_RESULTS.exists():
        return MAIN_RESULTS
    raise FileNotFoundError("results dir missing in worktree and main checkout")


def results_relative_uri(uri, results: Path):
    if not uri:
        return uri
    try:
        return str(Path("results") / Path(uri).resolve().relative_to(results.resolve()))
    except Exception:
        return lc.repo_relative_uri(uri)


def measured_pool(transactions, split_records):
    """Exclude reserved issue IDs and snapshots from the actual measured rows."""
    issues, snapshots = pairing_reserved(split_records)
    return exclude_reserved(transactions, issues, snapshots)


def new_issue_ids(measured_rows, historical_v2_report):
    """Compare this scan with frozen v2, not with this scan's own preparation."""
    historical = set((historical_v2_report.get("prepare") or {}).get("tasks") or [])
    current = {row["issue"] for row in measured_rows}
    return sorted(current - historical - set(PAIRING_RESERVED_ISSUES))


def collect_stored_sha256s(inventory):
    found = set()
    for row in inventory.get("files") or []:
        digest = row.get("sha256")
        if isinstance(digest, str) and len(digest) == 64:
            found.add(digest)
    return found


def apply_previous_inventory(files, prev_inventory):
    """A file is new only if its sha256 is not in the 20261010 inventory.

    That inventory stored issue counts, not per-file hashes. When counts still
    match, current hashes are the reconstructed previous set. Otherwise keep the
    oldest n_traces per issue as already counted, then apply sha256 membership
    so copies of old content are not treated as new files.
    """
    stored = collect_stored_sha256s(prev_inventory)
    prev_n = {item["task_id"]: int(item["n_traces"])
              for item in prev_inventory.get("issues") or []}
    if stored:
        hashes, source = stored, "stored_files_sha256"
    else:
        current_n = {}
        for row in files:
            current_n[row["task_id"]] = current_n.get(row["task_id"], 0) + 1
        counts_match = (
            len(files) == int(prev_inventory.get("n_trace_files") or 0)
            and sorted({row["task_id"] for row in files})
            == sorted(prev_inventory.get("unique_issues_with_traces") or [])
            and prev_n == current_n
        )
        if counts_match:
            hashes = {row["sha256"] for row in files}
            source = "reconstructed_counts_match"
        else:
            hashes = set()
            grouped = {}
            for row in files:
                grouped.setdefault(row["task_id"], []).append(row)
            for task_id, group in grouped.items():
                keep = prev_n.get(task_id, 0)
                ordered = sorted(group, key=lambda r: (r.get("mtime_ns") or 0, r["path"]))
                hashes.update(item["sha256"] for item in ordered[:keep])
            source = "reconstructed_oldest_n_per_issue"
    for row in files:
        row["in_previous_inventory"] = row["sha256"] in hashes
    return hashes, source


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
    st = trace_path.stat()
    return {
        "task_id": task_id,
        "path": results_relative_uri(str(trace_path), results),
        "sha256": sha256_file(trace_path),
        "bytes": st.st_size,
        "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
        "split": rec.get("split"),
        "repo": repo,
        "commit": commit,
        "eligibility": eligible_by_id.get(task_id, "unlisted"),
        "pairing_reserved": task_id in reserved_issues,
        "reserved_repo_commit": (repo, commit) in reserved_snaps,
        "would_prepare": usable,
        "exclude_reasons": reasons,
        "has_result_json": result_path.exists(),
        "abs_path": str(trace_path.resolve()),
        "result_json_path": str(result_path.resolve()) if result_path.exists() else None,
    }


def load_split_eligibility(splits_path: Path, eligibility_path: Path):
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
    return records, allowed, heldout, reserved_issues, reserved_snaps, eligible, eligible_by_id


def inventory_traces(results: Path, splits_path: Path, eligibility_path: Path,
                     prev_inventory=None):
    (records, allowed, heldout, reserved_issues, reserved_snaps,
     eligible, eligible_by_id) = load_split_eligibility(splits_path, eligibility_path)
    traces = []
    for path in sorted(results.rglob("trace_*.json")):
        traces.append(classify_trace(
            path, results, allowed, heldout, eligible, reserved_issues,
            reserved_snaps, eligible_by_id))
    prev_inventory = prev_inventory or {}
    prev_hashes, prev_source = apply_previous_inventory(traces, prev_inventory)
    by_issue = {}
    for row in traces:
        item = by_issue.setdefault(row["task_id"], {
            "task_id": row["task_id"], "n_traces": 0, "would_prepare": 0,
            "n_new_files": 0, "eligibility": row["eligibility"], "split": row["split"],
            "repo": row["repo"], "commit": row["commit"],
            "exclude_reasons": sorted(set(row["exclude_reasons"])),
        })
        item["n_traces"] += 1
        if row["would_prepare"]:
            item["would_prepare"] += 1
        if not row["in_previous_inventory"]:
            item["n_new_files"] += 1
        for reason in row["exclude_reasons"]:
            if reason not in item["exclude_reasons"]:
                item["exclude_reasons"].append(reason)
    public_files = []
    for row in traces:
        public_files.append({k: row[k] for k in (
            "task_id", "path", "sha256", "bytes", "split", "repo", "commit",
            "eligibility", "pairing_reserved", "reserved_repo_commit",
            "would_prepare", "exclude_reasons", "has_result_json",
            "in_previous_inventory")})
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
        "files": public_files,
        "previous_inventory": {
            "path": lc.rel_to_root(PREV_INVENTORY),
            "n_trace_files": prev_inventory.get("n_trace_files"),
            "eligible_issues": prev_inventory.get("eligible_issues"),
            "stored_file_sha256_count": len(collect_stored_sha256s(prev_inventory)),
            "applied_sha256_count": len(prev_hashes),
            "sha256_source": prev_source,
        },
        "n_new_files": sum(1 for row in traces if not row["in_previous_inventory"]),
        "_private_files": traces,
        "_records": records,
        "_reserved_snaps": sorted(
            [list(item) for item in reserved_snaps], key=lambda x: (x[0] or "", x[1] or "")),
    }


def already_counted_issues(historical_v2_report, prev_inventory):
    historical = set((historical_v2_report.get("prepare") or {}).get("tasks") or [])
    previous = set(prev_inventory.get("would_prepare_issues") or [])
    previous.update(prev_inventory.get("eligible_issues") or [])
    return historical | previous


def classify_new_file(row, counted_issues, historical_v2_issues):
    new_file = not row["in_previous_inventory"]
    prefix = bool(new_file and row["task_id"] in counted_issues)
    independent = bool(
        new_file and row.get("would_prepare") and row["task_id"] not in historical_v2_issues
        and row["task_id"] not in PAIRING_RESERVED_ISSUES
        and not row.get("pairing_reserved") and not row.get("reserved_repo_commit"))
    note = None
    if new_file and prefix:
        note = "another prefix of an already-counted issue; not an independent issue"
    elif new_file and not row.get("would_prepare"):
        note = "new file is not an eligible non-reserved protocol-compatible trace"
    elif independent:
        note = "candidate independent issue pending protocol-compatible prepare"
    return {
        "new_file": new_file,
        "prefix_of_already_counted_issue": prefix,
        "independent_issue_candidate": independent,
        "note": note,
    }


def eligible_non_reserved_without_unused(records, eligible, reserved_issues,
                                         reserved_snaps, files):
    unused_ok = {
        row["task_id"] for row in files
        if (not row["in_previous_inventory"]) and row.get("would_prepare")
    }
    out = []
    for rec in records:
        tid = rec["instance_id"]
        if rec.get("split") == "heldout" or tid not in eligible:
            continue
        repo, commit = rec.get("repo"), rec.get("base_commit") or rec.get("commit")
        if tid in reserved_issues or (repo, commit) in reserved_snaps:
            continue
        if tid not in unused_ok:
            out.append(tid)
    return sorted(out)


def non_reserved_split_without_compatible_trace(records, reserved_issues,
                                                reserved_snaps, files):
    compatible = {row["task_id"] for row in files if row.get("would_prepare")}
    out = []
    for rec in records:
        tid = rec["instance_id"]
        if rec.get("split") == "heldout":
            continue
        repo, commit = rec.get("repo"), rec.get("base_commit") or rec.get("commit")
        if tid in reserved_issues or (repo, commit) in reserved_snaps:
            continue
        if tid not in compatible:
            out.append({"issue": tid, "split": rec.get("split"), "repo": rec.get("repo")})
    return out


def stage_new_traces(new_rows, results: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    staged = []
    for row in new_rows:
        src = Path(row["abs_path"])
        rel = src.resolve().relative_to(results.resolve())
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        result_src = src.parent.parent / "result.json"
        if result_src.exists():
            result_rel = result_src.resolve().relative_to(results.resolve())
            result_dest = dest / result_rel
            result_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result_src, result_dest)
        staged.append(str(target))
    return staged


def pack_rows(rows):
    lengths = [r["raw_token_length"] for r in rows]
    events = sorted({eid for r in rows for eid in (r.get("recovery_event_ids") or [])})
    return {
        "n": len(rows),
        "independent_issues": sorted({r["issue"] for r in rows}),
        "independent_recovery_events": len(events),
        "recovery_event_ids": events,
        "behavior_counts": lc.behavior_counts(rows) if rows else None,
        "quantiles": lc.length_quantiles(lengths) if lengths else None,
        "histogram": lc.length_histogram(lengths) if lengths else None,
        "windows": {str(w): lc.window_coverage(rows, w) for w in WINDOWS} if rows else None,
    }


def compact_row(row, results: Path):
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
        "source_sha256": row.get("source_sha256"),
        "fits": {str(w): n <= w for w in WINDOWS},
    }


def measure_new_traces(new_rows, results: Path, splits_path: Path,
                       eligibility_path: Path, tools_path: Path, tokenizer_dir):
    tok_fp = lc.verify_ct_v2_tokenizer(tokenizer_dir)
    if tok_fp["sha256"] != CT_V2_TOKENIZER_SHA256 or tok_fp["bytes"] != CT_V2_TOKENIZER_BYTES:
        raise RuntimeError("tokenizer.json mismatch: %s" % tok_fp)
    with tempfile.TemporaryDirectory(prefix="p005-data-expansion-new-") as tmp:
        staged = Path(tmp) / "results"
        stage_new_traces(new_rows, results, staged)
        out = Path(tmp) / "prepared"
        prepare_report = prepare(
            staged, splits_path, eligibility_path, out, tools_path,
            include_failures=True)
        all_tx, dropped_reserved = measured_pool(
            lc.read_jsonl(out / "transactions.jsonl"),
            json.loads(splits_path.read_text(encoding="utf-8")).get("records") or [])
        enc = lc._load_encoding()
        tokenizer = enc.load_tokenizer(str(tokenizer_dir), allow_download=False)
        tok_hash = enc.tokenizer_fingerprint(tokenizer)
        tools, protocol = enc.normalize_tools(
            json.loads(tools_path.read_text(encoding="utf-8")))
        if protocol and protocol != PROTOCOL_ID:
            raise RuntimeError("protocol must remain %s, got %s" % (PROTOCOL_ID, protocol))
        rows, errors = lc.measure_rows(
            all_tx, tokenizer, tools, enc, encode_supervised=False)
    return {
        "tokenizer": tok_fp,
        "tokenizer_hash": tok_hash,
        "prepare": prepare_report,
        "rows": rows,
        "errors": errors,
        "dropped_reserved": dropped_reserved,
    }


def write_note(report):
    growth = report["independent_issue_growth"]
    issues = report["eligible_non_reserved_issues_without_unused_protocol_compatible_trace"]
    lines = [
        "# CPU independent-issue expansion (2026-10-11)",
        "",
        "No GPU. No new rollouts. v2 census files not overwritten. Reserved eight "
        "IDs and same repo+commit stay out. Protocol `%s`." % PROTOCOL_ID,
        "",
        "**Independent issues cannot grow without new rollouts.** "
        "Walked %d `trace_*.json` files under `results/` (20261010 inventory: %s traces, "
        "%s eligible issues). New files by sha256: **%d**. New independent non-reserved "
        "issues versus frozen v2: **%s**. New recovery events measured: **%s**."
        % (report["trace_inventory"]["n_trace_files"],
           report["previous_inventory"]["n_trace_files"],
           len(report["previous_inventory"].get("eligible_issues") or []),
           report["n_new_files"],
           growth["new_eligible_issues_with_traces"] or "none",
           report.get("new_recovery_event_count", 0)),
        "",
    ]
    if report.get("did_not_reprepare_old_prefixes"):
        lines.append(
            "Did not re-prepare the old 214 complete prefixes; no new eligible "
            "protocol-compatible traces appeared.")
        lines.append("")
    if report.get("new_files"):
        lines.append("New files (sha256 not in the 20261010 inventory):")
        for row in report["new_files"]:
            extra = row.get("note") or ""
            lines.append("- `%s` issue `%s` sha256 `%s` %s" % (
                row["path"], row["task_id"], row["sha256"][:12], extra))
        lines.append("")
    lines.append(
        "Eligible non-reserved issue ids in splits that still have **no** unused "
        "protocol-compatible trace:")
    lines.append("")
    for tid in issues:
        lines.append("- `%s`" % tid)
    lines.append("")
    n_gap = len(report.get("non_reserved_split_issues_with_no_protocol_compatible_trace") or [])
    lines.append(
        "%d other non-reserved, non-heldout split issues have zero protocol-compatible "
        "traces (not P002-eligible; listed in `window_report.json`). Do not invent them "
        "into train." % n_gap)
    lines.append("")
    lines.append(growth["reason"])
    lines.append("")
    (HERE / "note.md").write_text("\n".join(lines), encoding="utf-8")


def public_inventory(inv):
    return {k: inv[k] for k in inv if not k.startswith("_")}


def run(args):
    v2_before = snapshot_v2()
    prev_inventory_sha = sha256_file(PREV_INVENTORY)
    prev_inventory = json.loads(PREV_INVENTORY.read_text(encoding="utf-8"))
    results = Path(args.results)
    splits_path = ROOT / "research/task_splits.json"
    eligibility_path = ROOT / "research/002/verification_eligibility.json"
    tools_path = ROOT / "training/configs/harness_tools.json"
    inv = inventory_traces(results, splits_path, eligibility_path, prev_inventory)
    files = inv["_private_files"]
    records = inv["_records"]
    historical_v2_report = json.loads(V2_CENSUS[0].read_text(encoding="utf-8"))
    v2_issues_file = json.loads(V2_CENSUS[2].read_text(encoding="utf-8"))
    historical_ids = set((historical_v2_report.get("prepare") or {}).get("tasks") or [])
    counted = already_counted_issues(historical_v2_report, prev_inventory)
    reserved_issues, reserved_snaps = pairing_reserved(records)

    new_files = []
    for row in files:
        extra = classify_new_file(row, counted, historical_ids)
        if extra["new_file"]:
            item = {k: row[k] for k in (
                "task_id", "path", "sha256", "bytes", "split", "repo", "commit",
                "eligibility", "pairing_reserved", "reserved_repo_commit",
                "would_prepare", "exclude_reasons", "has_result_json",
                "in_previous_inventory")}
            item.update(extra)
            new_files.append(item)

    new_eligible_private = [
        row for row in files
        if (not row["in_previous_inventory"]) and row.get("would_prepare")
    ]
    without_unused = eligible_non_reserved_without_unused(
        records, set(inv["eligible_issues"]), reserved_issues, reserved_snaps, files)
    no_compatible = non_reserved_split_without_compatible_trace(
        records, reserved_issues, reserved_snaps, files)

    measured = None
    rows = []
    errors = []
    dropped_reserved = []
    tokenizer_info = None
    tokenizer_hash = None
    if new_eligible_private:
        tokenizer_dir = args.tokenizer or lc.default_tokenizer_dir()
        if not tokenizer_dir:
            raise FileNotFoundError("pass --tokenizer or set P005_GEMMA_TOKENIZER_DIR")
        measured = measure_new_traces(
            new_eligible_private, results, splits_path, eligibility_path,
            tools_path, tokenizer_dir)
        rows = measured["rows"]
        errors = measured["errors"]
        dropped_reserved = measured["dropped_reserved"]
        tokenizer_info = measured["tokenizer"]
        tokenizer_hash = measured["tokenizer_hash"]
        new_issues = new_issue_ids(rows, historical_v2_report)
        recovery_ids = sorted({eid for r in rows for eid in (r.get("recovery_event_ids") or [])})
        prepared_issues = sorted({r["issue"] for r in rows})
        prefix_only = sorted(
            tid for tid in prepared_issues if tid in counted and tid not in new_issues)
    else:
        new_issues = new_issue_ids([], historical_v2_report)
        recovery_ids = []
        prepared_issues = []
        prefix_only = sorted({
            row["task_id"] for row in new_files
            if row.get("prefix_of_already_counted_issue")})

    v2_after = snapshot_v2()
    if v2_before != v2_after:
        raise RuntimeError("v2 census files changed during scan")
    if sha256_file(PREV_INVENTORY) != prev_inventory_sha:
        raise RuntimeError("20261010 inventory changed during scan")

    growth_reason = (
        "New eligible independent issues were measured relative to frozen v2."
        if new_issues else
        "Independent issues cannot grow without new rollouts. No new eligible "
        "non-reserved protocol-compatible issue id appeared beyond the historical "
        "v2 issue set. Repeated shorter prefixes of the same 9 issues are not "
        "independent issues. Reserved issues and snapshots remain excluded."
    )
    compact_new = [compact_row(r, results) for r in sorted(
        rows, key=lambda x: (x["raw_token_length"], x["transaction_id"]))] if rows else []
    report = {
        "generated_at": "2026-10-11",
        "schema": SCHEMA,
        "gpu_launched": False,
        "did_not_overwrite_v2_census": True,
        "did_not_invent_issues": True,
        "did_not_truncate": True,
        "did_not_launch_new_rollouts": True,
        "did_not_reprepare_old_prefixes": not bool(new_eligible_private),
        "json_clipped": False,
        "protocol_id": PROTOCOL_ID,
        "tokenizer": tokenizer_info,
        "tokenizer_hash": tokenizer_hash,
        "v2_census_sha256_unchanged": v2_after,
        "previous_inventory": inv["previous_inventory"],
        "trace_inventory": {
            "n_trace_files": inv["n_trace_files"],
            "unique_issues_with_traces": inv["unique_issues_with_traces"],
            "eligible_issues": inv["eligible_issues"],
            "would_prepare_issues": inv["would_prepare_issues"],
            "excluded_issue_groups": inv["excluded_issue_groups"],
            "issues": inv["issues"],
            "n_new_files": inv["n_new_files"],
        },
        "n_new_files": inv["n_new_files"],
        "new_files": new_files,
        "new_eligible_traces": [
            {k: row[k] for k in ("task_id", "path", "sha256", "would_prepare",
                                 "pairing_reserved", "reserved_repo_commit")}
            for row in new_files if row.get("would_prepare")
        ],
        "new_usable_trace_lengths": compact_new,
        "new_usable_summary": pack_rows(rows) if rows else {
            "n": 0, "independent_issues": [], "independent_recovery_events": 0,
            "recovery_event_ids": [],
        },
        "new_recovery_event_ids": recovery_ids,
        "new_recovery_event_count": len(recovery_ids),
        "new_files_that_are_prefixes_of_already_counted_issues": prefix_only,
        "independent_issue_growth": {
            "v2_prepare_issues": (historical_v2_report.get("prepare") or {}).get("tasks"),
            "v2_train_issues": v2_issues_file.get("train_issues"),
            "inventory_would_prepare_issues": inv["would_prepare_issues"],
            "current_measured_issues": prepared_issues,
            "new_eligible_issues_with_traces": new_issues,
            "corpus_cannot_grow_without_new_rollouts": not new_issues,
            "reason": growth_reason,
        },
        "eligible_non_reserved_issues_without_unused_protocol_compatible_trace":
            without_unused,
        "non_reserved_split_issues_with_no_protocol_compatible_trace": no_compatible,
        "measure_errors": errors,
        "reserved_pairing_issues": list(PAIRING_RESERVED_ISSUES),
        "reserved_transactions_excluded_before_measurement": dropped_reserved,
        "reserved_in_measured": sorted({
            r["issue"] for r in rows if r["issue"] in PAIRING_RESERVED_ISSUES}),
        "prepare_of_new_traces": None if not measured else {
            "traces": len((measured["prepare"] or {}).get("traces") or []),
            "rejected": [
                {"path": results_relative_uri(row.get("path"), results),
                 "reason": row.get("reason")}
                for row in ((measured["prepare"] or {}).get("rejected") or [])
            ],
            "transactions_before_cap": (measured["prepare"] or {}).get("transactions"),
            "tasks": (measured["prepare"] or {}).get("tasks"),
            "independent_task_count": (measured["prepare"] or {}).get("independent_task_count"),
        },
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "trace_inventory.json").write_text(
        json.dumps(public_inventory(inv), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (HERE / "window_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_note(report)
    print(json.dumps({
        "ok": True,
        "n_trace_files": inv["n_trace_files"],
        "n_new_files": inv["n_new_files"],
        "new_eligible_traces": len(new_eligible_private),
        "new_eligible_issues": new_issues,
        "cannot_grow_without_new_rollouts": not new_issues,
        "did_not_reprepare_old_prefixes": report["did_not_reprepare_old_prefixes"],
        "eligible_without_unused_trace": without_unused,
        "v2_untouched": True,
        "gpu_launched": False,
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
