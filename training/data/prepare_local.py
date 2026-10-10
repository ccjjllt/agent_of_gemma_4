#!/usr/bin/env python3
"""Import existing ATIF artifacts without reading task answers or running models.

This is a small, exposed-task diagnostic corpus, NOT an independent evaluation
set or a substitute for multi-repository training data. Default: resolved traces
on previously eligible tasks. No snippets, observations or thoughts are invented.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from canonicalize import canonicalize_trace, load_tools, SCHEMA_PATH
from transactions import slice_transactions, build_transaction

ROOT = Path(__file__).resolve().parents[2]


def prepare(results, splits, eligibility, out, tools_path=SCHEMA_PATH, include_failures=False):
    records = json.loads(Path(splits).read_text())["records"]
    allowed = {r["instance_id"]: r for r in records if r["split"] != "heldout"}
    heldout = {r["instance_id"] for r in records if r["split"] == "heldout"}
    eligible = set()
    data = json.loads(Path(eligibility).read_text())
    tasks = data["tasks"]
    if isinstance(tasks, dict):
        tasks = [dict(v, task_id=k) for k, v in tasks.items()]
    for row in tasks:
        if row.get("status") == "eligible" or row.get("eligibility") == "eligible":
            eligible.add(row.get("task_id", row.get("instance_id")))
    tools = load_tools(tools_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"scope": "exposed historical traces; diagnostic corpus, no new rollouts",
              "heldout_excluded": True, "eligible_only": True, "resolved_only": not include_failures,
              "traces": [], "rejected": [], "transactions": 0, "tasks": []}
    seen, used = set(), set()
    with (out/"canonical.jsonl").open("w") as canonical, (out/"transactions.jsonl").open("w") as txout:
        for trace_path in sorted(Path(results).rglob("trace_*.json")):
            task_id = trace_path.stem.removeprefix("trace_")
            # Filter using filenames and the metadata-only split registry before parsing content.
            if task_id in heldout or task_id not in allowed or task_id not in eligible:
                continue
            result_path = trace_path.parent.parent / "result.json"
            if not result_path.exists():
                report["rejected"].append({"path": str(trace_path), "reason": "missing result.json"})
                continue
            result = json.loads(result_path.read_text())
            if not include_failures and result.get("resolved") is not True:
                continue
            raw_bytes = trace_path.read_bytes()
            digest = hashlib.sha256(raw_bytes).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            record = allowed[task_id]
            sample_id = f"{task_id}-{digest[:12]}"
            payload = {"trace_id": sample_id, "trajectory": json.loads(raw_bytes),
                "origin_kind": "student", "synthetic": False,
                "group": {"issue": task_id, "repo": record["repo"],
                          "commit": record["base_commit"], "split": "diagnostic"},
                "source": {"uri": str(trace_path.resolve()), "sha256": digest,
                           "license": "competition-restricted; not for automatic publication",
                           "teacher": "gemma-4-31b-it-qat-w4a16-ct",
                           "channel_basis": "ATIF extra.event_type when present"},
                "observed_outcome": {"task_resolved": result.get("resolved"),
                                     "action_succeeded": None},
                "behavior": "local_edit"}
            try:
                sample = canonicalize_trace(payload, tools)
                if sample["needs_replay"]:
                    raise ValueError("incompatible tool protocol in trace")
                groups, _ = slice_transactions(sample)
                transactions = [build_transaction(sample, n, positions)[0]
                                for n, (positions, _) in enumerate(groups, 1)]
            except (ValueError, KeyError, TypeError) as e:
                report["rejected"].append({"path": str(trace_path), "reason": str(e)})
                continue
            canonical.write(json.dumps(sample, ensure_ascii=False)+"\n")
            for tx in transactions:
                txout.write(json.dumps(tx, ensure_ascii=False)+"\n")
            report["transactions"] += len(transactions)
            report["traces"].append({"sample_id": sample_id, "path": str(trace_path),
                                     "sha256": digest, "transactions": len(transactions)})
            used.add(task_id)
    report["tasks"] = sorted(used)
    report["independent_task_count"] = len(used)
    (out/"manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(ROOT/"results"))
    p.add_argument("--splits", default=str(ROOT/"research/task_splits.json"))
    p.add_argument("--eligibility", default=str(ROOT/"research/002/verification_eligibility.json"))
    p.add_argument("--tools", default=str(SCHEMA_PATH))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--include-failures", action="store_true")
    a = p.parse_args()
    report = prepare(a.results, a.splits, a.eligibility, a.output_dir, a.tools, a.include_failures)
    print(json.dumps({"traces": len(report["traces"]), "tasks": report["independent_task_count"],
                      "transactions": report["transactions"], "rejected": len(report["rejected"]),
                      "manifest": str(Path(a.output_dir)/"manifest.json")}))
    return 0 if report["transactions"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as e:
        print(f"prepare_local: {e}", file=sys.stderr)
        sys.exit(2)
