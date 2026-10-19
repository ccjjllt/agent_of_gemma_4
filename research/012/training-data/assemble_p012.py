"""Assemble P012 256-issue SFT input (p012-repair-v1, five tools, no submit).

Reads research/012/training-data/data-usage-p012.json order,
takes the first 256 successfully replayed p012-local-2 transactions,
encodes with the 12B CT tokenizer + frozen chat template + five-tool schema,
skips whole-transaction oversize (>8192, no truncation), freezes
encoded-train-12b.jsonl + dataset-manifest.json. Refuses to overwrite.
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TARGET = 256


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, rows):
    Path(path).write_text(
        "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
    )


def main():
    final = HERE / "encoded-train-12b.jsonl"
    if final.exists():
        raise RuntimeError("Frozen output exists; create a new version instead of overwriting")
    usage = json.loads((HERE / "data-usage-p012.json").read_text())
    assert usage["protocol_id"] == "p012-repair-v1", usage.get("protocol_id")
    ordered = usage["selected"]
    results = json.loads((HERE / "production-results-p012.json").read_text())
    processed = {r["instance_id"]: r for r in results}
    transactions = []
    for row in ordered:
        result = processed.get(row["instance_id"])
        if not result or result.get("status") != "local_edits_applied":
            continue
        tx = json.loads((HERE / result["transaction"]).read_text())
        if tx.get("producer_revision") != "p012-local-2":
            raise ValueError("Mixed producer revision: %s" % tx.get("producer_revision"))
        if tx.get("protocol_id") != "p012-repair-v1":
            raise ValueError("Mixed protocol: %s" % tx.get("protocol_id"))
        transactions.append(tx)
    if len(transactions) < TARGET:
        print(json.dumps({"replayed": len(transactions), "needed": TARGET, "frozen": False,
                          "note": "not enough replayed transactions; extend production"}))
        return
    inputs = HERE / "transactions-all-12b.jsonl"
    write_rows(inputs, transactions)
    encoded = HERE / "encoded-all-12b.jsonl"
    subprocess.run(
        [sys.executable, str(ROOT / "training/encoding/gemma_tool_chat.py"), "encode",
         "--input", str(inputs), "--output", str(encoded),
         "--tokenizer", str(ROOT / "training/artifacts/tokenizers/gemma4-12b-ct-v2"),
         "--tools", str(ROOT / "research/012/agent/repair_tools_v1.json"),
         "--chat-template", str(ROOT / "training/configs/chat_templates/gemma4_12b_ct_v2.jinja"),
         "--thinking", "on", "--thought-loss", "mask", "--max-seq-len", "8192",
         "--on-oversize", "skip"],
        check=True,
    )
    available = [json.loads(x) for x in encoded.read_text().splitlines()]
    if len(available) < TARGET:
        print(json.dumps({"available": len(available), "needed": TARGET, "frozen": False,
                          "note": "oversize skipped; extend production limit and rerun"}))
        return
    fresh = available[:TARGET]
    new_ids = [r["meta"]["group"]["issue"] for r in fresh]
    if len(set(new_ids)) != TARGET:
        raise ValueError("Duplicate issue in fresh data")
    write_rows(final, fresh)
    provenance = {key: fresh[0]["meta"][key]
                  for key in ("tokenizer_hash", "chat_template_hash", "tools_hash",
                              "protocol_id", "protocol_hash")}
    for row in fresh:
        if any(row["meta"][key] != value for key, value in provenance.items()):
            raise ValueError("Protocol provenance differs")
    entries = []
    for iid in new_ids:
        folder = HERE / "outputs" / iid
        assets = [p for p in folder.rglob("*") if p.is_file()]
        entries.append({"issue": iid,
                        "repo": next(r["meta"]["group"]["repo"] for r in fresh
                                     if r["meta"]["group"]["issue"] == iid),
                        "tier": "local_edit_application",
                        "behavior_verified": False, "task_resolved": None,
                        "files": [{"path": str(p.relative_to(ROOT)),
                                   "bytes": p.stat().st_size,
                                   "sha256": sha(p)} for p in sorted(assets)]})
    dump(HERE / "admitted-source-manifest.json", entries)
    lengths = sorted(len(r["input_ids"]) for r in fresh)
    manifest = {
        "status": "frozen", "split": "train",
        "encoded": {"path": str(final.relative_to(ROOT)), "sha256": sha(final)},
        "rows": len(fresh), "independent_issues": len(set(new_ids)),
        "independent_repos": len({r["meta"]["group"]["repo"] for r in fresh}),
        "max_sequence_length": max(lengths),
        "training_sequence_limit": 8192,
        "provenance": provenance,
        "input_tokens": sum(lengths),
        "supervised_tokens": sum(r["meta"]["supervised_tokens"] for r in fresh),
        "length_distribution": {
            "min": lengths[0], "median": lengths[len(lengths) // 2],
            "p90": lengths[int(0.9 * (len(lengths) - 1))],
            "max": lengths[-1],
            "le6144": sum(n <= 6144 for n in lengths), "le8192": len(lengths)},
        "selection": {
            "order": "data-usage-p012.json selected order; all replayed p012-local-2 transactions encoded, first 256 whole-transaction encodable (<=8192) taken in frozen order; oversize skipped whole, never truncated",
            "metadata_candidates": len(ordered),
            "processed_candidates": len(results),
            "skipped_oversize_before_taken": len(transactions) - len(available),
            "processed_status_counts": dict(Counter(r["status"] for r in results))},
        "layers": {"p012_repair_local_edit": {
            "rows": TARGET, "behavior_verified": False, "task_resolved": None,
            "note": "five-tool repair-role transactions ending with handover report, no submit supervision"}},
        "truncated": False, "gpu_or_teacher_calls_for_data": 0,
        "training_executed_by_this_producer": False,
        "source_manifest": {
            "path": str((HERE / "admitted-source-manifest.json").relative_to(ROOT)),
            "sha256": sha(HERE / "admitted-source-manifest.json")},
        "frozen_policy": "Do not overwrite this encoded file after training is launched. New work requires another version."}
    dump(HERE / "dataset-manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in
                      ("rows", "independent_issues", "independent_repos", "max_sequence_length",
                       "input_tokens", "supervised_tokens", "length_distribution")},
                     ensure_ascii=False))
    print("manifest:", HERE / "dataset-manifest.json")


if __name__ == "__main__":
    main()
