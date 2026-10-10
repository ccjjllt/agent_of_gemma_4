#!/usr/bin/env python3
"""Census full-template token lengths and choose a training window.

CPU only. No JSON clipping, no history truncation, no 31B weight load.
Default prepare path stays resolved-only; this runner passes --include-failures.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from prepare_local import prepare
from split_registry import assign, materialize, register

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = (2048, 4096, 8192, 16384)
COMPARISON_ONLY_WINDOWS = (2048,)
CT_V2_TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
CT_V2_TOKENIZER_BYTES = 32169626
PROTOCOL_ID = "p005-harness-tools-v1"
PLACEHOLDER_T0_SIX_TOOLS_TOKENS = 1435


def _load_encoding():
    path = Path(__file__).resolve().parents[1] / "encoding" / "gemma_tool_chat.py"
    spec = importlib.util.spec_from_file_location("p005_encoding_census", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quantile(sorted_vals, p):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1 or p <= 0:
        return int(sorted_vals[0])
    if p >= 100:
        return int(sorted_vals[-1])
    rank = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return int(round(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac))


def length_quantiles(lengths, percents=(0, 25, 50, 75, 90, 95, 100)):
    ordered = sorted(int(x) for x in lengths)
    return {("p%d" % p): quantile(ordered, p) for p in percents}


def window_coverage(rows, window):
    fit = [r for r in rows if r["raw_token_length"] <= window]
    issues = sorted({r["issue"] for r in fit})
    return {
        "transactions": len(fit),
        "independent_issues": len(issues),
        "issues": issues,
        "comparison_only": window in COMPARISON_ONLY_WINDOWS,
    }


def length_histogram(lengths):
    counts = {"(0,2048]": 0, "(2048,4096]": 0, "(4096,8192]": 0,
              "(8192,16384]": 0, "(16384,inf)": 0}
    for n in lengths:
        if n <= 2048:
            counts["(0,2048]"] += 1
        elif n <= 4096:
            counts["(2048,4096]"] += 1
        elif n <= 8192:
            counts["(4096,8192]"] += 1
        elif n <= 16384:
            counts["(8192,16384]"] += 1
        else:
            counts["(16384,inf)"] += 1
    return counts


def decide_max_seq_len(train_lengths):
    """Choose a data window; CPU length coverage does not prove GPU capacity."""
    n = len(train_lengths)
    train_counts = {w: sum(1 for x in train_lengths if x <= w) for w in WINDOWS}
    base = {
        "n_train_transactions": n,
        "window_counts_train": train_counts,
        "majority_fit_4096": False,
        "majority_fit_8192": False,
        "majority_fit_16384": False,
        "four_k_capacity_unproven": True,
        "coverage_gap": None,
        "placeholder_t0_six_tools_tokens": PLACEHOLDER_T0_SIX_TOOLS_TOKENS,
        "note_2048": "2048 is a comparison bucket only; T0+six tools+official instructions "
                     "already use about %d tokens on a tiny placeholder issue" % PLACEHOLDER_T0_SIX_TOOLS_TOKENS,
    }
    if n == 0:
        base.update(chosen_max_seq_len=4096, trainable_ok=False,
                    decision_reason="no train transactions after whole-repo split and per-issue cap")
        return base
    maj4096 = train_counts[4096] * 2 > n
    maj8192 = train_counts[8192] * 2 > n
    maj16384 = train_counts[16384] * 2 > n
    base["majority_fit_4096"] = maj4096
    base["majority_fit_8192"] = maj8192
    base["majority_fit_16384"] = maj16384
    if maj4096:
        base.update(
            chosen_max_seq_len=4096, trainable_ok=True, four_k_capacity_unproven=True,
            decision_reason="majority of representative train transactions fully encode at 4096")
        return base
    if maj8192:
        base.update(
            chosen_max_seq_len=8192, trainable_ok=True, four_k_capacity_unproven=True,
            coverage_gap={"window": 4096, "fit": train_counts[4096], "n": n},
            decision_reason="majority exceed 4096 but 8192 covers the main train set; "
                            "4K capacity is not proven in this round")
        return base
    base.update(
        chosen_max_seq_len=8192, trainable_ok=False, four_k_capacity_unproven=True,
        coverage_gap={"window": 8192, "fit": train_counts[8192], "n": n,
                      "fit_16384": train_counts[16384],
                      "majority_fit_16384": maj16384},
        decision_reason="majority of representative train transactions exceed 8192; "
                        "gap recorded; no truncation; 2048 is comparison-only; "
                        "16384 is a census bucket only and is not chosen as the training window")
    return base


def verify_ct_v2_tokenizer(tokenizer_dir):
    path = Path(tokenizer_dir) / "tokenizer.json"
    if not path.is_file():
        raise ValueError("tokenizer.json missing in %s" % tokenizer_dir)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != CT_V2_TOKENIZER_BYTES or digest != CT_V2_TOKENIZER_SHA256:
        raise ValueError("CT v2 tokenizer.json mismatch: bytes=%d sha256=%s (expected %d / %s)"
                         % (len(data), digest, CT_V2_TOKENIZER_BYTES, CT_V2_TOKENIZER_SHA256))
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": digest}


def heldout_and_panel(splits_path):
    data = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    heldout = sorted(r["instance_id"] for r in data["records"] if r["split"] == "heldout")
    panel = list(data.get("dev_panel_ids") or [])
    return heldout, panel


def read_jsonl(path):
    rows = []
    text = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def issues_from_registry(registry, split):
    found = []
    seen = set()
    for group in registry["groups"].values():
        if group.get("split") != split:
            continue
        issue = group["issue"]
        if issue not in seen:
            seen.add(issue)
            found.append(issue)
    return sorted(found)


def measure_rows(transactions, tokenizer, tools, enc):
    rows, errors = [], []
    for tx in transactions:
        group = tx.get("group") or {}
        item = {
            "transaction_id": tx.get("transaction_id"),
            "sample_id": tx.get("sample_id"),
            "issue": group.get("issue"),
            "repo": group.get("repo"),
            "commit": group.get("commit"),
            "split": group.get("split"),
            "behavior": tx.get("behavior"),
            "n_messages": len(tx.get("messages") or []),
            "target_assistant_steps": list(tx.get("target_assistant_steps") or []),
        }
        try:
            n = enc.count_template_tokens(
                tx, tokenizer, tools, enable_thinking=True, preserve_thinking=True)
        except (enc.EncodingError, ValueError, TypeError, KeyError) as err:
            item["error"] = str(err)
            errors.append(item)
            continue
        item["raw_token_length"] = int(n)
        item["fits"] = {str(w): n <= w for w in WINDOWS}
        rows.append(item)
    return rows, errors


def encode_fitting(enc, rows, chosen, tokenizer_dir, tools_path, src, dest, report_path):
    fitting_ids = {r["transaction_id"] for r in rows
                   if r.get("raw_token_length") is not None and r["raw_token_length"] <= chosen}
    keep = [tx for tx in read_jsonl(src) if tx.get("transaction_id") in fitting_ids]
    dest = Path(dest)
    report_path = Path(report_path)
    if not keep:
        dest.write_text("", encoding="utf-8")
        report = {"encoding_version": enc.ENCODING_VERSION, "transactions": 0, "written": 0,
                  "skipped_oversize": [], "ok": True, "empty": True,
                  "reason": "no transactions fit chosen_max_seq_len=%d" % chosen}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    filtered = dest.with_name(dest.stem + ".fitting-src.jsonl")
    write_jsonl(filtered, keep)
    argv = ["encode", "--input", str(filtered), "--output", str(dest), "--report", str(report_path),
            "--tokenizer", str(tokenizer_dir), "--tools", str(tools_path),
            "--thinking", "on", "--thought-loss", "mask",
            "--max-seq-len", str(chosen), "--on-oversize", "skip"]
    code = enc.main(argv)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if code != 0:
        raise RuntimeError("encode failed for %s: %s" % (dest, report.get("error")))
    return report


def run_census(args):
    results = Path(args.results)
    if not results.exists():
        raise FileNotFoundError("results dir missing: %s (gitignored; use the main working tree copy)"
                                % results)
    tokenizer_dir = args.tokenizer or os.environ.get("P005_GEMMA_TOKENIZER_DIR")
    if not tokenizer_dir:
        raise FileNotFoundError("pass --tokenizer or set P005_GEMMA_TOKENIZER_DIR; no 31B download")
    tok_fp = verify_ct_v2_tokenizer(tokenizer_dir)
    out = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    tools_path = Path(args.tools)
    splits_path = Path(args.splits)
    heldout, panel = heldout_and_panel(splits_path)
    (out / "heldout-ids.txt").write_text("\n".join(heldout) + ("\n" if heldout else ""), encoding="utf-8")

    prepare_report = prepare(results, splits_path, args.eligibility, out, tools_path,
                             include_failures=True)
    (out / "prepare_manifest.json").write_text(
        json.dumps(prepare_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tx_path = out / "transactions.jsonl"
    registry = register(tx_path, max_per_issue=args.max_per_issue)
    assign(registry, args.select_fraction, whole_repo=True, seed=args.seed, heldout=set(heldout))
    (out / "splits.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    counts = materialize(tx_path, registry, out / "train.jsonl", out / "select.jsonl")
    train_issues = issues_from_registry(registry, "train")
    select_issues = issues_from_registry(registry, "select")

    enc = _load_encoding()
    tokenizer = enc.load_tokenizer(str(tokenizer_dir), allow_download=False)
    tools, _protocol = enc.normalize_tools(json.loads(tools_path.read_text(encoding="utf-8")))
    train_tx = read_jsonl(out / "train.jsonl")
    select_tx = read_jsonl(out / "select.jsonl")
    train_rows, train_err = measure_rows(train_tx, tokenizer, tools, enc)
    select_rows, select_err = measure_rows(select_tx, tokenizer, tools, enc)
    all_rows = train_rows + select_rows
    decision = decide_max_seq_len([r["raw_token_length"] for r in train_rows])
    chosen = decision["chosen_max_seq_len"]

    train_encode = encode_fitting(
        enc, train_rows, chosen, tokenizer_dir, tools_path,
        out / "train.jsonl", out / "encoded-train.jsonl", out / "encoded-train.jsonl.report.json")
    select_encode = encode_fitting(
        enc, select_rows, chosen, tokenizer_dir, tools_path,
        out / "select.jsonl", out / "encoded-select.jsonl", out / "encoded-select.jsonl.report.json")
    encoded_train_n = int(train_encode.get("written") or 0)
    encoded_select_n = int(select_encode.get("written") or 0)
    if encoded_train_n == 0:
        decision["trainable_ok"] = False
        if "no fully encoded train" not in decision["decision_reason"]:
            decision["decision_reason"] += "; no fully encoded train JSONL at chosen window"

    truncated_written = []
    for encoded_path in (out / "encoded-train.jsonl", out / "encoded-select.jsonl"):
        for rec in read_jsonl(encoded_path):
            meta = rec.get("meta") or {}
            if meta.get("truncated") or meta.get("sequence_length", 0) > chosen:
                truncated_written.append(meta.get("transaction_id"))
    if truncated_written:
        raise RuntimeError("encoder wrote truncated/oversize records: %s" % truncated_written)

    panel_in_train = [i for i in panel if i in set(train_issues)]
    panel_not_in_train = [i for i in panel if i not in set(train_issues)]
    train_issues_doc = {
        "generated_at": "2026-10-10",
        "protocol_id": PROTOCOL_ID,
        "include_failures": True,
        "resolved_only_default_unchanged": True,
        "max_per_issue": args.max_per_issue,
        "whole_repo": True,
        "select_fraction": args.select_fraction,
        "seed": args.seed,
        "heldout_excluded": True,
        "heldout_ids": heldout,
        "train_issues": train_issues,
        "select_issues": select_issues,
        "dev_panel_ids": panel,
        "dev_panel_in_train": panel_in_train,
        "dev_panel_not_in_train": panel_not_in_train,
        "pairing_note": "Issues in train_issues cannot be used as independent repair-gain evidence. "
                        "Task 5 pairing must use exposed panel issues disjoint from train_issues.",
    }
    (report_dir / "train_issues.json").write_text(
        json.dumps(train_issues_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def pack(rows):
        lengths = [r["raw_token_length"] for r in rows]
        return {
            "n": len(rows),
            "independent_issues": sorted({r["issue"] for r in rows}),
            "quantiles": length_quantiles(lengths),
            "histogram": length_histogram(lengths),
            "windows": {str(w): window_coverage(rows, w) for w in WINDOWS},
        }

    length_report = {
        "generated_at": "2026-10-10",
        "protocol_id": PROTOCOL_ID,
        "tools": str(tools_path),
        "tokenizer": tok_fp,
        "tokenizer_hash": enc.tokenizer_fingerprint(tokenizer),
        "encoding": "training/encoding/gemma_tool_chat.py",
        "on_oversize": "skip",
        "truncated": False,
        "json_clipped": False,
        "include_failures": True,
        "resolved_only_default_unchanged": True,
        "max_per_issue": args.max_per_issue,
        "whole_repo": True,
        "select_fraction": args.select_fraction,
        "seed": args.seed,
        "heldout_excluded": True,
        "prepare": {
            "traces": len(prepare_report.get("traces") or []),
            "rejected": len(prepare_report.get("rejected") or []),
            "transactions_before_cap": prepare_report.get("transactions"),
            "tasks": prepare_report.get("tasks"),
            "independent_task_count": prepare_report.get("independent_task_count"),
        },
        "split_materialize_counts": counts,
        "measure_errors": train_err + select_err,
        "all_capped": pack(all_rows),
        "train": pack(train_rows),
        "select": pack(select_rows),
        "transactions": all_rows,
        "chosen_max_seq_len": chosen,
        "trainable_ok": decision["trainable_ok"],
        "decision": decision,
        "encoded": {
            "train_written": encoded_train_n,
            "select_written": encoded_select_n,
            "train_report": str(out / "encoded-train.jsonl.report.json"),
            "select_report": str(out / "encoded-select.jsonl.report.json"),
            "train_path": str(out / "encoded-train.jsonl"),
            "select_path": str(out / "encoded-select.jsonl"),
        },
        "train_issues": train_issues,
        "select_issues": select_issues,
        "dev_panel_in_train": panel_in_train,
        "dev_panel_not_in_train": panel_not_in_train,
        "artifacts_dir": str(out),
        "did_not_overwrite": "training/artifacts/local-diagnostic",
    }
    (report_dir / "length_report.json").write_text(
        json.dumps(length_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "chosen_max_seq_len": chosen,
        "trainable_ok": decision["trainable_ok"],
        "train_issues": train_issues,
        "encoded_train": encoded_train_n,
        "encoded_select": encoded_select_n,
        "report": str(report_dir / "length_report.json"),
    }, ensure_ascii=False))
    return 0 if not (train_err or select_err) else 1


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--splits", default=str(ROOT / "research/task_splits.json"))
    p.add_argument("--eligibility", default=str(ROOT / "research/002/verification_eligibility.json"))
    p.add_argument("--tools", default=str(ROOT / "training/configs/harness_tools.json"))
    p.add_argument("--output-dir", default=str(ROOT / "training/artifacts/repair-sft-v1"))
    p.add_argument("--report-dir", default=str(ROOT / "research/005/length-census-20261010"))
    p.add_argument("--tokenizer", default=os.environ.get("P005_GEMMA_TOKENIZER_DIR"))
    p.add_argument("--max-per-issue", type=int, default=3)
    p.add_argument("--select-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20261010)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_census(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, RuntimeError, FileNotFoundError) as e:
        print(json.dumps({"ok": False, "error": str(e)}), file=sys.stderr)
        sys.exit(2)
