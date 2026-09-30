#!/usr/bin/env python3
"""P002 paired analysis (stage 0C capacity + stage-1 skill). Stdlib only.

Reads a frozen 002 run root produced by the joint carriers:
  run_manifest.json, full_schedule.json / selected_schedule.json,
  <variant>/.../attempts/<attempt_id>/{attempt_manifest.json, result.json,
  context_metrics.json, patches/*.patch, traces/trace_*.json}

Never guesses missing fields: absent trace/finish_reason/patch stays null and
is excluded from the corresponding denominator with explicit counts.

Usage:
  python3 research/002/analyze.py --run results/002/<RUN_ID> --baseline t0 --candidate c8192
  python3 research/002/analyze.py --run results/002/<RUN_ID> --baseline t0 --candidate t1

Writes <run>/analysis_002.json + stdout summary. Exit non-zero on pairing gaps.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def quantile(vals, p):
    vals = sorted(vals)
    if not vals:
        return None
    idx = (len(vals) - 1) * p
    lo, hi = math.floor(idx), math.ceil(idx)
    return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)


def numeric(vals):
    v = [x for x in vals if x is not None]
    return {"n": len(v), "missing": len(list(vals)) - len(v),
            "sum": sum(v) if v else None, "mean": statistics.mean(v) if v else None,
            "median": statistics.median(v) if v else None,
            "min": min(v) if v else None, "max": max(v) if v else None,
            "q25": quantile(v, 0.25), "q75": quantile(v, 0.75)}


def patch_production_changed(patch_text: str):
    """Parse unified diff paths; production = non-test, non-tmp source change."""
    if not patch_text or not patch_text.strip():
        return False, []
    files = []
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                b = parts[3]
                path = b[2:] if b.startswith("b/") else b
                files.append(path)
    prod = [f for f in files
            if not (f.startswith("tests/") or "/tests/" in f
                    or Path(f).name.startswith("test_") or f.endswith("_test.py"))
            and not f.split("/")[-1].startswith(("repro_", "scratch", "debug_"))
            and not f.endswith((".log", ".tmp", ".bak", ".orig", ".rej", ".pyc"))]
    return (len(prod) > 0), files


def scan_trace(trace_path: Path):
    """Best-effort trace scan; returns dict with explicit unknowns, never raises."""
    info = {"trace_present": trace_path.is_file(), "context_anomaly": None,
            "truncation_evidence": None, "successful_prod_edit_step": None,
            "tool_error_counts": {}, "steps": None}
    if not trace_path.is_file():
        return info
    try:
        raw = trace_path.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        info["bytes"] = len(raw)
        info["sha256"] = hashlib.sha256(raw).hexdigest()
        data = json.loads(text)
    except Exception as e:
        info["parse_error"] = f"{type(e).__name__}: {e}"
        return info
    # Generic string-level flags (multi-label, not causal)
    low = text.lower()
    if "contextwindowexceeded" in low or "context window" in low and "exceed" in low:
        info["context_anomaly"] = True
    else:
        info["context_anomaly"] = False
    if "finish_reason" in low and ("length" in low or "truncat" in low):
        info["truncation_evidence"] = True
    else:
        info["truncation_evidence"] = False  # absence of marker, not proof of no truncation
    # Steps: try common shapes
    steps = None
    if isinstance(data, dict):
        for key in ("steps", "tool_calls", "events", "trace"):
            if isinstance(data.get(key), list):
                steps = data[key]
                break
    if isinstance(data, list):
        steps = data
    if steps is not None:
        info["steps"] = len(steps)
        errs = Counter()
        for s in steps:
            try:
                blob = json.dumps(s)
            except Exception:
                continue
            if "FileReadError" in blob or "FileEditError" in blob:
                errs["read_edit_error_step"] += 1
        info["tool_error_counts"] = dict(errs)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run root dir")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    assert run.is_dir(), f"run root missing: {run}"
    manifest = load_json(run / "run_manifest.json")
    full = load_json(run / "full_schedule.json")
    # collect attempts
    rows = []
    for entry in full:
        if entry["variant"] not in (args.baseline, args.candidate):
            continue
        adir = Path(entry["results_dir"])
        # run root may be relocated; fall back to relative layout
        if not adir.is_dir():
            alt = run / entry["variant"] / "attempts" / entry["attempt_id"]
            alt2 = run / entry["variant"] / "r1" / "attempts" / entry["attempt_id"]
            alt3 = run / entry["variant"] / "r2" / "attempts" / entry["attempt_id"]
            for c in (alt, alt2, alt3):
                if c.is_dir():
                    adir = c
                    break
        rec = dict(entry)
        rj = adir / "result.json" if adir.is_dir() else None
        if rj is not None and rj.is_file():
            r = load_json(rj)
            rec.update({k: r.get(k) for k in
                        ("instance_id", "resolved", "test_exit_code", "duration_seconds",
                         "tool_calls", "total_llm_calls", "error")})
            rec["agent_patch_size"] = r.get("agent_patch_size")
        else:
            for k in ("instance_id", "resolved", "test_exit_code", "duration_seconds",
                      "tool_calls", "total_llm_calls", "error", "agent_patch_size"):
                rec.setdefault(k, None)
            rec["result_missing"] = True
        cm = adir / "context_metrics.json" if adir.is_dir() else None
        if cm is not None and cm.is_file():
            c = load_json(cm)
            rec["agent_elapsed_seconds"] = c.get("agent_elapsed_seconds")
            rec["patch_submitted"] = c.get("patch_submitted")
        else:
            rec["agent_elapsed_seconds"] = None
            rec["patch_submitted"] = None
        # patch production change from stored patch file
        safe = (rec.get("task_id") or "").replace("/", "__")
        pf = adir / "patches" / f"{safe}.patch" if adir.is_dir() else None
        if pf is not None and pf.is_file():
            changed, files = patch_production_changed(pf.read_text(encoding="utf-8", errors="replace"))
            rec["production_source_changed"] = changed
            rec["patch_files"] = files
        else:
            rec["production_source_changed"] = None
            rec["patch_files"] = None
        # trace scan (best effort)
        tp = adir / "traces" / f"trace_{safe}.json" if adir.is_dir() else None
        rec["trace_info"] = scan_trace(tp) if tp is not None else {"trace_present": False}
        # multi-label flags (descriptive, overlapping allowed)
        err = str(rec.get("error") or "")
        tinfo = rec["trace_info"]
        rec["flag_context_anomaly"] = bool(tinfo.get("context_anomaly") or "ContextWindowExceeded" in err)
        rec["flag_tool_error"] = bool("FileReadError" in err or "FileEditError" in err
                                      or (tinfo.get("tool_error_counts") or {}))
        rows.append(rec)
    # pairing key: (task_id, repeat?) — attempt_id embeds repeat for 0C; task_slot otherwise
    def pair_key(e):
        return (e.get("task_id"), e.get("task_slot"))
    base = {pair_key(e): e for e in rows if e["variant"] == args.baseline}
    cand = {pair_key(e): e for e in rows if e["variant"] == args.candidate}
    assert base and cand, "empty arm"
    assert set(base) == set(cand), f"incomplete pairing: base={len(base)} cand={len(cand)}"
    ids = sorted(base)
    wins = [k for k in ids if cand[k].get("resolved") and not base[k].get("resolved")]
    losses = [k for k in ids if base[k].get("resolved") and not cand[k].get("resolved")]
    n = len(wins) + len(losses)
    exact_p = (min(1, 2 * sum(math.comb(n, k) for k in range(min(len(wins), len(losses)) + 1)) / 2 ** n)
               if n else 1.0)
    metrics = {}
    for f in ("duration_seconds", "agent_elapsed_seconds", "tool_calls", "total_llm_calls", "agent_patch_size"):
        diffs = [cand[k].get(f) - base[k].get(f) for k in ids
                 if cand[k].get(f) is not None and base[k].get(f) is not None]
        metrics[f] = {"paired_diffs": numeric(diffs),
                      "eligible": len(diffs), "excluded_missing": len(ids) - len(diffs)}
    flags = {}
    for f in ("flag_context_anomaly", "flag_tool_error"):
        flags[f] = {"both": sum(bool(base[k].get(f)) and bool(cand[k].get(f)) for k in ids),
                    "baseline_only": sum(bool(base[k].get(f)) and not bool(cand[k].get(f)) for k in ids),
                    "candidate_only": sum(bool(cand[k].get(f)) and not bool(base[k].get(f)) for k in ids),
                    "neither": sum(not bool(base[k].get(f)) and not bool(cand[k].get(f)) for k in ids),
                    "missing_either": sum(base[k].get(f) is None or cand[k].get(f) is None for k in ids)}
    prod = {"both": sum(bool(base[k].get("production_source_changed")) and bool(cand[k].get("production_source_changed")) for k in ids),
            "baseline_only": sum(bool(base[k].get("production_source_changed")) and not bool(cand[k].get("production_source_changed")) for k in ids),
            "candidate_only": sum(bool(cand[k].get("production_source_changed")) and not bool(base[k].get("production_source_changed")) for k in ids)}
    out = {
        "schema_version": 1,
        "run": str(run), "baseline": args.baseline, "candidate": args.candidate,
        "pairs": len(ids),
        "baseline_resolved": sum(bool(base[k].get("resolved")) for k in ids),
        "candidate_resolved": sum(bool(cand[k].get("resolved")) for k in ids),
        "wins": [f"{k[0]}@slot{k[1]}" for k in wins], "losses": [f"{k[0]}@slot{k[1]}" for k in losses],
        "net": len(wins) - len(losses),
        "exact_two_sided_discordant_binomial_p": exact_p,
        "exact_p_caveat": "Descriptive only; no power/equivalence claim; task independence not guaranteed; small n.",
        "metrics": metrics, "flags": flags, "production_changed_pairs": prod,
        "method": "One attempt per (task, slot) per arm; missing stays null, never zero-filled. "
                  "Context/tool flags are overlapping descriptive labels, not causal classification. "
                  "Truncation requires finish_reason evidence; absence of marker is not proof.",
        "pair_rows": [{"task_id": k[0], "slot": k[1],
                       "baseline_resolved": base[k].get("resolved"),
                       "candidate_resolved": cand[k].get("resolved"),
                       "baseline_production": base[k].get("production_source_changed"),
                       "candidate_production": cand[k].get("production_source_changed"),
                       "baseline_error": base[k].get("error"),
                       "candidate_error": cand[k].get("error")} for k in ids],
    }
    dest = Path(args.out) if args.out else (run / "analysis_002.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(dest), "pairs": len(ids),
                      "baseline_resolved": out["baseline_resolved"],
                      "candidate_resolved": out["candidate_resolved"],
                      "wins": out["wins"], "losses": out["losses"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
