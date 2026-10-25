"""F5-fresh: new sweep for small full-suite (F2P+P2P<50) new snapshots.

Sources: Smith-py cached pages/task.json + Gym/V2 local parquet.
Excludes: 75 banned snapshots + 5 f5-replace/small used snapshots (+3 f5-new already in banned).
Also excludes S600 600 ids, 4 competition repos, empty problem_statement, multi-file, hunks!=1.
F5 timing signals mirror f5-new select_40/fetch_scan.
Output: fresh-strict-scan.json sorted by total.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path("/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4")
HERE = ROOT / "competition/vfarl/acquisition/f5-new/f5-fresh"
REV_SMITHPY = "77cab9055d42ab4a5c25c89a8f937096db13558e"

F5_POS = [
    r"attempt_number", r"wait_chain", r"wait_fixed", r"wait_exponential",
    r"wait_random", r"wait_incrementing", r"stop_after", r"stop_before",
    r"stop_all", r"stop_any", r"retry_state", r"iter_state",
    r"statistics\[", r"prepare_for_next_attempt", r"DoAttempt", r"DoSleep",
    r"_run_stop", r"_begin_iter", r"add_hook", r"trigger_hook",
    r"__hook_reversed", r"_hooks\[", r"Bottle\.install", r"plugin",
    r"_Multipart", r"\.feed\(", r"header_to_body", r"chunk", r"rewind",
    r"_notify", r"on_run_", r"on_job_", r"callbacks",
    r"_extract_next_receive_event", r"_extract_next_send_event",
    r"start_next_cycle", r"NEED_DATA", r"PAUSED", r"CLOSED",
    r"next_event", r"receive_data",
    r"visited_new_blank_line", r"next_logical_line", r"build_logical_line",
    r"logical_line", r"blank_line", r"EOL", r"noqa",
    r"Arbiter", r"halt", r"reload", r"murder_workers", r"wakeup",
    r"Worker", r"boot\(", r"handle_exit",
    r"nursery", r"cancel_scope", r"checkpoint", r"UnboundedQueue",
    r"put_nowait", r"_get_batch", r"parking_lot", r"unpark",
    r"IOLoop", r"add_future", r"run_in_executor", r"PeriodicCallback",
    r"chain_future", r"_run_callback", r"add_done_callback",
    r"_do_exit", r"_on_timeout", r"_reschedule", r"_State\.",
    r"deadline", r"shift\(", r"expired", r"reject\(",
    r"isalive", r"PtyProcess", r"\.wait\(", r"\.read\(", r"terminate", r"spawn",
    r"MidiFile", r"\btick\b", r"merge_tracks", r"\.play\(", r"MidiTrack",
    r"upsert", r"truncate", r"drop_table", r"_read_table", r"flush",
    r"handle_starttag", r"handle_endtag", r"handle_data", r"paginate",
    r"__enter__", r"__exit__", r"call_on_close", r"enter_context", r"exit_stack",
    r"lifespan", r"StreamingResponse", r"BackgroundTask", r"send_wrapper",
    r"ResultIter", r"__iter__", r"commit", r"rollback",
    r"move_to", r"MockClock", r"autojump", r"get_nowait", r"getecho",
    r"waitnoecho", r"handle_reply", r"handle_exception", r"_clean_expired",
]
F5_NEG_SSE = [r"ServerSentEvent", r"EventSource", r"text/event-stream", r"sse_", r"html\.escape"]
EXCLUDED = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}

def load_banned():
    d = json.loads((ROOT / "competition/vfarl/acquisition/f5-new/f5-small/banned-snapshots.json").read_text())
    banned = set(d["banned_snapshot_group_ids"])
    # shorts for quick check
    shorts = set()
    for b in banned:
        canon, rev = b.split("::")
        shorts.add((canon.lower(), rev[:8].lower()))
    # 5 used fresh snapshots (full SHAs from f5-small README/banned note)
    used5 = {
        ("python-trio/trio", "cfbbe2c1"),
        ("spulec/freezegun", "5f171db0"),
        ("django/channels", "a144b4b8"),
        ("django/daphne", "32ac73e1"),
        ("tornadoweb/tornado", "d5ac65c1"),
    }
    return banned, shorts, used5

def load_s600_used():
    man = json.loads((ROOT / "competition/vfarl/s600/course-manifest.json").read_text())
    return set(man["courses"]["common"]["instance_ids"]) | set(man["courses"]["specialist"]["instance_ids"])

def load_used_instances():
    used = set()
    for p in [
        ROOT / "competition/vfarl/acquisition/f5-new/selected-40.json",
        ROOT / "competition/vfarl/acquisition/f5-new/f5-replace/selected-40.json",
        ROOT / "competition/vfarl/acquisition/f5-new/f5-small/selected-40.json",
    ]:
        if p.exists():
            d = json.loads(p.read_text())
            cands = d.get("candidates", [])
            for c in cands:
                used.add(c["instance_id"])
    # also f5-small extended + rejected
    return used

def canon_of_smith(row):
    repo = row["repo"]
    assert repo.startswith("swesmith/")
    suffix = repo.split("/")[-1].rsplit(".", 1)[0]
    assert "__" in suffix, suffix
    return suffix.replace("__", "/")

def short_of_smith(iid):
    return iid.split(".")[1]

def score_patch(patch):
    if any(re.search(p, patch) for p in F5_NEG_SSE):
        return None, "SSE"
    hits = [p for p in F5_POS if re.search(p, patch)]
    if not hits:
        return None, "NO_SIGNAL"
    return hits, None

def main():
    HERE.mkdir(parents=True, exist_ok=True)
    banned, banned_shorts, used5 = load_banned()
    s600_used = load_s600_used()
    used_inst = load_used_instances()
    print(f"banned={len(banned)} banned_shorts={len(banned_shorts)} s600={len(s600_used)} used_inst={len(used_inst)}")

    # ---- Smith-py cached rows ----
    rows = {}
    def load_pages(pattern):
        for p in (ROOT / "competition/vfarl/acquisition").glob(pattern):
            if "receipt" in str(p):
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            entries = d.get("rows", []) if isinstance(d, dict) else (d if isinstance(d, list) else [])
            for e in entries:
                if not isinstance(e, dict):
                    continue
                r = e.get("row", e if "instance_id" in e else None)
                if r and "instance_id" in r:
                    if r["instance_id"] not in rows:
                        rows[r["instance_id"]] = (e.get("row_idx", r.get("row_idx", -1)), r, str(p.relative_to(ROOT)))
    for pat in ["f5-new/pages/train-*.json",
                "f2-new/pages/train-*.json",
                "f1-new/pages/train-*.json",
                "expand-resource/*.json",
                "s600-specialist-a/artifacts/pages/click-*.json",
                "f5-new/f5-replace/*.task.json",
                "f5-new/f5-small/*.task.json",
                "f5-new/*.task.json",
                "f2-new/*.task.json",
                "f1-new/*.task.json"]:
        # rglob for task.json patterns
        if pat.endswith(".task.json"):
            for f in (ROOT / "competition/vfarl/acquisition").rglob(pat.split("/")[-1]):
                # restrict to f5/f2/f1 dirs to avoid huge walk? rglob is ok
                try:
                    d = json.loads(f.read_text())
                    r = d.get("row")
                    if r and "instance_id" in r and r["instance_id"] not in rows:
                        rows[r["instance_id"]] = (d.get("row_idx", -1), r, str(f.relative_to(ROOT)))
                except Exception:
                    pass
        else:
            load_pages(pat)
    print(f"smith cached rows={len(rows)}")

    smith_hits = []
    for iid, (idx, row, src) in rows.items():
        if iid in s600_used or iid in used_inst:
            continue
        if not row.get("repo", "").startswith("swesmith/"):
            continue
        try:
            canon = canon_of_smith(row)
        except Exception:
            continue
        if canon.lower() in EXCLUDED:
            continue
        short = short_of_smith(iid)
        if (canon.lower(), short.lower()) in banned_shorts or (canon.lower(), short.lower()) in {(c.lower(), s.lower()) for c, s in used5}:
            continue
        patch = row.get("patch", "")
        f2p = row.get("FAIL_TO_PASS") or []
        p2p = row.get("PASS_TO_PASS") or []
        if isinstance(f2p, str):
            f2p = json.loads(f2p)
        if isinstance(p2p, str):
            p2p = json.loads(p2p)
        if not f2p or not p2p or set(f2p) & set(p2p):
            continue
        if not (row.get("problem_statement") or "").strip():
            continue
        total = len(f2p) + len(p2p)
        if total >= 50:
            continue
        files = re.findall(r"diff --git a/(\S+)", patch)
        if len(files) != 1 or any(".." in f for f in files):
            continue
        hunks = len([l for l in patch.splitlines() if l.startswith("@@ ")])
        if hunks != 1:
            continue
        hits, rej = score_patch(patch)
        if hits is None:
            continue
        smith_hits.append({
            "pool": "smith-py", "instance_id": iid, "row_idx": idx, "repo": row["repo"],
            "canon": canon, "short": short, "hits": hits[:6], "file": files[0],
            "patch_len": len(patch), "hunks": hunks, "f2p": len(f2p), "p2p": len(p2p),
            "total": total, "F2P_sample": f2p[:2], "source_page": src,
        })
    smith_hits.sort(key=lambda d: (d["total"], d["f2p"]))
    print(f"smith small-full<50 single-hunk timing hits={len(smith_hits)}")
    for h in smith_hits[:40]:
        print(f'{h["total"]:3d} F2P={h["f2p"]} P2P={h["p2p"]} {h["instance_id"]} {h["canon"]}::{h["short"]} {h["file"]} {h["hits"][:3]}')

    # ---- Gym/V2 parquet F5 scan ----
    gym_hits, v2_hits = [], []
    try:
        import pyarrow.parquet as pq
        # Gym
        gym_path = Path("/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/gym-train.parquet")
        if gym_path.exists():
            f = pq.ParquetFile(str(gym_path))
            for rg in range(f.num_row_groups):
                t = f.read_row_group(rg, columns=["instance_id", "repo", "base_commit", "patch", "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS"])
                for r in t.to_pylist():
                    iid = r["instance_id"]
                    if iid in s600_used or iid in used_inst:
                        continue
                    repo = (r.get("repo") or "").strip()
                    if repo.lower() in EXCLUDED:
                        continue
                    base = r.get("base_commit") or ""
                    # snapshot check: canon::full and canon::short
                    full_id = f"{repo}::{base}"
                    short_id = f"{repo}::{base[:8]}"
                    if full_id in banned or short_id in banned:
                        continue
                    if (repo.lower(), base[:8].lower()) in {(c.lower(), s.lower()) for c, s in used5}:
                        continue
                    # also check banned_shorts
                    if (repo.lower(), base[:8].lower()) in banned_shorts:
                        continue
                    patch = r.get("patch") or ""
                    f2p = r.get("FAIL_TO_PASS") or []
                    p2p = r.get("PASS_TO_PASS") or []
                    if not f2p or set(f2p) & set(p2p):
                        continue
                    if not (r.get("problem_statement") or "").strip():
                        continue
                    total = len(f2p) + len(p2p)
                    if total >= 50 or total == 0:
                        continue
                    files = re.findall(r"diff --git a/(\S+)", patch)
                    if len(files) != 1 or any(".." in fl for fl in files):
                        continue
                    if len([l for l in patch.splitlines() if l.startswith("@@ ")]) != 1:
                        continue
                    hits, rej = score_patch(patch)
                    if hits is None:
                        continue
                    gym_hits.append({"pool": "gym", "instance_id": iid, "repo": repo, "base_commit": base,
                                     "hits": hits[:6], "files": files, "F2P": len(f2p), "P2P": len(p2p), "total": total,
                                     "F2P_list": f2p[:2], "P2P_sample": p2p[:2], "row_group": rg})
            gym_hits.sort(key=lambda d: (d["total"], d["F2P"]))
            print(f"gym F5 small-full hits={len(gym_hits)}")
            for h in gym_hits[:40]:
                print(f'{h["total"]:3d} {h["instance_id"]} {h["repo"]} {h["files"]} {h["hits"][:3]}')
        # V2
        v2_path = Path("/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet")
        if v2_path.exists():
            f = pq.ParquetFile(str(v2_path))
            for rg in range(f.num_row_groups):
                t = f.read_row_group(rg, columns=["instance_id", "language", "repo", "base_commit", "patch", "problem_statement", "FAIL_TO_PASS", "PASS_TO_PASS"])
                for r in t.to_pylist():
                    if (r.get("language") or "").lower() != "python":
                        continue
                    iid = r["instance_id"]
                    if iid in s600_used or iid in used_inst:
                        continue
                    repo = (r.get("repo") or "").strip()
                    if repo.lower() in EXCLUDED:
                        continue
                    base = r.get("base_commit") or ""
                    full_id = f"{repo}::{base}"
                    short_id = f"{repo}::{base[:8]}"
                    if full_id in banned or short_id in banned:
                        continue
                    if (repo.lower(), base[:8].lower()) in {(c.lower(), s.lower()) for c, s in used5}:
                        continue
                    if (repo.lower(), base[:8].lower()) in banned_shorts:
                        continue
                    patch = r.get("patch") or ""
                    f2p = r.get("FAIL_TO_PASS") or []
                    p2p = r.get("PASS_TO_PASS") or []
                    if not f2p or set(f2p) & set(p2p):
                        continue
                    if not (r.get("problem_statement") or "").strip():
                        continue
                    total = len(f2p) + len(p2p)
                    if total >= 50 or total == 0:
                        continue
                    files = re.findall(r"diff --git a/(\S+)", patch)
                    if len(files) != 1 or any(".." in fl for fl in files):
                        continue
                    if len([l for l in patch.splitlines() if l.startswith("@@ ")]) != 1:
                        continue
                    hits, rej = score_patch(patch)
                    if hits is None:
                        continue
                    v2_hits.append({"pool": "v2", "instance_id": iid, "repo": repo, "base_commit": base,
                                    "hits": hits[:6], "files": files, "F2P": len(f2p), "P2P": len(p2p), "total": total,
                                    "F2P_list": f2p[:2], "P2P_sample": p2p[:2], "row_group": rg})
            v2_hits.sort(key=lambda d: (d["total"], d["F2P"]))
            print(f"v2 F5 small-full hits={len(v2_hits)}")
            for h in v2_hits[:60]:
                print(f'{h["total"]:3d} {h["instance_id"]} {h["repo"]} {h["files"]} {h["hits"][:3]}')
    except Exception as e:
        print("parquet scan error", repr(e))
        import traceback; traceback.print_exc()

    out = {"date": "2026-10-25", "revision_smithpy": REV_SMITHPY,
           "policy": "F5-fresh small full-suite F2P+P2P<50, single-file single-hunk, nonempty whitelist prompt, exclude 75 banned +5 used snapshots +S600 +4 repos +used instances",
           "banned_count": len(banned), "s600_used": len(s600_used),
           "smith_hits": smith_hits, "gym_hits": gym_hits, "v2_hits": v2_hits,
           "counts": {"smith": len(smith_hits), "gym": len(gym_hits), "v2": len(v2_hits)}}
    (HERE / "fresh-strict-scan.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("wrote fresh-strict-scan.json", out["counts"])

if __name__ == "__main__":
    main()
