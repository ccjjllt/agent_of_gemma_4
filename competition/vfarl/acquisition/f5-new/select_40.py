"""Select 40 F5 timing candidates from locally cached Smith train rows."""
import json, re, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).resolve().parent
REV = "77cab9055d42ab4a5c25c89a8f937096db13558e"

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
]
F5_NEG_SSE = [r"ServerSentEvent", r"EventSource", r"text/event-stream", r"sse_", r"html\.escape"]
EXCLUDED = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}

man = json.loads((ROOT / "competition/vfarl/s600/course-manifest.json").read_text())
used = set(man["courses"]["common"]["instance_ids"]) | set(man["courses"]["specialist"]["instance_ids"])

rows = {}
def load_pages(pattern):
    for p in pathlib.Path(ROOT / "competition/vfarl/acquisition").glob(pattern):
        if "receipt" in str(p):
            continue
        try:
            d = json.loads(pathlib.Path(p).read_text())
        except Exception:
            continue
        for e in d.get("rows", []):
            r = e.get("row")
            if r and "instance_id" in r:
                rows[r["instance_id"]] = (e.get("row_idx"), r, str(pathlib.Path(p).relative_to(ROOT)))
for pat in ["f5-new/pages/train-*.json",
            "expand-resource/mido__mido-train-*.json",
            "expand-resource/pudo__dataset-train-*.json",
            "expand-resource/pexpect__ptyprocess.1067dbda-train.json",
            "expand-resource/aio-libs__async-timeout.d0baa9f1-train.json",
            "s600-specialist-a/artifacts/pages/click-*.json"]:
    load_pages(pat)
# add cached task.json rows
for f in pathlib.Path(ROOT / "competition/vfarl/acquisition").rglob("*.task.json"):
    try:
        d = json.loads(f.read_text())
        r = d.get("row")
        if r and "instance_id" in r and r["instance_id"] not in rows:
            rows[r["instance_id"]] = (d.get("row_idx", -1), r, str(f.relative_to(ROOT)))
    except Exception:
        pass

print(f"total_rows={len(rows)} novel={sum(1 for k in rows if k not in used)}")

cands = []
for iid, (idx, row, src) in rows.items():
    if iid in used:
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
    if any(re.search(p, patch) for p in F5_NEG_SSE):
        continue
    # canonical exclude
    try:
        suffix = row["repo"].split("/")[-1].split(".")[0].replace("__", "/")
        if suffix.lower() in EXCLUDED:
            continue
    except Exception:
        pass
    # only Smith-py train (repo startswith swesmith/) for main; Gym/V2 supplement allowed but prefer Smith
    if not row.get("repo", "").startswith("swesmith/"):
        continue
    hits = [p for p in F5_POS if re.search(p, patch)]
    if not hits:
        continue
    # prefer small single-file patches, synthetic mutations
    files = re.findall(r"diff --git a/(\S+)", patch)
    if len(files) != 1:
        continue
    if len(patch) > 6000:
        continue
    # reject combine with huge hunks? allow but penalize
    hunks = patch.count("@@")
    is_synth = ("func_basic" in iid or "func_pm" in iid or "lm_rewrite" in iid)
    score = len(hits) * 3 + (3 if is_synth else 0) - max(0, hunks - 2) - (len(patch) // 2000)
    cands.append({"instance_id": iid, "row_idx": idx, "repo": row["repo"],
                  "score": score, "hits": hits[:6], "files": files,
                  "patch_len": len(patch), "hunks": hunks,
                  "f2p": len(f2p), "p2p": len(p2p),
                  "is_synth": is_synth, "source": src,
                  "kind": iid.split(".")[-1].split("__")[0] if "." in iid else "?"})

cands.sort(key=lambda d: -d["score"])
# per-repo cap for diversity: max 10 tenacity, 8 click, 8 mido, rest fill
caps = {"jd__tenacity": 10, "pallets__click": 8, "mido__mido": 8,
        "pudo__dataset": 4, "pexpect__ptyprocess": 5, "aio-libs__async-timeout": 4}
counts = {}
selected = []
for c in cands:
    # repo key
    key = c["repo"].split("/")[-1].split(".")[0]
    # map to caps key
    cap_key = None
    for k in caps:
        if k in key:
            cap_key = k
            break
    if cap_key and counts.get(cap_key, 0) >= caps[cap_key]:
        continue
    selected.append(c)
    if cap_key:
        counts[cap_key] = counts.get(cap_key, 0) + 1
    if len(selected) >= 40:
        break

print(f"selected={len(selected)} counts={counts}")
for s in selected:
    print(s["score"], s["instance_id"], f'F2P={s["f2p"]} P2P={s["p2p"]} files={s["files"]} hunks={s["hunks"]} hits={s["hits"]}')

(HERE / "selected-40.json").write_text(json.dumps(
    {"date": "2026-10-25", "revision": REV, "split": "train",
     "policy": "Smith-py train main, exclude four competition repos, exclude S600 600, single-file small patches with timing signals, SSE rejected, single-point demotion left to manual review",
     "counts": counts, "candidates": selected}, ensure_ascii=False, indent=2) + "\n")
print("wrote selected-40.json")
