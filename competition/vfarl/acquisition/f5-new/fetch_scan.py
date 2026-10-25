"""F5-new batch: fetch pinned train pages and heuristic-scan for F5 timing candidates.

Reads-only HF datasets-server at pinned revision. Saves pages + receipts.
Scans patches for start/stop/stream/event-sequence signals.
Single-point values and SSE-string escapes are down-scored for manual demotion.
"""
import hashlib, json, re, sys, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PAGES = HERE / "pages"
REV = "77cab9055d42ab4a5c25c89a8f937096db13558e"
BASE = ("https://datasets-server.huggingface.co/rows?dataset=SWE-bench%2FSWE-smith-py"
        "&config=default&split=train&revision=" + REV)

TARGETS = {
    "jd__tenacity.0d40e76f": (1003, 1154),
    "bottlepy__bottle.a8dfef30": (4658, 5030),
    "python-hyper__h11.bed0dd4a": (48841, 49037),
    "PyCQA__flake8.cf1542ce": (5170, 5623),
    "facebookresearch__hydra.0f03eb60": (46816, 47243),
    "benoitc__gunicorn.bacbf8aa": (30111, 30483),
    "python-trio__trio.cfbbe2c1": (44716, 45493),
    "tornadoweb__tornado.d5ac65c1": (38136, 38447),
    "aio-libs__async-timeout.d0baa9f1": (16834, 16852),
    "pexpect__ptyprocess.1067dbda": (21096, 21127),
    "mido__mido.a0158ff9": (29508, 29868),
    "msiemens__tinydb.10644a0e": (17765, 17973),
    "prettytable__prettytable.ca90b055": (16558, 16833),
    "pallets__click.fde47b4b": (32065, 32709),
    "encode__starlette.db5063c2": (27459, 28179),
    "pudo__dataset.5c2dc8d3": (2745, 2851),
}

F5_POS = [
    r"attempt_number", r"wait_chain", r"wait_fixed", r"wait_exponential",
    r"stop_after", r"stop_before", r"stop_all", r"retry_state", r"iter_state",
    r"statistics\[", r"prepare_for_next_attempt", r"DoAttempt", r"DoSleep",
    r"add_hook", r"trigger_hook", r"__hook_reversed", r"_hooks\[",
    r"Bottle\.install", r"plugin", r"_Multipart", r"\.feed\(",
    r"header_to_body", r"chunk", r"rewind", r"route", r"dispatch",
    r"_notify", r"on_run_", r"on_job_", r"callbacks",
    r"_extract_next_receive_event", r"_extract_next_send_event",
    r"start_next_cycle", r"NEED_DATA", r"PAUSED", r"CLOSED", r"MUST_CLOSE",
    r"next_event", r"receive_data", r"send\(",
    r"visited_new_blank_line", r"next_logical_line", r"build_logical_line",
    r"logical_line", r"blank_line", r"_ logical", r"EOL", r"noqa",
    r"Arbiter", r"halt", r"reload", r"murder_workers", r"wakeup",
    r"Worker", r"boot\(", r"handle_exit",
    r"nursery", r"cancel_scope", r"checkpoint", r"UnboundedQueue",
    r"put_nowait", r"_get_batch", r"parking_lot", r"unpark",
    r"IOLoop", r"add_future", r"run_in_executor", r"PeriodicCallback",
    r"chain_future", r"_run_callback", r"add_done_callback",
    r"_do_exit", r"_on_timeout", r"_reschedule", r"_state\s*=\s*_State",
    r"deadline", r"shift\(", r"expired", r"reject\(",
    r"isalive", r"PtyProcess", r"\.wait\(", r"\.read\(", r"terminate", r"spawn",
    r"MidiFile", r"tick", r"merge_tracks", r"\.play\(", r"MidiTrack",
    r"upsert", r"truncate", r"drop_table", r"_read_table", r"flush",
    r"handle_starttag", r"handle_endtag", r"handle_data", r"paginate",
    r"__enter__", r"__exit__", r"close\(", r"lifespan", r"StreamingResponse",
    r"BackgroundTask", r"send_wrapper", r"receive", r"on_startup", r"on_shutdown",
]

F5_NEG_SSE = [
    r"ServerSentEvent", r"EventSource", r"text/event-stream",
    r"sse_", r"data:\s*\\n", r"html\.escape", r"quote\(",
]

F5_NEG_SINGLE = [
    r"domain\s*[:=]", r"config\.set", r"settings\[",
]


def fetch_page(offset, length=100):
    import time
    path = PAGES / f"train-{offset}.json"
    url = BASE + f"&offset={offset}&length={length}"
    if path.exists():
        return json.loads(path.read_text())
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                body = r.read()
            rev = r.headers.get("X-Revision", "")
            blob = json.loads(body)
            path.write_bytes(json.dumps(blob, ensure_ascii=False).encode())
            (PAGES / f"train-{offset}.receipt.json").write_text(json.dumps(
                {"url": url, "offset": offset, "length": length, "split": "train",
                 "revision": REV, "response_X_Revision": rev,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "rows": len(blob.get("rows", []))}, indent=2) + "\n")
            time.sleep(2)
            return blob
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"fetch offset={offset} attempt={attempt} error={e!r} sleep={wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"fetch failed offset={offset}")


def score(row):
    patch = row.get("patch", "")
    iid = row["instance_id"]
    f2p = row.get("FAIL_TO_PASS") or []
    p2p = row.get("PASS_TO_PASS") or []
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)
    if not f2p or not p2p or set(f2p) & set(p2p):
        return None
    if not (row.get("problem_statement") or "").strip():
        return None
    # SSE-string escape impersonating timing: hard reject flag
    sse_hits = [pat for pat in F5_NEG_SSE if re.search(pat, patch)]
    if sse_hits:
        return {"score": -100, "pos": 0, "neg_sse": sse_hits,
                "f2p": len(f2p), "p2p": len(p2p),
                "empty_stmt": False, "reject": "SSE_STRING_ESCAPE",
                "patch_head": patch[:600]}
    pos_hits = [pat for pat in F5_POS if re.search(pat, patch)]
    pos = len(pos_hits)
    if pos == 0:
        return {"score": -50, "pos": 0, "neg_sse": [],
                "f2p": len(f2p), "p2p": len(p2p),
                "empty_stmt": False, "reject": "NO_TIMING_SIGNAL_DEMOTE_SG",
                "patch_head": patch[:600]}
    single_hits = [pat for pat in F5_NEG_SINGLE if re.search(pat, patch)]
    # sequence syntax in patch? attempt loops, order lists
    seq_bonus = 2 if re.search(r"for .* in .*attempt|seen\s*==|values\s*==|order", patch) else 0
    return {"score": pos * 2 - len(single_hits) * 3 + seq_bonus,
            "pos": pos, "pos_hits": pos_hits[:8],
            "single_hits": single_hits, "neg_sse": [],
            "f2p": len(f2p), "p2p": len(p2p),
            "empty_stmt": False, "reject": None,
            "patch_head": patch[:600]}


def main():
    only = sys.argv[1:] or list(TARGETS)
    PAGES.mkdir(parents=True, exist_ok=True)
    index = {}
    for l in (ROOT / "competition/vfarl/acquisition/general/smith-task-index.jsonl").read_text().splitlines():
        r = json.loads(l)
        index[r["instance_id"]] = r
    import pickle, os
    used = set()
    # load S600 600 ids directly (no pickle dependency)
    man = json.loads((ROOT / "competition/vfarl/s600/course-manifest.json").read_text())
    used |= set(man["courses"]["common"]["instance_ids"])
    used |= set(man["courses"]["specialist"]["instance_ids"])
    # also exclude four competition repos (canonical lower)
    excluded = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}
    out = []
    for repo_key in only:
        if repo_key not in TARGETS:
            print(f"unknown repo {repo_key}", flush=True)
            continue
        start, end = TARGETS[repo_key]
        off = start - (start % 100)
        while off <= end:
            blob = fetch_page(off)
            for entry in blob.get("rows", []):
                if not (start <= entry["row_idx"] <= end):
                    continue
                row = entry["row"]
                if row["instance_id"] in used:
                    continue
                # exclude four competition repos by canonical guess
                # canonical: swesmith/org__repo.sha -> org/repo
                try:
                    suffix = row["repo"].split("/")[-1].split(".")[0].replace("__", "/")
                    if suffix.lower() in excluded:
                        continue
                except Exception:
                    pass
                s = score(row)
                if s is None:
                    continue
                out.append({"instance_id": row["instance_id"], "row_idx": entry["row_idx"],
                            "repo": row["repo"], **s})
            off += 100
    out.sort(key=lambda d: -d["score"])
    (HERE / "scan-hits.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"scanned_hits={len(out)}")
    pos = [d for d in out if (d.get("reject") is None)]
    rej = [d for d in out if (d.get("reject") is not None)]
    print(f"timing_signal={len(pos)} rejected_or_demote={len(rej)}")
    for d in out[:100]:
        print(d["score"], d["instance_id"], f'F2P={d["f2p"]} P2P={d["p2p"]} reject={d.get("reject")} pos={d.get("pos")}')
    # save used snapshot for audit
    (HERE / "scan-used.json").write_text(json.dumps({"used_600": len(used), "excluded": sorted(excluded)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
