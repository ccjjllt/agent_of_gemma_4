"""TPU full feasibility probe (Stage 2).
Fixed stack: tpu-inference @ merge SHA of upstream PR #3653 (1bd2d5a...).
Weight: google/gemma-4-31b-it-qat-w4a16-ct (mounted via model_sources).
Sequence: forensics -> pinned install -> serve 2048 -> minimal generate ->
tool-call multi-round -> 8192 -> 32768 (stop wherever time/budget ends).
Every stage is timestamped; partial evidence is ALWAYS written.
Global soft deadline: 80 min (kernel timeout 90 min leaves margin).
"""
import datetime
import glob
import json
import os
import subprocess
import sys
import time
import urllib.request

PIN_SHA = "1bd2d5a42af324706baff94f85cf6b8e8db8eea2"
PIN_URL = f"git+https://github.com/vllm-project/tpu-inference.git@{PIN_SHA}"
MODEL_ID = "google/gemma-4-31b-it-qat-w4a16-ct"
SOFT_DEADLINE_S = 80 * 60
PORT = 8000
# Upstream serving setup (PR #3653 body): vLLM 0.29.1rc1.dev468 + this SHA,
# --max-model-len 2048 --max-num-seqs 16, single v6e chip. Keep the same flags.
os.environ.setdefault("VLLM_XLA_CACHE_PATH", "/kaggle/working/xla_cache")

T0 = datetime.datetime.now(datetime.timezone.utc)
EV = {"probe": "tpu-full-20261001", "started_utc": T0.isoformat(),
      "pin_sha": PIN_SHA, "pin_url": PIN_URL, "model_id": MODEL_ID,
      "requested_machine_shape": "TpuV6E8", "stages": {}}


def log(msg):
    print(f"[full-probe] {msg}", flush=True)


def elapsed():
    return (datetime.datetime.now(datetime.timezone.utc) - T0).total_seconds()


def over_budget():
    return elapsed() > SOFT_DEADLINE_S


def sh(cmd, timeout=600):
    t = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"cmd": " ".join(cmd), "rc": p.returncode,
                "dt_s": round(time.time() - t, 1),
                "stdout_tail": p.stdout[-3000:], "stderr_tail": p.stderr[-2000:]}
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return {"cmd": " ".join(cmd), "rc": -99, "dt_s": round(time.time() - t, 1),
                "stdout_tail": str(out)[-3000:], "stderr_tail": f"TIMEOUT {timeout}s " + str(err)[-1500:]}


def http_post(path, payload, timeout=300):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://localhost:{PORT}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"rc": 200, "dt_s": round(time.time() - t, 1),
                    "body": r.read().decode()[:4000]}
    except Exception as e:
        return {"rc": -1, "dt_s": round(time.time() - t, 1), "body": f"ERROR {e!r}"[:2000]}


def http_get(path, timeout=30):
    t = time.time()
    try:
        with urllib.request.urlopen(f"http://localhost:{PORT}{path}", timeout=timeout) as r:
            return {"rc": r.status, "dt_s": round(time.time() - t, 1),
                    "body": r.read().decode()[:2000]}
    except Exception as e:
        return {"rc": -1, "dt_s": round(time.time() - t, 1), "body": f"ERROR {e!r}"[:1000]}


def write_evidence(tag):
    EV["now_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    EV["elapsed_s"] = round(elapsed(), 1)
    EV["stop_tag"] = tag
    with open("/kaggle/working/probe_evidence.json", "w") as f:
        json.dump(EV, f, indent=2, ensure_ascii=False)
    log(f"evidence written ({tag}) elapsed={elapsed():.0f}s")


def find_model_dir():
    cands = glob.glob("/kaggle/input/**/config.json", recursive=True)
    EV["config_candidates"] = cands[:10]
    log(f"config candidates: {cands[:10]}")
    for c in cands:
        if "gemma" in c.lower() or "31b" in c.lower() or "qat" in c.lower():
            return os.path.dirname(c)
    if cands:
        return os.path.dirname(cands[0])
    return None


def wait_server(deadline_s, label):
    t_start = time.time()
    last = None
    while time.time() - t_start < deadline_s:
        if over_budget():
            return {"ready": False, "dt_s": round(time.time() - t_start, 1),
                    "note": "SOFT_DEADLINE hit while waiting"}
        r = http_get("/v1/models", timeout=15)
        last = r
        if r["rc"] == 200:
            return {"ready": True, "dt_s": round(time.time() - t_start, 1),
                    "models_body": r["body"]}
        time.sleep(20)
    return {"ready": False, "dt_s": round(time.time() - t_start, 1),
            "note": f"not ready in {deadline_s}s", "last": last}


def main():
    log(f"start {EV['started_utc']} pin={PIN_SHA}")
    # A. device forensics (fast)
    EV["stages"]["forensics"] = {
        "jax_devices": sh([sys.executable, "-c",
            "import jax; print('backend=', jax.default_backend()); print('n=', jax.device_count()); [print(d) for d in jax.devices()]"]),
        "pip_before": sh(["sh", "-c",
            "pip list 2>/dev/null | grep -i -E '^(jax|jaxlib|libtpu|vllm|tpu|transformers|torch) ' || pip list 2>/dev/null | grep -i -E 'jax|tpu|vllm' || echo NOMATCH"]),
    }
    model_dir = find_model_dir()
    EV["model_dir"] = model_dir
    if not model_dir:
        EV["fatal"] = "model mount not found; cannot proceed to load"
        log("FATAL: model mount not found")
        write_evidence("no-model-mount")
        return

    # B. pinned install (needs internet)
    if over_budget():
        write_evidence("over-budget-before-install")
        return
    log(f"pip install {PIN_URL}")
    EV["stages"]["install"] = sh(
        [sys.executable, "-m", "pip", "install", "--quiet", PIN_URL], timeout=1500)
    r = EV["stages"]["install"]
    log(f"install rc={r['rc']} dt={r['dt_s']}s tail={r['stdout_tail'][-500:]} {r['stderr_tail'][-500:]}")
    EV["stages"]["fingerprints"] = {
        "pip_after": sh(["sh", "-c",
            "pip list 2>/dev/null | grep -i -E 'jax|tpu|vllm|transformers' || echo NOMATCH"]),
        "tpu_inf_version": sh([sys.executable, "-c",
            "import importlib.metadata as m; print('tpu_inference', m.version('tpu_inference'))"]),
        "vllm_version": sh([sys.executable, "-c",
            "import vllm, importlib.metadata as m; print('vllm', vllm.__version__); print('tpu_inf', m.version('tpu_inference'))"]),
        "wna16_present": sh([sys.executable, "-c",
            "import tpu_inference.layers.jax.quantization.wna16 as w; print('wna16 OK', w.__file__)"], timeout=120),
        "git_sha_check": sh(["sh", "-c",
            "pip show tpu_inference 2>/dev/null | head -n 12; python -c \"import tpu_inference, os; print(tpu_inference.__file__)\""]),
    }
    write_evidence("post-install")

    # C. serve tiers
    tiers = [2048, 8192, 32768]
    serve_deadlines = {2048: 2700, 8192: 1800, 32768: 1800}  # 45/30/30 min
    for tier in tiers:
        if over_budget():
            EV.setdefault("tiers", {})[str(tier)] = {"status": "SKIPPED_BUDGET"}
            log(f"tier {tier}: skipped, soft deadline reached")
            continue
        log(f"=== tier max_model_len={tier} ===")
        srv = subprocess.Popen(
            [sys.executable, "-m", "vllm", "serve", model_dir,
             "--tensor_parallel_size=1", f"--max-model-len={tier}",
             "--max-num-seqs=16",
             "--port", str(PORT), "--download_dir", "/tmp"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        wait = wait_server(serve_deadlines[tier], f"tier-{tier}")
        tier_ev = {"wait": wait}
        if not wait.get("ready"):
            # capture server log tail then kill
            try:
                srv.kill()
            except Exception:
                pass
            try:
                out, _ = srv.communicate(timeout=60)
                tier_ev["server_log_tail"] = out[-6000:]
            except Exception as e:
                tier_ev["server_log_tail"] = f"LOGCAP_ERR {e!r}"
            tier_ev["status"] = "SERVE_NOT_READY"
            EV.setdefault("tiers", {})[str(tier)] = tier_ev
            log(f"tier {tier}: SERVER NOT READY dt={wait['dt_s']}s")
            write_evidence(f"tier-{tier}-not-ready")
            continue
        log(f"tier {tier}: server ready dt={wait['dt_s']}s (load+cold-compile)")
        tier_ev["status"] = "READY"

        if tier == 2048:
            # C1. minimal generation
            g = http_post("/v1/completions", {
                "model": model_dir, "prompt": "Hello, my name is",
                "max_tokens": 32, "temperature": 0.0}, timeout=600)
            tier_ev["minimal_generate"] = g
            log(f"minimal_generate rc={g['rc']} dt={g['dt_s']}s body={g['body'][:800]}")
            # C2. tool-call multi-round (quotes/backslash/unicode)
            tool_arg_str = 'C:\\temp\\a"b \u6d4b\u8bd5 \u00e9'
            chat1 = http_post("/v1/chat/completions", {
                "model": model_dir,
                "messages": [{"role": "user",
                               "content": "Call the tool repo_navigation with query containing quotes."}],
                "tools": [{"type": "function", "function": {
                    "name": "repo_navigation",
                    "description": "Find symbols in the repo",
                    "parameters": {"type": "object",
                                   "properties": {"query": {"type": "string"}},
                                   "required": ["query"]}}}],
                "tool_choice": {"type": "function",
                                "function": {"name": "repo_navigation"}},
                "max_tokens": 128, "temperature": 0.0}, timeout=600)
            tier_ev["tool_round1"] = chat1
            log(f"tool_round1 rc={chat1['rc']} dt={chat1['dt_s']}s body={chat1['body'][:1200]}")
            # round 2: feed synthetic tool response back (only if round1 parsed)
            try:
                b = json.loads(chat1["body"]) if chat1["rc"] == 200 else None
                msg = (b.get("choices", [{}])[0].get("message", {})) if b else {}
                tcs = msg.get("tool_calls") or msg.get("toolCalls") or []
                if tcs:
                    tc = tcs[0]
                    tc_id = tc.get("id", "call_1")
                    tool_resp = json.dumps(
                        {"candidates": [{"path": "a.py", "kind": "def"}],
                         "echo": tool_arg_str}, ensure_ascii=False)
                    chat2 = http_post("/v1/chat/completions", {
                        "model": model_dir,
                        "messages": [
                            {"role": "user", "content": "Find symbol Foo."},
                            {"role": "assistant", "tool_calls": [tc]},
                            {"role": "tool", "tool_call_id": tc_id,
                             "content": tool_resp}],
                        "max_tokens": 128, "temperature": 0.0}, timeout=600)
                    tier_ev["tool_round2"] = chat2
                    log(f"tool_round2 rc={chat2['rc']} dt={chat2['dt_s']}s body={chat2['body'][:1200]}")
                else:
                    tier_ev["tool_round2"] = {"status": "NO_TOOL_CALL_IN_ROUND1"}
                    log("tool_round2 skipped: round1 had no tool_calls")
            except Exception as e:
                tier_ev["tool_round2"] = {"status": f"PARSE_ERR {e!r}"}
        else:
            # acceptance: long input within tier capacity, tiny output
            n_chars = {8192: 20000, 32768: 80000}[tier]
            big = "def foo():\n    pass\n" * (n_chars // 20)
            acc = http_post("/v1/completions", {
                "model": model_dir, "prompt": big[-n_chars:],
                "max_tokens": 16, "temperature": 0.0}, timeout=900)
            tier_ev["acceptance"] = {"input_chars": n_chars, "result": acc}
            log(f"tier {tier} acceptance rc={acc['rc']} dt={acc['dt_s']}s")

        # stop server before next tier
        try:
            srv.terminate()
            srv.wait(timeout=120)
        except Exception:
            try:
                srv.kill()
            except Exception:
                pass
        try:
            out = srv.stdout.read() if srv.stdout else ""
            tier_ev["server_log_tail"] = str(out)[-4000:]
        except Exception as e:
            tier_ev["server_log_tail"] = f"LOGCAP_ERR {e!r}"
        EV.setdefault("tiers", {})[str(tier)] = tier_ev
        write_evidence(f"tier-{tier}-done")

    write_evidence("finished-all-attempted")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        EV["top_error"] = repr(e)
        import traceback
        EV["traceback"] = traceback.format_exc()[-4000:]
        log(f"TOP ERROR {e!r}")
        write_evidence("top-error")
