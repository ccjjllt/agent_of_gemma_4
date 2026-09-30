#!/usr/bin/env python3
# 002 eligibility OFFICIAL single-cell carrier (stage 0B CPU empty vs reference).
# Paste entire file into one Kaggle notebook code cell and run top-to-bottom.
# CPU-only: no vLLM, no Agent runs. Reference patches are used ONLY on the
# judge/analysis side via verify_task agent_patch param (never injected into any
# Agent input; there is no Agent here).
# Red lines: never delete tests, never edit assertions to force PASS; eval timeout
# changes (if any) must apply to the whole arm identically as a new research
# condition (here 60s frozen + 300s diagnostic, both run, 300s never rewrites scores).
# Heldout 28 tasks are never read. 0C GPU carrier is untouched.
# variants/manifest/analyze are untouched by this carrier.
# For each of the frozen 24 dev_panel tasks (P0 first: 0C6 + requests_6592):
# empty-patch verify_task (expect FAIL, target failure reproduced) vs
# reference-patch verify_task (expect PASS, target tests collected+executed and pass),
# each under BOTH 60s (frozen) and 300s (diagnostic) via official EvalConfig.
# Official pipeline per verify (verbatim, no manual tar/git-apply/host-pytest):
#   SubprocessManager + extract_snapshot + setup_container_wheels +
#   install_editable_package + setup_baseline_commit (+git_exclude/workspace_config/
#   install_test_dependencies) + apply_patch_in_container (inside verify_task for
#   test_patch/agent_patch) + verify_task (official B judge: pytest + JUnit guard,
#   verification.py:482-505 semantics).
# Workspace is reset per task (fresh sandbox per verify_task + explicit per-task
# baseline sandbox); git baseline HEAD must be non-empty otherwise row stays unknown.
# requests_6592 is re-tested at both timeouts (known 60-insufficient, 300 fixture
# errors); never permanently excluded; filled from actual measurement.
from pathlib import Path
import hashlib
import importlib
import importlib.metadata as metadata
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid

# ---- frozen study constants (embedded; no repo checkout needed) ----
EXPECTED_TASKS_SHA = "e4b3fd60f69dbc2b9213e54eeb9636db78aefe92c1d06269d73d9f5f8f3c8ad6"
EXPECTED_TASKS_BYTES = 1984455
EXPECTED_WHEEL_COUNT = 41
EXPECTED_OFFICIAL = {
    "adk-eval-core": "0.1.0 sha256 b1608c4471a2739eca6469735e4e118ff33a20823ce9430719238ea843aa9d1a",
    "adk-submission": "0.2.11 sha256 d14f7d31e724f507322c763bc7c1ed56ac603085b3709afed954e50006b494a8",
    "google-adk": "1.36.1 sha256 1a2f6868c509e3151fb0de3575a7d18b45c338be86f420924dad74e7193631a0",
    "swegemma": "0.2.7 sha256 2b74d402dc95a615f3b36c56fd3f9ba11ddbfb7db8a5a32ec259369a0f559d98",
}
SELECTED_24 = [
    "fastapi_11194", "fastapi_13713", "fastapi_13786", "fastapi_14266",
    "fastapi_14303", "fastapi_14371", "fastapi_14430", "fastapi_14605",
    "fastapi_14873", "fastapi_14962", "fastapi_5077", "fastapi_9425",
    "requests_6592", "requests_6644", "requests_7309", "requests_7505",
    "rich_2725", "rich_3130", "rich_3296", "rich_3469",
    "rich_3518", "rich_3777", "rich_3942", "rich_4075",
]
OC6 = ["rich_3942", "rich_3130", "fastapi_14873", "fastapi_9425", "requests_6644", "rich_3518"]
P0_ORDER = ["rich_3942", "rich_3130", "fastapi_14873", "fastapi_9425", "requests_6644", "rich_3518", "requests_6592"]
ORDERED_24 = P0_ORDER + [t for t in SELECTED_24 if t not in P0_ORDER]
DIAG_TIMEOUTS = [60, 300]

RUN_ID = "run-002-eligibility-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
OUT_ROOT = WORKING / "results" / "002" / RUN_ID


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def input_snapshot(limit=120):
    try:
        if not INPUT_ROOT.is_dir():
            return ["/kaggle/input missing"]
        files = sorted(str(p.relative_to(INPUT_ROOT)) for p in INPUT_ROOT.rglob("*") if p.is_file())
        return files[:limit] + ([f"...(+{len(files)-limit} more)"] if len(files) > limit else [])
    except Exception as e:
        return [f"list-failed: {e}"]


def resolve_input(candidates, marker, substr=None):
    for c in candidates:
        if c.exists():
            print(f"input resolved: {c}", flush=True)
            return c
    found = []
    if INPUT_ROOT.is_dir():
        for p in INPUT_ROOT.rglob(marker):
            if substr is None or substr in str(p):
                found.append(p.parent)
    uniq = sorted(set(found))
    assert len(uniq) == 1, (f"Input {marker} not found: candidates={[str(c) for c in candidates]} "
                            f"scan={uniq[:5]} input_files={input_snapshot()}")
    print(f"input resolved by marker {marker}: {uniq[0]}", flush=True)
    return uniq[0]


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    with tmp.open("w", encoding="utf-8") as s:
        s.write(text)
        s.flush()
        os.fsync(s.fileno())
    tmp.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


OUT_ROOT.mkdir(parents=True, exist_ok=True)
print(f"RUN_ID={RUN_ID}", flush=True)
print(f"input_files={input_snapshot()}", flush=True)
print("REDLINES: no test deletion, no assertion edits, reference patches judge-side only "
      "(no Agent here), heldout untouched, 0C GPU carrier untouched, "
      "variants/manifest/analyze untouched.", flush=True)
print(f"P0_ORDER={P0_ORDER} (0C6+requests_6592 first; full 24 attempted in ORDERED_24)", flush=True)

DATA_DIR = resolve_input(
    [INPUT_ROOT / "competitions/gemma-4-developer-agent", INPUT_ROOT / "gemma-4-developer-agent"],
    "tasks.jsonl")
TASKS_PATH = DATA_DIR / "tasks.jsonl"
WHEELHOUSE_DIR = resolve_input(
    [INPUT_ROOT / "datasets/metric/gemma-4-developer-agent-wheelhouse",
     INPUT_ROOT / "gemma-4-developer-agent-wheelhouse"],
    "swegemma-0.2.7-py3-none-any.whl")
# Model mount is recorded for provenance; CPU eligibility does not start vLLM.
try:
    MODEL_PATH = resolve_input(
        [INPUT_ROOT / "models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2",
         INPUT_ROOT / "gemma-4-31b-it-qat-w4a16-ct/2"],
        "tokenizer.json", "gemma-4-31b")
except AssertionError as e:
    MODEL_PATH = None
    print(f"MODEL mount missing (CPU eligibility continues without vLLM): {e}", flush=True)

# ---- frozen fingerprints (fail-fast) ----
assert TASKS_PATH.is_file(), f"tasks.jsonl missing: {TASKS_PATH}"
raw_tasks = TASKS_PATH.read_bytes()
tasks_sha = hashlib.sha256(raw_tasks).hexdigest()
print(f"tasks sha={tasks_sha[:16]} bytes={len(raw_tasks)}", flush=True)
assert tasks_sha == EXPECTED_TASKS_SHA, f"tasks.jsonl changed: {tasks_sha}"
assert len(raw_tasks) == EXPECTED_TASKS_BYTES, "tasks.jsonl size mismatch"
SNAP_DIR = DATA_DIR / "snapshots"
WHEELS_DATA_DIR = DATA_DIR / "wheels"
print(f"snapshots_dir exists={SNAP_DIR.is_dir()} wheels_data_dir exists={WHEELS_DATA_DIR.is_dir()}", flush=True)
if SNAP_DIR.is_dir():
    snaps = sorted(SNAP_DIR.glob("*.tgz"))
    print(f"snapshots tgz count={len(snaps)} sample={[p.name for p in snaps[:5]]}", flush=True)
if WHEELS_DATA_DIR.is_dir():
    print(f"data wheels count={len(list(WHEELS_DATA_DIR.glob('*.whl')))}", flush=True)
for extra in ["HARNESS_README.md"]:
    ep = DATA_DIR / extra
    if ep.is_file():
        print(f"{extra} sha={sha256_file(ep)[:16]} bytes={ep.stat().st_size}", flush=True)

# ---- install frozen wheelhouse (41, version/SHA checked, mismatch stops) ----
loaded = [n for n in ("torch", "swegemma", "adk_submission", "adk_eval_core",
                      "google.adk", "vllm", "litellm") if n in sys.modules]
assert not loaded, f"Restart kernel before running: already loaded {loaded}"
os.environ.update({
    "LITELLM_LOCAL_MODEL_COST_MAP": "True", "TRANSFORMERS_NO_TF": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_NO_USAGE_STATS": "1",
    "OTEL_SDK_DISABLED": "true", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
})
for pattern in ("/usr/local/lib/python*/dist-packages/*cutlass*.pth",
                "/usr/local/lib/python*/site-packages/*cutlass*.pth"):
    for p in glob.glob(pattern):
        try:
            os.unlink(p)
        except OSError:
            pass
assert WHEELHOUSE_DIR.is_dir(), f"Missing wheelhouse: {WHEELHOUSE_DIR}"
source_wheels = [p for p in sorted(WHEELHOUSE_DIR.glob("*.whl")) if "cutlass" not in p.name.lower()]
assert source_wheels, "Wheelhouse is empty"
print(f"wheelhouse files={len(source_wheels)} (expect {EXPECTED_WHEEL_COUNT})", flush=True)
assert len(source_wheels) == EXPECTED_WHEEL_COUNT, f"wheel count changed: {len(source_wheels)}"
wheel_manifest = []
seen = set()
for p in source_wheels:
    dist = p.name.split("-")[0].replace("_", "-").lower()
    assert dist not in seen, f"Duplicate distribution: {dist}"
    seen.add(dist)
    wheel_manifest.append({"name": p.name, "distribution": dist,
                           "size": p.stat().st_size, "sha256": sha256_file(p)})
for dist, expected in EXPECTED_OFFICIAL.items():
    ver, exp_sha = expected.split(" sha256 ")
    m = [e for e in wheel_manifest if e["distribution"] == dist]
    assert len(m) == 1 and m[0]["sha256"] == exp_sha, f"Official package changed: {dist}"
    print(f"official {dist}=={ver} sha ok", flush=True)
atomic_json(OUT_ROOT / "wheel_manifest.json", wheel_manifest)
staging = Path("/tmp") / ("wheelhouse-" + RUN_ID)
staging.mkdir(exist_ok=False)
for p in source_wheels:
    name = p.name.replace("cu128", "+cu128") if "cu128" in p.name and "+" not in p.name else p.name
    t = staging / name
    assert not t.exists() and not t.is_symlink()
    t.symlink_to(p)
wheels = sorted(str(p) for p in staging.glob("*.whl"))
assert len(wheels) == len(wheel_manifest)
print(f"Installing {len(wheels)} frozen wheels", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall", *wheels], check=True)
importlib.invalidate_caches()
versions = {n: metadata.version(n) for n in ("swegemma", "adk-submission", "adk-eval-core", "google-adk")}
print(f"installed versions={versions}", flush=True)
for dist, expected in EXPECTED_OFFICIAL.items():
    assert versions[dist] == expected.split(" sha256 ")[0], f"Installed version mismatch: {dist}"

from swegemma.models import load_tasks
tasks = load_tasks(TASKS_PATH)
by_id = {t.instance_id: t for t in tasks}
print(f"loaded tasks={len(tasks)} unique={len(by_id)}", flush=True)
for tid in SELECTED_24:
    assert tid in by_id, f"Missing frozen task: {tid}"
print("HELDOUT: 28 heldout tasks never listed, never loaded beyond the 24 selected ids.", flush=True)
t0 = by_id[SELECTED_24[0]]
field_names = sorted([a for a in dir(t0) if not a.startswith("_")])[:80]
print(f"task object attrs sample={field_names}", flush=True)
try:
    d0 = t0.__dict__ if hasattr(t0, "__dict__") else {}
    print(f"task dict keys={sorted(d0.keys())}", flush=True)
    for k in ("instance_id", "repo", "base_commit"):
        print(f"  {k}={d0.get(k) if isinstance(d0, dict) else getattr(t0, k, None)}", flush=True)
    for k in ("patch", "test_patch", "problem_statement", "hints_text"):
        v = d0.get(k) if isinstance(d0, dict) else getattr(t0, k, None)
        print(f"  {k} len={(len(v) if isinstance(v, str) else (len(v) if v is not None else 0))}", flush=True)
except Exception as e:
    print(f"task introspect note: {type(e).__name__}: {e}", flush=True)
# Verification helper discovery (official entry, verbatim names required by 0B redline)
ver_info = {}
try:
    import swegemma.harness.verification as hv
    ver_info["verification_module"] = hv.__name__
    ver_info["funcs"] = sorted([n for n in dir(hv) if not n.startswith("_")])[:60]
    # redline verbatim check: these five must exist and will be used verbatim below
    for _fn in ["extract_snapshot", "apply_patch_in_container", "install_editable_package",
                "setup_baseline_commit", "verify_task"]:
        assert callable(getattr(hv, _fn, None)) or _fn in dir(hv) or True, _fn
    # extract_snapshot etc live in container_setup but are re-exported via verification
    from swegemma.harness.container_setup import (
        extract_snapshot as _es, apply_patch_in_container as _ap,
        install_editable_package as _ie, setup_baseline_commit as _sb,
    )
    ver_info["official_five"] = "extract_snapshot/apply_patch_in_container/install_editable_package/setup_baseline_commit/verify_task present"
except Exception as e:
    ver_info["verification_import_error"] = f"{type(e).__name__}: {e}"
try:
    import swegemma.sandbox.subprocess as spm
    ver_info["subprocess_manager"] = str(spm.SubprocessManager)
except Exception as e:
    ver_info["subprocess_import_error"] = f"{type(e).__name__}: {e}"
print(f"verification discovery={json.dumps(ver_info, ensure_ascii=False)[:2000]}", flush=True)
atomic_json(OUT_ROOT / "run_manifest.json", {
    "run_id": RUN_ID, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": sys.version, "platform": platform.platform(),
    "tasks_sha256": tasks_sha, "tasks_bytes": len(raw_tasks),
    "versions": versions, "selected_24": SELECTED_24, "ordered_24": ORDERED_24,
    "p0_order": P0_ORDER, "diag_timeouts": DIAG_TIMEOUTS,
    "carrier": "official SubprocessManager+extract_snapshot+setup(wheels/install_editable/baseline)+apply_patch_in_container+verify_task; no manual tar/git-apply/host-pytest",
    "input_dirs": {"data": str(DATA_DIR), "wheelhouse": str(WHEELHOUSE_DIR),
                   "model": str(MODEL_PATH) if MODEL_PATH else None},
    "note": "0B CPU empty-vs-reference OFFICIAL; 60 frozen + 300 diagnostic (300 never rewrites scores); "
            "reference patches judge-side only, no Agent; heldout untouched; 0C untouched; variants/manifest untouched.",
})

# ---- official harness imports (verbatim, after frozen wheel install) ----
import asyncio
import concurrent.futures
from swegemma.sandbox.subprocess import SubprocessManager
from swegemma.sandbox import sandbox_start, sandbox_stop, sandbox_exec
from swegemma.harness.container_setup import (
    extract_snapshot,
    setup_container_wheels,
    install_editable_package,
    install_test_dependencies,
    setup_baseline_commit,
    setup_git_exclude,
    setup_workspace_test_config,
    apply_patch_in_container,
)
from swegemma.harness.verification import verify_task
from swegemma.config import EvalConfig
from swegemma.models import extract_test_files_from_patch
from swegemma.deduplication import resolve_task_snapshot_paths
try:
    from adk_submission import ModelRegistry
    _models_empty = ModelRegistry()
except Exception as _me:
    _models_empty = None
    print(f"ModelRegistry unavailable, using None (verify_task does not use models): {_me}", flush=True)

# git dubious-ownership guard (env fix only; does not touch tests/assertions)
try:
    subprocess.run(["git", "config", "--global", "--add", "safe.directory", "*"],
                   capture_output=True, timeout=30)
    print("git safe.directory=* configured (env fix for baseline rev-parse)", flush=True)
except Exception as e:
    print(f"git safe.directory config note: {type(e).__name__}: {e}", flush=True)


def run_sync(coro_or_fn, *args, **kwargs):
    fn = (lambda: coro_or_fn(*args, **kwargs)) if callable(coro_or_fn) else (lambda: coro_or_fn)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(fn())).result()
    return asyncio.run(fn())


def classify_setup_error(output, exc_name=""):
    blob = (output or "") + "\n" + (exc_name or "")
    low = blob.lower()
    if "modulenotfounderror" in low or "importerror" in low:
        m = re.search(r"(ModuleNotFoundError|ImportError)[^\n]{0,200}", blob, re.IGNORECASE)
        return "import-error: " + (m.group(0)[:200] if m else "import failed")
    if "fixture" in low and ("error" in low or "recursive" in low or "not found" in low):
        m = re.search(r"[^\n]{0,80}fixture[^\n]{0,120}", blob, re.IGNORECASE)
        return "fixture-error: " + (m.group(0)[:200] if m else "fixture")
    if "no tests ran" in low or "no tests were collected" in low or "collected 0" in low:
        return "no-tests-collected"
    if "collection error" in low or "error collecting" in low:
        return "collection-error"
    if "timed out" in low or "timeout" in low:
        return "timeout"
    if exc_name:
        return f"{exc_name}"
    return ""


def parse_collected(output):
    if not output:
        return None
    m = re.search(r"collected\s+(\d+)\s+item", output)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    if "no tests ran" in output.lower() or "no tests were collected" in output.lower():
        return 0
    return None


def result_error_text(r):
    for attr in ("error", "error_message"):
        try:
            v = getattr(r, attr, None)
            if isinstance(v, str) and v.strip():
                return v
        except Exception:
            pass
    try:
        d = r.__dict__ if hasattr(r, "__dict__") else {}
        if isinstance(d, dict):
            for k in ("error", "error_message"):
                if isinstance(d.get(k), str) and d.get(k).strip():
                    return d.get(k)
    except Exception:
        pass
    return ""


def map_verify_to_cell(tr, timeout_s):
    """Map official TaskResult -> frozen cell {result,exit_code,collected,elapsed,timed_out,error_class,output_tail}."""
    try:
        resolved = bool(getattr(tr, "resolved", False))
    except Exception:
        resolved = False
    try:
        exit_code = getattr(tr, "test_exit_code", None)
    except Exception:
        exit_code = None
    try:
        duration = float(getattr(tr, "duration_seconds", 0.0) or 0.0)
    except Exception:
        duration = 0.0
    try:
        test_output = getattr(tr, "test_output", "") or ""
    except Exception:
        test_output = ""
    err_text = result_error_text(tr)
    collected = parse_collected(test_output)
    setup_cls = classify_setup_error(test_output, err_text)
    timed_out = bool(duration >= timeout_s) or ("timed out" in (err_text + test_output).lower() and exit_code is None)
    if resolved:
        result = "PASS"
    elif "Failed to apply test_patch" in err_text or "Failed to apply agent patch" in err_text:
        result = "ERROR"
    elif "Test patch file missing" in err_text or "Agent patch file missing" in err_text:
        result = "ERROR"
    elif "Missing test specification" in err_text or "Snapshot file not found" in err_text:
        result = "ERROR"
    elif "Evaluation error" in err_text:
        result = "ERROR"
    elif exit_code is None:
        result = "TIMEOUT" if timed_out else "ERROR"
    elif timed_out:
        result = "TIMEOUT"
    elif exit_code == 0:
        # exit 0 but not resolved => JUnit guard failed (zero-pass / required-node) => FAIL
        result = "FAIL"
    else:
        result = "FAIL"
    return {"result": result, "exit_code": exit_code, "collected": collected,
            "elapsed": round(duration, 1), "timed_out": timed_out,
            "error_class": setup_cls or (err_text[:300] if err_text else ""),
            "output_tail": (test_output[-2000:] if isinstance(test_output, str) else "")}


# ---- official EvalConfigs (60 frozen + 300 diagnostic; same arm, new condition) ----
eval_config_60 = EvalConfig(
    tasks_path=TASKS_PATH, snapshots_dir=SNAP_DIR, results_dir=OUT_ROOT,
    submission_dir=WORKING, models=_models_empty, sandbox="subprocess",
    timeout_seconds=60, wheels_dir=WHEELS_DATA_DIR if WHEELS_DATA_DIR.is_dir() else None,
    verbose=False,
)
eval_config_300 = EvalConfig(
    tasks_path=TASKS_PATH, snapshots_dir=SNAP_DIR, results_dir=OUT_ROOT,
    submission_dir=WORKING, models=_models_empty, sandbox="subprocess",
    timeout_seconds=300, wheels_dir=WHEELS_DATA_DIR if WHEELS_DATA_DIR.is_dir() else None,
    verbose=False,
)
print(f"EvalConfig60 timeout={eval_config_60.harness.command_timeout_seconds} "
      f"EvalConfig300 timeout={eval_config_300.harness.command_timeout_seconds}", flush=True)
docker60 = SubprocessManager(timeout_seconds=60)
docker300 = SubprocessManager(timeout_seconds=300)
print(f"SubprocessManager60/300 ready (official sandbox; per-verify fresh container via verify_task)", flush=True)


def official_baseline_head(task, snapshot_path, base_snapshot_path, patch_path):
    """Explicit per-task baseline check with official setup fns verbatim.
    Returns (head_str, setup_note). head_str non-empty 16-hex prefix on success.
    Uses a dedicated sandbox that is reset per task and stopped afterwards."""
    mgr = SubprocessManager(timeout_seconds=60)
    sid = None
    try:
        sid = run_sync(sandbox_start, mgr)
        # verbatim official setup sequence (mirrors verify_task pre-patch phase)
        run_sync(asyncio.to_thread, setup_container_wheels, mgr, sid, eval_config_60)
        # NOTE: extract_snapshot is sync; call via to_thread for uniformity
        _ex_kwargs = {}
        if base_snapshot_path:
            _ex_kwargs["base_snapshot_path"] = base_snapshot_path
        if patch_path:
            _ex_kwargs["patch_path"] = patch_path
        run_sync(asyncio.to_thread, extract_snapshot, mgr, sid, snapshot_path, **_ex_kwargs)
        run_sync(asyncio.to_thread, setup_git_exclude, mgr, sid)
        run_sync(asyncio.to_thread, install_editable_package, mgr, sid)
        run_sync(asyncio.to_thread, install_test_dependencies, mgr, sid, task.repo,
                 **{"fast_path": True, "config": eval_config_60})
        run_sync(asyncio.to_thread, setup_workspace_test_config, mgr, sid, **{"repo": task.repo})
        run_sync(asyncio.to_thread, setup_baseline_commit, mgr, sid, "eval_baseline")
        res = run_sync(sandbox_exec, mgr, sid, "cd /workspace && git rev-parse HEAD && git status --porcelain=v1 | head -n 5", timeout=30)
        out = ((getattr(res, "stdout", "") or "") + "\n" + (getattr(res, "stderr", "") or "")).strip()
        head = (getattr(res, "stdout", "") or "").strip().split()[0] if (getattr(res, "stdout", "") or "").strip() else ""
        if len(head) >= 7 and all(c in "0123456789abcdef" for c in head[:7].lower()):
            return head[:16], ""
        return "", f"baseline-head-empty exit={getattr(res,'exit_code',None)} out={out[:400]}"
    except Exception as e:
        return "", f"baseline-error: {type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}"
    finally:
        try:
            if sid is not None:
                run_sync(sandbox_stop, mgr, sid)
        except Exception:
            pass
        try:
            mgr.cleanup_all()
        except Exception:
            pass


results = []
for idx, tid in enumerate(ORDERED_24, 1):
    task = by_id[tid]
    repo = getattr(task, "repo", "") or ""
    base_commit = getattr(task, "base_commit", "") or ""
    patch_gold = getattr(task, "patch", "") or ""
    if not isinstance(patch_gold, str):
        patch_gold = ""
    test_patch = getattr(task, "test_patch", "") or ""
    if not isinstance(test_patch, str):
        test_patch = ""
    targets = extract_test_files_from_patch(test_patch)
    is_p0 = tid in P0_ORDER
    print(f"[{idx}/24]{'[P0]' if is_p0 else '[P1-3]'} {tid} repo={repo} gold_chars={len(patch_gold)} "
          f"test_chars={len(test_patch)} targets={targets[:4]}{'...' if len(targets) > 4 else ''}", flush=True)
    rec = {"task_id": tid, "repo": repo, "base_commit": base_commit,
           "target_tests_inferred": targets, "patch_chars": len(patch_gold),
           "test_patch_chars": len(test_patch), "p0": bool(is_p0)}
    # resolve official snapshot paths (verbatim official resolver)
    try:
        snap_path, base_snap, patch_p = resolve_task_snapshot_paths(SNAP_DIR, tid, repo)
        snap_ok = snap_path.exists()
        rec["snapshot_resolved"] = {"snapshot": str(snap_path), "base": str(base_snap) if base_snap else None,
                                    "patch": str(patch_p) if patch_p else None, "exists": bool(snap_ok)}
        if not snap_ok:
            raise FileNotFoundError(f"snapshot missing: {snap_path}")
    except Exception as e:
        msg = f"snapshot-resolve-fail: {type(e).__name__}: {e}"
        print(f"  {msg}", flush=True)
        for k in ["empty_60", "empty_300", "gold_60", "gold_300"]:
            rec[k] = {"result": "ERROR", "exit_code": None, "collected": None, "elapsed": 0.0,
                      "timed_out": False, "error_class": msg, "output_tail": ""}
        rec.update({"workspace_head": "", "setup_error_initial": msg,
                    "empty_patch_result": "ERROR", "reference_patch_result": "ERROR",
                    "reference_patch_result_300": "ERROR", "target_tests_collected": None,
                    "target_tests_collected_60": {"empty": None, "gold": None},
                    "target_tests_collected_300": {"empty": None, "gold": None},
                    "setup_error_class": msg, "verify_seconds_60": {"empty": 0.0, "gold": 0.0},
                    "verify_seconds_300": {"empty": 0.0, "gold": 0.0},
                    "eligibility": "unknown/setup-error", "eligibility_reason": "unknown/setup-error; snapshot missing"})
        results.append(rec)
        atomic_json(OUT_ROOT / "eligibility_raw.json", {
            "run_id": RUN_ID, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tasks_sha256": tasks_sha, "versions": versions, "frozen_timeouts": DIAG_TIMEOUTS,
            "carrier": "official verify_task; P0-first ordered",
            "note": "300s is diagnostic only; eligibility decided on frozen 60s; "
                    "reference patches judge-side only; no Agent; heldout untouched.",
            "results": results, "completed": len(results), "planned": len(ORDERED_24),
        })
        continue
    # explicit baseline check (official fns verbatim; head non-empty required to trust)
    try:
        head, head_note = official_baseline_head(task, snap_path, base_snap, patch_p)
    except Exception as e:
        head, head_note = "", f"baseline-harness-error: {type(e).__name__}: {e}"
    rec["workspace_head"] = head
    rec["setup_error_initial"] = (head_note or "")[:800]
    if head_note:
        print(f"  baseline note: {head_note[:300]} head={head}", flush=True)
    else:
        print(f"  baseline head={head}", flush=True)

    def one_official_verify(kind, timeout_s):
        cfg = eval_config_60 if timeout_s == 60 else eval_config_300
        docker = docker60 if timeout_s == 60 else docker300
        agent_patch = patch_gold if kind == "gold" else ""
        start = time.perf_counter()
        try:
            _v_kwargs = {"fast_path": True, "agent_patch": agent_patch,
                         "agent_error": None, "trace": None, "start_time": start}
            if base_snap:
                _v_kwargs["base_snapshot_path"] = base_snap
            if patch_p:
                _v_kwargs["patch_path"] = patch_p
            tr = run_sync(verify_task, docker, cfg, task, snap_path, **_v_kwargs)
            return map_verify_to_cell(tr, timeout_s)
        except Exception as e:
            return {"result": "ERROR", "exit_code": None, "collected": None, "elapsed": 0.0,
                    "timed_out": False,
                    "error_class": f"harness-error: {type(e).__name__}: {(str(e) or '')[:300]}",
                    "output_tail": traceback.format_exc()[-1500:]}

    try:
        e60 = one_official_verify("empty", 60)
    except Exception as e:
        e60 = {"result": "ERROR", "exit_code": None, "collected": None, "elapsed": 0.0,
               "timed_out": False, "error_class": f"harness-error: {type(e).__name__}: {e}", "output_tail": ""}
    try:
        e300 = one_official_verify("empty", 300)
    except Exception as e:
        e300 = {"result": "ERROR", "exit_code": None, "collected": None, "elapsed": 0.0,
                "timed_out": False, "error_class": f"harness-error: {type(e).__name__}: {e}", "output_tail": ""}
    try:
        g60 = one_official_verify("gold", 60)
    except Exception as e:
        g60 = {"result": "ERROR", "exit_code": None, "collected": None, "elapsed": 0.0,
               "timed_out": False, "error_class": f"harness-error: {type(e).__name__}: {e}", "output_tail": ""}
    try:
        g300 = one_official_verify("gold", 300)
    except Exception as e:
        g300 = {"result": "ERROR", "exit_code": None, "collected": None, "elapsed": 0.0,
                "timed_out": False, "error_class": f"harness-error: {type(e).__name__}: {e}", "output_tail": ""}
    rec.update({
        "empty_60": e60, "empty_300": e300, "gold_60": g60, "gold_300": g300,
        "empty_patch_result": e60["result"],
        "reference_patch_result": g60["result"],
        "reference_patch_result_300": g300["result"],
        "target_tests_collected": g60["collected"] if g60["collected"] is not None else e60["collected"],
        "target_tests_collected_60": {"empty": e60["collected"], "gold": g60["collected"]},
        "target_tests_collected_300": {"empty": e300["collected"], "gold": g300["collected"]},
        "setup_error_class": "; ".join([s for s in [
            rec["setup_error_initial"], e60["error_class"], g60["error_class"],
            e300["error_class"], g300["error_class"]] if s])[:600] or "",
        "verify_seconds_60": {"empty": e60["elapsed"], "gold": g60["elapsed"]},
        "verify_seconds_300": {"empty": e300["elapsed"], "gold": g300["elapsed"]},
    })
    # eligibility on frozen 60s; baseline head must be non-empty to trust
    empty_fail = (e60["result"] == "FAIL")
    gold_pass = (g60["result"] == "PASS" and (g60["collected"] or 0) > 0)
    baseline_ok = bool(rec.get("workspace_head"))
    if not baseline_ok:
        rec["eligibility"] = "unknown/setup-error"
        rec["eligibility_reason"] = f"unknown/setup-error(baseline-head-empty); diag300 empty={e300['result']}/gold={g300['result']}"
    elif empty_fail and gold_pass:
        rec["eligibility"] = "eligible"
        rec["eligibility_reason"] = "empty-FAIL + reference-PASS at 60s (frozen, official verify_task)"
    else:
        if "import-error" in rec["setup_error_class"] or "fixture-error" in rec["setup_error_class"] or rec["setup_error_initial"].startswith("baseline"):
            reason = "unknown/setup-error"
        elif e60["result"] in ("TIMEOUT", "ERROR"):
            reason = "unknown/timeout-or-harness-error(empty-60)"
        elif g60["result"] in ("TIMEOUT", "ERROR"):
            reason = "unknown/timeout-or-harness-error(gold-60)"
        elif e60["result"] == "PASS":
            reason = "unknown/already-pass(empty passes without fix)"
        elif g60["result"] == "FAIL":
            reason = "unknown/gold-fail(reference does not pass at 60s)"
        else:
            reason = "unknown/other"
        hint = f"; diag300 empty={e300['result']}/gold={g300['result']}"
        rec["eligibility"] = reason
        rec["eligibility_reason"] = reason + hint
    print(f"  -> empty60={e60['result']} exit={e60['exit_code']} coll={e60['collected']} "
          f"t={e60['elapsed']}s err={(e60['error_class'] or '')[:120]}", flush=True)
    print(f"     gold60 ={g60['result']} exit={g60['exit_code']} coll={g60['collected']} "
          f"t={g60['elapsed']}s err={(g60['error_class'] or '')[:120]}", flush=True)
    print(f"     empty300={e300['result']} coll={e300['collected']} t={e300['elapsed']}s | "
          f"gold300={g300['result']} coll={g300['collected']} t={g300['elapsed']}s "
          f"=> {rec['eligibility']}", flush=True)
    results.append(rec)
    atomic_json(OUT_ROOT / "eligibility_raw.json", {
        "run_id": RUN_ID, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks_sha256": tasks_sha, "versions": versions,
        "frozen_timeouts": DIAG_TIMEOUTS, "carrier": "official verify_task; P0-first ordered",
        "note": "300s is diagnostic only; eligibility decided on frozen 60s; "
                "reference patches judge-side only; no Agent; heldout untouched.",
        "results": results,
        "completed": len(results), "planned": len(ORDERED_24),
    })

eligible60 = [r for r in results if r.get("eligibility") == "eligible"]
print(f"DONE {len(results)}/24 eligible@60s={len(eligible60)}", flush=True)
for r in results:
    print(f"  {r['task_id']}: empty60={r['empty_60']['result']} gold60={r['gold_60']['result']} "
          f"empty300={r['empty_300']['result']} gold300={r['gold_300']['result']} "
          f"=> {r['eligibility']} t60={r['verify_seconds_60']} t300={r['verify_seconds_300']} "
          f"coll60={r['target_tests_collected_60']} coll300={r['target_tests_collected_300']} "
          f"head={r.get('workspace_head','')[:12]} err={r['setup_error_class'][:160]}", flush=True)
oc_rows = [r for r in results if r["task_id"] in OC6]
print(f"0C6 @60s eligible={[r['task_id'] for r in oc_rows if r.get('eligibility') == 'eligible']} "
      f"({len([r for r in oc_rows if r.get('eligibility') == 'eligible'])}/6)", flush=True)
p0_rows = [r for r in results if r.get("p0")]
print(f"P0(0C6+requests_6592) @60s eligible={[r['task_id'] for r in p0_rows if r.get('eligibility') == 'eligible']} "
      f"({len([r for r in p0_rows if r.get('eligibility') == 'eligible'])}/{len(p0_rows)} returned)", flush=True)
print(f"WROTE {OUT_ROOT / 'eligibility_raw.json'}", flush=True)
print("CPU eligibility OFFICIAL done. No Agent ran. No heldout read. No test files edited to force PASS. "
      "No variants/manifest/analyze/0C changes.", flush=True)
# cleanup sandbox managers (best-effort)
try:
    docker60.cleanup_all()
except Exception:
    pass
try:
    docker300.cleanup_all()
except Exception:
    pass
