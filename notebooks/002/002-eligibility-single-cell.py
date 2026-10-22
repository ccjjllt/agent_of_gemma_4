#!/usr/bin/env python3
# 002 eligibility single-cell carrier (stage 0B CPU empty vs reference).
# Paste entire file into one Kaggle notebook code cell and run top-to-bottom.
# CPU-only: no vLLM, no Agent runs. Reference patches are used ONLY on the
# judge/analysis side (never injected into any Agent input; there is no Agent here).
# Red lines: never delete tests, never edit assertions to force PASS; eval timeout
# changes (if any) must apply to the whole arm identically as a new research
# condition (here 60s frozen + 300s diagnostic, both run, 300s never rewrites scores).
# Heldout 28 tasks are never read. 0C GPU carrier is untouched.
# For each of the frozen 24 dev_panel tasks: empty-patch verify (expect FAIL,
# target failure reproduced) vs reference-patch verify (expect PASS, target tests
# collected+executed and pass), each under BOTH 60s (frozen) and 300s (diagnostic).
# eligibility = empty FAILS AND reference PASSES under SAME frozen A-env+B-judge,
# decided BEFORE any agent对照 results are known; otherwise unknown with reason
# class (setup-error / gold-fail / timeout / already-pass).
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
import tarfile
import tempfile
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
      "(no Agent here), heldout untouched, 0C GPU carrier untouched.", flush=True)

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
# Introspect task schema (proves patch/test_patch present without leaking heldout)
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
# Verification helper discovery (official entry preferred)
ver_info = {}
try:
    import swegemma.harness.verification as hv
    ver_info["verification_module"] = hv.__name__
    ver_info["funcs"] = sorted([n for n in dir(hv) if not n.startswith("_")])[:60]
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
    "versions": versions, "selected_24": SELECTED_24, "diag_timeouts": DIAG_TIMEOUTS,
    "input_dirs": {"data": str(DATA_DIR), "wheelhouse": str(WHEELHOUSE_DIR),
                   "model": str(MODEL_PATH) if MODEL_PATH else None},
    "note": "0B CPU empty-vs-reference; 60 frozen + 300 diagnostic (300 never rewrites scores); "
            "reference patches judge-side only, no Agent; heldout untouched.",
})


def get_task_str(task, names):
    for n in names:
        try:
            v = getattr(task, n, None)
            if isinstance(v, str) and v:
                return v
        except Exception:
            pass
        try:
            if isinstance(task, dict) and isinstance(task.get(n), str) and task.get(n):
                return task.get(n)
        except Exception:
            pass
    # pydantic / dataclass fallback via __dict__
    try:
        d = task.__dict__ if hasattr(task, "__dict__") else {}
        if isinstance(d, dict) and isinstance(d.get(names[0] if isinstance(names, list) else names), str):
            return d.get(names[0] if isinstance(names, list) else names)
    except Exception:
        pass
    return ""


def task_field(task, name):
    try:
        v = getattr(task, name, None)
        if v is not None:
            return v
    except Exception:
        pass
    try:
        if hasattr(task, "__dict__") and isinstance(task.__dict__, dict):
            return task.__dict__.get(name)
    except Exception:
        pass
    return None


def parse_test_targets(test_patch):
    if not isinstance(test_patch, str) or not test_patch.strip():
        return []
    targets = []
    for line in test_patch.splitlines():
        if line.startswith("+++ "):
            p = line[4:].strip().split()[0]
            if p.startswith("b/"):
                p = p[2:]
            if p.endswith(".py") and p != "/dev/null":
                targets.append(p)
    # preserve order, dedupe
    seen_t = set()
    out = []
    for t in targets:
        if t not in seen_t:
            seen_t.add(t)
            out.append(t)
    # prefer non-conftest test files (mirror official verification filter)
    non_conftest = [t for t in out if Path(t).name != "conftest.py" and t.endswith(".py")]
    return non_conftest or out


def classify_setup_error(output, exc_name=""):
    blob = (output or "") + "\n" + (exc_name or "")
    low = blob.lower()
    if "modulenotfounderror" in low or "importerror" in low:
        # keep original module name hint (first 200 chars around the error)
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


def apply_patch_text(workspace, patch_text, label):
    if not isinstance(patch_text, str) or not patch_text.strip():
        return True, ""
    p = workspace / f"_elig_{label}.diff"
    p.write_text(patch_text, encoding="utf-8")
    # try git apply first (official style unified diff)
    for cmd in (["git", "apply", "--whitespace=fix", str(p)],
                ["patch", "-p1", "-i", str(p)]):
        try:
            r = subprocess.run(cmd, cwd=str(workspace), capture_output=True, text=True, timeout=60)
            if r.returncode == 0:
                return True, ""
            last_err = (r.stderr or r.stdout or "")[:800]
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return False, f"patch-apply-fail({label}): {last_err[:400]}"


def run_pytest(workspace, targets, timeout_s):
    junit = f"/tmp/_swegemma_junit_{uuid.uuid4().hex[:12]}.xml"
    try:
        subprocess.run(["rm", "-f", junit], timeout=10)
    except Exception:
        pass
    quoted = [t for t in targets if t.endswith(".py")]
    # mirror official verification.py: exclude conftest from positional args, -p no:anyio, timeout=0
    test_str = " ".join([f"'{t}'" for t in quoted]) if quoted else "."
    cmd = (f"cd '{workspace}' && PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -m pytest {test_str} "
           f"--junitxml={junit} -p no:anyio -o timeout=0 -o python_classes=\"Test* *Test\" -q")
    start = time.perf_counter()
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout_s + 10)
        elapsed = time.perf_counter() - start
        out = (r.stdout or "") + "\n---STDERR---\n" + (r.stderr or "")
        # timed_out if wall exceeded timeout (official gate) even if process returned
        timed_out = elapsed >= timeout_s
        return {"exit_code": r.returncode, "output": out[-6000:], "elapsed": elapsed,
                "timed_out": timed_out, "junit": junit}
    except subprocess.TimeoutExpired as e:
        elapsed = time.perf_counter() - start
        out = ""
        try:
            out = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""))[-4000:]
        except Exception:
            pass
        return {"exit_code": None, "output": (out + f"\n[TIMEOUT after {timeout_s}s]")[-6000:],
                "elapsed": elapsed, "timed_out": True, "junit": junit}


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


ELIG_TMP = Path(tempfile.mkdtemp(prefix="elig_0b_"))
print(f"elig tmp={ELIG_TMP}", flush=True)
results = []
for idx, tid in enumerate(SELECTED_24, 1):
    task = by_id[tid]
    repo = task_field(task, "repo") or ""
    base_commit = task_field(task, "base_commit") or ""
    patch_gold = get_task_str(task, ["patch"]) if not isinstance(task_field(task, "patch"), str) else (task_field(task, "patch") or "")
    # get_task_str fallback already; ensure string
    if not isinstance(patch_gold, str):
        patch_gold = ""
    test_patch = task_field(task, "test_patch") or ""
    if not isinstance(test_patch, str):
        test_patch = ""
    targets = parse_test_targets(test_patch)
    print(f"[{idx}/24] {tid} repo={repo} gold_chars={len(patch_gold)} test_chars={len(test_patch)} "
          f"targets={targets[:4]}{'...' if len(targets) > 4 else ''} collected_hint=pending", flush=True)
    rec = {"task_id": tid, "repo": repo, "base_commit": base_commit,
           "target_tests_inferred": targets, "patch_chars": len(patch_gold),
           "test_patch_chars": len(test_patch)}
    # ---- workspace setup (extract snapshot once per task) ----
    ws_base = ELIG_TMP / f"ws_{tid}"
    setup_err = ""
    snap = SNAP_DIR / f"{tid}.tgz"
    try:
        if ws_base.exists():
            shutil.rmtree(ws_base)
        ws_base.mkdir(parents=True)
        assert snap.is_file(), f"snapshot missing: {snap.name}"
        with tarfile.open(snap, "r:gz") as tf:
            # snapshot contains a git repo; extract into ws_base
            # find top-level dir: extract then normalise to ws_base/workspace
            tmp_ex = ELIG_TMP / f"ex_{tid}"
            if tmp_ex.exists():
                shutil.rmtree(tmp_ex)
            tmp_ex.mkdir(parents=True)
            tf.extractall(path=str(tmp_ex))
            # locate workspace: if single top dir, use it; else use tmp_ex
            kids = [p for p in tmp_ex.iterdir()]
            if len(kids) == 1 and kids[0].is_dir():
                # move contents into ws_base
                for child in kids[0].iterdir():
                    shutil.move(str(child), str(ws_base / child.name))
                shutil.rmtree(tmp_ex)
            else:
                for child in kids:
                    shutil.move(str(child), str(ws_base / child.name))
                shutil.rmtree(tmp_ex)
        # baseline commit hash for reset
        try:
            br = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ws_base),
                                capture_output=True, text=True, timeout=30)
            rec["workspace_head"] = (br.stdout or "").strip()[:16]
        except Exception as e:
            rec["workspace_head"] = f"rev-parse-fail: {type(e).__name__}"
        # editable install from offline wheels (best-effort; record class, do not stop)
        wheels_link = str(WHEELS_DATA_DIR) if WHEELS_DATA_DIR.is_dir() else None
        if wheels_link:
            try:
                ir = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "--no-index",
                     f"--find-links={wheels_link}", "-e", "."],
                    cwd=str(ws_base), capture_output=True, text=True, timeout=180)
                if ir.returncode != 0:
                    setup_err = f"pip-install-fail: {(ir.stderr or ir.stdout or '')[-400:]}"
            except subprocess.TimeoutExpired:
                setup_err = "pip-install-timeout(180s)"
            except Exception as e:
                setup_err = f"pip-install-error: {type(e).__name__}: {e}"
        # ensure baseline commit exists for clean resets
        try:
            subprocess.run(["git", "add", "-N", "."], cwd=str(ws_base), timeout=30)
            subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=str(ws_base), timeout=30)
        except Exception:
            pass
    except AssertionError as e:
        setup_err = f"snapshot-missing: {e}"
    except Exception as e:
        setup_err = f"setup-error: {type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"
    if setup_err:
        print(f"  setup note: {setup_err[:400]}", flush=True)
    rec["setup_error_initial"] = setup_err[:800] if setup_err else ""

    def one_verify(kind, timeout_s):
        # fresh reset to baseline before each verify
        try:
            subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=str(ws_base),
                           capture_output=True, timeout=60)
            subprocess.run(["git", "clean", "-fdx"], cwd=str(ws_base),
                           capture_output=True, timeout=60)
        except Exception as e:
            return {"result": "ERROR", "exit_code": None, "collected": None,
                    "elapsed": 0.0, "timed_out": False,
                    "error_class": f"reset-fail: {type(e).__name__}",
                    "output_tail": ""}
        # apply test_patch always (official B restores/applies test_patch)
        ok1, err1 = apply_patch_text(ws_base, test_patch, "test_patch")
        if not ok1:
            return {"result": "ERROR", "exit_code": None, "collected": None,
                    "elapsed": 0.0, "timed_out": False,
                    "error_class": err1, "output_tail": ""}
        if kind == "gold":
            ok2, err2 = apply_patch_text(ws_base, patch_gold, "gold")
            if not ok2:
                return {"result": "ERROR", "exit_code": None, "collected": None,
                        "elapsed": 0.0, "timed_out": False,
                        "error_class": err2, "output_tail": ""}
        # else empty: no source patch (""), only test_patch
        pr = run_pytest(ws_base, targets, timeout_s)
        collected = parse_collected(pr["output"])
        err_cls = classify_setup_error(pr["output"])
        if pr["timed_out"] and pr["exit_code"] is None:
            res = "TIMEOUT"
        elif pr["exit_code"] == 0:
            res = "PASS"
        elif pr["exit_code"] is None:
            res = "TIMEOUT"
        else:
            res = "FAIL"
        # timeout with no collection is still TIMEOUT class for eligibility reason
        return {"result": res, "exit_code": pr["exit_code"], "collected": collected,
                "elapsed": round(pr["elapsed"], 1), "timed_out": pr["timed_out"],
                "error_class": err_cls, "output_tail": pr["output"][-2000:]}

    # 4 verifies per task: empty×{60,300}, gold×{60,300}
    try:
        e60 = one_verify("empty", 60)
    except Exception as e:
        e60 = {"result": "ERROR", "exit_code": None, "collected": None, "elapsed": 0.0,
               "timed_out": False, "error_class": f"harness-error: {type(e).__name__}: {e}", "output_tail": ""}
    try:
        e300 = one_verify("empty", 300)
    except Exception as e:
        e300 = {"result": "ERROR", "exit_code": None, "collected": None, "elapsed": 0.0,
                "timed_out": False, "error_class": f"harness-error: {type(e).__name__}: {e}", "output_tail": ""}
    try:
        g60 = one_verify("gold", 60)
    except Exception as e:
        g60 = {"result": "ERROR", "exit_code": None, "collected": None, "elapsed": 0.0,
               "timed_out": False, "error_class": f"harness-error: {type(e).__name__}: {e}", "output_tail": ""}
    try:
        g300 = one_verify("gold", 300)
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
    # eligibility on frozen 60s (official diagnostic rule); 300s kept as diagnostic only
    # eligible = empty FAILS (target failure reproduced) AND reference PASSES (collected+pass)
    empty_fail = (e60["result"] == "FAIL")
    gold_pass = (g60["result"] == "PASS" and (g60["collected"] or 0) > 0)
    if empty_fail and gold_pass:
        rec["eligibility"] = "eligible"
        rec["eligibility_reason"] = "empty-FAIL + reference-PASS at 60s (frozen)"
    else:
        if rec["setup_error_initial"].startswith("snapshot-missing") or "pip-install-fail" in rec["setup_error_class"] or "import-error" in rec["setup_error_class"] or "fixture-error" in rec["setup_error_class"]:
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
        # 300s diagnostic hint appended (never rewrites 60s eligibility)
        hint = f"; diag300 empty={e300['result']}/gold={g300['result']}"
        rec["eligibility"] = reason
        rec["eligibility_reason"] = reason + hint
    print(f"  -> empty60={e60['result']} exit={e60['exit_code']} coll={e60['collected']} "
          f"t={e60['elapsed']}s err={e60['error_class'][:120]}", flush=True)
    print(f"     gold60 ={g60['result']} exit={g60['exit_code']} coll={g60['collected']} "
          f"t={g60['elapsed']}s err={g60['error_class'][:120]}", flush=True)
    print(f"     empty300={e300['result']} coll={e300['collected']} t={e300['elapsed']}s | "
          f"gold300={g300['result']} coll={g300['collected']} t={g300['elapsed']}s "
          f"=> {rec['eligibility']}", flush=True)
    results.append(rec)
    atomic_json(OUT_ROOT / "eligibility_raw.json", {
        "run_id": RUN_ID, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks_sha256": tasks_sha, "versions": versions,
        "frozen_timeouts": DIAG_TIMEOUTS,
        "note": "300s is diagnostic only; eligibility decided on frozen 60s; "
                "reference patches judge-side only; no Agent; heldout untouched.",
        "results": results,
        "completed": len(results), "planned": len(SELECTED_24),
    })
    # free workspace to bound disk
    try:
        shutil.rmtree(ws_base)
    except Exception:
        pass

eligible60 = [r for r in results if r.get("eligibility") == "eligible"]
print(f"DONE {len(results)}/24 eligible@60s={len(eligible60)}", flush=True)
for r in results:
    print(f"  {r['task_id']}: empty60={r['empty_60']['result']} gold60={r['gold_60']['result']} "
          f"empty300={r['empty_300']['result']} gold300={r['gold_300']['result']} "
          f"=> {r['eligibility']} t60={r['verify_seconds_60']} t300={r['verify_seconds_300']} "
          f"coll60={r['target_tests_collected_60']} coll300={r['target_tests_collected_300']} "
          f"err={r['setup_error_class'][:160]}", flush=True)
oc_rows = [r for r in results if r["task_id"] in OC6]
print(f"0C6 @60s eligible={[r['task_id'] for r in oc_rows if r.get('eligibility') == 'eligible']} "
      f"({len([r for r in oc_rows if r.get('eligibility') == 'eligible'])}/6)", flush=True)
print(f"WROTE {OUT_ROOT / 'eligibility_raw.json'}", flush=True)
print("CPU eligibility done. No Agent ran. No heldout read. No test files edited to force PASS.", flush=True)
