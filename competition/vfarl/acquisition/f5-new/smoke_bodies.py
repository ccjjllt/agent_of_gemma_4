"""Smoke-test F5 bodies against upstream (fixed) workspaces; expect pass."""
import json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).resolve().parent
REPLAY = ROOT / "competition/vfarl/replay"
ART = REPLAY / "artifacts/f5-new"
PY = str(REPLAY / ".venv/bin/python")

contracts = json.load(open(HERE / "selected-contracts.json"))["tasks"]
acq = {t["key"]: t for t in json.load(open(HERE / "acquired-40.json"))["tasks"]}
results = []
for c in contracts:
    key = c["key"]
    a = acq.get(key)
    if a is None:
        # key mismatch? match by instance_id
        cand = [t for t in json.load(open(HERE / "acquired-40.json"))["tasks"] if t["instance_id"] == c["instance_id"]]
        a = cand[0] if cand else None
    if a is None:
        results.append((key, "NO_ACQ", ""))
        continue
    canon = c["canonical_repo"]
    ups = ART / (canon.replace("/", "__") + "-upstream")
    body = c["body"]
    layout = c.get("source_layout", ".")
    # run with cwd=ups, PYTHONPATH set so `import <module>` resolves to workspace;
    # task-profile site-packages (same as S per repo) supplies third-party deps.
    env_path = str(ups) if layout == "." else str(ups / "src")
    prof = REPLAY / c.get("task_profile", ".venv/lib/python3.12/site-packages")
    code = "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n%s" % (str(prof), env_path, body)
    try:
        r = subprocess.run([PY, "-c", code], cwd=str(ups), capture_output=True, text=True, timeout=120)
        ok = (r.returncode == 0)
        out = (r.stdout[-1500:] if r.stdout else "") + (("\nSTDERR:" + r.stderr[-1500:]) if r.stderr and not ok else "")
        results.append((key, "PASS" if ok else "FAIL", out.strip().splitlines()[-1] if out.strip() else ""))
        print(f"{'PASS' if ok else 'FAIL'} {key}", flush=True)
        if not ok:
            print("---stdout---\n" + (r.stdout or "")[-2000:], flush=True)
            print("---stderr---\n" + (r.stderr or "")[-2000:], flush=True)
    except subprocess.TimeoutExpired:
        results.append((key, "TIMEOUT", ""))
        print(f"TIMEOUT {key}", flush=True)

npass = sum(1 for _, s, _ in results if s == "PASS")
print(f"smoke upstream: {npass}/{len(results)} pass")
json.loads(json.dumps(results))  # sanity
(HERE / "smoke-upstream.json").write_text(json.dumps([{"key": k, "status": s, "tail": t} for k, s, t in results], indent=2) + "\n")
