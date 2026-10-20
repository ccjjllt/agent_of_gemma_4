"""P013 GPU pre-verification (V1->V2->V3->V4), single GPU machine, no model/harness/network.

Reads task material from a mounted private dataset (see dataset-metadata.json):
  <dataset>/tasks-13.jsonl
  <dataset>/snapshots/<instance_id>.tar.gz
  <dataset>/criteria/test_p013_*.py            (criterion test files = test_patch source)
  <dataset>/criteria/replay/<instance_id>.json (teacher edit ops for V3 calibration)
  <dataset>/criteria/criteria-index.json       (task -> test_file / F2P node map)

Per task (execution order: t12 highest-risk first, then t01 R3 gate, then rest):
  V1 unpack snapshot -> build venv -> `pip install -e . --no-build-isolation`
      (record python/pip versions; --no-build-isolation keeps it offline-friendly)
  V2 copy criterion file into tests/ (test_patch adds ONLY the criterion file,
      never any source fix) -> run F2P node, expect exit != 0 behavior failure.
      Collection errors / missing deps classify as ENV failure, not prefail.
  V3 replay teacher edits (str_replace_editor semantics: create/str_replace,
      exact match first, then leading/trailing-blankline-tolerant match per
      apply_warning in replay JSON) -> run F2P node (expect exit 0) + full
      same-file run as P2P (expect exit 0).
  Write per-task result.json (includes sha256 of raw log text).

V4 gating (selection only, no fabrication):
  Only V2+V3-double-pass tasks enter the 12 slots. t01 failure -> R3
  freeze_invalid, stop. One non-t01 failure may consume floating spare s01
  (R1/R2); more failures than spare capacity -> R4 freeze_invalid, stop.
  The script writes dev12-freeze.json with ONLY actually-passed tasks.

Usage (on GPU box, dataset mounted, no internet):
  python3 preverify.py --dataset /kaggle/input/p013-task-material \\
      --workdir /kaggle/working/preverify [--tasks t12,t01] [--no-stop-on-r3]
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import venv
from pathlib import Path

TASK_ORDER = ["t12", "t01", "t02", "t03", "t04", "t05", "t06",
              "t07", "t08", "t09", "t10", "t11", "s01"]
SLOT_TASKS = [t for t in TASK_ORDER if t != "s01"]  # 12 freeze slots; s01 floating spare

INSTALL_TIMEOUT = 1200
PYTEST_TIMEOUT = 900


SNAP_DIR = CRIT_DIR = REPLAY_DIR = None
INDEX_PATH = None


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def run(cmd, cwd, timeout):
    try:
        r = subprocess.run(cmd, cwd=str(cwd), text=True,
                           capture_output=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(
            e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(
            e.stderr, bytes) else (e.stderr or "")
        return 124, out + err + f"\n[TIMEOUT after {timeout}s]\n"


def classify_v2(exit_code: int, log: str) -> str:
    """V2 verdict on the buggy snapshot with criterion applied.
    prefail_confirmed: exit!=0 AND a real assertion/behavior failure.
    env_failure:       exit!=0 but collection error / missing dep / no tests ran.
    unexpected_pass:   exit==0 (node passes on buggy -> invalid F2P).
    """
    low = log.lower()
    env_markers = ["modulenotfounderror", "no module named", "importerror",
                   "collection error", "error: unrecognized", "no tests ran",
                   "file or directory not found"]
    if exit_code == 0:
        return "unexpected_pass"
    if exit_code in (4, 5) or any(m in low for m in env_markers):
        return "env_failure"
    if "failed" in low or "assertionerror" in low or "assert" in low:
        return "prefail_confirmed"
    return "env_failure"  # conservative: unknown non-zero output is not a behavior fail


def apply_replay_op(work_tree: Path, op: dict) -> tuple:
    """Apply one replay op. Returns (ok, warning_or_error)."""
    kind = op.get("op")
    rel = op.get("path", "")
    target = work_tree / rel
    if kind == "create":
        text = op.get("new_str", op.get("content", op.get("file_text")))
        if text is None:
            return False, "create op missing new_str/content"
        if target.exists():
            return False, f"create target exists: {rel}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return True, ""
    if kind == "str_replace":
        old, new = op.get("old_str"), op.get("new_str")
        if old is None or new is None:
            return False, "str_replace op missing old_str/new_str"
        try:
            cur = target.read_text()
        except OSError as e:
            return False, f"cannot read {rel}: {e}"
        if old in cur:
            target.write_text(cur.replace(old, new, 1))
            return True, ""
        # tolerant match per replay apply_warning: strip one leading newline,
        # then leading/trailing blank-line tolerance.
        for variant_i, variant in enumerate(
                [old.lstrip("\n"), old.strip("\n"), old.strip()]):
            if variant and variant in cur:
                target.write_text(cur.replace(variant, new.strip("\n") if variant_i == 2 else new, 1))
                return True, (f"blankline-tolerant match (level {variant_i}) for {rel}")
        return False, f"old_str not found in {rel}"
    return False, f"unknown op kind: {kind}"


def verify_task(task: dict, crit: dict, ds: Path, out_root: Path) -> dict:
    tid = task["task_id"]
    iid = task["instance_id"]
    res = {"task_id": tid, "instance_id": iid, "repo": task.get("repo"),
           "base_commit": task.get("base_commit")}
    work = out_root / tid
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # V1: unpack snapshot (offline; no git/network).
    snap = SNAP_DIR / task["snapshot"]
    tree = work / "tree"
    tree.mkdir()
    try:
        with tarfile.open(snap, "r:gz") as t:
            t.extractall(work / "stage")
        staged = list((work / "stage").iterdir())
        src = staged[0] if len(staged) == 1 and staged[0].is_dir() else work / "stage"
        for p in src.iterdir():
            shutil.move(str(p), str(tree / p.name))
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        res.update({"verdict": "ENV_FAIL", "stage": "V1_unpack",
                    "detail": f"{type(e).__name__}: {e}"[:300]})
        return res

    # V1: venv + install (offline-friendly flags; versions recorded).
    vpy = work / ".venv" / "bin" / "python"
    try:
        venv.create(work / ".venv", with_pip=True)
    except Exception as e:  # noqa: BLE001
        res.update({"verdict": "ENV_FAIL", "stage": "V1_venv",
                    "detail": f"{type(e).__name__}: {e}"[:300]})
        return res
    pyver = run([str(vpy), "--version"], tree, 60)[1].strip()
    inst_exit, inst_log = run(
        [str(vpy), "-m", "pip", "install", "-e", ".", "--no-build-isolation",
         "--no-index" if os.environ.get("P013_PIP_NO_INDEX") == "1" else "--no-input"],
        tree, INSTALL_TIMEOUT)
    (work / "install.log").write_text(inst_log)
    freeze = run([str(vpy), "-m", "pip", "freeze"], tree, 120)[1]
    res["venv"] = {"python": pyver, "install_exit": inst_exit,
                   "install_log_sha256": sha256_text(inst_log),
                   "pip_freeze": freeze[:4000]}
    if inst_exit != 0:
        res.update({"verdict": "ENV_FAIL", "stage": "V1_install",
                    "detail": f"pip install -e . exit {inst_exit}"})
        return res

    # V2: test_patch = copy ONLY the criterion file into tests/.
    test_file = crit["test_file"]
    f2p_node = crit["f2p_node"]
    src_crit = CRIT_DIR / test_file
    dst_crit = tree / "tests" / test_file
    try:
        shutil.copyfile(src_crit, dst_crit)
    except OSError as e:
        res.update({"verdict": "ENV_FAIL", "stage": "V2_test_patch",
                    "detail": f"copy criterion: {e}"[:200]})
        return res
    v2_exit, v2_log = run(
        [str(vpy), "-m", "pytest", f"tests/{test_file}::{f2p_node}",
         "-x", "-q", "-p", "no:cacheprovider"], tree, PYTEST_TIMEOUT)
    (work / "v2-prefail.log").write_text(v2_log)
    v2_verdict = classify_v2(v2_exit, v2_log)
    res["v2"] = {"node": f2p_node, "exit": v2_exit,
                 "log_sha256": sha256_text(v2_log), "verdict": v2_verdict}
    if v2_verdict != "prefail_confirmed":
        res.update({"verdict": "V2_FAIL" if v2_verdict == "unexpected_pass" else "ENV_FAIL",
                    "stage": "V2_prefail", "detail": f"V2 classified {v2_verdict}"})
        return res

    # V3: replay teacher edits, then F2P + full-file P2P must both pass.
    replay_path = REPLAY_DIR / f"{iid}.json"
    try:
        replay = json.loads(replay_path.read_text())
    except OSError as e:
        res.update({"verdict": "REPLAY_FAIL", "stage": "V3_replay_load",
                    "detail": f"replay unreadable: {e}"[:200]})
        return res
    warnings = []
    for op in replay.get("ops", []):
        ok, msg = apply_replay_op(tree, op)
        if not ok:
            res.update({"verdict": "REPLAY_FAIL", "stage": "V3_replay",
                        "replay_applied": op.get("seq"), "detail": msg[:300]})
            return res
        if msg:
            warnings.append(msg)
    res["replay"] = {"ops": len(replay.get("ops", [])),
                     "teacher_traj": replay.get("teacher_traj_id"),
                     "warnings": warnings}
    v3_exit, v3_log = run(
        [str(vpy), "-m", "pytest", f"tests/{test_file}::{f2p_node}",
         "-q", "-p", "no:cacheprovider"], tree, PYTEST_TIMEOUT)
    (work / "v3-postpass.log").write_text(v3_log)
    p2p_exit, p2p_log = run(
        [str(vpy), "-m", "pytest", f"tests/{test_file}",
         "-q", "-p", "no:cacheprovider"], tree, PYTEST_TIMEOUT)
    (work / "v3-p2p.log").write_text(p2p_log)
    res["v3"] = {"f2p_exit": v3_exit, "f2p_log_sha256": sha256_text(v3_log),
                 "p2p_exit": p2p_exit, "p2p_log_sha256": sha256_text(p2p_log)}
    if v3_exit == 0 and p2p_exit == 0:
        res.update({"verdict": "PASS", "stage": "V3_done"})
    else:
        res.update({"verdict": "V3_FAIL", "stage": "V3_postpass",
                    "detail": f"F2P exit {v3_exit}, P2P exit {p2p_exit}"})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.environ.get(
        "P013_TASK_MATERIAL", "/kaggle/input/p013-task-material"))
    ap.add_argument("--workdir", default="/kaggle/working/preverify")
    ap.add_argument("--tasks", default=",".join(TASK_ORDER),
                    help="comma-separated subset, run in given order")
    ap.add_argument("--no-stop-on-r3", action="store_true")
    args = ap.parse_args()

    ds = Path(args.dataset)
    out = Path(args.workdir)
    out.mkdir(parents=True, exist_ok=True)
    # --dir-mode zip datasets mount folders as <name>.zip; unpack-into-place
    # fallback (P012 precedent). Flat-file datasets need no action.
    import zipfile as _zf, tempfile as _tf, shutil as _sh
    for _name in ("snapshots", "criteria"):
        if not (ds / _name).is_dir() and (ds / f"{_name}.zip").is_file():
            (ds / _name).mkdir(parents=True, exist_ok=True)
            with _tf.TemporaryDirectory() as _td:
                with _zf.ZipFile(ds / f"{_name}.zip") as _z:
                    _z.extractall(_td)
                _inner = Path(_td) / _name
                _src = _inner if _inner.is_dir() else Path(_td)
                for _p in _src.iterdir():
                    _dst = ds / _name / _p.name
                    if _dst.exists():
                        continue
                    if _p.is_dir():
                        _sh.copytree(_p, _dst)
                    else:
                        _sh.copy2(_p, _dst)
    # Layout detection: nested (snapshots/ + criteria/ dirs, incl. zip-fallback
    # output) vs flat (all 41 files at dataset root, v3+). Module-level dirs
    # are set once; verify_task resolves through them.
    global SNAP_DIR, CRIT_DIR, REPLAY_DIR, INDEX_PATH
    if (ds / "snapshots").is_dir() and (ds / "criteria").is_dir():
        SNAP_DIR, CRIT_DIR = ds / "snapshots", ds / "criteria"
        REPLAY_DIR = CRIT_DIR / "replay"
        INDEX_PATH = CRIT_DIR / "criteria-index.json"
    else:
        SNAP_DIR = CRIT_DIR = REPLAY_DIR = ds
        INDEX_PATH = ds / "criteria-index.json"
    assert INDEX_PATH.is_file(), f"missing criteria index under {ds}: " + \
        ",".join(sorted(p.name for p in ds.iterdir())[:10])
    tasks = {json.loads(line)["task_id"]: json.loads(line)
             for line in (ds / "tasks-13.jsonl").read_text().splitlines() if line.strip()}
    index = {c["task_id"]: c for c in
             json.loads(INDEX_PATH.read_text())["items"]}

    ordered = [t for t in args.tasks.split(",") if t in tasks]
    # R3/R4 state.
    results, freeze_invalid, stop_reason = {}, False, ""
    for tid in ordered:
        r = verify_task(tasks[tid], index[tid], ds, out)
        results[tid] = r
        (out / f"result-{tid}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"task": tid, "verdict": r["verdict"],
                          "stage": r.get("stage")}), flush=True)
        if tid == "t01" and r["verdict"] != "PASS" and not args.no_stop_on_r3:
            freeze_invalid, stop_reason = True, (
                f"R3: t01 (click, no same-repo spare) {r['verdict']} "
                f"at {r.get('stage')}; 12-task freeze INVALID, stop.")
            print(stop_reason, flush=True)
            break

    # V4 selection: only double-pass tasks; s01 floats as R1/R2 spare.
    passed = {t: results[t] for t in ordered
              if results.get(t, {}).get("verdict") == "PASS"}
    items, substitutions = [], []
    if not freeze_invalid:
        failed_slots = [t for t in SLOT_TASKS if t in ordered and t not in passed]
        spare_ok = "s01" in passed
        if failed_slots and not spare_ok:
            # No spare consumed yet but spare itself unavailable/failed.
            if "s01" in ordered and "s01" in results and results["s01"].get("verdict") != "PASS":
                freeze_invalid, stop_reason = True, (
                    "R2/R4: slot failures without a passing spare "
                    f"(failed={failed_slots}, s01={results['s01'].get('verdict')}); "
                    "freeze INVALID.")
            elif len(failed_slots) > 1 or "s01" not in ordered:
                freeze_invalid, stop_reason = True, (
                    f"R4: {len(failed_slots)} slot failures exceed single-spare "
                    f"capacity (failed={failed_slots}); freeze INVALID.")
        if not freeze_invalid:
            pending = [t for t in failed_slots if t != "s01"]
            use_spare = bool(pending) and spare_ok
            if use_spare:
                substitutions.append(
                    f"R1/R2: {pending[0]} failed; floating spare s01 substitutes "
                    "(quota shifts, still 12 tasks).")
            if len(pending) > 1:
                freeze_invalid, stop_reason = True, (
                    f"R4: multiple slot failures {pending}; single spare "
                    "cannot cover; freeze INVALID.")
            if not freeze_invalid:
                skipped = set(pending[:1]) if use_spare else set()
                for t in SLOT_TASKS:
                    if t in passed and t not in skipped:
                        r = passed[t]
                        items.append({
                            "task_id": t, "repo": r["repo"],
                            "instance_id": r["instance_id"],
                            "base_commit": r["base_commit"],
                            "criterion": index[t]["test_file"],
                            "f2p_node": index[t]["f2p_node"],
                            "v2_log_sha256": r["v2"]["log_sha256"],
                            "v3_f2p_log_sha256": r["v3"]["f2p_log_sha256"],
                            "v3_p2p_log_sha256": r["v3"]["p2p_log_sha256"],
                            "pip_freeze_tail": r["venv"]["pip_freeze"][-500:]})
                if use_spare:
                    r = passed["s01"]
                    items.append({
                        "task_id": "s01", "substitute_for": pending[0],
                        "repo": r["repo"], "instance_id": r["instance_id"],
                        "base_commit": r["base_commit"],
                        "criterion": index["s01"]["test_file"],
                        "f2p_node": index["s01"]["f2p_node"],
                        "v2_log_sha256": r["v2"]["log_sha256"],
                        "v3_f2p_log_sha256": r["v3"]["f2p_log_sha256"],
                        "v3_p2p_log_sha256": r["v3"]["p2p_log_sha256"],
                        "pip_freeze_tail": r["venv"]["pip_freeze"][-500:]})
    verdict = ("FREEZE INVALID (" + stop_reason + ")" if freeze_invalid
               else ("FREEZE VALID" if len(items) == 12 else
                     f"INCOMPLETE ({len(items)}/12 passed; GPU run unfinished, "
                     "no fabrication)"))
    freeze = {"produced_by": "preverify.py (GPU, V1->V2->V3->V4)",
              "content_read": True, "tests_executed": True,
              "count": len(items), "substitutions": substitutions,
              "stop_reason": stop_reason, "verdict": verdict, "items": items}
    (out / "dev12-freeze.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"verdict": verdict, "count": len(items)}), flush=True)
    return 2 if freeze_invalid else 0


if __name__ == "__main__":
    sys.exit(main())
