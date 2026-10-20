"""P013 step-2 CPU materialization (metadata-only, no GPU judging).

Scope (strict):
- Reads ONLY identifier columns from SWE-smith shards:
  file_row_number / instance_id / resolved / model / traj_id.
  NEVER selects messages / patch / test content, never reads issue text,
  never reads heldout or reserved tasks.
- Resolves each issue's buggy commit via `git ls-remote` on the swesmith
  mirror branch refs/heads/<instance_id> (commit hash only, no content).
- Shallow single-branch clone per issue into work/snapshots/<instance_id>/,
  verifies HEAD == buggy commit, packs snapshots/<instance_id>.tar.gz.
- Per-task CPU environment checks ONLY:
  snapshot exists, top package importable (PYTHONPATH=snapshot),
  `pytest --collect-only` starts. NEVER runs FAIL_TO_PASS / PASS_TO_PASS.
- Deps: static build-manifest record only (pyproject/setup.cfg/setup.py read
  as text metadata, never executed); real wheel fetch deferred to GPU Linux.
- Substitution rules R1-R4 from candidate-pool.md; R3/R4 invalidate freeze.

Outputs (under research/013/eval-31b-pair/):
  tasks-13.jsonl, materialize-manifest.json, materialize-report.md,
  gpu-prevalidate-checklist.md, deps/<iid>.requirements-record.json,
  snapshots/<iid>.tar.gz (git-ignored), work/ (git-ignored scratch).
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

SHARDS = [
    ROOT / "research/011/data-scaling-20261018/shards/tool-00000-of-00008.parquet",
    ROOT / "research/011/data-scaling-20261018/shards/tool-00001-of-00008.parquet",
    ROOT / "research/011/data-scaling-20261018/shards/tool-00002-of-00008.parquet",
    ROOT / "research/011/data-scaling-20261018/shards/tool-00003-of-00008.parquet",
    ROOT / "training/artifacts/external/swe-smith/tool-00004-of-00008.parquet",
    ROOT / "research/011/data-scaling-20261018/shards/tool-00005-of-00008.parquet",
    ROOT / "research/011/data-scaling-20261018/shards/tool-00006-of-00008.parquet",
    ROOT / "research/011/data-scaling-20261018/shards/tool-00007-of-00008.parquet",
]

# slot, repo, instance_id, hash-rank, sha16 prefix (metadata from candidate-pool.md)
TASKS = [
    ("t01", "pallets/click", "pallets__click.fde47b4b.pr_2604", 0, "bd4d6faeaa3721a8"),
    ("t02", "pallets/markupsafe", "pallets__markupsafe.620c06c9.pr_402", 0, "3eff07e0ae4bbfb4"),
    ("t03", "pallets/markupsafe", "pallets__markupsafe.620c06c9.pr_418", 1, "43929120a8c952e4"),
    ("t04", "pallets/markupsafe", "pallets__markupsafe.620c06c9.pr_379", 2, "514d8660aa8807a"),
    ("t05", "pallets/markupsafe", "pallets__markupsafe.620c06c9.pr_400", 3, "cdc07f0062821eb8"),
    ("t06", "pydicom/pydicom", "pydicom__pydicom.7d361b3d.pr_1938", 0, "1799dc4b50293cca"),
    ("t07", "pydicom/pydicom", "pydicom__pydicom.7d361b3d.pr_2069", 1, "1cf4dad182a95bb9"),
    ("t08", "pydicom/pydicom", "pydicom__pydicom.7d361b3d.pr_1867", 2, "44d8358933f9c08b"),
    ("t09", "pydicom/pydicom", "pydicom__pydicom.7d361b3d.pr_2163", 3, "4bf520de1fd4e2b"),
    ("t10", "pydicom/pydicom", "pydicom__pydicom.7d361b3d.pr_1859", 4, "4ebe922ef77c5026"),
    ("t11", "pydicom/pydicom", "pydicom__pydicom.7d361b3d.pr_1995", 5, "83fd1a730733751b"),
    ("t12", "pydicom/pydicom", "pydicom__pydicom.7d361b3d.pr_1987", 6, "a11fbe936e43f847"),
]
SPARE = ("s01", "pydicom/pydicom", "pydicom__pydicom.7d361b3d.pr_1920", 7, "d53ea788402e88da")

TOP_PKG = {"pallets/click": "click", "pallets/markupsafe": "markupsafe", "pydicom/pydicom": "pydicom"}

LS_TIMEOUT = 90
CLONE_TIMEOUT = 300
IMPORT_TIMEOUT = 60
COLLECT_TIMEOUT = 180


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_bytes(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        if f.is_file() and ".git/" not in f.as_posix().replace(str(p), ""):
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def verify_metadata(iid: str):
    """Identifier columns only. No messages/patch/test projection, ever."""
    import duckdb

    c = duckdb.connect()
    rows = 0
    models = set()
    first = None
    for s in SHARDS:
        # metadata-only projection (identifiers + labels, no content columns)
        for number, model, traj in c.execute(
            "SELECT file_row_number, model, traj_id FROM read_parquet(?)"
            " WHERE instance_id=? AND resolved=true",
            [str(s), iid],
        ).fetchall():
            rows += 1
            models.add(model)
            if first is None:
                first = {"source_file": str(s.relative_to(ROOT)), "row_index": number, "traj_id": traj}
    return {"resolved_rows": rows, "models": sorted(models), "first_row": first}


def ls_remote(mirror: str, iid: str):
    r = subprocess.run(
        ["git", "ls-remote", f"https://github.com/{mirror}.git", f"refs/heads/{iid}"],
        text=True, capture_output=True, timeout=LS_TIMEOUT,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ls-remote exit {r.returncode}: {(r.stderr or '')[:300]}")
    out = (r.stdout or "").strip()
    if not out:
        raise RuntimeError("ls-remote empty (branch missing)")
    commit = out.split()[0]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(f"bad commit: {out[:120]}")
    return commit, out


def clone_snapshot(mirror: str, iid: str, commit: str):
    dest = HERE / "work" / "snapshots" / iid
    if (dest / ".git").is_dir():
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=dest,
                text=True, capture_output=True, timeout=30,
            )
            if head.returncode == 0 and head.stdout.strip() == commit:
                return dest, True  # reuse
        except subprocess.TimeoutExpired:
            pass
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["git", "clone", "--branch", iid, "--depth", "1",
         f"https://github.com/{mirror}.git", str(dest)],
        text=True, capture_output=True, timeout=CLONE_TIMEOUT,
    )
    if r.returncode != 0:
        raise RuntimeError(f"clone exit {r.returncode}: {(r.stderr or '')[:300]}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=dest,
        text=True, capture_output=True, timeout=30,
    )
    if head.stdout.strip() != commit:
        raise RuntimeError(f"HEAD {head.stdout.strip()[:12]} != ls-remote {commit[:12]}")
    return dest, False


def pack_snapshot(iid: str, dest: Path):
    snap_dir = HERE / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    arc = snap_dir / f"{iid}.tar.gz"
    if arc.exists():
        arc.unlink()
    with tarfile.open(arc, "w:gz") as t:
        t.add(dest, arcname=iid)
    return arc


def import_check(snapshot: Path, pkg: str):
    src = snapshot / "src"
    probe_path = str(src if src.is_dir() else snapshot)
    env = dict(os.environ, PYTHONPATH=probe_path)
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             f"import {pkg}; print(getattr({pkg}, '__version__', 'ok')); print({pkg}.__file__)"],
            cwd=snapshot, env=env, text=True,
            capture_output=True, timeout=IMPORT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "detail": f"import {pkg} >{IMPORT_TIMEOUT}s"}
    out = ((r.stdout or "") + (r.stderr or ""))[-300:]
    if r.returncode == 0:
        lines = (r.stdout or "").strip().splitlines()
        origin = lines[-1] if lines else ""
        snap = str(snapshot)
        status = "import_ok_snapshot" if origin.startswith(snap) else "import_ok_sitepackages"
        return {"status": status, "detail": " | ".join(lines)[:200]}
    return {"status": "import_failed", "detail": out.replace("\n", " | ")[:300]}


def collect_check(snapshot: Path):
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
            cwd=snapshot, text=True,
            capture_output=True, timeout=COLLECT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"status": "collect_timeout", "detail": f"collect >{COLLECT_TIMEOUT}s"}
    tail = (r.stdout or "")[-1500:]
    m = re.search(r"(\d+)\s+tests?\s+collected", tail)
    n = int(m.group(1)) if m else None
    if r.returncode == 0:
        return {"status": "collect_ok", "detail": f"{n} collected" if n is not None else "exit0"}
    if n is not None:
        return {"status": "collect_partial", "detail": f"exit={r.returncode},{n} collected"}
    err = ((r.stdout or "") + (r.stderr or ""))[-300:].replace("\n", " | ")
    return {"status": "collect_failed", "detail": f"exit={r.returncode}: {err[:250]}"}


def deps_record(snapshot: Path, iid: str):
    rec = {"build_files": {}, "license": None, "note": "static manifest only; wheels deferred to GPU Linux"}
    for name in ("pyproject.toml", "setup.cfg", "setup.py", "requirements.txt"):
        f = snapshot / name
        if f.is_file():
            try:
                text = f.read_text(errors="replace")
            except OSError as e:
                rec["build_files"][name] = {"error": str(e)[:100]}
                continue
            entry = {"bytes": len(text.encode()), "head": text[:1200]}
            if name == "pyproject.toml":
                m = re.search(r"\[build-system\][^\[]*requires\s*=\s*([^\n]+)", text)
                entry["build_requires"] = m.group(1).strip()[:200] if m else None
                m = re.search(r"requires-python\s*=\s*\"([^\"]+)\"", text)
                entry["requires_python"] = m.group(1) if m else None
                m = re.search(r"(?m)^dependencies\s*=\s*\[(.*?)\]", text, re.S)
                entry["dependencies"] = (m.group(1).strip()[:800] if m else None)
            if name == "setup.cfg":
                m = re.search(r"install_requires\s*=\s*([^\[]+?)(?=\n\[|\Z)", text, re.S)
                entry["install_requires"] = (m.group(1).strip()[:800] if m else None)
            if name == "setup.py":
                m = re.search(r"install_requires\s*=\s*\[(.*?)\]", text, re.S)
                entry["install_requires"] = (m.group(1).strip()[:800] if m else None)
            rec["build_files"][name] = entry
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
        if (snapshot / name).is_file():
            rec["license"] = name
            break
    out = HERE / "deps" / f"{iid}.requirements-record.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n")
    return str(out.relative_to(HERE))


def materialize(slot: str, repo: str, iid: str):
    mirror = "swesmith/" + iid.split(".pr_")[0]
    rec = {"task_id": slot, "slot": slot, "repo": repo, "instance_id": iid,
           "mirror": mirror, "base_commit": None, "snapshot": None,
           "snapshot_bytes": None, "snapshot_sha256": None,
           "build_status": None, "substitute_for": None, "standby": False}
    try:
        meta = verify_metadata(iid)
        if meta["resolved_rows"] == 0:
            raise RuntimeError("no resolved rows in shards (metadata)")
        rec["shard_resolved_rows"] = meta["resolved_rows"]
        rec["teacher_models"] = meta["models"]
        commit, _ = ls_remote(mirror, iid)
        rec["base_commit"] = commit
        dest, reused = clone_snapshot(mirror, iid, commit)
        rec["reused_clone"] = reused
        arc = pack_snapshot(iid, dest)
        rec["snapshot"] = arc.name
        rec["snapshot_bytes"] = arc.stat().st_size
        rec["snapshot_sha256"] = sha256_file(arc)
        rec["work_tree_bytes"] = dir_bytes(dest)
        rec["import_check"] = import_check(dest, TOP_PKG[repo])
        rec["collect_check"] = collect_check(dest)
        rec["deps_record"] = deps_record(dest, iid)
        rec["build_status"] = "staged_ok"
    except Exception as e:  # noqa: BLE001 - record, then R1-R4 engine decides
        rec["build_status"] = "failed"
        rec["fail_reason"] = f"{type(e).__name__}: {str(e)[:300]}"
    return rec


def main():
    print(json.dumps({"phase": "metadata_verify_only", "shards": len(SHARDS)}), flush=True)
    for s in SHARDS:
        assert s.is_file(), f"missing shard {s}"
    records = []
    actions = []

    def emit(rec):
        records.append(rec)
        print(json.dumps({k: rec.get(k) for k in
              ("task_id", "instance_id", "build_status", "base_commit",
               "snapshot_bytes", "fail_reason", "substitute_for") if rec.get(k) is not None}),
              flush=True)

    # R3 gate: t01 first (no same-repo substitute exists).
    slot, repo, iid, _, _ = TASKS[0]
    r01 = materialize(slot, repo, iid)
    emit(r01)
    if r01["build_status"] != "staged_ok":
        actions.append("R3: t01 (click pr_2604) materialization failed; no same-repo substitute;"
                       " 12-task freeze INVALID, stop, no out-of-repo addition.")
        for slot2, repo2, iid2, _, _ in TASKS[1:] + [SPARE]:
            records.append({"task_id": slot2, "slot": slot2, "repo": repo2,
                            "instance_id": iid2, "mirror": "swesmith/" + iid2.split(".pr_")[0],
                            "base_commit": None, "snapshot": None, "build_status": "not_attempted_R3_stop"})
        return finish(records, actions, freeze_valid=False,
                      verdict="FREEZE INVALID (R3: click t01 single point failed)")

    s01_used = False

    def use_spare(failed_slot: str):
        nonlocal s01_used
        sslot, srepo, siid, _, _ = SPARE
        s = materialize(sslot, srepo, siid)
        s["substitute_for"] = failed_slot
        s01_used = True
        emit(s)
        return s

    # t02..t12 in slot order.
    for slot, repo, iid, _, _ in TASKS[1:]:
        r = materialize(slot, repo, iid)
        emit(r)
        if r["build_status"] == "staged_ok":
            continue
        if not s01_used:
            actions.append(f"R1/R2: {slot} ({iid}) failed ({r.get('fail_reason', '?')});"
                           " substituting floating spare s01 (pydicom pr_1920).")
            s = use_spare(slot)
            if s["build_status"] == "staged_ok":
                actions.append(f"{slot} filled by s01; quota shifts to 1/3/8 or 1/4/7, still 12 tasks.")
                continue
            actions.append("spare s01 also failed; no further substitute (R2 exhausted).")
        else:
            actions.append(f"{slot} ({iid}) failed and spare already consumed; no substitute left.")
        # R4 path: freeze invalid, stop remaining.
        actions.append("R4: failure without available substitute; FREEZE INVALID, stop.")
        done_slots = {x["slot"] for x in records}
        for slot2, repo2, iid2, _, _ in TASKS[1:] + [SPARE]:
            if slot2 not in done_slots and slot2 != slot:
                records.append({"task_id": slot2, "slot": slot2, "repo": repo2,
                                "instance_id": iid2, "mirror": "swesmith/" + iid2.split(".pr_")[0],
                                "base_commit": None, "snapshot": None,
                                "build_status": "not_attempted_R4_stop"})
        return finish(records, actions, freeze_valid=False,
                      verdict="FREEZE INVALID (R4: unsubstitutable failure)")

    # All 12 staged: materialize s01 as standby spare.
    sslot, srepo, siid, _, _ = SPARE
    s = materialize(sslot, srepo, siid)
    s["standby"] = True
    emit(s)
    if s["build_status"] != "staged_ok":
        actions.append(f"s01 standby spare failed ({s.get('fail_reason', '?')});"
                       " 12-task freeze still VALID but with NO spare (any GPU V2/V3 failure"
                       " will trigger R2-exhausted/R4 path).")

    # R4 whole-repo check: 12 filled slots must cover all 3 repos.
    covered = sorted({r["repo"] for r in records
                      if r["build_status"] == "staged_ok" and not r.get("standby")
                      or (r["build_status"] == "staged_ok" and r.get("substitute_for"))})
    filled = [r for r in records if r["build_status"] == "staged_ok"
              and (not r.get("standby") or r.get("substitute_for"))]
    if len(filled) != 12 or len(covered) != 3:
        actions.append(f"R4: filled={len(filled)} repos={covered}; FREEZE INVALID.")
        return finish(records, actions, freeze_valid=False,
                      verdict="FREEZE INVALID (R4: repo coverage incomplete)")
    actions.append("All 12 slots staged_ok covering 3 repos; freeze VALID pending GPU V2/V3.")
    return finish(records, actions, freeze_valid=True, verdict="FREEZE VALID (CPU stage)")


def finish(records, actions, freeze_valid: bool, verdict: str):
    ok = [r for r in records if r.get("build_status") == "staged_ok" and not r.get("standby")]
    manifest = {
        "produced_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source": "candidate-pool.md 13 issues; SWE-smith shards revision 08e109b4 (P012 precedent)",
        "content_read": False,
        "tests_executed": False,
        "freeze_valid_cpu": freeze_valid,
        "verdict": verdict,
        "staged_ok_12": sum(1 for r in ok if not r.get("substitute_for")) + sum(
            1 for r in records if r.get("substitute_for") and r.get("build_status") == "staged_ok"),
        "substitutions": actions,
        "counts": {s: sum(1 for r in records if r.get("build_status") == s) for s in
                   {r.get("build_status") for r in records}},
    }
    (HERE / "tasks-13.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    (HERE / "materialize-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_report(records, actions, freeze_valid, verdict)
    write_gpu_checklist(records, freeze_valid)
    print(json.dumps({"freeze_valid_cpu": freeze_valid, "verdict": verdict,
                      "ok": manifest["counts"]}, ensure_ascii=False), flush=True)
    return 0 if freeze_valid else 2


def write_report(records, actions, freeze_valid, verdict):
    L = ["# P013 step-2 物化报告（CPU）", "",
         f"日期：{datetime.now(timezone.utc).strftime('%Y-%m-%d')}。 verdict: **{verdict}**。", "",
         "纪律：只读分片标识列（instance_id/resolved/model/traj_id/行号），未读任何问题正文、"
         "参考补丁、test_patch；未读 heldout；未执行 FAIL_TO_PASS / PASS_TO_PASS（GPU V2/V3 阶段）；"
         "一次性手动命令；未建测试文件；snapshots/work 大体积产物不进 git。", "",
         "## 1. 逐题结果", "",
         "| task | repo | instance_id | base_commit | snapshot | 字节 | import | collect | 状态 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r.get("task_id"), r.get("repo"), r.get("instance_id"),
            (r.get("base_commit") or "-")[:12], r.get("snapshot") or "-",
            r.get("snapshot_bytes") if r.get("snapshot_bytes") is not None else "-",
            (r.get("import_check") or {}).get("status", "-"),
            (r.get("collect_check") or {}).get("status", "-"),
            r.get("build_status") + (" (替" + r["substitute_for"] + ")" if r.get("substitute_for") else "")
            + (" [备]" if r.get("standby") else "")))
    L += ["", "## 2. 依赖记录（静态清单，wheel 实取延后 GPU Linux）", ""]
    for r in records:
        if r.get("build_status") != "staged_ok":
            L.append(f"- {r.get('task_id')} {r.get('instance_id')}: 未物化（{r.get('fail_reason') or r.get('build_status')}）。")
            continue
        try:
            rec = json.loads((HERE / r["deps_record"]).read_text())
            files = ", ".join(sorted(rec["build_files"])) or "无构建文件"
            L.append(f"- {r['task_id']} {r['instance_id']}: 构建文件[{files}]；license={rec['license']}；"
                     f"import={r['import_check']['status']}；collect={r['collect_check']['status']}。")
        except (KeyError, OSError) as e:
            L.append(f"- {r.get('task_id')}: deps 记录读取失败 {e}.")
    L += ["", "## 3. 失败与替补动作（R1/R2/R3/R4）", ""]
    L += [f"- {a}" for a in actions] or ["- 无失败，无替补。"]
    L += ["", "## 4. 冻结有效性", "",
          f"- CPU 阶段 verdict: **{verdict}**；freeze_valid_cpu={freeze_valid}。",
          "- GPU V2（修前失败）+ V3（参考通过）均实际跑通后方可进入 24 attempts（V4 门控），"
          "执行清单见 `gpu-prevalidate-checklist.md`。", ""]
    (HERE / "materialize-report.md").write_text("\n".join(L) + "\n")


def write_gpu_checklist(records, freeze_valid):
    L = ["# P013 GPU 预验证作业设计（V2 修前失败 / V3 参考通过）", "",
         "供主智能体推送 GPU 执行。CPU 阶段未执行任何裁判；以下全部待 GPU 实机跑通，log 指针入 `dev12-freeze.json` 后方可 V4 放行。",
         f"前置：CPU 物化 verdict `{ 'VALID' if freeze_valid else 'INVALID' }`；"
         + ("INVALID 则本清单暂停，先走 R3/R4 重冻结。" if not freeze_valid else "逐题执行 V1→V2→V3→V4。"), "",
         "## 0. 通用环境", "",
         "- GPU Linux 镜像，Python（以各题 requires-python 为准），`pip install -e .` 或按 deps 记录安装；",
         "- 官方 SWE-smith loader 按 `instance_id` 物化（不消费教师轨迹）；快照 HEAD 必须等于下表 base_commit；",
         "- 每题作业隔离目录；V2/V3 日志落盘并记录 sha256 指针；判据区分性（A/B 拉开差距）**不是**预验证门槛。", "",
         "## 1. V1 物化（官方 loader 重建 + 构建/许可记录）", "",
         "```bash",
         "# <iid> 逐题：",
         "git clone --branch <instance_id> https://github.com/<mirror>.git task && cd task",
         "git rev-parse HEAD  # 必须等于 base_commit",
         "python -m venv .venv && . .venv/bin/activate && pip install -e .  # 按需加测试 extras",
         "# 记录： snapshot 字节 / 构建系统 / 许可证（许可文件见 deps 记录）",
         "```", "",
         "## 2. V2 修前失败（buggy 环境真实失败）", "",
         "```bash",
         "# 在 buggy 快照上应用当题 criterion test_patch（GPU 阶段编写 tests/test_p013_*.py），然后：",
         "pytest tests/test_p013_<name>.py::<F2P_node> -x -q 2>&1 | tee v2-prefail.log",
         "# 要求： exit != 0 且为断言/行为失败（非收集错误/缺依赖）；log 指针入 dev12-freeze.json",
         "```", "",
         "## 3. V3 参考通过（同环境 F2P 通过 + P2P 通过）", "",
         "```bash",
         "# 应用参考补丁后同环境：",
         "pytest tests/test_p013_<name>.py::<F2P_node> -q 2>&1 | tee v3-postpass.log        # 要求通过",
         "pytest tests/test_p013_<name>.py -q 2>&1 | tee v3-p2p.log                         # 同文件 P2P 通过",
         "# 要求： 两者 exit == 0；log 指针入 dev12-freeze.json",
         "```", "",
         "## 4. V4 门控与换题", "",
         "- V2+V3 均实际跑通 → 该题放行 24 attempts；任一不满足 → 走 R1（同仓下一 rank）/R2（s01 跨仓浮动）；",
         "- R3：click t01 无替补，失败即冻结无效（重走冻结）；R4：整仓不可物化即冻结无效；",
         "- 永远禁止：按模型成败换题；重召 3 个触碰 issue；跳哈希顺序挑题；列 heldout/预留题 ID。", "",
         "## 5. 逐题基线（CPU 物化实测，供 GPU 对照）", "",
         "| task | mirror | base_commit | snapshot 字节 | import | collect | 构建文件 |",
         "|---|---|---|---|---|---|---|"]
    for r in records:
        if r.get("build_status") != "staged_ok" or r.get("standby"):
            continue
        try:
            rec = json.loads((HERE / r["deps_record"]).read_text())
            bf = ",".join(sorted(rec["build_files"])) or "-"
        except (KeyError, OSError):
            bf = "?"
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            r["task_id"], r["mirror"], (r["base_commit"] or "")[:12],
            r.get("snapshot_bytes"), r["import_check"]["status"],
            r["collect_check"]["status"], bf))
    L += ["", "注：import/collect 为 macOS CPU 抽查（PYTHONPATH 直引、collect-only），不代表 GPU Linux 结论；"
          "pydicom 若 CPU import 缺 numpy，属环境缺件记录，GPU 建全量 env 后重测。", ""]
    (HERE / "gpu-prevalidate-checklist.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
