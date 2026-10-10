# Train-mode 6032 save_on_cpu redo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the same 6,032-token two-arm comparison with verified training-mode decoder checkpointing, host RSS/available/cgroup records, and a 0.6 GPU-quota-hour cap; then gate 9,841/12,832, data expansion, and SFT/pairing on measured results.

**Architecture:** Keep the existing DDP-QLoRA recipe. Put the corrected carrier in a **new** evidence directory so historical v1 Kaggle outputs stay untouched. The only comparison variable remains backbone `save_on_cpu`. Wrapper host sampling survives SIGKILL; worker records RSS plus actual decoder checkpoint **calls**, not just flags.

**Tech Stack:** Python 3.13 Kaggle training path, NF4 cache `ccjjllt/p005-cpu-nf4-cache`, 7-wheel `ccjjllt/p005-training-cpu-stage`, Transformers 5.13.1, PEFT LoRA rank8/alpha16, `selected_chunked` CE, `torch.autograd.graph.save_on_cpu(pin_memory=True)`, Kaggle CLI `kernels push`.

**Spec:** `docs/plan/005-save-on-cpu.md` (binding follow-up is §2026-10-11 合并审查与下一步). This plan argues from that section.

## Global Constraints

- 不覆盖 P002 冻结变体、历史结果、heldout、v1/v2 普查文件。
- 不正式提交比赛。不截断历史。不把诊断 adapter 当修复收益。
- 训练路径：Python 3.13 + `ccjjllt/p005-cpu-nf4-cache` + `ccjjllt/p005-training-cpu-stage`。不重量化 62.5GB，不装 41 个推理 wheel，不重跑四步诊断。
- Kernel 用 CLI `kaggle kernels push`。从**当前仓库**重新生成载体，不要沿用已删除 worktree 的绝对路径。
- 本轮第一项 GPU 最多 **0.6 个额度小时**。4×L4 按两倍 wall 计费。启动前重读配额。快照不是当前余额。
- NF4、LoRA rank8/alpha16、标签、注意力实现、CE chunk、同一 6032-token 输入在对照两臂保持固定。只改变是否对 backbone 保存张量启用 `torch.autograd.graph.save_on_cpu(pin_memory=True)`。
- 容量阶段用过的 select 更新须丢弃；正式 SFT 从干净 adapter 初始化。
- 12,832 成功 ≠ 16,384 已验证。不自动连续重试 16K。
- 预留八题及其 repo＋commit 不得进 train/select：`fastapi_13713`、`requests_6592`、`requests_6644`、`requests_7309`、`requests_7505`、`rich_3130`、`rich_3296`、`rich_3469`。
- 实现者自己 commit。
- 不改写 `research/005/save-on-cpu-6032-20261010/kaggle-output/`、旧 `quota-before.json`/`quota-after.json`、旧 `cpu-grad-check.json` 原始数字。
- 不再把开关配置本身当作 checkpoint 实际执行的证据。
- 旧结论已撤回：不得写“Live MLP/dequant 已定位”或“host pin_memory 导致 SIGKILL”除非本次实测证明。
- 研究优先，不要新建门控平台。本地检查只防止再漏训练态/主机内存/错误载体路径。
- Work from this worktree only: `/Users/zhaohuaxiang/ccjjllt/worktrees/agent_of_gemma_4-p005-trainmode-save-on-cpu` on branch `p005-trainmode-save-on-cpu`. Never commit on `main`.
- Local Python for checks: `/tmp/p005-training-check/bin/python`. Do not load 31B locally.

## File map

- Create: `research/005/save-on-cpu-6032-trainmode-20261011/` — corrected carrier, GPU evidence, honest result.
- Create: `research/005/data-expansion-cpu-20261011/` — independent-issue/recovery inventory.
- Copy (untracked jsonl, same SHA): `encoded-capacity-8192.jsonl` sha256 `38cd361b7c1f15d78492963eddc226c17699ae11ec1d0c850071a7d02b585ce5`.
- Leave unchanged: `training/memory.py` default `save_on_cpu=False`; historical `research/005/save-on-cpu-6032-20261010/kaggle-output/`; v1/v2 census files.
- Kernel slug: `ccjjllt/p005-save-on-cpu-6032-tm` (new kernel; do not overwrite historical `p005-save-on-cpu-6032` v1 identity).
- Timeouts: `ARM_TIMEOUT=450`, `WRAPPER_DEADLINE=1000` (session covers both arms), `KAGGLE_TIMEOUT=1080` (0.6h billed at 2×).

---

### Task 1: Corrected carrier with host memory and actual checkpoint calls

**Files:**
- Create: `research/005/save-on-cpu-6032-trainmode-20261011/worker.py`
- Create: `research/005/save-on-cpu-6032-trainmode-20261011/build.py`
- Create: `research/005/save-on-cpu-6032-trainmode-20261011/test_save_on_cpu.py`
- Create: `research/005/save-on-cpu-6032-trainmode-20261011/chat_template.jinja` (copy from `research/005/save-on-cpu-6032-20261010/chat_template.jinja` or `training/configs/chat_template.jinja`; bytes/hash must match the training template already used)
- Create: `research/005/save-on-cpu-6032-trainmode-20261011/encoded/encoded-capacity-8192.jsonl` (copy; do not re-encode)
- Modify: none of `training/memory.py`, none of the historical 20261010 kaggle-output files

**Interfaces:**
- Consumes: existing `prepare_training_mode` contract (outer/base/decoder training + checkpoint flags/functions; fail closed). Same TX `fastapi_11194-3f51fb716edd-tx01`, SEQ 6032, encoded SHA above.
- Produces: `host_memory()` dict; `attach_checkpoint_counters(layers, tracker)` that mutates `tracker["decoder_checkpoint_calls"]`; wrapper monitor samples host meminfo + cgroup events; `build()` writes kernel id `ccjjllt/p005-save-on-cpu-6032-tm` and launch command whose `-p` path is **this worktree**, never `agent_of_gemma_4-p005-save-on-cpu`.

Start from the current `research/005/save-on-cpu-6032-20261010/{worker,build,test_save_on_cpu}.py` (commit `170888c`). Keep two fresh torchrun processes, OOM `os._exit(0)`, adapter discard, no 16K retry, `model.train()` + `prepare_training_mode`.

- [ ] **Step 1: Copy sources into the new evidence directory**

```bash
HERE=research/005/save-on-cpu-6032-trainmode-20261011
OLD=research/005/save-on-cpu-6032-20261010
mkdir -p "$HERE/encoded"
cp "$OLD/worker.py" "$OLD/build.py" "$OLD/test_save_on_cpu.py" "$OLD/chat_template.jinja" "$HERE/"
cp "$OLD/encoded/encoded-capacity-8192.jsonl" "$OLD/encoded/encoded-capacity-8192.jsonl.report.json" "$HERE/encoded/"
```

Do not copy `kaggle-output/`, old `result.json`, old quotas, or `review-20261011.json`.

- [ ] **Step 2: Add `host_memory()` to worker.py**

Place near `cuda_memory`. Never require `psutil`. Missing paths go into `missing`, do not crash the step.

```python
def host_memory():
    info = {
        "rss_bytes": None, "maxrss_bytes": None,
        "mem_available_bytes": None, "mem_total_bytes": None,
        "cgroup_memory_current_bytes": None,
        "cgroup_memory_events": None, "cgroup_memory_events_path": None,
        "missing": [],
    }
    try:
        import resource
        info["maxrss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception as exc:
        info["missing"].append("maxrss:" + repr(exc))
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                info["rss_bytes"] = int(line.split()[1]) * 1024
    except Exception as exc:
        info["missing"].append("VmRSS:" + repr(exc))
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                info["mem_available_bytes"] = int(line.split()[1]) * 1024
            elif line.startswith("MemTotal:"):
                info["mem_total_bytes"] = int(line.split()[1]) * 1024
        if info["mem_available_bytes"] is None:
            info["missing"].append("MemAvailable")
    except Exception as exc:
        info["missing"].append("meminfo:" + repr(exc))
    events_candidates = (
        Path("/sys/fs/cgroup/memory.events"),
        Path("/sys/fs/cgroup/memory/memory.events"),
    )
    current_candidates = (
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    )
    found_events = False
    for path in events_candidates:
        try:
            if path.is_file():
                parsed = {}
                for line in path.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1].isdigit():
                        parsed[parts[0]] = int(parts[1])
                info["cgroup_memory_events"] = parsed or path.read_text()
                info["cgroup_memory_events_path"] = str(path)
                found_events = True
                break
        except Exception as exc:
            info["missing"].append("cgroup_events:" + str(path) + ":" + repr(exc))
    if not found_events:
        info["missing"].append("cgroup_memory_events")
    for path in current_candidates:
        try:
            if path.is_file():
                info["cgroup_memory_current_bytes"] = int(path.read_text().strip())
                break
        except Exception as exc:
            info["missing"].append("cgroup_current:" + str(path) + ":" + repr(exc))
    return info
```

Call `host_memory()` from `emit`, from `begin`/`end` phase records, and on failure. Rank reports must contain a `host_memory` object even when OOM/`os._exit`.

- [ ] **Step 3: Count actual decoder checkpoint calls during forward**

Keep `prepare_training_mode` fail-closed on flags. Add runtime counters. Flags alone are not evidence.

```python
def attach_checkpoint_counters(layers, tracker):
    originals = []
    tracker["decoder_checkpoint_calls"] = [0] * len(layers)

    def wrap(index, function):
        def counted(*args, **kwargs):
            tracker["decoder_checkpoint_calls"][index] += 1
            return function(*args, **kwargs)
        return counted

    for index, layer in enumerate(layers):
        function = getattr(layer, "_gradient_checkpointing_func", None)
        originals.append((layer, function))
        if callable(function):
            layer._gradient_checkpointing_func = wrap(index, function)
    return originals


def restore_checkpoint_functions(originals):
    for layer, function in originals:
        if function is not None:
            layer._gradient_checkpointing_func = function
```

In `run_one_step`, attach counters with the layer hooks. After a **completed** forward (`end("forward", ...)`), if `sum(tracker["decoder_checkpoint_calls"]) == 0`, raise `RuntimeError("decoder checkpointing did not execute during forward")`. On OOM/failure, still write `decoder_checkpoint_calls` into the rank report. Include the list in the forward phase payload and in `emit`.

SCHEMA constant: `"p005-save-on-cpu-6032-tm"`.

- [ ] **Step 4: Wrapper host sampling and new kernel identity**

In `build.py`:

- `KERNEL_ID = "ccjjllt/p005-save-on-cpu-6032-tm"`
- Keep `KAGGLE_TIMEOUT = 1080`, `ARM_TIMEOUT = 450`, `WRAPPER_DEADLINE = 1000`
- Wrapper `OUT = Path('/kaggle/working/save-on-cpu-6032-tm')`
- Wrapper schema `p005-save-on-cpu-wrapper-v2`
- `build()` must not launch Kaggle
- Launch plan `-p` must be the destination path actually passed to `build()`, which on the real rebuild is this worktree's `research/005/save-on-cpu-6032-trainmode-20261011/kernel`. The JSON must not contain `agent_of_gemma_4-p005-save-on-cpu` (deleted worktree).
- Title: `P005 save_on_cpu 6032 train-mode`
- Same kernel_sources, private, GPU, no internet, no model_sources

Extend the wrapper `monitor()` sample (same 15s cadence) with host fields:

```python
sample['host'] = {
    'meminfo': Path('/proc/meminfo').read_text() if Path('/proc/meminfo').is_file() else None,
    'cgroup_memory_events': None,
    'cgroup_memory_current': None,
}
for candidate in ('/sys/fs/cgroup/memory.events', '/sys/fs/cgroup/memory/memory.events'):
    p = Path(candidate)
    if p.is_file():
        sample['host']['cgroup_memory_events'] = p.read_text()
        sample['host']['cgroup_memory_events_path'] = candidate
        break
else:
    sample['host']['cgroup_memory_events_missing'] = True
for candidate in ('/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory/memory.usage_in_bytes'):
    p = Path(candidate)
    if p.is_file():
        sample['host']['cgroup_memory_current'] = p.read_text().strip()
        break
```

If a path is unreadable, set the field to missing/true rather than crashing the monitor. This is the SIGKILL survival path.

Keep two-arm loop, `kill_process_group` after each arm, remaining-time session deadline, adapter discard. Do not treat skipped second arm as success.

- [ ] **Step 5: Local checks (not a new platform)**

Update `test_save_on_cpu.py` for the new directory and slug. Keep the existing 6032-row, train-mode, two-arm, no-launch checks. Add:

1. `host_memory()` returns a dict with the keys above; on macOS `/proc` is absent so `missing` is non-empty and the call does not raise.
2. Source of `worker.py` contains `decoder checkpointing did not execute during forward` and `attach_checkpoint_counters`.
3. Built `probe.py` contains `cgroup_memory_events` and `WRAPPER_DEADLINE = 1000` and kernel id `ccjjllt/p005-save-on-cpu-6032-tm`.
4. Launch plan timeout is `1080`, path is the build destination, and `agent_of_gemma_4-p005-save-on-cpu` is absent.
5. Existing tiny-Gemma test still proves eval backbone has 0 checkpoint calls and `prepare_training_mode` then makes calls > 0.

Run:

```bash
cd /Users/zhaohuaxiang/ccjjllt/worktrees/agent_of_gemma_4-p005-trainmode-save-on-cpu
/tmp/p005-training-check/bin/python -m unittest research.005.save-on-cpu-6032-trainmode-20261011.test_save_on_cpu
/tmp/p005-training-check/bin/python -m unittest training.tests.test_memory
```

If the research path is not a package, run by file path instead:

```bash
/tmp/p005-training-check/bin/python research/005/save-on-cpu-6032-trainmode-20261011/test_save_on_cpu.py
```

Do not rewrite historical `cpu-grad-check.json`. `training/memory.py` default remains `save_on_cpu=False`.

- [ ] **Step 6: Commit**

```bash
git add docs/plan/005-trainmode-save-on-cpu.md \
  research/005/save-on-cpu-6032-trainmode-20261011/worker.py \
  research/005/save-on-cpu-6032-trainmode-20261011/build.py \
  research/005/save-on-cpu-6032-trainmode-20261011/test_save_on_cpu.py \
  research/005/save-on-cpu-6032-trainmode-20261011/chat_template.jinja
# encoded jsonl is gitignored; do not force-add it
git commit -m "$(cat <<'EOF'
feat(005): record host memory and live decoder checkpoint calls

Redo the 6032-token carrier in a new evidence directory with training-mode
checkpoint-call counts and RSS/cgroup sampling. Default save_on_cpu stays off.
EOF
)"
```

---

### Task 2: CPU independent-issue and recovery source expansion

**Files:**
- Create: `research/005/data-expansion-cpu-20261011/scan.py`
- Create: `research/005/data-expansion-cpu-20261011/test_scan.py`
- Create: `research/005/data-expansion-cpu-20261011/note.md`
- Create (generated, commit JSON reports if small): `trace_inventory.json`, `window_report.json` or equivalent
- Do not modify: `research/005/length-census-v2-20261010/*`, `training/artifacts/repair-sft-v1/`, `training/artifacts/repair-sft-v2/` census reports, heldout content

**Interfaces:**
- Consumes: `training/data/{length_census,prepare_local,sample_behaviors}.py`; `PAIRING_RESERVED_ISSUES`; historical v2 issue set from `research/005/length-census-v2-20261010/length_report.json`; previous inventory `research/005/data-expansion-cpu-20261010/trace_inventory.json` (92 traces, 9 eligible issues).
- Produces: a new-directory report stating whether any **new independent non-reserved** issue or recovery trace appeared, with source path and length. Growth is versus the historical v2 issue set, not versus this run's own prepare list.

No GPU. Do not invent issues. Do not treat truncated prefixes as original complete samples. Do not launch new rollouts in this task (that would spend the 0.6h 6032 budget).

- [ ] **Step 1: Inventory every local `trace_*.json` under `results/`**

Reuse the reserved-snapshot filter and `new_issue_ids(..., frozen_v2)` logic from `research/005/data-expansion-cpu-20261010/scan.py` (copy into the new directory, do not overwrite the 20261010 reports). Walk `results/` (worktree `results` is a symlink to the main checkout). Record path, issue id, repo, commit, sha256, whether pairing-reserved or same repo+commit, whether already in the 92-file inventory.

- [ ] **Step 2: Classify new vs already-counted**

A file is new only if its sha256 is not in the 20261010 inventory. A new **independent issue** is an eligible, non-heldout, non-reserved issue id (and not the same repo+commit as a reserved pairing task) that is absent from the historical v2 issue set. Protocol must remain `p005-harness-tools-v1`. If a new file is only another prefix of an already-counted issue, say so; do not call it an independent issue.

- [ ] **Step 3: Length/source stats for any new usable traces**

If new eligible traces exist, run the existing prepare/length path on them only, `--on-oversize skip`, no truncation. Tokenizer sha `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` (32169626 bytes) if encoding. If none exist, do not re-prepare the old 214 prefixes just to reprint the same table; write that independent issues cannot grow without new rollouts, and list eligible non-reserved issue ids in splits that still have **no** unused protocol-compatible trace.

- [ ] **Step 4: Checks**

Port the reserved-snapshot and historical-v2 growth tests into `test_scan.py` pointed at the new scanner. Assert v2 census sha256 unchanged:

- `research/005/length-census-v2-20261010/length_report.json` `2a219102b6779f5ea9aa37e696c1089e67cc0283b77c5fe23f73720c246eef76`
- `sample_list.json` `05cbf49bcd7b67cf8fc8d98937a35031d6192b183a3ac2dcdc8adcd138ee79c1`
- `train_issues.json` `b25fe2c97599d4d15d5e0c32c69b1ef6af4a7f6a95a662109b446e865a9159b1`

Run:

```bash
/tmp/p005-training-check/bin/python research/005/data-expansion-cpu-20261011/scan.py
/tmp/p005-training-check/bin/python research/005/data-expansion-cpu-20261011/test_scan.py
```

- [ ] **Step 5: Commit**

```bash
git add research/005/data-expansion-cpu-20261011
git commit -m "$(cat <<'EOF'
docs(005): inventory independent traces without new rollouts

Record whether any non-reserved protocol-compatible repair or recovery
trace appeared beyond the v2 issue set. Do not overwrite the v2 census.
EOF
)"
```

---

### Task 3: Rebuild from this worktree, re-read quota, run the 6032 GPU comparison

**Files:**
- Create (generated): `research/005/save-on-cpu-6032-trainmode-20261011/kernel/{probe.py,kernel-metadata.json,bundle-manifest.json,launch-plan.json}`
- Create: `research/005/save-on-cpu-6032-trainmode-20261011/quota-before.json`
- Create: `research/005/save-on-cpu-6032-trainmode-20261011/quota-after.json`
- Create: `research/005/save-on-cpu-6032-trainmode-20261011/kaggle-output/` (new; do not write into the 20261010 kaggle-output)
- Modify: none of the historical 20261010 kernel outputs

**Interfaces:**
- Consumes: Task 1 `build.py` / `worker.py` in this worktree; encoded jsonl SHA `38cd361b7c1f15d78492963eddc226c17699ae11ec1d0c850071a7d02b585ce5`.
- Produces: a pushed private kernel `ccjjllt/p005-save-on-cpu-6032-tm`, downloaded rank reports, wrapper-report, gpu-telemetry (with host samples), quota before/after.

Building NEVER launches Kaggle. Push is a separate step after quota.

- [ ] **Step 1: Rebuild the carrier from the current worktree**

```bash
cd /Users/zhaohuaxiang/ccjjllt/worktrees/agent_of_gemma_4-p005-trainmode-save-on-cpu
/tmp/p005-training-check/bin/python research/005/save-on-cpu-6032-trainmode-20261011/build.py
```

Confirm `launch-plan.json` `command_for_operator_only` `-p` is this worktree kernel directory, timeout `1080`, `launch_performed` false, and the deleted worktree path is absent. Confirm bundled `worker.py` hash matches the file just committed. Confirm encoded SHA. Confirm `kernel-metadata.json` id `ccjjllt/p005-save-on-cpu-6032-tm`, private, GPU, no internet, kernel_sources `ccjjllt/p005-cpu-nf4-cache` and `ccjjllt/p005-training-cpu-stage`.

- [ ] **Step 2: Re-read accelerator quota**

Use MCP `kaggle__get_accelerator_quota` with `{"request": {}}`. Write `quota-before.json` with `time_used`, `total_time_allowed`, captured UTC time, `this_run_wall_timeout_s=1080`, `l4_billing_multiplier=2`, `this_run_estimated_billed_hours=0.6`, remaining hours, and `push_allowed`.

Stop without pushing if remaining GPU seconds < 2160 + 600 margin, or if another GPU job is already running on this account.

- [ ] **Step 3: Push with CLI, then watch to a terminal state**

```bash
kaggle kernels push -p research/005/save-on-cpu-6032-trainmode-20261011/kernel \
  --accelerator NvidiaL4 --timeout 1080
```

Do not use MCP `save_notebook`. Do not pass competition submit. Kernel must stay private.

Watch with `kaggle kernels status ccjjllt/p005-save-on-cpu-6032-tm` (and/or MCP `kaggle__get_notebook_session_status`) until COMPLETE or ERROR or a hard timeout well past 1080s wall. 4×L4 billed time is 2× wall; the session timeout in the wrapper is the two-arm cap.

If the kernel is queued more than ~20 minutes with no start, record that and do not start a second GPU job.

- [ ] **Step 4: Download outputs into the new evidence directory**

```bash
kaggle kernels output ccjjllt/p005-save-on-cpu-6032-tm \
  -p research/005/save-on-cpu-6032-trainmode-20261011/kaggle-output
```

Need at least wrapper-report, both arms' rank-000 reports if they exist, worker logs, gpu-telemetry.jsonl, source-manifest. If SIGKILL left an arm without rank JSON, keep the telemetry/host samples and say the rank report is missing.

Re-read quota into `quota-after.json`. Compute delta seconds/hours. Do not edit 20261010 quota files.

- [ ] **Step 5: Commit process files that belong in git**

Commit `build.py` outputs that are source (kernel-metadata, launch-plan, bundle-manifest) plus quota JSON. Do not git-add large logs if gitignored (`*.log`). Rank JSON reports in `kaggle-output/` may be committed if they are small JSON; do not force-add zip/safetensors.

```bash
git add research/005/save-on-cpu-6032-trainmode-20261011
git commit -m "$(cat <<'EOF'
feat(005): run train-mode 6032 save_on_cpu comparison

Rebuild the carrier from this worktree and push a new private 4xL4 kernel
capped at 0.6 quota hours. Historical v1 outputs are left unchanged.
EOF
)"
```

If push was blocked by quota, commit the quota-before file and a `blocked.json` instead, status BLOCKED, `gpu_launched=false`.

---

### Task 4: Honest result, then gate 9841/12832 and SFT/pairing

**Files:**
- Create: `research/005/save-on-cpu-6032-trainmode-20261011/result.json`
- Create: `research/005/save-on-cpu-6032-trainmode-20261011/results.md`
- Create if gated off: `research/005/save-on-cpu-6032-trainmode-20261011/blocked-longseq.json`
- Create if no SFT: `research/005/sft-candidate-20261011/blocked.json` and/or pairing skip under `research/005/base-adapter-pair-trainmode-20261011/skipped.json`
- Modify: `docs/plan/005-save-on-cpu.md` (append a short next-step note only; do not rewrite v1 observations)
- Modify: `docs/progress/005-medal-and-prize-research.md` (new latest section; keep historical sections)

**Interfaces:**
- Consumes: Task 3 kaggle-output + quotas; Task 2 inventory (`new_independent_issues`, recovery events).
- Produces: conclusions that only restate measured fields. Gate: offload arm completed first Adam (`optimizer_reached=true` and `adam_state.tensors>0`) **and** decoder checkpoint calls > 0.

- [ ] **Step 1: Write the 6032 result from files, not from hope**

`result.json` schema `p005-save-on-cpu-6032-tm-result`. Required fields:

- `comparison_valid_for_checkpointed_training`: true only if **both** arms recorded `training_mode` with `decoder_checkpointing_active` all true **and** at least one rank shows `decoder_checkpoint_calls` summing to > 0 (or, if forward died before any layer, record calls and set this false with reason).
- Per arm: save_on_cpu flag, load GiB/seconds, forward/backward/optimizer reached, last decoder layer, traceback (or missing), GPU peaks, worker RSS / host MemAvailable / cgroup events (or `missing`), decoder_checkpoint_calls, optimizer_reached, adam_state.
- `first_adam_step_completed` per arm.
- `did_not_truncate`, `did_not_retry_16k`, `adapter_discarded`, `loss_is_not_repair_gain`, `competition_submit=false`.
- Quota delta.
- Do not use the 3.62 GiB activation estimate as a claim.
- Do not copy v1 “gate_proj is the root cause” language unless this run’s traceback with **active** checkpointing shows it.

`results.md` is the human summary of the same numbers.

- [ ] **Step 2: Gate 9841 / 12832**

If the corrected **save_on_cpu** arm completed the first Adam step:

1. Re-read quota. Additional GPU must still be justified; do not blow past a further 0.6h without writing the new estimate. Prefer a **new** kernel slug such as `ccjjllt/p005-capacity-9841-tm`. One complete train transaction at 9841 tokens, then 12832, each one Adam step, discard adapters. 12832 success does not claim 16384. Do not retry 16K.
2. Record peaks and whether Adam state was created.

If the offload arm did **not** complete the first Adam step: write `blocked-longseq.json` with `status=BLOCKED`, `gpu_launched=false` for 9841/12832, `did_not_push_9841=true`, `did_not_push_12832=true`, `did_not_retry_16k=true`, `did_not_rewrite_fsdp=true`. Point `traceback_class` at the **actual** failure stage from this run:

- host memory growth / cgroup oom_kill / MemAvailable collapse → next cut is saved-tensor volume, not FSDP
- live operator peak with checkpoint calls > 0 → next cut is that operator
- attention path → actual kernel/mask/KV
- checkpoint calls == 0 → the comparison is still invalid; do not claim offload failed

Do not start FSDP-QLoRA in this task unless the evidence paragraph above names it.

- [ ] **Step 3: Gate real SFT and eight-task T0 pairing**

Run SFT + pairing only if (a) at least one complete **train** length backprop succeeded in Step 2, **and** (b) Task 2 produced independent non-reserved trainable samples. Otherwise write skip records. Diagnostic adapter is not a repair-gain arm. Reserved eight stay out. No competition submit.

If skipping pairing, copy the skip shape from `research/005/base-adapter-pair-save-on-cpu-20261010/skipped.json` with updated reasons and paths.

- [ ] **Step 4: Update plan/progress**

Add a new dated section at the top of `docs/progress/005-medal-and-prize-research.md`. Append a short “本次实测” note to `docs/plan/005-save-on-cpu.md` without deleting the 2026-10-11 review. Next-step language must match the measured gate, not a pre-commitment that offload saves N GiB.

- [ ] **Step 5: Commit**

```bash
git add research/005/save-on-cpu-6032-trainmode-20261011/result.json \
  research/005/save-on-cpu-6032-trainmode-20261011/results.md \
  docs/plan/005-save-on-cpu.md docs/progress/005-medal-and-prize-research.md
# plus blocked/skip/SFT files actually written
git commit -m "$(cat <<'EOF'
docs(005): record train-mode 6032 comparison and gated follow-up

State only measured peaks, checkpoint-call counts, host memory, and whether
the first Adam step completed. Long-seq, SFT, and pairing follow the gate.
EOF
)"
```

---

## Spec coverage (controller self-check)

| Spec requirement | Task |
|---|---|
| Same 6032-token tx, NF4/LoRA/loss/attn fixed, only save_on_cpu varies | 1, 3 |
| Both arms `model.train()` + decoder checkpoint actually runs; flags ≠ evidence | 1, 4 |
| Fresh DDP worker per arm; session timeout covers both arms | 1, 3 |
| RSS, host available, cgroup memory.events or missing | 1, 3, 4 |
| ≤0.6h GPU, re-read quota, 4×L4 bills 2×, CLI push from current repo | 3 |
| New evidence dir; do not rewrite v1 kaggle-output | 1, 3, 4 |
| 9841/12832 only if offload completed first Adam; 12832 ≠ 16384; no auto 16K | 4 |
| Next cut from actual failure stage; withdrawn v1 root-cause claims | 4 |
| Expand independent non-reserved traces on CPU; prefixes ≠ new issues | 2 |
| SFT + eight-task T0 pairing only with real adapter + independent samples | 4 |
| Reserved eight and same repo+commit out; no competition submit | all |
