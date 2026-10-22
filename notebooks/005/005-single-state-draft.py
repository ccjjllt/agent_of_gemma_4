# DRAFT-005-single-state — DRAFT草稿，未冻结，不可直接推送（no push, no model run）
"""P005 single_state 联合载体草稿（本地 DRAFT，未冻结）。

状态：DRAFT 草稿，仅供本地评审；不可直接推送 Kaggle，不运行模型。
推送前必须完成 TODO v30 pins（指纹/BUNDLE 哈希/调度/manifest/配额重查），
并按 handoff §5 预注册改任务名单。详见同目录 README。

借鉴（只借组织，不抄旧指纹）：
- notebooks/002-joint-explore-single-cell.py 的 input_snapshot / resolve_input /
  双布局 fail-fast、单次模型加载、逐题 attempt 目录、独立归档、服务日志归档后停止。
- docs/plan/005-implementation-handoff.md §2（资产表+建议文件组织）、§3 I1-I4、
  §5（复用联合载体思想：单次启动、逐题重置、小探索先行）。
- research/005/candidates/single_state/agent 与 source_context 现有实现
  （以其 SKILL/脚本为准；本文件只内嵌四文件全文，三技能包只嵌清单+SHA）。

本载体内嵌：
- single_state agent 四文件全文：agent.yaml / prompts/system.md /
  configs/sampling.yaml / eval_config.yaml（前三份逐字拷贝自候选目录，
  eval_config 为 DRAFT 占位，复用 002 冻结预算，待 v30 重 pin）。
- working_state / source_probe / source_edit 三技能包：只嵌文件清单+SHA256，
  完整脚本不 whole-copy（大段注明），执行前从候选目录真实拷贝并校验。
"""
from pathlib import Path
import asyncio
import concurrent.futures
import glob
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

IS_DRAFT = True
DRAFT_NOTE = (
    'DRAFT-005-single-state: draft only, NOT FROZEN, DO NOT PUSH. '
    'Fill TODO v30 pins + pre-register task list per handoff S5 before any push.'
)
VARIANT = 'single_state'
REPEAT_ID = 'r1'
RUN_ID = 'run-005-single-state-DRAFT-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


def input_snapshot(limit=120):
    # 借鉴 002 组织：区分"未挂载 inputs"与"路径错误"，fail-fast 窗口内诊断。
    # 注意：本函数只列文件名快照，不抄 002 任何指纹/哈希值。
    root = Path('/kaggle/input')
    try:
        if not root.is_dir():
            return ['/kaggle/input missing']
        files = sorted(str(path.relative_to(root)) for path in root.rglob('*') if path.is_file())
        return files[:limit] + ([f'...(+{len(files) - limit} more)'] if len(files) > limit else [])
    except Exception as exc:
        return [f'list-failed: {exc}']


INPUT_ROOT = Path('/kaggle/input')


def resolve_input(candidates, marker, marker_substr=None):
    # 借鉴 002 组织：双布局兼容（nested datasets/.../competitions/.../models/... 或 flat <slug>/）。
    # 先试已知候选路径，否则按唯一 marker 文件扫描；找不到则 fail-fast，不静默替代。
    for cand in candidates:
        if cand.exists():
            print(f'input resolved: {cand}', flush=True)
            return cand
    found = []
    if INPUT_ROOT.is_dir():
        for path in INPUT_ROOT.rglob(marker):
            if marker_substr is None or marker_substr in str(path):
                found.append(path.parent)
    uniq = sorted(set(found))
    assert len(uniq) == 1, (f'Input {marker} not found: '
                            f'candidates={[str(c) for c in candidates]} scan={uniq[:5]} '
                            f'input_files={input_snapshot()}')
    print(f'input resolved by marker {marker}: {uniq[0]}', flush=True)
    return uniq[0]


# ---- 内嵌 single_state agent 四文件（全文） ----
# agent.yaml / prompts/system.md / configs/sampling.yaml 三份逐字拷贝自
# research/005/candidates/single_state/agent/（2026-10-10 现状，以其为准）；
# eval_config.yaml 为 DRAFT 占位（复用 002 冻结预算数值，待 v30 重 pin，见 TODO）。
BUNDLE_SINGLE_STATE = {
    'agent.yaml': (
        'name: p005_single_state\n'
        'model: gemma-4-31b-it-qat-w4a16-ct\n'
        'instruction: !include prompts/system.md\n'
        'tools:\n'
        '  - run_command\n'
        '  - read_file\n'
        '  - edit_file\n'
        '  - write_file\n'
        '  - get_status\n'
        '  - submit_patch\n'
        'skills:\n'
        '  - skills/source_edit\n'
        '  - skills/working_state\n'
        '  - skills/source_probe\n'
        'generate_content_config: !include configs/sampling.yaml\n'
    ),
    'prompts/system.md': (
        'You are a single repository-fix agent. Resolve the injected issue with minimal, evidence-backed edits, then submit.\n'
        '\n'
        'Initial problem is injected by the harness. Reference `patch`/`test_patch` never enter your prompt. '
        'If session state contains non-empty `hints`, treat them as auxiliary context only; '
        'if absent, proceed without them. Never fail on a missing hints variable.\n'
        '\n'
        'Only call tools actually registered for this run. `run_skill_script` is auto-provided when `skills:` is declared, '
        'never added to `tools:`. Check the real tool schema before calling.\n'
        '\n'
        '## Contract and working state\n'
        '\n'
        'Record the issue contract first (`working_state record`): trigger (inputs/config), expected (output/exception/state), '
        'preserve (adjacent behavior that must not change). Missing info stays `unknown`; never invent expectations. '
        'Keep at most two competing hypotheses with layer (`producer|transformer|consumer|boundary`), '
        'supporting facts (span/tool refs) and contradictions; one clear explanation means proceed. '
        'Update the state when preparing an edit or when evidence overturns an assumption; '
        '`show` surfaces `changed_fields` plus the current `next_action`, which must answer the discrepancy.\n'
        '\n'
        '## Read to decide\n'
        '\n'
        'Each `source_probe` call states one `question` it answers. `locate` (1-4 queries) and `inspect` (1-3 targets) '
        'yield `textual_reference` only, never proven calls. `exact` gives complete functions; '
        'truncated output carries no editable handle and never seeds edits. '
        '`probe` runs one command for trigger/path evidence with named `--assert` checks; '
        'exit 0 alone never means reproduced or fixed. Missing fixtures/imports are `setup_error`, not success; '
        'keep reasoning from source.\n'
        '\n'
        '## Edit with custody\n'
        '\n'
        'Hold a `source_edit inspect` handle (complete display only), then `apply_span(span_id, replacement)`; '
        'you write only the replacement. On `stale_source`, re-inspect narrowed and retry once. '
        'Complete source, raw tool output, and diffs are kept by tools; your "done" text is not success evidence. '
        'Snapshot (at most 4, explicit `parent_patch_id`) before risky rewrites; '
        '`restore` one named id explicitly when an assumption falls; nothing auto-picks a best patch.\n'
        '\n'
        '## Check and submit\n'
        '\n'
        'Run `check_patch`-equivalent hygiene via `run_command` (`git diff --check`, changed-file list) '
        'and only the targeted behavior check; never full-repo sweeps. Use `get_status()` at stage transitions; '
        'it costs no tool calls. Reserve time to close out: `submit_patch()` once, then a short summary '
        '(changed files, verification command and result, stop reason). '
        'Keep temporary scripts in `/tmp`, out of the final diff; never touch tests.\n'
    ),
    'configs/sampling.yaml': (
        '# P005 sampling: same 16384 output ceiling as the P002 frozen base, so the\n'
        '# state/handoff increment is isolated (same isolation logic as the T2 panel).\n'
        '# New thinking semantics (verified in adk_submission 0.2.13\n'
        '# resolvers/generation.py:359-453): a POSITIVE thinking_budget is forwarded as\n'
        '# extra_body.thinking_token_budget (real token budget, not just an on/off\n'
        '# switch); 0 (or level none) disables thinking; include_thoughts:false strips\n'
        '# thought parts from history AND outgoing reasoning -- closing thinking and\n'
        '# not retaining thoughts are different interventions, study them separately.\n'
        '# 1024/4096 below are experiment STARTING points under the new semantics, not\n'
        '# known optima. Output ceiling stays 16384 to separate the budget variable.\n'
        'temperature: 0.2\n'
        'top_p: 0.95\n'
        'max_output_tokens: 16384\n'
        'thinking_config:\n'
        '  thinking_budget: 4096\n'
        '  include_thoughts: true\n'
    ),
    # TODO v30 pins: eval_config 占位，复用 002 冻结预算数值；推送前按实际 runner/官方包重 pin。
    'eval_config.yaml': (
        '# P005 single_state DRAFT eval budgets — TODO v30 pins: placeholder reuses 002 frozen budgets.\n'
        '# Do NOT treat as pinned until the pre-push checklist re-pins timeout/tools/time/turns\n'
        '# against the actual v30 runner + official packages.\n'
        'evaluation:\n'
        '  timeout_seconds: 60\n'
        '  max_tool_calls: 50\n'
        '  max_time_minutes: 5\n'
        '  max_turns: 80\n'
    ),
}

# ---- 三技能包：只嵌清单+SHA256，不 whole-copy 完整脚本（大段注明） ----
# 说明：完整脚本执行前从 research/005/candidates/single_state/agent/skills/*/scripts/
# 真实拷贝进打包目录并按以下 SHA 校验；本 DRAFT 文件不内嵌脚本全文，不引用仓库外路径、
# 不用 symlink（交接 §2 打包规则）。SHA 以候选目录现状为准（2026-10-10）。
SKILL_INVENTORY = {
    'skills/source_edit': {
        'SKILL.md': '2888df0efbeb67ac67ef1645a2833b0cb4b5155cacc20790c68e6c240633d576',
        'scripts/source_edit.py': '466847b9c964465396fb8a55ceac09abcfb0aaed4371f9813b2569e4c8b99bd1',
        'scripts/span_store.py': '01011ed554086b4ab3c34c4c5bff37145371d80f1be18b7a1a652ee824f913d1',
        'scripts/patch_store.py': '250acb724fd71948abbb98a9cda7e9749d43df6b3f9b5bab031182be245665b1',
        'note': 'vendored T2 + patch_store 快照扩展；脚本全文不内嵌，打包时真实拷贝+SHA 校验',
    },
    'skills/working_state': {
        'SKILL.md': '1295c7cd1acc1bdafa5830a84497e966d70cf2b56521b41d120a888783efd652',
        'scripts/working_state.py': 'e7723f2ee4aa5839a90dcf3bcf86304ffb986443fbdcf4f578f11f83d99285fb',
        'note': 'record/show + handoff 往返；脚本全文不内嵌，打包时真实拷贝+SHA 校验',
    },
    'skills/source_probe': {
        'SKILL.md': '7a8ae1930d35da87f7fd904ef84b94d0d9551113e5ab354e1ad95e4f25217669',
        'scripts/source_probe.py': 'e6a3862b2bef9f63da1854f7d6952356a791bd45312819febec8ae19eb0798b9',
        'note': 'locate/inspect/probe；脚本全文不内嵌，打包时真实拷贝+SHA 校验',
    },
}
CORE_FILE_SHA_DRAFT = {
    # TODO v30 pins: 以下为候选目录现状 SHA（草稿记录值），推送前重算并填入冻结哈希表。
    'agent.yaml': '0f6c5f6f2b6adc5b4516d9ad20d3e866a2fdb8f82cc8703ad72523bbc2a450f9',
    'prompts/system.md': 'a2f9a4b7f490ac83ef52c6e63214f5670441ab7655d2e76cc2bac55325db73de',
    'configs/sampling.yaml': '24477150474b5cd59418ff6db342d04934ffcb6a58980a0661e9d7c08fb4838b',
    'eval_config.yaml': 'TODO-v30-pins',
}

# TODO v30 pins：指纹校验位留空，待 v30 值填入。推送前必填，否则 fail-fast 停止。
TODO_V30_PINS = {
    'model_revision': None,  # TODO v30 pins: e.g. 'gemma-4-31b-it-qat-w4a16-ct/<N>' + tokenizer 指纹
    'official_packages': {},  # TODO v30 pins: {dist: 'version sha256 ...'}，执行前重查
    'tasks_sha256': None,  # TODO v30 pins: tasks.jsonl 精确哈希
    'tasks_bytes': None,  # TODO v30 pins: tasks.jsonl 精确字节数
    'bundle_sha256': {},  # TODO v30 pins: 四文件冻结哈希（替换 CORE_FILE_SHA_DRAFT 草稿值）
    'skill_sha256': {},  # TODO v30 pins: 三技能包冻结哈希（替换 SKILL_INVENTORY 草稿值）
}

# ---- 任务名单：与 002 相同的 6 题占位；待执行前按 §5 预注册改名单 ----
# 注意：此名单为占位，未预注册；推送前必须按 handoff §5 改为正式名单并冻结。
# 6 题与 002 探索集相同（dev 池，非 heldout；不含参考答案，不硬编码 heldout）。
PLACEHOLDER_TASKS = [
    {'id': 'rich_3942', 'reason': 'context-fault case (placeholder, same as 002)'},
    {'id': 'rich_3130', 'reason': 'late-edit failure case (placeholder, same as 002)'},
    {'id': 'fastapi_14873', 'reason': 'tool-loop case (placeholder, same as 002)'},
    {'id': 'fastapi_5077', 'reason': 'fastapi eligible (placeholder, same as 002)'},
    {'id': 'requests_6644', 'reason': 'stable success + requests (placeholder, same as 002)'},
    {'id': 'rich_3518', 'reason': 'stable success control (placeholder, same as 002)'},
]
PLACEHOLDER_NOTE = '待执行前按§5预注册改名单：正式名单+顺序需预注册并冻结，本占位不作数。'

WORKING_DIR = Path('/kaggle/working')
RUN_ROOT = WORKING_DIR / 'results' / '005-single-state' / RUN_ID
AGENT_ROOT = WORKING_DIR / 'agent_exec' / RUN_ID


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    with tmp.open('w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def build_schedule(tasks):
    # 单臂 single_state：每题一次 attempt，共 6 次（占位；正式调度待预注册）。
    schedule = []
    for ordinal, item in enumerate(tasks, 1):
        attempt_id = f'{ordinal:02d}-{item["id"]}-{VARIANT}'
        schedule.append({
            'ordinal': ordinal, 'task_id': item['id'], 'variant': VARIANT,
            'attempt_id': attempt_id,
            'results_dir': str(RUN_ROOT / VARIANT / REPEAT_ID / 'attempts' / attempt_id),
        })
    return schedule


SCHEDULE_DRAFT = build_schedule(PLACEHOLDER_TASKS)

EXPERIMENT_DRAFT = {
    'study': 'P005 single_state (DRAFT, NOT FROZEN, DO NOT PUSH)',
    'variant': VARIANT,
    'single_variable': 'state+handoff increment vs P002 frozen base (working_state + probe + diff snapshots)',
    'sampling_note': '16384 ceiling + thinking_budget 4096 starting point (new forwarding semantics)',
    'placeholder_note': PLACEHOLDER_NOTE,
    'todo_v30_pins': sorted(TODO_V30_PINS.keys()),
}

# ---- DRAFT 护栏：任何模型加载/安装/评测之前 fail-fast；通过 py_compile 但不可执行 ----
if IS_DRAFT:
    raise RuntimeError(
        'DRAFT-005-single-state: draft only, NOT FROZEN, DO NOT PUSH. '
        'Resolve TODO v30 pins + pre-register task list per handoff S5 first. '
        + DRAFT_NOTE
    )

# 以下为推送版骨架（DRAFT 护栏后不可达；保留以说明与 002 相同的执行结构）。
# 推送前需：填 TODO_V30_PINS → 移除护栏 → 冻结 BUNDLE/SKILL 哈希 → 重查配额。
assert TODO_V30_PINS['tasks_sha256'] is not None, 'Fill TODO v30 pins before push'
assert not RUN_ROOT.exists() and not AGENT_ROOT.exists(), 'Never overwrite a prior run'
RUN_ROOT.mkdir(parents=True)
AGENT_ROOT.mkdir(parents=True)

FROZEN_DIR = RUN_ROOT / 'frozen_bundle'
FROZEN_DIR.mkdir()
for relative, content in BUNDLE_SINGLE_STATE.items():
    atomic_text(FROZEN_DIR / 'variants' / VARIANT / relative, content)
    atomic_text(AGENT_ROOT / VARIANT / relative, content)
# 三技能包：推送版在此从候选目录真实拷贝脚本并按 SKILL_INVENTORY SHA 校验
# （本 DRAFT 不内嵌全文；禁止 symlink；禁止引用仓库外路径）。
atomic_json(FROZEN_DIR / 'skill_inventory.json', SKILL_INVENTORY)
atomic_json(RUN_ROOT / 'draft_experiment.json', EXPERIMENT_DRAFT)

DATA_DIR = resolve_input(
    [INPUT_ROOT / 'competitions/gemma-4-developer-agent', INPUT_ROOT / 'gemma-4-developer-agent'],
    'tasks.jsonl')
TASKS_PATH = DATA_DIR / 'tasks.jsonl'
WHEELHOUSE_DIR = resolve_input(
    [INPUT_ROOT / 'datasets/metric/gemma-4-developer-agent-wheelhouse',
     INPUT_ROOT / 'gemma-4-developer-agent-wheelhouse'],
    'swegemma-0.2.7-py3-none-any.whl')
MODEL_PATH = resolve_input(
    [INPUT_ROOT / 'models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2',
     INPUT_ROOT / 'gemma-4-31b-it-qat-w4a16-ct/2'],
    'tokenizer.json', 'gemma-4-31b')

loaded = [name for name in ('torch', 'swegemma', 'adk_submission', 'adk_eval_core',
                           'google.adk', 'vllm', 'litellm') if name in sys.modules]
assert not loaded, f'Restart the kernel before running: packages already loaded: {loaded}'
os.environ.update({
    'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'TRANSFORMERS_NO_TF': '1',
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'VLLM_ENGINE_READY_TIMEOUT_S': '1200', 'VLLM_NO_USAGE_STATS': '1',
    'OTEL_SDK_DISABLED': 'true', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
})

# 指纹校验（推送版）：官方包版本+哈希、tasks.jsonl 哈希+字节数，任一不符即停。
# 具体期望值填 TODO_V30_PINS 后在此比对（本 DRAFT 留空，不抄 002 旧指纹）。


def run_sync(function, **kwargs):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(function(**kwargs))).result()
    return asyncio.run(function(**kwargs))


class ArchivingVllmServer:
    # 包装官方 server：stop() 前先归档服务日志（含启动失败路径），再执行官方清理。
    # 此处仅保留结构说明；推送版继承官方 VllmServer 并覆盖 stop()。
    def stop(self, timeout=10.0):
        raise NotImplementedError('push version inherits official VllmServer; DRAFT unreachable')


# 执行结构（与 002 相同思想，单臂版）：
# 1. 单次模型加载：server.start() 一次，整轮 6 题共用；server_start_count=1。
# 2. 逐题重置：每 attempt 新建 Evaluator（新 SubprocessManager/沙箱）+ 新会话上下文；
#    attempt 目录独立（attempt_manifest/result.json/patches/test_outputs/traces）。
# 3. 独立结果归档：每题 artifact_manifest.json + attempt_status.json；variant 级增量汇总。
# 4. 服务日志归档后停止：finally 中 server.stop()（先归档 vllm 日志），再写 joint_summary。
