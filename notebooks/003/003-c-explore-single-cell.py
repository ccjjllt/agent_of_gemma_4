# 003-C explore：浏览器单 cell 跑法（粘贴到官方 Getting Started 的 fork 里运行）
# 前置：Copy and Edit（继承比赛+wheelhouse+模型三数据源），Accelerator 选 GPU L4。
# 6题约40分钟。跑完保留 working 产物。
# 变体提示词与 research/003/variants/c/agent 四文件逐字节一致（见生成校验）。
# R01-R10/R13 已修版 + A0 共同修订基线（证据才改、临时文件不入diff、仅注册工具、截断重叠重读、提交后停止）。
# A0 SYSTEM_MD 与 research/003/variants/a0/agent/prompts/system.md 逐字节一致；其余三段与变体目录一致。
#
# 前置（Kaggle 页面点选，无需代码）：
# 1. 打开官方 Getting Started Notebook → Copy and Edit（fork 会继承全部 3 个数据源：
#    competition gemma-4-developer-agent、dataset wheelhouse、model gemma-4-31b…/2）。
# 2. 右侧 Settings → Accelerator 选 GPU L4（可复现为 4×L4；其他卡数为独立可行性问题，R05）。
# 3. 新建一个空 Code cell，把本文件全文粘贴进去，Run（预计 30~45 分钟：装包约1分钟、
#    模型启动约7分钟、3 题×5分钟上限）。
# 4. 跑完把输出（含 ROWS/RESOLVED 行与 EVAL_WALL_MIN）贴回，存档到 docs/progress/001-baseline.md。
#
# Smoke 任务（splits固化顺序）：fastapi_11194 / requests_6592 / rich_2725；预算 60s/50calls/5min/80turns。
# 注意：MCP 的 kaggle_save_notebook 通道挂不上数据源（2026-09-28 实测 4 种传参组合均失败），
# 故 GPU 运行走浏览器，MCP 只做状态/输出读取。

from pathlib import Path
import shutil

# ---------- 0. 写入本项目 001 基线文件（与仓库 agent/ 完全一致） ----------
# R04：不删除固定目录复用；按 RUN_ID 隔离执行目录与结果目录；源与执行目录不可相同或互为祖先。
RUN_ID = 'run-003-c-explore'
VARIANT = 'c'
REPEAT_ID = 'r1'
AGENT_DIR = Path('/kaggle/working/agent_exec') / RUN_ID / VARIANT / REPEAT_ID
assert not AGENT_DIR.exists(), f'R04：{AGENT_DIR} 已存在，请换新 RUN_ID，不要覆盖'
AGENT_DIR.parent.mkdir(parents=True, exist_ok=True)
(AGENT_DIR / 'prompts').mkdir(parents=True)
(AGENT_DIR / 'configs').mkdir(parents=True)

AGENT_YAML = '''name: baseline_single_agent_c
model: gemma-4-31b-it-qat-w4a16-ct
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
generate_content_config: !include configs/sampling.yaml
'''

SYSTEM_MD = '''You are a single repository-fix agent. Resolve the injected issue with minimal, evidence-backed edits, then submit.

Initial problem is injected by the harness. Reference `patch`/`test_patch` never enter your prompt. If session state contains non-empty `hints`, treat them as auxiliary context only; if absent, proceed without them. Never fail on a missing hints variable.

Only call tools actually registered for this run. Check the real tool schema before calling; a graph directory existing on disk does not mean graph tools are available. Do not call unexposed tools.

## Flow: locate -> read -> edit -> check -> submit

1. Locate
- Extract file/function/class/error strings from the problem statement first.
- Prefer `read_file` on explicit paths; otherwise use `run_command` with commands actually available in the environment (check with `command -v` first; do not assume `rg` exists).
- Keep every search scoped; preserve hit locations. Graph/retrieval tools are not registered in this round.

2. Read
- Use `read_file(filepath, start_line, end_line)` with 1-based inclusive range, function- or local-scope first, expand only as needed.
- Save the original call arguments and the response `start_line`/`end_line`/`is_truncated`/text length. Character truncation may leave the returned `end_line` beyond actually shown text; do not blindly continue from `end_line + 1`.
- On `is_truncated`, re-read with a smaller window and overlap, then verify against the source. Quote the evidence you will edit before editing.

3. Edit
- Apply the smallest source-only fix via `edit_file`/`write_file`, only when evidence is sufficient. Do not manufacture edits without evidence.
- Keep temporary reproduction scripts and logs out of the source diff. Never modify, create, or delete tests (`test_*.py`, `*_test.py`, `tests/`). Never reformat unrelated code.
- Match required error strings, exception types, status codes, and API signatures exactly.

4. Check
- Run only the targeted test for the changed behavior (e.g. `pytest tests/test_target.py -k test_feature`).
- Never run bare `pytest`, `pytest .`, or full-repo sweeps: they exhaust time and turn budgets.
- If a failure is a pre-existing breakage, missing fixture, or environment import error unrelated to your edit, record it and continue; do not repair tests or scoring code.
- On tool error or single-command timeout: record the raw message, narrow scope, retry once with a smaller range; single-command timeout is not whole-task termination.

## No-progress detection (B only, same budget)
- Since the last relevant source, input, or testable-hypothesis change, the same action repeated with equivalent results and no new verifiable evidence is a no-progress pattern. Watch adjacent repeats and short loops such as run-script -> read-same-file -> run-script. Timestamp or print-order noise is not new evidence.
- On two repeats or one short loop with no new evidence, briefly state known facts, excluded hypotheses, and the next different testable direction, then switch scope, check callers, or edit a confirmed issue. Rephrasing alone is not progress. Switching direction does not abandon the task.
- Re-running after a code change, reading a truly undisplayed next segment, or waiting on a genuinely changed state is not invalid repetition.

5. Budget and submit
- Use `get_status()` at stage transitions, before long commands, after edits, and before finalization to check `time_seconds_remaining`, `agent_elapsed_seconds`, `tool_calls_remaining`, `patch_submitted`. It costs no tool calls; do not poll in a tight loop.
- Reserve time to close out. After edits and necessary checks, call `submit_patch()` once with current edits while session time remains, then return a short plain-text summary with no further tool edits. A successfully submitted patch is captured at submit time; later file edits do not replace it.
- Soft close-out (C only): when remaining time is at most 20 seconds, stop opening new investigation directions. Review the existing production diff first and call `submit_patch()`. With no patch, complete only an evidence-sufficient local edit; do not manufacture edits to raise the submit rate. The 20-second value is an experimental candidate, not an optimal threshold or a reliable hard cutoff; free status/submit tools can still be rejected past the deadline.
- End with a short summary: changed files, verification command and result, stop reason. Keep the agent end state separate from the judge result.
'''

SAMPLING_YAML = '''# 003 common sampling (same as 001 v1, frozen across A0/B/C)
# 沿用官方 v2 初值；刻意省略 thinking_level，避免映射到服务端不兼容参数。
# 输出上限含在 32,768 token 总上下文内，不额外叠加。
temperature: 0.2
top_p: 0.95
max_output_tokens: 16384
thinking_config:
  thinking_budget: 4096
  include_thoughts: true
'''

EVAL_YAML = '''# 003 common eval budgets (frozen across A0/B/C; same as 001 dev-start)
# 官方 sample 默认: 60 / 10 / 1 / 50。本轮保持 60 / 50 / 5 / 80 作行为对照，
# timeout_seconds 同时限制 Agent 命令与最终验证（swegemma 0.2.7），改动须全组一致。
evaluation:
  timeout_seconds: 60
  max_tool_calls: 50
  max_time_minutes: 5
  max_turns: 80
'''

(AGENT_DIR / 'agent.yaml').write_text(AGENT_YAML, encoding='utf-8')
(AGENT_DIR / 'prompts' / 'system.md').write_text(SYSTEM_MD, encoding='utf-8')
(AGENT_DIR / 'configs' / 'sampling.yaml').write_text(SAMPLING_YAML, encoding='utf-8')
(AGENT_DIR / 'eval_config.yaml').write_text(EVAL_YAML, encoding='utf-8')
print(f'AGENT_DIR={AGENT_DIR}', flush=True)

# ---------- 1. 环境安装（官方 v2 原样；数据源缺失时直接报错） ----------
# R01：单cell为全新kernel；若重跑已用kernel且已加载目标模块，须重启后重跑，不要循环重装代替重启。
# R02：版本化staging，只装本清单内wheel；源空/悬空/多版本均报错；记录文件名/大小/SHA-256。
import glob
import hashlib
import importlib
import importlib.metadata as _md
import os
import subprocess
import sys

for _m in ['torch', 'swegemma', 'adk_submission', 'adk_eval_core', 'vllm', 'litellm']:
    if _m in sys.modules:
        print(f'WARNING(R01)：kernel已加载 {_m}，建议重启kernel后重跑', flush=True)

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS'] = '1'
os.environ['VLLM_ENGINE_READY_TIMEOUT_S'] = '1200'
os.environ['VLLM_NO_USAGE_STATS'] = '1'
os.environ['OTEL_SDK_DISABLED'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

for pat in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth', '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for pth in glob.glob(pat):
        try:
            os.unlink(pth)
        except OSError:
            pass

WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
assert WHEELHOUSE_DIR.exists(), f'wheelhouse 未挂载：{WHEELHOUSE_DIR}（确认 fork 继承了 dataset 输入）'
src_wheels = sorted(WHEELHOUSE_DIR.glob('*.whl'))
assert len(src_wheels) > 0, 'wheelhouse 为空，中断（不要继续跑）'
manifest = []
_seen_dist = {}
for w in src_wheels:
    if 'cutlass' in w.name.lower():
        continue
    _dist = w.name.split('-')[0].replace('_', '-').lower()
    assert _dist not in _seen_dist, f'R02：同一distribution多版本 {_dist}: {_seen_dist[_dist]} vs {w.name}'
    _seen_dist[_dist] = w.name
    manifest.append((w.name, w.stat().st_size, hashlib.sha256(w.read_bytes()).hexdigest()))
print(f'wheelhouse src count={len(manifest)}', flush=True)
for name, size, h in manifest:
    print(f'  {name} size={size} sha256={h[:16]}…', flush=True)
tmp_whl = Path('/tmp/wheelhouse-003-c-explore-v1')
assert not tmp_whl.exists() or not any(tmp_whl.iterdir()), f'R02：staging已有旧内容 {tmp_whl}，请换新干净目录'
tmp_whl.mkdir(parents=True, exist_ok=True)
for w in WHEELHOUSE_DIR.glob('*.whl'):
    if 'cutlass' in w.name.lower():
        continue
    target_name = (w.name.replace('cu128', '+cu128')
                   if ('cu128' in w.name and '+' not in w.name) else w.name)
    target = tmp_whl / target_name
    assert not target.exists() and not target.is_symlink(), f'R02 staging冲突：{target}'
    os.symlink(w, target)
wheels = sorted(str(w) for w in tmp_whl.glob('*.whl'))
assert len(wheels) == len(manifest), 'staging清单与来源不一致，中断'
print(f'Installing {len(wheels)} wheels from {tmp_whl}...', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                '--no-deps', '--force-reinstall', *wheels], check=True)
importlib.invalidate_caches()
print('Wheelhouse installation complete. 执行阶段用importlib.metadata核对实际版本；__import__(google.adk)不能打印ADK版本 (R02).', flush=True)

# ---------- 2. 版本记录 + 任务 + 基线校验 ----------
import hashlib
import json
import platform
import yaml

yaml.SafeLoader.add_constructor('!include', lambda l, n: l.construct_scalar(n))
print(f'python={sys.version.split()[0]} platform={platform.platform()}', flush=True)
for dist in ['swegemma', 'adk-submission', 'adk-eval-core', 'google-adk', 'vllm', 'litellm', 'torch']:
    try:
        print(f'{dist} dist-version={_md.version(dist)}', flush=True)
    except Exception as e:
        print(f'{dist} NOT installed yet ({e})', flush=True)
import torch
assert torch.cuda.is_available(), 'R05：无CUDA时在加载前停止'
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} '
      f'gpus={torch.cuda.device_count()}', flush=True)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  gpu{i}: {p.name} mem={p.total_memory/2**30:.1f}GiB', flush=True)
print(f'torch cuda_build={torch.version.cuda} bf16={torch.cuda.is_bf16_supported()}', flush=True)

from swegemma.models import load_tasks
DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
WORKING_DIR = Path('/kaggle/working')
RESULTS_DIR = WORKING_DIR / 'results' / '003' / RUN_ID / VARIANT / REPEAT_ID
assert not RESULTS_DIR.exists(), f'R04：{RESULTS_DIR} 已存在，请换新RUN_ID'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASKS_PATH = DATA_DIR / 'tasks.jsonl'
assert TASKS_PATH.exists(), f'比赛数据未挂载：{TASKS_PATH}'
raw = TASKS_PATH.read_bytes()
print(f'tasks.jsonl bytes={len(raw)} sha256={hashlib.sha256(raw).hexdigest()}', flush=True)
tasks = load_tasks(TASKS_PATH)
print(f'Loaded {len(tasks)} tasks', flush=True)

# R03：探索集=smoke(splits固化顺序)+dev_future各1题（种子20260928，不看成功率/参考补丁；见experiment_manifest）。
SMOKE_IDS = ['fastapi_11194', 'requests_6592', 'rich_2725']
EXTRA_IDS = ['fastapi_15280', 'rich_3905', 'requests_7315']
by_id = {t.instance_id: t for t in tasks}
assert len(by_id) == len(tasks), 'R03：任务ID不唯一'
for _rid in SMOKE_IDS + EXTRA_IDS:
    assert _rid in by_id, f'R03：划分ID不存在 {_rid}，不得静默跳过'
SELECTED = [by_id[i] for i in SMOKE_IDS] + [by_id[i] for i in EXTRA_IDS]
print(f'SELECTED={[t.instance_id for t in SELECTED]} FROZEN (R03)', flush=True)
print(f'tasks_sha256={hashlib.sha256(raw).hexdigest()} RESULTS_DIR={RESULTS_DIR}', flush=True)

cfg = yaml.safe_load((AGENT_DIR / 'agent.yaml').read_text(encoding='utf-8'))
assert 'adapter' not in cfg, '001 不设 adapter'
assert set(cfg['tools']) == {'run_command', 'read_file', 'edit_file',
                             'write_file', 'get_status', 'submit_patch'}
ev0 = yaml.safe_load((AGENT_DIR / 'eval_config.yaml').read_text(encoding='utf-8'))['evaluation']
assert set(ev0) == {'timeout_seconds', 'max_tool_calls',
                    'max_time_minutes', 'max_turns'}
print(f'agent OK budgets={ev0}', flush=True)

# ---------- 3. 模型服务 ----------
import time
import litellm
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
from swegemma.models.discovery import validate_single_declared_model

litellm.drop_params = True
TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'
MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
assert MODEL_PATH.exists(), f'模型未挂载：{MODEL_PATH}'
try:
    _mst = MODEL_PATH.stat()
    print(f'model_path stat size={_mst.st_size} mtime={_mst.st_mtime}', flush=True)
except Exception as e:
    print(f'model_path stat failed: {e}', flush=True)
for _probe in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
    print(f'  {_probe} exists={(MODEL_PATH / _probe).exists()}', flush=True)
t0 = time.time()
declared_model = validate_single_declared_model(AGENT_DIR)
assert declared_model == TARGET_MODEL_NAME, f'R05：{declared_model} != {TARGET_MODEL_NAME}'
adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
assert hasattr(adapters, 'adapters') and isinstance(adapters.adapters, dict), \
    f'R06：未知AdapterManifest形状 type={type(adapters).__name__}，记录版本并停止'
n_adapters = len(adapters.adapters)
print(f'adapter_manifest_type={type(adapters).__name__} n_adapters={n_adapters}',
      flush=True)
assert n_adapters == 0, f'003-c 应无 adapter：{adapters!r}'
print(f'declared_model={declared_model} adapters={adapters}', flush=True)
gpu_count = torch.cuda.device_count()
assert gpu_count >= 1, 'R05：无GPU停止'
tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
print(f'tp_size={tp_size}（可复现4xL4；其他卡数为独立可行性，R05）', flush=True)
vllm_cfg = VllmConfig(
    model=str(MODEL_PATH), port=8000, host='127.0.0.1',
    tool_call_parser='gemma4', reasoning_parser='gemma4', max_model_len=32768,
    dtype='bfloat16' if (torch.cuda.is_available()
                         and torch.cuda.is_bf16_supported()) else 'auto',
    gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
    enable_lora=True, max_loras=8, max_lora_rank=128,
    tensor_parallel_size=tp_size, startup_timeout=60 * 20)
server_instance = VllmServer(vllm_cfg, adapter_manifest=adapters)
try:
    server_instance.start()
except Exception as e:
    print(f'vLLM start failed: {type(e).__name__}: {e} output={str(getattr(e, "output", ""))[:1000]}', flush=True)
    print(f'log_path={getattr(server_instance, "log_path", None)}（R07：原日志丢失则记证据缺失）', flush=True)
    raise
print(f'vLLM started on {server_instance.base_url} tp={tp_size} '
      f'startup={time.time()-t0:.1f}s log_path={getattr(server_instance, "log_path", None)}', flush=True)
print('R11：图目录存在不代表图工具可用；以注册6工具与实际schema为准', flush=True)
models = server_instance.create_model_registry(
    aliases=[declared_model, TARGET_MODEL_NAME],
    model_prefix='openai/', api_key='EMPTY')
print(f'model_registry_content={str(models)[:500]}', flush=True)

# ---------- 4. 逐题执行（顺序） ----------
import asyncio
import concurrent.futures
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator


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


sec = yaml.safe_load((AGENT_DIR / 'eval_config.yaml').read_text(encoding='utf-8'))['evaluation']
timeout_seconds = int(sec['timeout_seconds'])
max_tool_calls = int(sec['max_tool_calls'])
max_time_minutes = float(sec['max_time_minutes'])
max_turns = int(sec['max_turns'])
print(f'budgets: {sec}', flush=True)
limits, gen_constraints = build_submission_limits()
# R04：结果目录已按RUN_ID隔离；先写manifest再开始；R13：SwegemmaContext分段计时
(agent_yaml_txt, system_txt) = ((AGENT_DIR / 'agent.yaml').read_text(encoding='utf-8'), (AGENT_DIR / 'prompts' / 'system.md').read_text(encoding='utf-8'))
run_manifest = {
    'run_id': RUN_ID, 'variant': VARIANT, 'task_ids': [t.instance_id for t in SELECTED],
    'agent_yaml_sha256': hashlib.sha256(agent_yaml_txt.encode()).hexdigest(),
    'system_md_sha256': hashlib.sha256(system_txt.encode()).hexdigest(),
    'tasks_sha256': hashlib.sha256(TASKS_PATH.read_bytes()).hexdigest(),
    'eval_budgets': sec, 'sandbox': 'subprocess',
    'model_alias': TARGET_MODEL_NAME, 'declared_model': declared_model,
    'server_base_url': str(getattr(server_instance, 'base_url', '')),
    'outer_start': time.time(),
    'python_version': sys.version.split()[0],
    'executable': sys.executable,
    'gpu_count': gpu_count,
    'wheel_manifest': [{'name': n, 'size': s, 'sha256': h} for n, s, h in manifest],
}
(RESULTS_DIR / 'run_manifest.json').write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding='utf-8')
print(f'manifest written: {RESULTS_DIR / "run_manifest.json"}', flush=True)
eval_config = EvalConfig(
    tasks_path=TASKS_PATH, snapshots_dir=DATA_DIR / 'snapshots',
    results_dir=RESULTS_DIR, submission_dir=AGENT_DIR,
    models=models, sandbox='subprocess',
    timeout_seconds=timeout_seconds, max_time_minutes=max_time_minutes,
    max_tool_calls=max_tool_calls, max_turns=max_turns,
    limits=limits, generation_constraints=gen_constraints,
    adapter_manifest=adapters,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800,
                                            cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=15, overlap_size=2, token_threshold=14336,
        event_retention_size=5),
    graph_dir=str(DATA_DIR / 'graphs'),
    embeddings_dir=str(DATA_DIR / 'embeddings'),
    wheels_dir=DATA_DIR / 'wheels', verbose=False)
evaluator = Evaluator(eval_config)
from swegemma.results import append_task_result
try:
    from swegemma.context import SwegemmaContext as _Ctx
    HAS_CTX = True
except Exception as e:
    _Ctx, HAS_CTX = None, False
    print(f'R13：SwegemmaContext不可用 ({e})', flush=True)
try:
    from swegemma.harness.verification import save_trace_artifact as _save_trace
    HAS_SAVE_TRACE = True
except Exception as e:
    _save_trace, HAS_SAVE_TRACE = None, False
    print(f'R08：save_trace_artifact不可用 ({e})', flush=True)
running_results = []
status = {}
for _t in SELECTED:
    status[_t.instance_id] = 'planned'
t_start = time.time()
try:
    for idx, task in enumerate(SELECTED, start=1):
        status[task.instance_id] = 'running'
        print(f'[{idx}/{len(SELECTED)}] {task.instance_id} ({task.repo})...', flush=True)
        ctx = None
        if HAS_CTX:
            try:
                ctx = _Ctx(docker_manager=evaluator.docker, task=task, graph_dir=eval_config.graph_dir, embeddings_dir=eval_config.embeddings_dir, budget=eval_config.budget, harness=eval_config.harness)
            except Exception as e:
                print(f'  ctx create failed: {type(e).__name__}: {e}', flush=True)
        t0_outer = time.perf_counter()
        try:
            if ctx is not None:
                try:
                    r = run_sync(evaluator.evaluate_task, task=task, task_index=idx, total_tasks=len(SELECTED), context=ctx)
                except TypeError:
                    print('  context参数不受支持，退回无context调用 (R13)', flush=True)
                    r = run_sync(evaluator.evaluate_task, task=task, task_index=idx, total_tasks=len(SELECTED))
            else:
                r = run_sync(evaluator.evaluate_task, task=task, task_index=idx, total_tasks=len(SELECTED))
        except KeyboardInterrupt:
            status[task.instance_id] = 'interrupted'
            for _t in SELECTED:
                if status.get(_t.instance_id) == 'planned':
                    status[_t.instance_id] = 'not_run'
            print('  interrupted；保留现场 (R08)', flush=True)
            break
        except Exception as e:
            status[task.instance_id] = f'error:{type(e).__name__}:{str(e)[:300]}'
            print(f'  error: {type(e).__name__}: {str(e)[:500]}', flush=True)
            continue
        running_results.append(r)
        try:
            append_task_result(r, RESULTS_DIR, running_results)
        except Exception as e:
            print(f'  append failed: {type(e).__name__}: {str(e)[:300]}', flush=True)
        try:
            tr_file = RESULTS_DIR / 'traces' / f'trace_{task.instance_id}.json'
            if getattr(r, 'trace', None) is not None and not tr_file.exists() and HAS_SAVE_TRACE:
                _save_trace(r.trace, RESULTS_DIR, task.instance_id)
        except Exception as e:
            print(f'  trace supplement failed: {type(e).__name__}', flush=True)
        status[task.instance_id] = 'completed'
        psize = getattr(r, 'agent_patch_size', None)
        if psize is None:
            psize = len(getattr(r, 'agent_patch', None) or '')
        _cinfo = ''
        if ctx is not None:
            _fs = ['task_start_time', 'agent_start_time', 'agent_end_time', 'agent_elapsed_seconds', 'patch_submitted', 'tool_calls', 'total_llm_calls']
            _cinfo = ' ' + ' '.join(f'{_f}={getattr(ctx, _f, "")}' for _f in _fs)
        print(f'  -> resolved={r.resolved} exit={r.test_exit_code} '
              f'patch_size={psize} calls={r.tool_calls} '
              f'dur={r.duration_seconds:.1f}s outer={time.perf_counter()-t0_outer:.1f}s{_cinfo}', flush=True)
finally:
    (RESULTS_DIR / 'attempt_status.json').write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')
    try:
        lp = getattr(server_instance, 'log_path', None)
        if lp is not None and Path(lp).exists():
            shutil.copy(Path(lp), RESULTS_DIR / 'vllm_server.archived.log')
            print('archived server log (R07)', flush=True)
    except Exception as e:
        print(f'log archive failed: {e}', flush=True)
    try:
        server_instance.stop(timeout=10.0)
        print('server stopped (仅本批，R07)', flush=True)
    except Exception as e:
        print(f'server stop failed: {e}', flush=True)
print(f'EVAL_WALL_MIN={(time.time()-t_start)/60:.1f}', flush=True)

# ---------- 5. 结果汇总 + 打包 ----------
import hashlib as _hl
import pandas as pd
import zipfile
from swegemma.config import ALLOWED_SUBMISSION_EXTENSIONS, MAX_SUBMISSION_SIZE_BYTES

rows = []
tr = RESULTS_DIR / 'task_results.jsonl'
print(f'task_results exists={tr.exists()} dir={RESULTS_DIR}', flush=True)
bad = []
if tr.exists():
    for ln, line in enumerate(tr.read_text(encoding='utf-8').splitlines(), start=1):
        if not line.strip():
            continue
        try:
            j = json.loads(line)
        except Exception as e:
            bad.append((ln, str(e)[:200]))
            continue
        pid = j.get('instance_id', '')
        pf = RESULTS_DIR / 'patches' / f'{pid}.patch'
        pt = pf.read_text(encoding='utf-8') if pf.exists() else ''
        rows.append({
            'instance_id': pid,
            'resolved': j.get('resolved', ''),
            'test_exit_code': j.get('test_exit_code', ''),
            'agent_patch_size': j.get('agent_patch_size', ''),
            'patch_file_chars': len(pt),
            'patch_nonempty': bool(pt.strip()),
            'tool_calls': j.get('tool_calls', ''),
            'total_llm_calls': j.get('total_llm_calls', ''),
            'duration_seconds': j.get('duration_seconds', ''),
            'error': j.get('error', ''),
            'test_output_bytes': (RESULTS_DIR / 'test_outputs' / f'{pid}.log').stat().st_size if (RESULTS_DIR / 'test_outputs' / f'{pid}.log').exists() else '',
        })
df = pd.DataFrame(rows)
print(f'ROWS={len(df)} RESOLVED={(df["resolved"] == True).sum() if len(df) else 0} bad_lines={len(bad)}',
      flush=True)
if bad:
    print(f'坏尾行保留为证据，不静默丢弃 (R09): {bad[:3]}', flush=True)
if len(df) == 0:
    print('无数据则报告缺失，不生成0分汇总 (R09)', flush=True)
else:
    print(df.to_string() if len(df) else 'no rows', flush=True)
    df.to_csv(RESULTS_DIR / 'analysis_003_c_explore.csv', index=False)
print('result files:', flush=True)
for p in sorted(RESULTS_DIR.rglob('*')):
    if p.is_file():
        print(' ', p.relative_to(RESULTS_DIR), p.stat().st_size, flush=True)

assert (AGENT_DIR / 'agent.yaml').exists(), 'R10：冻结目录缺agent.yaml'
_atxt = (AGENT_DIR / 'agent.yaml').read_text(encoding='utf-8')
assert 'prompts/system.md' in _atxt, 'R10：include目标须在包内'
for _ln in _atxt.splitlines():
    if '!include' in _ln:
        assert '..' not in _ln, f'R10：include越界：{_ln.strip()}'
for sub in sorted(AGENT_DIR.rglob('*')):
    if sub.is_symlink():
        raise AssertionError(f'R10禁symlink：{sub}')
zip_path = Path(shutil.make_archive(str(RESULTS_DIR / 'submission-003-c-explore'), 'zip',
                                    root_dir=AGENT_DIR))
with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
    assert 'agent.yaml' in names, f'包根缺 agent.yaml：{names[:10]}'
    assert not any(n.startswith('sub_agents/') or n.startswith('adapters/')
                   for n in names), '003-c 不带 sub_agents/adapters'
    total = sum(i.file_size for i in zf.infolist())
    assert total <= MAX_SUBMISSION_SIZE_BYTES, f'超 3GiB：{total}'
    for info in zf.infolist():
        if not info.is_dir():
            assert Path(info.filename).suffix.lower() in ALLOWED_SUBMISSION_EXTENSIONS, \
                f'非法扩展名：{info.filename}'
print(f'submission.zip OK {zip_path.stat().st_size} bytes files={names}', flush=True)
print(f'zip sha256={_hl.sha256(zip_path.read_bytes()).hexdigest()}', flush=True)
print('R10：Notebook/参考答案/日志/其他候选/训练数据留在包外', flush=True)
