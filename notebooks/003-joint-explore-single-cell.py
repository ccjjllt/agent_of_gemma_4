# 003 joint exploration: one GPU allocation, one model startup, sequential attempts.
# Frozen A0/B/C configs and task/order manifest; no result from another arm enters an agent.
# Required Kaggle inputs: competition, official wheelhouse, and model version 2.
# The harness performs the same official evaluation as the existing single-arm carriers.
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

# Change only this list when replacing still-queued arms. Completed arms remain historical.
SELECTED_VARIANTS = ('a0', 'b', 'c')
REPEAT_ID = 'r1'
RUN_ID = 'run-003-joint-explore-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
BUNDLE = json.loads('{"a0": {"agent.yaml": "name: baseline_single_agent_a0\\nmodel: gemma-4-31b-it-qat-w4a16-ct\\ninstruction: !include prompts/system.md\\ntools:\\n  - run_command\\n  - read_file\\n  - edit_file\\n  - write_file\\n  - get_status\\n  - submit_patch\\ngenerate_content_config: !include configs/sampling.yaml\\n", "prompts/system.md": "You are a single repository-fix agent. Resolve the injected issue with minimal, evidence-backed edits, then submit.\\n\\nInitial problem is injected by the harness. Reference `patch`/`test_patch` never enter your prompt. If session state contains non-empty `hints`, treat them as auxiliary context only; if absent, proceed without them. Never fail on a missing hints variable.\\n\\nOnly call tools actually registered for this run. Check the real tool schema before calling; a graph directory existing on disk does not mean graph tools are available. Do not call unexposed tools.\\n\\n## Flow: locate -> read -> edit -> check -> submit\\n\\n1. Locate\\n- Extract file/function/class/error strings from the problem statement first.\\n- Prefer `read_file` on explicit paths; otherwise use `run_command` with commands actually available in the environment (check with `command -v` first; do not assume `rg` exists).\\n- Keep every search scoped; preserve hit locations. Graph/retrieval tools are not registered in this round.\\n\\n2. Read\\n- Use `read_file(filepath, start_line, end_line)` with 1-based inclusive range, function- or local-scope first, expand only as needed.\\n- Save the original call arguments and the response `start_line`/`end_line`/`is_truncated`/text length. Character truncation may leave the returned `end_line` beyond actually shown text; do not blindly continue from `end_line + 1`.\\n- On `is_truncated`, re-read with a smaller window and overlap, then verify against the source. Quote the evidence you will edit before editing.\\n\\n3. Edit\\n- Apply the smallest source-only fix via `edit_file`/`write_file`, only when evidence is sufficient. Do not manufacture edits without evidence.\\n- Keep temporary reproduction scripts and logs out of the source diff. Never modify, create, or delete tests (`test_*.py`, `*_test.py`, `tests/`). Never reformat unrelated code.\\n- Match required error strings, exception types, status codes, and API signatures exactly.\\n\\n4. Check\\n- Run only the targeted test for the changed behavior (e.g. `pytest tests/test_target.py -k test_feature`).\\n- Never run bare `pytest`, `pytest .`, or full-repo sweeps: they exhaust time and turn budgets.\\n- If a failure is a pre-existing breakage, missing fixture, or environment import error unrelated to your edit, record it and continue; do not repair tests or scoring code.\\n- On tool error or single-command timeout: record the raw message, narrow scope, retry once with a smaller range; single-command timeout is not whole-task termination.\\n\\n5. Budget and submit\\n- Use `get_status()` at stage transitions, before long commands, after edits, and before finalization to check `time_seconds_remaining`, `agent_elapsed_seconds`, `tool_calls_remaining`, `patch_submitted`. It costs no tool calls; do not poll in a tight loop.\\n- Reserve time to close out. After edits and necessary checks, call `submit_patch()` once with current edits while session time remains, then return a short plain-text summary with no further tool edits. A successfully submitted patch is captured at submit time; later file edits do not replace it.\\n- End with a short summary: changed files, verification command and result, stop reason. Keep the agent end state separate from the judge result.\\n", "configs/sampling.yaml": "# 003 common sampling (same as 001 v1, frozen across A0/B/C)\\n# 沿用官方 v2 初值；刻意省略 thinking_level，避免映射到服务端不兼容参数。\\n# 输出上限含在 32,768 token 总上下文内，不额外叠加。\\ntemperature: 0.2\\ntop_p: 0.95\\nmax_output_tokens: 16384\\nthinking_config:\\n  thinking_budget: 4096\\n  include_thoughts: true\\n", "eval_config.yaml": "# 003 common eval budgets (frozen across A0/B/C; same as 001 dev-start)\\n# 官方 sample 默认: 60 / 10 / 1 / 50。本轮保持 60 / 50 / 5 / 80 作行为对照，\\n# timeout_seconds 同时限制 Agent 命令与最终验证（swegemma 0.2.7），改动须全组一致。\\nevaluation:\\n  timeout_seconds: 60\\n  max_tool_calls: 50\\n  max_time_minutes: 5\\n  max_turns: 80\\n"}, "b": {"agent.yaml": "name: baseline_single_agent_b\\nmodel: gemma-4-31b-it-qat-w4a16-ct\\ninstruction: !include prompts/system.md\\ntools:\\n  - run_command\\n  - read_file\\n  - edit_file\\n  - write_file\\n  - get_status\\n  - submit_patch\\ngenerate_content_config: !include configs/sampling.yaml\\n", "prompts/system.md": "You are a single repository-fix agent. Resolve the injected issue with minimal, evidence-backed edits, then submit.\\n\\nInitial problem is injected by the harness. Reference `patch`/`test_patch` never enter your prompt. If session state contains non-empty `hints`, treat them as auxiliary context only; if absent, proceed without them. Never fail on a missing hints variable.\\n\\nOnly call tools actually registered for this run. Check the real tool schema before calling; a graph directory existing on disk does not mean graph tools are available. Do not call unexposed tools.\\n\\n## Flow: locate -> read -> edit -> check -> submit\\n\\n1. Locate\\n- Extract file/function/class/error strings from the problem statement first.\\n- Prefer `read_file` on explicit paths; otherwise use `run_command` with commands actually available in the environment (check with `command -v` first; do not assume `rg` exists).\\n- Keep every search scoped; preserve hit locations. Graph/retrieval tools are not registered in this round.\\n\\n2. Read\\n- Use `read_file(filepath, start_line, end_line)` with 1-based inclusive range, function- or local-scope first, expand only as needed.\\n- Save the original call arguments and the response `start_line`/`end_line`/`is_truncated`/text length. Character truncation may leave the returned `end_line` beyond actually shown text; do not blindly continue from `end_line + 1`.\\n- On `is_truncated`, re-read with a smaller window and overlap, then verify against the source. Quote the evidence you will edit before editing.\\n\\n3. Edit\\n- Apply the smallest source-only fix via `edit_file`/`write_file`, only when evidence is sufficient. Do not manufacture edits without evidence.\\n- Keep temporary reproduction scripts and logs out of the source diff. Never modify, create, or delete tests (`test_*.py`, `*_test.py`, `tests/`). Never reformat unrelated code.\\n- Match required error strings, exception types, status codes, and API signatures exactly.\\n\\n4. Check\\n- Run only the targeted test for the changed behavior (e.g. `pytest tests/test_target.py -k test_feature`).\\n- Never run bare `pytest`, `pytest .`, or full-repo sweeps: they exhaust time and turn budgets.\\n- If a failure is a pre-existing breakage, missing fixture, or environment import error unrelated to your edit, record it and continue; do not repair tests or scoring code.\\n- On tool error or single-command timeout: record the raw message, narrow scope, retry once with a smaller range; single-command timeout is not whole-task termination.\\n\\n## No-progress detection (B only, same budget)\\n- Since the last relevant source, input, or testable-hypothesis change, the same action repeated with equivalent results and no new verifiable evidence is a no-progress pattern. Watch adjacent repeats and short loops such as run-script -> read-same-file -> run-script. Timestamp or print-order noise is not new evidence.\\n- On two repeats or one short loop with no new evidence, briefly state known facts, excluded hypotheses, and the next different testable direction, then switch scope, check callers, or edit a confirmed issue. Rephrasing alone is not progress. Switching direction does not abandon the task.\\n- Re-running after a code change, reading a truly undisplayed next segment, or waiting on a genuinely changed state is not invalid repetition.\\n\\n5. Budget and submit\\n- Use `get_status()` at stage transitions, before long commands, after edits, and before finalization to check `time_seconds_remaining`, `agent_elapsed_seconds`, `tool_calls_remaining`, `patch_submitted`. It costs no tool calls; do not poll in a tight loop.\\n- Reserve time to close out. After edits and necessary checks, call `submit_patch()` once with current edits while session time remains, then return a short plain-text summary with no further tool edits. A successfully submitted patch is captured at submit time; later file edits do not replace it.\\n- End with a short summary: changed files, verification command and result, stop reason. Keep the agent end state separate from the judge result.\\n", "configs/sampling.yaml": "# 003 common sampling (same as 001 v1, frozen across A0/B/C)\\n# 沿用官方 v2 初值；刻意省略 thinking_level，避免映射到服务端不兼容参数。\\n# 输出上限含在 32,768 token 总上下文内，不额外叠加。\\ntemperature: 0.2\\ntop_p: 0.95\\nmax_output_tokens: 16384\\nthinking_config:\\n  thinking_budget: 4096\\n  include_thoughts: true\\n", "eval_config.yaml": "# 003 common eval budgets (frozen across A0/B/C; same as 001 dev-start)\\n# 官方 sample 默认: 60 / 10 / 1 / 50。本轮保持 60 / 50 / 5 / 80 作行为对照，\\n# timeout_seconds 同时限制 Agent 命令与最终验证（swegemma 0.2.7），改动须全组一致。\\nevaluation:\\n  timeout_seconds: 60\\n  max_tool_calls: 50\\n  max_time_minutes: 5\\n  max_turns: 80\\n"}, "c": {"agent.yaml": "name: baseline_single_agent_c\\nmodel: gemma-4-31b-it-qat-w4a16-ct\\ninstruction: !include prompts/system.md\\ntools:\\n  - run_command\\n  - read_file\\n  - edit_file\\n  - write_file\\n  - get_status\\n  - submit_patch\\ngenerate_content_config: !include configs/sampling.yaml\\n", "prompts/system.md": "You are a single repository-fix agent. Resolve the injected issue with minimal, evidence-backed edits, then submit.\\n\\nInitial problem is injected by the harness. Reference `patch`/`test_patch` never enter your prompt. If session state contains non-empty `hints`, treat them as auxiliary context only; if absent, proceed without them. Never fail on a missing hints variable.\\n\\nOnly call tools actually registered for this run. Check the real tool schema before calling; a graph directory existing on disk does not mean graph tools are available. Do not call unexposed tools.\\n\\n## Flow: locate -> read -> edit -> check -> submit\\n\\n1. Locate\\n- Extract file/function/class/error strings from the problem statement first.\\n- Prefer `read_file` on explicit paths; otherwise use `run_command` with commands actually available in the environment (check with `command -v` first; do not assume `rg` exists).\\n- Keep every search scoped; preserve hit locations. Graph/retrieval tools are not registered in this round.\\n\\n2. Read\\n- Use `read_file(filepath, start_line, end_line)` with 1-based inclusive range, function- or local-scope first, expand only as needed.\\n- Save the original call arguments and the response `start_line`/`end_line`/`is_truncated`/text length. Character truncation may leave the returned `end_line` beyond actually shown text; do not blindly continue from `end_line + 1`.\\n- On `is_truncated`, re-read with a smaller window and overlap, then verify against the source. Quote the evidence you will edit before editing.\\n\\n3. Edit\\n- Apply the smallest source-only fix via `edit_file`/`write_file`, only when evidence is sufficient. Do not manufacture edits without evidence.\\n- Keep temporary reproduction scripts and logs out of the source diff. Never modify, create, or delete tests (`test_*.py`, `*_test.py`, `tests/`). Never reformat unrelated code.\\n- Match required error strings, exception types, status codes, and API signatures exactly.\\n\\n4. Check\\n- Run only the targeted test for the changed behavior (e.g. `pytest tests/test_target.py -k test_feature`).\\n- Never run bare `pytest`, `pytest .`, or full-repo sweeps: they exhaust time and turn budgets.\\n- If a failure is a pre-existing breakage, missing fixture, or environment import error unrelated to your edit, record it and continue; do not repair tests or scoring code.\\n- On tool error or single-command timeout: record the raw message, narrow scope, retry once with a smaller range; single-command timeout is not whole-task termination.\\n\\n## No-progress detection (B only, same budget)\\n- Since the last relevant source, input, or testable-hypothesis change, the same action repeated with equivalent results and no new verifiable evidence is a no-progress pattern. Watch adjacent repeats and short loops such as run-script -> read-same-file -> run-script. Timestamp or print-order noise is not new evidence.\\n- On two repeats or one short loop with no new evidence, briefly state known facts, excluded hypotheses, and the next different testable direction, then switch scope, check callers, or edit a confirmed issue. Rephrasing alone is not progress. Switching direction does not abandon the task.\\n- Re-running after a code change, reading a truly undisplayed next segment, or waiting on a genuinely changed state is not invalid repetition.\\n\\n5. Budget and submit\\n- Use `get_status()` at stage transitions, before long commands, after edits, and before finalization to check `time_seconds_remaining`, `agent_elapsed_seconds`, `tool_calls_remaining`, `patch_submitted`. It costs no tool calls; do not poll in a tight loop.\\n- Reserve time to close out. After edits and necessary checks, call `submit_patch()` once with current edits while session time remains, then return a short plain-text summary with no further tool edits. A successfully submitted patch is captured at submit time; later file edits do not replace it.\\n- Soft close-out (C only): when remaining time is at most 20 seconds, stop opening new investigation directions. Review the existing production diff first and call `submit_patch()`. With no patch, complete only an evidence-sufficient local edit; do not manufacture edits to raise the submit rate. The 20-second value is an experimental candidate, not an optimal threshold or a reliable hard cutoff; free status/submit tools can still be rejected past the deadline.\\n- End with a short summary: changed files, verification command and result, stop reason. Keep the agent end state separate from the judge result.\\n", "configs/sampling.yaml": "# 003 common sampling (same as 001 v1, frozen across A0/B/C)\\n# 沿用官方 v2 初值；刻意省略 thinking_level，避免映射到服务端不兼容参数。\\n# 输出上限含在 32,768 token 总上下文内，不额外叠加。\\ntemperature: 0.2\\ntop_p: 0.95\\nmax_output_tokens: 16384\\nthinking_config:\\n  thinking_budget: 4096\\n  include_thoughts: true\\n", "eval_config.yaml": "# 003 common eval budgets (frozen across A0/B/C; same as 001 dev-start)\\n# 官方 sample 默认: 60 / 10 / 1 / 50。本轮保持 60 / 50 / 5 / 80 作行为对照，\\n# timeout_seconds 同时限制 Agent 命令与最终验证（swegemma 0.2.7），改动须全组一致。\\nevaluation:\\n  timeout_seconds: 60\\n  max_tool_calls: 50\\n  max_time_minutes: 5\\n  max_turns: 80\\n"}}')
MANIFEST_TEXT = '{"common":{"adapter_manifest":"none","eval_budgets":{"max_time_minutes":5.0,"max_tool_calls":50,"max_turns":80,"timeout_seconds":60},"get_status_rhythm":"stage-transition + before-long-command + after-edit + before-finalize (shared A0/B/C; C adds soft-close decision only)","graph_tools":[],"model_alias":"gemma-4-31b-it-qat-w4a16-ct","official_packages":{"adk-eval-core":"0.1.0 sha256 b1608c4471a2739eca6469735e4e118ff33a20823ce9430719238ea843aa9d1a","adk-submission":"0.2.11 sha256 d14f7d31e724f507322c763bc7c1ed56ac603085b3709afed954e50006b494a8","google-adk":"1.36.1 sha256 1a2f6868c509e3151fb0de3575a7d18b45c338be86f420924dad74e7193631a0","swegemma":"0.2.7 sha256 2b74d402dc95a615f3b36c56fd3f9ba11ddbfb7db8a5a32ec259369a0f559d98"},"sandbox":"subprocess","sampling":{"include_thoughts":true,"max_output_tokens":16384,"temperature":0.2,"thinking_budget":4096,"top_p":0.95},"tool_list":["run_command","read_file","edit_file","write_file","get_status","submit_patch"]},"data":{"source_sha256":"e4b3fd60f69dbc2b9213e54eeb9636db78aefe92c1d06269d73d9f5f8f3c8ad6","splits_path":"research/task_splits.json","splits_seed":20260928,"tasks_bytes":1984455},"exploration_set":{"order_seed":20260928,"orders":["ABC","ACB","BAC","BCA","CAB","CBA"],"tasks":[{"id":"fastapi_11194","role":"smoke"},{"id":"requests_6592","role":"smoke"},{"id":"rich_2725","role":"smoke"},{"id":"fastapi_15280","repo":"fastapi/fastapi","role":"dev_future-extra"},{"id":"rich_3905","repo":"Textualize/rich","role":"dev_future-extra"},{"id":"requests_7315","repo":"psf/requests","role":"dev_future-extra"}]},"heldout_note":"heldout 28 (26 snapshot groups: FastAPI13/Rich11/Requests4) frozen until final freeze; any early use must be logged as exposed, no longer independent","note":"A0 is revised common baseline (differs from 001); 001 kept as history only, not strict paired control. B=A0+no-progress; C=B+20s soft close. Six tasks x three variants = 18 runs, ABC orders balanced, cold/hot cache recorded. Panel 24 (12/8/4) paired only after exploration picks at most one candidate.","panel":{"dev_panel_ids":["fastapi_11194","fastapi_13713","fastapi_13786","fastapi_14266","fastapi_14303","fastapi_14371","fastapi_14430","fastapi_14605","fastapi_14873","fastapi_14962","fastapi_5077","fastapi_9425","requests_6592","requests_6644","requests_7309","requests_7505","rich_2725","rich_3130","rich_3296","rich_3469","rich_3518","rich_3777","rich_3942","rich_4075"],"pairing":"A0-first vs candidate-first 12/12, balanced within repo; first full panel is primary; no post-hoc cherry-picked reruns merged into mean","report_smoke_separately":true},"repeats":{"attempt":"per-task retry gets new attempt id AND new actual results_dir (official filenames contain only task id)","repeat_id":"full batch rerun uses new repeat dir; never overwrite; manifest references attempt originals"},"run_id_template":"results/003/<run_id>/<variant>/<repeat_id>/","variants":{"a0":{"agent_dir":"research/003/variants/a0/agent","desc":"common-revised baseline: evidence-only edits, no temp files in diff, registered-tools only, overlap reread on truncation, submit-then-stop"},"b":{"agent_dir":"research/003/variants/b/agent","desc":"A0 + no-progress identification (adjacent + short-loop, 2-repeat trigger) and direction switch"},"c":{"agent_dir":"research/003/variants/c/agent","desc":"B + 20s soft close-out (review diff and submit; no unfounded edits)"}}}'
EXPERIMENT = json.loads(MANIFEST_TEXT)
EXPECTED_CONFIG_HASHES = {'a0': {'agent.yaml': 'dc61fe02ca6684a8f55deef1d6db6a7658737f2f616e06b421d98ce1418cbd4c', 'prompts/system.md': '85aeb0eff37193988b59f38c439490eb3aa83029896b7617466216326201506c', 'configs/sampling.yaml': '377ff2f96f400d0b6b8d23dfa225e129ce6cd395838058113731d86c5b046f0b', 'eval_config.yaml': '1ff0866e840fbb586a9466dbd81df3f42b8ce045aa740d6fd0eecada66312f2e'}, 'b': {'agent.yaml': 'ba8995db22ae4c6c7b44288f018a7f14f3a82a9e76437913c1020be061ad9408', 'prompts/system.md': 'c9859eb6527b828ca91a1569b1e4cc8251a919f07675d86ed43e30ac7cb3fe25', 'configs/sampling.yaml': '377ff2f96f400d0b6b8d23dfa225e129ce6cd395838058113731d86c5b046f0b', 'eval_config.yaml': '1ff0866e840fbb586a9466dbd81df3f42b8ce045aa740d6fd0eecada66312f2e'}, 'c': {'agent.yaml': '4f1bd53ca5ab470851807a15d359271030c0142f58b4407ef84563d20fccde70', 'prompts/system.md': '35d89afac8bf5eee5e90d2b5adf945ec1b1d78ab213b0d7c60dcc7f65913ca44', 'configs/sampling.yaml': '377ff2f96f400d0b6b8d23dfa225e129ce6cd395838058113731d86c5b046f0b', 'eval_config.yaml': '1ff0866e840fbb586a9466dbd81df3f42b8ce045aa740d6fd0eecada66312f2e'}}
WORKING_DIR = Path('/kaggle/working')
RUN_ROOT = WORKING_DIR / 'results' / '003' / RUN_ID
AGENT_ROOT = WORKING_DIR / 'agent_exec' / RUN_ID
DATA_DIR = Path('/kaggle/input/competitions/gemma-4-developer-agent')
TASKS_PATH = DATA_DIR / 'tasks.jsonl'
WHEELHOUSE_DIR = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
MODEL_PATH = Path('/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2')
TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'
assert SELECTED_VARIANTS and len(set(SELECTED_VARIANTS)) == len(SELECTED_VARIANTS)
assert set(SELECTED_VARIANTS) <= {'a0', 'b', 'c'}, 'Unknown variant: stop, never substitute'
assert not RUN_ROOT.exists() and not AGENT_ROOT.exists(), 'Never overwrite a prior run'
RUN_ROOT.mkdir(parents=True)
AGENT_ROOT.mkdir(parents=True)


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


FROZEN_DIR = RUN_ROOT / 'frozen_bundle'
FROZEN_DIR.mkdir()
atomic_text(FROZEN_DIR / 'experiment_manifest.json', MANIFEST_TEXT)
atomic_json(FROZEN_DIR / 'embedded_config_sha256.json', EXPECTED_CONFIG_HASHES)
for variant, file_map in BUNDLE.items():
    for relative, content in file_map.items():
        digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
        assert digest == EXPECTED_CONFIG_HASHES[variant][relative], f'Config changed: {variant}/{relative}'
        atomic_text(FROZEN_DIR / 'variants' / variant / relative, content)
        atomic_text(AGENT_ROOT / variant / relative, content)

# Preserve the original 18-entry schedule even when an already-completed arm is omitted.
letters = {'A': 'a0', 'B': 'b', 'C': 'c'}
exploration = EXPERIMENT['exploration_set']
assert len(exploration['tasks']) == len(exploration['orders']) == 6
assert exploration['orders'] == ['ABC', 'ACB', 'BAC', 'BCA', 'CAB', 'CBA']
full_schedule = []
for task_slot, (item, order) in enumerate(zip(exploration['tasks'], exploration['orders']), 1):
    for within_task, letter in enumerate(order, 1):
        variant = letters[letter]
        ordinal = len(full_schedule) + 1
        attempt_id = f'{ordinal:02d}-{item["id"]}-{variant}'
        full_schedule.append({
            'ordinal': ordinal, 'task_slot': task_slot, 'within_task': within_task,
            'task_id': item['id'], 'variant': variant, 'attempt_id': attempt_id,
            'selected': variant in SELECTED_VARIANTS,
            'results_dir': str(RUN_ROOT / variant / REPEAT_ID / 'attempts' / attempt_id),
            'config_sha256': EXPECTED_CONFIG_HASHES[variant],
        })
schedule = [entry for entry in full_schedule if entry['selected']]
assert len(schedule) == 6 * len(SELECTED_VARIANTS)
state = {entry['attempt_id']: {**entry, 'status': 'planned' if entry['selected'] else 'excluded_variant'}
         for entry in full_schedule}
atomic_json(RUN_ROOT / 'full_schedule.json', full_schedule)
atomic_json(RUN_ROOT / 'selected_schedule.json', schedule)
atomic_json(RUN_ROOT / 'attempt_status.json', state)
print(f'RUN_ID={RUN_ID} SELECTED_VARIANTS={SELECTED_VARIANTS} ATTEMPTS={len(schedule)}', flush=True)
for entry in schedule:
    print(f'  {entry["ordinal"]:02d}: {entry["task_id"]} {entry["variant"]}', flush=True)

# Install before importing target packages. A warm kernel is rejected, never silently reused.
loaded = [name for name in ('torch', 'swegemma', 'adk_submission', 'adk_eval_core',
                           'google.adk', 'vllm', 'litellm') if name in sys.modules]
assert not loaded, f'Restart the kernel before running: packages already loaded: {loaded}'
os.environ.update({
    'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'TRANSFORMERS_NO_TF': '1',
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'VLLM_ENGINE_READY_TIMEOUT_S': '1200', 'VLLM_NO_USAGE_STATS': '1',
    'OTEL_SDK_DISABLED': 'true', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
})
for pattern in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth',
                '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for path in glob.glob(pattern):
        try:
            os.unlink(path)
        except OSError:
            pass
assert WHEELHOUSE_DIR.is_dir(), f'Missing wheelhouse: {WHEELHOUSE_DIR}'
source_wheels = [path for path in sorted(WHEELHOUSE_DIR.glob('*.whl')) if 'cutlass' not in path.name.lower()]
assert source_wheels, 'Wheelhouse is empty'
wheel_manifest = []
seen_distributions = set()
for path in source_wheels:
    distribution = path.name.split('-')[0].replace('_', '-').lower()
    assert distribution not in seen_distributions, f'Duplicate distribution: {distribution}'
    seen_distributions.add(distribution)
    wheel_manifest.append({'name': path.name, 'distribution': distribution,
                           'size': path.stat().st_size, 'sha256': sha256_file(path)})
for distribution, expected in EXPERIMENT['common']['official_packages'].items():
    version, expected_sha = expected.split(' sha256 ')
    matching = [entry for entry in wheel_manifest if entry['distribution'] == distribution]
    assert len(matching) == 1 and matching[0]['sha256'] == expected_sha, f'Official package changed: {distribution}'
atomic_json(FROZEN_DIR / 'wheel_manifest.json', wheel_manifest)
staging = Path('/tmp') / ('wheelhouse-' + RUN_ID)
staging.mkdir(exist_ok=False)
for path in source_wheels:
    name = path.name.replace('cu128', '+cu128') if 'cu128' in path.name and '+' not in path.name else path.name
    target = staging / name
    assert not target.exists() and not target.is_symlink(), f'Staging collision: {name}'
    target.symlink_to(path)
wheels = sorted(str(path) for path in staging.glob('*.whl'))
assert len(wheels) == len(wheel_manifest)
print(f'Installing {len(wheels)} frozen wheels', flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall', *wheels], check=True)
importlib.invalidate_caches()

import torch
import yaml
import litellm
from adk_submission import VllmConfig, VllmServer, discover_adapters
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import (ALLOWED_ADAPTER_EXTENSIONS, ALLOWED_SUBMISSION_EXTENSIONS,
                            MAX_SUBMISSION_SIZE_BYTES, EvalConfig, build_submission_limits)
from swegemma.context import SwegemmaContext
from swegemma.evaluate import Evaluator
from swegemma.harness.verification import save_trace_artifact
from swegemma.models import load_tasks
from swegemma.models.discovery import validate_single_declared_model
from swegemma.results import append_task_result

yaml.SafeLoader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))
versions = {name: metadata.version(name) for name in
            ('swegemma', 'adk-submission', 'adk-eval-core', 'google-adk', 'vllm', 'litellm', 'torch')}
for name, expected in EXPERIMENT['common']['official_packages'].items():
    assert versions[name] == expected.split(' sha256 ')[0], f'Installed version mismatch: {name}'
assert torch.cuda.is_available() and torch.cuda.device_count() >= 1, 'CUDA unavailable'
gpus = [{'index': i, 'name': torch.cuda.get_device_properties(i).name,
         'memory_bytes': torch.cuda.get_device_properties(i).total_memory}
        for i in range(torch.cuda.device_count())]
assert TASKS_PATH.is_file() and MODEL_PATH.is_dir(), 'Competition data or pinned model is missing'
raw_tasks = TASKS_PATH.read_bytes()
tasks_sha = hashlib.sha256(raw_tasks).hexdigest()
assert tasks_sha == EXPERIMENT['data']['source_sha256'], 'tasks.jsonl changed; stop before inference'
assert len(raw_tasks) == EXPERIMENT['data']['tasks_bytes'], 'tasks.jsonl size mismatch'
(FROZEN_DIR / 'tasks.jsonl').write_bytes(raw_tasks)
tasks = load_tasks(TASKS_PATH)
by_id = {task.instance_id: task for task in tasks}
assert len(by_id) == len(tasks), 'Duplicate task IDs'
for entry in schedule:
    assert entry['task_id'] in by_id, f'Missing frozen task: {entry["task_id"]}'

common = EXPERIMENT['common']
common_sampling = None
common_config = None
budgets = None
for variant in ('a0', 'b', 'c'):
    agent_dir = AGENT_ROOT / variant
    config = yaml.safe_load(BUNDLE[variant]['agent.yaml'])
    sampling = yaml.safe_load(BUNDLE[variant]['configs/sampling.yaml'])
    budget = yaml.safe_load(BUNDLE[variant]['eval_config.yaml'])['evaluation']
    comparable = {key: value for key, value in config.items() if key != 'name'}
    if common_config is None:
        common_config, common_sampling, budgets = comparable, sampling, budget
    assert comparable == common_config and sampling == common_sampling and budget == budgets
    assert budget == common['eval_budgets']
    assert config['tools'] == common['tool_list'] and 'adapter' not in config
    assert validate_single_declared_model(agent_dir) == TARGET_MODEL_NAME
    for path in agent_dir.rglob('*'):
        assert not path.is_symlink(), f'No symlinks in submission: {path}'
assert sampling == {'temperature': common['sampling']['temperature'], 'top_p': common['sampling']['top_p'],
                    'max_output_tokens': common['sampling']['max_output_tokens'],
                    'thinking_config': {'thinking_budget': common['sampling']['thinking_budget'],
                                        'include_thoughts': common['sampling']['include_thoughts']}}
eval_signature = inspect.signature(Evaluator.evaluate_task)
assert 'context' in eval_signature.parameters, f'Unsupported evaluate_task API: {eval_signature}'
# Check support before any call; never catch TypeError and rerun an already-started task.
adapters = discover_adapters(str(AGENT_ROOT / SELECTED_VARIANTS[0]), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
assert hasattr(adapters, 'adapters') and isinstance(adapters.adapters, dict) and not adapters.adapters
litellm.drop_params = True
gpu_count = torch.cuda.device_count()
tp_size = 4 if gpu_count >= 4 else (2 if gpu_count >= 2 else 1)
run_manifest = {
    'run_id': RUN_ID, 'repeat_id': REPEAT_ID, 'selected_variants': list(SELECTED_VARIANTS),
    'full_schedule': full_schedule, 'config_sha256': EXPECTED_CONFIG_HASHES,
    'experiment_manifest_sha256': hashlib.sha256(MANIFEST_TEXT.encode('utf-8')).hexdigest(),
    'tasks_sha256': tasks_sha, 'tasks_bytes': len(raw_tasks), 'versions': versions,
    'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
    'gpus': gpus, 'tensor_parallel_size': tp_size, 'cuda_build': torch.version.cuda,
    'budgets': budgets, 'sampling': common_sampling, 'sandbox': 'subprocess',
    'model_path': str(MODEL_PATH), 'model_alias': TARGET_MODEL_NAME,
    'server_start_count': 0, 'model_requests': 'sequential',
    'context_cache': {'min_tokens': 2048, 'ttl_seconds': 1800, 'cache_intervals': 10},
    'events_compaction': {'compaction_interval': 15, 'overlap_size': 2,
                          'token_threshold': 14336, 'event_retention_size': 5},
    'cache_note': 'Shared server may retain prefix cache; schedule position recorded, not claimed cache-equivalent.',
    'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
print(f'GPUS={gpus} versions={versions}', flush=True)

# Export each unchanged config independently; these ZIPs are outputs, not competition submissions.
for variant in SELECTED_VARIANTS:
    output_dir = RUN_ROOT / variant / REPEAT_ID
    output_dir.mkdir(parents=True)
    zip_path = output_dir / f'submission-003-{variant}-explore.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for relative in BUNDLE[variant]:
            path = AGENT_ROOT / variant / relative
            assert path.suffix.lower() in ALLOWED_SUBMISSION_EXTENSIONS
            archive.write(path, relative)
    with zipfile.ZipFile(zip_path) as archive:
        assert 'agent.yaml' in archive.namelist()
        assert sum(info.file_size for info in archive.infolist()) <= MAX_SUBMISSION_SIZE_BYTES
    atomic_json(output_dir / 'submission_manifest.json', {
        'zip_sha256': sha256_file(zip_path), 'config_sha256': EXPECTED_CONFIG_HASHES[variant]})


def run_sync(function, **kwargs):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(function(**kwargs))).result()
    return asyncio.run(function(**kwargs))


class ArchivingVllmServer(VllmServer):
    # The official start() can call stop() on failure; archive before that internal cleanup too.
    def stop(self, timeout=10.0):
        log_path = getattr(self, 'log_path', None)
        try:
            if log_path and Path(log_path).is_file():
                shutil.copyfile(log_path, RUN_ROOT / 'vllm_server.archived.log')
        except Exception:
            atomic_text(RUN_ROOT / 'server_log_archive_error.txt', traceback.format_exc())
        finally:
            super().stop(timeout=timeout)


server = ArchivingVllmServer(VllmConfig(
    model=str(MODEL_PATH), port=8000, host='127.0.0.1', tool_call_parser='gemma4',
    reasoning_parser='gemma4', max_model_len=32768,
    dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'auto',
    gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
    enable_lora=True, max_loras=8, max_lora_rank=128,
    tensor_parallel_size=tp_size, startup_timeout=60 * 20), adapter_manifest=adapters)
results_by_variant = {variant: [] for variant in SELECTED_VARIANTS}
analysis_rows = []
batch_start = time.perf_counter()
interrupted = False
try:
    startup_start = time.perf_counter()
    run_manifest['server_start_count'] = 1
    atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
    server.start()
    run_manifest['server_startup_seconds'] = time.perf_counter() - startup_start
    run_manifest['server_base_url'] = str(server.base_url)
    atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
    models = server.create_model_registry(aliases=[TARGET_MODEL_NAME], model_prefix='openai/', api_key='EMPTY')
    print(f'vLLM started once in {run_manifest["server_startup_seconds"]:.1f}s', flush=True)
    for execution_index, entry in enumerate(schedule, 1):
        task = by_id[entry['task_id']]
        variant = entry['variant']
        attempt_id = entry['attempt_id']
        attempt_dir = Path(entry['results_dir'])
        attempt_dir.mkdir(parents=True, exist_ok=False)
        variant_dir = RUN_ROOT / variant / REPEAT_ID
        attempt_state = state[attempt_id]
        attempt_state.update(status='running', execution_index=execution_index,
                             started_at=time.time(), server_uptime_seconds=time.perf_counter()-startup_start,
                             first_attempt_on_server=execution_index == 1)
        atomic_json(attempt_dir / 'attempt_manifest.json', attempt_state)
        atomic_json(RUN_ROOT / 'attempt_status.json', state)
        evaluator = None
        ctx = None
        result = None
        attempt_start = time.perf_counter()
        handler = logging.FileHandler(attempt_dir / 'harness.log', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
        logging.getLogger().addHandler(handler)
        print(f'[{execution_index}/{len(schedule)}] {attempt_id}', flush=True)
        try:
            limits, generation_constraints = build_submission_limits()
            eval_config = EvalConfig(
                tasks_path=TASKS_PATH, snapshots_dir=DATA_DIR / 'snapshots', results_dir=attempt_dir,
                submission_dir=AGENT_ROOT / variant, models=models, sandbox='subprocess',
                timeout_seconds=int(budgets['timeout_seconds']), max_tool_calls=int(budgets['max_tool_calls']),
                max_time_minutes=float(budgets['max_time_minutes']), max_turns=int(budgets['max_turns']),
                limits=limits, generation_constraints=generation_constraints, adapter_manifest=adapters,
                context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
                events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                               token_threshold=14336, event_retention_size=5),
                graph_dir=str(DATA_DIR / 'graphs'), embeddings_dir=str(DATA_DIR / 'embeddings'),
                wheels_dir=DATA_DIR / 'wheels', verbose=False)
            evaluator = Evaluator(eval_config)
            # Each Evaluator owns a new SubprocessManager; each task creates fresh agent/judge workspaces.
            assert not evaluator.docker.sandboxes, 'A new evaluator must not reuse a repository'
            ctx = SwegemmaContext(docker_manager=evaluator.docker, task=task,
                                  graph_dir=eval_config.graph_dir, embeddings_dir=eval_config.embeddings_dir,
                                  budget=eval_config.budget, harness=eval_config.harness)
            result = run_sync(evaluator.evaluate_task, task=task, task_index=execution_index,
                              total_tasks=len(schedule), context=ctx)
            # Persist raw result fields before calling the official incremental writers.
            payload = {'instance_id': result.instance_id, 'repo': result.repo, 'resolved': result.resolved,
                       'agent_patch_size': len(result.agent_patch or ''), 'test_exit_code': result.test_exit_code,
                       'duration_seconds': result.duration_seconds, 'tool_calls': result.tool_calls,
                       'total_llm_calls': result.total_llm_calls, 'error': result.error,
                       'trace_json_path': getattr(result, 'trace_json_path', None)}
            atomic_json(attempt_dir / 'result.json', payload)
            safe_id = task.instance_id.replace('/', '__')
            trace_path = attempt_dir / 'traces' / f'trace_{safe_id}.json'
            if not trace_path.is_file() and getattr(result, 'trace', None) is not None:
                save_trace_artifact(result.trace, attempt_dir, task.instance_id)
            atomic_text(attempt_dir / 'patches' / f'{safe_id}.patch', result.agent_patch or '')
            atomic_text(attempt_dir / 'test_outputs' / f'{safe_id}.log', result.test_output or '')
            append_task_result(result, attempt_dir, [result])
            results_by_variant[variant].append(result)
            append_task_result(result, variant_dir, results_by_variant[variant])
            attempt_state['trace_present'] = trace_path.is_file()
            attempt_state['status'] = 'completed' if trace_path.is_file() else 'completed_evidence_gap'
            attempt_state['resolved'] = result.resolved
            analysis_rows.append({**entry, **payload, 'execution_index': execution_index,
                                  'trace_present': trace_path.is_file()})
            atomic_json(RUN_ROOT / 'analysis_rows.json', analysis_rows)
            print(f'  resolved={result.resolved} exit={result.test_exit_code} '
                  f'patch_chars={len(result.agent_patch or "")} llm_calls={ctx.llm_calls_used}', flush=True)
        except KeyboardInterrupt:
            interrupted = True
            attempt_state['status'] = 'interrupted'
            atomic_text(attempt_dir / 'exception.txt', traceback.format_exc())
        except Exception as exc:
            attempt_state.update(status='error', error_type=type(exc).__name__, error=str(exc))
            atomic_text(attempt_dir / 'exception.txt', traceback.format_exc())
            print(f'  ERROR {type(exc).__name__}: {exc}; no automatic retry', flush=True)
        finally:
            attempt_state['outer_seconds'] = time.perf_counter() - attempt_start
            attempt_state['finished_at'] = time.time()
            if ctx is not None:
                context_fields = ('task_start_time', 'agent_start_time', 'agent_end_time',
                                  'agent_elapsed_seconds', 'task_elapsed_seconds', 'patch_submitted',
                                  'tool_calls_used', 'llm_calls_used')
                atomic_json(attempt_dir / 'context_metrics.json',
                            {field: getattr(ctx, field, None) for field in context_fields})
                if result is None and getattr(ctx, 'submitted_patch', None) is not None:
                    atomic_text(attempt_dir / 'submitted_patch_before_exception.patch', ctx.submitted_patch)
            if evaluator is not None:
                try:
                    evaluator.docker.cleanup_all()
                except Exception:
                    atomic_text(attempt_dir / 'sandbox_cleanup_error.txt', traceback.format_exc())
            logging.getLogger().removeHandler(handler)
            handler.close()
            evidence = {str(path.relative_to(attempt_dir)): {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
                        for path in attempt_dir.rglob('*') if path.is_file()
                        and path.name not in ('artifact_manifest.json', 'attempt_status.json')}
            atomic_json(attempt_dir / 'artifact_manifest.json', evidence)
            atomic_json(attempt_dir / 'attempt_status.json', attempt_state)
            atomic_json(RUN_ROOT / 'attempt_status.json', state)
        if interrupted:
            break
except BaseException as exc:
    atomic_text(RUN_ROOT / 'batch_exception.txt', traceback.format_exc())
    if getattr(exc, 'output', None):
        atomic_text(RUN_ROOT / 'server_exception_output.log', str(exc.output))
    raise
finally:
    # stop() archives the server log before official cleanup, including startup-failure paths.
    try:
        server.stop(timeout=10.0)
    except Exception:
        atomic_text(RUN_ROOT / 'server_stop_error.txt', traceback.format_exc())
    for item in state.values():
        if item['status'] == 'planned':
            item['status'] = 'not_run'
        elif item['status'] == 'running':
            item['status'] = 'interrupted_or_batch_error'
    run_manifest['batch_wall_seconds'] = time.perf_counter() - batch_start
    atomic_json(RUN_ROOT / 'attempt_status.json', state)
    atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
    final_summary = {variant: {
        'planned': sum(entry['variant'] == variant for entry in schedule),
        'returned_results': len(results_by_variant[variant]),
        'resolved': sum(bool(result.resolved) for result in results_by_variant[variant]),
        'states': {entry['task_id']: state[entry['attempt_id']]['status']
                   for entry in schedule if entry['variant'] == variant},
    } for variant in SELECTED_VARIANTS}
    atomic_json(RUN_ROOT / 'joint_summary.json', final_summary)
    print(f'FINAL {json.dumps(final_summary)} OUTPUT={RUN_ROOT}', flush=True)
