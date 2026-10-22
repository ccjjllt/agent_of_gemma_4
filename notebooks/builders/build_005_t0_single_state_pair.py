#!/usr/bin/env python3
"""Build a bounded, single-server P005 T0/single_state paired experiment.

Only stdlib is imported at build time. The frozen P002 carrier and existing P005
builder supply official v30 setup and provenance; neither input is modified.
"""
import ast
import hashlib
import json
from pathlib import Path

import build_005_single_state as single

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks/005/005-t0-single-state-pair"
TASKS = ("rich_3942", "fastapi_14873", "fastapi_5077", "rich_3518")
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

# Used by the notebook supervisor and coordinator. Descendants may create their
# own sessions (official sandbox commands do); killing only a process group is
# insufficient. Snapshot the tree before signalling its parents.
PROCESS_HELPERS = r'''
def descendant_pids(root_pid):
    parents = {}
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            row = next(line for line in path.read_text().splitlines() if line.startswith('PPid:'))
            parents[int(path.parent.name)] = int(row.split()[1])
        except (OSError, ValueError, StopIteration):
            pass
    found, frontier = [], [root_pid]
    while frontier:
        frontier = [pid for pid, parent in parents.items() if parent in frontier and pid not in found]
        found.extend(frontier)
    return found


def stop_process_tree(process, grace=3, own_group=False):
    children = descendant_pids(process.pid)
    pids = list(reversed(children)) + [process.pid]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if own_group:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        for pid in pids:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(grace)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
'''

WORKER = r'''
from pathlib import Path
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import traceback
import uuid

SPEC = json.loads(Path(sys.argv[1]).read_text())
ENTRY = SPEC['entry']
ATTEMPT = Path(ENTRY['results_dir'])
STATUS = dict(ENTRY, status='initializing', started_at=time.time(), pid=os.getpid())


def phase(name):
    STATUS['phase'] = name
    STATUS['phase_at'] = time.time()
    atomic_json(ATTEMPT / 'attempt_status.json', STATUS)
    print(json.dumps({'attempt_id': ENTRY['attempt_id'], 'phase': name, 'time': time.time()}), flush=True)


def execute():
    evaluator = ctx = result = None
    start = time.monotonic()
    handler = logging.FileHandler(ATTEMPT / 'harness.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    logging.getLogger().addHandler(handler)
    try:
        phase('import')
        import litellm
        from adk_submission import ModelRegistry, discover_adapters
        from adk_submission.resolvers.generation import install_litellm_reasoning_patch
        from google.adk.models.lite_llm import LiteLlm
        from google.adk.agents.context_cache_config import ContextCacheConfig
        from google.adk.apps._configs import EventsCompactionConfig
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
        from swegemma.context import SwegemmaContext
        from swegemma.evaluate import Evaluator
        from swegemma.harness.verification import save_trace_artifact
        from swegemma.models import load_tasks
        from swegemma.results import append_task_result
        # Exactly the official VllmServer.create_model_registry() no-adapter path,
        # without constructing or starting another inference server.
        install_litellm_reasoning_patch()
        litellm.drop_params = True
        models = ModelRegistry()
        models.register(SPEC['model_alias'], LiteLlm(
            model='openai/' + SPEC['served_model'], api_base=SPEC['base_url'], api_key='EMPTY'))
        data = Path(SPEC['data_dir'])
        task = next(task for task in load_tasks(Path(SPEC['tasks_subset'])) if task.instance_id == ENTRY['task_id'])
        budgets = SPEC['budgets']
        adapters = discover_adapters(SPEC['agent_dir'], adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
        assert not adapters.adapters
        limits, generation_constraints = build_submission_limits()
        config = EvalConfig(
            tasks_path=Path(SPEC['tasks_subset']), snapshots_dir=data / 'snapshots', results_dir=ATTEMPT,
            submission_dir=Path(SPEC['agent_dir']), models=models, sandbox='subprocess',
            timeout_seconds=int(budgets['timeout_seconds']), max_tool_calls=int(budgets['max_tool_calls']),
            max_time_minutes=float(budgets['max_time_minutes']), max_turns=int(budgets['max_turns']),
            limits=limits, generation_constraints=generation_constraints, adapter_manifest=adapters,
            context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
            events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                           token_threshold=14336, event_retention_size=5),
            graph_dir=str(data / 'graphs'), embeddings_dir=str(data / 'embeddings'),
            wheels_dir=data / 'wheels', verbose=False)
        evaluator = Evaluator(config)
        assert not evaluator.docker.sandboxes
        # Official SubprocessManager accepts a base_dir. Fresh attempt-specific
        # directories isolate repository, venv, skill staging, and judge state.
        evaluator.docker.base_dir = Path(SPEC['scratch']) / 'sandboxes'
        evaluator.docker.base_dir.mkdir(parents=True, exist_ok=False)
        ctx = SwegemmaContext(docker_manager=evaluator.docker, task=task,
                              graph_dir=config.graph_dir, embeddings_dir=config.embeddings_dir,
                              budget=config.budget, harness=config.harness)
        STATUS['status'] = 'running'
        phase('evaluate_task')
        result = asyncio.run(evaluator.evaluate_task(
            task=task, task_index=ENTRY['ordinal'], total_tasks=8, context=ctx))
        phase('archive')
        payload = {'instance_id': result.instance_id, 'repo': result.repo, 'resolved': result.resolved,
                   'agent_patch_size': len(result.agent_patch or ''), 'test_exit_code': result.test_exit_code,
                   'duration_seconds': result.duration_seconds, 'tool_calls': result.tool_calls,
                   'total_llm_calls': result.total_llm_calls, 'error': result.error,
                   'trace_json_path': getattr(result, 'trace_json_path', None)}
        atomic_json(ATTEMPT / 'result.json', payload)
        safe_id = task.instance_id.replace('/', '__')
        trace = ATTEMPT / 'traces' / f'trace_{safe_id}.json'
        if not trace.is_file() and getattr(result, 'trace', None) is not None:
            save_trace_artifact(result.trace, ATTEMPT, task.instance_id)
        atomic_text(ATTEMPT / 'patches' / f'{safe_id}.patch', result.agent_patch or '')
        atomic_text(ATTEMPT / 'test_outputs' / f'{safe_id}.log', result.test_output or '')
        append_task_result(result, ATTEMPT, [result])
        STATUS.update(status='completed' if trace.is_file() else 'completed_evidence_gap',
                      trace_present=trace.is_file(), resolved=result.resolved)
    except BaseException as exc:
        STATUS.update(status='error', error_type=type(exc).__name__, error=str(exc))
        atomic_text(ATTEMPT / 'exception.txt', traceback.format_exc())
        if not (ATTEMPT / 'result.json').exists():
            atomic_json(ATTEMPT / 'result.json', {'instance_id': ENTRY['task_id'], 'resolved': None,
                        'error_type': type(exc).__name__, 'error': str(exc)})
    finally:
        STATUS.update(finished_at=time.time(), outer_seconds=time.monotonic() - start)
        if ctx is not None:
            fields = ('task_start_time', 'agent_start_time', 'agent_end_time', 'agent_elapsed_seconds',
                      'task_elapsed_seconds', 'patch_submitted', 'tool_calls_used', 'llm_calls_used')
            atomic_json(ATTEMPT / 'context_metrics.json', {field: getattr(ctx, field, None) for field in fields})
            if result is None and getattr(ctx, 'submitted_patch', None) is not None:
                atomic_text(ATTEMPT / 'submitted_patch_before_exception.patch', ctx.submitted_patch)
        if evaluator is not None:
            try:
                evaluator.docker.cleanup_all()
            except Exception:
                atomic_text(ATTEMPT / 'sandbox_cleanup_error.txt', traceback.format_exc())
        logging.getLogger().removeHandler(handler)
        handler.close()
        atomic_json(ATTEMPT / 'attempt_status.json', STATUS)
    return 0 if STATUS['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(execute())
'''

SCHEDULE = r'''
RUN_ID = os.environ['P005_RUN_ID']
RUN_ROOT = Path(os.environ['P005_RUN_ROOT'])
AGENT_ROOT = Path('/kaggle/working/agent_exec') / RUN_ID
INPUT_ROOT = Path('/kaggle/input')
WORKING_DIR = Path('/kaggle/working')
TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'
SELECTED_VARIANTS = ('t0', 'single_state')
REPEAT_ID = 'r1'
MAX_CONCURRENT = 2
ATTEMPT_WALL_SECONDS = 480


def set_phase(name, **extra):
    atomic_json(RUN_ROOT / 'phase.json', {'phase': name, 'time': time.time(), **extra})

'''

SETUP = r'''
set_phase('setup')
DATA_DIR = resolve_input([INPUT_ROOT / 'competitions/gemma-4-developer-agent',
                          INPUT_ROOT / 'gemma-4-developer-agent'], 'tasks.jsonl')
TASKS_PATH = DATA_DIR / 'tasks.jsonl'
WHEELHOUSE_DIR = resolve_input([INPUT_ROOT / 'datasets/metric/gemma-4-developer-agent-wheelhouse',
                               INPUT_ROOT / 'gemma-4-developer-agent-wheelhouse'],
                              'swegemma-0.2.11-py3-none-any.whl')
MODEL_PATH = resolve_input([INPUT_ROOT / 'models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2',
                            INPUT_ROOT / 'gemma-4-31b-it-qat-w4a16-ct/2'],
                           'tokenizer.json', 'gemma-4-31b-it-qat-w4a16-ct/2')
assert MODEL_PATH.name == '2' and 'gemma-4-31b-it-qat-w4a16-ct' in str(MODEL_PATH)
tokenizer = MODEL_PATH / 'tokenizer.json'
assert tokenizer.stat().st_size == MODEL_PIN['tokenizer_bytes']
assert sha256_file(tokenizer) == MODEL_PIN['tokenizer_sha256']
assert not AGENT_ROOT.exists()
AGENT_ROOT.mkdir(parents=True)
FROZEN_DIR = RUN_ROOT / 'frozen_bundle'
FROZEN_DIR.mkdir()
atomic_text(FROZEN_DIR / 'experiment_manifest.json', MANIFEST_TEXT)
atomic_json(FROZEN_DIR / 'embedded_config_sha256.json', EXPECTED_CONFIG_HASHES)
atomic_json(FROZEN_DIR / 'model_pin.json', MODEL_PIN)
for variant, file_map in BUNDLE.items():
    for relative, content in file_map.items():
        assert hashlib.sha256(content.encode()).hexdigest() == EXPECTED_CONFIG_HASHES[variant][relative]
        atomic_text(FROZEN_DIR / 'variants' / variant / relative, content)
        atomic_text(AGENT_ROOT / variant / relative, content)
schedule = []
for slot, task_id in enumerate(TASK_IDS, 1):
    order = ('t0', 'single_state') if slot % 2 else ('single_state', 't0')
    for launch_order, variant in enumerate(order, 1):
        ordinal = len(schedule) + 1
        attempt_id = f'{ordinal:02d}-{task_id}-{variant}'
        schedule.append({'ordinal': ordinal, 'task_slot': slot, 'within_pair_launch_order': launch_order,
                         'task_id': task_id, 'variant': variant, 'attempt_id': attempt_id,
                         'results_dir': str(RUN_ROOT / variant / REPEAT_ID / 'attempts' / attempt_id),
                         'config_sha256': EXPECTED_CONFIG_HASHES[variant]})
full_schedule = schedule
state = {entry['attempt_id']: dict(entry, status='planned') for entry in schedule}
atomic_json(RUN_ROOT / 'full_schedule.json', schedule)
atomic_json(RUN_ROOT / 'attempt_status.json', state)
'''

CONFIG_CHECK = r'''
common = EXPERIMENT['common']
budgets = common['eval_budgets']
common_sampling = common['sampling']
for variant in SELECTED_VARIANTS:
    config = yaml.safe_load(BUNDLE[variant]['agent.yaml'])
    assert config['tools'] == common['tool_list'] and 'adapter' not in config
    assert yaml.safe_load(BUNDLE[variant]['configs/sampling.yaml']) == common_sampling
    assert yaml.safe_load(BUNDLE[variant]['eval_config.yaml'])['evaluation'] == budgets
    assert (config.get('skills') or []) == ([] if variant == 't0' else [
        'skills/source-edit', 'skills/working-state', 'skills/source-probe'])
    assert discover_declared_models(AGENT_ROOT / variant) == {TARGET_MODEL_NAME}
    assert not any(path.is_symlink() for path in (AGENT_ROOT / variant).rglob('*'))
'''

TASK_SUBSET = r'''
# Hash the complete source as opaque bytes; parse task objects for the four
# approved development IDs only. Never archive the full input JSONL.
import re
digest = hashlib.sha256()
tasks_size = 0
selected_lines = {}
id_pattern = re.compile(rb'(?<!\\)"instance_id"\s*:\s*("(?:[^"\\]|\\.)*")')
with TASKS_PATH.open('rb') as stream:
    for line in stream:
        digest.update(line)
        tasks_size += len(line)
        match = id_pattern.search(line)
        if match is None:
            raise ValueError('Missing instance_id in source JSONL row')
        identifier = json.loads(match.group(1))
        if identifier in TASK_IDS:
            assert identifier not in selected_lines, 'Duplicate approved task ID'
            selected_lines[identifier] = line.rstrip(b'\r\n') + b'\n'
tasks_sha = digest.hexdigest()
assert tasks_sha == EXPERIMENT['data']['source_sha256'], 'tasks.jsonl changed; stop before inference'
assert tasks_size == EXPERIMENT['data']['tasks_bytes'], 'tasks.jsonl size mismatch'
assert set(selected_lines) == set(TASK_IDS), 'Missing approved development task'
TASKS_SUBSET_PATH = FROZEN_DIR / 'tasks-selected.jsonl'
TASKS_SUBSET_PATH.write_bytes(b''.join(selected_lines[identifier] for identifier in TASK_IDS))
tasks_subset_sha = sha256_file(TASKS_SUBSET_PATH)
tasks = load_tasks(TASKS_SUBSET_PATH)
by_id = {task.instance_id: task for task in tasks}
assert len(by_id) == len(TASK_IDS) and set(by_id) == set(TASK_IDS)
atomic_json(FROZEN_DIR / 'task-source.json', {'source_sha256': tasks_sha, 'source_bytes': tasks_size,
            'selected_sha256': tasks_subset_sha, 'selected_bytes': TASKS_SUBSET_PATH.stat().st_size,
            'selected_task_ids': list(TASK_IDS)})
'''

RUN_PAIRS = r'''
results_by_variant = {variant: [] for variant in SELECTED_VARIANTS}
analysis_rows = []
batch_start = time.monotonic()
active = []
failure = None
worker_path = RUN_ROOT / 'attempt_worker.py'
atomic_text(worker_path, WORKER_SOURCE)


def collect(entry, process, log, started, forced=None):
    log.close()
    attempt = Path(entry['results_dir'])
    status_path = attempt / 'attempt_status.json'
    item = json.loads(status_path.read_text()) if status_path.exists() else dict(entry, status='error')
    item.update(exit_code=process.returncode, coordinator_outer_seconds=time.monotonic() - started)
    if forced or process.returncode != 0:
        item.update(status=forced or 'error')
    result_path = attempt / 'result.json'
    if not result_path.exists():
        atomic_json(result_path, {'instance_id': entry['task_id'], 'resolved': None,
                                 'error': forced or f'worker exited {process.returncode} before result'})
    result = json.loads(result_path.read_text())
    state[entry['attempt_id']] = item
    atomic_json(status_path, item)
    analysis_rows.append({**entry, **result, 'status': item['status'], 'exit_code': process.returncode})
    atomic_json(RUN_ROOT / 'analysis_rows.json', analysis_rows)
    atomic_json(RUN_ROOT / 'attempt_status.json', state)
    files = {str(path.relative_to(attempt)): {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
             for path in attempt.rglob('*') if path.is_file() and path.name != 'artifact_manifest.json'}
    atomic_json(attempt / 'artifact_manifest.json', files)
    print(json.dumps({'attempt_id': entry['attempt_id'], 'status': item['status'],
                      'resolved': result.get('resolved'), 'time': time.time()}), flush=True)


try:
    set_phase('server_start')
    run_manifest['server_start_count'] = 1
    atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
    startup = time.monotonic()
    server.start()
    run_manifest['server_startup_seconds'] = time.monotonic() - startup
    run_manifest['server_base_url'] = str(server.base_url)
    atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
    for pair_start in range(0, len(schedule), 2):
        pair = schedule[pair_start:pair_start + 2]
        # Each wave contains both arms on the same task. AB/BA launch order is
        # balanced; scheduling and prefix-cache behavior are observed, not equal.
        set_phase('evaluation', task_slot=pair[0]['task_slot'], task_id=pair[0]['task_id'])
        for entry in pair:
            attempt = Path(entry['results_dir'])
            attempt.mkdir(parents=True, exist_ok=False)
            scratch = Path('/tmp') / RUN_ID / entry['attempt_id']
            scratch.mkdir(parents=True, exist_ok=False)
            agent = scratch / 'agent'
            shutil.copytree(AGENT_ROOT / entry['variant'], agent)
            specification = {'entry': entry, 'data_dir': str(DATA_DIR), 'agent_dir': str(agent),
                             'tasks_subset': str(TASKS_SUBSET_PATH),
                             'scratch': str(scratch), 'base_url': str(server.base_url),
                             'served_model': str(getattr(server, 'effective_model_path', MODEL_PATH)),
                             'model_alias': TARGET_MODEL_NAME, 'budgets': budgets}
            spec_path = attempt / 'worker_spec.json'
            atomic_json(spec_path, specification)
            env = dict(os.environ, TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch),
                       PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                       TOKENIZERS_PARALLELISM='false')
            log = (attempt / 'worker.log').open('w')
            process = subprocess.Popen([sys.executable, '-u', str(worker_path), str(spec_path)],
                                       cwd=scratch, env=env, stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            active.append((entry, process, log, started))
            state[entry['attempt_id']].update(status='running', pid=process.pid, started_at=time.time())
            atomic_json(attempt / 'attempt_manifest.json', state[entry['attempt_id']])
            atomic_json(RUN_ROOT / 'attempt_status.json', state)
            print(f'Launched {entry["attempt_id"]} pid={process.pid}', flush=True)
        assert len(active) <= MAX_CONCURRENT
        while active:
            for record in active[:]:
                entry, process, log, started = record
                forced = None
                if process.poll() is None and time.monotonic() - started >= ATTEMPT_WALL_SECONDS:
                    forced = 'attempt_wall_timeout'
                    stop_process_tree(process)
                if process.poll() is not None:
                    collect(entry, process, log, started, forced)
                    active.remove(record)
            if active:
                time.sleep(1)
except BaseException as exc:
    failure = exc
    atomic_text(RUN_ROOT / 'batch_exception.txt', traceback.format_exc())
    if getattr(exc, 'output', None):
        atomic_text(RUN_ROOT / 'server_exception_output.log', str(exc.output))
finally:
    set_phase('cleanup')
    for entry, process, log, started in active:
        stop_process_tree(process)
        collect(entry, process, log, started, 'interrupted')
    try:
        server.stop(timeout=10.0)
    except Exception:
        atomic_text(RUN_ROOT / 'server_stop_error.txt', traceback.format_exc())
    for item in state.values():
        if item['status'] == 'planned':
            item['status'] = 'not_run'
    run_manifest['batch_wall_seconds'] = time.monotonic() - batch_start
    atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
    atomic_json(RUN_ROOT / 'attempt_status.json', state)
    summary = {variant: {'planned': 4,
        'completed': sum(s['variant'] == variant and s['status'] == 'completed' for s in state.values()),
        'resolved': sum(row['variant'] == variant and row.get('resolved') is True for row in analysis_rows),
        'states': {s['task_id']: s['status'] for s in state.values() if s['variant'] == variant}}
        for variant in SELECTED_VARIANTS}
    atomic_json(RUN_ROOT / 'joint_summary.json', summary)
    print('FINAL ' + json.dumps(summary), flush=True)
if failure is not None:
    raise failure
if any(item['status'] != 'completed' for item in state.values()):
    raise RuntimeError('Some attempts are incomplete; retained evidence is partial, no automatic retry')
set_phase('completed')
'''

SUPERVISOR = r'''
# Execution starts here. This cell supervises a fresh interpreter containing the
# entire installation/server/evaluation lifecycle, including a hard wall bound.
from pathlib import Path
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
import uuid
import zipfile

WALL_SECONDS = 2550  # Includes pip installation, model loading and compilation.
RUN_ID = 'run-005-t0-single-state-pair-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
RUN_ROOT = Path('/kaggle/working/results/005') / RUN_ID
RUN_ROOT.mkdir(parents=True, exist_ok=False)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sample():
    row = {'time': time.time(), 'elapsed_seconds': time.monotonic() - started}
    phase_path = RUN_ROOT / 'phase.json'
    try:
        row.update(json.loads(phase_path.read_text()))
        row['phase_at'] = row.pop('time')
        row['time'] = time.time()
    except (OSError, ValueError):
        row['phase'] = 'setup'
    try:
        response = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw',
                                   '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        row['gpu_csv'] = response.stdout.strip()
        row['nvidia_smi_exit_code'] = response.returncode
        row['gpus'] = []
        for line in response.stdout.splitlines():
            index, name, util, used, total, power = [part.strip() for part in line.split(',')]
            row['gpus'].append({'index': int(index), 'name': name, 'utilization_percent': float(util),
                                'memory_used_mib': float(used), 'memory_total_mib': float(total),
                                'power_w': None if power == '[N/A]' else float(power)})
    except Exception as exc:
        row['gpu_sampling_error'] = str(exc)
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/metrics', timeout=2) as response:
            metrics = response.read().decode()
        prefixes = ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:gpu_cache_usage_perc',
                    'vllm:kv_cache_usage_perc', 'vllm:generation_tokens_total', 'vllm:prompt_tokens_total')
        selected = [line for line in metrics.splitlines() if line.startswith(prefixes)]
        row['vllm_metrics'] = selected
        row['vllm_requests_running'] = sum(float(line.rsplit(' ', 1)[1]) for line in selected
                                          if line.startswith('vllm:num_requests_running'))
        row['vllm_requests_waiting'] = sum(float(line.rsplit(' ', 1)[1]) for line in selected
                                          if line.startswith('vllm:num_requests_waiting'))
    except Exception as exc:
        row['vllm_metrics_unavailable'] = str(exc)
    with (RUN_ROOT / 'gpu-telemetry.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
        stream.flush()
    return row


if sys.version_info[:2] != (3, 12) or platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
    raise RuntimeError('Use the pinned Starter Linux CPython 3.12 image; no packages or model loaded')
runner = RUN_ROOT / 'coordinator.py'
runner.write_text(COORDINATOR_SOURCE)
write_json(RUN_ROOT / 'carrier-source.json', SOURCE_MANIFEST)
env = dict(os.environ, P005_RUN_ID=RUN_ID, P005_RUN_ROOT=str(RUN_ROOT), PYTHONUNBUFFERED='1')
started = time.monotonic()
process = None
timed_out = False
error = None
rows = []
try:
    with (RUN_ROOT / 'coordinator.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', str(runner)], env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        offset = 0
        next_sample = 0.0
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= WALL_SECONDS:
                timed_out = True
                break
            if elapsed >= next_sample:
                rows.append(sample())
                next_sample = time.monotonic() - started + 10
                with (RUN_ROOT / 'coordinator.log').open() as follow:
                    follow.seek(offset)
                    text = follow.read()
                    offset = follow.tell()
                if text:
                    print(text[-16000:], end='', flush=True)
            time.sleep(min(1, max(0, WALL_SECONDS - (time.monotonic() - started))))
except BaseException:
    error = traceback.format_exc()
    (RUN_ROOT / 'supervisor_exception.txt').write_text(error)
finally:
    if process is not None:
        # Includes server processes, both Evaluators, and separately sessioned
        # sandbox descendants. No retry or second model startup.
        stop_process_tree(process, own_group=True)
    result_path = RUN_ROOT / 'attempt_status.json'
    states = json.loads(result_path.read_text()) if result_path.exists() else {}
    for attempt_id, state in states.items():
        local = Path(state['results_dir']) / 'attempt_status.json'
        if local.is_file():
            states[attempt_id] = json.loads(local.read_text())
        if states[attempt_id].get('status') in ('running', 'initializing', 'planned'):
            states[attempt_id]['status'] = 'outer_wall_timeout' if timed_out else 'interrupted_or_not_run'
    write_json(RUN_ROOT / 'attempt_status.json', states)
    complete = len(states) == 8 and all(item.get('status') == 'completed' for item in states.values())
    ok = process is not None and process.returncode == 0 and complete and not timed_out and error is None
    report = {'run_id': RUN_ID, 'status': 'completed' if ok else 'partial_or_failed',
              'exit_code': None if process is None else process.returncode, 'timed_out': timed_out,
              'wall_seconds': time.monotonic() - started, 'hard_worker_limit_seconds': WALL_SECONDS,
              'attempt_states': states, 'completed_attempts': sum(s.get('status') == 'completed' for s in states.values()),
              'configured_max_concurrent_attempts': 2, 'image_pin': SOURCE_MANIFEST['image_pin'],
              'quota_note': 'Kaggle quota accounting is external; actual wall and GPU samples are retained.'}
    # Averages are sampled utilization, never throughput or complete GPU busy time.
    report['sampled_phase_summary'] = {}
    for phase in sorted({row['phase'] for row in rows}):
        selected = [row for row in rows if row['phase'] == phase]
        by_gpu = {}
        for row in selected:
            for gpu in row.get('gpus', []):
                by_gpu.setdefault(str(gpu['index']), []).append(gpu['utilization_percent'])
        report['sampled_phase_summary'][phase] = {
            'samples': len(selected), 'gpu_mean_utilization_percent': {k: sum(v) / len(v) for k, v in by_gpu.items()},
            'vllm_samples_available': sum('vllm_requests_running' in row for row in selected),
            'maximum_sampled_requests_running': max((row['vllm_requests_running'] for row in selected
                                                     if 'vllm_requests_running' in row), default=None)}
    write_json(RUN_ROOT / 'supervisor-report.json', report)
    # A hard-killed official server cannot run stop(); retain its original log.
    locator = RUN_ROOT / 'server-log-path.json'
    output_deadline = started + 2680
    if locator.is_file():
        source = Path(json.loads(locator.read_text())['path'])
        if source.is_file():
            with source.open('rb') as incoming, (RUN_ROOT / 'vllm_server.archived.log').open('wb') as outgoing:
                while time.monotonic() < output_deadline:
                    chunk = incoming.read(1024 * 1024)
                    if not chunk:
                        break
                    outgoing.write(chunk)
    archive_path = RUN_ROOT.parent / (RUN_ID + '.zip')
    try:
        # Stored ZIP avoids compression using the remaining output budget.
        # Original files remain complete even if archive time is exhausted.
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_STORED) as archive:
            for path in sorted(RUN_ROOT.rglob('*')):
                if path.is_file():
                    with path.open('rb') as incoming, archive.open(str(path.relative_to(RUN_ROOT)), 'w') as outgoing:
                        while True:
                            if time.monotonic() >= output_deadline:
                                raise TimeoutError('Archive budget exhausted; use original per-file outputs')
                            chunk = incoming.read(1024 * 1024)
                            if not chunk:
                                break
                            outgoing.write(chunk)
    except Exception as exc:
        report['archive_error'] = str(exc)
        write_json(RUN_ROOT / 'supervisor-report.json', report)
    print(json.dumps({'status': report['status'], 'output': str(RUN_ROOT), 'archive': str(archive_path),
                      'completed_attempts': report['completed_attempts'], 'wall_seconds': report['wall_seconds']}), flush=True)
if not ok:
    raise RuntimeError('Paired experiment incomplete; see retained supervisor report, logs and per-attempt evidence')
'''


def render():
    baseline = single.BASE.read_text()
    t0 = {}
    for path in sorted((ROOT / 'research/002/variants/t0/agent').rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlink in frozen T0: {path}')
        if path.is_file() and '__pycache__' not in path.parts:
            t0[path.relative_to(ROOT / 'research/002/variants/t0/agent').as_posix()] = path.read_text()
    original_bundle = json.loads(ast.literal_eval(
        next(n.value.args[0] for n in ast.parse(baseline).body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'BUNDLE' for t in n.targets))))
    if t0 != original_bundle['t0']:
        raise ValueError('Frozen T0 differs from its v30 carrier; do not silently relabel a changed baseline')
    bundle = {'t0': t0, **single.collect_bundle()}
    hashes = {arm: {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}
              for arm, files in bundle.items()}
    original = json.loads(single.literal_assignment(baseline, 'MANIFEST_TEXT'))
    common = dict(original['common'])
    common.pop('single_variable', None)
    manifest = {'study': 'P005 T0 vs whole single_state package, four exposed development tasks',
                'common': common, 'data': original['data'], 'task_ids': TASKS,
                'orders': ['AB', 'BA', 'AB', 'BA'], 'max_concurrent_attempts': 2,
                'research_condition': 'Whole single_state package (prompt + source-edit/working-state/source-probe skills); not a state-card-only ablation. Six official tools match; candidate also receives official SkillToolset tools.',
                'interpretation': 'Four previously exposed development tasks, one pair each; exploratory mechanism evidence, not independent evaluation or established improvement.',
                'process_isolation': 'Separate Evaluator interpreters, submission copies, tmp/sandbox directories and artifacts; one shared vLLM server. No cross-arm result injection.',
                'cache_note': 'Shared prefix cache and concurrent scheduling are not cache-equivalent; AB/BA launch order and live concurrency are retained.',
                'server_lora': 'Explicitly disabled for both arms; official v30 also omits LoRA CLI flags when no modules are mounted, so no speedup is attributed to this setting.',
                'server_max_num_seqs': 2,
                'accelerator': 'NvidiaL4 (4 physical L4 GPUs, quota factor 2)',
                'image_pin': IMAGE, 'wall_limit_seconds': 2550, 'account_quota_ceiling_hours': 2,
                'source_carriers': ['notebooks/builders/build_005_single_state.py', str(single.BASE.relative_to(ROOT))]}
    model_pin = {'ref': 'google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2',
                 'tokenizer_bytes': single.MODEL_TOKENIZER_BYTES, 'tokenizer_sha256': single.MODEL_TOKENIZER_SHA256}
    inventory = json.loads((ROOT / 'research/005/official-20261010/wheel-files.json').read_text())
    wheels = {item['name']: int(item['total_bytes']) for item in inventory['dataset_files']
              if item['name'].endswith('.whl') and 'cutlass' not in item['name'].lower()}
    helpers1 = baseline[baseline.index('def input_snapshot'):baseline.index('BUNDLE =')]
    helpers2 = baseline[baseline.index('def sha256_file'):baseline.index('FROZEN_DIR =')]
    worker = WORKER.replace("SPEC = json.loads", helpers2 + "\nSPEC = json.loads", 1)
    body = baseline[baseline.index('# Install before importing target packages.'):baseline.index('results_by_variant =')]
    cutlass_start = body.index("for pattern in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth'")
    cutlass_end = body.index('assert WHEELHOUSE_DIR.is_dir()', cutlass_start)
    body = body[:cutlass_start] + body[cutlass_end:]
    body = single.replace_once(body, "assert source_wheels, 'Wheelhouse is empty'", "assert source_wheels, 'Wheelhouse is empty'\n" + single.WHEEL_CHECK)
    # v30 has no swegemma.models.discovery or validate_single_declared_model.
    # Use the current public discovery API, which returns all declared aliases.
    body = single.replace_once(body,
                               'from swegemma.models.discovery import validate_single_declared_model',
                               'from adk_submission.discovery import discover_declared_models')
    tasks_start = body.index('raw_tasks = TASKS_PATH.read_bytes()')
    tasks_end = body.index("common = EXPERIMENT['common']", tasks_start)
    body = body[:tasks_start] + TASK_SUBSET + '\n' + body[tasks_end:]
    body = single.replace_once(body, "'tasks_bytes': len(raw_tasks)", "'tasks_bytes': tasks_size, 'selected_tasks_sha256': tasks_subset_sha")
    start = body.index("common = EXPERIMENT['common']")
    end = body.index('eval_signature =', start)
    body = body[:start] + CONFIG_CHECK + '\n' + body[end:]
    body = single.replace_once(body, "'single_variable': 'skill only (T1 = T0 + skills/repo_navigation)',", "'research_condition': EXPERIMENT['research_condition'],")
    body = single.replace_once(body, "'model_requests': 'sequential'", "'model_requests': 'two isolated concurrent Evaluators; sampled via /metrics'")
    body = body.replace('submission-002-explore-', 'agent-bundle-005-pair-')
    body = single.replace_once(body, 'enable_lora=True, max_loras=8, max_lora_rank=128,',
                               "enable_lora=False, extra_args=['--max-num-seqs', '2'],")
    # Process-safe workers use HTTP only. The inference server remains in the
    # coordinator and is instantiated exactly once; no models are pickled.
    body += "\natomic_json(RUN_ROOT / 'server-log-path.json', {'path': str(server.log_path)})\n"
    body += "run_manifest['server_command'] = server.build_cmd()\natomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)\n"
    imports = baseline[baseline.index('from pathlib import Path'):baseline.index('# Change only this list')]
    imports += '\nimport signal\n'
    constants = ('BUNDLE = json.loads(' + repr(json.dumps(bundle, ensure_ascii=False)) + ')\n'
                 + 'EXPECTED_CONFIG_HASHES = ' + repr(hashes) + '\n'
                 + 'MANIFEST_TEXT = ' + repr(json.dumps(manifest, ensure_ascii=False)) + '\n'
                 + 'EXPERIMENT = json.loads(MANIFEST_TEXT)\n'
                 + 'TASK_IDS = ' + repr(TASKS) + '\nMODEL_PIN = ' + repr(model_pin) + '\n'
                 + 'WHEELHOUSE_V30_FILES = ' + repr(wheels) + '\nWORKER_SOURCE = ' + repr(worker) + '\n')
    coordinator = imports + single.RUNTIME_CHECK + SCHEDULE + helpers1 + helpers2 + constants + SETUP + PROCESS_HELPERS + body + RUN_PAIRS
    for name, text in (('worker', worker), ('coordinator', coordinator)):
        ast.parse(text, filename=name)
    source_manifest = {'image_pin': IMAGE, 'coordinator_sha256': hashlib.sha256(coordinator.encode()).hexdigest(),
                       'worker_sha256': hashlib.sha256(worker.encode()).hexdigest(), 'task_ids': TASKS,
                       'config_sha256': hashes, 'official_wheelhouse_version': 30, 'model_pin': model_pin,
                       'build_sources': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in (Path(__file__), Path(single.__file__), single.BASE)}}
    header = '# Generated by notebooks/builders/build_005_t0_single_state_pair.py; executing starts the bounded GPU experiment.\n'
    # Function names resolve at execution time; helpers must precede supervisor's run loop.
    insertion = SUPERVISOR.index("if sys.version_info[:2]")
    supervisor = SUPERVISOR[:insertion] + PROCESS_HELPERS + '\n' + SUPERVISOR[insertion:]
    source = header + 'COORDINATOR_SOURCE = ' + repr(coordinator) + '\nSOURCE_MANIFEST = ' + repr(source_manifest) + '\n' + supervisor
    ast.parse(source)
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3.12', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python', 'version': '3.12'}},
                'cells': [{'cell_type': 'markdown', 'id': 'p005-paired-intro', 'metadata': {}, 'source': [
                    '# P005 T0 vs single_state — four exposed development pairs\n',
                    'One CT v2 vLLM service, wheelhouse v30, pinned Starter Python 3.12; two isolated Evaluator subprocesses.\n',
                    'Executing the cell starts GPU work. Agent budget: 5 minutes each; whole lifecycle hard limit: 2550 seconds plus bounded cleanup/output.\n',
                    'The candidate is an entire prompt/skill package, not a state-card-only ablation. No competition submission is performed.\n']},
                          {'cell_type': 'code', 'id': 'p005-paired-run', 'metadata': {}, 'execution_count': None,
                           'outputs': [], 'source': source.splitlines(keepends=True)}]}
    return source, json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', source_manifest


def main():
    source, notebook, manifest = render()
    Path(str(OUT) + '-single-cell.py').write_text(source)
    Path(str(OUT) + '.ipynb').write_text(notebook)
    print(json.dumps({'python': str(OUT.relative_to(ROOT)) + '-single-cell.py',
                      'notebook': str(OUT.relative_to(ROOT)) + '.ipynb', 'task_ids': TASKS,
                      'coordinator_sha256': manifest['coordinator_sha256'],
                      'source_sha256': hashlib.sha256(source.encode()).hexdigest()}))


if __name__ == '__main__':
    main()
