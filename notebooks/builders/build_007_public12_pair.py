#!/usr/bin/env python3
"""Build P007 Public-12 paired evaluation: public022_v10 vs behavior_v1.

Uses stdlib only at build time. Inherits the proven P005/P006 single-server
supervisor, wheelhouse v31 installer, and isolated worker architecture.
Maximizes 4xL4 GPU utilization via a rolling pair dispatcher (MAX_CONCURRENT=4,
vLLM --max-num-seqs=4) that always launches both arms of a task simultaneously
while immediately refilling freed pair slots without wave-barrier stalls.
"""
import ast
import hashlib
import json
from pathlib import Path

import build_006_reference_pair as p006

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'notebooks/007/007-public12-pair'
KERNEL = ROOT / 'research/007/public12-pair-20261017/kernel'
CANDIDATES = ROOT / 'research/007/candidates'
DEV12_MANIFEST = ROOT / 'research/007/evaluation-design-20261017/public-dev12-manifest.json'
LICENSE_SOURCE = ROOT / 'research/006/high-score-notebooks-20261013/extracted/dmitriigluzdov__gemma-4-task-soundness-serving-and-evaluation.json'
ARMS = ('public022_v10', 'behavior_v1')
IMAGE = p006.IMAGE
OFFICIAL_TASKS_SHA256 = 'e4b3fd60f69dbc2b9213e54eeb9636db78aefe92c1d06269d73d9f5f8f3c8ad6'
OFFICIAL_TASKS_BYTES = 1984455


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one inherited source anchor, got {source.count(old)}: {old[:100]}')
    return source.replace(old, new, 1)


def assignment(source, name, value):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            lines = source.splitlines(keepends=True)
            return ''.join(lines[:node.lineno - 1]) + name + ' = ' + repr(value) + '\n' + ''.join(lines[node.end_lineno:])
    raise ValueError('Missing inherited assignment: ' + name)


SCHEDULE = r'''
RUN_ID = os.environ['P007_RUN_ID']
RUN_ROOT = Path(os.environ['P007_RUN_ROOT'])
AGENT_ROOT = Path('/kaggle/working/agent_exec') / RUN_ID
INPUT_ROOT = Path('/kaggle/input')
WORKING_DIR = Path('/kaggle/working')
TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'
SELECTED_VARIANTS = ('public022_v10', 'behavior_v1')
REPEAT_ID = 'r1'
MAX_CONCURRENT = 4
ATTEMPT_WALL_SECONDS = 1020  # 480 agent + 300 judge + up to 240 setup/cleanup.
VERIFICATION_TIMEOUT_SECONDS = 300
PREPARE_ONLY = os.environ.get('P007_PREPARE_ONLY') == '1'
EVALUATION_DEADLINE = float(os.environ['P007_EVALUATION_DEADLINE'])


def set_phase(name, **extra):
    atomic_json(RUN_ROOT / 'phase.json', {'phase': name, 'time': time.time(), **extra})

'''

SCHEDULE_LOOP = r'''
schedule = []
for slot, task_id in enumerate(TASK_IDS, 1):
    order = SELECTED_VARIANTS if slot % 2 else tuple(reversed(SELECTED_VARIANTS))
    for launch_order, variant in enumerate(order, 1):
        ordinal = len(schedule) + 1
        attempt_id = f'{ordinal:02d}-{task_id}-{REPEAT_ID}-{variant}'
        schedule.append({'ordinal': ordinal, 'task_slot': slot, 'repeat_id': REPEAT_ID,
                         'within_pair_launch_order': launch_order, 'task_id': task_id,
                         'variant': variant, 'attempt_id': attempt_id,
                         'results_dir': str(RUN_ROOT / variant / REPEAT_ID / 'attempts' / attempt_id),
                         'config_sha256': EXPECTED_CONFIG_HASHES[variant]})
full_schedule = list(schedule)
assert len(full_schedule) == 24
if PREPARE_ONLY:
    schedule = schedule[:2]  # Construct both official agents; no model or judge.
state = {entry['attempt_id']: dict(entry, status='planned') for entry in schedule}
atomic_json(RUN_ROOT / 'full_schedule.json', full_schedule)
atomic_json(RUN_ROOT / 'attempt_status.json', state)
'''

TASK_SUBSET = r'''
# Hash the complete source as opaque bytes; extract ONLY the 12 frozen dev_future
# task rows. Never archive the full tasks.jsonl or read heldout/reserved tasks.
import re
TASKS_PATH = DATA_DIR / 'tasks.jsonl'
assert TASKS_PATH.is_file(), f'Missing tasks.jsonl under {DATA_DIR}'
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
            assert identifier not in selected_lines, f'Duplicate approved task ID: {identifier}'
            selected_lines[identifier] = line.rstrip(b'\r\n') + b'\n'
tasks_sha = digest.hexdigest()
assert tasks_sha == EXPERIMENT['data']['source_sha256'], 'tasks.jsonl changed; stop before inference'
assert tasks_size == EXPERIMENT['data']['source_bytes'], 'tasks.jsonl size mismatch'
assert set(selected_lines) == set(TASK_IDS), f'Missing approved task IDs: {set(TASK_IDS) - set(selected_lines)}'
TASKS_SUBSET_PATH = FROZEN_DIR / 'tasks-selected.jsonl'
subset_bytes = b''.join(selected_lines[identifier] for identifier in TASK_IDS)
TASKS_SUBSET_PATH.write_bytes(subset_bytes)
tasks_subset_sha = sha256_file(TASKS_SUBSET_PATH)
tasks = load_tasks(TASKS_SUBSET_PATH)
by_id = {task.instance_id: task for task in tasks}
assert len(tasks) == 12 and set(by_id) == set(TASK_IDS)
from swegemma.deduplication import resolve_task_snapshot_paths
snapshot_verification = {}
for task in tasks:
    assert task.patch and task.test_patch, f'Frozen public task must be complete: {task.instance_id}'
    expected_meta = EXPECTED_TASK_META[task.instance_id]
    assert task.repo == expected_meta['repo'] and task.base_commit == expected_meta['base_commit'], (
        f'Task metadata mismatch for {task.instance_id}: {task.repo}@{task.base_commit}')
    snapshot, base_snapshot, patch_file = resolve_task_snapshot_paths(
        DATA_DIR / 'snapshots', task.instance_id, task.repo)
    assert snapshot.is_file(), f'Missing selected snapshot: {task.instance_id}'
    assert base_snapshot is None or base_snapshot.is_file()
    assert patch_file is None or patch_file.is_file()
    snapshot_verification[task.instance_id] = {
        'repo': task.repo, 'base_commit': task.base_commit,
        'snapshot_name': snapshot.name, 'snapshot_bytes': snapshot.stat().st_size}
atomic_json(FROZEN_DIR / 'task-source.json', {
    **EXPERIMENT['data'], 'selected_sha256': tasks_subset_sha,
    'selected_bytes': len(subset_bytes), 'verified_snapshots': snapshot_verification})
'''

CONFIG_CHECK = r'''
common = EXPERIMENT['common']
for variant in SELECTED_VARIANTS:
    condition = VARIANT_CONDITIONS[variant]
    config = yaml.safe_load(BUNDLE[variant]['agent.yaml'])
    assert 'adapter' not in config and config['tools'] == condition['tools']
    assert yaml.safe_load(BUNDLE[variant]['configs/sampling.yaml']) == condition['sampling']
    assert yaml.safe_load(BUNDLE[variant]['eval_config.yaml'])['evaluation'] == condition['budgets']
    assert condition['budgets']['max_time_minutes'] == 8
    assert (config.get('skills') or []) == condition['skills']
    assert discover_declared_models(AGENT_ROOT / variant) == {TARGET_MODEL_NAME}
    assert not any(path.is_symlink() for path in (AGENT_ROOT / variant).rglob('*'))
assert 'verification_timeout_seconds' in inspect.signature(EvalConfig).parameters
'''

RUN_BODY = r'''
try:
    if not PREPARE_ONLY:
        set_phase('server_start')
        run_manifest['server_start_count'] = 1
        atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
        startup = time.monotonic()
        server.start()
        run_manifest['server_startup_seconds'] = time.monotonic() - startup
        run_manifest['server_base_url'] = str(server.base_url)
        atomic_json(RUN_ROOT / 'run_manifest.json', run_manifest)
    pair_queue = [schedule[i:i + 2] for i in range(0, len(schedule), 2)]
    while pair_queue or active:
        if pair_queue and time.monotonic() >= EVALUATION_DEADLINE:
            raise TimeoutError('Whole-run evaluation deadline; keep partial evidence, no retry')
        while pair_queue and len(active) + len(pair_queue[0]) <= MAX_CONCURRENT:
            if time.monotonic() >= EVALUATION_DEADLINE:
                break
            pair = pair_queue.pop(0)
            set_phase('cpu_prepare' if PREPARE_ONLY else 'evaluation',
                      task_slot=pair[0]['task_slot'], task_id=pair[0]['task_id'],
                      repeat_id=pair[0]['repeat_id'],
                      active_workers=len(active) + len(pair),
                      remaining_pairs=len(pair_queue))
            for entry in pair:
                attempt = Path(entry['results_dir'])
                attempt.mkdir(parents=True, exist_ok=False)
                scratch = Path('/tmp') / RUN_ID / entry['attempt_id']
                scratch.mkdir(parents=True, exist_ok=False)
                agent = scratch / 'agent'
                shutil.copytree(AGENT_ROOT / entry['variant'], agent)
                specification = {'entry': entry, 'data_dir': str(DATA_DIR), 'agent_dir': str(agent),
                                 'tasks_subset': str(TASKS_SUBSET_PATH), 'scratch': str(scratch),
                                 'base_url': 'http://127.0.0.1:8000/v1' if PREPARE_ONLY else str(server.base_url),
                                 'served_model': str(MODEL_PATH) if PREPARE_ONLY else str(getattr(server, 'effective_model_path', MODEL_PATH)),
                                 'model_alias': TARGET_MODEL_NAME,
                                 'budgets': VARIANT_CONDITIONS[entry['variant']]['budgets'],
                                 'prepare_only': PREPARE_ONLY, 'total_tasks': 24,
                                 'verification_timeout_seconds': VERIFICATION_TIMEOUT_SECONDS}
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
                print(f'Launched {entry["attempt_id"]} pid={process.pid} active={len(active)}', flush=True)
            assert len(active) <= MAX_CONCURRENT
        for record in active[:]:
            entry, process, log, started = record
            forced = None
            if process.poll() is None:
                if time.monotonic() >= EVALUATION_DEADLINE:
                    forced = 'whole_run_deadline'
                elif time.monotonic() - started >= ATTEMPT_WALL_SECONDS:
                    forced = 'attempt_wall_timeout'
                if forced:
                    stop_process_tree(process)
            if process.poll() is not None:
                collect(entry, process, log, started, forced)
                active.remove(record)
        if active and (not pair_queue or len(active) + len(pair_queue[0]) > MAX_CONCURRENT):
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
    if server is not None:
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
    summary = {variant: {
        'planned': sum(entry['variant'] == variant for entry in full_schedule),
        'mode': 'cpu_prepare' if PREPARE_ONLY else 'evaluate',
        'prepared': sum(s['variant'] == variant and s['status'] == 'prepared' for s in state.values()),
        'completed': sum(s['variant'] == variant and s['status'] == 'completed' for s in state.values()),
        'resolved': None if PREPARE_ONLY else sum(row['variant'] == variant and row.get('resolved') is True for row in analysis_rows),
        'states': {s['attempt_id']: s['status'] for s in state.values() if s['variant'] == variant}}
        for variant in SELECTED_VARIANTS}
    atomic_json(RUN_ROOT / 'joint_summary.json', summary)
    print('FINAL ' + json.dumps(summary), flush=True)
if failure is not None:
    raise failure
expected_state = 'prepared' if PREPARE_ONLY else 'completed'
if any(item['status'] != expected_state for item in state.values()):
    raise RuntimeError('Some attempts are incomplete; retain partial evidence, no automatic retry')
set_phase('prepared' if PREPARE_ONLY else 'completed')
'''


def load_candidate(name):
    manifest_path = CANDIDATES / name / 'source-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    files = {}
    for rel, expected in manifest['files'].items():
        path = CANDIDATES / name / 'agent' / rel
        raw = path.read_bytes()
        assert len(raw) == expected['bytes'], f'Size mismatch {name}/{rel}'
        assert sha(raw) == expected['sha256'], f'SHA mismatch {name}/{rel}'
        files[rel] = raw.decode('utf-8')
    assert set(files) == {'agent.yaml', 'configs/sampling.yaml', 'eval_config.yaml', 'prompts/system.md'}
    return files, manifest


def frozen_inputs():
    a_files, a_manifest = load_candidate('public022_v10')
    b_files, b_manifest = load_candidate('behavior_v1')
    for rel in ('agent.yaml', 'configs/sampling.yaml', 'eval_config.yaml'):
        assert a_files[rel] == b_files[rel], f'Single-variable violation in {rel}'
    assert a_files['prompts/system.md'] != b_files['prompts/system.md']
    dev12 = json.loads(DEV12_MANIFEST.read_text())
    selected = dev12['selected_tasks']
    assert len(selected) == 12
    task_ids = tuple(item['instance_id'] for item in selected)
    assert len(set(task_ids)) == 12
    task_meta = {item['instance_id']: {'repo': item['repo'], 'base_commit': item['base_commit'], 'split': item['split']}
                 for item in selected}
    return {ARMS[0]: a_files, ARMS[1]: b_files}, {ARMS[0]: a_manifest, ARMS[1]: b_manifest}, task_ids, task_meta, dev12


def render():
    p006_source, _, p006_provenance, p006_manifest = p006.render()
    coordinator = p006.previous.single.literal_assignment(p006_source, 'COORDINATOR_SOURCE')
    worker = p006.previous.single.literal_assignment(coordinator, 'WORKER_SOURCE')
    bundle, arm_manifests, task_ids, task_meta, dev12 = frozen_inputs()
    reference_license = json.loads(LICENSE_SOURCE.read_text())['ISSUE_CONTRACT_FILES']['LICENSE.txt']
    hashes = {arm: {name: sha(text.encode()) for name, text in files.items()} for arm, files in bundle.items()}
    common_condition = {
        'tools': ['run_command', 'read_file', 'edit_file', 'write_file', 'get_status', 'submit_patch',
                  'get_code_neighbors', 'search_similar_code', 'get_code_subgraph'],
        'skills': [],
        'budgets': {'timeout_seconds': 240, 'max_tool_calls': 28, 'max_time_minutes': 8, 'max_turns': 80},
        'sampling': {'temperature': 0.2, 'top_p': 0.95, 'max_output_tokens': 8192,
                     'thinking_config': {'thinking_budget': 4096, 'include_thoughts': True}},
    }
    conditions = {arm: dict(common_condition) for arm in ARMS}
    packages = dict(p006_manifest['common']['official_packages'])
    manifest = {
        'study': 'P007 Public-12 Paired Evaluation: exact public022_v10 vs single-variable behavioral verification (behavior_v1)',
        'common': {'official_packages': packages},
        'variant_conditions': conditions,
        'verification_timeout_seconds': 300,
        'task_ids': task_ids,
        'repeats': ['r1'],
        'planned_attempts': 24,
        'max_concurrent_attempts': 4,
        'server_max_num_seqs': 4,
        'image_pin': IMAGE,
        'research_condition': (
            'Single-variable prompt ablation on top of exact public022_v10 (8-minute / 28-call / 80-turn / 240s command timeout, '
            '31B CT v2, temp 0.2, top-p 0.95, output 8192, thinking 4096). Only behavioral verification instructions in '
            'prompts/system.md differ. Rolling pair queue launches both arms of each task simultaneously (balanced AB/BA '
            'across the 12 tasks and 2 AB / 2 BA within each repository) while keeping up to 2 pairs (4 workers) active '
            'to maximize 4xL4 GPU utilization.'
        ),
        'data': {
            'selection_manifest': str(DEV12_MANIFEST.relative_to(ROOT)),
            'selection_manifest_sha256': sha(DEV12_MANIFEST.read_bytes()),
            'source': 'tasks.jsonl',
            'source_sha256': OFFICIAL_TASKS_SHA256,
            'source_bytes': OFFICIAL_TASKS_BYTES,
            'selected_task_ids': task_ids,
            'selected_rows': 12,
            'heldout_or_reserved_accessed': False,
        },
        'source_arms': {
            'public022_v10': arm_manifests['public022_v10'],
            'behavior_v1': arm_manifests['behavior_v1'],
        },
        'whole_run': {
            'platform_session_limit_seconds': 5760,
            'evaluation_deadline_seconds': 5400,
            'coordinator_force_stop_seconds': 5520,
            'output_deadline_seconds': 5640,
            'internal_lifecycle_target_seconds': 5660,
            'quota_upper_estimate_hours': 3.2,
            'quota_note': 'Rolling 4-worker paired evaluation on 4xL4 (factor 2); expected wall ~40-50m (~1.4-1.7h quota), hard cap 96m (3.2h quota).',
        },
        'attempt_outer_seconds': 1020,
        'process_isolation': p006_manifest['process_isolation'],
        'cache_note': p006_manifest['cache_note'],
    }
    inventory = json.loads((ROOT / 'research/006/official-data-20261013/wheel-files.json').read_text())
    wheels = {item['name']: int(item['total_bytes']) for item in inventory['dataset_files']
              if item['name'].endswith('.whl') and 'cutlass' not in item['name'].lower()}
    for name, value in [
        ('BUNDLE', bundle),
        ('EXPECTED_CONFIG_HASHES', hashes),
        ('MANIFEST_TEXT', json.dumps(manifest, ensure_ascii=False)),
        ('TASK_IDS', task_ids),
        ('WHEELHOUSE_V31_FILES', wheels),
        ('VARIANT_CONDITIONS', conditions),
        ('REFERENCE_LICENSE_TEXT', reference_license),
    ]:
        coordinator = assignment(coordinator, name, value)
    # Remove P006's embedded TASKS_SELECTED_TEXT and add EXPECTED_TASK_META.
    coordinator = assignment(coordinator, 'TASKS_SELECTED_TEXT', '')
    coordinator = once(
        coordinator,
        "TASKS_SELECTED_TEXT = ''\n",
        'EXPECTED_TASK_META = ' + repr(task_meta) + '\n',
    )
    coordinator = once(coordinator, p006.SCHEDULE, SCHEDULE)
    coordinator = once(coordinator, p006.SCHEDULE_LOOP, SCHEDULE_LOOP)
    coordinator = once(coordinator, p006.TASK_SUBSET, TASK_SUBSET)
    coordinator = once(coordinator, p006.CONFIG_CHECK, CONFIG_CHECK)
    coordinator = once(coordinator, "atomic_text(FROZEN_DIR / 'budget-fit-LICENSE.txt', REFERENCE_LICENSE_TEXT)",
                       "atomic_text(FROZEN_DIR / 'public022-v10-LICENSE.txt', REFERENCE_LICENSE_TEXT)")
    coordinator = once(coordinator, "atomic_json(FROZEN_DIR / 'budget-fit-origin.json', EXPERIMENT['source_arms']['budget_fit_5m'])",
                       "atomic_json(FROZEN_DIR / 'source-arms.json', EXPERIMENT['source_arms'])")
    coordinator = once(coordinator, "'repeats': ['r1', 'r2']", "'repeats': ['r1']")
    coordinator = coordinator.replace('agent-bundle-006-reference-', 'agent-bundle-007-public12-')
    coordinator = once(coordinator, "extra_args=['--max-num-seqs', '2']", "extra_args=['--max-num-seqs', '4']")
    coordinator = once(coordinator, "'two isolated concurrent Evaluators; sampled via /metrics'",
                       "'four isolated concurrent Evaluators (two simultaneous pairs); sampled via /metrics'")
    coordinator = once(coordinator, p006.RUN_BODY, RUN_BODY)
    ast.parse(worker, filename='p007-worker')
    ast.parse(coordinator, filename='p007-coordinator')
    provenance = {
        'image_pin': IMAGE,
        'official_wheelhouse_version': 31,
        'coordinator_sha256': sha(coordinator.encode()),
        'worker_sha256': sha(worker.encode()),
        'task_ids': task_ids,
        'config_sha256': hashes,
        'source_arms': manifest['source_arms'],
        'whole_run': manifest['whole_run'],
        'model_pin': p006_provenance['model_pin'],
        'build_sources': {
            str(p.relative_to(ROOT)): sha(p.read_bytes())
            for p in [Path(__file__), Path(p006.__file__), Path(p006.previous.__file__), p006.previous.single.BASE]
        },
    }
    supervisor = p006_source[p006_source.index('# Execution starts here.'):]
    supervisor = supervisor.replace('P006_RUN_ID', 'P007_RUN_ID').replace('P006_RUN_ROOT', 'P007_RUN_ROOT')
    supervisor = supervisor.replace('P006_PREPARE_ONLY', 'P007_PREPARE_ONLY').replace('P006_EVALUATION_DEADLINE', 'P007_EVALUATION_DEADLINE')
    supervisor = supervisor.replace('run-006-reference-pair-', 'run-007-public12-pair-').replace('/results/006', '/results/007')
    supervisor = once(supervisor, 'WALL_SECONDS = 2460', 'WALL_SECONDS = 5520')
    supervisor = once(supervisor, 'P007_EVALUATION_DEADLINE=str(started + 2400)', 'P007_EVALUATION_DEADLINE=str(started + 5400)')
    supervisor = once(supervisor, 'len(states) == (2 if PREPARE_ONLY else 12)', 'len(states) == (2 if PREPARE_ONLY else 24)')
    supervisor = once(supervisor, "'internal_lifecycle_target_seconds': 2550, 'platform_session_limit_seconds': 2700,",
                      "'internal_lifecycle_target_seconds': 5660, 'platform_session_limit_seconds': 5760,")
    supervisor = once(supervisor, "'configured_max_concurrent_attempts': 2,", "'configured_max_concurrent_attempts': 4,")
    supervisor = once(supervisor, 'output_deadline = started + 2535', 'output_deadline = started + 5640')
    source = (
        '# Generated by notebooks/builders/build_007_public12_pair.py; default execution starts the bounded GPU experiment.\n'
        + 'COORDINATOR_SOURCE = ' + repr(coordinator) + '\n'
        + 'SOURCE_MANIFEST = ' + repr(provenance) + '\n'
        + supervisor
    )
    ast.parse(source)
    notebook = {
        'nbformat': 4,
        'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3.12', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.12'},
        },
        'cells': [
            {
                'cell_type': 'markdown',
                'id': 'p007-public12-intro',
                'metadata': {},
                'source': [
                    '# P007 Public-12 Paired Evaluation: exact public022_v10 vs behavior_v1\n',
                    'Twelve unseen dev_future tasks (4 Rich, 4 FastAPI, 4 Requests), 8-minute Agent budget, single-variable behavioral verification prompt comparison.\n',
                    'One 31B CT v2 vLLM server on 4xL4 with rolling two-pair concurrency (`MAX_CONCURRENT=4`, `--max-num-seqs=4`) to maximize GPU utilization.\n',
                    'Default execution runs GPU evaluation. CPU preparation: run `prepare.py` in a CPU session (`enable_gpu=false`). No competition submission.\n',
                ],
            },
            {
                'cell_type': 'code',
                'id': 'p007-public12-run',
                'metadata': {},
                'execution_count': None,
                'outputs': [],
                'source': source.splitlines(keepends=True),
            },
        ],
    }
    return source, json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', provenance, manifest


def main():
    source, notebook, provenance, manifest = render()
    KERNEL.mkdir(parents=True, exist_ok=True)
    Path(str(OUT) + '-single-cell.py').write_text(source)
    Path(str(OUT) + '.ipynb').write_text(notebook)
    (KERNEL / 'pair.py').write_text(source)
    CPU_KERNEL = KERNEL.parent / 'cpu-kernel'
    CPU_KERNEL.mkdir(parents=True, exist_ok=True)
    prepare_src = "import os\nos.environ['P007_PREPARE_ONLY'] = '1'\n" + source
    (KERNEL / 'prepare.py').write_text(prepare_src)
    (CPU_KERNEL / 'prepare.py').write_text(prepare_src)
    (KERNEL / 'source-manifest.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    (KERNEL / 'experiment-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (KERNEL / 'LICENSE-public022-v10.txt').write_text(json.loads(LICENSE_SOURCE.read_text())['ISSUE_CONTRACT_FILES']['LICENSE.txt'])
    meta = json.loads((ROOT / 'research/005/t0-single-state-pair-20261011/kernel/kernel-metadata.json').read_text())
    meta.update(id='ccjjllt/p007-public12-pair', title='P007 Public12 Pair', code_file='pair.py', enable_gpu=True)
    (KERNEL / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    meta_cpu = dict(meta, id='ccjjllt/p007-public12-pair-cpu-prepare', title='P007 Public12 Pair CPU Prepare',
                    code_file='prepare.py', enable_gpu=False)
    (KERNEL / 'kernel-metadata-cpu.json').write_text(json.dumps(meta_cpu, indent=2) + '\n')
    (CPU_KERNEL / 'kernel-metadata.json').write_text(json.dumps(meta_cpu, indent=2) + '\n')
    print(json.dumps({
        'source_sha256': sha(source.encode()),
        'coordinator_sha256': provenance['coordinator_sha256'],
        'worker_sha256': provenance['worker_sha256'],
        'kernel': str(KERNEL.relative_to(ROOT)),
        'tasks': manifest['task_ids'],
        'attempts': manifest['planned_attempts'],
        'max_concurrent': manifest['max_concurrent_attempts'],
    }))


if __name__ == '__main__':
    main()
