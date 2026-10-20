#!/usr/bin/env python3
"""Build P006 E2 from two frozen sources; never executes model or external code.

The proven P005 supervisor, installation and worker loop are source inputs only.
Generated scripts are standalone. No complete competition task file is accessed.
"""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

import build_005_t0_single_state_pair as previous

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks/006-reference-pair'
KERNEL = ROOT / 'research/006/reference-pair-20261013/kernel'
HISTORY = ROOT / 'research/006/recovered-exec-closure-20261013/output/results/005/run-005-exec-closure-pair-20261011T082035Z-7fcb29d6'
A_ZIP = HISTORY / 'exec_closure/agent-bundle-005-pair-exec_closure.zip'
B_SOURCE = ROOT / 'research/006/high-score-notebooks-20261013/extracted/verracodeguacas__gemma-4-agent-budget-fit-single-agent.json'
LICENSE_SOURCE = ROOT / 'research/006/high-score-notebooks-20261013/extracted/dmitriigluzdov__gemma-4-task-soundness-serving-and-evaluation.json'
TASK_SOURCE = ROOT / 'research/005/t0-single-state-pair-20261011/output-v3/experiment/frozen_bundle/tasks-selected.jsonl'
TASK_SOURCE_SHA = '5f463bcc479f0228876b501351153055c2be017a07c9f9514fb1ae1730c7256f'
TASKS = ('fastapi_14873', 'fastapi_5077', 'rich_3518')
ARMS = ('exec_closure', 'budget_fit_5m')
IMAGE = previous.IMAGE


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
RUN_ID = os.environ['P006_RUN_ID']
RUN_ROOT = Path(os.environ['P006_RUN_ROOT'])
AGENT_ROOT = Path('/kaggle/working/agent_exec') / RUN_ID
INPUT_ROOT = Path('/kaggle/input')
WORKING_DIR = Path('/kaggle/working')
TARGET_MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'
SELECTED_VARIANTS = ('exec_closure', 'budget_fit_5m')
REPEAT_ID = 'frozen'
MAX_CONCURRENT = 2
ATTEMPT_WALL_SECONDS = 900  # 300 agent + 300 judge + up to 300 setup/cleanup.
VERIFICATION_TIMEOUT_SECONDS = 300
PREPARE_ONLY = os.environ.get('P006_PREPARE_ONLY') == '1'
EVALUATION_DEADLINE = float(os.environ['P006_EVALUATION_DEADLINE'])


def set_phase(name, **extra):
    atomic_json(RUN_ROOT / 'phase.json', {'phase': name, 'time': time.time(), **extra})

'''

SCHEDULE_LOOP = r'''
schedule = []
for repeat_index, repeat_id in enumerate(('r1', 'r2')):
    for slot, task_id in enumerate(TASK_IDS, 1):
        order = SELECTED_VARIANTS if (slot + repeat_index) % 2 else tuple(reversed(SELECTED_VARIANTS))
        for launch_order, variant in enumerate(order, 1):
            ordinal = len(schedule) + 1
            attempt_id = f'{ordinal:02d}-{task_id}-{repeat_id}-{variant}'
            schedule.append({'ordinal': ordinal, 'task_slot': slot, 'repeat_id': repeat_id,
                             'within_pair_launch_order': launch_order, 'task_id': task_id,
                             'variant': variant, 'attempt_id': attempt_id,
                             'results_dir': str(RUN_ROOT / variant / repeat_id / 'attempts' / attempt_id),
                             'config_sha256': EXPECTED_CONFIG_HASHES[variant]})
full_schedule = list(schedule)
assert len(full_schedule) == 12
if PREPARE_ONLY:
    schedule = schedule[:2]  # Construct both official agents; no model or judge.
state = {entry['attempt_id']: dict(entry, status='planned') for entry in schedule}
atomic_json(RUN_ROOT / 'full_schedule.json', full_schedule)
atomic_json(RUN_ROOT / 'attempt_status.json', state)
'''

TASK_SUBSET = r'''
# Only the three previously exposed task rows are embedded. Never read, scan or
# hash the attached full tasks.jsonl. Reference fields stay on the harness side.
TASKS_SUBSET_PATH = FROZEN_DIR / 'tasks-selected.jsonl'
subset_bytes = TASKS_SELECTED_TEXT.encode('utf-8')
assert hashlib.sha256(subset_bytes).hexdigest() == EXPERIMENT['data']['selected_sha256']
TASKS_SUBSET_PATH.write_bytes(subset_bytes)
tasks_subset_sha = sha256_file(TASKS_SUBSET_PATH)
tasks_sha, tasks_size = tasks_subset_sha, len(subset_bytes)
tasks = load_tasks(TASKS_SUBSET_PATH)
by_id = {task.instance_id: task for task in tasks}
assert len(tasks) == 3 and set(by_id) == set(TASK_IDS)
from swegemma.deduplication import resolve_task_snapshot_paths
for task in tasks:
    assert task.patch and task.test_patch, 'Frozen public task must be complete; never hydrate from another dataset'
    snapshot, base_snapshot, patch_file = resolve_task_snapshot_paths(DATA_DIR / 'snapshots', task.instance_id, task.repo)
    assert snapshot.is_file(), f'Missing selected snapshot: {task.instance_id}'
    assert base_snapshot is None or base_snapshot.is_file()
    assert patch_file is None or patch_file.is_file()
atomic_json(FROZEN_DIR / 'task-source.json', EXPERIMENT['data'])
'''

CONFIG_CHECK = r'''
common = EXPERIMENT['common']
for variant in SELECTED_VARIANTS:
    condition = VARIANT_CONDITIONS[variant]
    config = yaml.safe_load(BUNDLE[variant]['agent.yaml'])
    assert 'adapter' not in config and config['tools'] == condition['tools']
    assert yaml.safe_load(BUNDLE[variant]['configs/sampling.yaml']) == condition['sampling']
    assert yaml.safe_load(BUNDLE[variant]['eval_config.yaml'])['evaluation'] == condition['budgets']
    assert condition['budgets']['max_time_minutes'] == 5
    assert (config.get('skills') or []) == condition['skills']
    assert discover_declared_models(AGENT_ROOT / variant) == {TARGET_MODEL_NAME}
    assert not any(path.is_symlink() for path in (AGENT_ROOT / variant).rglob('*'))
assert 'verification_timeout_seconds' in inspect.signature(EvalConfig).parameters
'''

CPU_COMPILE = r'''
        assert config.harness.verification_timeout_seconds == 300
        assert config.harness.command_timeout_seconds == int(budgets['timeout_seconds'])
        if SPEC.get('prepare_only'):
            phase('compile_without_model_or_judge')
            from adk_submission import compile_submission
            from swegemma.sandbox import AdkSandboxCodeExecutor
            ctx.container_id = evaluator.docker.start()
            executor = AdkSandboxCodeExecutor(
                sandbox=ctx.sandbox, timeout_seconds=config.harness.command_timeout_seconds,
                start_time=ctx.task_start_time, max_time_minutes=config.budget.time_minutes,
                budget_check_fn=ctx.check_budget)
            agent = compile_submission(
                submission_dir=config.submission_dir, tool_registry=ctx.create_tools(),
                model_registry=models, code_executor=executor,
                script_timeout=config.harness.command_timeout_seconds,
                limits=limits, generation_constraints=generation_constraints,
                adapter_manifest=adapters)
            STATUS.update(status='prepared', agent_type=type(agent).__name__, agent_name=agent.name,
                          tool_types=([type(tool).__name__ for tool in agent.tools] if hasattr(agent, 'tools') else
                                      [type(sub).__name__ + ':' + getattr(sub, 'name', '?') for sub in getattr(agent, 'sub_agents', [])]),
                          command_timeout_seconds=config.harness.command_timeout_seconds,
                          verification_timeout_seconds=config.harness.verification_timeout_seconds,
                          model_requests=0, judge_runs=0)
            atomic_json(ATTEMPT / 'prepared.json', STATUS)
            print(json.dumps(STATUS), flush=True)
            return 0
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
    for pair_start in range(0, len(schedule), 2):
        if time.monotonic() >= EVALUATION_DEADLINE:
            raise TimeoutError('Whole-run evaluation deadline; keep partial evidence, no retry')
        pair = schedule[pair_start:pair_start + 2]
        set_phase('cpu_prepare' if PREPARE_ONLY else 'evaluation',
                  task_slot=pair[0]['task_slot'], task_id=pair[0]['task_id'], repeat_id=pair[0]['repeat_id'])
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
                             'prepare_only': PREPARE_ONLY, 'total_tasks': 12,
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
            print(f'Launched {entry["attempt_id"]} pid={process.pid}', flush=True)
        assert len(active) <= MAX_CONCURRENT
        while active:
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


def frozen_inputs():
    source_manifest = json.loads((A_ZIP.parent / 'submission_manifest.json').read_text())
    assert sha(A_ZIP.read_bytes()) == source_manifest['zip_sha256'] == 'a377164fb6dd9b3a8397c9a93d27e5fbaeaa5ac07133d1770529ccca0ee38b0e'
    with zipfile.ZipFile(A_ZIP) as archive:
        a = {name: archive.read(name).decode() for name in archive.namelist()}
    assert {name: sha(text.encode()) for name, text in a.items()} == source_manifest['config_sha256']
    original_b = json.loads(B_SOURCE.read_text())['FILES']
    assert len(original_b) == 4
    # Exact V5 file dictionary was extracted without executing the Notebook.
    raw_b = json.loads((ROOT / 'research/006/high-score-notebooks-20261013/raw/verracodeguacas__gemma-4-agent-budget-fit-single-agent.json').read_text())
    assert raw_b['metadata']['current_version_number'] == 5
    embedded = []
    for cell in json.loads(raw_b['blob']['source'])['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FILES' for t in node.targets):
                embedded.append(ast.literal_eval(node.value))
    assert embedded == [original_b]
    b = dict(original_b)
    b['eval_config.yaml'] = once(b['eval_config.yaml'], 'max_time_minutes: 8', 'max_time_minutes: 5')
    assert b['prompts/system.md'].count('8 minutes') == 2
    b['prompts/system.md'] = b['prompts/system.md'].replace('8 minutes', '5 minutes')
    raw = TASK_SOURCE.read_bytes()
    assert sha(raw) == TASK_SOURCE_SHA
    rows = {}
    for line in raw.splitlines():
        task = json.loads(line)
        assert task['instance_id'] in ('rich_3942', *TASKS)
        if task['instance_id'] in TASKS:
            assert task['instance_id'] not in rows and task.get('patch') and task.get('test_patch')
            rows[task['instance_id']] = line.decode()
    assert set(rows) == set(TASKS)
    subset = '\n'.join(rows[task_id] for task_id in TASKS) + '\n'
    return {ARMS[0]: a, ARMS[1]: b}, subset, source_manifest, original_b


def render():
    old_source, _, old_provenance = previous.render()
    coordinator = previous.single.literal_assignment(old_source, 'COORDINATOR_SOURCE')
    worker = previous.single.literal_assignment(coordinator, 'WORKER_SOURCE')
    bundle, subset, a_manifest, original_b = frozen_inputs()
    # Standard Apache-2.0 text already archived locally. The V5 page itself was
    # independently checked on 2026-10-13 and links this license explicitly.
    reference_license = json.loads(LICENSE_SOURCE.read_text())['ISSUE_CONTRACT_FILES']['LICENSE.txt']
    hashes = {arm: {name: sha(text.encode()) for name, text in files.items()} for arm, files in bundle.items()}
    conditions = {
        'exec_closure': {'tools': ['run_command', 'read_file', 'edit_file', 'write_file', 'get_status', 'submit_patch'],
                         'skills': ['skills/source-edit', 'skills/working-state', 'skills/source-probe'],
                         'budgets': {'timeout_seconds': 60, 'max_tool_calls': 50, 'max_time_minutes': 5, 'max_turns': 80},
                         'sampling': {'temperature': 0.2, 'top_p': 0.95, 'max_output_tokens': 16384,
                                      'thinking_config': {'thinking_budget': 4096, 'include_thoughts': True}}},
        'budget_fit_5m': {'tools': ['run_command', 'read_file', 'edit_file', 'write_file', 'get_status', 'submit_patch',
                                  'get_code_neighbors', 'search_similar_code', 'get_code_subgraph'],
                          'skills': [],
                          'budgets': {'timeout_seconds': 240, 'max_tool_calls': 28, 'max_time_minutes': 5, 'max_turns': 80},
                          'sampling': {'temperature': 0.2, 'top_p': 0.95, 'max_output_tokens': 8192,
                                       'thinking_config': {'thinking_budget': 4096, 'include_thoughts': True}}}}
    old_manifest = json.loads(previous.single.literal_assignment(coordinator, 'MANIFEST_TEXT'))
    packages = dict(old_manifest['common']['official_packages'])
    for item in json.loads((ROOT / 'research/006/official-data-20261013/wheel-manifest.json').read_text()):
        dist = item['name'].split('-')[0].replace('_', '-')
        packages[dist] = item['name'].split('-')[1] + ' sha256 ' + item['sha256']
    # v31 rearchives packages without changing their version strings. Use the
    # downloaded v31 archive, never a predecessor's archive hash.
    core_wheel = ROOT / 'research/006/data-replay-20261013/wheels/adk_eval_core-0.1.0-py3-none-any.whl'
    core_sha = sha(core_wheel.read_bytes())
    assert core_sha == 'e614e1b2fe8d886a634871e4c57f9887e1cc66e8c27300f8503a43c6e3ca82f3'
    packages['adk-eval-core'] = '0.1.0 sha256 ' + core_sha
    google_wheel = ROOT / 'research/006/official-data-20261013/wheels/google_adk-1.36.1-py3-none-any.whl'
    google_sha = sha(google_wheel.read_bytes())
    assert google_sha == '1a2f6868c509e3151fb0de3575a7d18b45c338be86f420924dad74e7193631a0'
    packages['google-adk'] = '1.36.1 sha256 ' + google_sha
    manifest = {
        'study': 'P006 E2: frozen exec_closure vs Budget-Fit V5 matched-five-minute reference',
        'common': {'official_packages': packages}, 'variant_conditions': conditions,
        'additional_wheel_sources': [{'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                                      'sha256': sha(path.read_bytes())} for path in (core_wheel, google_wheel)],
        'verification_timeout_seconds': 300, 'task_ids': TASKS, 'repeats': ['r1', 'r2'],
        'planned_attempts': 12, 'max_concurrent_attempts': 2, 'image_pin': IMAGE,
        'research_condition': 'Whole-agent comparison at common five-minute Agent wall budget. Tools, command budgets and output reserves differ; not a prompt-only ablation. Budget-Fit reference is modified and has no inherited 0.18 score.',
        'data': {'source': str(TASK_SOURCE.relative_to(ROOT)), 'source_sha256': TASK_SOURCE_SHA,
                 'source_rows': 4, 'selected_task_ids': TASKS, 'selected_rows': 3,
                 'selected_sha256': sha(subset.encode()), 'selected_bytes': len(subset.encode()),
                 'full_competition_tasks_file_accessed': False},
        'source_arms': {'exec_closure': {'path': str(A_ZIP.relative_to(ROOT)), 'zip_sha256': a_manifest['zip_sha256'], 'modifications': []},
                        'budget_fit_5m': {'url': 'https://www.kaggle.com/code/verracodeguacas/gemma-4-agent-budget-fit-single-agent?scriptVersionId=355310852',
                                          'version': 5, 'source_file_sha256': {name: sha(text.encode()) for name, text in original_b.items()},
                                          'author': 'Wilmer E. Henao (verracodeguacas)',
                                          'license': 'Apache-2.0', 'license_url': 'https://www.apache.org/licenses/LICENSE-2.0',
                                          'license_evidence': 'Exact V5 page License section visible through Chrome accessibility on 2026-10-13: Apache 2.0 link',
                                          'license_text_sha256': sha(reference_license.encode()),
                                          'modifications': ['eval_config max_time_minutes 8 -> 5', 'two prompt occurrences: 8 minutes -> 5 minutes'],
                                          'original_version_score': 0.18, 'matched_reference_score': None}},
        'whole_run': {'platform_session_limit_seconds': 2700, 'evaluation_deadline_seconds': 2400,
                      'coordinator_force_stop_seconds': 2460, 'output_deadline_seconds': 2535,
                      'internal_lifecycle_target_seconds': 2550, 'quota_upper_estimate_hours': 1.5,
                      'quota_note': '45 minute 4xL4 session at factor2; actual account charge is external'},
        'attempt_outer_seconds': 900,
        'interpretation': 'Three previously exposed tasks, two pairs each; repeated attempts are not independent tasks. No hidden or reserved evaluation content.',
        'process_isolation': old_manifest['process_isolation'], 'cache_note': old_manifest['cache_note']}
    inventory = json.loads((ROOT / 'research/006/official-data-20261013/wheel-files.json').read_text())
    wheels = {item['name']: int(item['total_bytes']) for item in inventory['dataset_files']
              if item['name'].endswith('.whl') and 'cutlass' not in item['name'].lower()}
    for name, value in [('BUNDLE', bundle), ('EXPECTED_CONFIG_HASHES', hashes), ('MANIFEST_TEXT', json.dumps(manifest, ensure_ascii=False)),
                        ('TASK_IDS', TASKS), ('WHEELHOUSE_V30_FILES', wheels)]:
        coordinator = assignment(coordinator, name, value)
    coordinator = coordinator.replace('WHEELHOUSE_V30_FILES', 'WHEELHOUSE_V31_FILES').replace('wheelhouse v30', 'wheelhouse v31').replace('recorded v30', 'recorded v31')
    coordinator = once(coordinator, previous.SCHEDULE, SCHEDULE)
    setup = previous.SETUP
    setup = setup.replace("'tasks.jsonl')\nTASKS_PATH = DATA_DIR / 'tasks.jsonl'", "'snapshots')")
    start = setup.index('schedule = []')
    setup = setup[:start] + SCHEDULE_LOOP
    setup += "\natomic_text(FROZEN_DIR / 'budget-fit-LICENSE.txt', REFERENCE_LICENSE_TEXT)\natomic_json(FROZEN_DIR / 'budget-fit-origin.json', EXPERIMENT['source_arms']['budget_fit_5m'])\n"
    coordinator = once(coordinator, previous.SETUP, setup)
    coordinator = once(coordinator, previous.TASK_SUBSET, TASK_SUBSET)
    coordinator = once(coordinator, previous.CONFIG_CHECK, CONFIG_CHECK)
    coordinator = once(coordinator, "assert TASKS_PATH.is_file() and MODEL_PATH.is_dir()", "assert (DATA_DIR / 'snapshots').is_dir() and MODEL_PATH.is_dir()")
    coordinator = once(coordinator, "assert torch.cuda.is_available() and torch.cuda.device_count() >= 1, 'CUDA unavailable'",
                       "if PREPARE_ONLY:\n    assert not torch.cuda.is_available(), 'CPU preparation must use a CPU session'\nelse:\n    assert torch.cuda.is_available() and torch.cuda.device_count() == 4, 'Requires four L4 GPUs'\n    assert all('L4' in torch.cuda.get_device_properties(i).name for i in range(4))")
    coordinator = once(coordinator, "'run_id': RUN_ID, 'repeat_id': REPEAT_ID,", "'run_id': RUN_ID, 'repeats': ['r1', 'r2'], 'mode': 'cpu_prepare' if PREPARE_ONLY else 'evaluate',")
    coordinator = once(coordinator, "'budgets': budgets, 'sampling': common_sampling,", "'variant_conditions': VARIANT_CONDITIONS, 'verification_timeout_seconds': VERIFICATION_TIMEOUT_SECONDS,")
    coordinator = coordinator.replace('agent-bundle-005-pair-', 'agent-bundle-006-reference-')
    coordinator = once(coordinator, "'model_requests': 'two isolated concurrent Evaluators; sampled via /metrics'",
                       "'model_requests': 0 if PREPARE_ONLY else 'two isolated concurrent Evaluators; sampled via /metrics'")
    worker = once(worker, "timeout_seconds=int(budgets['timeout_seconds']), max_tool_calls=int(budgets['max_tool_calls']),",
                  "timeout_seconds=int(budgets['timeout_seconds']), verification_timeout_seconds=int(SPEC['verification_timeout_seconds']),\n            max_tool_calls=int(budgets['max_tool_calls']),")
    worker = once(worker, "STATUS['status'] = 'running'", CPU_COMPILE + "\n        STATUS['status'] = 'running'")
    worker = once(worker, 'total_tasks=8', "total_tasks=SPEC['total_tasks']")
    coordinator = assignment(coordinator, 'WORKER_SOURCE', worker)
    constants = '\nTASKS_SELECTED_TEXT = ' + repr(subset) + '\nVARIANT_CONDITIONS = ' + repr(conditions) + '\nREFERENCE_LICENSE_TEXT = ' + repr(reference_license) + '\n'
    coordinator = once(coordinator, "set_phase('setup')", constants + "\nset_phase('setup')")
    start = coordinator.index('class ArchivingVllmServer')
    end = coordinator.index('results_by_variant =', start)
    server_block = coordinator[start:end]
    coordinator = coordinator[:start] + 'server = None\nif not PREPARE_ONLY:\n' + ''.join('    ' + line if line.strip() else line for line in server_block.splitlines(keepends=True)) + coordinator[end:]
    runner_prefix = previous.RUN_PAIRS[:previous.RUN_PAIRS.index("try:\n    set_phase('server_start')")]
    runner_prefix = once(runner_prefix, "    result_path = attempt / 'result.json'", "    result_path = attempt / ('prepared.json' if PREPARE_ONLY else 'result.json')")
    coordinator = once(coordinator, previous.RUN_PAIRS, runner_prefix + RUN_BODY)
    # Discovery of mounted inputs is path-only. The forbidden full task read is absent.
    assert 'TASKS_PATH' not in coordinator and 'with TASKS_PATH' not in coordinator
    ast.parse(worker, filename='p006-worker')
    ast.parse(coordinator, filename='p006-coordinator')
    provenance = {'image_pin': IMAGE, 'official_wheelhouse_version': 31,
                  'coordinator_sha256': sha(coordinator.encode()), 'worker_sha256': sha(worker.encode()),
                  'task_ids': TASKS, 'selected_tasks_sha256': sha(subset.encode()), 'config_sha256': hashes,
                  'source_arms': manifest['source_arms'], 'whole_run': manifest['whole_run'],
                  'model_pin': old_provenance['model_pin'],
                  'build_sources': {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in [Path(__file__), Path(previous.__file__), previous.single.BASE]}}
    supervisor = previous.SUPERVISOR.replace('P005_RUN_ID', 'P006_RUN_ID').replace('P005_RUN_ROOT', 'P006_RUN_ROOT')
    supervisor = supervisor.replace('run-005-t0-single-state-pair-', 'run-006-reference-pair-').replace("/results/005", "/results/006")
    supervisor = once(supervisor, 'WALL_SECONDS = 2550', 'WALL_SECONDS = 2460')
    supervisor = once(supervisor, "RUN_ID = 'run-006-reference-pair-'", "PREPARE_ONLY = os.environ.get('P006_PREPARE_ONLY') == '1' or '--prepare-only' in sys.argv\nRUN_ID = 'run-006-reference-pair-'")
    supervisor = once(supervisor, "env = dict(os.environ, P006_RUN_ID=RUN_ID, P006_RUN_ROOT=str(RUN_ROOT), PYTHONUNBUFFERED='1')\nstarted = time.monotonic()",
                      "started = time.monotonic()\nenv = dict(os.environ, P006_RUN_ID=RUN_ID, P006_RUN_ROOT=str(RUN_ROOT), PYTHONUNBUFFERED='1',\n           P006_PREPARE_ONLY='1' if PREPARE_ONLY else '0',\n           P006_EVALUATION_DEADLINE=str(started + 2400))")
    supervisor = once(supervisor, 'rows.append(sample())', "rows.append({'phase': 'cpu_prepare', 'time': time.time(), 'elapsed_seconds': elapsed} if PREPARE_ONLY else sample())")
    supervisor = once(supervisor, "complete = len(states) == 8 and all(item.get('status') == 'completed' for item in states.values())",
                      "expected = 'prepared' if PREPARE_ONLY else 'completed'\n    complete = len(states) == (2 if PREPARE_ONLY else 12) and all(item.get('status') == expected for item in states.values())")
    supervisor = once(supervisor, "'status': 'completed' if ok else 'partial_or_failed',", "'status': ('prepared' if PREPARE_ONLY else 'completed') if ok else 'partial_or_failed',\n              'mode': 'cpu_prepare' if PREPARE_ONLY else 'evaluate',\n              'model_requests': 0 if PREPARE_ONLY else None, 'judge_runs': 0 if PREPARE_ONLY else None,")
    supervisor = once(supervisor, "'completed_attempts': sum(s.get('status') == 'completed' for s in states.values()),",
                      "'completed_attempts': sum(s.get('status') == 'completed' for s in states.values()),\n              'prepared_agents': sum(s.get('status') == 'prepared' for s in states.values()),\n              'internal_lifecycle_target_seconds': 2550, 'platform_session_limit_seconds': 2700,")
    supervisor = once(supervisor, 'output_deadline = started + 2680', 'output_deadline = started + 2535')
    insertion = supervisor.index('if sys.version_info[:2]')
    supervisor = supervisor[:insertion] + previous.PROCESS_HELPERS + '\n' + supervisor[insertion:]
    source = '# Generated by notebooks/build_006_reference_pair.py; default execution starts the bounded GPU experiment.\n' + 'COORDINATOR_SOURCE = ' + repr(coordinator) + '\nSOURCE_MANIFEST = ' + repr(provenance) + '\n' + supervisor
    ast.parse(source)
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3.12', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3.12'}},
                'cells': [{'cell_type': 'markdown', 'id': 'p006-reference-intro', 'metadata': {}, 'source': [
                    '# P006 E2: historical exec_closure vs Budget-Fit V5 matched-five-minute\n',
                    'Three exposed tasks, two repeats per arm, one CT v2 service and two isolated evaluators.\n',
                    'Different tools/command/output budgets are part of each design; this is not a prompt-only ablation.\n',
                    'Pinned wheelhouse v31; verification 300 seconds. No inherited 0.18 claim for the modified reference.\n',
                    'Default execution uses GPU. CPU preparation: run prepare.py in a CPU session. No competition submission.\n']},
                          {'cell_type': 'code', 'id': 'p006-reference-run', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': source.splitlines(keepends=True)}]}
    return source, json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', provenance, manifest


def main():
    source, notebook, provenance, manifest = render()
    KERNEL.mkdir(parents=True, exist_ok=True)
    Path(str(OUT) + '-single-cell.py').write_text(source)
    Path(str(OUT) + '.ipynb').write_text(notebook)
    (KERNEL / 'pair.py').write_text(source)
    (KERNEL / 'prepare.py').write_text("import os\nos.environ['P006_PREPARE_ONLY'] = '1'\n" + source)
    (KERNEL / 'source-manifest.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    (KERNEL / 'experiment-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (KERNEL / 'LICENSE-Budget-Fit.txt').write_text(json.loads(LICENSE_SOURCE.read_text())['ISSUE_CONTRACT_FILES']['LICENSE.txt'])
    meta = json.loads((ROOT / 'research/005/t0-single-state-pair-20261011/kernel/kernel-metadata.json').read_text())
    meta.update(id='ccjjllt/p006-reference-pair', title='P006 Reference Pair', code_file='pair.py')
    (KERNEL / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    meta.update(id='ccjjllt/p006-reference-pair-cpu-prepare', title='P006 Reference Pair CPU Prepare', code_file='prepare.py', enable_gpu=False)
    (KERNEL / 'kernel-metadata-cpu.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({'source_sha256': sha(source.encode()), 'coordinator_sha256': provenance['coordinator_sha256'],
                      'kernel': str(KERNEL.relative_to(ROOT)), 'tasks': TASKS, 'attempts': 12}))


if __name__ == '__main__':
    main()
