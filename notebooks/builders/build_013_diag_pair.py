#!/usr/bin/env python3
"""Build P013 31B diagnosis pair: public022_v10 vs p013_candidate (repair->review).

CPU-only file work. Adapts the proven P007 paired-evaluation carrier
(notebooks/builders/build_007_public12_pair.py on top of build_006_reference_pair.py):
same 31B model pin, same budgets (8min/28calls/80turns, temp 0.2/top_p 0.95),
same wheelhouse v31 + image pin + competition sources as P007 kernel-metadata.

Differences from P007:
- ARMS = ('public022_v10', 'p013_candidate'); p013 arm is a multi-file
  SequentialAgent (agent.yaml -> repair_agent.yaml -> review_agent.yaml with
  !include prompts/configs). compile_submission discovers multi-file dirs via
  validate_directory + sandboxed !include + sub-agent config_path resolution
  (same mechanism as the P012 agent/ coordinator with agent_tool sub-configs);
  the worker copies the whole AGENT_ROOT/variant dir (shutil.copytree), so
  BUNDLE carries ALL agent files and EXPECTED_CONFIG_HASHES covers every file.
- TASK_IDS = 4 diagnostic tasks only (rich_4006, rich_3180, requests_7205,
  requests_6757); EXPECTED_TASK_META reuses those 4 entries from the P007
  dev12 manifest; tasks.jsonl hash pin kept from P007; row count asserted 4;
  schedule 8 attempts (prepare-only: first 2 = one per arm).
- CONFIG_CHECK is per-arm: baseline keeps the exact P007 asserts; the p013 arm
  checks the SequentialAgent root, repair (5 tools, no submit_patch,
  output_key) and review (6 tools incl. submit_patch, include_contents none)
  stages, shared sampling/budgets, and single declared 31B model.

Outputs research/013/eval-31b-diag/kernel/ (pair.py single-cell +
kernel-metadata.json + experiment-manifest.json + source-manifest.json,
P007 kernel dir layout). Does NOT push to Kaggle. Does NOT read heldout.
"""
import ast
import hashlib
import json
from pathlib import Path

import build_006_reference_pair as p006
import build_007_public12_pair as p007

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'notebooks/013/013-31b-diag'
KERNEL = ROOT / 'research/013/eval-31b-diag/kernel'
BASELINE = ROOT / 'research/007/candidates/public022_v10'
P013_AGENT = ROOT / 'research/013/agent'
DEV12_MANIFEST = ROOT / 'research/007/evaluation-design-20261017/public-dev12-manifest.json'
LICENSE_SOURCE = ROOT / 'research/006/high-score-notebooks-20261013/extracted/dmitriigluzdov__gemma-4-task-soundness-serving-and-evaluation.json'
ARMS = ('public022_v10', 'p013_candidate')
TASK_IDS = ('rich_4006', 'rich_3180', 'requests_7205', 'requests_6757')
P013_FILES = ('agent.yaml', 'agent-single.yaml', 'repair_agent.yaml', 'review_agent.yaml',
              'eval_config.yaml', 'configs/sampling.yaml',
              'prompts/repair.md', 'prompts/review.md', 'prompts/single.md')
V10_FILES = ('agent.yaml', 'configs/sampling.yaml', 'eval_config.yaml', 'prompts/system.md')
IMAGE = p006.IMAGE
OFFICIAL_TASKS_SHA256 = p007.OFFICIAL_TASKS_SHA256
OFFICIAL_TASKS_BYTES = p007.OFFICIAL_TASKS_BYTES


def sha(data):
    return hashlib.sha256(data).hexdigest()


once = p007.once
assignment = p007.assignment

SCHEDULE = once(
    p007.SCHEDULE.replace('P007_RUN_ID', 'P013_RUN_ID').replace(
        'P007_RUN_ROOT', 'P013_RUN_ROOT').replace(
        'P007_PREPARE_ONLY', 'P013_PREPARE_ONLY').replace(
        'P007_EVALUATION_DEADLINE', 'P013_EVALUATION_DEADLINE'),
    "SELECTED_VARIANTS = ('public022_v10', 'behavior_v1')",
    'SELECTED_VARIANTS = ' + repr(ARMS),
)

SCHEDULE_LOOP = once(
    p007.SCHEDULE_LOOP,
    'assert len(full_schedule) == 24',
    'assert len(full_schedule) == 8',
)

TASK_SUBSET = once(
    p007.TASK_SUBSET,
    'assert len(tasks) == 12 and set(by_id) == set(TASK_IDS)',
    'assert len(tasks) == 4 and set(by_id) == set(TASK_IDS)',
)

CONFIG_CHECK = r'''
for variant in SELECTED_VARIANTS:
    condition = VARIANT_CONDITIONS[variant]
    assert yaml.safe_load(BUNDLE[variant]['configs/sampling.yaml']) == condition['sampling']
    assert yaml.safe_load(BUNDLE[variant]['eval_config.yaml'])['evaluation'] == condition['budgets']
    assert condition['budgets']['max_time_minutes'] == 8
    assert (condition.get('skills') or []) == []
    assert discover_declared_models(AGENT_ROOT / variant) == {TARGET_MODEL_NAME}
    assert not any(path.is_symlink() for path in (AGENT_ROOT / variant).rglob('*'))
v10 = yaml.safe_load(BUNDLE['public022_v10']['agent.yaml'])
assert 'adapter' not in v10 and v10['tools'] == VARIANT_CONDITIONS['public022_v10']['tools']
assert (v10.get('skills') or []) == []
root = yaml.safe_load(BUNDLE['p013_candidate']['agent.yaml'])
assert root.get('agent_class') == 'SequentialAgent'
assert [s['config_path'] for s in root['sub_agents']] == ['repair_agent.yaml', 'review_agent.yaml']
repair = yaml.safe_load(BUNDLE['p013_candidate']['repair_agent.yaml'])
assert repair['model'] == TARGET_MODEL_NAME
assert repair['tools'] == VARIANT_CONDITIONS['p013_candidate']['tools']['repair']
assert 'submit_patch' not in repair['tools'] and repair.get('output_key') == 'repair_handoff'
review = yaml.safe_load(BUNDLE['p013_candidate']['review_agent.yaml'])
assert review['model'] == TARGET_MODEL_NAME
assert review['tools'] == VARIANT_CONDITIONS['p013_candidate']['tools']['review']
assert 'submit_patch' in review['tools'] and review.get('include_contents') == 'none'
fallback = yaml.safe_load(BUNDLE['p013_candidate']['agent-single.yaml'])
assert fallback['model'] == TARGET_MODEL_NAME and 'submit_patch' in fallback['tools']
for name in ('repair_agent.yaml', 'review_agent.yaml', 'agent-single.yaml'):
    assert 'adapter' not in yaml.safe_load(BUNDLE['p013_candidate'][name])
assert 'verification_timeout_seconds' in inspect.signature(EvalConfig).parameters
'''

RUN_BODY = once(p007.RUN_BODY, "'total_tasks': 24", "'total_tasks': 8")


def load_baseline():
    manifest = json.loads((BASELINE / 'source-manifest.json').read_text())
    files = {}
    for rel in V10_FILES:
        path = BASELINE / 'agent' / rel
        raw = path.read_bytes()
        expected = manifest['files'][rel]
        assert len(raw) == expected['bytes'], f'Size mismatch public022_v10/{rel}'
        assert sha(raw) == expected['sha256'], f'SHA mismatch public022_v10/{rel}'
        files[rel] = raw.decode('utf-8')
    assert set(files) == set(V10_FILES)
    return files, manifest


def load_p013():
    files = {}
    for rel in P013_FILES:
        raw = (P013_AGENT / rel).read_bytes()
        assert b'\r' not in raw, f'CRLF in research/013/agent/{rel}'
        files[rel] = raw.decode('utf-8')
    return files


def frozen_inputs():
    v10_files, v10_manifest = load_baseline()
    p013_files = load_p013()
    assert p013_files['configs/sampling.yaml'] == v10_files['configs/sampling.yaml'], \
        'sampling budget drift between arms'
    import yaml as _yaml
    assert _yaml.safe_load(p013_files['eval_config.yaml'])['evaluation'] == \
        _yaml.safe_load(v10_files['eval_config.yaml'])['evaluation'], \
        'eval budget drift between arms (parsed evaluation blocks must match)'
    root = _yaml.safe_load(p013_files['agent.yaml'])
    assert root.get('agent_class') == 'SequentialAgent'
    assert [s['config_path'] for s in root['sub_agents']] == ['repair_agent.yaml', 'review_agent.yaml']
    dev12 = json.loads(DEV12_MANIFEST.read_text())
    selected = [item for item in dev12['selected_tasks'] if item['instance_id'] in TASK_IDS]
    assert tuple(item['instance_id'] for item in selected) == TASK_IDS, \
        'diag task order must follow the P007 dev12 manifest order'
    task_meta = {item['instance_id']: {'repo': item['repo'], 'base_commit': item['base_commit'],
                                       'split': item['split']} for item in selected}
    bundle = {ARMS[0]: v10_files, ARMS[1]: p013_files}
    return bundle, v10_manifest, TASK_IDS, task_meta, dev12


def render():
    p006_source, _, p006_provenance, p006_manifest = p006.render()
    coordinator = p006.previous.single.literal_assignment(p006_source, 'COORDINATOR_SOURCE')
    worker = p006.previous.single.literal_assignment(coordinator, 'WORKER_SOURCE')
    bundle, v10_manifest, task_ids, task_meta, dev12 = frozen_inputs()
    reference_license = json.loads(LICENSE_SOURCE.read_text())['ISSUE_CONTRACT_FILES']['LICENSE.txt']
    hashes = {arm: {name: sha(text.encode()) for name, text in files.items()} for arm, files in bundle.items()}
    budgets = {'timeout_seconds': 240, 'max_tool_calls': 28, 'max_time_minutes': 8, 'max_turns': 80}
    sampling = {'temperature': 0.2, 'top_p': 0.95, 'max_output_tokens': 8192,
                'thinking_config': {'thinking_budget': 4096, 'include_thoughts': True}}
    repair_tools = ['run_command', 'read_file', 'edit_file', 'write_file', 'get_status']
    conditions = {
        'public022_v10': {
            'tools': repair_tools + ['submit_patch', 'get_code_neighbors',
                                     'search_similar_code', 'get_code_subgraph'],
            'skills': [], 'budgets': dict(budgets), 'sampling': dict(sampling),
        },
        'p013_candidate': {
            'tools': {'repair': list(repair_tools), 'review': repair_tools + ['submit_patch'],
                      'fallback_single': repair_tools + ['submit_patch']},
            'skills': [], 'budgets': dict(budgets), 'sampling': dict(sampling),
        },
    }
    packages = dict(p006_manifest['common']['official_packages'])
    p013_arm_manifest = {
        'created_at': '2026-10-20',
        'candidate': 'p013_candidate',
        'base_note': 'P013 repair-then-review SequentialAgent; same 31B base, budgets and sampling as public022_v10',
        'training_weights': False,
        'license': 'Apache-2.0',
        'files': {rel: {'bytes': len(bundle[ARMS[1]][rel].encode()),
                        'sha256': hashes[ARMS[1]][rel]} for rel in P013_FILES},
    }
    manifest = {
        'study': 'P013 31B diagnosis pair: public022_v10 vs p013_candidate (repair-then-review SequentialAgent)',
        'common': {'official_packages': packages},
        'variant_conditions': conditions,
        'verification_timeout_seconds': 300,
        'task_ids': task_ids,
        'repeats': ['r1'],
        'planned_attempts': 8,
        'max_concurrent_attempts': 4,
        'server_max_num_seqs': 4,
        'image_pin': IMAGE,
        'research_condition': (
            'Whole-agent comparison on 4 diagnostic tasks (rich_4006, rich_3180, requests_7205, requests_6757). '
            'p013_candidate is a repair-then-review SequentialAgent with the P0-P1 execution fixes '
            '(short timeouts, /workspace source check, early full edit, clean close); baseline is the frozen '
            'public022_v10 single agent. Shared 8-minute / 28-call / 80-turn / 240s budgets, 31B CT v2, '
            'temp 0.2, top-p 0.95. Rolling pair queue launches both arms of each task simultaneously '
            'while keeping up to 2 pairs (4 workers) active on 4xL4.'
        ),
        'data': {
            'selection_manifest': str(DEV12_MANIFEST.relative_to(ROOT)),
            'selection_manifest_sha256': sha(DEV12_MANIFEST.read_bytes()),
            'source': 'tasks.jsonl',
            'source_sha256': OFFICIAL_TASKS_SHA256,
            'source_bytes': OFFICIAL_TASKS_BYTES,
            'selected_task_ids': task_ids,
            'selected_rows': 4,
            'heldout_or_reserved_accessed': False,
        },
        'source_arms': {
            'public022_v10': v10_manifest,
            'p013_candidate': p013_arm_manifest,
        },
        'whole_run': {
            'platform_session_limit_seconds': 5760,
            'evaluation_deadline_seconds': 5400,
            'coordinator_force_stop_seconds': 5520,
            'output_deadline_seconds': 5640,
            'internal_lifecycle_target_seconds': 5660,
            'quota_upper_estimate_hours': 3.2,
            'quota_note': 'Rolling 4-worker paired diagnosis on 4xL4 (factor 2); expected wall ~15-25m (~0.5-0.9h quota), hard cap 96m (3.2h quota).',
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
    coordinator = coordinator.replace('agent-bundle-006-reference-', 'agent-bundle-013-diag-')
    coordinator = once(coordinator, "extra_args=['--max-num-seqs', '2']", "extra_args=['--max-num-seqs', '4']")
    coordinator = once(coordinator, "'two isolated concurrent Evaluators; sampled via /metrics'",
                       "'four isolated concurrent Evaluators (two simultaneous pairs); sampled via /metrics'")
    coordinator = once(coordinator, p006.RUN_BODY, RUN_BODY)
    ast.parse(worker, filename='p013-worker')
    ast.parse(coordinator, filename='p013-coordinator')
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
            for p in [Path(__file__), Path(p007.__file__), Path(p006.__file__),
                      Path(p006.previous.__file__), p006.previous.single.BASE]
        },
    }
    supervisor = p006_source[p006_source.index('# Execution starts here.'):]
    supervisor = supervisor.replace('P006_RUN_ID', 'P013_RUN_ID').replace('P006_RUN_ROOT', 'P013_RUN_ROOT')
    supervisor = supervisor.replace('P006_PREPARE_ONLY', 'P013_PREPARE_ONLY').replace('P006_EVALUATION_DEADLINE', 'P013_EVALUATION_DEADLINE')
    supervisor = supervisor.replace('run-006-reference-pair-', 'run-013-31b-diag-').replace('/results/006', '/results/013')
    supervisor = once(supervisor, 'WALL_SECONDS = 2460', 'WALL_SECONDS = 5520')
    supervisor = once(supervisor, 'P013_EVALUATION_DEADLINE=str(started + 2400)', 'P013_EVALUATION_DEADLINE=str(started + 5400)')
    supervisor = once(supervisor, 'len(states) == (2 if PREPARE_ONLY else 12)', 'len(states) == (2 if PREPARE_ONLY else 8)')
    supervisor = once(supervisor, "'internal_lifecycle_target_seconds': 2550, 'platform_session_limit_seconds': 2700,",
                      "'internal_lifecycle_target_seconds': 5660, 'platform_session_limit_seconds': 5760,")
    supervisor = once(supervisor, "'configured_max_concurrent_attempts': 2,", "'configured_max_concurrent_attempts': 4,")
    supervisor = once(supervisor, 'output_deadline = started + 2535', 'output_deadline = started + 5640')
    source = (
        '# Generated by notebooks/builders/build_013_diag_pair.py; default execution starts the bounded GPU experiment.\n'
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
                'id': 'p013-diag-intro',
                'metadata': {},
                'source': [
                    '# P013 31B Diagnosis: public022_v10 vs p013_candidate (repair-then-review)\n',
                    'Four diagnostic tasks (rich_4006, rich_3180, requests_7205, requests_6757), 8-minute Agent budget per arm.\n',
                    'One 31B CT v2 vLLM server on 4xL4 with rolling two-pair concurrency (`MAX_CONCURRENT=4`, `--max-num-seqs=4`).\n',
                    'Default execution runs GPU evaluation. CPU preparation: run `prepare.py` in a CPU session (`enable_gpu=false`). No competition submission.\n',
                ],
            },
            {
                'cell_type': 'code',
                'id': 'p013-diag-run',
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
    prepare_src = "import os\nos.environ['P013_PREPARE_ONLY'] = '1'\n" + source
    ast.parse(prepare_src)
    (KERNEL / 'prepare.py').write_text(prepare_src)
    (CPU_KERNEL / 'prepare.py').write_text(prepare_src)
    (KERNEL / 'source-manifest.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    (KERNEL / 'experiment-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (KERNEL / 'LICENSE-public022-v10.txt').write_text(json.loads(LICENSE_SOURCE.read_text())['ISSUE_CONTRACT_FILES']['LICENSE.txt'])
    meta = json.loads((ROOT / 'research/005/t0-single-state-pair-20261011/kernel/kernel-metadata.json').read_text())
    meta.update(id='ccjjllt/p013-31b-diag', title='P013 31B Diag Pair', code_file='pair.py', enable_gpu=True)
    (KERNEL / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    meta_cpu = dict(meta, id='ccjjllt/p013-31b-diag-cpu-prepare', title='P013 31B Diag Pair CPU Prepare',
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
        'config_sha256': provenance['config_sha256'],
    }))


if __name__ == '__main__':
    main()
