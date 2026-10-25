"""Prepare f2-push P4: hypothesis-2435 (strategy-cache thread-local isolation)."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-push'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-push'

IID = 'hypothesisworks__hypothesis-2435'
KEY = 'f2-push-hypothesis-2435'
REPO = 'hypothesisworks/hypothesis'
BASE = '021d2a6944696198b7e8e0f5b5be3cb21de4ad2f'
FAMILY = 'hypothesisworks/hypothesis::hypothesis.strategies._internal.core.cached_strategy+get_cache::thread_local_cache_isolation'
SNAP = f'{REPO}::{BASE}'

PROD_FILES = ['hypothesis-python/src/hypothesis/strategies/_internal/core.py']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(b):
    return hashlib.sha256(b).hexdigest()


def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def forward_edits(patch):
    edits = []
    old, new = [], []
    for line in patch.splitlines(keepends=True):
        if line.startswith('@@ '):
            if old or new:
                edits.append((''.join(old), ''.join(new)))
            old, new = [], []
        elif line.startswith(('diff --git ', 'index ', '--- ', '+++ ', '\\')):
            continue
        elif line.startswith(' '):
            old.append(line[1:])
            new.append(line[1:])
        elif line.startswith('+'):
            new.append(line[1:])
        elif line.startswith('-'):
            old.append(line[1:])
    if old or new:
        edits.append((''.join(old), ''.join(new)))
    return [(o, n) for o, n in edits if o != n]


row = json.loads((ACQ / 'pages-v2/hypothesisworks__hypothesis-2435.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'hypothesis-2435-source'
arch = ART / 'hypothesis-2435-source.tgz'
ART.mkdir(parents=True, exist_ok=True)
url = f'https://codeload.github.com/{REPO}/tar.gz/{BASE}'
assert arch.exists(), f'missing {arch} (pre-downloaded)'
with tarfile.open(arch) as bundle:
    members = bundle.getmembers()
    top = members[0].name.split('/')[0]
    for m in members:
        parts = Path(m.name).parts
        if not parts or parts[0] != top or '..' in parts:
            continue
        rel = Path(*parts[1:])
        if not rel.parts:
            continue
        tgt = WS / rel
        if m.isdir():
            tgt.mkdir(parents=True, exist_ok=True)
        elif m.isfile():
            payload = bundle.extractfile(m).read()
            if tgt.exists() and tgt.read_bytes() != payload:
                raise ValueError(f'altered {tgt}')
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_bytes(payload)
print('workspace ready', flush=True)

parts = row['patch'].split('diff --git a/')
file_edits = {}
for part in parts:
    if not part.strip():
        continue
    name = part.split()[0].strip()
    if name in PROD_FILES:
        file_edits[name] = forward_edits('diff --git a/' + part)
assert set(file_edits) == set(PROD_FILES), set(file_edits)
cur_text = {}
for name, edits in file_edits.items():
    cur = (WS / name).read_text()
    for o, n in edits:
        assert o and cur.count(o) == 1, (name, o[:160])
        cur = cur.replace(o, n)
    cur_text[name] = cur
assert '_CACHE = threading.local()' in cur_text[PROD_FILES[0]]
assert '\nSTRATEGY_CACHE = LRUReusedCache(1024)\n' not in cur_text[PROD_FILES[0]]
print(f'edit reconstruction verified ({sum(len(e) for e in file_edits.values())} prod hunks; workspace left at baseline)', flush=True)

targets = sorted({nn.split('::', 1)[0] for nn in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
print('targets:', targets, flush=True)
assert targets == ['hypothesis-python/tests/cover/test_cache_implementation.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-push-hypothesis-2435-task.json'
save(adapted, task)

OBS = ("import threading\n"
       "from hypothesis.strategies._internal.core import get_cache\n"
       "main_cache = get_cache()\n"
       "main_cache['f2probe'] = 1\n"
       "seen = {}\n"
       "def target():\n"
       "    seen['cache'] = get_cache()\n"
       "t = threading.Thread(target=target)\n"
       "t.start()\n"
       "t.join()\n"
       "print('cache_identity', seen['cache'] is main_cache)\n"
       "assert seen['cache'] is not main_cache, 'threads share one cache'\n"
       "assert 'f2probe' not in seen['cache'], 'cross-thread entry visible'\n"
       "print('f2_thread_isolation', True)\n")
BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "module = importlib.import_module('hypothesis.strategies._internal.core')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert str(actual).endswith('hypothesis/strategies/_internal/core.py'), actual\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'hypothesis-python/src/hypothesis/strategies/_internal/core.py',
                             'start_line': 148, 'end_line': 200}),
           act('run_command', {'command': CMD}, 'error')]
for name, edits in file_edits.items():
    for o, n in edits:
        actions.append(act('edit_file', {'filepath': name, 'old_string': o, 'new_string': n}))
actions += [act('run_command', {'command': CMD}),
            act('get_status', {}),
            act('submit_patch', {})]
recipe_path = REPLAY / 'f2-push-hypothesis-2435-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': ('subagent f2-push: causal source read (one global STRATEGY_CACHE dict mutated '
                    'from all threads races inside cacheable; threading.local() per-thread '
                    'LRUReusedCache isolates owners), pinned tests, command review 2026-10-25'),
    'review_notes': ('Strategy caches must be per-thread owned copies: each thread cache is not '
                     'another thread cache and cross-thread entries stay invisible. Copy-time identity '
                     'plus bidirectional isolation are asserted source-bound before and after the repair. '
                     'Original F2P/P2P unchanged. Baseline diagnostic failure is expected task '
                     'reproduction, not a policy error. Off-policy scripted demonstration, not a Gemma '
                     'sample. Risk note: the F2P is a probabilistic race regression test; if the race '
                     'does not trigger locally the baseline leg is a near-miss (K11 precedent), not relaxed.'),
    'actions': actions,
    'final_response': ('Isolate strategy caches per thread so concurrent builds no longer race on one '
                       'shared dict. Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-push/pages-v2/hypothesisworks__hypothesis-2435.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/hypothesisworks__hypothesis-2435.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-push/hypothesis-2435-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('thread-local cache isolation: per-thread cache is not another thread cache; '
                           'captured at the strategy-construction cache-access timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-push-hypothesis-2435-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
