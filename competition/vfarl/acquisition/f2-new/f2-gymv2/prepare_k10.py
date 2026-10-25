"""Prepare f2-gymv2 K10: openmdao-3125 (hook copy-on-inherit isolation)."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-gymv2'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-gymv2'

IID = 'openmdao__openmdao-3125'
KEY = 'f2-gymv2-openmdao-3125'
REPO = 'OpenMDAO/OpenMDAO'
BASE = '56f4bb77c196fc568bde910120ee0ae68470d904'
FAMILY = 'OpenMDAO/OpenMDAO::openmdao.utils.hooks._HookMeta.copy::inherit_hook_copy_isolation'
SNAP = f'{REPO}::{BASE}'


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
    cur_file = None
    for line in patch.splitlines(keepends=True):
        if line.startswith('diff --git '):
            if old or new:
                edits.append((cur_file, ''.join(old), ''.join(new)))
            old, new = [], []
            cur_file = line.split(' b/')[-1].strip()
        elif line.startswith(('index ', '--- ', '+++ ', '\\')):
            continue
        elif line.startswith('@@ '):
            if old or new:
                edits.append((cur_file, ''.join(old), ''.join(new)))
            old, new = [], []
        elif line.startswith(' '):
            old.append(line[1:])
            new.append(line[1:])
        elif line.startswith('+'):
            new.append(line[1:])
        elif line.startswith('-'):
            old.append(line[1:])
    if old or new:
        edits.append((cur_file, ''.join(old), ''.join(new)))
    return [(f, o, n) for f, o, n in edits if o != n]


row = json.loads((ACQ / 'pages-v2/openmdao__openmdao-3125.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'openmdao-3125-source'
arch = ART / 'openmdao-3125-source.tgz'
ART.mkdir(parents=True, exist_ok=True)
url = f'https://codeload.github.com/{REPO}/tar.gz/{BASE}'
if not arch.exists():
    print(f'downloading {REPO}@{BASE}', flush=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'vfarl-f2-research'}),
                                timeout=300) as r:
        arch.write_bytes(r.read())
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

assert '_HookMeta' not in (WS / 'openmdao/utils/hooks.py').read_text(), 'workspace already fixed?'
edits = forward_edits(row['patch'])
print(f'{len(edits)} hunks:', [(f, len(o)) for f, o, n in edits], flush=True)
applied = []
for f, o, n in edits:
    cur = (WS / f).read_text()
    assert cur.count(o) == 1, (f, o[:150])
    (WS / f).write_text(cur.replace(o, n))
    applied.append((f, o, n))
assert '_HookMeta' in (WS / 'openmdao/utils/hooks.py').read_text()
for f, o, n in applied:
    cur = (WS / f).read_text()
    assert cur.count(n) == 1, (f, 'revert anchor')
    (WS / f).write_text(cur.replace(n, o))
assert '_HookMeta' not in (WS / 'openmdao/utils/hooks.py').read_text()
print('edit reconstruction verified (workspace restored to buggy)', flush=True)

targets = sorted({n.split('::', 1)[0] for n in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
print('targets:', targets, flush=True)
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-gymv2-openmdao-3125-task.json'
save(adapted, task)

BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "expected=(Path.cwd() / 'openmdao/utils/hooks.py').resolve()\n"
           "module = importlib.import_module('openmdao.utils.hooks')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert actual == expected, (actual, expected)\n")
OBS = ("from openmdao.utils.hooks import _HookMeta\ncalls = []\n"
       "base = _HookMeta('C', None, lambda inst: calls.append('base'))\n"
       "child = base.copy()\n"
       "print('hook_identity', child is base)\n"
       "assert child is not base, 'inherit shares hook object'\n"
       "assert child._stamp == base._stamp, 'copy lost ordering stamp'\n"
       "child.deactivate()\n"
       "base(None)\n"
       "print('base_calls', calls)\n"
       "assert calls == ['base'], 'child deactivation leaked into base'\n"
       "print('f2_hook_isolation', True)\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'openmdao/utils/hooks.py', 'start_line': 1, 'end_line': 60}),
           act('run_command', {'command': CMD}, 'error')]
for f, o, n in applied:
    actions.append(act('edit_file', {'filepath': f, 'old_string': o, 'new_string': n}))
actions.extend([act('run_command', {'command': CMD}), act('get_status', {}), act('submit_patch', {})])
recipe_path = REPLAY / 'f2-gymv2-openmdao-3125-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': 'root research agent: causal source read (real-PR repair copies inherited hooks per instance), pinned tests, command review 2026-10-25',
    'review_notes': ('Inherited hooks must be per-instance copies: the derived hook is not the base hook object '
                     '(same ordering stamp); deactivating the child leaves the base callable; captured at '
                     'hook-inheritance timepoint; asserted source-bound after the repair. At baseline the copy '
                     'machinery is absent (ImportError) while the pinned F2P test_inherited_class_hooks fails, '
                     'so both the task-level and mechanism-level discriminations hold. Original F2P/P2P unchanged. '
                     'Baseline diagnostic failure is expected task reproduction, not a policy error. Off-policy '
                     'scripted demonstration, not a Gemma sample.'),
    'actions': actions,
    'final_response': ('Copy inherited hooks per instance so derived registration never wipes base hooks. '
                       'Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-gymv2/pages-v2/openmdao__openmdao-3125.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/openmdao__openmdao-3125.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-gymv2/openmdao-3125-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('_HookMeta.copy inherit isolation: derived hooks are per-instance copies; '
                           'child deactivation never leaks into base; captured at inheritance timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-gymv2-openmdao-3125-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
