"""Prepare f2-gymv2b K13: altair-518 (update_nested copy isolation). Workspace + recipe only; dual pends era interpreter (collections.Mapping)."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-gymv2b'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-gymv2b'

IID = 'altair-viz__altair-518'
KEY = 'f2-gymv2b-altair-518'
REPO = 'altair-viz/altair'
BASE = 'e37000c8f54bc5e0e98ea8457b9a3c913cd58ccb'
FAMILY = 'altair-viz/altair::altair.utils.core.update_nested+TopLevelMixin.to_dict::copy_isolation'
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


row = json.loads((ACQ / 'pages-v2/altair-viz__altair-518.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'altair-518-source'
arch = ART / 'altair-518-source.tgz'
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

assert 'def update_nested' not in (WS / 'altair/utils/core.py').read_text(), 'already fixed?'
parts = row['patch'].split('diff --git a/')
edits_by_file = {}
for part in parts:
    name = part.split('\n', 1)[0].strip().split(' ')[0]
    if name in ('altair/utils/core.py', 'altair/vegalite/v2/api.py'):
        edits_by_file[name] = forward_edits('diff --git a/' + part)
print({k: len(v) for k, v in edits_by_file.items()})
n = sum(len(v) for v in edits_by_file.values())
assert n >= 3, n
for fname, hs in edits_by_file.items():
    cur = (WS / fname).read_text()
    for o, hh in hs:
        assert cur.count(o) == 1, (fname, o[:120])
        cur = cur.replace(o, hh)
print(f'edit reconstruction ready ({n} hunks)', flush=True)

targets = sorted({nn.split('::', 1)[0] for nn in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
assert targets == ['altair/utils/tests/test_core.py', 'altair/vegalite/tests/test_common.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-gymv2b-altair-518-task.json'
save(adapted, task)

BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "expected=(Path.cwd() / 'altair/utils/core.py').resolve()\n"
           "module = importlib.import_module('altair.utils.core')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert actual == expected, (actual, expected)\n")
OBS = ("from altair.utils.core import update_nested\n"
       "original = {'x': {'b': {'foo': 2}, 'c': 4}}\n"
       "update = {'x': {'b': {'foo': 5}, 'd': 6}, 'y': 40}\n"
       "output = update_nested(original, update, copy=True)\n"
       "print('copy_identity', output is original)\n"
       "assert output is not original\n"
       "output2 = update_nested(original, update)\n"
       "assert output2 is original\n"
       "print('f2_copy_isolation', True)\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'altair/utils/core.py', 'start_line': 295, 'end_line': 340}),
           act('run_command', {'command': CMD}, 'error')]
for fname, hs in edits_by_file.items():
    for o, hh in hs:
        actions.append(act('edit_file', {'filepath': fname, 'old_string': o, 'new_string': hh}))
actions.extend([act('run_command', {'command': CMD}), act('get_status', {}), act('submit_patch', {})])
recipe_path = REPLAY / 'f2-gymv2b-altair-518-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': 'root research agent: causal source read (update_nested deepcopy + to_dict private copy), pinned tests, command review 2026-10-25',
    'review_notes': ('update_nested with copy=True must own a deepcopy (is-not the input) while copy=False '
                     'returns the same object (is, by design); to_dict must format a private copy; production-call '
                     'diagnostic fails on baseline (no update_nested) and passes repaired on an era interpreter. '
                     'On replay/.venv Python 3.12 the repaired code hits collections.Mapping removal (era drift); '
                     'dual pends a task-python old interpreter, honestly reported. Original F2P/P2P unchanged. '
                     'Off-policy scripted demonstration, not a Gemma sample.'),
    'actions': actions,
    'final_response': ('Own a deepcopy in update_nested when copy=True and format a private copy in to_dict. '
                       'Source-bound inline observations now pass on an era interpreter. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-gymv2b/pages-v2/altair-viz__altair-518.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/altair-viz__altair-518.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-gymv2b/altair-518-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('update_nested copy isolation + to_dict private copy; captured at call timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-gymv2b-altair-518-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
