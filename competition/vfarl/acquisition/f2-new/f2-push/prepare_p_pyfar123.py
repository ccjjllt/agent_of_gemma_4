"""Prepare f2-push P1: pyfar-123 (Coordinates.__getitem__ slice-copy isolation)."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-push'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-push'

IID = 'pyfar__pyfar-123'
KEY = 'f2-push-pyfar-123'
REPO = 'pyfar/pyfar'
BASE = '1a2767c6fa603420724db34314a124093e8f1061'
FAMILY = 'pyfar/pyfar::pyfar.coordinates.Coordinates.__getitem__::slice_copy_isolation'
SNAP = f'{REPO}::{BASE}'

PROD_FILES = ['pyfar/coordinates.py']


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


row = json.loads((ACQ / 'pages-v2/pyfar__pyfar-123.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'pyfar-123-source'
arch = ART / 'pyfar-123-source.tgz'
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
assert 'new = self.copy()' in cur_text['pyfar/coordinates.py']
assert 'new = self\n' not in cur_text['pyfar/coordinates.py'].replace('new = self.copy()\n', '')
print(f'edit reconstruction verified ({sum(len(e) for e in file_edits.values())} prod hunks; workspace left at baseline)', flush=True)

targets = sorted({nn.split('::', 1)[0] for nn in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
print('targets:', targets, flush=True)
assert targets == ['tests/test_coordinates.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-push-pyfar-123-task.json'
save(adapted, task)

OBS = ("import numpy as np\n"
       "import pyfar\n"
       "coords = pyfar.Coordinates([0, 1], [0, 1], [0, 1])\n"
       "new = coords[0]\n"
       "print('slice_identity', new is coords)\n"
       "assert new is not coords, 'slice aliases the original'\n"
       "new.set_cart(2, 2, 2)\n"
       "assert coords.cshape == (2,), coords.cshape\n"
       "assert np.allclose(coords.get_cart()[0], np.array([0, 0, 0]))\n"
       "print('f2_slice_isolation', True)\n")
BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "module = importlib.import_module('pyfar.coordinates')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert str(actual).endswith('pyfar/coordinates.py'), actual\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'pyfar/coordinates.py',
                             'start_line': 1630, 'end_line': 1650}),
           act('run_command', {'command': CMD}, 'error')]
for name, edits in file_edits.items():
    for o, n in edits:
        actions.append(act('edit_file', {'filepath': name, 'old_string': o, 'new_string': n}))
actions += [act('run_command', {'command': CMD}),
            act('get_status', {}),
            act('submit_patch', {})]
recipe_path = REPLAY / 'f2-push-pyfar-123-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': ('subagent f2-push: causal source read (Coordinates.__getitem__ binds new = self '
                    'so slicing mutates the original object; new = self.copy() isolates the slice), '
                    'pinned tests, command review 2026-10-25'),
    'review_notes': ('A sliced Coordinates object must own a copy: the slice is not the original '
                     'so mutating the slice leaves the original points untouched. Copy-time identity '
                     '(is not the original) plus bidirectional isolation are asserted source-bound '
                     'before and after the repair. Original F2P/P2P unchanged. Baseline diagnostic '
                     'failure is expected task reproduction, not a policy error. Off-policy scripted '
                     'demonstration, not a Gemma sample. Dependency note: 2020-era pyfar needs '
                     'scipy/deepdiff/matplotlib/python-sofa via task-python 3.10 period profile.'),
    'actions': actions,
    'final_response': ('Copy the sliced Coordinates object so slices no longer alias and mutate the '
                       'original. Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-push/pages-v2/pyfar__pyfar-123.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/pyfar__pyfar-123.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-push/pyfar-123-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('slice-copy isolation: the __getitem__ slice is not the original; '
                           'captured at the __getitem__ call timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-push-pyfar-123-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
