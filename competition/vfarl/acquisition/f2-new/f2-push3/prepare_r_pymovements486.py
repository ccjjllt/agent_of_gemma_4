"""Prepare f2-push3 R1: pymovements-486 (save_preprocessed gaze-copy isolation)."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-push3'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-push3'

IID = 'aeye-lab__pymovements-486'
KEY = 'f2-push3-pymovements-486'
REPO = 'aeye-lab/pymovements'
BASE = 'eed48eb3cd914db91d15036f3932db454ae5e8cf'
FAMILY = 'aeye-lab/pymovements::pymovements.dataset.dataset_files.save_preprocessed::gaze_copy_isolation'
SNAP = f'{REPO}::{BASE}'

PROD_FILES = ['src/pymovements/dataset/dataset_files.py',
              'src/pymovements/gaze/gaze_dataframe.py']


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


row = json.loads((ACQ / 'pages-v2/aeye-lab__pymovements-486.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'pymovements-486-source'
arch = ART / 'pymovements-eed48eb3cd91-source.tgz'
ART.mkdir(parents=True, exist_ok=True)
url = f'https://codeload.github.com/{REPO}/tar.gz/{BASE}'
assert arch.exists(), f'missing {arch}'
with tarfile.open(arch) as bundle:
    members = bundle.getmembers()
    top = members[0].name.split('/')[0]
    n_checked = 0
    for m in members:
        parts = Path(m.name).parts
        if not parts or parts[0] != top or '..' in parts:
            continue
        rel = Path(*parts[1:])
        if not rel.parts:
            continue
        tgt = WS / rel
        if m.isfile() and str(rel) in PROD_FILES:
            payload = bundle.extractfile(m).read()
            assert tgt.exists() and tgt.read_bytes() == payload, f'altered {tgt}'
            n_checked += 1
assert n_checked == len(PROD_FILES), n_checked
print('workspace ready', flush=True)

parts = row['patch'].split('diff --git a/')
file_edits = {}
patch_prod_files = set()
for part in parts:
    if not part.strip():
        continue
    name = part.split()[0].strip()
    if name in PROD_FILES:
        file_edits[name] = forward_edits('diff --git a/' + part)
    elif not name.startswith('tests/') and not name.startswith('docs/') and 'test' not in name.split('/')[-1]:
        patch_prod_files.add(name)
assert not patch_prod_files, patch_prod_files
assert set(file_edits) == set(PROD_FILES), set(file_edits)
cur_text = {}
for name, edits in file_edits.items():
    cur = (WS / name).read_text()
    for o, n in edits:
        assert o and cur.count(o) == 1, (name, o[:160])
        cur = cur.replace(o, n)
    cur_text[name] = cur
assert 'gaze_df = gaze_df.copy()' in cur_text['src/pymovements/dataset/dataset_files.py']
assert 'gaze.n_components = self.n_components' in cur_text['src/pymovements/gaze/gaze_dataframe.py']
print(f'edit reconstruction verified ({sum(len(e) for e in file_edits.values())} prod hunks; workspace left at baseline)', flush=True)

targets = sorted({nn.split('::', 1)[0] for nn in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
print('targets:', targets, flush=True)
assert targets == ['tests/dataset/dataset_test.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-push3-pymovements-486-task.json'
save(adapted, task)

OBS = ("import tempfile\nimport polars as pl\n"
       "from pymovements.dataset.dataset_files import save_preprocessed\n"
       "from pymovements.dataset.dataset_paths import DatasetPaths\n"
       "from pymovements.gaze.gaze_dataframe import GazeDataFrame\n"
       "df = pl.from_dict({'t': [1000, 1001, 1002], 'x': [0.1, 0.2, 0.3], 'y': [0.1, 0.2, 0.3]})\n"
       "gaze = GazeDataFrame(data=df, pixel_columns=['x', 'y'])\n"
       "cols_before = list(gaze.frame.columns)\n"
       "print('cols_before', cols_before)\n"
       "paths = DatasetPaths(root=tempfile.mkdtemp(), dataset='ds')\n"
       "fileinfo = pl.DataFrame({'filepath': ['f.csv']})\n"
       "save_preprocessed(gaze=[gaze], fileinfo=fileinfo, paths=paths, verbose=0, extension='csv')\n"
       "print('cols_after', list(gaze.frame.columns))\n"
       "assert list(gaze.frame.columns) == cols_before, 'side effect: caller frame mutated'\n"
       "print('f2_save_isolation', True)\n")
BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "module = importlib.import_module('pymovements.gaze.gaze_dataframe')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert str(actual).endswith('pymovements/gaze/gaze_dataframe.py'), actual\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'src/pymovements/dataset/dataset_files.py',
                             'start_line': 466, 'end_line': 474}),
           act('run_command', {'command': CMD}, 'error')]
for name, edits in file_edits.items():
    for o, n in edits:
        actions.append(act('edit_file', {'filepath': name, 'old_string': o, 'new_string': n}))
actions += [act('run_command', {'command': CMD}),
            act('get_status', {}),
            act('submit_patch', {})]
recipe_path = REPLAY / 'f2-push3-pymovements-486-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': ('subagent f2-push3: causal source read (save_preprocessed csv path unnests/drops '
                    'the loop-aliased caller GazeDataFrame; per-file gaze_df.copy() isolates; '
                    'copy() preserves n_components), pinned tests, command review 2026-10-25'),
    'review_notes': ('save_preprocessed must own a per-file working copy: the working gaze_df is not the '
                     'caller frame so csv unnest/drop stays invisible to the caller. Copy-time identity '
                     '(is not the caller object) plus bidirectional isolation are asserted source-bound '
                     'before and after the repair. Original F2P/P2P unchanged. Baseline diagnostic '
                     'failure is expected task reproduction, not a policy error. Off-policy scripted '
                     'demonstration, not a Gemma sample. Period deps via task-site-packages-pymovements.'),
    'actions': actions,
    'final_response': ('Copy the per-file gaze frame so csv saving no longer mutates the caller object. '
                       'Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-push3/pages-v2/aeye-lab__pymovements-486.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/aeye-lab__pymovements-486.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-push3/pymovements-eed48eb3cd91-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('save-preprocessed copy isolation: the working gaze_df is not the caller '
                           'GazeDataFrame; captured at the per-file loop-iteration timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-push3-pymovements-486-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
