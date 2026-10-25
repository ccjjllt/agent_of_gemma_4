"""Prepare f2-gymv2 K9: tablib-579 (Row.copy + export format_row isolation)."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-gymv2'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-gymv2'

IID = 'jazzband__tablib-579'
KEY = 'f2-gymv2-tablib-579'
REPO = 'jazzband/tablib'
BASE = '4fd9a68c24a2a95fc471d0fc29a2a911b469f93f'
FAMILY = 'jazzband/tablib::Dataset.export+Row.copy::export_row_copy_isolation'
SNAP = f'{REPO}::{BASE}'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(b):
    return hashlib.sha256(b).hexdigest()


def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def forward_edits(patch):
    """Split unified diff into (old_block, new_block) per hunk (context + removed -> context + added)."""
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


row = json.loads((ACQ / 'pages-v2/jazzband__tablib-579.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'tablib-579-source'
arch = ART / 'tablib-579-source.tgz'
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

cur = (WS / 'src/tablib/core.py').read_text()
assert 'def copy(self):\n        return Row(self._row.copy(), self.tags.copy())' not in cur
edits = forward_edits(row['patch'])
assert len(edits) == 4, len(edits)
for o, n in edits:
    assert cur.count(o) == 1, o[:120]
    cur = cur.replace(o, n)
print('edit reconstruction ready (4 hunks)', flush=True)

targets = sorted({n.split('::', 1)[0] for n in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
assert targets == ['tests/test_tablib.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-gymv2-tablib-579-task.json'
save(adapted, task)

BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "expected=(Path.cwd() / 'src/tablib/core.py').resolve()\n"
           "module = importlib.import_module('tablib.core')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert actual == expected, (actual, expected)\n")
OBS = ("import tablib\nds = tablib.Dataset()\nds.headers = ['text']\nds.append(['aAbBcC'])\n"
       "ds.add_formatter('text', lambda s: s[2:])\n"
       "first = ds.export('csv')\nsecond = ds.export('csv')\n"
       "print('export_stable', first == second)\n"
       "assert first == second, 'export mutated internal data'\n"
       "assert ds['text'][0] == 'aAbBcC', 'stored data corrupted'\n"
       "print('f2_export_isolation', True)\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'src/tablib/core.py', 'start_line': 265, 'end_line': 300}),
           act('run_command', {'command': CMD}, 'error')]
for o, n in edits:
    actions.append(act('edit_file', {'filepath': 'src/tablib/core.py',
                                    'old_string': o, 'new_string': n}))
actions.extend([act('run_command', {'command': CMD}), act('get_status', {}), act('submit_patch', {})])
recipe_path = REPLAY / 'f2-gymv2-tablib-579-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': 'root research agent: causal source read (real-PR repair formats a private Row.copy), pinned tests, command review 2026-10-25',
    'review_notes': ('Export must format a private Row.copy() so formatted rows are not the internal rows; '
                     'repeated exports never mutate internal data, captured at the export() call timepoint; '
                     'asserted source-bound before and after the repair. Original F2P/P2P unchanged. Baseline '
                     'diagnostic failure is expected task reproduction, not a policy error. Off-policy scripted '
                     'demonstration, not a Gemma sample.'),
    'actions': actions,
    'final_response': ('Format a private Row.copy() during export so internal data is never mutated. '
                       'Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-gymv2/pages-v2/jazzband__tablib-579.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/jazzband__tablib-579.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-gymv2/tablib-579-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('Row.copy + export format_row isolation: formatted rows are not internal rows; '
                           'export leaves internal data untouched; captured at export() timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-gymv2-tablib-579-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
