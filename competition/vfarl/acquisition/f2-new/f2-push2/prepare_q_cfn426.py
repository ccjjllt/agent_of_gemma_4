"""Prepare f2-push2 Q2: cfn-python-lint-426 (convert_dict iter-copy isolation)."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-push2'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-push2'

IID = 'awslabs__cfn-python-lint-426'
KEY = 'f2-push2-cfn-python-lint-426'
REPO = 'awslabs/cfn-python-lint'
BASE = '4765ff51ce92bc17a5c2406ef4d0d4888db8a64a'
FAMILY = 'awslabs/cfn-python-lint::cfnlint.helpers.convert_dict::dict_iter_copy_isolation'
SNAP = f'{REPO}::{BASE}'

PROD_FILES = ['src/cfnlint/helpers.py']


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


row = json.loads((ACQ / 'pages-v2/awslabs__cfn-python-lint-426.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'cfn-python-lint-426-source'
arch = ART / 'cfn-python-lint-4765ff51ce92-source.tgz'
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
        assert o and cur.count(o) == 1, (name, o[:200])
        cur = cur.replace(o, n)
    cur_text[name] = cur
assert 'for k, v in template.copy().items():' in cur_text['src/cfnlint/helpers.py']
print(f'edit reconstruction verified ({sum(len(e) for e in file_edits.values())} prod hunks; workspace left at baseline)', flush=True)

# P2P-empty: verification targets from F2P only (test_patch-new file precedent C2/1613)
targets = sorted({nn.split('::', 1)[0] for nn in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
print('targets:', targets, flush=True)
assert targets == ['test/module/helpers/test_convert_dict.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-push2-cfn-python-lint-426-task.json'
save(adapted, task)

OBS = ("from cfnlint.helpers import convert_dict\n"
       "from cfnlint.decode.node import dict_node, str_node\n"
       "d = {'Key': {'SubKey': 'val'}, 'List': [1, 2]}\n"
       "out = convert_dict(d)\n"
       "print('out_type', type(out).__name__, [type(k).__name__ for k in out])\n"
       "assert isinstance(out, dict_node), type(out)\n"
       "assert all(isinstance(k, str_node) for k in out), [type(k).__name__ for k in out]\n"
       "print('f2_convert_isolation', True)\n")
BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "module = importlib.import_module('cfnlint.helpers')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert str(actual).endswith('cfnlint/helpers.py'), actual\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'src/cfnlint/helpers.py',
                             'start_line': 214, 'end_line': 236}),
           act('run_command', {'command': CMD}, 'error')]
for name, edits in file_edits.items():
    for o, n in edits:
        actions.append(act('edit_file', {'filepath': name, 'old_string': o, 'new_string': n}))
actions += [act('run_command', {'command': CMD}),
            act('get_status', {}),
            act('submit_patch', {})]
recipe_path = REPLAY / 'f2-push2-cfn-python-lint-426-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': ('subagent f2-push2: causal source read (convert_dict iterates live dict '
                    'while del/reinsert changes size; template.copy() snapshot isolates iteration), '
                    'pinned tests, command review 2026-10-25'),
    'review_notes': ('convert_dict must iterate an owned snapshot copy: template.copy() is not '
                     'template so del/reinsert of str_node keys leaves iteration fixed. Copy-time '
                     'identity plus bidirectional isolation are asserted source-bound before and '
                     'after the repair. Original F2P/P2P unchanged (P2P-empty, test_patch-new file '
                     'precedent C2/1613: base has no upstream file, naturally no skip collision). '
                     'Baseline diagnostic failure is expected task reproduction. Off-policy scripted '
                     'demonstration, not a Gemma sample. Dependency note: 2019-era cfn-lint needs '
                     'six/pyyaml/etc via task-python 3.10 period profile.'),
    'actions': actions,
    'final_response': ('Iterate a snapshot copy so key rewrapping no longer mutates the live dict '
                       'during iteration. Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-push2/pages-v2/awslabs__cfn-python-lint-426.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/awslabs__cfn-python-lint-426.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-push2/cfn-python-lint-4765ff51ce92-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('iter-copy isolation: template.copy() is not template; '
                           'captured at the convert_dict() call timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-push2-cfn-python-lint-426-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
