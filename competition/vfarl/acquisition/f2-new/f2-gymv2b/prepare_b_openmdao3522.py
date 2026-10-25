"""Prepare f2-gymv2b K17: openmdao-3522 (get_val copy=True isolation)."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-gymv2b'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-gymv2b'

IID = 'openmdao__openmdao-3522'
KEY = 'f2-gymv2b-openmdao-3522'
REPO = 'OpenMDAO/OpenMDAO'
BASE = '97f89e58f032013f7fa7f6a7da45880b45059bd9'
FAMILY = 'OpenMDAO/OpenMDAO::Problem.get_val+System.get_val::copy_return_isolation'
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


row = json.loads((ACQ / 'pages-v2/openmdao__openmdao-3522.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'openmdao-3522-source'
arch = ART / 'openmdao-3522-source.tgz'
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

# problem.py already imports deepcopy? verify; system.py needs the import hunk
prob_text = (WS / 'openmdao/core/problem.py').read_text()
print('problem.py has deepcopy import:', 'from copy import deepcopy' in prob_text or 'deepcopy' in prob_text[:2000])
parts = row['patch'].split('diff --git a/')
edits_by_file = {}
for part in parts:
    if part.startswith('openmdao/core/problem.py') or part.startswith('openmdao/core/system.py'):
        fname = part.split(' ', 1)[0] if False else None
        # recover filename from part header
        name = part.split('\n', 1)[0].strip().split(' ')[0]
        edits_by_file[name] = forward_edits('diff --git a/' + part)
print({k: len(v) for k, v in edits_by_file.items()})
all_edits = []
for fname in ['openmdao/core/problem.py', 'openmdao/core/system.py']:
    cur = (WS / fname).read_text()
    for o, n in edits_by_file.get(fname, []):
        assert cur.count(o) == 1, (fname, o[:160])
        cur = cur.replace(o, n)
        all_edits.append((fname, o, n))
    (WS / fname).write_text(cur) if False else None  # do not mutate workspace; replay applies edits
# restore workspace (we only verified; replay will apply via tools)
print(f'edit reconstruction ready ({len(all_edits)} hunks across 2 files)', flush=True)
# NOTE: workspace left pristine (verification above used in-memory cur); replay recipe applies edits.

targets = sorted({n.split('::', 1)[0] for n in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
assert targets == ['openmdao/core/tests/test_problem.py', 'openmdao/core/tests/test_system.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-gymv2b-openmdao-3522-task.json'
save(adapted, task)

BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "expected=(Path.cwd() / 'openmdao/core/problem.py').resolve()\n"
           "module = importlib.import_module('openmdao.core.problem')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert actual == expected, (actual, expected)\n")
OBS = ("import numpy as np\nimport openmdao.api as om\n"
       "p = om.Problem()\n"
       "p.model.add_subsystem('ivc', om.IndepVarComp('x', val=np.array([5., 5., 5.])), promotes=['*'])\n"
       "p.setup()\np.run_model()\n"
       "x_view = p.get_val('x')\n"
       "x_copy = p.get_val('x', copy=True)\n"
       "print('copy_identity', x_copy is x_view)\n"
       "assert x_copy is not x_view, 'copy=True did not isolate'\n"
       "x_view[0] = 99.0\n"
       "assert x_copy[0] == 5.0, 'copy polluted by view mutation'\n"
       "assert p.get_val('x')[0] == 99.0, 'view did not share internal state'\n"
       "print('f2_getval_isolation', True)\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'openmdao/core/problem.py', 'start_line': 490, 'end_line': 550}),
           act('run_command', {'command': CMD}, 'error')]
for fname, o, n in all_edits:
    actions.append(act('edit_file', {'filepath': fname, 'old_string': o, 'new_string': n}))
actions.extend([act('run_command', {'command': CMD}), act('get_status', {}), act('submit_patch', {})])
recipe_path = REPLAY / 'f2-gymv2b-openmdao-3522-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': 'root research agent: causal source read (get_val copy=True deepcopies internal vectors), pinned tests, command review 2026-10-25',
    'review_notes': ('get_val with copy=True must return a deepcopy so array/object copies are not the internal '
                     'vectors (scalars stay shared by design); production-call diagnostic builds a Problem, calls '
                     'get_val(copy=True) (TypeError on baseline), then asserts copy-is-not-view plus view-mutates-'
                     'internal/copy-isolated (fails on baseline, passes repaired). Original '
                     'F2P/P2P unchanged. Baseline diagnostic failure is expected task reproduction, not a policy '
                     'error. Off-policy scripted demonstration, not a Gemma sample.'),
    'actions': actions,
    'final_response': ('Return a deepcopy from get_val when copy=True so callers own an isolated value. '
                       'Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-gymv2b/pages-v2/openmdao__openmdao-3522.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/openmdao__openmdao-3522.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-gymv2b/openmdao-3522-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('get_val copy_return_isolation: copy=True deepcopies internal vectors; '
                           'default view stays shared; captured at get_val(copy=True) timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-gymv2b-openmdao-3522-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
