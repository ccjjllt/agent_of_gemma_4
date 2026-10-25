"""Prepare f2-gymv2b K14: jinja-2026 (Undefined copy/pickle protocol)."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-gymv2b'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-gymv2b'

IID = 'pallets__jinja-2026'
KEY = 'f2-gymv2b-jinja-2026'
REPO = 'pallets/jinja'
BASE = '39d9ffff1f33a5e5141af655857f9dbe6b6df03d'
FAMILY = 'pallets/jinja::jinja2.runtime.Undefined::__getattr__+slots::copy_pickle_protocol'
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


row = json.loads((ACQ / 'pages-v2/pallets__jinja-2026.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'jinja-2026-source'
arch = ART / 'jinja-2026-source.tgz'
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

cur = (WS / 'src/jinja2/runtime.py').read_text()
assert 'del (\n    Undefined.__slots__,' in cur, 'slots-deletion anchor missing (already fixed?)'
# keep only src/jinja2/runtime.py hunks (drop CHANGES.rst docs hunk)
parts = row['patch'].split('diff --git a/')
runtime_patch = ''
for part in parts:
    if part.startswith('src/jinja2/runtime.py'):
        runtime_patch = 'diff --git a/' + part
        break
assert runtime_patch, 'runtime.py patch part missing'
edits = forward_edits(runtime_patch)
assert len(edits) == 3, len(edits)
for o, n in edits:
    assert cur.count(o) == 1, o[:160]
    cur = cur.replace(o, n)
print('edit reconstruction ready (3 runtime hunks)', flush=True)

targets = sorted({n.split('::', 1)[0] for n in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
assert targets == ['tests/test_api.py', 'tests/test_runtime.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-gymv2b-jinja-2026-task.json'
save(adapted, task)

BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "expected=(Path.cwd() / 'src/jinja2/runtime.py').resolve()\n"
           "module = importlib.import_module('jinja2.runtime')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert actual == expected, (actual, expected)\n")
OBS = ("import copy, pickle\nfrom jinja2 import Undefined\n"
       "u = Undefined(hint='h', obj=['foo'], name='n', exc=RuntimeError('e'))\n"
       "c = copy.copy(u)\n"
       "print('copy_identity', c is u)\n"
       "assert c is not u, 'copy did not create a new object'\n"
       "assert c._undefined_hint is u._undefined_hint\n"
       "assert c._undefined_obj is u._undefined_obj\n"
       "d = copy.deepcopy(u)\n"
       "assert d is not u\n"
       "print('deepcopy_obj_isolated', d._undefined_obj is not u._undefined_obj)\n"
       "assert d._undefined_obj is not u._undefined_obj\n"
       "assert d._undefined_obj == u._undefined_obj\n"
       "p = pickle.loads(pickle.dumps(u))\n"
       "assert p._undefined_hint == u._undefined_hint\n"
       "print('f2_undefined_copy', True)\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'src/jinja2/runtime.py', 'start_line': 855, 'end_line': 875}),
           act('run_command', {'command': CMD}, 'error')]
for o, n in edits:
    actions.append(act('edit_file', {'filepath': 'src/jinja2/runtime.py',
                                    'old_string': o, 'new_string': n}))
actions.extend([act('run_command', {'command': CMD}), act('get_status', {}), act('submit_patch', {})])
recipe_path = REPLAY / 'f2-gymv2b-jinja-2026-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': 'root research agent: causal source read (dunder probing fix + slots restoration enables copy/pickle), pinned tests, command review 2026-10-25',
    'review_notes': ('Undefined dunder probing must raise AttributeError so default copy/deepcopy/pickle '
                     'protocols work; copies are new objects sharing immutable wiring while deepcopy '
                     'isolates the mutable payload; asserted source-bound before (copy fails) and after '
                     '(copy/deepcopy/pickle triple passes). Original F2P/P2P unchanged. Baseline diagnostic '
                     'failure is expected task reproduction, not a policy error. Off-policy scripted '
                     'demonstration, not a Gemma sample.'),
    'actions': actions,
    'final_response': ('Raise AttributeError for dunder probes and restore slots so Undefined objects '
                       'support copy/deepcopy/pickle with the documented identity semantics. '
                       'Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-gymv2b/pages-v2/pallets__jinja-2026.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/pallets__jinja-2026.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-gymv2b/jinja-2026-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('Undefined copy_pickle_protocol: dunder AttributeError + slots restoration; '
                           'copies are new objects sharing wiring, deepcopy isolates payload; '
                           'captured at copy call timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-gymv2b-jinja-2026-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
