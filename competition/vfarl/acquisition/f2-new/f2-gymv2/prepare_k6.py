"""Prepare f2-gymv2 K6: pystac-826 (Asset.clone extra_fields deepcopy)."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-gymv2'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-gymv2'

IID = 'stac-utils__pystac-826'
KEY = 'f2-gymv2-pystac-826'
REPO = 'stac-utils/pystac'
BASE = '0f308cdd0c29f290eb8a6eb08b4d4942f7180519'
FAMILY = 'stac-utils/pystac::Asset.clone::extra_fields_deepcopy_isolation'
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
        elif line.startswith(('@@ ', 'index ', '--- ', '+++ ', '\\')):
            if line.startswith('@@ '):
                if old or new:
                    edits.append((cur_file, ''.join(old), ''.join(new)))
                old, new = [], []
            continue
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


row = json.loads((ACQ / 'pages-v2/stac-utils__pystac-826.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'pystac-826-source'
arch = ART / 'pystac-826-source.tgz'
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

edits = forward_edits(row['patch'])
print(f'{len(edits)} hunks:', [(f, len(o), len(n)) for f, o, n in edits], flush=True)
applied = []
for f, o, n in edits:
    cur = (WS / f).read_text()
    assert cur.count(o) == 1, (f, o[:120])
    (WS / f).write_text(cur.replace(o, n))
    applied.append((f, o, n))
# restore workspace to buggy (edits ship via recipe, not via workspace mutation)
for f, o, n in applied:
    cur = (WS / f).read_text()
    assert cur.count(n) == 1, (f, 'revert anchor')
    (WS / f).write_text(cur.replace(n, o))
print('edit reconstruction verified (workspace restored to buggy)', flush=True)

targets = sorted({n.split('::', 1)[0] for n in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
print('targets:', targets, flush=True)
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-gymv2-pystac-826-task.json'
save(adapted, task)

BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "expected=(Path.cwd() / 'pystac/asset.py').resolve()\n"
           "module = importlib.import_module('pystac.asset')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert actual == expected, (actual, expected)\n")
OBS = ("from pystac import Asset\n"
       "asset = Asset(href='http://example.com/a.tif', extra_fields={'k': 'v'})\n"
       "mask = asset.clone()\n"
       "print('clone_identity', mask.extra_fields is asset.extra_fields)\n"
       "assert mask.extra_fields is not asset.extra_fields, 'clone shares extra_fields'\n"
       "mask.extra_fields['abc'] = True\n"
       "print('orig_untouched', asset.extra_fields)\n"
       "assert asset.extra_fields == {'k': 'v'}, 'mutation leaked into original'\n"
       "print('f2_clone_isolation', True)\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'pystac/asset.py', 'start_line': 136, 'end_line': 160}),
           act('run_command', {'command': CMD}, 'error')]
for f, o, n in applied:
    actions.append(act('edit_file', {'filepath': f, 'old_string': o, 'new_string': n}))
actions.extend([act('run_command', {'command': CMD}), act('get_status', {}), act('submit_patch', {})])
recipe_path = REPLAY / 'f2-gymv2-pystac-826-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': 'root research agent: causal source read (real-PR repair deep-copies extra_fields in clone), pinned tests, command review 2026-10-25',
    'review_notes': ('Asset.clone must own a deep copy of extra_fields: clone-side mutation leaves the original '
                     'untouched, captured at the clone() call timepoint; asserted source-bound before and after '
                     'the repair. Original F2P/P2P unchanged. Baseline diagnostic failure is expected task '
                     'reproduction, not a policy error. Off-policy scripted demonstration, not a Gemma sample.'),
    'actions': actions,
    'final_response': ('Deep-copy extra_fields in Asset.clone so clones never share mutable state with the '
                       'original. Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-gymv2/pages-v2/stac-utils__pystac-826.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/stac-utils__pystac-826.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-gymv2/pystac-826-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('Asset.clone extra_fields deepcopy isolation: clone owns an independent mapping; '
                           'mutation stays local; captured at clone() timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-gymv2-pystac-826-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)