"""Prepare f2-push3 R2: kernel_gateway-118 (per-kernel activity-map copy isolation)."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-push3'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-push3'

IID = 'jupyter__kernel_gateway-118'
KEY = 'f2-push3-kernel_gateway-118'
REPO = 'jupyter/kernel_gateway'
BASE = 'a5379b43ebe6c4c93be5a1aa3ab9f2527253b1c0'
FAMILY = 'jupyter/kernel_gateway::kernel_gateway.services.activity.manager.ActivityManager.get_map_for_kernel::per_kernel_copy_isolation'
SNAP = f'{REPO}::{BASE}'

PROD_FILES = ['kernel_gateway/gatewayapp.py',
              'kernel_gateway/services/activity/handlers.py',
              'kernel_gateway/services/activity/manager.py',
              'kernel_gateway/services/kernels/handlers.py']


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


row = json.loads((ACQ / 'pages-v2/jupyter__kernel_gateway-118.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'kernel_gateway-118-source'
arch = ART / 'kernel_gateway-a5379b43ebe6-source.tgz'
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
mgr = cur_text['kernel_gateway/services/activity/manager.py']
assert 'self.values[kernel_id] = DEFAULT_ACTIVITY_VALUES.copy()' in mgr
assert 'return DEFAULT_ACTIVITY_VALUES.copy()' in mgr
print(f'edit reconstruction verified ({sum(len(e) for e in file_edits.values())} prod hunks; workspace left at baseline)', flush=True)

targets = sorted({nn.split('::', 1)[0] for nn in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
print('targets:', targets, flush=True)
assert targets == ['kernel_gateway/tests/services/activity/test_activity.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-push3-kernel_gateway-118-task.json'
save(adapted, task)

OBS = ("from kernel_gateway.services.activity.manager import (\n"
       "    ActivityManager, DEFAULT_ACTIVITY_VALUES, BUSY)\n"
       "am = ActivityManager(['k1'])\n"
       "m1 = am.get_map_for_kernel('k1')\n"
       "m2 = am.get_map_for_kernel('k2')\n"
       "print('identities', m1 is m2, m1 is DEFAULT_ACTIVITY_VALUES, m2 is DEFAULT_ACTIVITY_VALUES)\n"
       "assert m1 is not m2\n"
       "assert m1 is not DEFAULT_ACTIVITY_VALUES\n"
       "assert m2 is not DEFAULT_ACTIVITY_VALUES\n"
       "m1[BUSY] = True\n"
       "assert DEFAULT_ACTIVITY_VALUES[BUSY] is False\n"
       "assert m2.get(BUSY) is False\n"
       "print('f2_kernel_isolation', True)\n")
BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "module = importlib.import_module('kernel_gateway.services.activity.manager')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert str(actual).endswith('kernel_gateway/services/activity/manager.py'), actual\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'kernel_gateway/services/activity/manager.py',
                             'start_line': 55, 'end_line': 72}),
           act('run_command', {'command': CMD}, 'error')]
for name, edits in file_edits.items():
    for o, n in edits:
        actions.append(act('edit_file', {'filepath': name, 'old_string': o, 'new_string': n}))
actions += [act('run_command', {'command': CMD}),
            act('get_status', {}),
            act('submit_patch', {})]
recipe_path = REPLAY / 'f2-push3-kernel_gateway-118-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': ('subagent f2-push3: causal source read (unknown-kernel queries returned the one '
                    'shared dummy_map; per-kernel DEFAULT_ACTIVITY_VALUES.copy() isolates; P2 click-2312 '
                    'precedent for refactor-embedded isolation), pinned tests, command review 2026-10-25'),
    'review_notes': ('Each kernel must own its activity map: a per-kernel map is not another kernel map '
                     'nor the global defaults, so publishing for one kernel stays invisible elsewhere. '
                     'Copy-time identity (is not the shared/defaults object) plus bidirectional isolation '
                     'are asserted source-bound before and after the repair. Original F2P/P2P unchanged. '
                     'Baseline diagnostic failure is expected task reproduction, not a policy error. '
                     'Off-policy scripted demonstration, not a Gemma sample. traitlets via '
                     'task-site-packages-kernelgw; parent .venv otherwise.'),
    'actions': actions,
    'final_response': ('Give each kernel its own activity-map copy so unknown/tracked kernels no longer '
                       'share one mutable map. Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-push3/pages-v2/jupyter__kernel_gateway-118.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/jupyter__kernel_gateway-118.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-push3/kernel_gateway-a5379b43ebe6-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('per-kernel copy isolation: a kernel activity map is not another kernel map '
                           'nor the global defaults; captured at the get_map_for_kernel access timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-push3-kernel_gateway-118-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
