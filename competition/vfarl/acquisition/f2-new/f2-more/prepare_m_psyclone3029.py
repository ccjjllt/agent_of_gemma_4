"""Prepare f2-more M1: psyclone-3029 (get_node_list single-schedule children copy)."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-more'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-more'

IID = 'stfc__psyclone-3029'
KEY = 'f2-more-psyclone-3029'
REPO = 'stfc/PSyclone'
BASE = '70a0a9c3f0b039f5f2dc8c0c77c4f7209937772f'
FAMILY = 'stfc/PSyclone::psyclone.psyir.transformations.region_trans.RegionTrans.get_node_list::single_schedule_children_copy'
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


row = json.loads((ACQ / 'pages-v2/stfc__psyclone-3029.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'psyclone-3029-source'
arch = ART / 'psyclone-3029-source.tgz'
ART.mkdir(parents=True, exist_ok=True)
url = f'https://codeload.github.com/{REPO}/tar.gz/{BASE}'
assert arch.exists(), f'missing {arch} (pre-downloaded)'
with tarfile.open(arch) as bundle:
    members = bundle.getmembers()
    top = members[0].name.split('/')[0]
    seen_lower = {}
    collisions = []
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
            # Case-insensitive macOS FS: repo holds .F90/.f90 twins (LFRic
            # fixtures, 46 groups, all under test_files/lfric/infrastructure/,
            # none referenced by the 41 target tests). First member wins;
            # every collision recorded honestly.
            key = str(rel).lower()
            if key in seen_lower:
                collisions.append({'kept': seen_lower[key], 'skipped': m.name,
                                   'kept_size': seen_lower[key + '#size'],
                                   'skipped_size': m.size})
                continue
            seen_lower[key] = m.name
            seen_lower[key + '#size'] = m.size
            payload = bundle.extractfile(m).read()
            if tgt.exists() and tgt.read_bytes() != payload:
                raise ValueError(f'altered {tgt}')
            tgt.parent.mkdir(parents=True, exist_ok=True)
            tgt.write_bytes(payload)
    save(ART / 'psyclone-3029-case-collisions.json',
         {'instance_id': IID, 'reason': 'case-insensitive local FS vs case-sensitive upstream (both twins kept upstream)',
          'count': len(collisions), 'policy': 'first-tar-member-wins', 'collisions': collisions})
    print(f'workspace ready ({len(collisions)} case-collisions recorded)', flush=True)

parts = row['patch'].split('diff --git a/')
prod_patch = ''
for part in parts:
    if part.startswith('src/psyclone/psyir/transformations/region_trans.py'):
        prod_patch = 'diff --git a/' + part
        break
assert prod_patch, 'prod patch part missing'
edits = forward_edits(prod_patch)
print(f'prod hunks: {len(edits)}', flush=True)
cur = (WS / 'src/psyclone/psyir/transformations/region_trans.py').read_text()
applied = 0
for o, n in edits:
    if o and cur.count(o) == 1:
        cur = cur.replace(o, n)
        applied += 1
    elif not o:
        raise ValueError('pure-addition hunk needs anchor review')
    else:
        raise ValueError(f'non-unique hunk anchor (count={cur.count(o)}): {o[:160]!r}')
# NOTE: workspace stays at BASELINE (buggy); the recipe's edit_file actions
# apply the repair inside the dual. Only verify reconstruction on a copy.
assert 'return nodes[0].children[:]' in cur
print(f'edit reconstruction verified ({applied} prod hunks; workspace left at baseline)', flush=True)

targets = sorted({nn.split('::', 1)[0] for nn in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
print('targets:', targets, flush=True)
assert targets == ['src/psyclone/tests/psyir/transformations/region_trans_test.py',
                   'src/psyclone/tests/psyir/transformations/transformations_test.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-more-psyclone-3029-task.json'
save(adapted, task)

OBS = ("from psyclone.psyir.nodes import Node, Schedule, IntrinsicCall, Reference\n"
       "from psyclone.psyir.symbols import DataSymbol, REAL_TYPE\n"
       "from psyclone.psyir.transformations import RegionTrans\n"
       "class MyRT(RegionTrans):\n"
       "    excluded_node_types = ()\n"
       "    @property\n"
       "    def name(self):\n"
       "        return 'my_rt'\n"
       "    def apply(self, node, options=None):\n"
       "        pass\n"
       "rt = MyRT()\n"
       "sched = Schedule()\n"
       "sym = DataSymbol('x', REAL_TYPE)\n"
       "sched.children = [IntrinsicCall.create(IntrinsicCall.Intrinsic.SIN, [Reference(sym)])]\n"
       "lst = rt.get_node_list([sched])\n"
       "print('list_identity', lst is not sched.children, len(lst))\n"
       "assert lst is not sched.children, 'node list shares the live children list'\n"
       "assert len(lst) == 1, lst\n"
       "assert lst[0] is sched.children[0], 'single-schedule list must unwrap to children'\n"
       "lst.append(Node.__new__(Node))\n"
       "assert len(sched.children) == 1, sched.children\n"
       "lst2 = rt.get_node_list(sched.children)\n"
       "assert lst2 is not sched.children and lst2 == list(sched.children)\n"
       "print('f2_node_list_isolation', True)\n")
BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "module = importlib.import_module('psyclone.psyir.transformations.region_trans')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert str(actual).endswith('psyclone/psyir/transformations/region_trans.py'), actual\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


o0, n0 = edits[0]
actions = [act('get_status', {}),
           act('read_file', {'filepath': 'src/psyclone/psyir/transformations/region_trans.py',
                             'start_line': 60, 'end_line': 115}),
           act('run_command', {'command': CMD}, 'error')]
for o, n in edits:
    actions.append(act('edit_file', {'filepath': 'src/psyclone/psyir/transformations/region_trans.py',
                                     'old_string': o, 'new_string': n}))
actions += [act('run_command', {'command': CMD}),
            act('get_status', {}),
            act('submit_patch', {})]
recipe_path = REPLAY / 'f2-more-psyclone-3029-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': ('subagent f2-more: causal source read (get_node_list list branch returns [sched] '
                    'instead of children copy; nodes[0].children[:] unwraps single-schedule lists), pinned tests, command review 2026-10-25'),
    'review_notes': ('get_node_list([Schedule]) must return a slice copy of the Schedule children: '
                     'the returned list is not the live children list so later reordering stays invisible, '
                     'while elements stay identical (shallow by design). Copy-time identity '
                     '(is not the live list) plus bidirectional isolation are asserted source-bound '
                     'before and after the repair. Original F2P/P2P unchanged. Baseline diagnostic '
                     'failure is expected task reproduction, not a policy error. Off-policy scripted '
                     'demonstration, not a Gemma sample.'),
    'actions': actions,
    'final_response': ('Unwrap single-Schedule lists to a children slice copy so the returned node list '
                       'owns no alias to the live tree order. Source-bound inline observations now pass. '
                       'Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-more/pages-v2/stfc__psyclone-3029.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/stfc__psyclone-3029.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-more/psyclone-3029-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('get_node_list single-schedule children copy: returned list owns no alias '
                           'to the live children list (is not), elements stay identical; captured at '
                           'get_node_list() timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-more-psyclone-3029-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
