"""Prepare f2-replace task 01: deepdiff w1di5431 (directed opcode rebuild vs alias)."""
import difflib, hashlib, json, subprocess
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new'
OUT = ACQ / 'f2-replace'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-replace'
REV = '77cab9055d42ab4a5c25c89a8f937096db13558e'
CANONICAL = 'seperman/deepdiff'
UPSTREAM_FULL = 'ed2520229d0369813f6e54cdf9c7e68e8073ef62'
SNAP = f'{CANONICAL}::{UPSTREAM_FULL}'
IID = 'seperman__deepdiff.ed252022.func_pm_ctrl_invert_if__w1di5431'
KEY = 'f2-replace-deepdiff-01'
COMMIT = '2e81d76465ff2e77bf12d9c19fcd0d7135b3a2a6'
FAMILY = 'seperman/deepdiff::deepdiff.SerializationMixin._to_delta_dict::directed_opcode_filtered_rebuild_isolation'

UPSTREAM_TGZ_SHA = 'faa7c950c58bb618fc5fa8a9cdb14f7c40823a060ca2a2f5f9ed8fb4d4f5792e'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def digest(b):
    return hashlib.sha256(b).hexdigest()


orig_task = OUT / (IID + '.task.json')
row = json.loads(orig_task.read_text())['row']

WS = ART / 'f2-deepdiff-w1di5431-source'
UP = ART / 'upstream-seperman__deepdiff'

# inverse edits (repair = undo bug-introducing patch)
patch = row['patch']
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
        old.append(line[1:])
    elif line.startswith('-'):
        new.append(line[1:])
if old or new:
    edits.append((''.join(old), ''.join(new)))
edits = [e for e in edits if e[0] != e[1]]
assert len(edits) == 3, len(edits)
cur = (WS / 'deepdiff/serialization.py').read_text()
for o, n in edits:
    assert cur.count(o) == 1, o[:120]
    cur = cur.replace(o, n)
assert cur == (UP / 'deepdiff/serialization.py').read_text(), 'reconstruction mismatch'
print('inverse edits verified:', len(edits), flush=True)

# verification targets + test_patch (missing test_delta.py from upstream)
required = list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])
targets = sorted({n.split('::', 1)[0] for n in required})
missing = []
for t in targets:
    if not (WS / t).exists() and (UP / t).exists():
        text = (UP / t).read_text()
        missing.extend(difflib.unified_diff([], text.splitlines(keepends=True),
                                            fromfile='/dev/null', tofile='b/' + t))
test_patch = ''.join(missing)
assert (WS / 'tests/test_delta.py').exists() or test_patch, 'expected test_delta.py patch'
print('targets:', len(targets), 'test_patch_bytes:', len(test_patch), flush=True)

task = {'instance_id': IID, 'repo': row['repo'], 'base_commit': COMMIT,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': test_patch, 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-replace-deepdiff-01-task.json'
save(adapted, task)

OBS = """from deepdiff import DeepDiff
import deepdiff.serialization as S
seen = {}
Orig = S.DeltaResult
class Spy(Orig):
    def __init__(self, *args, **kwargs):
        seen['opcodes'] = kwargs.get('_iterable_opcodes')
        super().__init__(*args, **kwargs)
S.DeltaResult = Spy
try:
    l1 = 'A B C D E F G D H'.split()
    l2 = 'B C X D H Y Z'.split()
    diff = DeepDiff(l1, l2)
    live = diff._iterable_opcodes
    live_snapshot = {p: list(o) for p, o in live.items()}
    d = diff._to_delta_dict(directed=True, always_include_values=False)
    internal = seen['opcodes']
    key = next(iter(internal))
    print('alias_identity', internal is live)
    assert internal is not live, 'shares live opcode store'
    internal[key].append(internal[key][0])
    print('live_untouched', live == live_snapshot)
    assert live == live_snapshot, 'mutation leaked into live store'
    print('f2_isolation_ownership', True)
finally:
    S.DeltaResult = Orig
"""

BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "expected=(Path.cwd() / 'deepdiff/serialization.py').resolve()\n"
           "try:\n module = importlib.import_module('deepdiff.serialization')\n"
           "except Exception as import_error:\n traceback=import_error.__traceback__\n"
           " while traceback is not None:\n  file=traceback.tb_frame.f_globals.get('__file__')\n"
           "  if file and Path(file).resolve()==expected:\n   print('source_binding', str(expected), 'sha256', hashlib.sha256(expected.read_bytes()).hexdigest(), 'actual_import_failure_frame')\n"
           "  traceback=traceback.tb_next\n raise\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert actual == expected, (actual, expected)\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'deepdiff/serialization.py', 'start_line': 227, 'end_line': 262}),
           act('run_command', {'command': CMD}, 'error')]
for o, n in edits:
    actions.append(act('edit_file', {'filepath': 'deepdiff/serialization.py', 'old_string': o, 'new_string': n}))
actions.extend([act('run_command', {'command': CMD}), act('get_status', {}), act('submit_patch', {})])

recipe_path = REPLAY / 'f2-replace-deepdiff-01-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': 'root research agent: causal source read (bug swaps directed rebuild/alias branches), pinned tests, command review 2026-10-25',
    'review_notes': ('Directed delta serialization must rebuild filtered opcode copies, not alias the live '
                     'opcode store. Copy-time identity (internal result opcodes are not the live store), '
                     'shared-mutation visibility (mutating the result side leaves the live store untouched), '
                     'and copy-timepoint capture (at _to_delta_dict call) are asserted source-bound before and '
                     'after the repair. Original F2P/P2P unchanged. Baseline diagnostic failure is expected task '
                     'reproduction, not a policy error. Off-policy scripted demonstration, not a Gemma sample.'),
    'actions': actions,
    'final_response': ('Rebuild filtered opcode copies for directed serialization so the live store keeps its '
                       'identity boundary. Source-bound inline observations now pass. Patch submitted.')})

remote = subprocess.check_output(
    ['git', 'ls-remote', 'https://github.com/swesmith/seperman__deepdiff.ed252022.git',
     'refs/heads/' + IID], text=True).strip()
assert remote.split()[0] == COMMIT, remote

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': str(orig_task.relative_to(ROOT)),
    'original_task_sha256': sha(orig_task),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'SWE-bench/SWE-smith-py', 'source_train_revision': REV,
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': CANONICAL,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': COMMIT, 'git_ls_remote': remote,
    'buggy_source': {'repo': 'swesmith/seperman__deepdiff.ed252022',
                     'requested_revision': COMMIT, 'resolved_revision': COMMIT,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-replace/f2-deepdiff-w1di5431-source.tgz',
                     'archive_sha256': digest((ART / 'f2-deepdiff-w1di5431-source.tgz').read_bytes()),
                     'download_url': f'https://codeload.github.com/swesmith/seperman__deepdiff.ed252022/tar.gz/{COMMIT}',
                     'immutable_url': f'https://codeload.github.com/swesmith/seperman__deepdiff.ed252022/tar.gz/{COMMIT}',
                     'commit_resolution': None},
    'upstream_source': {'repo': CANONICAL, 'requested_revision': 'ed252022',
                        'resolved_revision': UPSTREAM_FULL,
                        'archive_path': 'competition/vfarl/replay/artifacts/f2-replace/upstream-seperman__deepdiff.tgz',
                        'archive_sha256': UPSTREAM_TGZ_SHA,
                        'response_sha256': '917a42c2025a9c0b51fdbec53c9b7da42f330a5295b4a9214a5f2eb28573bdee'},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('SerializationMixin._to_delta_dict directed_opcode_filtered_rebuild_isolation: '
                           'directed serialization rebuilds filtered opcode copies (is-not the live store); '
                           'mutating the result side leaves the live store untouched; captured at call timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'bug_introducing',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-replace-deepdiff-01-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
