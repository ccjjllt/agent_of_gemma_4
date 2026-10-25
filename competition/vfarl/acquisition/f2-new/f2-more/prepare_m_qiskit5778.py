"""Prepare f2-more M2: qiskit-terra-5778 (per-instruction op.copy() duration isolation)."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-more'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-more'

IID = 'qiskit__qiskit-terra-5778'
KEY = 'f2-more-qiskit-terra-5778'
REPO = 'Qiskit/qiskit-terra'
BASE = '62b521316e62868fd827fd115e88a58f91d6915d'
FAMILY = 'Qiskit/qiskit-terra::qiskit.transpiler.passes.scheduling.ASAPSchedule+ALAPSchedule::per_instruction_op_copy_duration_isolation'
SNAP = f'{REPO}::{BASE}'

PROD_FILES = ['qiskit/transpiler/passes/scheduling/asap.py',
              'qiskit/transpiler/passes/scheduling/alap.py']


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


row = json.loads((ACQ / 'pages-v2/qiskit__qiskit-terra-5778.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'qiskit-terra-5778-source'
arch = ART / 'qiskit-terra-5778-source.tgz'
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
assert all('need different op instance to store duration' in t for t in cur_text.values())
print(f'edit reconstruction verified ({sum(len(e) for e in file_edits.values())} prod hunks; workspace left at baseline)', flush=True)

targets = sorted({nn.split('::', 1)[0] for nn in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
print('targets:', targets, flush=True)
assert targets == ['test/python/circuit/test_scheduled_circuit.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-more-qiskit-terra-5778-task.json'
save(adapted, task)

OBS = ("from qiskit.circuit import QuantumRegister\n"
       "from qiskit.circuit.library import CXGate\n"
       "from qiskit.dagcircuit import DAGCircuit\n"
       "from qiskit.transpiler.passes.scheduling.asap import ASAPSchedule\n"
       "from qiskit.transpiler.instruction_durations import InstructionDurations\n"
       "from qiskit.test.mock.backends import FakeParis\n"
       "backend = FakeParis()\n"
       "durs = InstructionDurations.from_backend(backend)\n"
       "qr = QuantumRegister(3, 'q')\n"
       "shared = CXGate()\n"
       "dag = DAGCircuit()\n"
       "dag.add_qreg(qr)\n"
       "dag.apply_operation_back(shared, [qr[0], qr[1]], [])\n"
       "dag.apply_operation_back(shared, [qr[1], qr[2]], [])\n"
       "sched = ASAPSchedule(durs)\n"
       "sched.property_set['time_unit'] = 'dt'\n"
       "out = sched.run(dag)\n"
       "cxs = [(node, node.op) for node in out.op_nodes() if node.name == 'cx']\n"
       "assert len(cxs) == 2, len(cxs)\n"
       "print('op_identity', cxs[0][1] is cxs[1][1], cxs[0][1].duration, cxs[1][1].duration)\n"
       "assert cxs[0][1] is not cxs[1][1], 'scheduled gates share one op instance'\n"
       "assert cxs[0][1].duration != cxs[1][1].duration, (cxs[0][1].duration, cxs[1][1].duration)\n"
       "print('f2_sched_op_isolation', True)\n")
BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "module = importlib.import_module('qiskit.transpiler.passes.scheduling.asap')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert str(actual).endswith('qiskit/transpiler/passes/scheduling/asap.py'), actual\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'qiskit/transpiler/passes/scheduling/asap.py',
                             'start_line': 60, 'end_line': 95}),
           act('run_command', {'command': CMD}, 'error')]
for name, edits in file_edits.items():
    for o, n in edits:
        actions.append(act('edit_file', {'filepath': name, 'old_string': o, 'new_string': n}))
actions += [act('run_command', {'command': CMD}),
            act('get_status', {}),
            act('submit_patch', {})]
recipe_path = REPLAY / 'f2-more-qiskit-terra-5778-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': ('subagent f2-more: causal source read (ALAP/ASAP schedule writes duration '
                    'onto the shared node.op instance so the second CX overwrites the first; '
                    'per-node node.op.copy() isolates durations), pinned tests, command review 2026-10-25'),
    'review_notes': ('Each scheduled instruction must own a per-node op copy: the scheduled op '
                     'is not the original shared op so per-gate durations stay visible only on '
                     'their own node. Copy-time identity (is not the shared instance) plus '
                     'bidirectional isolation are asserted source-bound before and after the repair. '
                     'Original F2P/P2P unchanged. Baseline diagnostic failure is expected task '
                     'reproduction, not a policy error. Off-policy scripted demonstration, not a '
                     'Gemma sample. Heavy dependency note: terra-0.17 needs Cython stochastic_swap '
                     'build + retworkx; dual pending env build.'),
    'actions': actions,
    'final_response': ('Copy each scheduled op per instruction so durations no longer clobber '
                       'across gates sharing one instance. Source-bound inline observations now pass. '
                       'Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-more/pages-v2/qiskit__qiskit-terra-5778.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/qiskit__qiskit-terra-5778.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-more/qiskit-terra-5778-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('per-instruction op copy duration isolation: scheduled op is not the '
                           'shared instance; captured at the scheduling-pass write timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-more-qiskit-terra-5778-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
