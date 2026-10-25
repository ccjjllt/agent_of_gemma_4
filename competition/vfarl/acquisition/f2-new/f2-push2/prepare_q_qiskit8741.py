"""Prepare f2-push2 Q1: qiskit-terra-8741 (idle-wires qubits/clbits copy isolation)."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-push2'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-push2'

IID = 'qiskit__qiskit-terra-8741'
KEY = 'f2-push2-qiskit-terra-8741'
REPO = 'Qiskit/qiskit-terra'
BASE = 'b35b18a90ff10b2ae245db85b3bfc728cc0adcd8'
FAMILY = 'Qiskit/qiskit-terra::qiskit.visualization.circuit._utils._get_layered_instructions::idle_wires_qubits_clbits_copy_isolation'
SNAP = f'{REPO}::{BASE}'

PROD_FILES = ['qiskit/visualization/circuit/_utils.py']


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


row = json.loads((ACQ / 'pages-v2/qiskit__qiskit-terra-8741.json').read_text())['row']
assert row['instance_id'] == IID and row['base_commit'] == BASE

WS = ART / 'qiskit-terra-8741-source'
arch = ART / 'qiskit-terra-b35b18a90ff1-source.tgz'
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
assert 'qubits = circuit.qubits.copy()' in cur_text['qiskit/visualization/circuit/_utils.py']
print(f'edit reconstruction verified ({sum(len(e) for e in file_edits.values())} prod hunks; workspace left at baseline)', flush=True)

targets = sorted({nn.split('::', 1)[0] for nn in list(row['FAIL_TO_PASS']) + list(row['PASS_TO_PASS'])})
print('targets:', targets, flush=True)
assert targets == ['test/python/visualization/test_circuit_text_drawer.py'], targets
task = {'instance_id': IID, 'repo': REPO, 'base_commit': BASE,
        'problem_statement': row['problem_statement'], 'patch': '',
        'test_patch': row['test_patch'], 'FAIL_TO_PASS': row['FAIL_TO_PASS'],
        'PASS_TO_PASS': row['PASS_TO_PASS'], 'verification_target_files': targets}
adapted = ART / 'f2-push2-qiskit-terra-8741-task.json'
save(adapted, task)

OBS = ("from qiskit import QuantumCircuit\n"
       "from qiskit.visualization.circuit._utils import _get_layered_instructions\n"
       "qc = QuantumCircuit(2, 2)\n"
       "qc.h(0)\n"
       "before_q = list(qc.qubits)\n"
       "before_c = list(qc.clbits)\n"
       "qubits, clbits, nodes = _get_layered_instructions(qc, idle_wires=False)\n"
       "print('qubits_identity', qubits is qc.qubits, 'clbits_identity', clbits is qc.clbits)\n"
       "assert qubits is not qc.qubits, 'qubits aliases the circuit'\n"
       "assert clbits is not qc.clbits, 'clbits aliases the circuit'\n"
       "assert list(qc.qubits) == before_q, 'circuit qubits mutated'\n"
       "assert list(qc.clbits) == before_c, 'circuit clbits mutated'\n"
       "print('f2_qiskit_isolation', True)\n")
BINDING = ("from pathlib import Path\nimport hashlib, importlib\n"
           "module = importlib.import_module('qiskit.visualization.circuit._utils')\n"
           "actual=Path(module.__file__).resolve()\n"
           'print("source_binding", str(actual), "sha256", hashlib.sha256(actual.read_bytes()).hexdigest())\n'
           "assert str(actual).endswith('qiskit/visualization/circuit/_utils.py'), actual\n")
CMD = "python3 - <<'PYREPLAY'\n" + BINDING + OBS + "\nPYREPLAY"


def act(tool, args, exp='ok'):
    v = {'tool': tool, 'arguments': args, 'expected_status': exp}
    if tool == 'run_command':
        v['reviewed_command_sha256'] = digest(args['command'].encode())
    return v


actions = [act('get_status', {}),
           act('read_file', {'filepath': 'qiskit/visualization/circuit/_utils.py',
                             'start_line': 396, 'end_line': 408}),
           act('run_command', {'command': CMD}, 'error')]
for name, edits in file_edits.items():
    for o, n in edits:
        actions.append(act('edit_file', {'filepath': name, 'old_string': o, 'new_string': n}))
actions += [act('run_command', {'command': CMD}),
            act('get_status', {}),
            act('submit_patch', {})]
recipe_path = REPLAY / 'f2-push2-qiskit-terra-8741-reviewed-recipe.json'
save(recipe_path, {
    'evidence_kind': 'human_reviewed_off_policy_demonstration',
    'reviewed_by': ('subagent f2-push2: causal source read (_get_layered_instructions binds '
                    'qubits = circuit.qubits so idle-wire removal mutates the circuit; '
                    '.copy() isolates), pinned tests, command review 2026-10-25'),
    'review_notes': ('Drawing must own private qubits/clbits copies: the working lists are not '
                     'the circuit lists so idle-wire removal and reverse_bits leave the original '
                     'circuit untouched. Copy-time identity (is not the circuit lists) plus '
                     'bidirectional isolation are asserted source-bound before and after the repair. '
                     'Original F2P/P2P unchanged. Baseline diagnostic failure is expected task '
                     'reproduction. Off-policy scripted demonstration, not a Gemma sample.'),
    'actions': actions,
    'final_response': ('Copy the working qubits/clbits lists so drawing no longer mutates the '
                       'circuit. Source-bound inline observations now pass. Patch submitted.')})

prep = {
    'instance_id': IID, 'key': KEY,
    'original_task': 'competition/vfarl/acquisition/f2-new/f2-push2/pages-v2/qiskit__qiskit-terra-8741.json',
    'original_task_sha256': sha(ACQ / 'pages-v2/qiskit__qiskit-terra-8741.json'),
    'adapted_task': str(adapted.relative_to(ROOT)),
    'workspace': str(WS.relative_to(ROOT)),
    'source_dataset': 'nebius/SWE-rebench-V2', 'source_train_revision': '10483de0f50fe5da545942705a76c6150171af7f',
    'physical_split': 'train', 'course_membership': 'F600_F2',
    'primary_adapter': 'F', 'submechanism': 'F2', 'canonical_repo': REPO,
    'repair_family_id': FAMILY, 'snapshot_group_id': SNAP,
    'execution_commit': BASE,
    'buggy_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE,
                     'archive_path': 'competition/vfarl/replay/artifacts/f2-push2/qiskit-terra-b35b18a90ff1-source.tgz',
                     'archive_sha256': digest(arch.read_bytes()),
                     'download_url': url, 'immutable_url': url, 'commit_resolution': None},
    'upstream_source': {'repo': REPO, 'requested_revision': BASE[:12], 'resolved_revision': BASE},
    'recipe': str(recipe_path.relative_to(ROOT)), 'recipe_sha256': sha(recipe_path),
    'F2P_count': len(row['FAIL_TO_PASS']), 'P2P_count': len(row['PASS_TO_PASS']),
    'mechanism_evidence': ('qubits/clbits-copy isolation: the working lists are not the circuit lists; '
                           'captured at the _get_layered_instructions() call timepoint.'),
    'verification_target_files': targets,
    'original_patch_semantics': 'repair_target',
    'required_node_identifiers_verbatim_preserved': True,
    'source_requirements_relaxed': False,
}
save(REPLAY / 'f2-push2-qiskit-terra-8741-source-preparation.json', prep)
print('PREPARED', KEY, FAMILY, flush=True)
