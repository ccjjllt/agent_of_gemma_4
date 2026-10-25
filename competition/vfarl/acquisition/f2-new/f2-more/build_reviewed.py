"""F2 f2-more: build reviewed-candidates.json + catalog.json for 2 KEEPs (M1/M2)."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
OUT = ROOT / 'competition/vfarl/acquisition/f2-new/f2-more'

DS = {'v2': ('nebius/SWE-rebench-V2', '10483de0f50fe5da545942705a76c6150171af7f', 'CC-BY-4.0')}
LICFILE = {
    'stfc/PSyclone': ('psyclone.LICENSE', 'BSD-3-Clause'),
    'Qiskit/qiskit-terra': ('qiskit-terra.LICENSE', 'Apache-2.0'),
}

M1_OBS = (
    "from psyclone.psyir.nodes import Node, Schedule, IntrinsicCall, Reference\n"
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
    "print('f2_node_list_isolation', True)\n"
)

M2_OBS = (
    "from qiskit.circuit import QuantumRegister\n"
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
    "print('f2_sched_op_isolation', True)\n"
)

MECH = {
    'stfc__psyclone-3029': dict(
        family='stfc/PSyclone::psyclone.psyir.transformations.region_trans.RegionTrans.get_node_list::single_schedule_children_copy',
        symbols=['psyclone.psyir.transformations.region_trans.RegionTrans.get_node_list'],
        contract=('get_node_list unwraps a single-Schedule list to a slice copy of its '
                  'children: the returned list is not (is not) the live children list, '
                  'its elements are the children themselves (is); reordering the returned '
                  'list leaves the tree untouched; captured at the get_node_list() call timepoint.'),
        trigger='Pass [Schedule] to RegionTrans.get_node_list, then mutate the returned list.',
        wrong=('A list containing just a Schedule falls into the generic list branch and '
               'returns [sched] itself instead of the Schedule children.'),
        causal="Return nodes[0].children[:] for a single-Schedule list (same as Schedule input).",
        obs=M1_OBS,
        triple=('lst is not sched.children (new list); lst[0] is sched.children[0] (shallow elements)',
                'appending to the returned list leaves sched.children unchanged',
                'get_node_list() call timepoint capture'),
        direction='repair_target', semantics='repair_target'),
    'qiskit__qiskit-terra-5778': dict(
        family='Qiskit/qiskit-terra::qiskit.transpiler.passes.scheduling.ASAPSchedule+ALAPSchedule::per_instruction_op_copy_duration_isolation',
        symbols=['qiskit.transpiler.passes.scheduling.asap.ASAPSchedule',
                 'qiskit.transpiler.passes.scheduling.alap.ALAPSchedule'],
        contract=('Each scheduled instruction owns a per-node op copy: the scheduled op is not '
                  '(is not) the original shared op; writing a duration on one node leaves the '
                  'original and sibling nodes untouched; captured at the scheduling-pass write timepoint.'),
        trigger='Schedule a circuit reusing one op instance across nodes (cz chain), then compare durations.',
        wrong=('Scheduled nodes share one op instance; the second duration write overwrites the first.'),
        causal='new_op = node.op.copy() per node before storing duration (asap.py + alap.py).',
        obs=M2_OBS,
        triple=('cxs[0] is not cxs[1] (per-node op copies)',
                'per-gate durations differ and match the backend table; original circuit ops untouched',
                'ASAPSchedule/ALAPSchedule run() write timepoint'),
        direction='repair_target', semantics='repair_target'),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    keeps = json.loads((OUT / 'keep-list.json').read_text())
    assert len(keeps) == 2
    tasks = []
    for k in keeps:
        iid = k['instance_id']
        m = MECH[iid]
        lic, licname = LICFILE[k['repo']]
        row = json.loads((OUT / f"pages-v2/{iid}.json").read_text())['row']
        assert row['instance_id'] == iid and row['base_commit'] == k['base_commit']
        assert sorted(row['FAIL_TO_PASS']) == sorted(k['F2P_list'])
        env = {'buggy_source': {'repo': k['repo'], 'requested_revision': k['base_commit'][:12],
                                'resolved_revision': k['base_commit']},
               'upstream_source': {'repo': k['repo'], 'requested_revision': k['base_commit'][:12],
                                   'resolved_revision': k['base_commit']}}
        tasks.append({
            'instance_id': iid, 'canonical_repo': k['repo'], 'primary_adapter': 'F', 'submechanism': 'F2',
            'repair_family_id': m['family'], 'snapshot_group_id': f"{k['repo']}::{k['base_commit']}",
            'execution_commit': k['base_commit'], 'source_dataset': DS['v2'][0], 'source_revision': DS['v2'][1],
            'physical_split': 'train', 'source_task_path': f'competition/vfarl/acquisition/f2-new/f2-more/pages-v2/{iid}.json',
            'source_task_sha256': sha(OUT / f'pages-v2/{iid}.json'),
            'environment': env,
            'license_evidence': {
                'source': {'license': licname, 'path': f'competition/vfarl/acquisition/f2-new/f2-more/{lic}',
                           'sha256': sha(OUT / lic), 'revision': k['base_commit']},
                'dataset': {'license': DS['v2'][2], 'revision': DS['v2'][1]}},
            'mechanism_evidence': {
                'submechanism': 'F2', 'family': m['family'], 'contract': m['contract'], 'trigger': m['trigger'],
                'base_wrong_behavior': m['wrong'], 'production_symbols': m['symbols'], 'causal_change': m['causal'],
                'discriminating_observation': {'kind': 'explicitly_reviewed_source_bound_inline_execution',
                                               'description': m['obs']},
                'triple': {'is_identity': m['triple'][0], 'shared_mutation_visibility': m['triple'][1],
                           'copy_timepoint': m['triple'][2]},
                'bug_patch_direction': m['direction']},
            'original_patch_semantics': m['semantics'],
            'original_FAIL_TO_PASS_count': k['F2P_count'], 'original_PASS_TO_PASS_count': k['P2P_count'],
            'source_mechanism_review_complete': True,
            'root_second_read': {
                'date': '2026-10-25', 'reviewer': 'subagent f2-more',
                'finding': ('Read original repair diff, publisher problem_statement verbatim, and F2P/P2P lists. '
                            'F2 triple (is-not + bidirectional isolation + copy timepoint) holds; only-== would '
                            'demote to S, capability-only to R. No test/matcher edits; six-tool course; protected paths untouched.'),
                'classification_scope': 'F2 reference/ownership isolation, not value equality or capability enumeration.'},
            'use_partition': 'train_only', 'course_membership': 'F600_F2', 'whitelisted_prompt': True,
            'junit_dual_required': True, 'tests_unmodified': True, 'six_tools_only': True,
            'protected_paths_untouched': True})
    reviewed = {'schema_version': 'reviewed-f2-more-training-candidates-v1', 'date': '2026-10-25',
                'scope': 'F600-F2 f2-more (Gym + Rebench-V2 small-suite inline-is, total<=150)',
                'source_dataset': DS['v2'][0], 'source_revision': DS['v2'][1], 'physical_split': 'train',
                'tasks': tasks,
                'pools': {'v2': {'dataset': DS['v2'][0], 'revision': DS['v2'][1]}}}
    (OUT / 'reviewed-candidates.json').write_text(json.dumps(reviewed, ensure_ascii=False, indent=2) + '\n')
    cat_tasks = []
    for t in tasks:
        c = dict(t)
        c['source_review_path'] = 'competition/vfarl/acquisition/f2-new/f2-more/reviewed-candidates.json'
        c['source_review_sha256'] = sha(OUT / 'reviewed-candidates.json')
        cat_tasks.append(c)
    catalog = {'schema_version': 'reviewed-f2-more-training-catalog-v1', 'date': '2026-10-25',
               'selection_policy': ('Explicit F600-F2 train-only; Gym + Rebench-V2 small-suite inline-is total<=150; '
                                    'disjoint from common-400/F1-40/F5-old40/f2-round1-2/f2-gymv2-16/f2-gymv2b-5/f2-gymv2c-2/'
                                    'f2-recall-3/f2-inline-1 families/snapshots/instances; skip-collision prechecked; '
                                    'normalize eligible only.'),
               'tasks': cat_tasks}
    (OUT / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')
    print(f'reviewed {len(tasks)} catalog {len(cat_tasks)}')


if __name__ == '__main__':
    main()
