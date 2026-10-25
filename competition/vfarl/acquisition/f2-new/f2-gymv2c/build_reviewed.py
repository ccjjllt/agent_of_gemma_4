"""F2 round 5 (f2-gymv2c): verify zero-overlap and build reviewed-candidates + catalog."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT / 'competition/vfarl'
ACQ = VFARL / 'acquisition/f2-new'
OUT = ACQ / 'f2-gymv2c'

V2_REV = '10483de0f50fe5da545942705a76c6150171af7f'
LIC_SHA = '3b49dcee4105eb37bac10faf1be260408fe85d252b8e9df2e0979fc1e094437b'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_forbidden():
    forb_i, forb_s, forb_f = set(), set(), set()
    base = json.loads((ACQ / 'dedup-baseline.json').read_text())
    forb_i |= set(base['forbidden_instance_ids'])
    forb_s |= set(base.get('forbidden_snapshot_group_ids', []))
    forb_f |= set(base['forbidden_repair_family_ids'])
    f1 = json.loads((VFARL / 'acquisition/f1-new/reviewed-candidates.json').read_text())
    for t in f1.get('tasks', []):
        forb_i.add(t['instance_id'])
        if t.get('snapshot_group_id'):
            forb_s.add(t['snapshot_group_id'])
        if t.get('repair_family_id'):
            forb_f.add(t['repair_family_id'])
    f5 = json.loads((VFARL / 'acquisition/f5-new/acquired-40.json').read_text())
    for t in f5.get('tasks', []):
        forb_i.add(t.get('instance_id', t.get('canonical_task_id', '')))
        for key in ('snapshot_group_id', 'snapshot_id'):
            if t.get(key):
                forb_s.add(t[key])
        for key in ('repair_family_id', 'family_id'):
            if t.get(key):
                forb_f.add(t[key])
    for rel in ['f2-replace/catalog.json', 'f2-replace/reviewed-candidates.json',
                'candidate-list-40.json', 'reviewed-candidates.json', 'catalog.json']:
        p = ACQ / rel
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        for t in d.get('tasks', d.get('candidates', [])):
            if t.get('instance_id'):
                forb_i.add(t['instance_id'])
            for key in ('snapshot_group_id', 'snapshot_group_id_provisional'):
                if t.get(key):
                    forb_s.add(t[key])
            for key in ('repair_family_id', 'repair_family_id_provisional'):
                if t.get(key):
                    forb_f.add(t[key])
    forb_s |= {'seperman/deepdiff::ed252022', 'pallets/click::fde47b4b',
               'pydantic/pydantic::acb0f10f', 'stanfordnlp/dspy::651a4c71',
               'sunpy/sunpy::f8edfd5c', 'mido/mido::a0158ff9',
               'cantools/cantools::0c6a7871'}
    for kdir in ['f2-gymv2', 'f2-gymv2b']:
        for k in json.loads((ACQ / kdir / 'keep-list.json').read_text()):
            forb_i.add(k['instance_id'])
            forb_s.add(f"{k['repo']}::{k['base_commit']}")
        forb_f |= set(json.loads((ACQ / kdir / 'keep-families.json').read_text()).values())
    return forb_i, forb_s, forb_f


MECH = {
    'pallets__jinja-2029': {
        'family': 'pallets/jinja::jinja2.utils.missing/_MissingType::singleton_copy_pickle_identity',
        'contract': ('The internal missing singleton survives copy/deepcopy/pickle as itself: '
                     'copy.copy(missing) is missing, copy.deepcopy(missing) is missing, and '
                     'pickle.loads(pickle.dumps(missing)) is missing; captured at the '
                     'copy.copy/copy.deepcopy/pickle call timepoint.'),
        'trigger': 'Copy, deepcopy and pickle jinja2.utils.missing at every pickle protocol.',
        'base_wrong_behavior': ('missing is an instance of a dynamically created type with no '
                                '__reduce__; copy/deepcopy produce a distinct object and pickle '
                                'fails or rebuilds a non-identical object, breaking is/is-not '
                                'comparisons against the singleton.'),
        'production_symbols': ['jinja2.utils._MissingType.__reduce__', 'jinja2.utils.missing'],
        'causal_change': 'Define _MissingType with __repr__ and __reduce__ returning "missing".',
        'observation': ('import copy, pickle\nfrom jinja2.utils import missing\n'
                        'c = copy.copy(missing)\nprint("copy_identity", c is missing)\n'
                        'assert c is missing\nd = copy.deepcopy(missing)\nassert d is missing\n'
                        'u = pickle.loads(pickle.dumps(missing))\nprint("pickle_identity", u is missing)\n'
                        'assert u is missing\nprint("f2_missing_singleton", True)\n'),
        'triple_is': ('copy.copy(missing) is missing; deepcopy is missing; '
                      'pickle roundtrip is missing (is not a distinct rebuild)'),
        'triple_shared': ('singleton semantics globally visible: every holder of the missing '
                          'reference observes the identical object across copy boundaries'),
        'triple_time': 'copy.copy/copy.deepcopy/pickle.dumps+loads call timepoint capture',
    },
    'pallets__jinja-1613': {
        'family': 'pallets/jinja::jinja2.async_utils.async_variant::pickle_by_reference_identity',
        'contract': ('Filters/tests decorated with @async_variant pickle by reference to the '
                     'identical function object: pickle.loads(pickle.dumps(do_first)) is '
                     'jinja2.filters.do_first; captured at the pickle call timepoint.'),
        'trigger': 'Pickle an async_variant-decorated filter such as jinja2.filters.do_first.',
        'base_wrong_behavior': ('The decorator puts the wrong __name__/__qualname__ on the wrapper, '
                                'so pickle cannot resolve it ("not the same object as ...") and '
                                'raises PicklingError.'),
        'production_symbols': ['jinja2.async_utils.async_variant'],
        'causal_change': ('@wraps with split assigned attrs: name/module/qualname from the async '
                          'function, doc/annotations from the sync function.'),
        'observation': ('import pickle\nimport jinja2.filters\n'
                        'f = jinja2.filters.do_first\n'
                        'u = pickle.loads(pickle.dumps(f))\nprint("pickle_identity", u is f)\n'
                        'assert u is f\nassert u is jinja2.filters.do_first\n'
                        'f._f2_probe = 1\nassert u._f2_probe == 1\n'
                        'del f._f2_probe\nprint("f2_pickle_by_reference", True)\n'),
        'triple_is': ('unpickled filter is the module-level function object '
                      '(pickle-by-reference identity)'),
        'triple_shared': ('attribute set on the original is visible via the unpickled reference '
                          '(same object, shared by design)'),
        'triple_time': 'pickle.dumps/pickle.loads call timepoint capture',
    },
}

FINDING = ('Read real-PR fix diff, publisher problem_statement verbatim, and F2P/P2P node lists. '
           'F2 triple (is + shared visibility + copy timepoint) holds; only-== would demote to S, '
           'capability-only to R. No test/matcher edits; six-tool course; protected paths untouched.')


def main():
    forb_i, forb_s, forb_f = load_forbidden()
    keeps = json.loads((OUT / 'keep-list.json').read_text())
    fams = json.loads((OUT / 'keep-families.json').read_text())
    assert set(fams) == {k['instance_id'] for k in keeps}
    tasks = []
    for k in keeps:
        iid = k['instance_id']
        assert iid not in forb_i, f'instance overlap {iid}'
        snap = f"{k['repo']}::{k['base_commit']}"
        for s in forb_s:
            if '::' not in s:
                continue
            repo, rev = s.split('::', 1)
            if repo.lower() == k['repo'].lower():
                c, r = k['base_commit'].lower(), rev.lower()
                assert not (c.startswith(r) or r.startswith(c)), f'snapshot overlap {iid} vs {s}'
        assert fams[iid] not in forb_f, f"family overlap {iid}"
        forb_i.add(iid)
        forb_s.add(snap)
        forb_f.add(fams[iid])
        page = OUT / f"pages-v2/{iid}.json"
        row = json.loads(page.read_text())['row']
        assert row['instance_id'] == iid and row['base_commit'] == k['base_commit']
        assert row['FAIL_TO_PASS'] == k['F2P_list'] and row['PASS_TO_PASS'] == k['P2P_list']
        m = MECH[iid]
        assert m['family'] == fams[iid]
        lic_rel = 'competition/vfarl/acquisition/f2-new/f2-gymv2c/jinja.LICENSE'
        assert sha(ROOT / lic_rel) == LIC_SHA
        tasks.append({
            'instance_id': iid, 'canonical_repo': k['repo'],
            'primary_adapter': 'F', 'submechanism': 'F2',
            'repair_family_id': fams[iid], 'snapshot_group_id': snap,
            'execution_commit': k['base_commit'],
            'source_dataset': 'nebius/SWE-rebench-V2', 'source_revision': V2_REV,
            'physical_split': 'train',
            'source_task_path': f'competition/vfarl/acquisition/f2-new/f2-gymv2c/pages-v2/{iid}.json',
            'source_task_sha256': sha(page),
            'environment': {
                'buggy_source': {'repo': k['repo'], 'requested_revision': k['base_commit'][:12],
                                 'resolved_revision': k['base_commit']},
                'upstream_source': {'repo': k['repo'], 'requested_revision': k['base_commit'][:12],
                                    'resolved_revision': k['base_commit']}},
            'license_evidence': {
                'source': {'license': 'BSD-3-Clause', 'path': lic_rel,
                           'sha256': LIC_SHA, 'revision': k['base_commit']},
                'dataset': {'license': 'CC-BY-4.0', 'revision': V2_REV}},
            'mechanism_evidence': {
                'submechanism': 'F2', 'family': fams[iid], 'contract': m['contract'],
                'trigger': m['trigger'], 'base_wrong_behavior': m['base_wrong_behavior'],
                'production_symbols': m['production_symbols'], 'causal_change': m['causal_change'],
                'discriminating_observation': {
                    'kind': 'explicitly_reviewed_source_bound_inline_execution',
                    'description': m['observation']},
                'triple': {'is_identity': m['triple_is'],
                           'shared_mutation_visibility': m['triple_shared'],
                           'copy_timepoint': m['triple_time']},
                'bug_patch_direction': 'repair_target'},
            'original_patch_semantics': 'repair_target',
            'original_FAIL_TO_PASS_count': len(k['F2P_list']),
            'original_PASS_TO_PASS_count': len(k['P2P_list']),
            'source_mechanism_review_complete': True,
            'root_second_read': {'date': '2026-10-25', 'reviewer': 'root research agent',
                                 'finding': FINDING,
                                 'classification_scope': ('F2 reference/ownership isolation, not '
                                                          'value equality or capability enumeration.')},
            'use_partition': 'train_only', 'course_membership': 'F600_F2',
            'whitelisted_prompt': True, 'junit_dual_required': True,
            'tests_unmodified': True, 'six_tools_only': True,
            'protected_paths_untouched': True})
    reviewed = {
        'schema_version': 'f2-gymv2c-reviewed-candidates-v1',
        'date': '2026-10-25',
        'scope': ('F600 F2 reference/ownership isolation round 5 (Gym + Rebench-V2 only): 2 KEEP tasks, '
                  'each with is-identity + shared-mutation visibility + copy timepoint. Only-== demotes '
                  'to S; capability-only to R. Train only; four competition repos excluded; disjoint from '
                  'common-400/F1-40/F5-old40/f2-round1-2/f2-gymv2-16/f2-gymv2b-5.'),
        'pools': {'nebius/SWE-rebench-V2': V2_REV},
        'tasks': tasks}
    (OUT / 'reviewed-candidates.json').write_text(
        json.dumps(reviewed, ensure_ascii=False, indent=1) + '\n')
    rsha = sha(OUT / 'reviewed-candidates.json')
    catalog = {'schema_version': 'reviewed-f2-gymv2c-training-catalog-v1',
               'date': '2026-10-25',
               'selection_policy': ('Explicit F600-F2 train-only; Gym + Rebench-V2; disjoint from '
                                    'common-400/F1-40/F5-old40/f2-round1-2/f2-gymv2-16/f2-gymv2b-5 '
                                    'families/snapshots/instances; skip-collision prechecked; '
                                    'normalize eligible only.'),
               'tasks': []}
    for t in tasks:
        c = {k: t[k] for k in ('instance_id', 'canonical_repo', 'primary_adapter', 'submechanism',
                               'repair_family_id', 'snapshot_group_id', 'execution_commit',
                               'source_dataset', 'source_revision', 'physical_split',
                               'original_FAIL_TO_PASS_count', 'original_PASS_TO_PASS_count',
                               'mechanism_evidence', 'original_patch_semantics', 'use_partition',
                               'course_membership', 'source_task_path', 'source_task_sha256',
                               'environment', 'license_evidence', 'source_mechanism_review_complete',
                               'root_second_read')}
        c['source_review_path'] = 'competition/vfarl/acquisition/f2-new/f2-gymv2c/reviewed-candidates.json'
        c['source_review_sha256'] = rsha
        catalog['tasks'].append(c)
    (OUT / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n')
    print(f'keeps={len(tasks)} zero-overlap verified; reviewed + catalog written; review_sha={rsha[:12]}')


if __name__ == '__main__':
    main()
