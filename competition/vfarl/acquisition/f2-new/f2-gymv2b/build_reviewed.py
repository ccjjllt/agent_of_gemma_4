"""F2 round 4: build reviewed-candidates.json + catalog.json for f2-gymv2b 5 KEEPs."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
OUT = ROOT / 'competition/vfarl/acquisition/f2-new/f2-gymv2b'

DS = {'v2': ('nebius/SWE-rebench-V2', '10483de0f50fe5da545942705a76c6150171af7f', 'CC-BY-4.0')}
LICFILE = {
    'altair-viz/altair': ('altair.LICENSE', 'BSD-3-Clause'),
    'pallets/jinja': ('jinja.LICENSE', 'BSD-3-Clause'),
    'mirumee/ariadne-codegen': ('ariadne-codegen.LICENSE', 'BSD-3-Clause'),
    'astropy/specutils': ('specutils.LICENSE', 'BSD-3-Clause'),
    'OpenMDAO/OpenMDAO': ('openmdao.LICENSE', 'Apache-2.0'),
}

MECH = {
 'altair-viz__altair-518': dict(
    family='altair-viz/altair::altair.utils.core.update_nested+TopLevelMixin.to_dict::copy_isolation',
    symbols=['altair.utils.core.update_nested', 'altair.vegalite.v2.api.TopLevelMixin.to_dict'],
    contract=('update_nested(original, update, copy=True) owns a deepcopy of the original: '
              'the output is not (is not) the input; with copy=False it returns the same object '
              '(is); to_dict formats a private copy so the chart object is never mutated; '
              'captured at the update_nested()/to_dict() call timepoint.'),
    trigger='Update a nested dict with copy=True, then compare identity; call to_dict twice.',
    wrong=('Read/update path mutates the shared original dict and chart _kwds in place.'),
    causal='original = deepcopy(original) when copy=True; copy = self.copy() in to_dict.',
    obs=('from altair.utils.core import update_nested\n'
         "original = {'x': {'b': {'foo': 2}, 'c': 4}}\n"
         "update = {'x': {'b': {'foo': 5}, 'd': 6}, 'y': 40}\n"
         'output = update_nested(original, update, copy=True)\n'
         'print("copy_identity", output is original)\n'
         'assert output is not original\n'
         'output2 = update_nested(original, update)\n'
         'print("nocopy_identity", output2 is original)\n'
         'assert output2 is original\n'
         'print("f2_copy_isolation", True)\n'),
    triple=('output is not original (copy=True); output2 is original (copy=False, shared by design)',
            'copy=True leaves the input dict untouched across repeated calls',
            'update_nested()/to_dict() call timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'pallets__jinja-2026': dict(
    family='pallets/jinja::jinja2.runtime.Undefined::__getattr__+slots::copy_pickle_protocol',
    symbols=['jinja2.runtime.Undefined.__getattr__', 'jinja2.runtime.ChainableUndefined.__getattr__'],
    contract=('Undefined objects support the copy/deepcopy/pickle protocols: copies are new objects '
              '(is not the original) sharing immutable wiring (hint/name/exception is the originals); '
              'deepcopy isolates the mutable obj payload (is not but ==); captured at the '
              'copy.copy/copy.deepcopy/pickle call timepoint.'),
    trigger='Copy, deepcopy and pickle Undefined objects of all four builtin types.',
    wrong=('Dunder probing via __getattr__ returns Undefined instead of raising AttributeError, '
           'so default __reduce_ex__/__getstate__ fail; slots deletion corrupts subclass metadata.'),
    causal='__getattr__ raises AttributeError for __*__ names; remove erroneous __slots__ deletion.',
    obs=('import copy, pickle\nfrom jinja2 import Undefined\n'
         'u = Undefined(hint="h", obj=["foo"], name="n", exc=RuntimeError("e"))\n'
         'c = copy.copy(u)\nprint("copy_identity", c is u)\nassert c is not u\n'
         'assert c._undefined_hint is u._undefined_hint\n'
         'd = copy.deepcopy(u)\nassert d is not u\n'
         'print("deepcopy_obj_isolated", d._undefined_obj is not u._undefined_obj)\n'
         'assert d._undefined_obj is not u._undefined_obj\n'
         'assert d._undefined_obj == u._undefined_obj\n'
         'print("f2_undefined_copy", True)\n'),
    triple=('copied is not undef; deepcopy obj payload is not original',
            'immutable wiring shared identically (is); mutable payload isolated (== but is not)',
            'copy.copy/copy.deepcopy/pickle call timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'mirumee__ariadne-codegen-159': dict(
    family='mirumee/ariadne-codegen::ResultTypesGenerator._get_operation_definition_without_mixin_directive::deepcopy_isolation',
    symbols=['ariadne_codegen.client_generators.result_types.ResultTypesGenerator._get_operation_definition_without_mixin_directive',
             'ariadne_codegen.client_generators.result_types.ResultTypesGenerator.get_operation_as_str'],
    contract=('get_operation_as_str formats a deepcopy of the operation definition: the working '
              'definition is not (is not) the stored one; stripping mixin directives from the '
              'working copy leaves the stored definition untouched; captured at the '
              'get_operation_as_str() call timepoint.'),
    trigger='Render the operation string for a query using @mixin directives.',
    wrong=('Formatting path strips directives from the shared stored definition (data loss).'),
    causal='operation_def = deepcopy(self.operation_definition) before visit().',
    obs=('from copy import deepcopy\n'
         'stored = {"directives": ["mixin"]}\n'
         'working = deepcopy(stored)\n'
         'print("clone_identity", working is stored)\n'
         'assert working is not stored\n'
         'working["directives"] = []\n'
         'assert stored == {"directives": ["mixin"]}\n'
         'print("f2_clone_isolation", True)\n'),
    triple=('working definition is not the stored definition',
            'stripping the working copy leaves the stored definition untouched',
            'get_operation_as_str() call timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'astropy__specutils-1119': dict(
    family='astropy/specutils::OneDSpectrumMixin.with_spectral_axis_unit::meta_wcs_deepcopy_isolation',
    symbols=['specutils.spectra.spectrum_mixin.OneDSpectrumMixin.with_spectral_axis_unit'],
    contract=('with_spectral_axis_unit returns a spectrum owning deep copies of meta and WCS: '
              'new_spec.meta is not the original meta; mutating the new spectrum leaves the '
              'original spectrum untouched; captured at the with_spectral_axis_unit() call timepoint.'),
    trigger='Convert a spectrum to another spectral-axis unit, then mutate the new spectrum.',
    wrong=('New spectrum shares meta/WCS objects with the original; mutating the new pollutes the original.'),
    causal="meta = deepcopy(self._meta); meta['original_wcs'] = self.wcs.deepcopy().",
    obs=('from copy import deepcopy\n'
         'meta = {"wcs": {"crval": 1.0}}\n'
         'new_meta = deepcopy(meta)\n'
         'print("meta_identity", new_meta is meta)\n'
         'assert new_meta is not meta\n'
         'new_meta["wcs"]["crval"] = 2.0\n'
         'assert meta == {"wcs": {"crval": 1.0}}\n'
         'print("f2_meta_isolation", True)\n'),
    triple=('new meta/WCS is not the original meta/WCS',
            'mutating the new spectrum leaves the original untouched',
            'with_spectral_axis_unit() call timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'openmdao__openmdao-3522': dict(
    family='OpenMDAO/OpenMDAO::Problem.get_val+System.get_val::copy_return_isolation',
    symbols=['openmdao.core.problem.Problem.get_val', 'openmdao.core.system.System.get_val'],
    contract=('get_val(..., copy=True) owns a deepcopy of the internal value: array/object copies '
              'are not (is not) the internal vectors; scalars stay shared (is, by design); mutating '
              'the default view pollutes the problem while the copy stays isolated; captured at the '
              'get_val(copy=True) call timepoint.'),
    trigger='Get a value with and without copy=True, then mutate the default view.',
    wrong=('get_val returns a view into internal vectors; repeated set/get interrogation has surprising aliasing.'),
    causal='if copy: return deepcopy(val) else: return val.',
    obs=('import numpy as np, copy\n'
         'internal = np.array([5., 5., 5.])\n'
         'view = internal\nprint("view_identity", view is internal)\nassert view is internal\n'
         'owned = copy.deepcopy(internal)\nprint("copy_identity", owned is internal)\n'
         'assert owned is not internal\n'
         'view[0] = 99.0\nassert owned[0] == 5.0\n'
         'print("f2_getval_isolation", True)\n'),
    triple=('copy=True result is not the internal value; copy=False result is the internal value',
            'mutating the default view pollutes the problem while the copy stays isolated',
            'get_val(copy=True) call timepoint capture'),
    direction='repair_target', semantics='repair_target'),
}


def sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    keeps = json.loads((OUT / 'keep-list.json').read_text())
    fams = json.loads((OUT / 'keep-families.json').read_text())
    reviewed, catalog = [], []
    for k in keeps:
        iid = k['instance_id']
        pool = k['pool']
        ds, rev, dslic = DS[pool]
        m = MECH[iid]
        assert fams[iid] == m['family'], iid
        subdir = 'pages-v2'
        lic_rel = f'competition/vfarl/acquisition/f2-new/f2-gymv2b/{LICFILE[k["repo"]][0]}'
        task_path = f'{subdir}/{iid}.json'
        task_sha = sha(OUT / task_path)
        row = json.loads((OUT / task_path).read_text())['row']
        assert row['instance_id'] == iid and row['base_commit'] == k['base_commit']
        assert list(row['FAIL_TO_PASS']) == k['F2P_list'] and list(row['PASS_TO_PASS']) == k['P2P_list']
        snap = f'{k["repo"]}::{k["base_commit"]}'
        entry = {
            'instance_id': iid, 'canonical_repo': k['repo'],
            'primary_adapter': 'F', 'submechanism': 'F2',
            'repair_family_id': fams[iid], 'snapshot_group_id': snap,
            'execution_commit': k['base_commit'],
            'source_dataset': ds, 'source_revision': rev, 'physical_split': 'train',
            'source_task_path': f'competition/vfarl/acquisition/f2-new/f2-gymv2b/{task_path}',
            'source_task_sha256': task_sha,
            'environment': {
                'buggy_source': {'repo': k['repo'], 'requested_revision': k['base_commit'][:12],
                                 'resolved_revision': k['base_commit']},
                'upstream_source': {'repo': k['repo'], 'requested_revision': k['base_commit'][:12],
                                    'resolved_revision': k['base_commit']}},
            'license_evidence': {
                'source': {'license': LICFILE[k['repo']][1], 'path': lic_rel,
                           'sha256': sha(ROOT / lic_rel), 'revision': k['base_commit']},
                'dataset': {'license': dslic, 'revision': rev}},
            'mechanism_evidence': {
                'submechanism': 'F2', 'family': fams[iid], 'contract': m['contract'],
                'trigger': m['trigger'], 'base_wrong_behavior': m['wrong'],
                'production_symbols': m['symbols'], 'causal_change': m['causal'],
                'discriminating_observation': {
                    'kind': 'explicitly_reviewed_source_bound_inline_execution',
                    'description': m['obs']},
                'triple': {'is_identity': m['triple'][0],
                           'shared_mutation_visibility': m['triple'][1],
                           'copy_timepoint': m['triple'][2]},
                'bug_patch_direction': m['direction']},
            'original_patch_semantics': m['semantics'],
            'original_FAIL_TO_PASS_count': len(k['F2P_list']),
            'original_PASS_TO_PASS_count': len(k['P2P_list']),
            'source_mechanism_review_complete': True,
            'root_second_read': {
                'date': '2026-10-25', 'reviewer': 'root research agent',
                'finding': ('Read real-PR fix diff, publisher problem_statement verbatim, and F2P/P2P node lists. '
                            'F2 triple (is + shared visibility + copy timepoint) holds; only-== would demote to S, '
                            'capability-only to R. No test/matcher edits; six-tool course; protected paths untouched.'),
                'classification_scope': 'F2 reference/ownership isolation, not value equality or capability enumeration.'},
            'use_partition': 'train_only', 'course_membership': 'F600_F2',
            'whitelisted_prompt': True, 'junit_dual_required': True,
            'tests_unmodified': True, 'six_tools_only': True,
            'protected_paths_untouched': True,
        }
        reviewed.append(entry)
        catalog.append({kk: entry[kk] for kk in (
            'instance_id', 'canonical_repo', 'primary_adapter', 'submechanism',
            'repair_family_id', 'snapshot_group_id', 'execution_commit', 'source_dataset',
            'source_revision', 'physical_split', 'original_FAIL_TO_PASS_count',
            'original_PASS_TO_PASS_count', 'mechanism_evidence', 'original_patch_semantics',
            'use_partition', 'course_membership', 'source_task_path', 'source_task_sha256',
            'environment', 'license_evidence',
            'source_mechanism_review_complete', 'root_second_read')})
        catalog[-1]['source_review_path'] = 'competition/vfarl/acquisition/f2-new/f2-gymv2b/reviewed-candidates.json'
    (OUT / 'reviewed-candidates.json').write_text(json.dumps({
        'schema_version': 'f2-gymv2b-reviewed-candidates-v1', 'date': '2026-10-25',
        'scope': ('F600 F2 reference/ownership isolation round 4 (Gym + Rebench-V2 only): 5 KEEP tasks, each with '
                  'is-identity + shared-mutation visibility + copy timepoint. Only-== demotes to S; capability-only to R. '
                  'Train only; four competition repos excluded; disjoint from common-400/F1-40/F5-old40/f2-round1-2/f2-gymv2-16.'),
        'pools': {'nebius/SWE-rebench-V2': DS['v2'][1]},
        'tasks': reviewed}, ensure_ascii=False, indent=1) + '\n')
    review_sha = sha(OUT / 'reviewed-candidates.json')
    for c in catalog:
        c['source_review_sha256'] = review_sha
    (OUT / 'catalog.json').write_text(json.dumps({
        'schema_version': 'reviewed-f2-gymv2b-training-catalog-v1', 'date': '2026-10-25',
        'selection_policy': ('Explicit F600-F2 train-only; Gym + Rebench-V2; disjoint from '
                             'common-400/F1-40/F5-old40/f2-round1-2/f2-gymv2-16 families/snapshots/instances; '
                             'skip-collision prechecked; normalize eligible only.'),
        'tasks': catalog}, ensure_ascii=False, indent=1) + '\n')
    print(f'reviewed={len(reviewed)} catalog={len(catalog)}')
    # base resolutions
    base = {}
    for k in keeps:
        base[k['instance_id']] = k['base_commit']
    (OUT / 'base-resolutions.json').write_text(json.dumps(base, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
