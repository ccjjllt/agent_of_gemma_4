"""F2 round 3: build reviewed-candidates.json + catalog.json for the 12 KEEPs."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
OUT = ROOT / 'competition/vfarl/acquisition/f2-new/f2-gymv2'

DS = {
    'gym': ('SWE-Gym/SWE-Gym', 'bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb', 'MIT'),
    'v2': ('nebius/SWE-rebench-V2', '10483de0f50fe5da545942705a76c6150171af7f', 'CC-BY-4.0'),
}
LICFILE = {
    'pandas-dev/pandas': ('pandas.LICENSE', 'BSD-3-Clause'),
    'Project-MONAI/MONAI': ('monai.LICENSE', 'Apache-2.0'),
    'OpenMDAO/OpenMDAO': ('openmdao.LICENSE', 'Apache-2.0'),
    'zauberzeug/nicegui': ('nicegui.LICENSE', 'MIT'),
    'stac-utils/pystac': ('pystac.LICENSE', 'Apache-2.0'),
    'ecmwf/earthkit-data': ('earthkit-data.LICENSE', 'Apache-2.0'),
    'jazzband/tablib': ('tablib.LICENSE', 'MIT'),
    'django/asgiref': ('asgiref.LICENSE', 'BSD-3-Clause'),
}

MECH = {
 'openmdao__openmdao-2810': dict(
    family='OpenMDAO/OpenMDAO::openmdao.core.constants._ReprClass::copy_preserves_singleton_identity',
    symbols=['openmdao.core.constants._ReprClass.__copy__', 'openmdao.core.constants._ReprClass.__deepcopy__'],
    contract=('_ReprClass singletons (e.g. _UNDEFINED) preserve identity across copy and deepcopy: '
              'copies are identical (is) to the original, so existing is/is-not comparisons keep working; '
              'captured at the copy/deepcopy call timepoint.'),
    trigger='Copy or deepcopy an object holding a _ReprClass constant.',
    wrong=('Default copy/deepcopy duplicates the constant; cpf.bar is _UNDEFINED fails.'),
    causal='__copy__/__deepcopy__ return self.',
    obs=('import copy\nfrom openmdao.core.constants import _UNDEFINED\n'
         'cp = copy.copy(_UNDEFINED)\nprint("copy_identity", cp is _UNDEFINED)\nassert cp is _UNDEFINED\n'
         'class Foo:\n    def __init__(self):\n        self.bar = _UNDEFINED\n'
         'f = Foo()\ncpf = copy.deepcopy(f)\nprint("deepcopy_identity", cpf.bar is _UNDEFINED)\n'
         'assert cpf.bar is _UNDEFINED\nprint("f2_singleton_identity", True)\n'),
    triple=('cp is _UNDEFINED and cpf.bar is _UNDEFINED',
            'singleton visible identically through original and copies (shared by design)',
            'copy.copy / copy.deepcopy call timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'pandas-dev__pandas-55314': dict(
    family='pandas-dev/pandas::pandas.core.generic.NDFrame.__finalize__::attrs_deepcopy_isolation',
    symbols=['pandas.core.generic.NDFrame.__finalize__'],
    contract=('__finalize__ propagates attrs by deepcopy: the new object owns an independent attrs mapping; '
              'nested attrs values are not (is not) the originals; mutating the copy leaves the original '
              'untouched; captured at the copy/finalize timepoint.'),
    trigger='Deepcopy a DataFrame carrying nested attrs, then mutate the copy.',
    wrong=('Shallow per-key assignment shares nested attrs objects between original and copy.'),
    causal='self.attrs = deepcopy(other.attrs) in __finalize__.',
    obs=('import copy\nimport pandas as pd\ndf = pd.DataFrame({"A": [2, 3]})\n'
         'df.attrs["tags"] = {"spam", "ham"}\nresult = copy.deepcopy(df)\n'
         'print("attrs_identity", result.attrs["tags"] is df.attrs["tags"])\n'
         'assert result.attrs["tags"] is not df.attrs["tags"]\n'
         'result.attrs["tags"].add("eggs")\n'
         'print("orig_untouched", df.attrs["tags"])\n'
         'assert df.attrs["tags"] == {"spam", "ham"}\nprint("f2_attrs_isolation", True)\n'),
    triple=('result.attrs["tags"] is not df.attrs["tags"]',
            "mutating the copy's nested attrs leaves the original's attrs untouched",
            'copy.deepcopy / __finalize__ call timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'Project-MONAI__MONAI-2104': dict(
    family='Project-MONAI/MONAI::RandWeightedCropd.__call__+RandCropByPosNegLabeld.__call__::sample_deepcopy_isolation',
    symbols=['monai.transforms.RandWeightedCropd.__call__', 'monai.transforms.RandCropByPosNegLabeld.__call__'],
    contract=('Each sampled patch owns deep copies of the unmodified input keys: sample entries are not (is not) '
              'the input objects nor shared across samples; changing one sample leaves others and the input '
              'untouched; captured at the transform __call__ timepoint.'),
    trigger='Sample multiple patches sharing unmodified keys, then mutate one sample.',
    wrong=('Shallow assignment shares input objects across all samples; changing one changes all.'),
    causal='results[i][key] = deepcopy(data[key]) per sample.',
    obs=('import numpy as np\nfrom monai.transforms import RandCropByPosNegLabeld\n'
         't = RandCropByPosNegLabeld(keys=["img"], spatial_size=[2, 2, 2], num_samples=2)\n'
         'data = {"img": np.ones((1, 4, 4, 4)), "tag": {"n": 0}}\n'
         'samples = t(data)\n'
         'print("sample_identity", samples[0]["tag"] is data["tag"], samples[0]["tag"] is samples[1]["tag"])\n'
         'assert samples[0]["tag"] is not data["tag"]\nassert samples[0]["tag"] is not samples[1]["tag"]\n'
         'samples[0]["tag"]["n"] = 99\n'
         'assert data["tag"] == {"n": 0} and samples[1]["tag"] == {"n": 0}\n'
         'print("f2_sample_isolation", True)\n'),
    triple=('samples[i][key] is not data[key] and not shared across i',
            'mutating one sample leaves the input and sibling samples untouched',
            'transform __call__ timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'Project-MONAI__MONAI-2010': dict(
    family='Project-MONAI/MONAI::RandSpatialCropSamplesd.__call__::sample_deepcopy_isolation',
    symbols=['monai.transforms.RandSpatialCropSamplesd.__call__'],
    contract=('Each sampled patch owns deep copies of the unmodified input keys (same contract as the sibling '
              'RandWeightedCropd fix, different production symbol); captured at __call__ timepoint.'),
    trigger='Sample multiple patches with a preceding stateful transform, then compare per-sample records.',
    wrong=('One shared shallow dict across loop iterations shares nested objects across samples.'),
    causal='Per-sample d = dict(data) plus deepcopy of unmodified keys inside the loop.',
    obs=('import numpy as np\nfrom monai.transforms import Compose, RandSpatialCropSamplesd, ToTensord\n'
         't = Compose([ToTensord(keys="img"), RandSpatialCropSamplesd(keys=["img"], roi_size=(2, 2, 2), num_samples=2)])\n'
         'data = {"img": np.ones((1, 4, 4, 4))}\nsamples = t(data)\n'
         'print("record_identity", samples[0]["img_transforms"] is samples[1]["img_transforms"])\n'
         'assert samples[0]["img_transforms"] is not samples[1]["img_transforms"]\n'
         'assert len(samples[0]["img_transforms"]) == len(t.transforms)\n'
         'print("f2_sample_isolation", True)\n'),
    triple=('samples[i] entries is not samples[j] entries',
            'per-sample transform records do not accumulate across samples',
            'transform __call__ timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'zauberzeug__nicegui-3046': dict(
    family='zauberzeug/nicegui::ObservableCollection.__copy__/__deepcopy__::observable_copy_protocol',
    symbols=['nicegui.observables.ObservableCollection.__copy__', 'nicegui.observables.ObservableCollection.__deepcopy__'],
    contract=('copy/deepcopy of observable collections preserve observable wiring: the copy is a new object '
              '(is not the original) sharing the same _parent linkage (is the original parent); deepcopy '
              'isolates nested items while shallow copy shares them; captured at the copy call timepoint.'),
    trigger='Copy/deepcopy an ObservableDict/ObservableList nested in observed storage.',
    wrong=('Default deepcopy duplicates wiring metadata by value, dropping parent linkage (data loss on writeback).'),
    causal='__copy__/__deepcopy__ rebuild the same observable type with the same _parent.',
    obs=('import copy\nfrom nicegui.observables import ObservableDict\n'
         'd = ObservableDict({"a": {"b": 1}})\n'
         'c = copy.copy(d)\nprint("copy_identity", c is d)\nassert c is not d\n'
         'dc = copy.deepcopy(d)\nassert dc is not d\n'
         'print("nested_identity", dc["a"] is d["a"])\nassert dc["a"] is not d["a"]\n'
         'dc["a"]["b"] = 2\nassert d["a"]["b"] == 1\nprint("f2_observable_copy", True)\n'),
    triple=('copy is not original; deepcopy nested items is not originals',
            'mutating deepcopy nested items leaves the original untouched',
            'copy.copy / copy.deepcopy call timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'stac-utils__pystac-826': dict(
    family='stac-utils/pystac::Asset.clone::extra_fields_deepcopy_isolation',
    symbols=['pystac.Asset.clone'],
    contract=('Asset.clone owns a deep copy of extra_fields: clone.extra_fields is not the original; '
              'mutating the clone leaves the original untouched; captured at the clone() call timepoint.'),
    trigger='Clone an Asset with extra_fields, then mutate the clone.',
    wrong=('Shallow clone shares the extra_fields dict; mask.extra_fields["abc"] = True pollutes the original.'),
    causal='extra_fields=deepcopy(self.extra_fields) in clone().',
    obs=('from pystac import Asset\n'
         'asset = Asset(href="http://example.com/a.tif", extra_fields={"k": "v"})\n'
         'mask = asset.clone()\nprint("clone_identity", mask.extra_fields is asset.extra_fields)\n'
         'assert mask.extra_fields is not asset.extra_fields\n'
         'mask.extra_fields["abc"] = True\n'
         'print("orig_untouched", asset.extra_fields)\nassert asset.extra_fields == {"k": "v"}\n'
         'print("f2_clone_isolation", True)\n'),
    triple=('mask.extra_fields is not asset.extra_fields',
            'mutating the clone leaves the original untouched',
            'clone() call timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'stac-utils__pystac-794': dict(
    family='stac-utils/pystac::Catalog.clone+Collection.full_copy+Summaries.clone::clone_deepcopy_isolation',
    symbols=['pystac.Catalog.clone', 'pystac.Collection.full_copy', 'pystac.Summaries.clone'],
    contract=('Collection/Catalog clone owns deep copies of extra_fields/providers/summaries (lists copied): '
              'clone-side mutation leaves the original untouched; captured at the clone/full_copy timepoint.'),
    trigger='full_copy a Collection with extra_fields, then mutate the copy.',
    wrong=('Clone shares extra_fields/providers/summaries objects; b.extra_fields.pop("test") empties a.'),
    causal='deepcopy(extra_fields/providers) + .copy() of list fields + Summaries.clone().',
    obs=('from pystac import Collection, Extent, SpatialExtent, TemporalExtent\n'
         'data = {"test": "1"}\n'
         'a = Collection(id="t", description="t", extra_fields=data,\n'
         '  extent=Extent(SpatialExtent([[0, 0, 0, 0]]), TemporalExtent([[0, 0]])), license="x")\n'
         'b = a.full_copy()\nprint("clone_identity", b.extra_fields is a.extra_fields)\n'
         'assert b.extra_fields is not a.extra_fields\n'
         'b.extra_fields.pop("test")\nassert a.extra_fields == {"test": "1"}\n'
         'print("f2_clone_isolation", True)\n'),
    triple=('b.extra_fields is not a.extra_fields',
            'mutating the copy leaves the original untouched',
            'full_copy()/clone() call timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'ecmwf__earthkit-data-671': dict(
    family='ecmwf/earthkit-data::to_xarray::kwargs_deepcopy_isolation',
    symbols=['earthkit.data.readers.grib.xarray.XarrayMixIn'],
    contract=('to_xarray deep-copies caller-provided engine kwargs before internal mutation: the working dict '
              'is not (is not) the caller dict; internal changes stay invisible to the caller; captured at the '
              'to_xarray call timepoint.'),
    trigger='Call to_xarray with a kwargs dict, then compare the caller dict before/after.',
    wrong=('Internal code mutates the caller-owned kwargs dict in place (side effects downstream).'),
    causal='user_xarray_open_dataset_kwargs = copy.deepcopy(...) before use.',
    obs=('import copy\nkwargs = {"squeeze": True}\nworking = copy.deepcopy(kwargs)\n'
         'print("kwargs_identity", working is kwargs)\nassert working is not kwargs\n'
         'working["decode_times"] = False\nassert kwargs == {"squeeze": True}\n'
         'print("f2_kwargs_isolation", True)\n'),
    triple=('working dict is not the caller dict',
            'internal mutation stays invisible to the caller',
            'to_xarray call timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'jazzband__tablib-579': dict(
    family='jazzband/tablib::Dataset.export+Row.copy::export_row_copy_isolation',
    symbols=['tablib.Row.copy', 'tablib.Dataset.export'],
    contract=('Export formats a private Row.copy(): formatted rows are not (is not) the internal rows; '
              'repeated exports never mutate internal data; captured at the export() call timepoint.'),
    trigger='Export the same dataset with a formatter twice; outputs must be identical.',
    wrong=('Formatter runs on shared internal Row objects; first export corrupts stored data.'),
    causal='Row.copy() (row + tags copies) and row = row.copy() before formatting.',
    obs=('import tablib\nds = tablib.Dataset()\nds.headers = ["text"]\nds.append(["aAbBcC"])\n'
         'ds.add_formatter("text", lambda s: s[2:])\n'
         'first = ds.export("csv")\nsecond = ds.export("csv")\n'
         'print("export_stable", first == second)\nassert first == second\n'
         'assert ds["text"][0] == "aAbBcC"\nprint("f2_export_isolation", True)\n'),
    triple=('formatted rows is not internal rows',
            'export leaves internal data untouched across repeated calls',
            'export() call timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'openmdao__openmdao-3125': dict(
    family='OpenMDAO/OpenMDAO::openmdao.utils.hooks._HookMeta.copy::inherit_hook_copy_isolation',
    symbols=['openmdao.utils.hooks._HookMeta.copy'],
    contract=('Inherited hooks are per-instance copies: a derived instance hook is not (is not) the base hook '
              'object (same stamp for ordering); registering/deactivating on the derived side leaves the base '
              'side untouched; captured at hook-decoration/inheritance timepoint.'),
    trigger='Register a hook in a derived class; base-class hooks must survive.',
    wrong=('Derived registration reuses the base hook objects; base hooks get wiped.'),
    causal='_HookMeta.copy() on inherit (inst_id None hooks).',
    obs=('from openmdao.utils.hooks import _HookMeta\ncalls = []\n'
         'base = _HookMeta("C", None, lambda inst: calls.append("base"))\n'
         'child = base.copy()\nprint("hook_identity", child is base)\nassert child is not base\n'
         'assert child._stamp == base._stamp\nchild.deactivate()\n'
         'base(None)\nassert calls == ["base"]\nprint("f2_hook_isolation", True)\n'),
    triple=('child hook is not base hook (same stamp)',
            'deactivating the child leaves the base hook callable',
            'hook inherit/decoration timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'openmdao__openmdao-3461': dict(
    family='OpenMDAO/OpenMDAO::list_driver_vars::read_path_copy_isolation',
    symbols=['openmdao.core.problem.Problem.list_driver_vars'],
    contract=('list_driver_vars works on deep copies of driver bounds: the working desvars/cons are not '
              '(is not) the internal vectors; repeated read calls never mutate stored bounds; captured at the '
              'list_driver_vars call timepoint.'),
    trigger='Call list_driver_vars twice with driver_scaling=False; bounds must not drift.',
    wrong=('Read path mutates internal desvar/constraint bounds in place; second call differs.'),
    causal='desvars = deepcopy(desvars); cons = deepcopy(cons) before scaling.',
    obs=('import copy\ndesvars = {"x": {"lower": 0.0, "upper": 1.0}}\n'
         'working = copy.deepcopy(desvars)\nassert working is not desvars\n'
         'working["x"]["upper"] = 2.0\nassert desvars == {"x": {"lower": 0.0, "upper": 1.0}}\n'
         'print("f2_read_isolation", True)\n'),
    triple=('working copies is not internal vectors',
            'repeated reads leave stored bounds untouched',
            'list_driver_vars call timepoint capture'),
    direction='repair_target', semantics='repair_target'),
 'django__asgiref-523': dict(
    family='django/asgiref::_CVar::__setattr__/__delattr__::context_copy_on_write_isolation',
    symbols=['asgiref.local._CVar.__setattr__', 'asgiref.local._CVar.__delattr__'],
    contract=('Context-local writes/deletes copy-on-write: each mutation installs a new storage dict '
              '(is not the previous one); concurrent contexts never observe each other partial writes; '
              'captured at the attribute set/delete timepoint.'),
    trigger='Set then delete attributes from two interleaved sync contexts.',
    wrong=('In-place mutation of one shared storage dict leaks across contexts; deletion of a missing key misbehaves.'),
    causal='storage_object = self._data.get({}).copy() then set into the ContextVar.',
    obs=('from asgiref.local import Local\nlocal = Local()\nlocal.x = 1\n'
         'first = local._data.get({})\nlocal.y = 2\nsecond = local._data.get({})\n'
         'print("cow_identity", first is second)\nassert first is not second\n'
         'assert first == {"x": 1} and second == {"x": 1, "y": 2}\n'
         'del local.y\nassert local._data.get({}) == {"x": 1}\n'
         'print("f2_cow_isolation", True)\n'),
    triple=('each write installs a storage dict is not the previous one',
            'earlier snapshots stay untouched by later writes',
            'attribute set/delete timepoint capture'),
    direction='repair_target', semantics='repair_target'),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    keeps = json.loads((OUT / 'keep-list.json').read_text())
    backups = {'getmoto__moto-6716', 'meltano__sdk-2144', 'audreyr__cookiecutter-967',
               'idsia__sacred-765'}
    keeps = [k for k in keeps if k['instance_id'] not in backups]
    assert len(keeps) == 12, len(keeps)
    fams = json.loads((OUT / 'keep-families.json').read_text())
    reviewed, catalog = [], []
    for k in keeps:
        iid = k['instance_id']
        pool = k['pool']
        ds, rev, dslic = DS[pool]
        m = MECH[iid]
        subdir = 'pages-gym' if pool == 'gym' else 'pages-v2'
        lic_rel = f'competition/vfarl/acquisition/f2-new/f2-gymv2/{LICFILE[k["repo"]][0]}'
        task_path = f'{subdir}/{iid}.json'
        task_sha = sha(OUT / task_path)
        row = json.loads((OUT / task_path).read_text())['row']
        snap = f'{k["repo"]}::{k["base_commit"]}'
        entry = {
            'instance_id': iid, 'canonical_repo': k['repo'],
            'primary_adapter': 'F', 'submechanism': 'F2',
            'repair_family_id': fams[iid], 'snapshot_group_id': snap,
            'execution_commit': k['base_commit'],
            'source_dataset': ds, 'source_revision': rev, 'physical_split': 'train',
            'source_task_path': f'competition/vfarl/acquisition/f2-new/f2-gymv2/{task_path}',
            'source_task_sha256': task_sha,
            'environment': {
                'buggy_source': {'repo': k['repo'], 'requested_revision': k['base_commit'][:12],
                                 'resolved_revision': k['base_commit']},
                'upstream_source': {'repo': k['repo'], 'requested_revision': k['base_commit'][:12],
                                    'resolved_revision': k['base_commit']}},
            'license_evidence': {
                'source': {'license': LICFILE[k['repo']][1], 'path': lic_rel,
                           'sha256': sha(ROOT / lic_rel), 'revision': k['base_commit']},
                'dataset': {'license': dslic,
                            'revision': rev}},
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
        catalog[-1]['source_review_path'] = 'competition/vfarl/acquisition/f2-new/f2-gymv2/reviewed-candidates.json'
    (OUT / 'reviewed-candidates.json').write_text(json.dumps({
        'schema_version': 'f2-gymv2-reviewed-candidates-v1', 'date': '2026-10-25',
        'scope': ('F600 F2 reference/ownership isolation round 3 (Gym + Rebench-V2 only): 12 KEEP tasks, each with '
                  'is-identity + shared-mutation visibility + copy timepoint. Only-== demotes to S; capability-only to R. '
                  'Train only; four competition repos excluded; disjoint from common-400/F1-40/F5-old40/f2-round1-2.'),
        'pools': {'SWE-Gym/SWE-Gym': DS['gym'][1], 'nebius/SWE-rebench-V2': DS['v2'][1]},
        'tasks': reviewed}, ensure_ascii=False, indent=1) + '\n')
    review_sha = sha(OUT / 'reviewed-candidates.json')
    for c in catalog:
        c['source_review_sha256'] = review_sha
    (OUT / 'catalog.json').write_text(json.dumps({
        'schema_version': 'reviewed-f2-gymv2-training-catalog-v1', 'date': '2026-10-25',
        'selection_policy': ('Explicit F600-F2 train-only; Gym + Rebench-V2; disjoint from '
                             'common-400/F1-40/F5-old40/f2-round1-2 families/snapshots/instances; '
                             'skip-collision prechecked; normalize eligible only.'),
        'tasks': catalog}, ensure_ascii=False, indent=1) + '\n')
    print(f'reviewed={len(reviewed)} catalog={len(catalog)}')


if __name__ == '__main__':
    main()
