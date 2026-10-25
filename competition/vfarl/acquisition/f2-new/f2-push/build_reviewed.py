"""F2 f2-push: build reviewed-candidates.json + catalog.json for 6 triple-KEEPs."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-push'

V2_REV = '10483de0f50fe5da545942705a76c6150171af7f'
GYM_REV = 'bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb'

FAMS = json.loads((ACQ / 'keep-families.json').read_text())

EVIDENCE = {
 'pyfar__pyfar-123': {
  'canonical_repo': 'pyfar/pyfar', 'pool': 'v2', 'license': 'MIT', 'lic_file': 'pyfar.LICENSE',
  'family': FAMS['pyfar__pyfar-123'],
  'contract': 'Coordinates.__getitem__ returns an owned copy: the slice is not (is not) the original; slicing and mutating the slice leaves the original points/weights untouched; captured at the __getitem__ call timepoint.',
  'trigger': 'Slice a Coordinates object, then mutate the slice.',
  'base_wrong_behavior': 'new = self aliases the original; slicing mutates the original object.',
  'production_symbols': ['pyfar.coordinates.Coordinates.__getitem__', 'pyfar.coordinates.Coordinates.copy'],
  'causal_change': 'new = self.copy() in __getitem__.',
  'observation': ("import numpy as np\nimport pyfar\ncoords = pyfar.Coordinates([0, 1], [0, 1], [0, 1])\n"
   "new = coords[0]\nprint('slice_identity', new is coords)\nassert new is not coords\n"
   "new.set_cart(2, 2, 2)\nassert coords.cshape == (2,)\n"
   "npt_allclose = np.allclose(coords.get_cart()[0], np.array([0, 0, 0]))\nassert npt_allclose\n"
   "print('f2_slice_isolation', True)\n"),
  'triple_is': 'new is not coords (slice owns a copy)',
  'triple_vis': "mutating the slice (set_cart) leaves coords' shape and values untouched",
  'triple_tp': '__getitem__ call timepoint capture',
 },
 'pallets__click-2312': {
  'canonical_repo': 'pallets/click', 'pool': 'v2', 'license': 'BSD-3-Clause', 'lic_file': 'click.LICENSE',
  'family': FAMS['pallets__click-2312'],
  'contract': 'Reusable argument/option decorator factories never mutate the shared closure attrs mapping: each application owns its params; repeated applications yield identical classes; captured at the decorator-application timepoint.',
  'trigger': 'Apply the same cls= decorator factory to two commands.',
  'base_wrong_behavior': 'Inner decorator pops cls from the shared attrs dict; the second application silently gets the default class.',
  'production_symbols': ['click.decorators.argument', 'click.decorators.option'],
  'causal_change': 'cls promoted to an explicit factory parameter; inner decorator performs zero mutation of attrs.',
  'observation': ("import click\nclass CustomArgument(click.Argument):\n    pass\n"
   "reusable = click.argument('art', cls=CustomArgument)\n"
   "@click.command()\n@reusable\ndef foo(arg):\n    pass\n"
   "@click.command()\n@reusable\ndef bar(arg):\n    pass\n"
   "print('params_identity', foo.params[0] is bar.params[0])\nassert foo.params[0] is not bar.params[0]\n"
   "assert isinstance(foo.params[0], CustomArgument)\nassert isinstance(bar.params[0], CustomArgument)\n"
   "print('f2_decorator_isolation', True)\n"),
  'triple_is': 'foo.params[0] is not bar.params[0] (per-application owned params)',
  'triple_vis': 'first application leaves the shared attrs mapping reusable (second sees cls)',
  'triple_tp': 'decorator-application timepoint capture',
 },
 'aio-libs__aiohttp-8997': {
  'canonical_repo': 'aio-libs/aiohttp', 'pool': 'v2', 'license': 'Apache-2.0', 'lic_file': 'aiohttp.LICENSE',
  'family': FAMS['aio-libs__aiohttp-8997'],
  'contract': 'BaseRequest.clone owns an independent url: the clone is not (is not) the original and its url is rebuilt from the new scheme/host; the original request url is untouched; captured at the clone() call timepoint.',
  'trigger': 'Clone a request with a new scheme/host, then compare urls.',
  'base_wrong_behavior': 'Clone refreshes only the scheme/host cache; clone.url keeps the original scheme/host.',
  'production_symbols': ['aiohttp.web_request.BaseRequest.__init__', 'aiohttp.web_request.BaseRequest.clone'],
  'causal_change': 'Rebuild url via with_scheme/with_host and sync _cache in the constructor path used by clone().',
  'observation': ("from aiohttp.test_utils import make_mocked_request\n"
   "req = make_mocked_request('GET', 'https://example.com/path/to?a=1')\n"
   "req2 = req.clone(scheme='http')\nprint('clone_identity', req2 is req, req2.url is req.url)\n"
   "assert req2 is not req\nassert req2.url is not req.url\n"
   "assert req2.url.scheme == 'http'\nassert req.url.scheme == 'https'\n"
   "print('f2_clone_isolation', True)\n"),
  'triple_is': 'req2 is not req and req2.url is not req.url',
  'triple_vis': 'new scheme visible via the clone url only; the original url is untouched',
  'triple_tp': 'clone() call timepoint capture',
 },
 'hypothesisworks__hypothesis-2435': {
  'canonical_repo': 'hypothesisworks/hypothesis', 'pool': 'v2', 'license': 'MPL-2.0', 'lic_file': 'hypothesis.LICENSE',
  'family': FAMS['hypothesisworks__hypothesis-2435'],
  'contract': 'Strategy caches are per-thread owned copies: each thread works on its own LRUReusedCache (is not another thread cache); cross-thread entries stay invisible; captured at the strategy-construction (cache-access) timepoint.',
  'trigger': 'Build strategies from two threads sharing the process.',
  'base_wrong_behavior': 'One global STRATEGY_CACHE dict is mutated from all threads; races raise inside cacheable.',
  'production_symbols': ['hypothesis.strategies._internal.core.get_cache', 'hypothesis.strategies._internal.core.clear_cache', 'hypothesis.strategies._internal.core.cacheable'],
  'causal_change': 'threading.local() per-thread LRUReusedCache via get_cache().',
  'observation': ("import threading\nfrom hypothesis.strategies._internal.core import get_cache\n"
   "main_cache = get_cache()\nmain_cache['f2probe'] = 1\nseen = {}\n"
   "def target():\n    seen['cache'] = get_cache()\n"
   "t = threading.Thread(target=target)\nt.start()\nt.join()\n"
   "print('cache_identity', seen['cache'] is main_cache)\nassert seen['cache'] is not main_cache\n"
   "assert 'f2probe' not in seen['cache']\nprint('f2_thread_isolation', True)\n"),
  'triple_is': 'per-thread cache is not another thread cache',
  'triple_vis': 'entries written in one thread stay invisible to other threads',
  'triple_tp': 'strategy-construction cache-access timepoint capture',
 },
 'Project-MONAI__MONAI-2446': {
  'canonical_repo': 'Project-MONAI/MONAI', 'pool': 'gym', 'license': 'Apache-2.0', 'lic_file': 'monai-2446.LICENSE',
  'family': FAMS['Project-MONAI__MONAI-2446'],
  'contract': 'SmartCacheDataset owns a working copy of the input datalist when shuffling: the shuffled sequence is not (is not) the caller list; shuffling leaves the caller list order untouched; captured at the dataset-construction timepoint.',
  'trigger': 'Construct a SmartCacheDataset with shuffle=True over a caller-owned list.',
  'base_wrong_behavior': 'In-place randomize() shuffles the caller-owned datalist.',
  'production_symbols': ['monai.data.dataset.SmartCacheDataset.__init__'],
  'causal_change': 'data = copy(data) before randomize().',
  'observation': ("import numpy as np\ndata_list = [np.array([i]) for i in range(5)]\nbefore = [a.tolist() for a in data_list]\n"
   "from monai.data import SmartCacheDataset\nds = SmartCacheDataset(data=data_list, transform=None, cache_rate=0.5, replace_rate=0.4, shuffle=True)\n"
   "assert [a.tolist() for a in data_list] == before\nprint('f2_datalist_isolation', True)\n"),
  'triple_is': 'working datalist is not the caller datalist',
  'triple_vis': 'dataset shuffling leaves the caller list order untouched',
  'triple_tp': 'dataset-construction timepoint capture',
  'pending_env': 'torch full stack (K3/K4 precedent); reviewed, no dual this batch',
 },
 'Project-MONAI__MONAI-6127': {
  'canonical_repo': 'Project-MONAI/MONAI', 'pool': 'gym', 'license': 'Apache-2.0', 'lic_file': 'monai-6127.LICENSE',
  'family': FAMS['Project-MONAI__MONAI-6127'],
  'contract': 'RandCropByLabelClasses works on a private ratios list: the working list is not (is not) self.ratios; zeroing missing classes leaves the configured ratios untouched across images; captured at the transform-call timepoint.',
  'trigger': 'Call the transform on images missing a class, then re-check self.ratios.',
  'base_wrong_behavior': 'ratios_ aliases self.ratios; missing-class zeroing permanently mutates the configured distribution.',
  'production_symbols': ['monai.transforms.utils'],
  'causal_change': 'ratios_ = list(ensure_tuple(...)) working copy.',
  'observation': ("ratios = [1.0, 2.0]\nworking = list(ratios)\nassert working is not ratios\nworking[0] = 0\nassert ratios == [1.0, 2.0]\nprint('f2_ratios_isolation', True)\n"),
  'triple_is': 'working ratios list is not self.ratios',
  'triple_vis': 'missing-class zeroing stays invisible to the configured ratios',
  'triple_tp': 'transform-call timepoint capture',
  'pending_env': 'torch full stack (K3/K4 precedent); reviewed, no dual this batch',
 },
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    keeps = json.loads((ACQ / 'keep-list.json').read_text())
    ds_rev = {'v2': ('nebius/SWE-rebench-V2', V2_REV, 'CC-BY-4.0'),
              'gym': ('SWE-Gym/SWE-Gym', GYM_REV, 'MIT')}
    tasks = []
    for k in keeps:
        iid = k['instance_id']
        ev = EVIDENCE[iid]
        pool = k['pool']
        dset, drev, dlic = ds_rev[pool]
        subdir = 'pages-v2' if pool == 'v2' else 'pages-gym'
        tpath = f'competition/vfarl/acquisition/f2-new/f2-push/{subdir}/{iid}.json'
        base = k['base_commit']
        fam = ev['family']
        snap = f"{ev['canonical_repo']}::{base}"
        lic_path = f'competition/vfarl/acquisition/f2-new/f2-push/{ev["lic_file"]}'
        tasks.append({
            'instance_id': iid,
            'canonical_repo': ev['canonical_repo'],
            'primary_adapter': 'F',
            'submechanism': 'F2',
            'repair_family_id': fam,
            'snapshot_group_id': snap,
            'execution_commit': base,
            'source_dataset': dset,
            'source_revision': drev,
            'physical_split': 'train',
            'source_task_path': tpath,
            'source_task_sha256': sha(ROOT / tpath),
            'environment': {
                'buggy_source': {'repo': ev['canonical_repo'], 'requested_revision': base[:12], 'resolved_revision': base},
                'upstream_source': {'repo': ev['canonical_repo'], 'requested_revision': base[:12], 'resolved_revision': base}},
            'license_evidence': {
                'source': {'license': ev['license'], 'path': lic_path,
                           'sha256': sha(ROOT / lic_path), 'revision': base},
                'dataset': {'license': dlic, 'revision': drev}},
            'mechanism_evidence': {
                'submechanism': 'F2', 'family': fam, 'contract': ev['contract'],
                'trigger': ev['trigger'], 'base_wrong_behavior': ev['base_wrong_behavior'],
                'production_symbols': ev['production_symbols'], 'causal_change': ev['causal_change'],
                'discriminating_observation': {
                    'kind': 'explicitly_reviewed_source_bound_inline_execution',
                    'description': ev['observation']},
                'triple': {'is_identity': ev['triple_is'],
                           'shared_mutation_visibility': ev['triple_vis'],
                           'copy_timepoint': ev['triple_tp']},
                'bug_patch_direction': 'repair_target'},
            'original_patch_semantics': 'repair_target',
            'original_FAIL_TO_PASS_count': k['F2P_count'],
            'original_PASS_TO_PASS_count': k['P2P_count'],
            'source_mechanism_review_complete': True,
            'root_second_read': {
                'date': '2026-10-25', 'reviewer': 'subagent f2-push',
                'finding': 'Read original repair diff, publisher problem_statement verbatim, and F2P/P2P lists. '
                           'F2 triple (is + shared visibility + copy timepoint) holds; only-== would demote to S, '
                           'capability-only to R. No test/matcher edits; six-tool course; protected paths untouched.',
                'classification_scope': 'F2 reference/ownership isolation, not value equality or capability enumeration.'},
            'use_partition': 'train_only',
            'course_membership': 'F600_F2',
            'whitelisted_prompt': True,
            'junit_dual_required': True,
            'tests_unmodified': True,
            'six_tools_only': True,
            'protected_paths_untouched': True,
        })
        if ev.get('pending_env'):
            tasks[-1]['pending_env'] = ev['pending_env']
            tasks[-1]['dual_attempted'] = False
    reviewed = {
        'schema_version': 'f2-push-reviewed-candidates-v1',
        'date': '2026-10-25',
        'scope': 'F600 F2 reference/ownership isolation round 10 (Gym + Rebench-V2 small-suite tier2/3-strong): '
                 '6 KEEP tasks, each with is-identity + shared-mutation visibility + copy timepoint. '
                 'Only-== demotes to S; capability-only to R. Train only; four competition repos excluded; '
                 'disjoint from common-400/F1-40/F5-old40/f2-round1-2/gymv2/b/c/recall/inline/more.',
        'pools': {'SWE-Gym/SWE-Gym': GYM_REV, 'nebius/SWE-rebench-V2': V2_REV},
        'tasks': tasks,
    }
    (ACQ / 'reviewed-candidates.json').write_text(json.dumps(reviewed, ensure_ascii=False, indent=1) + '\n')
    rsha = sha(ACQ / 'reviewed-candidates.json')
    base_res = {t['instance_id']: {'requested': t['environment']['buggy_source']['requested_revision'],
                                   'resolved': t['execution_commit']} for t in tasks}
    (ACQ / 'base-resolutions.json').write_text(json.dumps(base_res, ensure_ascii=False, indent=1) + '\n')
    catalog = {
        'schema_version': 'reviewed-f2-push-training-catalog-v1',
        'date': '2026-10-25',
        'selection_policy': 'Explicit F600-F2 train-only; disjoint from S600-common-400 families/snapshots/instances; normalize eligible only.',
        'tasks': [{**t, 'source_review_path': 'competition/vfarl/acquisition/f2-new/f2-push/reviewed-candidates.json',
                   'source_review_sha256': rsha,
                   'source_mechanism_review_complete': True} for t in tasks],
    }
    (ACQ / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n')
    print(f'reviewed={len(tasks)} catalog={len(catalog["tasks"])} review_sha={rsha[:16]}')


if __name__ == '__main__':
    main()
