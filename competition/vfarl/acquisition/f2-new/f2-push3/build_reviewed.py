"""F2 f2-push3: build reviewed-candidates.json + catalog.json for 2 triple-KEEPs."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-push3'

V2_REV = '10483de0f50fe5da545942705a76c6150171af7f'
GYM_REV = 'bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb'

FAMS = json.loads((ACQ / 'keep-families.json').read_text())

EVIDENCE = {
 'aeye-lab__pymovements-486': {
  'canonical_repo': 'aeye-lab/pymovements', 'pool': 'v2', 'license': 'MIT', 'lic_file': 'pymovements.LICENSE',
  'family': FAMS['aeye-lab__pymovements-486'],
  'contract': 'save_preprocessed owns a per-file working copy: the working gaze_df is not (is not) the caller GazeDataFrame; csv unnest/drop of the working copy leaves the caller frame untouched; captured at the save_preprocessed per-file loop-iteration timepoint.',
  'trigger': 'Save a gaze dataset with nested pixel/position/velocity/acceleration columns to csv, then keep preprocessing the in-memory frames.',
  'base_wrong_behavior': 'The loop variable aliases the caller GazeDataFrame; csv unnest/drop mutates the caller frame in place.',
  'production_symbols': ['pymovements.dataset.dataset_files.save_preprocessed', 'pymovements.gaze.gaze_dataframe.GazeDataFrame.copy'],
  'causal_change': 'gaze_df = gaze_df.copy() at the loop top; GazeDataFrame.copy() preserves n_components.',
  'observation': ("import tempfile\nimport polars as pl\n"
   "from pymovements.dataset.dataset_files import save_preprocessed\n"
   "from pymovements.dataset.dataset_paths import DatasetPaths\n"
   "from pymovements.gaze.gaze_dataframe import GazeDataFrame\n"
   "df = pl.from_dict({'t': [1000, 1001, 1002], 'x': [0.1, 0.2, 0.3], 'y': [0.1, 0.2, 0.3]})\n"
   "gaze = GazeDataFrame(data=df, pixel_columns=['x', 'y'])\n"
   "cols_before = list(gaze.frame.columns)\n"
   "print('cols_before', cols_before)\n"
   "paths = DatasetPaths(root=tempfile.mkdtemp(), dataset='ds')\n"
   "fileinfo = pl.DataFrame({'filepath': ['f.csv']})\n"
   "save_preprocessed(gaze=[gaze], fileinfo=fileinfo, paths=paths, verbose=0, extension='csv')\n"
   "print('cols_after', list(gaze.frame.columns))\n"
   "assert list(gaze.frame.columns) == cols_before, 'side effect: caller frame mutated'\n"
   "print('f2_save_isolation', True)\n"),
  'triple_is': 'working gaze_df is not the caller GazeDataFrame (per-file owned copy)',
  'triple_vis': 'csv unnest/drop on the working copy stays invisible to the caller frame (assert_frame_equal in F2P)',
  'triple_tp': 'save_preprocessed per-file loop-iteration timepoint capture',
 },
 'jupyter__kernel_gateway-118': {
  'canonical_repo': 'jupyter/kernel_gateway', 'pool': 'v2', 'license': 'BSD-3-Clause', 'lic_file': 'kernel_gateway.LICENSE',
  'family': FAMS['jupyter__kernel_gateway-118'],
  'contract': 'Each kernel owns its activity map: a per-kernel map is not (is not) another kernel map nor the global defaults; publishing activity for one kernel leaves defaults and other kernels untouched; captured at the get_map_for_kernel access timepoint.',
  'trigger': 'Track two kernels (or query an unknown kernel id), then publish activity for one.',
  'base_wrong_behavior': 'Unknown-kernel queries return the single shared dummy_map object; mutations through it leak across kernels and the defaults.',
  'production_symbols': ['kernel_gateway.services.activity.manager.ActivityManager.get_map_for_kernel', 'kernel_gateway.services.activity.manager.ActivityManager'],
  'causal_change': 'Unknown kernels get DEFAULT_ACTIVITY_VALUES.copy(); each tracked kernel gets its own DEFAULT_ACTIVITY_VALUES.copy().',
  'observation': ("from kernel_gateway.services.activity.manager import (\n"
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
   "print('f2_kernel_isolation', True)\n"),
  'triple_is': 'per-kernel map is not another kernel map nor the global DEFAULT_ACTIVITY_VALUES',
  'triple_vis': 'publishing to one kernel stays invisible to defaults and other kernels',
  'triple_tp': 'get_map_for_kernel access timepoint capture',
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
        tpath = f'competition/vfarl/acquisition/f2-new/f2-push3/{subdir}/{iid}.json'
        base = k['base_commit']
        fam = ev['family']
        snap = f"{ev['canonical_repo']}::{base}"
        lic_path = f'competition/vfarl/acquisition/f2-new/f2-push3/{ev["lic_file"]}'
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
                'date': '2026-10-25', 'reviewer': 'subagent f2-push3',
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
    reviewed = {
        'schema_version': 'f2-push3-reviewed-candidates-v1',
        'date': '2026-10-25',
        'scope': 'F600 F2 reference/ownership isolation round 12 (Gym + Rebench-V2 mid-suite 151-400 + P2P-empty<=50 inline-is, push2 remainder): '
                 '2 KEEP tasks, each with is-identity + shared-mutation visibility + copy timepoint. '
                 'Only-== demotes to S; capability-only to R. Train only; four competition repos excluded; '
                 'disjoint from common-400/F1-40/F5-old40/f2-round1-2/gymv2/b/c/recall/inline/more/push/push2.',
        'pools': {'SWE-Gym/SWE-Gym': GYM_REV, 'nebius/SWE-rebench-V2': V2_REV},
        'tasks': tasks,
    }
    (ACQ / 'reviewed-candidates.json').write_text(json.dumps(reviewed, ensure_ascii=False, indent=1) + '\n')
    rsha = sha(ACQ / 'reviewed-candidates.json')
    base_res = {t['instance_id']: {'requested': t['environment']['buggy_source']['requested_revision'],
                                   'resolved': t['execution_commit']} for t in tasks}
    (ACQ / 'base-resolutions.json').write_text(json.dumps(base_res, ensure_ascii=False, indent=1) + '\n')
    catalog = {
        'schema_version': 'reviewed-f2-push3-training-catalog-v1',
        'date': '2026-10-25',
        'selection_policy': 'Explicit F600-F2 train-only; disjoint from S600-common-400 families/snapshots/instances; normalize eligible only.',
        'tasks': [{**t, 'source_review_path': 'competition/vfarl/acquisition/f2-new/f2-push3/reviewed-candidates.json',
                   'source_review_sha256': rsha,
                   'source_mechanism_review_complete': True} for t in tasks],
    }
    (ACQ / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n')
    print(f'reviewed={len(tasks)} catalog={len(catalog["tasks"])} review_sha={rsha[:16]}')


if __name__ == '__main__':
    main()
