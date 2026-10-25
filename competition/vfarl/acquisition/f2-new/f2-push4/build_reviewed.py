"""F2 f2-push4: build reviewed-candidates.json + catalog.json for 1 triple-KEEP."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-push4'

V2_REV = '10483de0f50fe5da545942705a76c6150171af7f'
GYM_REV = 'bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb'

FAMS = json.loads((ACQ / 'keep-families.json').read_text())

EVIDENCE = {
 'cs-si__eodag-253': {
  'canonical_repo': 'CS-SI/eodag', 'pool': 'v2', 'license': 'Apache-2.0', 'lic_file': 'eodag.LICENSE',
  'family': FAMS['cs-si__eodag-253'],
  'contract': 'download_all owns a per-call working sequence: the working products list is not (is not) the caller SearchResult; consuming (popping) the working copy leaves the caller sequence untouched; captured at the download_all entry timepoint.',
  'trigger': 'Download several products with download_all, then keep using the caller-side SearchResult.',
  'base_wrong_behavior': 'The working variable aliases the caller SearchResult; per-product removal empties the caller list in place.',
  'production_symbols': ['eodag.plugins.download.base.Download.download_all'],
  'causal_change': 'products = products[:] at the download_all entry (working snapshot before per-product removal).',
  'observation': ("from eodag.api.search_result import SearchResult\n"
   "prods = SearchResult([object(), object(), object()])\n"
   "n_before = len(prods)\n"
   "working = prods[:]\n"
   "print('identities', working is prods)\n"
   "assert working is not prods\n"
   "working.pop(0)\n"
   "assert len(prods) == n_before, 'side effect: caller sequence consumed'\n"
   "assert len(working) == n_before - 1\n"
   "print('f2_download_isolation', True)\n"),
  'triple_is': 'working products is not the caller SearchResult (per-call owned snapshot)',
  'triple_vis': 'per-product removal from the working copy stays invisible to the caller sequence (F2P download assertions + intact caller)',
  'triple_tp': 'download_all entry timepoint capture',
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
        tpath = f'competition/vfarl/acquisition/f2-new/f2-push4/{subdir}/{iid}.json'
        base = k['base_commit']
        fam = ev['family']
        snap = f"{ev['canonical_repo']}::{base}"
        lic_path = f'competition/vfarl/acquisition/f2-new/f2-push4/{ev["lic_file"]}'
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
                'date': '2026-10-25', 'reviewer': 'subagent f2-push4',
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
        'schema_version': 'f2-push4-reviewed-candidates-v1',
        'date': '2026-10-25',
        'scope': 'F600 F2 reference/ownership isolation round 13 (Gym+V2 weak-offset<=150 + V2 large-recall remainder 151-200 + P2P-empty 51-200 new window): '
                 '1 KEEP task with is-identity + shared-mutation visibility + copy timepoint. '
                 'Only-== demotes to S; capability-only to R. Train only; four competition repos excluded; '
                 'disjoint from common-400/F1-40/F5-old40/f2-round1-2/gymv2/b/c/recall/inline/more/push/push2/push3.',
        'pools': {'SWE-Gym/SWE-Gym': GYM_REV, 'nebius/SWE-rebench-V2': V2_REV},
        'tasks': tasks,
    }
    (ACQ / 'reviewed-candidates.json').write_text(json.dumps(reviewed, ensure_ascii=False, indent=1) + '\n')
    rsha = sha(ACQ / 'reviewed-candidates.json')
    base_res = {t['instance_id']: {'requested': t['environment']['buggy_source']['requested_revision'],
                                   'resolved': t['execution_commit']} for t in tasks}
    (ACQ / 'base-resolutions.json').write_text(json.dumps(base_res, ensure_ascii=False, indent=1) + '\n')
    catalog = {
        'schema_version': 'reviewed-f2-push4-training-catalog-v1',
        'date': '2026-10-25',
        'selection_policy': 'Explicit F600-F2 train-only; disjoint from S600-common-400 families/snapshots/instances; normalize eligible only.',
        'tasks': [{**t, 'source_review_path': 'competition/vfarl/acquisition/f2-new/f2-push4/reviewed-candidates.json',
                   'source_review_sha256': rsha,
                   'source_mechanism_review_complete': True} for t in tasks],
    }
    (ACQ / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=1) + '\n')
    print(f'reviewed={len(tasks)} catalog={len(catalog["tasks"])} review_sha={rsha[:16]}')


if __name__ == '__main__':
    main()
