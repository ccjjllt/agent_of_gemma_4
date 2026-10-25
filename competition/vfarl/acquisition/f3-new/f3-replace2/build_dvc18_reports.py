"""Emit f3r-dvc-18 replay-result + normalize catalog entry + run normalizer. Additive only."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
NEW = ROOT / 'competition/vfarl/acquisition/f3-new/f3-replace2'
REPLAY = ROOT / 'competition/vfarl/replay'
ART2 = REPLAY / 'artifacts/f3-replace2'

KEY = 'f3r-dvc-18'


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    prep = json.loads((REPLAY / f'f3r2-{KEY}-source-preparation.json').read_text())
    st = json.loads((NEW / f'{KEY}-stage1c.json').read_text())
    iid = prep['instance_id']
    run = ART2 / f'{KEY}-run-001'
    assert (run / 'run-summary.json').exists()
    report = {
        'instance_id': iid,
        'adapted_task': prep['adapted_task'],
        'run_directory': str(run.relative_to(ROOT)),
        'current_replay': str(run.relative_to(ROOT)),
        'mechanism': {'adapter': 'F', 'submechanism': 'F3',
                      'evidence': prep['mechanism_evidence']},
        'primary_adapter': 'F',
        'submechanism': 'F3',
        'canonical_repo': prep['canonical_repo'],
        'source_train_revision': '77cab9055d42ab4a5c25c89a8f937096db13558e',
        'source_preparation': str((REPLAY / f'f3r2-{KEY}-source-preparation.json').relative_to(ROOT)),
        'recipe': prep['recipe'],
        'run_present': True,
    }
    report_path = REPLAY / f'f3r2-{KEY}-replay-result.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    # catalog entry (mirror build_f3replace_reports fields)
    entry = {
        'instance_id': iid,
        'canonical_repo': prep['canonical_repo'],
        'primary_adapter': 'F',
        'submechanism': 'F3',
        'repair_family_id': prep['repair_family_id'],
        'snapshot_group_id': prep['snapshot_group_id'],
        'execution_commit': prep['execution_commit'],
        'source_dataset': 'SWE-bench/SWE-smith-py',
        'source_revision': '77cab9055d42ab4a5c25c89a8f937096db13558e',
        'physical_split': 'train',
        'source_task_path': prep['original_task'],
        'source_task_sha256': prep['original_task_sha256'],
        'source_review_path': prep['source_review_path'],
        'source_review_sha256': sha(ROOT / prep['source_review_path']),
        'course_membership': 'F600_F3_new',
        'use_partition': 'train_only',
        'original_FAIL_TO_PASS_count': prep['F2P_count'],
        'original_PASS_TO_PASS_count': prep['P2P_count'],
        'original_patch_semantics': 'bug_introducing',
        'mechanism_evidence': prep['mechanism_evidence'],
        'environment': {'buggy_source': prep['buggy_source'],
                        'upstream_source': prep['upstream_source']},
        'license_evidence': prep['license_evidence'],
    }
    (NEW / f'{KEY}-normalize-catalog.json').write_text(
        json.dumps({'tasks': [entry]}, ensure_ascii=False, indent=2) + '\n')
    catalog = {entry['instance_id']: entry}
    approvals = json.loads((REPLAY / 'approved-source-requirement-mappings.json').read_text())
    appr = {row['instance_id']: row for row in approvals.get('approvals', [])}
    spec = importlib.util.spec_from_file_location('nr', REPLAY / 'normalize_results.py')
    nr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nr)
    res = nr.normalize(report_path, catalog, appr)
    out = {'instance_id': iid, 'report': str(report_path.relative_to(ROOT)),
           'eligible': res.get('eligible_for_human_off_policy_SFT', False),
           'reasons': res.get('ineligibility_reasons', []),
           'complete': res.get('complete_official_replay', False),
           'baseline_f2p': res.get('baseline_verified_original_F2P_failure', False),
           'resolved': res.get('resolved_by_local_official_verify_task'),
           'strict_ids': res.get('strict_verbatim_required_ids_passed', False),
           'visible_profile': (res.get('visible_prompt_profile') or {}).get('kind'),
           'error': res.get('normalization_error')}
    (NEW / f'{KEY}-normalize-result.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(out, indent=1), flush=True)
    assert out['eligible'], out['reasons']


if __name__ == '__main__':
    main()
