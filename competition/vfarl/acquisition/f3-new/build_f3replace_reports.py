"""Stage 5: emit replay-result reports + normalize catalog, run normalize.

Writes:
  replay/f3r-<key>-replay-result.json   (one per ACTIVE task with a run)
Uses:
  acquisition/f3-new/f3-replace/f3-replace-normalize-catalog.json (built here)
  replay/approved-source-requirement-mappings.json (existing approvals)
Prints eligible count + reasons.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f3-new'
OUT = ACQ / 'f3-replace'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f3-replace'

sys.path.insert(0, str(ACQ))
from f3replace_pool import ACTIVE_TASKS  # noqa: E402


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else set()
    stage1c = json.loads((OUT / 'stage1c-manifest.json').read_text())
    by_key = {t['key']: t for t in stage1c['tasks']}
    catalog = {'tasks': []}
    for key, iid, repo, canonical, upstream_short, symbol, contract in ACTIVE_TASKS():
        if only and key not in only:
            continue
        m = by_key[key]
        prep = json.loads((REPLAY / f'f3r-{key}-source-preparation.json').read_text())
        entry = {
            'instance_id': iid,
            'canonical_repo': canonical,
            'primary_adapter': 'F',
            'submechanism': 'F3',
            'repair_family_id': prep['repair_family_id'],
            'snapshot_group_id': prep['snapshot_group_id'],
            'execution_commit': m['execution_commit'],
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
        if prep.get('visible_prompt_profile'):
            entry['visible_prompt_profile'] = prep['visible_prompt_profile']
        catalog['tasks'].append(entry)
        # replay report
        run = ART / f'{key}-run-001'
        report = {
            'instance_id': iid,
            'adapted_task': prep['adapted_task'],
            'run_directory': str((ART / f'{key}-run-001').relative_to(ROOT)),
            'current_replay': str((ART / f'{key}-run-001').relative_to(ROOT)),
            'mechanism': {'adapter': 'F', 'submechanism': 'F3',
                          'evidence': prep['mechanism_evidence']},
            'primary_adapter': 'F',
            'submechanism': 'F3',
            'canonical_repo': canonical,
            'source_train_revision': '77cab9055d42ab4a5c25c89a8f937096db13558e',
            'source_preparation': f'competition/vfarl/replay/f3r-{key}-source-preparation.json',
            'recipe': prep['recipe'],
            'run_present': (run / 'run-summary.json').exists(),
        }
        if prep.get('visible_prompt_profile'):
            report['visible_prompt_profile'] = prep['visible_prompt_profile']
        (REPLAY / f'f3r-{key}-replay-result.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'f3-replace-normalize-catalog.json').write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + '\n')
    print(f'catalog tasks={len(catalog["tasks"])}', flush=True)


if __name__ == '__main__':
    main()
