"""Stage 5b: run normalize_results.normalize over f3-replace reports.

Usage: python3 normalize_f3replace.py
Reads: f3-replace-normalize-catalog.json, replay/f3r-*-replay-result.json,
       replay/approved-source-requirement-mappings.json
Writes: f3-replace/normalize-results.json, f3-replace/admit.json
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f3-new'
OUT = ACQ / 'f3-replace'
REPLAY = ROOT / 'competition/vfarl/replay'


def main():
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else set()
    catalog_blob = json.loads((OUT / 'f3-replace-normalize-catalog.json').read_text())
    catalog = {row['instance_id']: row for row in catalog_blob['tasks']}
    approval_blob = json.loads((REPLAY / 'approved-source-requirement-mappings.json').read_text())
    approvals = {row['instance_id']: row for row in approval_blob.get('approvals', [])}
    spec = importlib.util.spec_from_file_location(
        'f3replace_normalizer', REPLAY / 'normalize_results.py')
    normalizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(normalizer)
    results, eligible = [], []
    for path in sorted(REPLAY.glob('f3r-*-replay-result.json')):
        report = json.loads(path.read_text())
        iid = report.get('instance_id', '')
        if only and iid not in only and path.stem not in only:
            continue
        try:
            res = normalizer.normalize(path, catalog, approvals)
        except Exception as error:
            res = {'instance_id': iid, 'eligible_for_human_off_policy_SFT': False,
                   'ineligibility_reasons': ['normalizer_exception'],
                   'normalization_error': f'{type(error).__name__}: {error}'}
        results.append({'instance_id': iid, 'report': str(path.relative_to(ROOT)),
                        'eligible': res.get('eligible_for_human_off_policy_SFT', False),
                        'reasons': res.get('ineligibility_reasons', []),
                        'complete': res.get('complete_official_replay', False),
                        'baseline_f2p': res.get('baseline_verified_original_F2P_failure', False),
                        'resolved': res.get('resolved_by_local_official_verify_task'),
                        'strict_ids': res.get('strict_verbatim_required_ids_passed', False),
                        'visible_profile': (res.get('visible_prompt_profile') or {}).get('kind'),
                        'error': res.get('normalization_error')})
        if res.get('eligible_for_human_off_policy_SFT'):
            eligible.append(iid)
        print(f"{iid}: eligible={res.get('eligible_for_human_off_policy_SFT')} "
              f"reasons={res.get('ineligibility_reasons')}", flush=True)
    (OUT / 'normalize-results.json').write_text(
        json.dumps({'results': results, 'eligible': eligible,
                    'eligible_count': len(eligible)}, ensure_ascii=False, indent=2) + '\n')
    print(f'\nELIGIBLE {len(eligible)}/{len(results)}', flush=True)


if __name__ == '__main__':
    main()
