"""F2 f2-recall: summarize normalize verdicts across dual replays."""
import glob
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-recall'


def main():
    recs = []
    for p in sorted(glob.glob(str(ACQ / 'f2-recall-*-normalize.json'))):
        d = json.loads(Path(p).read_text())
        recs.append({
            'instance_id': d['instance_id'],
            'eligible': d['eligible_for_human_off_policy_SFT'],
            'reasons': d['ineligibility_reasons'],
            'complete_replay': d['complete_official_replay'],
            'baseline_F2P_fail': d['baseline_verified_original_F2P_failure'],
            'file': Path(p).name,
        })
    reviewed = {t['instance_id'] for t in json.loads((ACQ / 'reviewed-candidates.json').read_text())['tasks']}
    elig = [r for r in recs if r['eligible']]
    near = [r for r in recs if (not r['eligible']) and r['complete_replay'] and r['baseline_F2P_fail']]
    summary = {
        'date': '2026-10-25',
        'scope': 'F600-F2 f2-recall (Gym + Rebench-V2 recall-large suite<=200)',
        'reviewed_count': len(reviewed),
        'dual_completed': len(recs),
        'eligible': len(elig),
        'eligible_instance_ids': [r['instance_id'] for r in elig],
        'near_miss_instance_ids': [r['instance_id'] for r in near],
        'gap_to_40': 30,
        'global_eligible': 10,
        'records': recs,
        'pending_reviewed_without_dual': sorted(reviewed - {r['instance_id'] for r in recs}),
        'blocked_triple_keep_without_dual': ['hdmf-dev__hdmf-848'],
        'note': ('normalize eligible only; global 10/40 unchanged (6+2+2); f2-recall 0 eligible / 2 near-miss '
                 '(nox-439 P2P missing-dist-metadata env; aiohttp-10672 extra flaky warn_large_cookie, all 152 P2P pass); '
                 'hdmf-848 triple-KEEP but skip-DROP blocked; supplemental=f2-recall/catalog.json; '
                 'no test/matcher/requirement edits; six-tool course; protected paths untouched'),
    }
    (ACQ / 'normalize-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
