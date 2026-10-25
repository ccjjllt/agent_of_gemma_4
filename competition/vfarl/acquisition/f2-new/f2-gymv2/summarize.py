"""F2 round 3: summarize normalize verdicts across f2-gymv2 dual replays."""
import glob
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-gymv2'


def main():
    recs = []
    for p in sorted(glob.glob(str(ACQ / 'f2-gymv2-*-normalize.json'))):
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
    summary = {
        'date': '2026-10-25',
        'scope': 'F600-F2 f2-gymv2 (Gym + Rebench-V2 round 3)',
        'reviewed_count': len(reviewed),
        'dual_completed': len(recs),
        'eligible': len(elig),
        'eligible_instance_ids': [r['instance_id'] for r in elig],
        'gap_to_40': 40 - len(elig),
        'records': recs,
        'pending_reviewed_without_dual': sorted(reviewed - {r['instance_id'] for r in recs}),
        'note': ('normalize eligible only; supplemental=f2-gymv2/catalog.json; '
                 'no test/matcher/requirement edits; six-tool course; protected paths untouched'),
    }
    (ACQ / 'normalize-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
