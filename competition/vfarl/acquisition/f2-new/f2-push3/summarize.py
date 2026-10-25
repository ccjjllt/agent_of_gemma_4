"""Summarize f2-push3 normalize outcomes (supplemental = f2-push3 catalog)."""
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
OUT = ROOT / 'competition/vfarl/acquisition/f2-new/f2-push3'

keeps = json.loads((OUT / 'keep-list.json').read_text())
records = []
for k in keeps:
    iid = k['instance_id']
    key = 'f2-push3-' + iid.replace('__', '-').replace('_', '-')
    # keys used: f2-push3-psyclone-3029, f2-push3-qiskit-terra-5778
    cand = list(OUT.glob(f'*-normalize.json'))
    rec = None
    for c in cand:
        d = json.loads(c.read_text())
        if d.get('instance_id') == iid:
            rec = {'instance_id': iid, 'eligible': d['eligible_for_human_off_policy_SFT'],
                   'reasons': d['ineligibility_reasons'],
                   'complete_replay': d['complete_official_replay'],
                   'baseline_F2P_fail': d['baseline_verified_original_F2P_failure'],
                   'file': c.name}
            break
    if rec is None:
        rec = {'instance_id': iid, 'eligible': False, 'reasons': ['dual_pending'],
               'complete_replay': False, 'baseline_F2P_fail': False, 'file': None}
    records.append(rec)

eligible = [r['instance_id'] for r in records if r['eligible']]
near_miss = [r['instance_id'] for r in records
             if not r['eligible'] and r['complete_replay'] and r['reasons'] != ['dual_pending']]
pending = [r['instance_id'] for r in records if not r['eligible'] and not r['complete_replay']]
blocked = [r['instance_id'] for r in records
           if not r['eligible'] and not r['complete_replay'] and r['reasons'] != ['dual_pending']]
summary = {'date': '2026-10-25',
           'scope': 'F600-F2 f2-push3 (Gym + Rebench-V2 mid-suite 151-400 + P2P-empty<=50 inline-is (push2 remainder))',
           'reviewed_count': len(keeps), 'dual_completed': sum(1 for r in records if r['complete_replay']),
           'eligible': len(eligible), 'eligible_instance_ids': eligible,
           'near_miss_instance_ids': near_miss,
           'gap_to_40': 22 - len(eligible), 'global_eligible': 18 + len(eligible),
           'records': records,
           'pending_reviewed_without_dual': pending,
           'blocked_triple_keep_without_dual': blocked,
           'note': ('normalize eligible only; global 16/40 + f2-push3 eligible; '
                    'supplemental=f2-push3/catalog.json; no test/matcher/requirement edits; '
                    'six-tool course; protected paths untouched; Smith frozen')}
(OUT / 'normalize-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
