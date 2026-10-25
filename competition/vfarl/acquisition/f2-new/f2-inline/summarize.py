"""F2 inline: summarize normalize verdicts (pending heavy-dep, no dual)."""
import glob, json
from pathlib import Path
ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new/f2-inline'
def main():
    recs = []
    for p in sorted(glob.glob(str(ACQ / 'f2-inline-*-normalize.json'))):
        d = json.loads(Path(p).read_text())
        recs.append({'instance_id': d['instance_id'], 'eligible': d['eligible_for_human_off_policy_SFT'], 'reasons': d['ineligibility_reasons'], 'complete_replay': d['complete_official_replay'], 'baseline_F2P_fail': d['baseline_verified_original_F2P_failure'], 'file': Path(p).name})
    reviewed = {t['instance_id'] for t in json.loads((ACQ / 'reviewed-candidates.json').read_text())['tasks']}
    elig = [r for r in recs if r['eligible']]
    base_global = 10  # f2-gymv2 6 + f2-gymv2b 2 + f2-gymv2c 2 (other batches, read-only)
    glob_elig = base_global + len(elig)
    summary = {'date': '2026-10-25', 'scope': 'F600-F2 f2-inline (Gym + Rebench-V2 inline-is rescreen, suite<=200)', 'reviewed_count': len(reviewed), 'dual_completed': len(recs), 'eligible': len(elig), 'eligible_instance_ids': [r['instance_id'] for r in elig], 'gap_to_40': 40 - glob_elig, 'global_eligible': glob_elig, 'records': recs, 'pending_reviewed_without_dual': sorted(reviewed - {r['instance_id'] for r in recs}), 'blocked_triple_keep_without_dual': [], 'note': 'normalize eligible only; global 11/40 (6+2+2+1 f2-inline qiskit-12620, task-python 3.10 + qiskit-terra 0.46.2 prebuilt _accelerate dual); supplemental=f2-inline/catalog.json; no test/matcher/requirement edits; six-tool course; protected paths untouched'}
    (ACQ / 'normalize-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=1))
if __name__ == '__main__':
    main()
