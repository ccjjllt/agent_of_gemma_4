"""F2 f2-push5: dump problem+patch+testpatch for bounded ranked review set.

Review set (bounded; remainder is signal-level SCREENED, not verdict):
  A all (gym 151-200 weak+support) +
  B all (v2 151-200 weak test-is-only) +
  C top30 (P2P-empty<=50 weak: fn/own/tis desc, total asc) +
  D top25 (weak-nosupport<=150: total asc, nfiles asc).
"""
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
OUT = ROOT / 'competition/vfarl/acquisition/f2-new/f2-push5'
GYM_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/gym-train.parquet')
V2_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet')

import pyarrow.parquet as pq

cands = json.loads((OUT / 'push5-candidates.json').read_text())['candidates']
A = [c for c in cands if 'A' in c['pools']]
B = [c for c in cands if 'B' in c['pools']]
C = sorted([c for c in cands if 'C' in c['pools'] and 'A' not in c['pools'] and 'B' not in c['pools']],
           key=lambda d: (-d['f2p_name_hits'], -d['own_problem_n'], -d['test_is_n'], d['total'], d['n_files']))
D = sorted([c for c in cands if 'D' in c['pools'] and 'A' not in c['pools'] and 'B' not in c['pools'] and 'C' not in c['pools']],
           key=lambda d: (d['total'], d['n_files']))
review = A + B + C[:30] + D[:25]
print(f'A={len(A)} B={len(B)} C={len([c for c in cands if "C" in c["pools"]])} C-reviewed={min(30, len(C))} D-only={len(D)} D-reviewed={min(25, len(D))} review={len(review)} screened={len(cands)-len(review)}')

gym_rows = {r['instance_id']: r for r in pq.read_table(str(GYM_PARQUET)).to_pylist()}
f = pq.ParquetFile(str(V2_PARQUET))
vrows = {}
for rg in range(f.num_row_groups):
    for r in f.read_row_group(rg).to_pylist():
        vrows[r['instance_id']] = r

def get_row(c):
    return gym_rows[c['instance_id']] if c['pool'] == 'gym' else vrows[c['instance_id']]

dump = []
tri = []
for c in review:
    r = get_row(c)
    dump.append('=' * 100)
    dump.append(f"{c['pool']} {c['instance_id']} pools={''.join(sorted(c['pools']))} repo={c['repo']} base={c['base_commit']}")
    dump.append(f"F2P({c['F2P']}): {r.get('FAIL_TO_PASS')}")
    p2p = r.get('PASS_TO_PASS') or []
    dump.append(f"P2P({len(p2p)}): {p2p[:12]}")
    dump.append(f"weak({c['weak_n']}): {c['weak_hits']}")
    dump.append(f"own_prob({c['own_problem_n']}): {c['own_problem']} || own_patch: {c['own_patch']}")
    dump.append(f"test_is({c['test_is_n']}): {c['test_is_lines']}")
    dump.append('---PROBLEM---')
    dump.append((r.get('problem_statement') or '')[:2500])
    dump.append('---PATCH---')
    dump.append((r.get('patch') or '')[:6000])
    dump.append('---TEST_PATCH(is/copy/clone lines)---')
    n = 0
    for l in (r.get('test_patch') or '').splitlines():
        ll = l.lower()
        if 'copy' in ll or 'clone' in ll or ' is ' in l or 'assertIs' in l:
            dump.append('  ' + l.strip()[:200]); n += 1
            if n > 25:
                break
    tri.append(f"t{c['total']:3d} F{c['F2P']:3d} P{c['P2P']:3d} pools={''.join(sorted(c['pools']))} w{c['weak_n']} fn{c['f2p_name_hits']} ownP{c['own_problem_n']} ownPa{c['own_patch_n']} tis{c['test_is_n']} nf{c['n_files']} {c['pool']:4s} {c['instance_id']} :: {(c['weak_hits']+c['inline_hits'])[0][:80] if (c['weak_hits']+c['inline_hits']) else ''} || F2P:{str(r.get('FAIL_TO_PASS'))[:150]}")

(OUT / 'push5-review-dump.txt').write_text('\n'.join(dump) + '\n')
(OUT / 'push5-triage.txt').write_text('\n'.join(tri) + '\n')
(OUT / 'push5-review-set.json').write_text(json.dumps(
    {'count': len(review), 'screened_count': len(cands) - len(review),
     'review_ids': [c['instance_id'] for c in review]}, indent=1) + '\n')
print('\n'.join(tri))
