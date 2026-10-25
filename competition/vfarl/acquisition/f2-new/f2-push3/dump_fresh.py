"""F2 f2-push3: dump problem+patch+testpatch for fresh (never-push2-verdict) rows."""
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
OUT = ROOT / 'competition/vfarl/acquisition/f2-new/f2-push3'
GYM_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/gym-train.parquet')
V2_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet')

# Every instance push2 gave a verdict on (KEEP/COLLIDE/REJECT/HELD/deep-read抽查).
PUSH2_VERDICT = {
 'aio-libs__aiohttp-10534', 'aio-libs__aiohttp-10551', 'pvlib__pvlib-python-2292',
 'pymc-devs__pymc-7637', 'modin-project__modin-6759', 'copier-org__copier-2189',
 'pandas-dev__pandas-53977', 'stfc__psyclone-1850', 'qiskit__qiskit-terra-8741',
 'modin-project__modin-6758', 'tdameritrade__stumpy-655', 'tdameritrade__stumpy-656',
 'pandas-dev__pandas-51005', 'stfc__psyclone-2685', 'pyccel__pyccel-1606',
 'laspy__laspy-342', 'jlowin__fastmcp-434', 'aeon-toolkit__aeon-1609',
 'pydantic__pydantic-7825', 'more-itertools__more-itertools-412',
 'iamconsortium__pyam-236', 'dask__dask-8930', 'networkx__networkx-5616',
 'dask__dask-7033', 'qiskit-community__ffsim-295', 'pandas-dev__pandas-57084',
 'pandas-dev__pandas-57078', 'pandas-dev__pandas-57046', 'cuthbertlab__music21-866',
 'qiskit__qiskit-terra-8764', 'conan-io__conan-13390',
 'aws-cloudformation__cfn-python-lint-1409', 'cta-observatory__ctapipe-1893',
 'python-trio__trio-2392', 'bids-standard__pybids-574', 'sktime__sktime-5570',
 'ericmjl__pyjanitor-735', 'pyjanitor-devs__pyjanitor-1317',
 'pyjanitor-devs__pyjanitor-1315', 'streamlink__streamlink-2654',
 'streamlink__streamlink-2653', 'keras-team__keras-nlp-64', 'benchopt__benchopt-361',
 'codezonediitj__pydatastructs-85', 'awslabs__cfn-python-lint-426',
}

import pyarrow.parquet as pq

cands = json.loads((OUT / 'push3-candidates.json').read_text())
allc = cands['candidates_A'] + cands['candidates_B']
fresh = [c for c in allc if c['instance_id'] not in PUSH2_VERDICT]
print(f'total={len(allc)} fresh={len(fresh)} excluded={len(allc)-len(fresh)}')

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
for c in fresh:
    r = get_row(c)
    dump.append('=' * 100)
    dump.append(f"{c['pool']} {c['instance_id']} repo={c['repo']} base={c['base_commit']}")
    dump.append(f"F2P({c['F2P']}): {r.get('FAIL_TO_PASS')}")
    p2p = r.get('PASS_TO_PASS') or []
    dump.append(f"P2P({len(p2p)}): {p2p[:12]}")
    dump.append('---PROBLEM---')
    dump.append((r.get('problem_statement') or '')[:2500])
    dump.append('---PATCH---')
    dump.append((r.get('patch') or '')[:6000])
    dump.append('---TEST_PATCH(is/copy lines)---')
    n = 0
    for l in (r.get('test_patch') or '').splitlines():
        if 'copy' in l.lower() or ' is ' in l or 'assertIs' in l:
            dump.append('  ' + l.strip()[:200]); n += 1
            if n > 25:
                break
    tri.append(f"t{c['total']:3d} F{c['F2P']:3d} P{c['P2P']:3d} ownP{c['own_problem_n']} s{c['strong_n']} tis{c['test_is_n']} fn{c['f2p_name_hits']} nf{c['n_files']} {c['pool']:4s} {c['instance_id']} :: {(c['strong_hits'][0][:80] if c['strong_hits'] else '')} || F2P:{str(r.get('FAIL_TO_PASS'))[:160]}")

(OUT / 'push3-review-dump.txt').write_text('\n'.join(dump) + '\n')
(OUT / 'push3-triage.txt').write_text('\n'.join(tri) + '\n')
print('\n'.join(tri))
