"""F2 f2-push3: build keep-list/keep-families for 2 triple-KEEPs from parquet rows."""
import json
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
OUT = ROOT / 'competition/vfarl/acquisition/f2-new/f2-push3'
GYM_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/gym-train.parquet')
V2_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet')

import pyarrow.parquet as pq

KEEPS = [
    {'instance_id': 'aeye-lab__pymovements-486', 'pool': 'v2',
     'repo': 'aeye-lab/pymovements',
     'family': 'aeye-lab/pymovements::pymovements.dataset.dataset_files.save_preprocessed::gaze_copy_isolation'},
    {'instance_id': 'jupyter__kernel_gateway-118', 'pool': 'v2',
     'repo': 'jupyter/kernel_gateway',
     'family': 'jupyter/kernel_gateway::kernel_gateway.services.activity.manager.ActivityManager.get_map_for_kernel::per_kernel_copy_isolation'},
]

f = pq.ParquetFile(str(V2_PARQUET))
vrows = {}
for rg in range(f.num_row_groups):
    for r in f.read_row_group(rg).to_pylist():
        vrows[r['instance_id']] = r

keeps = []
fams = {}
for k in KEEPS:
    r = vrows[k['instance_id']]
    assert r['repo'] == k['repo'], (k['instance_id'], r['repo'])
    keeps.append({
        'instance_id': k['instance_id'], 'pool': k['pool'], 'repo': k['repo'],
        'base_commit': r['base_commit'],
        'F2P_list': r['FAIL_TO_PASS'], 'F2P_count': len(r['FAIL_TO_PASS']),
        'P2P_list': r['PASS_TO_PASS'], 'P2P_count': len(r['PASS_TO_PASS']),
    })
    fams[k['instance_id']] = k['family']

(OUT / 'keep-list.json').write_text(json.dumps(keeps, ensure_ascii=False, indent=1) + '\n')
(OUT / 'keep-families.json').write_text(json.dumps(fams, ensure_ascii=False, indent=1) + '\n')
for k in keeps:
    print(k['instance_id'], k['base_commit'][:12], 'F2P=', k['F2P_count'], 'P2P=', k['P2P_count'])
