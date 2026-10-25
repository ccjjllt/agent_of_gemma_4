"""F2 f2-more: fetch per-keep source rows at pinned revisions (read-only old dirs)."""
import hashlib
import json
import urllib.request
import urllib.parse
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
OUT = ROOT / 'competition/vfarl/acquisition/f2-new/f2-more'

POOLS = {
    'v2': ('nebius/SWE-rebench-V2', '10483de0f50fe5da545942705a76c6150171af7f'),
}


def fetch_row(dataset, rev, offset):
    url = ('https://datasets-server.huggingface.co/rows?dataset='
           + urllib.parse.quote(dataset, safe='') + '&config=default&split=train'
           + '&revision=' + rev + f'&offset={offset}&length=1')
    req = urllib.request.Request(url, headers={'User-Agent': 'vfarl-f2-research'})
    with urllib.request.urlopen(req, timeout=120) as r:
        body = r.read()
        headers = dict(r.headers)
    return body, headers, url


def main():
    import pyarrow.parquet as pq
    keeps = json.loads((OUT / 'keep-list.json').read_text())
    f = pq.ParquetFile('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet')
    sizes = [f.read_row_group(rg, columns=['instance_id']).num_rows for rg in range(f.num_row_groups)]
    off = [0]
    for s in sizes:
        off.append(off[-1] + s)
    vidx = {}
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=['instance_id'])
        for j, iid in enumerate(t.column('instance_id').to_pylist()):
            vidx[iid] = off[rg] + j
    for k in keeps:
        pool = k['pool']
        iid = k['instance_id']
        dataset, rev = POOLS[pool]
        idx = vidx[iid]
        subdir = OUT / 'pages-v2'
        subdir.mkdir(parents=True, exist_ok=True)
        if (subdir / f'{iid}.json').exists():
            print(f'skip {pool} row {idx} {iid} (exists)')
            continue
        body, headers, url = fetch_row(dataset, rev, idx)
        doc = json.loads(body)
        rows = doc.get('rows', [])
        assert len(rows) == 1 and rows[0]['row']['instance_id'] == iid, (iid, idx, body[:200])
        row = rows[0]['row']
        assert row['base_commit'] == k['base_commit'], (iid, row['base_commit'], k['base_commit'])
        assert sorted(row['FAIL_TO_PASS']) == sorted(k['F2P_list']), iid
        assert len(row['PASS_TO_PASS']) == k['P2P_count'], (iid, len(row['PASS_TO_PASS']))
        (subdir / f'{iid}.json').write_text(json.dumps({'row': row, 'row_idx': idx}, ensure_ascii=False, indent=2) + '\n')
        (subdir / f'{iid}.receipt.json').write_text(json.dumps(
            {'url': url, 'dataset': dataset, 'revision': rev, 'offset': idx,
             'body_sha256': hashlib.sha256(body).hexdigest(),
             'response_headers': {kk: headers[kk] for kk in ('ETag', 'Date', 'Content-Type') if kk in headers}},
            ensure_ascii=False, indent=2) + '\n')
        print(f'fetched {pool} row {idx} {iid}')


if __name__ == '__main__':
    main()
