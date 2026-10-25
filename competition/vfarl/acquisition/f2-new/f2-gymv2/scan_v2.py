"""F2 round 3: scan SWE-rebench-V2 train Python subset (local pinned parquet)."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT / 'competition/vfarl'
ACQ = VFARL / 'acquisition/f2-new'
OUT = ACQ / 'f2-gymv2'
V2_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet')
V2_REV = '10483de0f50fe5da545942705a76c6150171af7f'

COPY_SIG = re.compile(r'__copy__|__deepcopy__|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|def copy\(|(?<![\w.])\.copy\(\)')
EXCLUDED_REPOS = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}

sys.path.insert(0, str(OUT))
from scan_gym import load_forbidden, snap_hit, save  # noqa: E402


def cmd_scan():
    import pyarrow.parquet as pq
    forb_i, forb_s, _, _ = load_forbidden()
    f = pq.ParquetFile(str(V2_PARQUET))
    names = f.schema.names
    out = []
    n_py = 0
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=['instance_id', 'language', 'repo', 'base_commit',
                                          'patch', 'problem_statement',
                                          'FAIL_TO_PASS', 'PASS_TO_PASS'])
        d = t.to_pylist()
        for r in d:
            if (r.get('language') or '').lower() != 'python':
                continue
            n_py += 1
            iid = r['instance_id']
            if iid in forb_i:
                continue
            repo = (r.get('repo') or '').strip()
            if repo.lower() in EXCLUDED_REPOS:
                continue
            patch = r.get('patch') or ''
            chg = [l for l in patch.splitlines()
                   if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++', '---'))]
            hits = [l.strip()[:200] for l in chg if COPY_SIG.search(l)]
            if not hits:
                continue
            f2p = r.get('FAIL_TO_PASS') or []
            p2p = r.get('PASS_TO_PASS') or []
            if not f2p or set(f2p) & set(p2p):
                continue
            if not (r.get('problem_statement') or '').strip():
                continue
            files = re.findall(r'diff --git a/(\S+)', patch)
            if any('..' in fl for fl in files):
                continue
            total = len(f2p) + len(p2p)
            if total > 150:
                continue
            snap = snap_hit(repo, r.get('base_commit') or '', forb_s)
            out.append({'instance_id': iid, 'repo': repo, 'base_commit': r.get('base_commit'),
                        'row_group': rg, 'F2P': len(f2p), 'P2P': len(p2p), 'total': total,
                        'files': files[:6], 'F2P_list': f2p, 'P2P_sample': p2p[:4],
                        'hits': hits[:6], 'snapshot_collision': snap})
    out.sort(key=lambda d: (d['total'], d['F2P']))
    save(OUT / 'v2-strict-scan.json', {
        'date': '2026-10-25', 'source': 'nebius/SWE-rebench-V2', 'revision': V2_REV,
        'split': 'train', 'language': 'python', 'python_rows_scanned': n_py,
        'parquet_bytes': V2_PARQUET.stat().st_size,
        'parquet_sha256': hashlib.sha256(V2_PARQUET.read_bytes()).hexdigest(),
        'suite_cap': 150, 'hits': out})
    print(f'v2 python rows={n_py} strict_hits={len(out)}')
    for d in out:
        flag = 'SNAPCOLLIDE ' if d['snapshot_collision'] else ''
        print(f"{flag}{d['total']:4d} F2P={d['F2P']:3d} P2P={d['P2P']:4d} {d['instance_id']} {d['repo']} {d['files'][:2]}")
        for h in d['hits'][:2]:
            print('    ', h)


def cmd_show():
    import pyarrow.parquet as pq
    scan = json.loads((OUT / 'v2-strict-scan.json').read_text())
    by_rg = {}
    for d in scan['hits']:
        by_rg.setdefault(d['row_group'], []).append(d)
    f = pq.ParquetFile(str(V2_PARQUET))
    want = sys.argv[2:] if len(sys.argv) > 2 else None
    for rg, ds in sorted(by_rg.items()):
        t = f.read_row_group(rg)
        rows = {r['instance_id']: r for r in t.to_pylist()}
        for d in ds:
            if want and d['instance_id'] not in want:
                continue
            r = rows[d['instance_id']]
            print('=' * 100)
            print(d['instance_id'], d['repo'], (d['base_commit'] or '')[:12],
                  'F2P=', d['F2P'], 'P2P=', d['P2P'], 'files=', d['files'])
            if d['snapshot_collision']:
                print('SNAPSHOT COLLISION:', d['snapshot_collision'])
            print('F2P:', json.dumps(d['F2P_list'], ensure_ascii=False)[:1200])
            print('--- problem (first 1800 chars) ---')
            print((r.get('problem_statement') or '')[:1800])
            print('--- patch FULL ---')
            print((r.get('patch') or '')[:6000])


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    OUT.mkdir(parents=True, exist_ok=True)
    if which == 'scan':
        cmd_scan()
    elif which == 'show':
        cmd_show()
    else:
        print('unknown cmd')
