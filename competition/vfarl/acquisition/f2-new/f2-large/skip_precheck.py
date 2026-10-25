"""F2 f2-large: skip-prefix-collision precheck for all 35 large-suite candidates.

For each candidate: fetch upstream test files at base_commit, find skip/xfail
markers, check official-matcher prefix collisions vs all F2P+P2P (full lists
from pinned parquets). Any collision -> DROP. Fetch errors -> FETCH_ERROR
(blocks file-scope validation; not counted as clean).
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT / 'competition/vfarl'
OUT = VFARL / 'acquisition/f2-new/f2-large'
GYM_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/gym-train.parquet')
V2_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet')

MARKER = re.compile(r'@pytest\.mark\.skip|@pytest\.mark\.xfail|@pytest\.mark\.skipif|'
                    r'@unittest\.skip|pytest\.skip\s*\(|@pytest\.fixture.*skip', re.M)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def fetch_raw(repo, commit, filepath, retries=4):
    url = f'https://raw.githubusercontent.com/{repo}/{commit}/{filepath}'
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'vfarl-f2-research'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(3 * (a + 1))
    raise RuntimeError(f'fetch failed {url}: {last!r}')


def req_base(node):
    return node.split('::')[-1].split('[')[0]


def collides(marked_short, requirement):
    rb = req_base(requirement)
    if marked_short == rb:
        return True
    if marked_short.startswith(rb + '_') or marked_short.startswith(rb + '['):
        return True
    if len(rb) >= 68 and marked_short.startswith(rb):
        return True
    m = re.match(r'^(.*?)(\d+)$', marked_short)
    if m and m.group(1).rstrip('_') == rb:
        return True
    return False


def test_file_of(node):
    return node.split('::')[0]


def load_full_rows():
    import pyarrow.parquet as pq
    gym = {r['instance_id']: ('gym', r) for r in pq.read_table(str(GYM_PARQUET)).to_pylist()}
    v2 = {}
    f = pq.ParquetFile(str(V2_PARQUET))
    for rg in range(f.num_row_groups):
        for r in f.read_row_group(rg).to_pylist():
            if (r.get('language') or '').lower() != 'python':
                continue
            v2[r['instance_id']] = ('v2', r)
    return gym, v2


def run():
    gym_rows, v2_rows = load_full_rows()
    gym_scan = json.loads((OUT / 'gym-large-scan.json').read_text())['hits']
    v2_scan = json.loads((OUT / 'v2-large-scan.json').read_text())['hits']
    cands = gym_scan + v2_scan
    cands.sort(key=lambda d: (d['total'], d['F2P']))
    results = []
    for k in cands:
        iid = k['instance_id']
        pool = k['pool']
        src = gym_rows if pool == 'gym' else v2_rows
        assert iid in src, iid
        _, r = src[iid]
        repo = (r.get('repo') or '').strip()
        commit = r.get('base_commit') or ''
        f2p = r.get('FAIL_TO_PASS') or []
        p2p = r.get('PASS_TO_PASS') or []
        nodes = list(f2p) + list(p2p)
        files = sorted({test_file_of(n) for n in nodes if '::' in n})
        fetched = {}
        markers = {}
        errors = {}
        for fp in files:
            try:
                raw = fetch_raw(repo, commit, fp)
                fetched[fp] = len(raw)
                text = raw.decode('utf-8', errors='replace')
            except Exception as e:
                errors[fp] = str(e)[:200]
                continue
            found = []
            cur_test = None
            for i, line in enumerate(text.splitlines()):
                mt = re.match(r'\s*(?:async\s+)?def\s+(test\w*)\s*\(', line)
                if mt:
                    cur_test = mt.group(1)
                mc = re.match(r'\s*class\s+(\w+)', line)
                if mc:
                    cur_test = None
                if MARKER.search(line):
                    look = '\n'.join(text.splitlines()[i:i + 6])
                    md = re.search(r'def\s+(test\w*)\s*\(', look)
                    found.append({'line': i + 1, 'marker': MARKER.search(line).group(0)[:40],
                                  'nearest_test': md.group(1) if md else cur_test})
            markers[fp] = found
        collisions = []
        for fp, found in markers.items():
            for mk in found:
                nt = mk.get('nearest_test')
                if not nt:
                    continue
                for req in nodes:
                    if collides(nt, req):
                        collisions.append({'file': fp, 'marker_line': mk['line'],
                                           'marked_test': nt, 'requirement': req})
        verdict = 'CLEAN' if not collisions and not errors else ('DROP' if collisions else 'FETCH_ERROR')
        results.append({'instance_id': iid, 'pool': pool, 'repo': repo,
                        'short_rev': commit[:8] if commit else None,
                        'base_commit': commit,
                        'F2P_count': len(f2p), 'P2P_count': len(p2p), 'total': len(f2p) + len(p2p),
                        'required_files': sorted(files), 'required_file_count': len(files),
                        'single_file_required_closure': len(files) == 1,
                        'upstream_files_fetched': sorted(fetched.keys()),
                        'upstream_file_count': len(fetched),
                        'bytes': [fetched[k2] for k2 in sorted(fetched.keys())],
                        'markers_found': markers, 'fetch_errors': errors,
                        'prefix_collisions': collisions, 'verdict': verdict})
        print(f"{verdict:12s} {iid} total={len(f2p)+len(p2p)} files={len(fetched)}/{len(files)} markers={sum(len(v) for v in markers.values())} collisions={len(collisions)}", flush=True)
    save(OUT / 'skip-precheck.json', {
        'date': '2026-10-25',
        'matcher': 'swegemma 0.2.11 harness.verification._node_matches_requirement (per round2): exact/short/base, startswith(req+_), startswith(req+[), len>=68 prefix, digit-suffix',
        'scope': 'all 35 large-suite strict-copy candidates (Gym 7 + V2 28), full F2P+P2P node lists from pinned parquets',
        'snapshots': results})
    drops = [r for r in results if r['verdict'] == 'DROP']
    cleans = [r for r in results if r['verdict'] == 'CLEAN']
    errs = [r for r in results if r['verdict'] == 'FETCH_ERROR']
    print(f'checked={len(results)} clean={len(cleans)} drop={len(drops)} errors={len(errs)}')


if __name__ == '__main__':
    run()
