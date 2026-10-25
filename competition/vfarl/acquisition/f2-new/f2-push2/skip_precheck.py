"""F2 f2-push: skip-prefix-collision precheck for KEEP candidates."""
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT / 'competition/vfarl'
OUT = VFARL / 'acquisition/f2-new/f2-push2'

MARKER = re.compile(r'@pytest\.mark\.skip|@pytest\.mark\.xfail|@pytest\.mark\.skipif|'
                    r'@unittest\.skip|pytest\.skip\s*\(|@pytest\.fixture.*skip', re.M)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def fetch_raw(repo, commit, filepath):
    url = f'https://raw.githubusercontent.com/{repo}/{commit}/{filepath}'
    req = urllib.request.Request(url, headers={'User-Agent': 'vfarl-f2-research'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


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


def run(keeps):
    results = []
    for k in keeps:
        repo = k['repo']
        commit = k['base_commit']
        nodes = list(k['F2P_list']) + list(k.get('P2P_list', []))
        files = sorted({test_file_of(n) for n in nodes})
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
        results.append({'instance_id': k['instance_id'], 'repo': repo,
                        'short_rev': commit[:8] if commit else None,
                        'F2P_count': len(k['F2P_list']), 'P2P_count': len(k.get('P2P_list', [])),
                        'upstream_files_fetched': list(fetched.keys()), 'bytes': list(fetched.values()),
                        'markers_found': markers, 'fetch_errors': errors,
                        'prefix_collisions': collisions, 'verdict': verdict})
        print(f"{verdict:12s} {k['instance_id']} files={len(fetched)} markers={sum(len(v) for v in markers.values())} collisions={len(collisions)}")
    save(OUT / 'skip-precheck.json', {
        'date': '2026-10-25',
        'matcher': 'swegemma 0.2.11 harness.verification._node_matches_requirement (per round2): exact/short/base, startswith(req+_), startswith(req+[), len>=68 prefix, digit-suffix',
        'snapshots': results})
    drops = [r for r in results if r['verdict'] == 'DROP']
    print(f'checked={len(results)} clean={sum(1 for r in results if r["verdict"]=="CLEAN")} drop={len(drops)} errors={sum(1 for r in results if r["verdict"]=="FETCH_ERROR")}')


KEEPS_FILE = OUT / 'keep-list.json'

if __name__ == '__main__':
    keeps = json.loads(KEEPS_FILE.read_text())
    run(keeps)
