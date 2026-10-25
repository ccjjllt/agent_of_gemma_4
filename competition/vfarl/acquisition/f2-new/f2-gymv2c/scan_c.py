"""F2 round 5 (f2-gymv2c): continue to 40 (have 8 eligible, need 32 more).

Pools (train only, pinned):
  SWE-Gym/SWE-Gym            rev bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb (2438 train)
  nebius/SWE-rebench-V2      rev 10483de0f50fe5da545942705a76c6150171af7f (Python subset 7243)

Constraints (unchanged):
  zero family/snapshot/instance overlap with common-400 + F1-40 + F5-old40
  + f2-round1/2 + f2-gymv2 16 keeps + f2-gymv2b 5 keeps (read-only, never
  rewrite old dirs); four repos excluded; F2 triple mandatory (is +
  shared-mutation + timepoint); only-== demotes to S; capability-only to R;
  bad paths filtered; <=150 allowed; skip-prefix-collision precheck
  mandatory; whitelisted problem_statement; JUnit dual conditions; six
  tools; protected paths untouched. replay/.venv dual rounds.

Usage:
  python3 scan_c.py dedup   # build forbidden sets (read-only old files + gymv2/b keeps)
  python3 scan_c.py scan    # strict changed-line copy scan (same signal as rounds 3-4)
  python3 scan_c.py recall  # expanded recall: clone-signal + ownership-word + test-side-is
  python3 scan_c.py show    # dump one row for manual triple review: show <pool> <instance_id>
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT / 'competition/vfarl'
ACQ = VFARL / 'acquisition/f2-new'
OUT = ACQ / 'f2-gymv2c'
GYM_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/gym-train.parquet')
V2_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet')
GYM_REV = 'bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb'
V2_REV = '10483de0f50fe5da545942705a76c6150171af7f'

COPY_SIG = re.compile(r'__copy__|__deepcopy__|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|def copy\(|(?<![\w.])\.copy\(\)')
CLONE_SIG = re.compile(r'\bclone\b|\.clone\(|clone\(|copyInto|copy_into|copyFrom|copy_from|__copy\b')
OWN_SIG = re.compile(r'alias|shar(?:ed|ing)|mutat|isolat|ownership|frozen|immutable|singleton|memo\b|shallow|deep.?copy|copy_on_write|copy-on-write', re.I)
F2P_NAME_SIG = re.compile(r'copy|clone|deep|shallow|alias|ident|isolat|shared|mutat|ownership|immutable|frozen|singleton|same|original|pollut|leak|retain|preserv', re.I)
PROB_SIG = re.compile(r'copy|copies|copied|clone|alias|shar(?:ed|ing)|mutat|isolat|ownership|pollut|leak|unexpected side effect|in place|original', re.I)
TEST_IS_SIG = re.compile(r'\bis\s+(?:not\s+)?(?:None|True|False|[A-Za-z_][\w.]*)\b|assertIs|assertIsNot|is not|isinstance')
EXCLUDED_REPOS = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def load_forbidden():
    forb_i, forb_s, forb_f = set(), set(), set()
    notes = []
    base = json.loads((ACQ / 'dedup-baseline.json').read_text())
    forb_i |= set(base['forbidden_instance_ids'])
    forb_s |= set(base.get('forbidden_snapshot_group_ids', []))
    forb_f |= set(base['forbidden_repair_family_ids'])
    notes.append(f"common-400: {len(base['forbidden_instance_ids'])}i")
    f1 = json.loads((VFARL / 'acquisition/f1-new/reviewed-candidates.json').read_text())
    for t in f1.get('tasks', []):
        forb_i.add(t['instance_id'])
        if t.get('snapshot_group_id'):
            forb_s.add(t['snapshot_group_id'])
        if t.get('repair_family_id'):
            forb_f.add(t['repair_family_id'])
    notes.append(f"F1-40: {len(f1.get('tasks', []))}")
    f5 = json.loads((VFARL / 'acquisition/f5-new/acquired-40.json').read_text())
    for t in f5.get('tasks', []):
        forb_i.add(t.get('instance_id', t.get('canonical_task_id', '')))
        for key in ('snapshot_group_id', 'snapshot_id'):
            if t.get(key):
                forb_s.add(t[key])
        for key in ('repair_family_id', 'family_id'):
            if t.get(key):
                forb_f.add(t[key])
    notes.append(f"F5-old40: {len(f5.get('tasks', []))}")
    for rel in ['f2-replace/catalog.json', 'f2-replace/reviewed-candidates.json',
                'candidate-list-40.json', 'reviewed-candidates.json', 'catalog.json']:
        p = ACQ / rel
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        for t in d.get('tasks', d.get('candidates', [])):
            if t.get('instance_id'):
                forb_i.add(t['instance_id'])
            for key in ('snapshot_group_id', 'snapshot_group_id_provisional'):
                if t.get(key):
                    forb_s.add(t[key])
            for key in ('repair_family_id', 'repair_family_id_provisional'):
                if t.get(key):
                    forb_f.add(t[key])
        notes.append(f"f2prior:{rel}: {len(d.get('tasks', d.get('candidates', [])))}")
    bans = ['seperman/deepdiff::ed252022', 'pallets/click::fde47b4b',
            'pydantic/pydantic::acb0f10f', 'stanfordnlp/dspy::651a4c71',
            'sunpy/sunpy::f8edfd5c', 'mido/mido::a0158ff9',
            'cantools/cantools::0c6a7871']
    forb_s |= set(bans)
    notes.append(f"explicit round1 snapshot bans: {len(bans)}")
    for kdir in ['f2-gymv2', 'f2-gymv2b']:
        g2keep = json.loads((ACQ / kdir / 'keep-list.json').read_text())
        g2fam = json.loads((ACQ / kdir / 'keep-families.json').read_text())
        for k in g2keep:
            forb_i.add(k['instance_id'])
            forb_s.add(f"{k['repo']}::{k['base_commit']}")
        forb_f |= set(g2fam.values())
        notes.append(f"{kdir}: {len(g2keep)} keeps (instances+snapshots+{len(g2fam)} families)")
    return forb_i, forb_s, forb_f, notes


def snap_hit(canonical_repo, commit, forb_s):
    if not commit:
        return None
    for s in forb_s:
        if '::' not in s:
            continue
        repo, rev = s.split('::', 1)
        if repo.lower() != canonical_repo.lower():
            continue
        c = commit.lower()
        r = rev.lower()
        if c.startswith(r) or r.startswith(c):
            return s
    return None


def base_ok(r, forb_i):
    iid = r['instance_id']
    if iid in forb_i:
        return False
    repo = (r.get('repo') or '').strip()
    if repo.lower() in EXCLUDED_REPOS:
        return False
    f2p = r.get('FAIL_TO_PASS') or []
    p2p = r.get('PASS_TO_PASS') or []
    if not f2p or set(f2p) & set(p2p):
        return False
    if not (r.get('problem_statement') or '').strip():
        return False
    patch = r.get('patch') or ''
    files = re.findall(r'diff --git a/(\S+)', patch)
    if any('..' in f for f in files):
        return False
    total = len(f2p) + len(p2p)
    if total > 150:
        return None  # big-suite excluded (counted separately)
    return True


def changed_hits(patch, sig):
    chg = [l for l in patch.splitlines()
           if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++', '---'))]
    return [l.strip()[:200] for l in chg if sig.search(l)]


def cmd_dedup():
    forb_i, forb_s, forb_f, notes = load_forbidden()
    save(OUT / 'dedup-evidence.json', {
        'date': '2026-10-25',
        'scope': 'F600-F2 f2-gymv2c vs common-400 + F1-40 + F5-old40 + f2-round1/2 + f2-gymv2-16 + f2-gymv2b-5 disjointness',
        'forbidden_instance_count': len(forb_i),
        'forbidden_snapshot_count': len(forb_s),
        'forbidden_family_count': len(forb_f),
        'sources': notes,
        'method': 'instance exact-match; snapshot repo-equality + commit prefix either way; family exact-match; four competition repos excluded',
        'physical_split': 'train only',
    })
    print(f'forbidden: {len(forb_i)}i/{len(forb_s)}s/{len(forb_f)}f')
    for n in notes:
        print('  ', n)


def gym_rows():
    import pyarrow.parquet as pq
    return pq.read_table(str(GYM_PARQUET)).to_pylist()


def v2_py_rows():
    import pyarrow.parquet as pq
    f = pq.ParquetFile(str(V2_PARQUET))
    out = []
    for rg in range(f.num_row_groups):
        tbl = f.read_row_group(rg)
        for j, r in enumerate(tbl.to_pylist()):
            if (r.get('language') or '').lower() != 'python':
                continue
            r['_row_group'] = rg
            r['_idx_in_rg'] = j
            out.append(r)
    return f, out


def entry(r, forb_s, hits, signal):
    f2p = r.get('FAIL_TO_PASS') or []
    p2p = r.get('PASS_TO_PASS') or []
    patch = r.get('patch') or ''
    repo = (r.get('repo') or '').strip()
    files = re.findall(r'diff --git a/(\S+)', patch)
    return {'instance_id': r['instance_id'], 'repo': repo,
            'base_commit': r.get('base_commit'),
            'F2P': len(f2p), 'P2P': len(p2p), 'total': len(f2p) + len(p2p),
            'files': files[:6], 'F2P_list': f2p, 'P2P_sample': p2p[:4],
            'hits': hits[:6], 'signal': signal,
            'snapshot_collision': snap_hit(repo, r.get('base_commit') or '', forb_s)}


def cmd_scan():
    import pyarrow.parquet as pq
    forb_i, forb_s, _, _ = load_forbidden()
    rows = gym_rows()
    gym_out, gym_big = [], 0
    for r in rows:
        ok = base_ok(r, forb_i)
        if ok is None:
            if changed_hits(r.get('patch') or '', COPY_SIG):
                gym_big += 1
            continue
        if not ok:
            continue
        hits = changed_hits(r.get('patch') or '', COPY_SIG)
        if not hits:
            continue
        gym_out.append(entry(r, forb_s, hits, 'strict-copy'))
    gym_out.sort(key=lambda d: (d['total'], d['F2P']))
    f = pq.ParquetFile(str(V2_PARQUET))
    v2_out, v2_big, n_py = [], 0, 0
    for rg in range(f.num_row_groups):
        tbl = f.read_row_group(rg, columns=['instance_id', 'language', 'repo', 'base_commit',
                                            'patch', 'problem_statement',
                                            'FAIL_TO_PASS', 'PASS_TO_PASS'])
        for r in tbl.to_pylist():
            if (r.get('language') or '').lower() != 'python':
                continue
            n_py += 1
            ok = base_ok(r, forb_i)
            if ok is None:
                if changed_hits(r.get('patch') or '', COPY_SIG):
                    v2_big += 1
                continue
            if not ok:
                continue
            hits = changed_hits(r.get('patch') or '', COPY_SIG)
            if not hits:
                continue
            d = entry(r, forb_s, hits, 'strict-copy')
            d['row_group'] = rg
            v2_out.append(d)
    v2_out.sort(key=lambda d: (d['total'], d['F2P']))
    save(OUT / 'gym-strict-scan.json', {
        'date': '2026-10-25', 'source': 'SWE-Gym/SWE-Gym', 'revision': GYM_REV,
        'split': 'train', 'rows_scanned': len(rows),
        'parquet_bytes': GYM_PARQUET.stat().st_size,
        'parquet_sha256': hashlib.sha256(GYM_PARQUET.read_bytes()).hexdigest(),
        'suite_cap': 150, 'excluded_over_150': gym_big, 'hits': gym_out})
    save(OUT / 'v2-strict-scan.json', {
        'date': '2026-10-25', 'source': 'nebius/SWE-rebench-V2', 'revision': V2_REV,
        'split': 'train', 'language': 'python', 'python_rows_scanned': n_py,
        'parquet_bytes': V2_PARQUET.stat().st_size,
        'parquet_sha256': hashlib.sha256(V2_PARQUET.read_bytes()).hexdigest(),
        'suite_cap': 150, 'excluded_over_150': v2_big, 'hits': v2_out})
    print(f'gym rows={len(rows)} strict_hits={len(gym_out)} big_excluded={gym_big}')
    for d in gym_out:
        flag = 'SNAPCOLLIDE ' if d['snapshot_collision'] else ''
        print(f"{flag}{d['total']:4d} F2P={d['F2P']:3d} P2P={d['P2P']:4d} {d['instance_id']}")
    print(f'v2 python rows={n_py} strict_hits={len(v2_out)} big_excluded={v2_big}')
    for d in v2_out:
        flag = 'SNAPCOLLIDE ' if d['snapshot_collision'] else ''
        print(f"{flag}{d['total']:4d} F2P={d['F2P']:3d} P2P={d['P2P']:4d} {d['instance_id']} {d['repo']}")


def recall_signals(r):
    """Return list of (signal, hits) for expanded recall (strict-copy excluded by caller)."""
    patch = r.get('patch') or ''
    test_patch = r.get('test_patch') or ''
    prob = r.get('problem_statement') or ''
    f2p = r.get('FAIL_TO_PASS') or []
    out = []
    ch = changed_hits(patch, CLONE_SIG)
    if ch:
        out.append(('clone-changed', ch))
    ch = changed_hits(patch, OWN_SIG)
    if ch and (any(F2P_NAME_SIG.search(n) for n in f2p) or PROB_SIG.search(prob)):
        out.append(('ownership-changed', ch))
    if TEST_IS_SIG.search(test_patch) and (COPY_SIG.search(patch) or CLONE_SIG.search(patch)):
        f2p_hit = [n for n in f2p if F2P_NAME_SIG.search(n)]
        if f2p_hit or PROB_SIG.search(prob):
            m = TEST_IS_SIG.search(test_patch)
            out.append(('test-side-is', [f'F2P_name_hits={len(f2p_hit)}']))
    return out


def cmd_recall():
    import pyarrow.parquet as pq
    forb_i, forb_s, _, _ = load_forbidden()
    strict_ids = set()
    for fn in ['gym-strict-scan.json', 'v2-strict-scan.json']:
        p = OUT / fn
        if p.exists():
            for d in json.loads(p.read_text())['hits']:
                strict_ids.add(d['instance_id'])
    rows = gym_rows()
    gym_out, gym_big = [], 0
    for r in rows:
        if r['instance_id'] in strict_ids:
            continue
        ok = base_ok(r, forb_i)
        if ok is None:
            gym_big += 1
            continue
        if not ok:
            continue
        for sig, hits in recall_signals(r):
            d = entry(r, forb_s, hits, sig)
            gym_out.append(d)
            break
    gym_out.sort(key=lambda d: (d['total'], d['F2P']))
    f = pq.ParquetFile(str(V2_PARQUET))
    v2_out, v2_big, n_py = [], 0, 0
    for rg in range(f.num_row_groups):
        tbl = f.read_row_group(rg)
        for r in tbl.to_pylist():
            if (r.get('language') or '').lower() != 'python':
                continue
            n_py += 1
            if r['instance_id'] in strict_ids:
                continue
            ok = base_ok(r, forb_i)
            if ok is None:
                v2_big += 1
                continue
            if not ok:
                continue
            for sig, hits in recall_signals(r):
                d = entry(r, forb_s, hits, sig)
                d['row_group'] = rg
                v2_out.append(d)
                break
    v2_out.sort(key=lambda d: (d['total'], d['F2P']))
    save(OUT / 'gym-recall-scan.json', {
        'date': '2026-10-25', 'source': 'SWE-Gym/SWE-Gym', 'revision': GYM_REV,
        'split': 'train', 'rows_scanned': len(rows), 'suite_cap': 150,
        'excluded_over_150_nontrivial': gym_big, 'hits': gym_out,
        'signals': ['clone-changed', 'ownership-changed', 'test-side-is'],
        'strict_excluded': len([i for i in strict_ids])})
    save(OUT / 'v2-recall-scan.json', {
        'date': '2026-10-25', 'source': 'nebius/SWE-rebench-V2', 'revision': V2_REV,
        'split': 'train', 'language': 'python', 'python_rows_scanned': n_py,
        'suite_cap': 150, 'excluded_over_150_nontrivial': v2_big, 'hits': v2_out,
        'signals': ['clone-changed', 'ownership-changed', 'test-side-is']})
    print(f'gym recall_hits={len(gym_out)}')
    for d in gym_out:
        print(f"{d['total']:4d} F2P={d['F2P']:3d} [{d['signal']}] {d['instance_id']}")
    print(f'v2 recall_hits={len(v2_out)}')
    for d in v2_out:
        print(f"{d['total']:4d} F2P={d['F2P']:3d} [{d['signal']}] {d['instance_id']} {d['repo']}")


def cmd_show(pool, iid):
    import pyarrow.parquet as pq
    if pool == 'gym':
        rows = gym_rows()
        hit = [r for r in rows if r['instance_id'] == iid]
    else:
        f, rows = v2_py_rows()
        hit = [r for r in rows if r['instance_id'] == iid]
    assert len(hit) == 1, (pool, iid, len(hit))
    r = hit[0]
    print('=' * 100)
    print(f"{r['instance_id']} repo={r.get('repo')} base={r.get('base_commit')}")
    print(f"F2P({len(r.get('FAIL_TO_PASS') or [])}):")
    for n in (r.get('FAIL_TO_PASS') or [])[:20]:
        print('   F', n)
    print(f"P2P({len(r.get('PASS_TO_PASS') or [])}):")
    for n in (r.get('PASS_TO_PASS') or [])[:8]:
        print('   P', n)
    print('--- problem_statement (first 2500 chars) ---')
    print((r.get('problem_statement') or '')[:2500])
    print('--- patch (first 6000 chars) ---')
    print((r.get('patch') or '')[:6000])
    print('--- test_patch (first 6000 chars) ---')
    print((r.get('test_patch') or '')[:6000])


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    OUT.mkdir(parents=True, exist_ok=True)
    if which == 'dedup':
        cmd_dedup()
    elif which == 'scan':
        cmd_scan()
    elif which == 'recall':
        cmd_recall()
    elif which == 'show':
        cmd_show(sys.argv[2], sys.argv[3])
    else:
        print('unknown cmd')
