"""F2 f2-xlarge: >200-suite batch basis.

Scope (this batch only, new ground not triaged before):
  Gym/V2 recall-nontrivial rows with total>200 (gym 658 + v2 899 = 1557,
  from f2-recall scan lists, read-only) + Smith-py cached large rows
  (total>200 among locally cached datasets-server pages, read-only).

Pools (train only, pinned):
  SWE-Gym/SWE-Gym            rev bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb (2438 train, local parquet)
  nebius/SWE-rebench-V2      rev 10483de0f50fe5da545942705a76c6150171af7f (python 7243, local parquet)
  SWE-bench/SWE-smith-py     rev 77cab9055d42ab4a5c25c89a8f937096db13558e (cached pages only, no refetch)

Constraints (unchanged, strict):
  zero family/snapshot/instance overlap with common-400 + F1-40 + F5-old40
  + f2-round1/2 + ALL f2-* keeps so far (41: gymv2 16 + gymv2b 5 + gymv2c 2
  + recall 3 + inline 1 + more 2 + push 6 + push2 3 + push3 2 + push4 1);
  four competition repos excluded; F2 triple mandatory (is-identity +
  shared-mutation visibility + copy timepoint); only-== demotes to S;
  capability-only to R; bad paths filtered; skip-prefix-collision precheck
  mandatory for every KEEP; whitelisted problem_statement; JUnit dual
  conditions; six tools; protected paths untouched. replay/.venv dual rounds.

Large-suite batch protocol (per f2-large basis):
  skip precheck first; only skip-CLEAN proceed to triple review;
  only triple-KEEP (is must pass) proceed to single-file file-scope
  validation (original matcher unchanged, no assertion shrinking).

Usage:
  python3 scan_xlarge.py dedup   # build forbidden sets (read-only old files)
  python3 scan_xlarge.py scan    # gym/v2 >200 rerank + smith cached large scan
  python3 scan_xlarge.py show <pool> <instance_id>  # full dump for triple review
                                   # pool in {gym,v2,smith}
"""
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT / 'competition/vfarl'
ACQ = VFARL / 'acquisition/f2-new'
OUT = ACQ / 'f2-xlarge'
GYM_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/gym-train.parquet')
V2_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet')
GYM_REV = 'bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb'
V2_REV = '10483de0f50fe5da545942705a76c6150171af7f'
SMITH_REV = '77cab9055d42ab4a5c25c89a8f937096db13558e'

COPY_SIG = re.compile(r'__copy__|__deepcopy__|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|def copy\(|(?<![\w.])\.copy\(\)')
CLONE_SIG = re.compile(r'\bclone\b|\.clone\(|clone\(|copyInto|copy_into|copyFrom|copy_from', re.I)
OWN_SIG = re.compile(r'alias|shar(?:ed|ing)|mutat|isolat|ownership|frozen|immutable|singleton|memo\b|shallow|deep.?copy|copy_on_write|copy-on-write', re.I)
F2P_NAME_SIG = re.compile(r'copy|clone|deep|shallow|alias|ident|isolat|shared|mutat|ownership|immutable|frozen|singleton|same|original|pollut|leak|retain|preserv', re.I)
TEST_IS_SIG = re.compile(r'\bis\s+(?:not\s+)?(?:None|True|False|[A-Za-z_][\w.]*)\b|assertIs|assertIsNot|is not|isinstance')
STRICT_IS = re.compile(r'assertIs(?:Not)?\b|\bis\s+(?:not\s+)?(?!None\b|True\b|False\b)[A-Za-z_][\w.]*\b')
EXCLUDED_REPOS = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}
KEEP_DIRS = ['f2-gymv2', 'f2-gymv2b', 'f2-gymv2c', 'f2-recall', 'f2-inline',
             'f2-more', 'f2-push', 'f2-push2', 'f2-push3', 'f2-push4', 'f2-push5']
SMITH_PAGE_DIRS = [ACQ / 'pages', VFARL / 'acquisition/f1-new/pages',
                   VFARL / 'acquisition/f5-new/pages', ACQ / 'f2-replace2/pages2']


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
    nkeep = 0
    for kdir in KEEP_DIRS:
        try:
            g2keep = json.loads((ACQ / kdir / 'keep-list.json').read_text())
            g2fam = json.loads((ACQ / kdir / 'keep-families.json').read_text())
        except Exception as e:
            notes.append(f"{kdir}: ERR {e}")
            continue
        items = g2keep if isinstance(g2keep, list) else list(g2keep.values())
        for k in items:
            if not isinstance(k, dict) or not k.get('instance_id'):
                continue
            forb_i.add(k['instance_id'])
            if k.get('repo') and k.get('base_commit'):
                forb_s.add(f"{k['repo']}::{k['base_commit']}")
            nkeep += 1
        vals = g2fam.values() if isinstance(g2fam, dict) else g2fam
        forb_f |= set(vals)
        notes.append(f"{kdir}: {len(items)} keeps listed")
    notes.append(f"total keeps listed: {nkeep}")
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


def changed_hits(patch, sig):
    chg = [l for l in (patch or '').splitlines()
           if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++', '---'))]
    return [l.strip()[:220] for l in chg if sig.search(l)]


def is_lines(text, pat):
    return [l.strip()[:220] for l in (text or '').splitlines() if pat.search(l)][:8]


def req_files_of(f2p, p2p):
    return sorted({n.split('::')[0] for n in list(f2p or []) + list(p2p or []) if '::' in n})


def cmd_dedup():
    forb_i, forb_s, forb_f, notes = load_forbidden()
    save(OUT / 'dedup-evidence.json', {
        'date': '2026-10-25',
        'scope': ('F600-F2 f2-xlarge vs common-400 + F1-40 + F5-old40 + f2-round1/2 + '
                  'ALL f2-* keeps (gymv2/gymv2b/gymv2c/recall/inline/more/push/push2/push3/push4/push5) disjointness'),
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


def parquet_row(pool, iid):
    import pyarrow.parquet as pq
    if pool == 'gym':
        for r in pq.read_table(str(GYM_PARQUET)).to_pylist():
            if r['instance_id'] == iid:
                return r
        return None
    f = pq.ParquetFile(str(V2_PARQUET))
    for rg in range(f.num_row_groups):
        for r in f.read_row_group(rg).to_pylist():
            if r['instance_id'] == iid:
                return r
    return None


def cmd_scan():
    import pyarrow.parquet as pq
    forb_i, forb_s, _, _ = load_forbidden()
    # --- large DONE (f2-large 35, never redo) ---
    large_done = set()
    for fn in ['gym-large-scan.json', 'v2-large-scan.json']:
        d = json.loads((ACQ / 'f2-large' / fn).read_text())
        for h in d.get('hits', []):
            large_done.add(h['instance_id'])
    # --- gym/v2 >200 from recall lists ---
    g_all = json.loads((ACQ / 'f2-recall/gym-recall-large-scan.json').read_text())['hits']
    v_all = json.loads((ACQ / 'f2-recall/v2-recall-large-scan.json').read_text())['hits']
    gym_big = [h for h in g_all if h['total'] > 200]
    v2_big = [h for h in v_all if h['total'] > 200]
    print(f'gym nontrivial={len(g_all)} gt200={len(gym_big)}; v2 nontrivial={len(v_all)} gt200={len(v2_big)}')

    def enrich(h, pool):
        r = parquet_row(pool, h['instance_id'])
        if r is None:
            h['body'] = 'PARQUET_MISSING'
            return h
        tp = r.get('test_patch') or ''
        patch = r.get('patch') or ''
        si = is_lines(tp, STRICT_IS)
        ti = is_lines(tp, TEST_IS_SIG)
        f2p = r.get('FAIL_TO_PASS') or []
        p2p = r.get('PASS_TO_PASS') or []
        f2pn = [n for n in f2p if F2P_NAME_SIG.search(n)]
        sc = changed_hits(patch, COPY_SIG)
        cl = changed_hits(patch, CLONE_SIG)
        own = list(dict.fromkeys(OWN_SIG.findall(patch)))[:8]
        req = req_files_of(f2p, p2p)
        h['strict_copy_now'] = bool(sc)
        h['strict_hits_now'] = sc[:4]
        h['clone_hits'] = cl[:4]
        h['own_hits'] = own
        h['test_is_strict_now'] = bool(si)
        h['test_is_any_now'] = bool(ti)
        h['test_is_lines_now'] = (si[:4] if si else ti[:4])
        h['f2p_name_hits_now'] = len(f2pn)
        h['f2p_name_sample'] = f2pn[:4]
        h['required_files'] = req
        h['single_file_required_closure'] = len(req) == 1
        h['large_done'] = h['instance_id'] in large_done
        h['snapshot_collision_now'] = snap_hit((r.get('repo') or '').strip(),
                                              r.get('base_commit') or '', forb_s)
        h['forbidden_instance'] = h['instance_id'] in forb_i
        return h

    for h in gym_big:
        enrich(h, 'gym')
    for h in v2_big:
        enrich(h, 'v2')

    def rank_key(d):
        tier = 6
        s, si, ai, fn = (d.get('strict_copy_now', d.get('strict_copy', False)),
                         d.get('test_is_strict_now', d.get('test_is_strict', False)),
                         d.get('test_is_any_now', d.get('test_is_any', False)),
                         d.get('f2p_name_hits_now', d.get('f2p_name_hits', 0)))
        if s and si:
            tier = 0
        elif s and ai:
            tier = 1
        elif s:
            tier = 2
        elif si:
            tier = 3
        elif ai:
            tier = 4
        elif fn:
            tier = 5
        return (tier, d['total'], d['F2P'])

    gym_big.sort(key=rank_key)
    v2_big.sort(key=rank_key)
    save(OUT / 'gym-xlarge-scan.json', {
        'date': '2026-10-25', 'source': 'SWE-Gym/SWE-Gym', 'revision': GYM_REV,
        'split': 'train', 'rows_scanned': pq.read_table(str(GYM_PARQUET)).num_rows,
        'parquet_sha256': hashlib.sha256(GYM_PARQUET.read_bytes()).hexdigest(),
        'mode': 'recall-nontrivial-total_gt_200 (f2-recall list filtered; enriched from parquet)',
        'count': len(gym_big), 'hits': gym_big})
    f = pq.ParquetFile(str(V2_PARQUET))
    save(OUT / 'v2-xlarge-scan.json', {
        'date': '2026-10-25', 'source': 'nebius/SWE-rebench-V2', 'revision': V2_REV,
        'split': 'train', 'language': 'python', 'python_rows_scanned': f.metadata.num_rows,
        'parquet_sha256': hashlib.sha256(V2_PARQUET.read_bytes()).hexdigest(),
        'mode': 'recall-nontrivial-total_gt_200 (f2-recall list filtered; enriched from parquet)',
        'count': len(v2_big), 'hits': v2_big})
    print(f'saved gym-xlarge={len(gym_big)} v2-xlarge={len(v2_big)}')
    for d in gym_big[:40]:
        print(f"  G t{d['total']:5d} F{d['F2P']:3d} strict={int(bool(d.get('strict_copy_now')))} "
              f"isS={int(bool(d.get('test_is_strict_now')))} isA={int(bool(d.get('test_is_any_now')))} "
              f"f2pn={d.get('f2p_name_hits_now')} 1f={int(bool(d.get('single_file_required_closure')))} "
              f"LD={int(bool(d.get('large_done')))} COLL={bool(d.get('snapshot_collision_now'))} "
              f"FORB={int(bool(d.get('forbidden_instance')))} {d['instance_id']}")
    for d in v2_big[:60]:
        print(f"  V t{d['total']:5d} F{d['F2P']:3d} strict={int(bool(d.get('strict_copy_now')))} "
              f"isS={int(bool(d.get('test_is_strict_now')))} isA={int(bool(d.get('test_is_any_now')))} "
              f"f2pn={d.get('f2p_name_hits_now')} 1f={int(bool(d.get('single_file_required_closure')))} "
              f"LD={int(bool(d.get('large_done')))} COLL={bool(d.get('snapshot_collision_now'))} "
              f"FORB={int(bool(d.get('forbidden_instance')))} {d['instance_id']} {d['repo']}")

    # --- smith cached large scan ---
    seen = {}
    pages = []
    for d in SMITH_PAGE_DIRS:
        pages += [p for p in glob.glob(str(d / 'train-*.json')) if 'receipt' not in p]
    print(f'smith cached page files: {len(pages)}')
    n_rows = 0
    smith_big = []
    for pf in sorted(pages):
        try:
            blob = json.loads(Path(pf).read_text())
        except Exception:
            continue
        for entry in blob.get('rows', []):
            row = entry.get('row', {})
            iid = row.get('instance_id')
            if not iid or iid in seen:
                continue
            seen[iid] = True
            n_rows += 1
            f2p = row.get('FAIL_TO_PASS') or []
            p2p = row.get('PASS_TO_PASS') or []
            if isinstance(f2p, str):
                try:
                    f2p = json.loads(f2p)
                except Exception:
                    f2p = []
            if isinstance(p2p, str):
                try:
                    p2p = json.loads(p2p)
                except Exception:
                    p2p = []
            total = len(f2p) + len(p2p)
            if total <= 200:
                continue
            if not f2p or set(f2p) & set(p2p):
                continue
            if not (row.get('problem_statement') or '').strip():
                continue
            repo = (row.get('repo') or '').strip()
            suffix = repo.split('/')[-1].split('.')[0].replace('__', '/') if '/' in repo else repo
            if suffix.lower() in EXCLUDED_REPOS or repo.lower() in EXCLUDED_REPOS:
                continue
            if iid in forb_i:
                continue
            patch = row.get('patch') or ''
            files = re.findall(r'diff --git a/(\S+)', patch)
            if any('..' in fl for fl in files):
                continue
            sc = changed_hits(patch, COPY_SIG)
            cl = changed_hits(patch, CLONE_SIG)
            own = list(dict.fromkeys(OWN_SIG.findall(patch)))[:8]
            f2pn = [n for n in f2p if F2P_NAME_SIG.search(n)]
            ps = (row.get('problem_statement') or '').lower()
            ps_hits = [k for k in ['copy', 'identical', 'identity', 'shared', 'alias',
                                   'ownership', 'sentinel', 'shallow', 'deepcopy', 'clone',
                                   ' is '] if k in ps]
            snap = None
            m = re.match(r'swesmith/(.+)', repo)
            canon = None
            if m:
                inner = m.group(1)
                parts = inner.rsplit('.', 1)
                if len(parts) == 2:
                    canon = parts[0].replace('__', '/')
                    snap = snap_hit(canon, parts[1], forb_s)
            req = req_files_of(f2p, p2p)
            smith_big.append({
                'instance_id': iid, 'pool': 'smith', 'row_idx': entry.get('row_idx'),
                'repo': repo, 'canonical_repo_guess': canon, 'base_commit_guess': parts[1] if m and len(parts) == 2 else None,
                'F2P': len(f2p), 'P2P': len(p2p), 'total': total,
                'strict_copy': bool(sc), 'strict_hits': sc[:4],
                'clone_hits': cl[:4], 'own_hits': own,
                'f2p_name_hits': len(f2pn), 'f2p_name_sample': f2pn[:4],
                'ps_hits': ps_hits,
                'F2P_sample': f2p[:4], 'P2P_sample': p2p[:2],
                'files': files[:8], 'required_files': req,
                'single_file_required_closure': len(req) == 1,
                'snapshot_collision': snap,
                'page': str(Path(pf).relative_to(ROOT)) if str(pf).startswith(str(ROOT)) else pf,
                'patch_head': patch[:600]})

    def srank(d):
        s, fn, psn = d['strict_copy'], d['f2p_name_hits'], len(d['ps_hits'])
        tier = 4
        if s and (fn or psn):
            tier = 0
        elif s:
            tier = 1
        elif fn and psn:
            tier = 2
        elif fn or psn:
            tier = 3
        return (tier, d['total'], d['F2P'])

    smith_big.sort(key=srank)
    save(OUT / 'smith-xlarge-scan.json', {
        'date': '2026-10-25', 'source': 'SWE-bench/SWE-smith-py', 'revision': SMITH_REV,
        'split': 'train', 'cached_page_files': len(pages), 'unique_rows_seen': n_rows,
        'mode': 'cached-pages-total_gt_200 (no refetch; bodies from local pages)',
        'count': len(smith_big), 'hits': smith_big})
    print(f'smith unique_rows={n_rows} gt200={len(smith_big)}')
    for d in smith_big[:60]:
        print(f"  S t{d['total']:5d} F{d['F2P']:3d} strict={int(d['strict_copy'])} "
              f"f2pn={d['f2p_name_hits']} ps={d['ps_hits']} 1f={int(d['single_file_required_closure'])} "
              f"COLL={bool(d['snapshot_collision'])} {d['instance_id']}")


def cmd_show(pool, iid):
    if pool == 'smith':
        for d in SMITH_PAGE_DIRS:
            for pf in sorted(glob.glob(str(d / 'train-*.json'))):
                if 'receipt' in pf:
                    continue
                try:
                    blob = json.loads(Path(pf).read_text())
                except Exception:
                    continue
                for entry in blob.get('rows', []):
                    row = entry.get('row', {})
                    if row.get('instance_id') == iid:
                        print('=' * 100)
                        print(f"{iid} repo={row.get('repo')} row_idx={entry.get('row_idx')} page={pf}")
                        f2p = row.get('FAIL_TO_PASS') or []
                        p2p = row.get('PASS_TO_PASS') or []
                        if isinstance(f2p, str):
                            f2p = json.loads(f2p)
                        if isinstance(p2p, str):
                            p2p = json.loads(p2p)
                        print(f"F2P({len(f2p)}):")
                        for n in f2p[:25]:
                            print('   F', n)
                        print(f"P2P({len(p2p)}):")
                        for n in p2p[:6]:
                            print('   P', n)
                        print('--- problem_statement (first 2200) ---')
                        print((row.get('problem_statement') or '')[:2200])
                        print('--- patch (first 5500) ---')
                        print((row.get('patch') or '')[:5500])
                        return
        print(f'smith row not in cache: {iid}')
        return
    r = parquet_row(pool, iid)
    assert r is not None, (pool, iid)
    print('=' * 100)
    print(f"{r['instance_id']} repo={r.get('repo')} base={r.get('base_commit')}")
    f2p = r.get('FAIL_TO_PASS') or []
    p2p = r.get('PASS_TO_PASS') or []
    print(f"F2P({len(f2p)}):")
    for n in f2p[:25]:
        print('   F', n)
    print(f"P2P({len(p2p)}):")
    for n in p2p[:6]:
        print('   P', n)
    print('--- problem_statement (first 2200) ---')
    print((r.get('problem_statement') or '')[:2200])
    print('--- patch (first 5500) ---')
    print((r.get('patch') or '')[:5500])
    print('--- test_patch STRICT_IS lines (up to 30) ---')
    cnt = 0
    for line in (r.get('test_patch') or '').splitlines():
        if STRICT_IS.search(line):
            print('   T:', line.strip()[:220])
            cnt += 1
            if cnt >= 30:
                break
    if cnt == 0:
        print('   (none)')
    print('--- test_patch (first 3500) ---')
    print((r.get('test_patch') or '')[:3500])


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    OUT.mkdir(parents=True, exist_ok=True)
    if which == 'dedup':
        cmd_dedup()
    elif which == 'scan':
        cmd_scan()
    elif which == 'show':
        cmd_show(sys.argv[2], sys.argv[3])
    else:
        print('unknown cmd')
