"""F2 f2-push5: fresh weak-offset slices, all total<=200 (Gym+V2, train only).

New ground vs all prior F2 batches (read-only, old dirs untouched):
  prior coverage recap (see f2-push4/scan_push4.py docstring):
    strict-copy <=150 (gymv2/b/c), strict-copy >150 (large),
    recall-nontrivial >150 REAL_IS<=200 (recall), recall 151-200 inline (inline),
    BROAD-inline <=150 strong tier0/1 (more) + tier2/3 (push),
    BROAD-inline 151-400 strong + P2P-empty<=50 strong (push2/3 A+B, exhausted),
    weak-offset<=150 + V2-151-200-weak-recall + P2P-empty-51-200 (push4 W/R/E).
  Fresh slices here (never individually triple-reviewed), all total<=200:
    A) gym-weak-151-200: Gym total 151-200, strict-copy excluded,
       strong_n==0, weak inline + support (F2P-name/ownership/test-is).
       (push4 R was V2-only; push2/3 A was strong-only.)
    B) v2-weak-151-200-testis-only: V2 total 151-200, strict excluded,
       strong_n==0, weak inline, NO recall support
       (clone_n==0, own_problem==0, own_patch==0, f2p_name==0),
       support from test-is only. (push4 R required recall support.)
    C) p2pempty-le50-weak: P2P==0, total<=50, strict excluded,
       strong_n==0, weak inline + support. (push2/3 B was strong-only.)
    D) weak-nosupport-le150: total<=150, strict excluded, strong_n==0,
       weak inline, NO support at all (lottery slice, rank by small total).
       (push4 W required support.)
  All with inline-is lens (is-identity + bidirectional visibility + copy
  timepoint; only-== -> S, capability-only -> R).

Constraints (same as f2-push4): train only; exclude
fastapi/rich/requests/httpx; zero overlap with common-400+F1-40+F5-old40+
f2-round1/2+f2-gymv2-16+f2-gymv2b-5+f2-gymv2c-2+f2-recall-3+f2-inline-1+
f2-more-2+f2-push-6+f2-push2-3+f2-push3-2+f2-push4-0 (read-only, never
rewrite old dirs); bad paths (..) filtered; Smith-py frozen; six tools;
verbatim prompts; JUnit dual; total<=200 batchable.

Usage:
  python3 scan_push5.py dedup
  python3 scan_push5.py scan
  python3 scan_push5.py cands
  python3 scan_push5.py show <pool> <instance_id>
"""
import hashlib, json, re, sys
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT / 'competition/vfarl'
ACQ = VFARL / 'acquisition/f2-new'
OUT = ACQ / 'f2-push5'
GYM_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/gym-train.parquet')
V2_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet')
GYM_REV = 'bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb'
V2_REV = '10483de0f50fe5da545942705a76c6150171af7f'
EXCLUDED = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}
COPY_SIG = re.compile(r'__copy__|__deepcopy__|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|def copy\(|(?<![\w.])\.copy\(\)')
INLINE_SIG = re.compile(r'\.copy\(\)|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|__copy__|__deepcopy__|\bclone\b|\.clone\(|CIMultiDict\s*\(|MultiDict\s*\(|clear_cache|copyInto|copy_into|\[:\]|\blist\s*\(|\bdict\s*\(|\bset\s*\(|tuple\s*\(|copy_module| shallow| deep.?copy', re.I)
STRONG_SIG = re.compile(r'\.copy\(\)|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|__copy__|__deepcopy__|\bclone\b|\.clone\(|CIMultiDict\s*\(|MultiDict\s*\(|clear_cache|copyInto|copy_into|\[:\]|copy_module| shallow| deep.?copy', re.I)
WEAK_SIG = re.compile(r'\blist\s*\(|\bdict\s*\(|\bset\s*\(|tuple\s*\(', re.I)
CLONE_SIG = re.compile(r'\bclone\b|\.clone\(|clone\(|copyInto|copy_into|copyFrom|copy_from|__copy\b')
F2P_NAME_SIG = re.compile(r'copy|clone|deep|shallow|alias|ident|isolat|shared|mutat|ownership|immutable|frozen|singleton|same|original|pollut|leak|retain|preserv|not_mutated|unchang|independ|separat', re.I)
STRICT_IS = re.compile(r'assertIs(?:Not)?\b|\bis\s+(?:not\s+)?(?!None\b|True\b|False\b)[A-Za-z_][\w.]*\b')
OWN_PROB = re.compile(r'alias|shar(?:ed|ing)|mutat|isolat|ownership|frozen|immutable|singleton|\bmemo\b|shallow|deep.?copy|copy_on_write|copy-on-write|identical|same object|same instance|independ|separat|pollut|leak|retain|preserv|side.?effect|unintended|not mutate|original\b.{0,20}(chang|modif|untouch|intact)', re.I)
OWN_PATCH = re.compile(r'alias|shar(?:ed|ing)|mutat|isolat|ownership|frozen|immutable|singleton|\bmemo\b|shallow|copy_on_write|copy-on-write|same object|same instance|independ|separat|pollut|leak|retain|preserv', re.I)
NOISE_SIG = re.compile(r'git clone|github\.com.*clone|Fixed .* by :user:|changelog|release notes', re.I)


def save(p, o):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(o, ensure_ascii=False, indent=2) + '\n')


def load_forbidden():
    fi, fs, ff = set(), set(), set()
    notes = []
    base = json.loads((ACQ / 'dedup-baseline.json').read_text())
    fi |= set(base['forbidden_instance_ids']); fs |= set(base.get('forbidden_snapshot_group_ids', [])); ff |= set(base['forbidden_repair_family_ids'])
    notes.append(f"common-400:{len(base['forbidden_instance_ids'])}i")
    f1 = json.loads((VFARL / 'acquisition/f1-new/reviewed-candidates.json').read_text())
    for t in f1.get('tasks', []):
        fi.add(t['instance_id'])
        if t.get('snapshot_group_id'): fs.add(t['snapshot_group_id'])
        if t.get('repair_family_id'): ff.add(t['repair_family_id'])
    notes.append(f"F1-40:{len(f1.get('tasks',[]))}")
    f5 = json.loads((VFARL / 'acquisition/f5-new/acquired-40.json').read_text())
    for t in f5.get('tasks', []):
        fi.add(t.get('instance_id', t.get('canonical_task_id', '')))
        for k in ('snapshot_group_id', 'snapshot_id'):
            if t.get(k): fs.add(t[k])
        for k in ('repair_family_id', 'family_id'):
            if t.get(k): ff.add(t[k])
    notes.append(f"F5-old40:{len(f5.get('tasks',[]))}")
    for rel in ['f2-replace/catalog.json', 'f2-replace/reviewed-candidates.json', 'candidate-list-40.json', 'reviewed-candidates.json', 'catalog.json']:
        p = ACQ / rel
        try: d = json.loads(p.read_text())
        except Exception: continue
        for t in d.get('tasks', d.get('candidates', [])):
            if t.get('instance_id'): fi.add(t['instance_id'])
            for k in ('snapshot_group_id', 'snapshot_group_id_provisional'):
                if t.get(k): fs.add(t[k])
            for k in ('repair_family_id', 'repair_family_id_provisional'):
                if t.get(k): ff.add(t[k])
        notes.append(f"f2prior:{rel}:{len(d.get('tasks', d.get('candidates',[])))}")
    fs |= {'seperman/deepdiff::ed252022', 'pallets/click::fde47b4b', 'pydantic/pydantic::acb0f10f',
           'stanfordnlp/dspy::651a4c71', 'sunpy/sunpy::f8edfd5c', 'mido/mido::a0158ff9', 'cantools/cantools::0c6a7871'}
    notes.append("explicit round1 snapshot bans:7")
    for kdir in ['f2-gymv2', 'f2-gymv2b', 'f2-gymv2c', 'f2-recall', 'f2-inline', 'f2-more', 'f2-push', 'f2-push2', 'f2-push3', 'f2-push4']:
        g2keep = json.loads((ACQ / kdir / 'keep-list.json').read_text())
        g2fam = json.loads((ACQ / kdir / 'keep-families.json').read_text())
        for k in g2keep:
            fi.add(k['instance_id']); fs.add(f"{k['repo']}::{k['base_commit']}")
        ff |= set(g2fam.values())
        notes.append(f"{kdir}:{len(g2keep)}keeps")
    return fi, fs, ff, notes


def snap_hit(repo, commit, fs):
    if not commit: return None
    for s in fs:
        if '::' not in s: continue
        r, rev = s.split('::', 1)
        if r.lower() != repo.lower(): continue
        c, rr = commit.lower(), rev.lower()
        if c.startswith(rr) or rr.startswith(c): return s
    return None


def changed_hits(patch, sig):
    chg = [l for l in patch.splitlines() if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++', '---'))]
    return [l.strip()[:220] for l in chg if sig.search(l)]


def base_rec(loc, r, pool, fi, fs):
    iid = r['instance_id']
    if iid in fi: return None
    repo = (r.get('repo') or '').strip()
    if repo.lower() in EXCLUDED: return None
    f2p = r.get('FAIL_TO_PASS') or []; p2p = r.get('PASS_TO_PASS') or []
    if not f2p or set(f2p) & set(p2p): return None
    if not (r.get('problem_statement') or '').strip(): return None
    patch = r.get('patch') or ''
    files = re.findall(r'diff --git a/(\S+)', patch)
    if any('..' in f for f in files): return None
    total = len(f2p) + len(p2p)
    if total > 200: return None
    if snap_hit(repo, r.get('base_commit') or '', fs): return None
    tp = r.get('test_patch') or ''
    si = [l.strip()[:220] for l in tp.splitlines() if STRICT_IS.search(l)][:6]
    sc = changed_hits(patch, COPY_SIG)
    if sc: return None  # strict-copy ground exhausted (gymv2/b/c + large)
    inl = changed_hits(patch, INLINE_SIG)
    if not inl: return None
    strong = [l for l in inl if STRONG_SIG.search(l)]
    weak = [l for l in inl if WEAK_SIG.search(l)]
    if strong: return None  # strong ground exhausted (more/push/push2/3/push4-E)
    if not weak: return None
    clone = changed_hits(patch, CLONE_SIG)
    f2pn = [n for n in f2p if F2P_NAME_SIG.search(n)]
    prob = r.get('problem_statement') or ''
    own_prob = [m.group(0)[:40] for m in OWN_PROB.finditer(prob)][:4]
    own_patch = changed_hits(patch, OWN_PATCH)[:4]
    noise = [l for l in inl if NOISE_SIG.search(l)]
    return {'instance_id': iid, 'pool': pool, 'loc': loc, 'repo': repo,
            'base_commit': r.get('base_commit'), 'F2P': len(f2p), 'P2P': len(p2p),
            'total': total, 'inline_hits': inl[:6], 'inline_n': len(inl),
            'strong_hits': [], 'strong_n': 0,
            'weak_hits': weak[:4], 'weak_n': len(weak),
            'clone_hits': clone[:4], 'clone_n': len(clone),
            'noise_hits': noise[:2], 'noise_n': len(noise),
            'test_is_lines': si, 'test_is_n': len(si),
            'f2p_name_hits': len(f2pn), 'f2p_sample': f2p[:3],
            'own_problem': own_prob, 'own_problem_n': len(own_prob),
            'own_patch': own_patch, 'own_patch_n': len(own_patch),
            'files': files[:4], 'n_files': len(files)}


def scan_pool(rows, pool, fi, fs):
    hitsA, hitsB, hitsC, hitsD = [], [], [], []
    for loc, r in rows:
        d = base_rec(loc, r, pool, fi, fs)
        if d is None: continue
        total = d['total']; p2p = d['P2P']
        support = (d['f2p_name_hits'] > 0 or d['own_problem_n'] > 0 or d['test_is_n'] > 0)
        recall_support = (d['clone_n'] > 0 or d['own_problem_n'] > 0 or d['own_patch_n'] > 0 or d['f2p_name_hits'] > 0)
        # A: gym 151-200 weak + support
        if pool == 'gym' and 151 <= total <= 200 and support:
            hitsA.append(d)
        # B: v2 151-200 weak, test-is-only support (no recall support)
        if pool == 'v2' and 151 <= total <= 200 and not recall_support and d['test_is_n'] > 0:
            hitsB.append(d)
        # C: P2P-empty <=50 weak + support
        if p2p == 0 and total <= 50 and support:
            hitsC.append(d)
        # D: <=150 weak, no support at all
        if total <= 150 and not support:
            hitsD.append(d)
    def rkA(d):
        return (-d['f2p_name_hits'], -d['own_problem_n'], -d['test_is_n'], d['total'])
    def rkB(d):
        return (d['total'], -d['test_is_n'])
    def rkC(d):
        return (d['total'], -d['f2p_name_hits'], -d['test_is_n'])
    def rkD(d):
        return (d['total'], d['n_files'])
    hitsA.sort(key=rkA); hitsB.sort(key=rkB); hitsC.sort(key=rkC); hitsD.sort(key=rkD)
    return hitsA, hitsB, hitsC, hitsD


def prior_named_ids():
    ids = set()
    for rel in ['f2-more/triple-review-log.md', 'f2-inline/triple-review-log.md',
                'f2-recall/triple-review-log.md', 'f2-large/triple-review-log.md',
                'f2-gymv2/triple-review-log.md', 'f2-gymv2b/triple-review-log.md',
                'f2-gymv2c/triple-review-log.md', 'f2-push/triple-review-log.md',
                'f2-push2/triple-review-log.md', 'f2-push3/triple-review-log.md',
                'f2-push4/triple-review-log.md']:
        try: txt = (ACQ / rel).read_text()
        except Exception: continue
        for m in re.finditer(r'`([A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+)`', txt):
            ids.add(m.group(1))
    return ids


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    OUT.mkdir(parents=True, exist_ok=True)
    if which == 'dedup':
        fi, fs, ff, notes = load_forbidden()
        save(OUT / 'dedup-evidence.json',
             {'date': '2026-10-25', 'scope': 'F600-F2 f2-push5 vs common-400+F1-40+F5-old40+f2-round1/2+f2-gymv2-16+f2-gymv2b-5+f2-gymv2c-2+f2-recall-3+f2-inline-1+f2-more-2+f2-push-6+f2-push2-3+f2-push3-2+f2-push4-0 disjointness',
              'forbidden_instance_count': len(fi), 'forbidden_snapshot_count': len(fs),
              'forbidden_family_count': len(ff), 'sources': notes,
              'method': 'instance exact; snapshot repo-equality+commit prefix either way; family exact; four repos excluded',
              'physical_split': 'train only'})
        print(f'forbidden {len(fi)}i/{len(fs)}s/{len(ff)}f')
        [print(' ', n) for n in notes]
    elif which == 'scan':
        import pyarrow.parquet as pq
        fi, fs, _, _ = load_forbidden()
        grows = [(idx, r) for idx, r in enumerate(pq.read_table(str(GYM_PARQUET)).to_pylist())]
        gA, gB, gC, gD = scan_pool(grows, 'gym', fi, fs)
        f = pq.ParquetFile(str(V2_PARQUET))
        vrows = []
        n_py = 0
        for rg in range(f.num_row_groups):
            for j, r in enumerate(f.read_row_group(rg).to_pylist()):
                if (r.get('language') or '').lower() != 'python': continue
                n_py += 1
                vrows.append(((rg, j), r))
        vA, vB, vC, vD = scan_pool(vrows, 'v2', fi, fs)
        save(OUT / 'gym-push5-scan.json',
             {'date': '2026-10-25', 'source': 'SWE-Gym/SWE-Gym', 'revision': GYM_REV, 'split': 'train',
              'rows_scanned': len(grows), 'parquet_sha256': hashlib.sha256(GYM_PARQUET.read_bytes()).hexdigest(),
              'mode': 'push5 weak-only fresh slices (strict+strong excluded, total<=200)',
              'count_A': len(gA), 'count_B': len(gB), 'count_C': len(gC), 'count_D': len(gD),
              'hitsA': gA, 'hitsB': gB, 'hitsC': gC, 'hitsD': gD})
        save(OUT / 'v2-push5-scan.json',
             {'date': '2026-10-25', 'source': 'nebius/SWE-rebench-V2', 'revision': V2_REV, 'split': 'train',
              'language': 'python', 'python_rows_scanned': n_py,
              'parquet_sha256': hashlib.sha256(V2_PARQUET.read_bytes()).hexdigest(),
              'mode': 'push5 weak-only fresh slices (strict+strong excluded, total<=200)',
              'count_A': len(vA), 'count_B': len(vB), 'count_C': len(vC), 'count_D': len(vD),
              'hitsA': vA, 'hitsB': vB, 'hitsC': vC, 'hitsD': vD})
        print(f"gym A={len(gA)} B={len(gB)} C={len(gC)} D={len(gD)} | v2 A={len(vA)} B={len(vB)} C={len(vC)} D={len(vD)} | total={len(gA)+len(gB)+len(gC)+len(gD)+len(vA)+len(vB)+len(vC)+len(vD)}")
    elif which == 'cands':
        g = json.loads((OUT / 'gym-push5-scan.json').read_text())
        v = json.loads((OUT / 'v2-push5-scan.json').read_text())
        named = prior_named_ids()
        union = {}
        for pool_tag, lst in [('A', g['hitsA']), ('B', v['hitsB']),
                              ('C', g['hitsC'] + v['hitsC']), ('D', g['hitsD'] + v['hitsD'])]:
            for h in lst:
                if h['instance_id'] in named: continue
                if h['instance_id'] not in union:
                    h = dict(h); h['pools'] = [pool_tag]; union[h['instance_id']] = h
                elif pool_tag not in union[h['instance_id']]['pools']:
                    union[h['instance_id']]['pools'].append(pool_tag)
        cands = sorted(union.values(), key=lambda d: (d['total'], -d['f2p_name_hits'], -d['test_is_n']))
        by_pool = {p: sum(1 for c in cands if p in c['pools']) for p in 'ABCD'}
        save(OUT / 'push5-candidates.json',
             {'date': '2026-10-25', 'scope': 'f2-push5 fresh weak-only slices A(gym151-200) + B(v2-151-200-testis-only) + C(p2pempty-le50) + D(weak-nosupport-le150), never-individually-reviewed, total<=200',
              'unique_count': len(cands), 'by_pool': by_pool, 'prior_named_excluded': len(named),
              'candidates': cands})
        print(f"push5 unique={len(cands)} by_pool={by_pool} (prior-named excluded={len(named)})")
        for d in cands[:80]:
            print(f" t{d['total']:3d} F{d['F2P']:2d} P{d['P2P']:3d} pools={''.join(sorted(d['pools']))} w{d['weak_n']} fn{d['f2p_name_hits']} ownP{d['own_problem_n']} tis{d['test_is_n']} nf{d['n_files']} {d['pool']:4s} {d['instance_id']} :: {d['weak_hits'][0][:80] if d['weak_hits'] else ''}")
    elif which == 'show':
        import pyarrow.parquet as pq
        pool, iid = sys.argv[2], sys.argv[3]
        if pool == 'gym':
            rows = pq.read_table(str(GYM_PARQUET)).to_pylist()
            r = next(x for x in rows if x['instance_id'] == iid)
        else:
            f = pq.ParquetFile(str(V2_PARQUET))
            r = None
            for rg in range(f.num_row_groups):
                for x in f.read_row_group(rg).to_pylist():
                    if x.get('instance_id') == iid:
                        r = x; break
                if r: break
        print('==REPO==', r.get('repo'), (r.get('base_commit') or '')[:12])
        print('==F2P==', r.get('FAIL_TO_PASS'))
        print('==P2P n==', len(r.get('PASS_TO_PASS') or []), (r.get('PASS_TO_PASS') or [])[:8])
        print('==PROBLEM=='); print((r.get('problem_statement') or '')[:3000])
        print('==PATCH=='); print((r.get('patch') or '')[:7000])
        print('==TEST_PATCH(is/copy lines)==')
        n = 0
        for l in (r.get('test_patch') or '').splitlines():
            if STRICT_IS.search(l) or 'copy' in l.lower() or 'clone' in l.lower():
                print(' ', l.strip()[:200]); n += 1
                if n > 30: break
