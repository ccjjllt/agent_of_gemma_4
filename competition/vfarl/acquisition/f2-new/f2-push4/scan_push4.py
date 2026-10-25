"""F2 f2-push4: weak-offset (<=150) + V2 large-recall remainder (151-200 weak)
+ P2P-empty new window (51-200) inline-is (Gym+V2).

New ground vs all prior F2 batches (read-only, old dirs untouched):
  - f2-gymv2/b/c covered strict-copy <=150 (+C2 P2P-empty total=1 strict exception).
  - f2-large covered strict-copy >150 (35 rows, KEEP 0).
  - f2-recall covered recall-nontrivial >150 REAL_IS<=200 (48 rows).
  - f2-inline covered recall le200 151-200 inline (451 rows, KEEP 1).
  - f2-more covered BROAD-inline <=150 tier0/1-strong.
  - f2-push covered BROAD-inline <=150 tier2/3-strong (128 rows).
  - f2-push2/3 covered BROAD-inline 151-400 strong + P2P-empty<=50 strong
    (85 cands; 35 verdict + fresh 39 verdict = exhausted, KEEP 3+2).
  New ground here (never individually triple-reviewed), all total<=200:
    W) weak-offset: total<=150, strict-copy excluded, strong_n==0,
       weak inline (list(/dict(/set(/tuple(/[:] etc) + support
       (F2P-name or ownership-problem or test-is). Prior small-suite pools
       all required strong_n>0, so weak-only <=150 is unswept (Gym+V2).
    R) V2 large-recall remainder: V2 total 151-200, strict excluded,
       non-strong inline with recall support (clone-changed or ownership
       problem/patch or F2P-name), not prior-named. Weak slice of the
       large-recall window beyond REAL_IS/strong lenses.
    E) P2P-empty new window: P2P==0, total 51-200 (prior B was <=50),
       strict excluded, inline (strong OR weak+support), Gym+V2.
  All with inline-is lens (is-identity + bidirectional visibility + copy
  timepoint; only-== -> S, capability-only -> R).

Constraints (same as f2-push3): train only; exclude
fastapi/rich/requests/httpx; zero overlap with common-400+F1-40+F5-old40+
f2-round1/2+f2-gymv2-16+f2-gymv2b-5+f2-gymv2c-2+f2-recall-3+f2-inline-1+
f2-more-2+f2-push-6+f2-push2-3+f2-push3-2 (read-only, never rewrite old
dirs); bad paths (..) filtered; Smith-py frozen; six tools; verbatim
prompts; JUnit dual; total<=200 batchable.

Usage:
  python3 scan_push4.py dedup
  python3 scan_push4.py scan
  python3 scan_push4.py cands
  python3 scan_push4.py show <pool> <instance_id>
"""
import hashlib, json, re, sys
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT / 'competition/vfarl'
ACQ = VFARL / 'acquisition/f2-new'
OUT = ACQ / 'f2-push4'
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
    for kdir in ['f2-gymv2', 'f2-gymv2b', 'f2-gymv2c', 'f2-recall', 'f2-inline', 'f2-more', 'f2-push', 'f2-push2', 'f2-push3']:
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
    if sc: return None  # strict-copy covered by gymv2/b/c + large rounds
    inl = changed_hits(patch, INLINE_SIG)
    if not inl: return None
    strong = [l for l in inl if STRONG_SIG.search(l)]
    weak = [l for l in inl if WEAK_SIG.search(l)]
    clone = changed_hits(patch, CLONE_SIG)
    f2pn = [n for n in f2p if F2P_NAME_SIG.search(n)]
    prob = r.get('problem_statement') or ''
    own_prob = [m.group(0)[:40] for m in OWN_PROB.finditer(prob)][:4]
    own_patch = changed_hits(patch, OWN_PATCH)[:4]
    noise = [l for l in inl if NOISE_SIG.search(l)]
    return {'instance_id': iid, 'pool': pool, 'loc': loc, 'repo': repo,
            'base_commit': r.get('base_commit'), 'F2P': len(f2p), 'P2P': len(p2p),
            'total': total, 'inline_hits': inl[:6], 'inline_n': len(inl),
            'strong_hits': strong[:4], 'strong_n': len(strong),
            'weak_hits': weak[:4], 'weak_n': len(weak),
            'clone_hits': clone[:4], 'clone_n': len(clone),
            'noise_hits': noise[:2], 'noise_n': len(noise),
            'test_is_lines': si, 'test_is_n': len(si),
            'f2p_name_hits': len(f2pn), 'f2p_sample': f2p[:3],
            'own_problem': own_prob, 'own_problem_n': len(own_prob),
            'own_patch': own_patch, 'own_patch_n': len(own_patch),
            'files': files[:4], 'n_files': len(files)}


def scan_pool(rows, pool, fi, fs):
    hitsW, hitsR, hitsE = [], [], []
    for loc, r in rows:
        d = base_rec(loc, r, pool, fi, fs)
        if d is None: continue
        total = d['total']; p2p = d['P2P']
        support = (d['f2p_name_hits'] > 0 or d['own_problem_n'] > 0 or d['test_is_n'] > 0)
        # W: weak-offset, total<=150, non-strong, weak present, support
        if total <= 150 and d['strong_n'] == 0 and d['weak_n'] > 0 and support:
            hitsW.append(d)
        # R: V2 large-recall remainder, 151-200, non-strong, recall support
        if pool == 'v2' and 151 <= total <= 200 and d['strong_n'] == 0 and \
                (d['clone_n'] > 0 or d['own_problem_n'] > 0 or d['own_patch_n'] > 0 or d['f2p_name_hits'] > 0):
            hitsR.append(d)
        # E: P2P-empty new window 51-200, inline + (strong or support)
        if p2p == 0 and 51 <= total <= 200 and (d['strong_n'] > 0 or (d['inline_n'] > 0 and support)):
            # E with strong 151-400 was covered by push2/3 A only when total<=400 strong;
            # here E-strong 51-200 P2P-empty overlaps push A/B only if prior-named -> cands step excludes.
            hitsE.append(d)
    def rkW(d):
        return (-d['f2p_name_hits'], -d['own_problem_n'], -d['test_is_n'], d['total'])
    def rkR(d):
        return (-d['clone_n'], -d['f2p_name_hits'], -d['own_problem_n'], d['total'])
    def rkE(d):
        return (d['total'], -d['strong_n'], -d['f2p_name_hits'])
    hitsW.sort(key=rkW); hitsR.sort(key=rkR); hitsE.sort(key=rkE)
    return hitsW, hitsR, hitsE


def prior_named_ids():
    ids = set()
    for rel in ['f2-more/triple-review-log.md', 'f2-inline/triple-review-log.md',
                'f2-recall/triple-review-log.md', 'f2-large/triple-review-log.md',
                'f2-gymv2/triple-review-log.md', 'f2-gymv2b/triple-review-log.md',
                'f2-gymv2c/triple-review-log.md', 'f2-push/triple-review-log.md',
                'f2-push2/triple-review-log.md', 'f2-push3/triple-review-log.md']:
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
             {'date': '2026-10-25', 'scope': 'F600-F2 f2-push4 vs common-400+F1-40+F5-old40+f2-round1/2+f2-gymv2-16+f2-gymv2b-5+f2-gymv2c-2+f2-recall-3+f2-inline-1+f2-more-2+f2-push-6+f2-push2-3+f2-push3-2 disjointness',
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
        gW, gR, gE = scan_pool(grows, 'gym', fi, fs)
        f = pq.ParquetFile(str(V2_PARQUET))
        vrows = []
        n_py = 0
        for rg in range(f.num_row_groups):
            for j, r in enumerate(f.read_row_group(rg).to_pylist()):
                if (r.get('language') or '').lower() != 'python': continue
                n_py += 1
                vrows.append(((rg, j), r))
        vW, vR, vE = scan_pool(vrows, 'v2', fi, fs)
        save(OUT / 'gym-push4-scan.json',
             {'date': '2026-10-25', 'source': 'SWE-Gym/SWE-Gym', 'revision': GYM_REV, 'split': 'train',
              'rows_scanned': len(grows), 'parquet_sha256': hashlib.sha256(GYM_PARQUET.read_bytes()).hexdigest(),
              'mode': 'push4-W-weak-le150 + R-v2only + E-p2pempty-51-200 (strict excluded, total<=200)',
              'count_W': len(gW), 'count_R': len(gR), 'count_E': len(gE),
              'hitsW': gW, 'hitsR': gR, 'hitsE': gE})
        save(OUT / 'v2-push4-scan.json',
             {'date': '2026-10-25', 'source': 'nebius/SWE-rebench-V2', 'revision': V2_REV, 'split': 'train',
              'language': 'python', 'python_rows_scanned': n_py,
              'parquet_sha256': hashlib.sha256(V2_PARQUET.read_bytes()).hexdigest(),
              'mode': 'push4-W-weak-le150 + R-v2only-151-200-weak + E-p2pempty-51-200 (strict excluded, total<=200)',
              'count_W': len(vW), 'count_R': len(vR), 'count_E': len(vE),
              'hitsW': vW, 'hitsR': vR, 'hitsE': vE})
        print(f"gym W={len(gW)} R={len(gR)} E={len(gE)} | v2 W={len(vW)} R={len(vR)} E={len(vE)} | total={len(gW)+len(gR)+len(gE)+len(vW)+len(vR)+len(vE)}")
    elif which == 'cands':
        g = json.loads((OUT / 'gym-push4-scan.json').read_text())
        v = json.loads((OUT / 'v2-push4-scan.json').read_text())
        named = prior_named_ids()
        # union with pool priority E > R > W; track origin pools
        union = {}
        for pool_tag, lst in [('W', g['hitsW'] + v['hitsW']), ('R', v['hitsR']),
                              ('E', g['hitsE'] + v['hitsE'])]:
            for h in lst:
                if h['instance_id'] in named: continue
                if h['instance_id'] not in union:
                    h = dict(h); h['pools'] = [pool_tag]; union[h['instance_id']] = h
                elif pool_tag not in union[h['instance_id']]['pools']:
                    union[h['instance_id']]['pools'].append(pool_tag)
        cands = sorted(union.values(), key=lambda d: (d['total'], -d['strong_n'], -d['f2p_name_hits'], -d['own_problem_n']))
        by_pool = {'W': sum(1 for c in cands if 'W' in c['pools']),
                   'R': sum(1 for c in cands if 'R' in c['pools']),
                   'E': sum(1 for c in cands if 'E' in c['pools'])}
        save(OUT / 'push4-candidates.json',
             {'date': '2026-10-25', 'scope': 'f2-push4 new-ground W(weak<=150) + R(v2-151-200-weak-recall) + E(p2pempty-51-200), never-individually-reviewed, total<=200',
              'unique_count': len(cands), 'by_pool': by_pool, 'prior_named_excluded': len(named),
              'candidates': cands})
        print(f"push4 unique={len(cands)} by_pool={by_pool} (prior-named excluded={len(named)})")
        for d in cands[:60]:
            print(f" t{d['total']:3d} F{d['F2P']:2d} P{d['P2P']:3d} pools={''.join(sorted(d['pools']))} s{d['strong_n']} w{d['weak_n']} fn{d['f2p_name_hits']} ownP{d['own_problem_n']} tis{d['test_is_n']} nf{d['n_files']} {d['pool']:4s} {d['instance_id']} :: {(d['strong_hits']+d['weak_hits'])[0][:80] if (d['strong_hits']+d['weak_hits']) else ''}")
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
