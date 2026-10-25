"""F2 f2-push: small-suite (total<=150) tier2/3-strong rescreen (Gym + V2).

New ground vs f2-more (all read-only, old dirs untouched):
  f2-more deep-read tier0-strong/tier1-strong/gym-F2P-copy-names and found M1/M2,
  but bulk-screened tier3 (733 rows, no test-is + no F2P-name) per "no is no-bet"
  and left tier2-strong (F2P-name + strong-inline, no test-is) largely
  unreviewed -- yet M2 itself is tier2 with NO test-is (is-leg carried by the
  source-bound inline observation, same as C2/K15/nox-439/aiohttp-10672/
  qiskit-12620). "No TEST is" != "no is triple", so tier2/3-strong rows with
  real ownership signals were never individually triple-reviewed. f2-push
  reviews exactly those (~125 rows) with the inline-is lens
  (is-identity + bidirectional visibility + copy timepoint; only-== -> S,
  capability-only -> R) plus a recall-nontrivial OWN lens on problem text.

Constraints (same as f2-more/f2-inline): train only; exclude
fastapi/rich/requests/httpx; zero overlap with common-400+F1-40+F5-old40+
f2-round1/2+f2-gymv2-16+f2-gymv2b-5+f2-gymv2c-2+f2-recall-3+f2-inline-1+
f2-more-2 (read-only, never rewrite old dirs); bad paths (..) filtered;
Smith-py frozen; total<=150, strict-copy excluded (covered by f2-gymv2/b/c);
suite<=200 file-scope batching for duals.

Usage:
  python3 scan_push.py dedup   # forbidden sets (read-only old files)
  python3 scan_push.py scan    # small-suite inline scan -> gym/v2-push-scan.json
  python3 scan_push.py cands   # push-candidates.json (tier2/3-strong, unreviewed)
  python3 scan_push.py show <pool> <instance_id>  # dump row for triple review
"""
import hashlib, json, re, sys
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT / 'competition/vfarl'
ACQ = VFARL / 'acquisition/f2-new'
OUT = ACQ / 'f2-push'
GYM_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/gym-train.parquet')
V2_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet')
GYM_REV = 'bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb'
V2_REV = '10483de0f50fe5da545942705a76c6150171af7f'
EXCLUDED = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}
COPY_SIG = re.compile(r'__copy__|__deepcopy__|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|def copy\(|(?<![\w.])\.copy\(\)')
INLINE_SIG = re.compile(r'\.copy\(\)|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|__copy__|__deepcopy__|\bclone\b|\.clone\(|CIMultiDict\s*\(|MultiDict\s*\(|clear_cache|copyInto|copy_into|\[:\]|\blist\s*\(|\bdict\s*\(|\bset\s*\(|tuple\s*\(|copy_module| shallow| deep.?copy', re.I)
STRONG_SIG = re.compile(r'\.copy\(\)|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|__copy__|__deepcopy__|\bclone\b|\.clone\(|CIMultiDict\s*\(|MultiDict\s*\(|clear_cache|copyInto|copy_into|\[:\]|copy_module| shallow| deep.?copy', re.I)
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
    for kdir in ['f2-gymv2', 'f2-gymv2b', 'f2-gymv2c', 'f2-recall', 'f2-inline', 'f2-more']:
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


def scan_pool(rows, pool, fi, fs):
    hits = []
    for loc, r in rows:
        iid = r['instance_id']
        if iid in fi: continue
        repo = (r.get('repo') or '').strip()
        if repo.lower() in EXCLUDED: continue
        f2p = r.get('FAIL_TO_PASS') or []; p2p = r.get('PASS_TO_PASS') or []
        if not f2p or set(f2p) & set(p2p): continue
        if not (r.get('problem_statement') or '').strip(): continue
        patch = r.get('patch') or ''
        files = re.findall(r'diff --git a/(\S+)', patch)
        if any('..' in f for f in files): continue
        total = len(f2p) + len(p2p)
        if total > 150 or total < 2: continue
        if snap_hit(repo, r.get('base_commit') or '', fs): continue
        tp = r.get('test_patch') or ''
        si = [l.strip()[:220] for l in tp.splitlines() if STRICT_IS.search(l)][:6]
        sc = changed_hits(patch, COPY_SIG)
        if sc: continue  # strict-copy <=150 already covered by f2-gymv2/b/c rounds
        inl = changed_hits(patch, INLINE_SIG)
        if not inl: continue  # require broad-inline changed-line signal (same universe as f2-more)
        f2pn = [n for n in f2p if F2P_NAME_SIG.search(n)]
        prob = r.get('problem_statement') or ''
        own_prob = [m.group(0)[:40] for m in OWN_PROB.finditer(prob)][:4]
        own_patch = changed_hits(patch, OWN_PATCH)[:4]
        strong = [l for l in inl if STRONG_SIG.search(l)]
        noise = [l for l in inl if NOISE_SIG.search(l)]
        hits.append({'instance_id': iid, 'pool': pool, 'loc': loc, 'repo': repo,
                     'base_commit': r.get('base_commit'), 'F2P': len(f2p), 'P2P': len(p2p),
                     'total': total, 'inline_hits': inl[:6], 'inline_n': len(inl),
                     'strong_hits': strong[:4], 'strong_n': len(strong),
                     'noise_hits': noise[:2], 'noise_n': len(noise),
                     'test_is_lines': si, 'test_is_n': len(si),
                     'f2p_name_hits': len(f2pn), 'f2p_sample': f2p[:3],
                     'own_problem': own_prob, 'own_problem_n': len(own_prob),
                     'own_patch': own_patch, 'own_patch_n': len(own_patch)})
    def rk(d):
        tier = 3
        if d['test_is_n'] > 0 and d['f2p_name_hits'] > 0: tier = 0
        elif d['test_is_n'] > 0: tier = 1
        elif d['f2p_name_hits'] > 0: tier = 2
        return (tier, -d['strong_n'], -d['own_problem_n'], d['total'])
    hits.sort(key=rk)
    return hits


def prior_named_ids():
    ids = set()
    for rel in ['f2-more/triple-review-log.md', 'f2-inline/triple-review-log.md',
                'f2-recall/triple-review-log.md', 'f2-large/triple-review-log.md',
                'f2-gymv2/triple-review-log.md', 'f2-gymv2b/triple-review-log.md',
                'f2-gymv2c/triple-review-log.md']:
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
             {'date': '2026-10-25', 'scope': 'F600-F2 f2-push vs common-400+F1-40+F5-old40+f2-round1/2+f2-gymv2-16+f2-gymv2b-5+f2-gymv2c-2+f2-recall-3+f2-inline-1+f2-more-2 disjointness',
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
        gym_hits = scan_pool(grows, 'gym', fi, fs)
        f = pq.ParquetFile(str(V2_PARQUET))
        vrows = []
        n_py = 0
        for rg in range(f.num_row_groups):
            for j, r in enumerate(f.read_row_group(rg).to_pylist()):
                if (r.get('language') or '').lower() != 'python': continue
                n_py += 1
                vrows.append(((rg, j), r))
        v2_hits = scan_pool(vrows, 'v2', fi, fs)
        save(OUT / 'gym-push-scan.json',
             {'date': '2026-10-25', 'source': 'SWE-Gym/SWE-Gym', 'revision': GYM_REV, 'split': 'train',
              'rows_scanned': len(grows), 'parquet_sha256': hashlib.sha256(GYM_PARQUET.read_bytes()).hexdigest(),
              'mode': 'small-suite-inline-rescreen-total_le150-strictCOPY-excluded-push', 'count': len(gym_hits), 'hits': gym_hits})
        save(OUT / 'v2-push-scan.json',
             {'date': '2026-10-25', 'source': 'nebius/SWE-rebench-V2', 'revision': V2_REV, 'split': 'train',
              'language': 'python', 'python_rows_scanned': n_py,
              'parquet_sha256': hashlib.sha256(V2_PARQUET.read_bytes()).hexdigest(),
              'mode': 'small-suite-inline-rescreen-total_le150-strictCOPY-excluded-push', 'count': len(v2_hits), 'hits': v2_hits})
        print(f"gym small-inline={len(gym_hits)} v2 small-inline={len(v2_hits)} total={len(gym_hits)+len(v2_hits)}")
        from collections import Counter
        print('tiers gym', Counter((min(d['test_is_n'],1), min(d['f2p_name_hits'],1)) for d in gym_hits))
        print('tiers v2', Counter((min(d['test_is_n'],1), min(d['f2p_name_hits'],1)) for d in v2_hits))
    elif which == 'cands':
        gym_hits = json.loads((OUT / 'gym-push-scan.json').read_text())['hits']
        v2_hits = json.loads((OUT / 'v2-push-scan.json').read_text())['hits']
        named = prior_named_ids()
        cands = []
        for h in gym_hits + v2_hits:
            tier = 3
            if h['test_is_n'] > 0 and h['f2p_name_hits'] > 0: tier = 0
            elif h['test_is_n'] > 0: tier = 1
            elif h['f2p_name_hits'] > 0: tier = 2
            if tier not in (2, 3): continue
            if h['strong_n'] == 0: continue
            if h['instance_id'] in named: continue  # already individually dispositioned
            cands.append(h)
        # review order: tier2 first (M2 pattern), then problem-OWN, then strong_n, smaller total
        cands.sort(key=lambda d: (0 if (d['test_is_n']==0 and d['f2p_name_hits']>0) else 1,
                                  -d['own_problem_n'], -d['own_patch_n'], -d['strong_n'], d['total']))
        save(OUT / 'push-candidates.json',
             {'date': '2026-10-25', 'scope': 'f2-push tier2/3-strong never-individually-reviewed (inline-is + recall-OWN lens)',
              'count': len(cands), 'prior_named_excluded': len(named), 'candidates': cands})
        print(f'push candidates={len(cands)} (prior-named excluded pool={len(named)})')
        for d in cands[:60]:
            t2 = d['f2p_name_hits'] > 0
            print(f" {'T2' if t2 else 'T3'} t{d['total']:3d} F{d['F2P']:2d} ownP{d['own_problem_n']} ownC{d['own_patch_n']} s{d['strong_n']} {d['instance_id']}")
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
        print('==TEST_PATCH(is lines)==')
        n = 0
        for l in (r.get('test_patch') or '').splitlines():
            if STRICT_IS.search(l) or 'is' in l[:160]:
                print(' ', l.strip()[:200]); n += 1
                if n > 30: break
