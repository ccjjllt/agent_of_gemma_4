"""F2 f2-smith: Smith-py train full weak-copy (non-strict) inline-is scan.

New ground vs all prior F2 (read-only, old dirs untouched):
  Smith strict (f2-replace 58 hits + f2-replace2 48 hits) covered
  strict changed-line copy (COPY_STRICT). This batch sweeps the
  complementary weak-only ground: strict excluded, strong_n==0,
  weak (list(/dict(/set(/tuple(/[:]) present on changed lines,
  with inline-is lens (is-identity + bidirectional visibility +
  copy timepoint; only-== -> S, capability-only -> R).

Constraints: train only (Smith-py rev 77cab905...); exclude
fastapi/rich/requests/httpx; zero overlap with common-400+F1-40+
F5-old40+f2-round1/2+f2-gymv2-16+f2-gymv2b-5+f2-gymv2c-2+f2-recall-3+
f2-inline-1+f2-more-2+f2-push-6+f2-push2-3+f2-push3-2+f2-push4-1+
f2-push5-0 (read-only); bad paths (..) filtered; total<=200 batchable;
six tools; verbatim prompts; JUnit dual; skip-CLEAN only to dual.

Usage:
  python3 scan_smith_weak.py dedup
  python3 scan_smith_weak.py scan
  python3 scan_smith_weak.py cands
  python3 scan_smith_weak.py show <instance_id>
"""
import hashlib, json, re, sys
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT / 'competition/vfarl'
ACQ = VFARL / 'acquisition/f2-new'
OUT = ACQ / 'f2-smith'
REV = '77cab9055d42ab4a5c25c89a8f937096db13558e'
EXCLUDED = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}
COPY_STRICT = re.compile(r'__copy__|__deepcopy__|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|def copy\(|(?<![\w.])\.copy\(\)')
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
        iid = t.get('instance_id', t.get('canonical_task_id', ''))
        if iid: fi.add(iid)
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
        try: n = len(d.get('tasks', d.get('candidates',[])))
        except Exception: n = 0
        notes.append(f"f2prior:{rel}:{n}")
    fs |= {'seperman/deepdiff::ed252022', 'pallets/click::fde47b4b', 'pydantic/pydantic::acb0f10f',
           'stanfordnlp/dspy::651a4c71', 'sunpy/sunpy::f8edfd5c', 'mido/mido::a0158ff9', 'cantools/cantools::0c6a7871'}
    notes.append("explicit round1 snapshot bans:7")
    for kdir in ['f2-gymv2', 'f2-gymv2b', 'f2-gymv2c', 'f2-recall', 'f2-inline', 'f2-more', 'f2-push', 'f2-push2', 'f2-push3', 'f2-push4', 'f2-push5']:
        try:
            g2keep = json.loads((ACQ / kdir / 'keep-list.json').read_text())
            g2fam = json.loads((ACQ / kdir / 'keep-families.json').read_text())
        except Exception:
            continue
        # keep-list may be list or dict
        if isinstance(g2keep, dict):
            items = []
        elif isinstance(g2keep, list):
            items = g2keep
        else:
            items = []
        for k in items:
            if not isinstance(k, dict): continue
            if k.get('instance_id'): fi.add(k['instance_id'])
            repo = k.get('repo') or k.get('canonical_repo') or ''
            base_commit = k.get('base_commit') or ''
            if repo and base_commit:
                fs.add(f"{repo}::{base_commit}")
            elif k.get('snapshot_group_id'):
                fs.add(k['snapshot_group_id'])
        if isinstance(g2fam, dict):
            ff |= set(g2fam.values())
        elif isinstance(g2fam, list):
            ff |= set([x for x in g2fam if isinstance(x, str)])
        notes.append(f"{kdir}:{len(items)}keeps")
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


def iter_cached_rows():
    seen = {}
    # f2-new/pages + f1-new/pages + f2-replace2/pages2 (read-only)
    dirs = [ACQ / 'pages', VFARL / 'acquisition/f1-new/pages', ACQ / 'f2-replace2/pages2']
    for d in dirs:
        if not d.exists(): continue
        for pg in sorted(d.glob('train-*.json')):
            if pg.name.endswith('.receipt.json'): continue
            try: blob = json.loads(pg.read_text())
            except Exception: continue
            for item in blob.get('rows', []):
                row = item.get('row', {})
                iid = row.get('instance_id')
                if not iid or iid in seen: continue
                seen[iid] = (item, str(pg))
    return seen


def base_rec(item, fi, fs):
    row = item['row']
    iid = row['instance_id']
    if iid in fi: return None
    repo_field = row.get('repo') or ''
    # repo_field like swesmith/pallets__click.fde47b4b ; derive canonical suffix
    try:
        suffix = repo_field.split('/')[-1].split('.')[0].replace('__', '/')
        if suffix.lower() in EXCLUDED: return None
        # canonical repo for snapshot check: need mapping? Use row's repo suffix? For Smith, base_commit not in row; need snapshot from instance_id?
        # instance_id like pallets__click.fde47b4b.func_basic__...
        # snapshot = canonical_repo :: short_rev ; canonical from suffix, short from instance middle
        parts = iid.split('.')
        # parts[0]=pallets__click, parts[1]=fde47b4b, rest=...
        if len(parts) >= 2:
            canon = parts[0].replace('__', '/')
            short = parts[1]
            # check forbidden snapshots by repo+short
            if snap_hit(canon, short, fs): return None
            # also check full? snap_hit handles prefix
            canon_repo = canon
            short_rev = short
        else:
            canon_repo = suffix
            short_rev = ''
    except Exception:
        canon_repo = ''
        short_rev = ''
    f2p = row.get('FAIL_TO_PASS') or []; p2p = row.get('PASS_TO_PASS') or []
    if isinstance(f2p, str):
        try: f2p = json.loads(f2p)
        except Exception: f2p = []
    if isinstance(p2p, str):
        try: p2p = json.loads(p2p)
        except Exception: p2p = []
    if not f2p or set(f2p) & set(p2p): return None
    if not (row.get('problem_statement') or '').strip(): return None
    patch = row.get('patch') or ''
    files = re.findall(r'diff --git a/(\S+)', patch)
    if any('..' in f for f in files): return None
    total = len(f2p) + len(p2p)
    if total > 200: return None
    tp = row.get('test_patch') or ''
    # test_patch in Smith rows? Check: Smith train rows have no test_patch? They have patch only? Let's handle both.
    si = [l.strip()[:220] for l in tp.splitlines() if STRICT_IS.search(l)][:6] if tp else []
    sc = changed_hits(patch, COPY_STRICT)
    if sc: return None  # strict ground exhausted
    inl = changed_hits(patch, INLINE_SIG)
    if not inl: return None
    strong = [l for l in inl if STRONG_SIG.search(l)]
    weak = [l for l in inl if WEAK_SIG.search(l)]
    if strong: return None  # strong ground exhausted (more/push covered Gym; Smith strong covered by replace/2)
    if not weak: return None
    # inline-is: patch changed lines with STRICT_IS
    patch_is = [l for l in changed_hits(patch, STRICT_IS)][:4]
    clone = changed_hits(patch, CLONE_SIG)
    f2pn = [n for n in f2p if F2P_NAME_SIG.search(n)]
    prob = row.get('problem_statement') or ''
    own_prob = [m.group(0)[:40] for m in OWN_PROB.finditer(prob)][:4]
    own_patch = changed_hits(patch, OWN_PATCH)[:4]
    noise = [l for l in inl if NOISE_SIG.search(l)]
    return {'instance_id': iid, 'repo_field': repo_field, 'canonical_repo': canon_repo, 'short_rev': short_rev,
            'row_idx': item.get('row_idx'), 'F2P': len(f2p), 'P2P': len(p2p), 'total': total,
            'inline_hits': inl[:6], 'inline_n': len(inl),
            'strong_n': 0, 'weak_hits': weak[:4], 'weak_n': len(weak),
            'clone_hits': clone[:4], 'clone_n': len(clone),
            'patch_is_lines': patch_is, 'patch_is_n': len(patch_is),
            'test_is_lines': si, 'test_is_n': len(si),
            'noise_hits': noise[:2], 'noise_n': len(noise),
            'f2p_name_hits': len(f2pn), 'f2p_sample': f2p[:3], 'F2P_list': f2p, 'P2P_list': p2p,
            'own_problem': own_prob, 'own_problem_n': len(own_prob),
            'own_patch': own_patch, 'own_patch_n': len(own_patch),
            'files': files[:4], 'n_files': len(files)}


def prior_named_ids():
    ids = set()
    for rel in ['f2-more/triple-review-log.md', 'f2-inline/triple-review-log.md',
                'f2-recall/triple-review-log.md', 'f2-large/triple-review-log.md',
                'f2-gymv2/triple-review-log.md', 'f2-gymv2b/triple-review-log.md',
                'f2-gymv2c/triple-review-log.md', 'f2-push/triple-review-log.md',
                'f2-push2/triple-review-log.md', 'f2-push3/triple-review-log.md',
                'f2-push4/triple-review-log.md', 'f2-push5/triple-review-log.md',
                'f2-replace/triple-review-log.md', 'f2-replace2/triple-review-log.md']:
        try: txt = (ACQ / rel).read_text()
        except Exception: continue
        for m in re.finditer(r'`([A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+__[A-Za-z0-9]+)`', txt):
            ids.add(m.group(1))
        for m in re.finditer(r'`([A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+)`', txt):
            ids.add(m.group(1))
    # also candidate lists
    for rel in ['candidate-list-40.json', 'f2-push/keep-list.json', 'f2-push2/keep-list.json']:
        try: d = json.loads((ACQ / rel).read_text())
        except Exception: continue
        for t in d.get('candidates', d if isinstance(d, list) else []):
            if isinstance(t, dict) and t.get('instance_id'): ids.add(t['instance_id'])
    return ids


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    OUT.mkdir(parents=True, exist_ok=True)
    if which == 'dedup':
        fi, fs, ff, notes = load_forbidden()
        save(OUT / 'dedup-evidence.json',
             {'date': '2026-10-25', 'scope': 'F600-F2 f2-smith vs common-400+F1-40+F5-old40+f2-round1/2+f2-gymv2-16+f2-gymv2b-5+f2-gymv2c-2+f2-recall-3+f2-inline-1+f2-more-2+f2-push-6+f2-push2-3+f2-push3-2+f2-push4-1+f2-push5-0 disjointness',
              'forbidden_instance_count': len(fi), 'forbidden_snapshot_count': len(fs),
              'forbidden_family_count': len(ff), 'sources': notes,
              'method': 'instance exact; snapshot repo-equality+commit prefix either way; family exact; four repos excluded; bad paths filtered',
              'physical_split': 'train only (Smith-py 77cab905...)'})
        print(f'forbidden {len(fi)}i/{len(fs)}s/{len(ff)}f')
        [print(' ', n) for n in notes]
    elif which == 'scan':
        fi, fs, _, _ = load_forbidden()
        seen = iter_cached_rows()
        print(f"cached unique instances: {len(seen)}")
        hitsW, hitsD = [], []
        n_strict = n_strong = n_noinline = n_noweak = n_total_gt200 = 0
        for iid, (item, pg) in seen.items():
            row = item['row']
            # quick total check for stats
            try:
                f2p = row.get('FAIL_TO_PASS') or []; p2p = row.get('PASS_TO_PASS') or []
                if isinstance(f2p, str): f2p = json.loads(f2p)
                if isinstance(p2p, str): p2p = json.loads(p2p)
                total = len(f2p)+len(p2p)
                if total>200: n_total_gt200+=1
            except Exception: pass
            d = base_rec(item, fi, fs)
            if d is None:
                # stats: why?
                patch = row.get('patch') or ''
                sc = changed_hits(patch, COPY_STRICT)
                if sc: n_strict+=1; continue
                inl = changed_hits(patch, INLINE_SIG)
                if not inl: n_noinline+=1; continue
                strong = [l for l in inl if STRONG_SIG.search(l)]
                if strong: n_strong+=1; continue
                weak = [l for l in inl if WEAK_SIG.search(l)]
                if not weak: n_noweak+=1; continue
                continue
            d['page'] = pg
            support = (d['f2p_name_hits']>0 or d['own_problem_n']>0 or d['test_is_n']>0 or d['own_patch_n']>0 or d['patch_is_n']>0)
            if support:
                hitsW.append(d)
            else:
                hitsD.append(d)
        def rkW(d): return (-d['f2p_name_hits'], -d['own_problem_n'], -d['patch_is_n'], -d['test_is_n'], d['total'])
        def rkD(d): return (d['total'], d['n_files'])
        hitsW.sort(key=rkW); hitsD.sort(key=rkD)
        save(OUT / 'smith-weak-scan.json',
             {'date': '2026-10-25', 'source': 'SWE-bench/SWE-smith-py', 'revision': REV, 'split': 'train',
              'cached_unique': len(seen),
              'stats': {'strict_excluded': n_strict, 'strong_excluded': n_strong, 'no_inline': n_noinline, 'no_weak': n_noweak, 'total_gt200': n_total_gt200},
              'mode': 'smith weak-only (strict excluded, strong_n==0, weak list/dict/set/tuple present, total<=200)',
              'count_W_support': len(hitsW), 'count_D_nosupport': len(hitsD),
              'hitsW': hitsW, 'hitsD': hitsD})
        print(f"weak W(support)={len(hitsW)} D(nosupport)={len(hitsD)} | strict={n_strict} strong={n_strong} noinline={n_noinline} noweak={n_noweak} gt200={n_total_gt200}")
        for d in (hitsW+hitsD)[:60]:
            print(f" t{d['total']:3d} F{d['F2P']:2d} P{d['P2P']:3d} fn{d['f2p_name_hits']} ownP{d['own_problem_n']} ownPa{d['own_patch_n']} pis{d['patch_is_n']} tis{d['test_is_n']} w{d['weak_n']} nf{d['n_files']} {d['instance_id']} :: {d['weak_hits'][0][:90] if d['weak_hits'] else ''}")
    elif which == 'cands':
        s = json.loads((OUT / 'smith-weak-scan.json').read_text())
        named = prior_named_ids()
        union = {}
        for pool_tag, lst in [('W', s['hitsW']), ('D', s['hitsD'])]:
            for h in lst:
                if h['instance_id'] in named: continue
                if h['instance_id'] not in union:
                    h = dict(h); h['pools'] = [pool_tag]; union[h['instance_id']] = h
                elif pool_tag not in union[h['instance_id']]['pools']:
                    union[h['instance_id']]['pools'].append(pool_tag)
        cands = sorted(union.values(), key=lambda d: (0 if 'W' in d['pools'] else 1, d['total'], -d['f2p_name_hits']))
        by_pool = {p: sum(1 for c in cands if p in c['pools']) for p in 'WD'}
        save(OUT / 'smith-candidates.json',
             {'date': '2026-10-25', 'scope': 'f2-smith weak-only W(support)+D(nosupport-lottery), never-individually-reviewed, total<=200, strict+strong excluded',
              'unique_count': len(cands), 'by_pool': by_pool, 'prior_named_excluded': len(named),
              'candidates': cands})
        print(f"smith unique={len(cands)} by_pool={by_pool} (prior-named excluded={len(named)})")
        for d in cands[:80]:
            print(f" t{d['total']:3d} F{d['F2P']:2d} P{d['P2P']:3d} pools={''.join(sorted(d['pools']))} fn{d['f2p_name_hits']} ownP{d['own_problem_n']} pis{d['patch_is_n']} tis{d['test_is_n']} nf{d['n_files']} {d['instance_id']} :: {(d['weak_hits'][0][:80] if d['weak_hits'] else '')}")
    elif which == 'show':
        iid = sys.argv[2]
        seen = iter_cached_rows()
        if iid not in seen:
            print(f"not in cache: {iid}"); sys.exit(1)
        item, pg = seen[iid]
        row = item['row']
        print('==PAGE==', pg, 'row_idx', item.get('row_idx'))
        print('==REPO==', row.get('repo'))
        print('==F2P==', row.get('FAIL_TO_PASS'))
        print('==P2P n==', len(row.get('PASS_TO_PASS') or []), (row.get('PASS_TO_PASS') or [])[:6])
        print('==PROBLEM=='); print((row.get('problem_statement') or '')[:3000])
        print('==PATCH=='); print((row.get('patch') or '')[:8000])
        tp = row.get('test_patch') or ''
        print('==TEST_PATCH len==', len(tp))
        if tp:
            print(tp[:4000])
