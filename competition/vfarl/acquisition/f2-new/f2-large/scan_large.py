"""F2 round 6 (f2-large): large-suite batch basis for the 35 strict-copy rows
excluded by the <=150 suite cap (Gym 7 + V2 28).

Pools (train only, pinned, read-only parquets):
  SWE-Gym/SWE-Gym            rev bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb (2438 train)
  nebius/SWE-rebench-V2      rev 10483de0f50fe5da545942705a76c6150171af7f (Python subset 7243)

Constraints (unchanged, strict):
  zero family/snapshot/instance overlap with common-400 + F1-40 + F5-old40
  + f2-round1/2 + f2-gymv2 16 keeps + f2-gymv2b 5 keeps + f2-gymv2c 2 keeps
  (read-only, never rewrite old dirs); four competition repos excluded;
  F2 triple mandatory (is-identity + shared-mutation visibility + copy timepoint);
  only-== demotes to S; capability-only to R; bad paths filtered;
  Smith-py NOT scanned (frozen); skip-prefix-collision precheck mandatory
  for every candidate; whitelisted problem_statement; JUnit dual conditions;
  six tools; protected paths untouched. replay/.venv dual rounds.

Large-suite batch protocol (new basis, established here):
  for each of the 35 rows: skip precheck first; only skip-CLEAN candidates
  proceed to triple review; only triple-KEEP (is must pass) proceed to
  single-file file-scope validation (run only the target test file(s),
  original matcher unchanged, no assertion shrinking). Multi-file suites
  cannot pass normalize missing_files unless every required file is executed,
  so file-scope eligibility additionally requires single-file required closure
  (all F2P+P2P in one file) or explicit multi-file target list executed together
  (still file-scope, still unmodified matcher). This batch triages all 35 and
  records per-candidate single-file feasibility as the batch basis.

Usage:
  python3 scan_large.py dedup  # build forbidden sets (read-only old files + 23 keeps)
  python3 scan_large.py scan   # list the 35 >150 strict-copy rows (counted, not filtered)
  python3 scan_large.py show <pool> <instance_id>  # full dump for manual triple review
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT / 'competition/vfarl'
ACQ = VFARL / 'acquisition/f2-new'
OUT = ACQ / 'f2-large'
GYM_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/gym-train.parquet')
V2_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/v2-train.parquet')
GYM_REV = 'bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb'
V2_REV = '10483de0f50fe5da545942705a76c6150171af7f'

COPY_SIG = re.compile(r'__copy__|__deepcopy__|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|def copy\(|(?<![\w.])\.copy\(\)')
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
    for kdir in ['f2-gymv2', 'f2-gymv2b', 'f2-gymv2c']:
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


def base_row_ok(r, forb_i):
    """Return 'big' if strict-copy candidate excluded only by suite size, else True/False."""
    iid = r['instance_id']
    if iid in forb_i:
        return False
    repo = (r.get('repo') or '').strip()
    if repo.lower() in EXCLUDED_REPOS:
        return False
    patch = r.get('patch') or ''
    chg = [l for l in patch.splitlines()
           if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++', '---'))]
    hits = [l.strip()[:200] for l in chg if COPY_SIG.search(l)]
    if not hits:
        return False
    f2p = r.get('FAIL_TO_PASS') or []
    p2p = r.get('PASS_TO_PASS') or []
    if not f2p or set(f2p) & set(p2p):
        return False
    if not (r.get('problem_statement') or '').strip():
        return False
    files = re.findall(r'diff --git a/(\S+)', patch)
    if any('..' in f for f in files):
        return False
    total = len(f2p) + len(p2p)
    if total <= 150:
        return False  # not part of the large-suite batch
    return True


def cmd_dedup():
    forb_i, forb_s, forb_f, notes = load_forbidden()
    save(OUT / 'dedup-evidence.json', {
        'date': '2026-10-25',
        'scope': 'F600-F2 f2-large vs common-400 + F1-40 + F5-old40 + f2-round1/2 + f2-gymv2-16 + f2-gymv2b-5 + f2-gymv2c-2 disjointness (Smith frozen, not scanned)',
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


def cmd_scan():
    import pyarrow.parquet as pq
    forb_i, forb_s, _, _ = load_forbidden()
    rows = pq.read_table(str(GYM_PARQUET)).to_pylist()
    gym_big = []
    for idx, r in enumerate(rows):
        if base_row_ok(r, forb_i) is not True:
            continue
        patch = r.get('patch') or ''
        chg = [l for l in patch.splitlines()
               if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++', '---'))]
        hits = [l.strip()[:200] for l in chg if COPY_SIG.search(l)]
        f2p = r.get('FAIL_TO_PASS') or []
        p2p = r.get('PASS_TO_PASS') or []
        files = re.findall(r'diff --git a/(\S+)', patch)
        req_files = sorted({n.split('::')[0] for n in list(f2p) + list(p2p) if '::' in n})
        gym_big.append({'instance_id': r['instance_id'], 'pool': 'gym', 'parquet_idx': idx,
                        'repo': (r.get('repo') or '').strip(), 'base_commit': r.get('base_commit'),
                        'F2P': len(f2p), 'P2P': len(p2p), 'total': len(f2p) + len(p2p),
                        'files': files[:8], 'required_files': req_files,
                        'single_file_required_closure': len(req_files) == 1,
                        'F2P_list': f2p, 'P2P_sample': p2p[:4], 'hits': hits[:6],
                        'snapshot_collision': snap_hit((r.get('repo') or '').strip(), r.get('base_commit') or '', forb_s)})
    gym_big.sort(key=lambda d: (d['total'], d['F2P']))
    f = pq.ParquetFile(str(V2_PARQUET))
    v2_big = []
    n_py = 0
    for rg in range(f.num_row_groups):
        tbl = f.read_row_group(rg, columns=['instance_id', 'language', 'repo', 'base_commit',
                                            'patch', 'problem_statement', 'FAIL_TO_PASS', 'PASS_TO_PASS'])
        for r in tbl.to_pylist():
            if (r.get('language') or '').lower() != 'python':
                continue
            n_py += 1
            if base_row_ok(r, forb_i) is not True:
                continue
            patch = r.get('patch') or ''
            chg = [l for l in patch.splitlines()
                   if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++', '---'))]
            hits = [l.strip()[:200] for l in chg if COPY_SIG.search(l)]
            f2p = r.get('FAIL_TO_PASS') or []
            p2p = r.get('PASS_TO_PASS') or []
            files = re.findall(r'diff --git a/(\S+)', patch)
            req_files = sorted({n.split('::')[0] for n in list(f2p) + list(p2p) if '::' in n})
            v2_big.append({'instance_id': r['instance_id'], 'pool': 'v2', 'row_group': rg,
                           'repo': (r.get('repo') or '').strip(), 'base_commit': r.get('base_commit'),
                           'F2P': len(f2p), 'P2P': len(p2p), 'total': len(f2p) + len(p2p),
                           'files': files[:8], 'required_files': req_files,
                           'single_file_required_closure': len(req_files) == 1,
                           'F2P_list': f2p, 'P2P_sample': p2p[:4], 'hits': hits[:6],
                           'snapshot_collision': snap_hit((r.get('repo') or '').strip(), r.get('base_commit') or '', forb_s)})
    v2_big.sort(key=lambda d: (d['total'], d['F2P']))
    save(OUT / 'gym-large-scan.json', {
        'date': '2026-10-25', 'source': 'SWE-Gym/SWE-Gym', 'revision': GYM_REV,
        'split': 'train', 'rows_scanned': len(rows),
        'parquet_bytes': GYM_PARQUET.stat().st_size,
        'parquet_sha256': hashlib.sha256(GYM_PARQUET.read_bytes()).hexdigest(),
        'suite_cap': 150, 'mode': 'large_suite_only_total_gt_150_strict_copy',
        'count': len(gym_big), 'hits': gym_big})
    save(OUT / 'v2-large-scan.json', {
        'date': '2026-10-25', 'source': 'nebius/SWE-rebench-V2', 'revision': V2_REV,
        'split': 'train', 'language': 'python', 'python_rows_scanned': n_py,
        'parquet_bytes': V2_PARQUET.stat().st_size,
        'parquet_sha256': hashlib.sha256(V2_PARQUET.read_bytes()).hexdigest(),
        'suite_cap': 150, 'mode': 'large_suite_only_total_gt_150_strict_copy',
        'count': len(v2_big), 'hits': v2_big})
    print(f'gym rows={len(rows)} large_hits={len(gym_big)}')
    for d in gym_big:
        print(f"  {d['total']:5d} F2P={d['F2P']:3d} P2P={d['P2P']:5d} singlefile={d['single_file_required_closure']} {d['instance_id']}")
    print(f'v2 python rows={n_py} large_hits={len(v2_big)}')
    for d in v2_big:
        print(f"  {d['total']:5d} F2P={d['F2P']:3d} P2P={d['P2P']:5d} singlefile={d['single_file_required_closure']} {d['instance_id']} {d['repo']}")


def cmd_show(pool, iid):
    import pyarrow.parquet as pq
    if pool == 'gym':
        rows = pq.read_table(str(GYM_PARQUET)).to_pylist()
        hit = [r for r in rows if r['instance_id'] == iid]
    else:
        f = pq.ParquetFile(str(V2_PARQUET))
        hit = []
        for rg in range(f.num_row_groups):
            for r in f.read_row_group(rg).to_pylist():
                if r['instance_id'] == iid:
                    hit.append(r)
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
    elif which == 'show':
        cmd_show(sys.argv[2], sys.argv[3])
    else:
        print('unknown cmd')
