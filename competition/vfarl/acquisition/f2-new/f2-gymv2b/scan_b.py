"""F2 round 4 (f2-gymv2b): continue to 40 (have 6 eligible, need 34 more).

Pools (train only, pinned):
  SWE-Gym/SWE-Gym            rev bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb (2438 train)
  nebius/SWE-rebench-V2      rev 10483de0f50fe5da545942705a76c6150171af7f (Python subset 7243)

Constraints (unchanged from f2-gymv2):
  zero family/snapshot/instance overlap with common-400 + F1-40 + F5-old40
  + f2-round1/2 + f2-gymv2 16 keeps (read-only, never rewrite old dirs);
  four repos excluded; F2 triple mandatory (is + shared-mutation + timepoint);
  only-== demotes to S; capability-only to R; bad paths filtered;
  small suites prioritized, <=150 allowed, >150 needs batch basis;
  skip-prefix-collision precheck mandatory; whitelisted problem_statement;
  JUnit dual conditions; six tools; protected paths untouched.
  replay/.venv dual rounds.

Usage:
  python3 scan_b.py scan   # strict + broad scans over Gym+V2 parquets
  python3 scan_b.py show   # dump rows for manual triple review
  python3 scan_b.py dedup  # build forbidden sets (read-only old files + f2-gymv2 keeps)
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT / 'competition/vfarl'
ACQ = VFARL / 'acquisition/f2-new'
OUT = ACQ / 'f2-gymv2b'
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
    # f2-gymv2 16 keeps (read-only, never rewrite): instances + snapshots + families
    g2keep = json.loads((ACQ / 'f2-gymv2/keep-list.json').read_text())
    g2fam = json.loads((ACQ / 'f2-gymv2/keep-families.json').read_text())
    for k in g2keep:
        forb_i.add(k['instance_id'])
        forb_s.add(f"{k['repo']}::{k['base_commit']}")
    forb_f |= set(g2fam.values())
    notes.append(f"f2-gymv2: {len(g2keep)} keeps (instances+snapshots+{len(g2fam)} families)")
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


def cmd_dedup():
    forb_i, forb_s, forb_f, notes = load_forbidden()
    save(OUT / 'dedup-evidence.json', {
        'date': '2026-10-25',
        'scope': 'F600-F2 f2-gymv2b vs common-400 + F1-40 + F5-old40 + f2-round1/2 + f2-gymv2-16 disjointness',
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
    # Gym strict
    t = pq.read_table(str(GYM_PARQUET))
    rows = t.to_pylist()
    gym_out = []
    gym_big = 0
    for idx, r in enumerate(rows):
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
        if any('..' in f for f in files):
            continue
        total = len(f2p) + len(p2p)
        if total > 150:
            gym_big += 1
            continue
        snap = snap_hit(repo, r.get('base_commit') or '', forb_s)
        gym_out.append({'instance_id': iid, 'parquet_idx': idx, 'repo': repo,
                        'base_commit': r.get('base_commit'), 'F2P': len(f2p), 'P2P': len(p2p),
                        'total': total, 'files': files[:6], 'F2P_list': f2p,
                        'P2P_sample': p2p[:4], 'hits': hits[:6],
                        'snapshot_collision': snap})
    gym_out.sort(key=lambda d: (d['total'], d['F2P']))
    # V2 strict (python only)
    f = pq.ParquetFile(str(V2_PARQUET))
    v2_out = []
    v2_big = 0
    n_py = 0
    for rg in range(f.num_row_groups):
        tbl = f.read_row_group(rg, columns=['instance_id', 'language', 'repo', 'base_commit',
                                            'patch', 'problem_statement',
                                            'FAIL_TO_PASS', 'PASS_TO_PASS'])
        for r in tbl.to_pylist():
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
                v2_big += 1
                continue
            snap = snap_hit(repo, r.get('base_commit') or '', forb_s)
            v2_out.append({'instance_id': iid, 'repo': repo, 'base_commit': r.get('base_commit'),
                           'row_group': rg, 'F2P': len(f2p), 'P2P': len(p2p), 'total': total,
                           'files': files[:6], 'F2P_list': f2p, 'P2P_sample': p2p[:4],
                           'hits': hits[:6], 'snapshot_collision': snap})
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


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    OUT.mkdir(parents=True, exist_ok=True)
    if which == 'dedup':
        cmd_dedup()
    elif which == 'scan':
        cmd_scan()
    else:
        print('unknown cmd')
