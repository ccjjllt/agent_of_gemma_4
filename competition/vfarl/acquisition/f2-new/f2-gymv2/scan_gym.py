"""F2 round 3 (f2-gymv2): scan SWE-Gym + SWE-rebench-V2 for F2 copy/ownership tasks.

Pools (train only, pinned):
  SWE-Gym/SWE-Gym            rev bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb (2438 train)
  nebius/SWE-rebench-V2      rev 10483de0f50fe5da545942705a76c6150171af7f (Python subset of 32079)

Constraints (unchanged): zero family/snapshot/instance overlap with
common-400 + F1-40 + F5-old40 (+ F2 round1/round2 claimed, read-only);
four repos excluded (fastapi/fastapi, textualize/rich, psf/requests, encode/httpx);
F2 triple (is-identity + shared-mutation visibility + copy timepoint) mandatory;
small suites preferred, real-task suites up to 150 tests allowed;
skip-prefix-collision precheck mandatory (drop snapshot on collision);
whitelisted problem_statement, JUnit dual conditions, six tools, protected paths.

Usage:
  python3 scan_gym.py scan      # strict changed-line copy scan over Gym parquet
  python3 scan_gym.py show      # full patch/problem/F2P dump for manual triple review
  python3 scan_gym.py dedup     # build forbidden sets from read-only old files
"""
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
VFARL = ROOT / 'competition/vfarl'
ACQ = VFARL / 'acquisition/f2-new'
OUT = ACQ / 'f2-gymv2'
GYM_PARQUET = Path('/private/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/opencode/gym-train.parquet')
GYM_REV = 'bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb'
V2_REV = '10483de0f50fe5da545942705a76c6150171af7f'

COPY_SIG = re.compile(r'__copy__|__deepcopy__|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|def copy\(|(?<![\w.])\.copy\(\)')
EXCLUDED_REPOS = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def load_forbidden():
    """Read-only load of all previously claimed identity sets. Never writes to old dirs."""
    forb_i, forb_s, forb_f = set(), set(), set()
    notes = []
    base = json.loads((ACQ / 'dedup-baseline.json').read_text())
    forb_i |= set(base['forbidden_instance_ids'])
    forb_s |= set(base.get('forbidden_snapshot_group_ids', []))
    forb_f |= set(base['forbidden_repair_family_ids'])
    notes.append(f"common-400: {len(base['forbidden_instance_ids'])}i/{len(base.get('forbidden_snapshot_group_ids', []))}s/{len(base['forbidden_repair_family_ids'])}f")
    f1 = json.loads((VFARL / 'acquisition/f1-new/reviewed-candidates.json').read_text())
    for t in f1.get('tasks', []):
        forb_i.add(t['instance_id'])
        if t.get('snapshot_group_id'):
            forb_s.add(t['snapshot_group_id'])
        if t.get('repair_family_id'):
            forb_f.add(t['repair_family_id'])
    notes.append(f"F1-40: {len(f1.get('tasks', []))} tasks")
    f5 = json.loads((VFARL / 'acquisition/f5-new/acquired-40.json').read_text())
    for t in f5.get('tasks', []):
        forb_i.add(t.get('instance_id', t.get('canonical_task_id', '')))
        for key in ('snapshot_group_id', 'snapshot_id'):
            if t.get(key):
                forb_s.add(t[key])
        for key in ('repair_family_id', 'family_id'):
            if t.get(key):
                forb_f.add(t[key])
    notes.append(f"F5-old40: {len(f5.get('tasks', []))} tasks")
    for rel in ['f2-replace/catalog.json', 'f2-replace/reviewed-candidates.json',
                'candidate-list-40.json', 'reviewed-candidates.json', 'catalog.json']:
        p = ACQ / rel if not rel.startswith('f2-') else ACQ / rel
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
        notes.append(f"f2prior:{rel}: {len(d.get('tasks', d.get('candidates', [])))} entries")
    bans = ['seperman/deepdiff::ed252022', 'pallets/click::fde47b4b',
            'pydantic/pydantic::acb0f10f', 'stanfordnlp/dspy::651a4c71',
            'sunpy/sunpy::f8edfd5c', 'mido/mido::a0158ff9',
            'cantools/cantools::0c6a7871']
    forb_s |= set(bans)
    notes.append(f"explicit round1 snapshot bans: {len(bans)}")
    return forb_i, forb_s, forb_f, notes


def snap_hit(canonical_repo, commit, forb_s):
    """Snapshot overlap: same repo (case-insensitive) and commit prefix either way."""
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
        'scope': 'F600-F2 f2-gymv2 vs common-400 + F1-40 + F5-old40 + f2-round1/round2 disjointness',
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
    t = pq.read_table(str(GYM_PARQUET))
    rows = t.to_pylist()
    out = []
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
            continue
        snap = snap_hit(repo, r.get('base_commit') or '', forb_s)
        out.append({'instance_id': iid, 'parquet_idx': idx, 'repo': repo,
                    'base_commit': r.get('base_commit'), 'F2P': len(f2p), 'P2P': len(p2p),
                    'total': total, 'files': files[:6], 'F2P_list': f2p,
                    'P2P_sample': p2p[:4], 'hits': hits[:6],
                    'snapshot_collision': snap})
    out.sort(key=lambda d: (d['total'], d['F2P']))
    save(OUT / 'gym-strict-scan.json', {
        'date': '2026-10-25', 'source': 'SWE-Gym/SWE-Gym', 'revision': GYM_REV,
        'split': 'train', 'rows_scanned': len(rows),
        'parquet_bytes': GYM_PARQUET.stat().st_size,
        'parquet_sha256': hashlib.sha256(GYM_PARQUET.read_bytes()).hexdigest(),
        'suite_cap': 150, 'hits': out})
    print(f'gym rows={len(rows)} strict_hits={len(out)}')
    for d in out:
        flag = 'SNAPCOLLIDE ' if d['snapshot_collision'] else ''
        print(f"{flag}{d['total']:4d} F2P={d['F2P']:3d} P2P={d['P2P']:4d} {d['instance_id']} {d['files'][:2]}")
        for h in d['hits'][:2]:
            print('    ', h)


def cmd_show():
    import pyarrow.parquet as pq
    scan = json.loads((OUT / 'gym-strict-scan.json').read_text())
    t = pq.read_table(str(GYM_PARQUET))
    rows = t.to_pylist()
    want = sys.argv[2:] if len(sys.argv) > 2 else None
    for d in scan['hits']:
        if want and d['instance_id'] not in want:
            continue
        r = rows[d['parquet_idx']]
        print('=' * 100)
        print(d['instance_id'], d['repo'], (d['base_commit'] or '')[:12],
              'F2P=', d['F2P'], 'P2P=', d['P2P'], 'files=', d['files'])
        if d['snapshot_collision']:
            print('SNAPSHOT COLLISION:', d['snapshot_collision'])
        print('F2P:', json.dumps(d['F2P_list'], ensure_ascii=False)[:1200])
        print('--- problem (first 1800 chars) ---')
        print((r.get('problem_statement') or '')[:1800])
        print('--- patch FULL ---')
        print((r.get('patch') or '')[:6000])
        if want and len(want) == 1:
            print('--- test_patch (first 3000 chars) ---')
            print((r.get('test_patch') or '')[:3000])


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    OUT.mkdir(parents=True, exist_ok=True)
    if which == 'dedup':
        cmd_dedup()
    elif which == 'scan':
        cmd_scan()
    elif which == 'show':
        cmd_show()
    else:
        print('unknown cmd')
