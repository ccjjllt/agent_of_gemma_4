"""F2-replace batch builder: from hand-reviewed selection to dual-replayed tasks.

Usage:
  python3 build.py scan            # strict changed-line copy-signal scan over cached pages
  python3 build.py shortlist       # print shortlist with full patches for manual triple review
  python3 build.py prepare         # for SELECTED tasks: task.json, acquire, inverse-check, recipe, source-prep
  python3 build.py replay          # dual official replay for prepared tasks
  python3 build.py report          # eligible count / gap via normalize

Selection lives in SELECTION below (hand triple-reviewed only).
"""
import hashlib, json, re, subprocess, sys, tarfile, urllib.request, difflib
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f2-new'
OUT = ACQ / 'f2-replace'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f2-replace'
REV = '77cab9055d42ab4a5c25c89a8f937096db13558e'
PAGES = ACQ / 'pages'

# Hand triple-reviewed selection: instance_id -> dict(family, symbol, contract,
#   obs (inline python asserting is-identity + shared-mutation visibility + timepoint),
#   read (filepath,start,end), upstream_repo, upstream_short_rev)
# Populated incrementally as manual review completes. Empty until reviewed.
SELECTION = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')


def digest(b):
    return hashlib.sha256(b).hexdigest()


COPY_SIG = re.compile(r'__copy__|__deepcopy__|copy\.copy|copy\.deepcopy|deepcopy\(|model_copy|def copy\(|(?<![\w.])\.copy\(\)')
EXCLUDED_REPOS = {"fastapi/fastapi", "textualize/rich", "psf/requests", "encode/httpx"}

import importlib.util
_spec = importlib.util.spec_from_file_location('f2_fetch_scan', str(ACQ / 'fetch_scan.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ALLOWED_REPO_KEYS = set(_mod.TARGETS)


def load_used():
    man = json.loads((ROOT / 'competition/vfarl/s600/course-manifest.json').read_text())
    used = set(man['courses']['common']['instance_ids']) | set(man['courses']['specialist']['instance_ids'])
    for rel in ['competition/vfarl/acquisition/supplemental/catalog.json',
                'competition/vfarl/pilot96/catalog.json']:
        try:
            c = json.loads((ROOT / rel).read_text())
            used |= {t['instance_id'] for t in c['tasks']}
        except Exception:
            pass
    base = json.loads((ACQ / 'dedup-baseline.json').read_text())
    used |= set(base['forbidden_instance_ids'])
    return used, set(base['forbidden_snapshot_group_ids']), set(base['forbidden_repair_family_ids'])


def iter_rows():
    for pg in sorted(PAGES.glob('train-*.json')):
        if pg.name.endswith('.receipt.json'):
            continue
        d = json.loads(pg.read_text())
        for item in d.get('rows', []):
            yield item, pg


def cmd_scan():
    used, _, _ = load_used()
    out = []
    for item, pg in iter_rows():
        row = item['row']
        iid = row['instance_id']
        if iid in used:
            continue
        repo_key = row['repo'].split('/')[-1]
        if repo_key not in ALLOWED_REPO_KEYS:
            continue
        try:
            suffix = row['repo'].split('/')[-1].split('.')[0].replace('__', '/')
            if '/' in suffix and suffix.lower() in EXCLUDED_REPOS:
                continue
        except Exception:
            pass
        patch = row.get('patch', '') or ''
        chg = [l for l in patch.splitlines()
               if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++', '---'))]
        hits = [l.strip()[:200] for l in chg if COPY_SIG.search(l)]
        if not hits:
            continue
        f2p = row.get('FAIL_TO_PASS') or []
        p2p = row.get('PASS_TO_PASS') or []
        if isinstance(f2p, str):
            f2p = json.loads(f2p)
        if isinstance(p2p, str):
            p2p = json.loads(p2p)
        if not f2p or not p2p or set(f2p) & set(p2p):
            continue
        if not (row.get('problem_statement') or '').strip():
            continue
        files = re.findall(r'diff --git a/(\S+)', patch)
        if any('..' in f for f in files):
            continue
        out.append({'instance_id': iid, 'row_idx': item['row_idx'], 'page': pg.name,
                    'F2P': len(f2p), 'P2P': len(p2p), 'files': files[:4],
                    'F2P_sample': f2p[:3], 'hits': hits[:6]})
    out.sort(key=lambda d: (d['P2P'], d['F2P']))
    save(OUT / 'strict-scan.json', out)
    print(f'strict_hits={len(out)}')
    for d in out:
        print(d['P2P'], d['F2P'], d['instance_id'], d['files'])
        for h in d['hits'][:3]:
            print('   ', h)


def cmd_shortlist():
    docs = json.loads((OUT / 'strict-scan.json').read_text())
    rows = {}
    for item, _ in iter_rows():
        rows[item['row']['instance_id']] = item['row']
    for d in docs:
        row = rows[d['instance_id']]
        print('=' * 100)
        print(d['instance_id'], 'row', d['row_idx'], 'F2P=', d['F2P'], 'P2P=', d['P2P'], 'files=', d['files'])
        print('F2P:', row.get('FAIL_TO_PASS'))
        print('--- problem ---')
        print((row.get('problem_statement') or '')[:1500])
        print('--- patch FULL ---')
        print((row.get('patch') or '')[:5000])


def inverse_edits(patch):
    edits = []
    old, new = [], []
    for line in patch.splitlines(keepends=True):
        if line.startswith('@@ '):
            if old or new:
                edits.append((''.join(old), ''.join(new)))
            old, new = [], []
        elif line.startswith(('diff --git ', 'index ', '--- ', '+++ ', '\\')):
            continue
        elif line.startswith(' '):
            old.append(line[1:])
            new.append(line[1:])
        elif line.startswith('+'):
            old.append(line[1:])
        elif line.startswith('-'):
            new.append(line[1:])
    if old or new:
        edits.append((''.join(old), ''.join(new)))
    return [e for e in edits if e[0] != e[1]]


def acquire(repo, rev, dest):
    dest.mkdir(parents=True, exist_ok=True)
    arch = dest.with_suffix('.tgz')
    url = f'https://codeload.github.com/{repo}/tar.gz/{rev}'
    if not arch.exists():
        print(f' downloading {repo}@{rev}', flush=True)
        with urllib.request.urlopen(url, timeout=180) as r:
            arch.write_bytes(r.read())
    with tarfile.open(arch) as bundle:
        members = bundle.getmembers()
        top = members[0].name.split('/')[0]
        for m in members:
            parts = Path(m.name).parts
            if not parts or parts[0] != top or '..' in parts:
                continue
            rel = Path(*parts[1:])
            if not rel.parts:
                continue
            tgt = dest / rel
            if m.isdir():
                tgt.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                payload = bundle.extractfile(m).read()
                if tgt.exists() and tgt.read_bytes() != payload:
                    raise ValueError(f'altered {tgt}')
                tgt.parent.mkdir(parents=True, exist_ok=True)
                tgt.write_bytes(payload)
    return {'repo': repo, 'requested_revision': rev, 'resolved_revision': rev,
            'archive_path': str(arch.relative_to(ROOT)), 'archive_sha256': digest(arch.read_bytes()),
            'download_url': url, 'immutable_url': url, 'commit_resolution': None}


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'scan'
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    if which == 'scan':
        cmd_scan()
    elif which == 'shortlist':
        cmd_shortlist()
    else:
        print('prepare/replay/report implemented per-task after manual review')
