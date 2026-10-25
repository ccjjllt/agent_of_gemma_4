"""Stage 1: resolve commits, download buggy+upstream sources, verify inverse edits.

Output: acquisition/f3-new/f3-replace/stage1-manifest.json
Workspaces: replay/artifacts/f3-replace/<key>-source (+ .tgz)
Upstreams: replay/artifacts/f3-replace/upstream-<tag>/ (+ .tgz)
"""
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path('/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4')
ACQ = ROOT / 'competition/vfarl/acquisition/f3-new'
OUT = ACQ / 'f3-replace'
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/f3-replace'
REV = '77cab9055d42ab4a5c25c89a8f937096db13558e'

sys.path.insert(0, str(ACQ))
from f3replace_pool import TASKS, ACTIVE_TASKS


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inverse_edits(patch: str):
    edits, old, new = [], [], []
    for line in patch.splitlines(keepends=True):
        if line.startswith('@@ '):
            if old or new:
                edits.append((''.join(old), ''.join(new)))
            old, new = [], []
        elif line.startswith(('diff --git ', 'index ', '--- ', '+++ ', '\\')):
            continue
        elif line.startswith(' '):
            old.append(line[1:]); new.append(line[1:])
        elif line.startswith('+'):
            old.append(line[1:])
        elif line.startswith('-'):
            new.append(line[1:])
    if old or new:
        edits.append((''.join(old), ''.join(new)))
    return [e for e in edits if e[0] != e[1]]


def patch_files(patch: str):
    return [m.group(1) for m in re.finditer(r'diff --git a/(\S+) b/\S+', patch)]


def acquire(repo: str, rev: str, dest: Path, label: str):
    dest.mkdir(parents=True, exist_ok=True)
    arch = dest.with_suffix('.tgz')
    url = f'https://codeload.github.com/{repo}/tar.gz/{rev}'
    if not arch.exists():
        print(f'  downloading {label} {repo}@{rev}', flush=True)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=180) as r:
                    arch.write_bytes(r.read())
                break
            except Exception as e:
                print(f'    attempt {attempt} failed: {e}', flush=True)
                if attempt == 4:
                    raise
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


def full_sha(canonical: str, short: str, cache: dict) -> str:
    if short in cache:
        return cache[short]
    url = f'https://github.com/{canonical}/commit/{short}'
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                html = r.read().decode('utf-8', 'replace')
            m = re.search(r'commit/([0-9a-f]{40})', html)
            if m:
                cache[short] = m.group(1)
                return m.group(1)
        except Exception as e:
            print(f'  full_sha attempt {attempt} failed: {e}', flush=True)
    raise RuntimeError(f'cannot resolve full sha for {canonical}@{short}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    sha_cache_path = OUT / 'upstream-sha-cache.json'
    sha_cache = json.loads(sha_cache_path.read_text()) if sha_cache_path.exists() else {}

    # locate rows
    pages = [p for p in (ACQ / 'artifacts/pages').glob('*.json') if not p.name.endswith('.receipt.json')]
    row_by_id, page_by_id = {}, {}
    for pg in pages:
        d = json.load(open(pg))
        for item in d['rows']:
            iid = item['row']['instance_id']
            if iid not in row_by_id:
                row_by_id[iid] = item['row']
                page_by_id[iid] = pg

    idx = {json.loads(l)['instance_id']: json.loads(l)
           for l in open(ROOT / 'competition/vfarl/acquisition/general/smith-task-index.jsonl')}

    upstreams = {}
    manifest = []
    for key, iid, repo, canonical, upstream_short, symbol, contract in ACTIVE_TASKS():
        print(f'=== {key} {iid} ===', flush=True)
        row = row_by_id.get(iid)
        assert row, f'row missing {iid}'
        pg = page_by_id[iid]
        remote = subprocess.check_output(
            ['git', 'ls-remote', f'https://github.com/{repo}.git', f'refs/heads/{iid}'],
            text=True).strip()
        commit, ref = remote.split()
        assert len(commit) == 40, remote
        ws = ART / (key + '-source')
        buggy = acquire(repo, commit, ws, key)
        tag = canonical.replace('/', '__') + '.' + upstream_short
        if tag not in upstreams:
            full = full_sha(canonical, upstream_short, sha_cache)
            sha_cache_path.write_text(json.dumps(sha_cache, indent=2) + '\n')
            up_dest = ART / f'upstream-{tag}'
            up = acquire(canonical, upstream_short, up_dest, 'upstream-' + tag)
            up['requested_revision'] = upstream_short
            up['resolved_revision'] = full
            upstreams[tag] = {'dir': str(up_dest.relative_to(ROOT)), 'meta': up, 'full': full,
                              'canonical': canonical, 'short': upstream_short}
        up_info = upstreams[tag]
        up_dest = ROOT / up_info['dir']
        # verify inverse edits reconstruct upstream for every patched file
        edits = inverse_edits(row['patch'])
        assert edits, f'no edits {iid}'
        files = patch_files(row['patch'])
        recon = {}
        for f in files:
            buggy_text = (ws / f).read_text()
            cur = buggy_text
            applied = 0
            for o, n in edits:
                if cur.count(o) == 1:
                    cur = cur.replace(o, n)
                    applied += 1
            up_text = (up_dest / f).read_text()
            recon[f] = {'edits_applied': applied, 'edits_total': len(edits),
                        'matches_upstream': cur == up_text,
                        'buggy_bytes': len(buggy_text.encode()),
                        'upstream_bytes': len(up_text.encode())}
        ok_all = all(v['matches_upstream'] for v in recon.values())
        print(f'  exec={commit[:12]} upstream_full={up_info["full"][:12]} files={files} recon_ok={ok_all}', flush=True)
        manifest.append({'key': key, 'instance_id': iid, 'repo': repo, 'canonical': canonical,
                         'upstream_short': upstream_short, 'upstream_full': up_info['full'],
                         'symbol': symbol, 'contract': contract, 'page': str(pg.relative_to(ROOT)),
                         'page_sha256': sha(pg), 'execution_commit': commit, 'git_ls_remote': remote,
                         'buggy_source': buggy, 'upstream': up_info['meta'],
                         'edits_total': len(edits), 'files': files, 'reconstruction': recon,
                         'reconstruction_ok': ok_all, 'F2P': row['FAIL_TO_PASS'], 'P2P': row['PASS_TO_PASS'],
                         'problem_statement': row['problem_statement'],
                         'problem_statement_sha256': digest(row['problem_statement'].encode()),
                         'smith_index': idx.get(iid)})
    (OUT / 'stage1-manifest.json').write_text(
        json.dumps({'tasks': manifest, 'upstreams': upstreams}, ensure_ascii=False, indent=2) + '\n')
    bad = [m['key'] for m in manifest if not m['reconstruction_ok']]
    print(f'\nSTAGE1 DONE tasks={len(manifest)} recon_failures={bad}', flush=True)


if __name__ == '__main__':
    main()
