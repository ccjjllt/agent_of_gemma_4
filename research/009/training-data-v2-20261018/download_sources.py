"""Download fixed train-only buggy snapshots and retain source provenance."""
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def fetch(row, excluded):
    if row['repo'].lower() in excluded:
        raise ValueError('Reserved repository is not a training source')
    iid = row['instance_id']
    mirror = 'swesmith/' + iid.split('.pr_')[0]
    raw = subprocess.run(['git', 'ls-remote', 'https://github.com/' + mirror + '.git',
                          'refs/heads/' + iid], capture_output=True, text=True, check=True).stdout
    sha = raw.split()[0]
    if len(sha) != 40:
        raise ValueError('Missing exact buggy branch commit')
    url = 'https://codeload.github.com/' + mirror + '/tar.gz/' + sha
    dest = HERE / 'sources'
    dest.mkdir(exist_ok=True)
    key = row.get('asset_key', row['repo'].split('/')[-1])
    archive = dest / (key + '.tar.gz')
    folder = dest / key
    folder.mkdir(exist_ok=True)
    if key == 'alive-progress':
        files = []
        for rel in ['alive_progress/core/configuration.py', 'LICENSE']:
            source_url = 'https://raw.githubusercontent.com/' + mirror + '/' + sha + '/' + rel
            with urllib.request.urlopen(source_url, timeout=60) as response:
                data = response.read(200000)
            target = folder / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files.append({'path': str(target.relative_to(ROOT)), 'url': source_url,
                          'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
        return dict(row, mirror=mirror, branch=iid, git_ls_remote_stdout=raw, buggy_commit=sha,
                    source_root=str(folder.relative_to(ROOT)), files=files,
                    licenses=[files[-1]], extraction_scope='fixed-commit configuration module only; full archive exceeded 20MB and was not saved')
    if archive.exists():
        content = archive.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=90) as response:
            content = response.read(20_000_001)
    if len(content) > 20_000_000:
        raise ValueError('Source archive exceeds this CPU preparation limit')
    archive.write_bytes(content)
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            parts = PurePosixPath(member.name).parts[1:]
            if not parts or '..' in parts or member.issym() or member.islnk():
                continue
            if any(p in ['.git', 'tests', 'test', 'test-data', 'docs', '.github'] for p in parts):
                continue
            if parts[-1].startswith('test_') or parts[-1].endswith('_test.py'):
                continue
            path = folder.joinpath(*parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(tf.extractfile(member).read())
    licenses = [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in folder.iterdir() if p.name.lower().startswith(('license', 'copying'))]
    return dict(row, mirror=mirror, branch=iid, git_ls_remote_stdout=raw, buggy_commit=sha,
                archive_url=url, archive_path=str(archive.relative_to(ROOT)), archive_bytes=len(content),
                archive_sha256=hashlib.sha256(content).hexdigest(),
                source_root=str(folder.relative_to(ROOT)), licenses=licenses,
                extraction_scope='production/config/resources; tests/docs/symlinks excluded')

def main():
    usage = json.loads((HERE / 'data-usage.json').read_text())
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(lambda row: fetch(row, set(usage['excluded_repositories'])), usage['selected']):
            results.append(result)
            (HERE / 'source-manifest.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
            print(result['repo'], result['buggy_commit'], result.get('archive_bytes', 'selected files'), flush=True)

if __name__ == '__main__':
    main()

