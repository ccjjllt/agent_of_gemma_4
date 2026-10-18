"""Download the authorized fixed-revision tool split; never interpret task bodies."""
import concurrent.futures, hashlib, json, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REV = '08e109b4a59eaeebf80e4675cd125d42e7ac99a4'

def fetch(row):
    name = Path(row['path']).name
    target = HERE / 'shards' / name
    target.parent.mkdir(exist_ok=True)
    if name == 'tool-00004-of-00008.parquet':
        target = HERE.parents[2] / 'training/artifacts/external/swe-smith' / name
    if not target.exists():
        url = f'https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/resolve/{REV}/{row["path"]}'
        temporary = target.with_suffix('.part')
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        temporary.rename(target)
    digest = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
    if target.stat().st_size != row['size'] or digest != row['lfs']['oid']:
        raise ValueError('Pinned artifact fingerprint mismatch: ' + name)
    result = {'path':str(target), 'bytes':target.stat().st_size, 'sha256':digest}
    print(json.dumps(result), flush=True)
    return result

if __name__ == '__main__':
    rows = [r for r in json.loads((HERE / 'hf-files.json').read_text()) if '/tool-' in r['path']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, rows))
    (HERE / 'shard-manifest.json').write_text(json.dumps(results, indent=2) + '\n')
