#!/usr/bin/env python3
"""Acquire a bounded, train-only Open-SWE pilot after metadata-only selection.

Never reads trajectory columns during selection. Content comes from exact
trajectory-ID queries only after the repository/issue policy has been frozen.
The dataset server is accepted only while main equals the pinned revision.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.request

PROJECT = Path(__file__).resolve().parents[3]
OUTPUT = PROJECT / 'research/016/data/open-swe'
DOC = Path(__file__).resolve().parent
DATASET = 'nvidia/Open-SWE-Traces'
REVISION = '1db674173119d71df45c22007af99e214600711a'
POLICY = PROJECT / 'research/016/data-preparation/train-family-policy.json'
SHARDS = [
    ('v1.2', 'openhands', 'data/openhands/qwen38_27b/swe-rebench-v2/train-00000-of-00006.parquet'),
    ('v1.2', 'minisweagent', 'data/minisweagent/qwen38_27b/swe-rebench-v2/train-00000-of-00022.parquet'),
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def get_json(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                return json.load(response)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata():
    import pyarrow.parquet as pq
    spec = importlib.util.spec_from_file_location('p015_acquisition', PROJECT / 'research/015/data-preparation/acquire_lego.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    api = json.loads((OUTPUT / 'source-api.json').read_text())
    if api['sha'] != REVISION:
        raise ValueError('API metadata revision differs from pinned source')
    files = {f['rfilename']: f for f in api['siblings']}
    policy = json.loads(POLICY.read_text())
    allowed = set(policy['allowed_train_repositories'])
    denied = set(policy['excluded_repositories'])
    excluded_issues = set(policy['excluded_issue_ids'])
    columns = ['instance_id', 'repo', 'license', 'language', 'trajectory_id', 'resolved']
    selections, sources = [], []
    seen_issues = set()
    for config, split, name in SHARDS:
        dest = OUTPUT / ('metadata-' + split + '.json')
        if dest.exists():
            record = json.loads(dest.read_text())
            if record['revision'] != REVISION or record['file'] != name:
                raise ValueError('Cached metadata identity mismatch')
        else:
            url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}'
            stream = module.RangeReader(url, files[name]['size'])
            parquet = pq.ParquetFile(stream)
            rows = parquet.read(columns=columns, use_threads=False).to_pylist()
            record = dict(revision=REVISION, file=name, file_sha256=files[name]['lfs']['sha256'],
                          file_bytes=files[name]['size'], projected_bytes=stream.bytes_read,
                          columns=columns, rows=rows)
            write_json(dest, record)
        eligible = [r for r in record['rows'] if r['repo'].lower() in allowed
                    and r['repo'].lower() not in denied and r['instance_id'] not in excluded_issues
                    and r['language'].lower() == 'python' and r['resolved'] == 1]
        eligible.sort(key=lambda r: hashlib.sha256(('p016-open-swe-' + r['trajectory_id']).encode()).hexdigest())
        # Round-robin across repositories before taking additional issues.
        by_repo = {}
        for row in eligible:
            by_repo.setdefault(row['repo'].lower(), []).append(row)
        chosen = []
        while len(chosen) < 32 and any(by_repo.values()):
            for repo in sorted(by_repo):
                if not by_repo[repo]:
                    continue
                row = by_repo[repo].pop(0)
                if row['instance_id'] in seen_issues:
                    continue
                seen_issues.add(row['instance_id'])
                chosen.append(dict(row, config=config, split=split, source_file=name))
                if len(chosen) == 32:
                    break
        selections.extend(chosen)
        sources.append({k: v for k, v in record.items() if k != 'rows'} | {
            'metadata_rows': len(record['rows']), 'eligible_rows': len(eligible), 'selected': len(chosen)})
        print(split, 'metadata', len(record['rows']), 'eligible', len(eligible), 'selected', len(chosen), flush=True)
    frozen = dict(dataset=DATASET, revision=REVISION, policy_path=str(POLICY.relative_to(PROJECT)),
                  policy_sha256=digest(POLICY), frozen_before_content_read=True,
                  sampling_scope='First pinned shard per named harness/teacher, not representative of the full dataset',
                  columns_read=columns, selection='resolved=1/Python/historical train families; hash order then repository round-robin; at most32 per harness; unique issues across harnesses',
                  sources=sources, selected=selections)
    dest = DOC / 'selection.json'
    if dest.exists() and json.loads(dest.read_text()) != frozen:
        raise ValueError('Refusing to overwrite a different frozen selection')
    write_json(dest, frozen)


def content():
    frozen = json.loads((DOC / 'selection.json').read_text())
    if frozen['revision'] != REVISION or digest(POLICY) != frozen['policy_sha256']:
        raise ValueError('Frozen source or policy changed')
    if get_json(f'https://huggingface.co/api/datasets/{DATASET}')['sha'] != REVISION:
        raise ValueError('Dataset server main differs from fixed revision')
    offsets_path = DOC / 'row-offsets.json'
    offsets = json.loads(offsets_path.read_text()) if offsets_path.exists() else None
    if offsets and offsets['selection_sha256'] != digest(DOC / 'selection.json'):
        raise ValueError('Row offsets belong to a different frozen selection')
    out = OUTPUT / 'rows'
    out.mkdir(parents=True, exist_ok=True)
    receipts = []
    for item in frozen['selected']:
        path = out / (item['trajectory_id'] + '.json')
        was_cached = path.exists()
        if was_cached:
            value = json.loads(path.read_text())
        else:
            params = dict(dataset=DATASET, config=item['config'], split=item['split'], offset=0, length=1)
            endpoint = 'filter'
            if offsets:
                params['offset'] = offsets['trajectory_offsets'][item['trajectory_id']]
                endpoint = 'rows'
            else:
                params['where'] = '"trajectory_id"=' + "'" + item['trajectory_id'].replace("'", "''") + "'"
            response = get_json('https://datasets-server.huggingface.co/' + endpoint + '?' + urllib.parse.urlencode(params))
            rows = response.get('rows', [])
            if len(rows) != 1:
                raise ValueError('Exact trajectory query did not return one row')
            if rows[0].get('truncated_cells'):
                raise ValueError('Dataset server returned a truncated trajectory')
            value = dict(row=rows[0]['row'], source=dict(dataset=DATASET, revision=REVISION,
                         config=item['config'], split=item['split'], trajectory_id=item['trajectory_id'],
                         row_offset=params['offset'] if offsets else None))
        row = value['row']
        for field in ('trajectory_id', 'instance_id', 'repo', 'language', 'resolved'):
            if row[field] != item[field]:
                raise ValueError('Returned row differs from frozen identity: ' + field)
        if not path.exists():
            write_json(path, value)
        receipts.append(dict(instance_id=row['instance_id'], trajectory_id=row['trajectory_id'],
                             split=item['split'], bytes=path.stat().st_size, sha256=digest(path)))
        print('cached' if was_cached else 'saved', item['split'], row['instance_id'], flush=True)
    if get_json(f'https://huggingface.co/api/datasets/{DATASET}')['sha'] != REVISION:
        raise ValueError('Dataset revision changed while retrieving rows')
    write_json(DOC / 'acquisition.json', dict(dataset=DATASET, revision=REVISION,
               selection_sha256=digest(DOC / 'selection.json'), rows=receipts))


def row_offsets():
    """Map fixed source physical rows to viewer offsets using hash-equal shards."""
    import pyarrow.parquet as pq
    spec = importlib.util.spec_from_file_location('p015_acquisition', PROJECT / 'research/015/data-preparation/acquire_lego.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frozen = json.loads((DOC / 'selection.json').read_text())
    original = json.loads((OUTPUT / 'source-api.json').read_text())
    converted = json.loads((OUTPUT / 'converted-api.json').read_text())
    original_files = {x['rfilename']: x for x in original['siblings']}
    converted_files = {x['rfilename']: x for x in converted['siblings']}
    offsets, evidence = {}, []
    for config, split, source_file in SHARDS:
        sha = original_files[source_file]['lfs']['sha256']
        files = sorted((f for name, f in converted_files.items()
                        if name.startswith(config + '/' + split + '/') and name.endswith('.parquet')),
                       key=lambda f: f['rfilename'])
        target = next(f for f in files if f.get('lfs', {}).get('sha256') == sha)
        previous = files[:files.index(target)]
        def count_rows(file):
            cache = OUTPUT / 'viewer-footer-rows' / (file['lfs']['sha256'] + '.json')
            if cache.exists():
                return json.loads(cache.read_text())
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{converted['sha']}/{file['rfilename']}"
            stream = module.RangeReader(url, file['size'])
            nrows = pq.ParquetFile(stream).metadata.num_rows
            value = dict(file=file['rfilename'], sha256=file['lfs']['sha256'], rows=nrows,
                         footer_bytes_read=stream.bytes_read)
            write_json(cache, value)
            return value
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            counts = list(executor.map(count_rows, previous))
        start = sum(x['rows'] for x in counts)
        rows = json.loads((OUTPUT / ('metadata-' + split + '.json')).read_text())['rows']
        index = {r['trajectory_id']: i for i, r in enumerate(rows)}
        for item in frozen['selected']:
            if item['split'] == split:
                offsets[item['trajectory_id']] = start + index[item['trajectory_id']]
        evidence.append(dict(split=split, converted_file=target['rfilename'], source_file=source_file,
                             identical_file_sha256=sha, offset=start, preceding_shards=counts))
        print(split, 'verified source/viewer hash equality; offset', start, flush=True)
    write_json(DOC / 'row-offsets.json', dict(selection_sha256=digest(DOC / 'selection.json'),
               converted_revision=converted['sha'], trajectory_offsets=offsets, evidence=evidence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['metadata', 'content', 'row-offsets'])
    args = parser.parse_args()
    {'metadata': metadata, 'content': content, 'row-offsets': row_offsets}[args.mode]()
