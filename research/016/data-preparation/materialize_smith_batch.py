#!/usr/bin/env python3
"""Materialize only an explicitly frozen train batch from cached Smith shards.

No network, source execution, GPU, task replay or final-patch column reads.
"""
import argparse
import collections
import hashlib
import json
import re
import time
from pathlib import Path
import duckdb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = HERE.parent / 'data'
REVISION = '08e109b4a59eaeebf80e4675cd125d42e7ac99a4'


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def read(path): return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, default=HERE / 'smith-next-1000-identity-freeze.json')
    parser.add_argument('--output', type=Path, default=DATA / 'source-rows-next1000')
    parser.add_argument('--manifest', type=Path, default=HERE / 'smith-next-1000-acquisition.json')
    args = parser.parse_args()
    start = time.monotonic()
    frozen = read(args.freeze)
    policy_path = HERE / 'train-family-policy.json'
    policy = read(policy_path)
    if frozen['policy_sha256'] != sha_file(policy_path): raise ValueError('Changed family policy')
    if frozen['source_revision'] != REVISION: raise ValueError('Unexpected source revision')
    selected = frozen['selected']
    old = {r['canonical_task_key'] for r in read(HERE / 'source-sample-freeze.json')['selected']}
    seen, groups = set(), collections.defaultdict(list)
    for row in selected:
        key = row['canonical_task_key']
        if (row['split'] != 'train' or row['repo'] not in policy['allowed_train_repositories']
                or row['repo'] in policy['excluded_repositories'] or key in seen or key in old):
            raise ValueError('Duplicate or nontrain identity: ' + key)
        if row['dataset'] != 'SWE-bench/SWE-smith-trajectories' or row['revision'] != REVISION:
            raise ValueError('Unexpected dataset pin')
        if '/' in row['instance_id'] or '..' in row['instance_id']:
            raise ValueError('Unsafe instance filename')
        seen.add(key)
        groups[row['source_file']].append(row)
    pins = {str(Path(r['path']).relative_to(ROOT)): r
            for r in read(ROOT / 'research/011/data-scaling-20261018/shard-manifest.json')}
    con = duckdb.connect()
    receipts, used_sources = [], []
    exact_meta = {r['instance_id']: r for r in read(HERE / 'smith-cached-task-provenance.json')['rows']}
    for filename, members in sorted(groups.items()):
        source = ROOT / filename
        pin = pins[filename]
        if source.stat().st_size != pin['bytes'] or sha_file(source) != pin['sha256']:
            raise ValueError('Cached source differs from pinned artifact: ' + filename)
        used_sources.append({'path': filename, 'bytes': pin['bytes'], 'sha256': pin['sha256']})
        con.execute('CREATE OR REPLACE TEMP TABLE wanted(row_number BIGINT, iid VARCHAR, tid VARCHAR)')
        con.executemany('INSERT INTO wanted VALUES (?, ?, ?)',
                        [(r['row_index'], r['instance_id'], r['traj_id']) for r in members])
        membership = {(r['row_index'], r['instance_id'], r['traj_id']): r for r in members}
        cursor = con.execute('SELECT p.file_row_number,p.instance_id,p.traj_id,p.messages,p.model,p.resolved '
            'FROM read_parquet(?,file_row_number=true) p JOIN wanted w '
            'ON p.file_row_number=w.row_number AND p.instance_id=w.iid AND p.traj_id=w.tid', [str(source)])
        actual = 0
        while True:
            batch = cursor.fetchmany(16)
            if not batch: break
            for number, iid, tid, messages, model, resolved in batch:
                if not resolved: raise ValueError('Frozen success metadata changed')
                item = membership[(number, iid, tid)]
                row = {'instance_id': iid, 'traj_id': tid, 'messages': json.loads(messages),
                       'model': model, 'resolved': resolved}
                target = args.output / (iid + '.json')
                if target.exists():
                    if read(target) != row: raise ValueError('Existing cached body differs: ' + iid)
                else: write(target, row)
                provenance = exact_meta.get(iid)
                receipts.append(dict(item, local_path=str(target.relative_to(ROOT)),
                    bytes=target.stat().st_size, sha256=sha_file(target),
                    transport={'method':'atomic cached Parquet join on file row, issue and trajectory','truncated':False},
                    full_buggy_commit_from_cached_provenance=provenance['execution_buggy_commit'] if provenance else None,
                    cached_license_provenance=provenance['provenance_path'] if provenance else None,
                    evidence='upstream resolved label; not locally revalidated',
                    training_admission_completed=False))
                actual += 1
        if actual != len(members): raise ValueError('Not all exact frozen rows were materialized')
        print('source complete', Path(filename).name, actual, 'cumulative',len(receipts),flush=True)
    if len(receipts) != len(selected): raise ValueError('Final frozen-count mismatch')
    receipts.sort(key=lambda r: r['canonical_task_key'])
    output = {'source_dataset':'SWE-bench/SWE-smith-trajectories','source_revision':REVISION,
       'frozen_selection_path':str(args.freeze.relative_to(ROOT)),
       'frozen_selection_sha256':sha_file(args.freeze), 'sources':used_sources,
       'scope':'Only explicitly frozen train rows; no network, patch column, replay, GPU or training.',
       'issues':len(receipts), 'repositories':len({r['repo'] for r in receipts}),
       'total_body_bytes':sum(r['bytes'] for r in receipts),
       'exact_buggy_commit_and_license_metadata_available':sum(r['full_buggy_commit_from_cached_provenance'] is not None for r in receipts),
       'wall_seconds':time.monotonic()-start, 'rows':receipts}
    write(args.manifest,output)
    print(json.dumps({k:v for k,v in output.items() if k not in ('rows','sources')},indent=2),flush=True)


if __name__ == '__main__': main()
