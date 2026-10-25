"""Acquire fixed train source pages and collect individually reviewed Jinja cache tasks."""
from pathlib import Path
import concurrent.futures
import hashlib
import importlib.util
import json
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
REPLAY=ROOT/'competition/vfarl/replay'
spec=importlib.util.spec_from_file_location('jinja_source_collector',HERE.parent/'s600-common-root/collect.py')
c=importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
c.HERE=HERE
c.ART=REPLAY/'artifacts/f600-review-120-jinja'
c.REPORT_PREFIX='f600-jinja-'
c.REVIEW_SCOPE='Individually reviewed Jinja LRU cache hit, invalidation and ordering contracts; original train full requirements retained.'
c.DEFAULT_COURSE='F600_specialist'

def fetch(offset):
    pages=HERE/'artifacts/pages'
    pages.mkdir(parents=True,exist_ok=True)
    path=pages/f'jinja-{offset}.json'
    receipt=path.with_suffix('.receipt.json')
    if path.exists():
        d=json.loads(receipt.read_text())
        if d['sha256']!=c.sha(path) or d['revision']!=c.REV or d['split']!='train':
            raise ValueError('Cached fixed-source page differs from receipt')
        return
    url=('https://datasets-server.huggingface.co/rows?dataset=SWE-bench%2FSWE-smith-py'
         '&config=default&split=train&revision='+c.REV+f'&offset={offset}&length=100')
    with urllib.request.urlopen(url,timeout=120) as response:
        body=response.read()
        revision=response.headers.get('X-Revision')
    if revision!=c.REV:
        raise ValueError('Fixed train revision was not confirmed by source server')
    payload=json.loads(body)
    path.write_bytes(body)
    c.save(receipt,dict(url=url,offset=offset,length=100,revision=c.REV,split='train',
        response_X_Revision=revision,sha256=c.sha(path),rows=len(payload['rows'])))

if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='fetch':
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for _ in pool.map(fetch,range(37179,38136,100)):
                pass
        print('acquired_fixed_jinja_train_pages',flush=True)
    else:
        configs=json.loads((HERE/'selected-jinja-cache.json').read_text())['tasks']
        c.ART.mkdir(parents=True,exist_ok=True)
        for config in configs:
            if stage=='prepare':
                c.prepare(config)
                print('prepared',config['key'],flush=True)
            elif stage=='replay':
                c.replay(config)
                print('replayed',config['key'],flush=True)
        if stage=='publish':
            if not all((c.ART/(x['key']+'-run-001/run-summary.json')).is_file() for x in configs):
                raise ValueError('Finish all selected actual attempts before publishing this complete batch')
            c.REPORT_PREFIX='f600-jinja-completed-'
            c.publish(configs,review_name='reviewed-jinja-cache-completed-v2')
