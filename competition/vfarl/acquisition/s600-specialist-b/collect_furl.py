"""Common-course Furl contracts in the disclosed Python 3.9 task profile."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import collect_common
import collect as base

HOME=Path(__file__).resolve().parent
base.HERE=HOME/'common-furl39-batch'
base.ART=base.REPLAY/'artifacts/s600-common-b'
base.CANDIDATES={'furl-strip-scheme-py39':dict(collect_common.base.CANDIDATES['furl-strip-scheme'])}
for config in base.CANDIDATES.values():
    config['task_python']='/Library/Developer/CommandLineTools/usr/bin/python3'
    config['task_site_packages']=str(base.REPLAY/'.venv-furl39-task/lib/python3.9/site-packages')
    config['task_runtime_profile_reason']='Original fixed Furl scope passes all 77 native cases on Python3.9.6; Python3.12 full scope retains 3 unrelated upstream URL/IPv6 stdlib incompatibilities. Modern official ADK parent stays Python3.12. Original tests, node identifiers and source are unchanged.'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keys',nargs='+',choices=list(base.CANDIDATES),default=list(base.CANDIDATES))
    parser.add_argument('--stage',choices=('prepare','replay','publish','all'),default='all')
    parser.add_argument('--workers',type=int,choices=(1,2),default=1)
    args=parser.parse_args()
    if args.stage in {'prepare','all'}:
        for key in args.keys:base.prepare(key);print('prepared',key,flush=True)
    if args.stage in {'replay','all'}:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for key in pool.map(lambda key:(base.replay(key),key)[1],args.keys):print('replayed',key,flush=True)
    if args.stage in {'publish','all'}:base.publish(args.keys)
