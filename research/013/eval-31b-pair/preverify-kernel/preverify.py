"""CPU experiment preparation; existing frozen criteria run inline, no test file writes."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

TASK_ORDER = [f't{i:02d}' for i in range(1,13)]+['s01']
PACKAGE = {'pallets/click':'click','pallets/markupsafe':'markupsafe','pydicom/pydicom':'pydicom'}
CRITERION_PROGRAM = '''import importlib.util,json,pathlib,sys
data=json.loads(sys.argv[1]); root=pathlib.Path(data['tree']).resolve()
sys.path[:0]=[str(root/'src'),str(root)]
spec=importlib.util.find_spec(data['package'])
assert spec and spec.origin and pathlib.Path(spec.origin).resolve().is_relative_to(root), (spec.origin if spec else None)
print('SOURCE_ORIGIN='+spec.origin,flush=True)
scope={'__file__':str(root/'tests'/data['test_file']),'__name__':'p014_frozen_criterion'}
exec(compile(data['source'],scope['__file__'],'exec'),scope)
scope[data['node']]()
print('CRITERION_PASS',flush=True)
'''

def digest(data): return hashlib.sha256(data).hexdigest()

def run(cmd,cwd,timeout=120,env=None):
    try:
        p=subprocess.run(cmd,cwd=str(cwd),env=env,capture_output=True,text=True,timeout=timeout)
        return p.returncode,(p.stdout or '')+(p.stderr or '')
    except subprocess.TimeoutExpired as e:
        text=lambda x: x.decode(errors='replace') if isinstance(x,bytes) else x or ''
        return 124,text(e.stdout)+text(e.stderr)+f'\nTIMEOUT {timeout}s\n'

def apply_replay_op(tree,op):
    rel=Path(op['path'])
    if rel.is_absolute() or '..' in rel.parts or 'tests' in rel.parts or rel.name.startswith('test_'):
        raise ValueError('reference replay must edit production source only: '+str(rel))
    target=tree/rel
    if op['op']=='create':
        assert not target.exists(),str(rel)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(op.get('new_str',op.get('content',op.get('file_text'))));return
    assert op['op']=='str_replace',op['op']
    text=target.read_text();old,new=op['old_str'],op['new_str']
    for level,needle in enumerate([old,old.lstrip('\n'),old.strip('\n'),old.strip()]):
        if needle and needle in text:
            target.write_text(text.replace(needle,new.strip('\n') if level==3 else new,1));return
    raise ValueError('reference anchor missing: '+str(rel))

def criterion(python,tree,crit,source):
    data={'tree':str(tree.resolve()),'package':PACKAGE[crit['repo']],'test_file':crit['test_file'],'node':crit['f2p_node'],'source':source}
    env=dict(os.environ,PYTHONPATH=str(tree.resolve()/'src')+os.pathsep+str(tree.resolve()),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    return run([str(python),'-c',CRITERION_PROGRAM,json.dumps(data)],tree,90,env)

def process_task(task,crit,args):
    out=Path(args.workdir).resolve();work=out/'attempts'/task['task_id'];work.mkdir(parents=True,exist_ok=True)
    iid=task['instance_id'];baseline=out/'snapshots'/iid
    source=Path(args.snapshot_root)/iid if args.snapshot_root else None
    if baseline.exists(): shutil.rmtree(baseline)
    if source and source.is_dir():shutil.copytree(source,baseline,symlinks=True)
    else:
        code,log=run(['git','clone','--depth','1','--branch',iid,'https://github.com/'+task['mirror']+'.git',str(baseline)],out,600)
        (work/'clone.log').write_text(log)
        if code:return {'task_id':task['task_id'],'instance_id':iid,'eligible':False,'reason':'clone failed','exit':code}
    head=run(['git','rev-parse','HEAD'],baseline,30)[1].strip()
    assert head==task['base_commit'],(iid,head,task['base_commit'])
    tree=work/'tree'
    if tree.exists():shutil.rmtree(tree)
    shutil.copytree(baseline,tree,symlinks=True,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    mat=Path(args.material_dir);source_text=(mat/crit['test_file']).read_text()
    pre,prelog=criterion(args.python,tree,crit,source_text);(work/'before.log').write_text(prelog)
    replay=json.loads((mat/'replay'/(iid+'.json')).read_text())
    for op in replay['ops']:apply_replay_op(tree,op)
    post,postlog=criterion(args.python,tree,crit,source_text);(work/'after.log').write_text(postlog)
    # Source ImportError is a valid bug if unchanged criterion passes after production edits.
    eligible=pre not in (0,124) and post==0 and 'SOURCE_ORIGIN=' in prelog
    result={'task_id':task['task_id'],'instance_id':iid,'repo':task['repo'],'base_commit':head,'eligible':eligible,'pre_exit':pre,'post_exit':post,'before_sha256':digest(prelog.encode()),'after_sha256':digest(postlog.encode()),'criterion_sha256':digest(source_text.encode()),'p2p':None,'p2p_note':'No independent regression run in the first CPU batch; repeated criterion is F2P only.','reason':None if eligible else 'criterion not discriminatory or reference/environment failure'}
    (work/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--material-dir',default=str(Path(__file__).resolve().parent));ap.add_argument('--snapshot-root');ap.add_argument('--workdir',default='/kaggle/working/p014-panel');ap.add_argument('--python',default=sys.executable);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--tasks',default=','.join(TASK_ORDER));ap.add_argument('--count',type=int,default=8);args=ap.parse_args()
    out=Path(args.workdir).resolve();out.mkdir(parents=True,exist_ok=True);(out/'snapshots').mkdir(exist_ok=True)
    mat=Path(args.material_dir);tasks={x['task_id']:x for x in [json.loads(line) for line in (mat/'tasks-13.jsonl').read_text().splitlines() if line]};criteria={x['task_id']:x for x in json.loads((mat/'criteria-index.json').read_text())['items']}
    ordered=[x for x in args.tasks.split(',') if x in tasks]
    def attempt(tid):
        try:return process_task(tasks[tid],criteria[tid],args)
        except Exception as e:
            import traceback
            p=out/'attempts'/tid;p.mkdir(parents=True,exist_ok=True);(p/'failure.txt').write_text(traceback.format_exc())
            return {'task_id':tid,'instance_id':tasks[tid]['instance_id'],'eligible':False,'reason':str(e)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(attempt,ordered))
    eligible=[r for r in results if r['eligible']];selected=eligible[:args.count]
    freeze={'protocol':'p014-dev-f2p-v1','selection':'predeclared t01..t12,s01 order; first eligible; no model outputs observed','requested_count':args.count,'count':len(selected),'repositories':sorted({r['repo'] for r in selected}),'items':selected,'all_results':results,'python':sys.version,'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'p2p_note':'No separate regression claim; only frozen criterion F2P executed.'}
    (out/'panel-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    host=[dict(criteria[r['task_id']],source=(mat/criteria[r['task_id']]['test_file']).read_text()) for r in selected]
    (out/'host-criteria.json').write_text(json.dumps(host,ensure_ascii=False,indent=2)+'\n')
    print('PANEL_FREEZE '+json.dumps({'count':len(selected),'tasks':[r['task_id'] for r in selected]}),flush=True)
    return 0 if selected else 1

if __name__=='__main__':raise SystemExit(main())
