"""Batch real official-tool local edit transactions, without executing source shell.

Source `patch` is intentionally unused: atomic row reads demonstrated mismatches.
This producer does not establish behavioral correctness or whole-issue resolution.
"""
import argparse, ast, concurrent.futures, hashlib, importlib.util, json, re
import shutil, subprocess, sys, tempfile, time, urllib.error, urllib.request
from collections import defaultdict
from pathlib import Path, PurePosixPath
import duckdb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REV = '08e109b4a59eaeebf80e4675cd125d42e7ac99a4'
SEED = 'p011-edit-replay-20261018'
spec = importlib.util.spec_from_file_location('p006_replay', ROOT / 'research/006/data-replay-20261013/replay.py')
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
TOOLS = R.official_tools()

def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()

def canonical(iid):
    prefix = re.split(r'\.[0-9a-f]{8}\.', iid)[0]
    return prefix.replace('__', '/', 1).lower()

def freeze(paths, name):
    old = json.loads((ROOT / 'research/009/training-data-20261018/data-usage.json').read_text())
    v2 = json.loads((ROOT / 'research/009/training-data-v2-20261018/data-usage.json').read_text())
    seen = {r['instance_id'] for r in old['selected'] + v2['selected']}
    excluded = set(old['excluded_repositories'])
    c = duckdb.connect(); rows=[]
    for p in paths:
        # Project metadata only; never materialize reserved task messages or patch.
        for number,iid,resolved,model,traj in c.execute('SELECT file_row_number,instance_id,resolved,model,traj_id FROM read_parquet(?, file_row_number=true)',[str(p)]).fetchall():
            if not resolved or not re.search(r'\.pr_\d+$',iid) or canonical(iid) in excluded or iid in seen:
                continue
            rows.append(dict(source_file=str(p.relative_to(ROOT)), row_index=number, instance_id=iid,
                             repo=canonical(iid), source_resolved=True, teacher_source=model,traj_id=traj))
    byissue={}
    for row in sorted(rows,key=lambda r:sha(SEED+r['instance_id']+r['traj_id']+r['source_file'])):
        byissue.setdefault(row['instance_id'],row)
    groups=defaultdict(list)
    for row in byissue.values(): groups[row['repo']].append(row)
    selected=[]
    while any(groups.values()):
        for repo in sorted(groups):
            if groups[repo]: selected.append(groups[repo].pop(0))
    usage={'frozen_before_content_read':True,'source_revision':REV,'seed':SEED,
           'rule':'Resolved-labeled real PR metadata; unique instance; hash order within repo then alphabetical repo round robin. No outcome-based replacements.',
           'excluded_repositories':sorted(excluded),'excluded_previously_read_issues':sorted(seen),
           'assigned_train_repositories':sorted({r['repo'] for r in selected}),
           'future_selection_replacements_must_skip_train_repos':True,'selected':selected,
           'task_resolved':None,'behavior_verified':False,'automatic_training':False}
    dump(HERE/name,usage); return usage

def production(path):
    p=PurePosixPath(path)
    return (not p.is_absolute() and '..' not in p.parts and p.suffix=='.py'
            and not any(x.lower() in ('test','tests','test-data','docs','doc','examples','.github') for x in p.parts)
            and not p.name.lower().startswith(('test','reproduce','debug','verify','check','temp','setup'))
            and not p.name.endswith('_test.py'))

def prepare(row):
    c=duckdb.connect()
    # Atomic selection by BOTH logical identity and physical row, not zipped columns.
    got=c.execute('SELECT file_row_number,instance_id,traj_id,messages FROM read_parquet(?, file_row_number=true) WHERE file_row_number=? AND instance_id=? AND traj_id=?',
                  [str(ROOT/row['source_file']),row['row_index'],row['instance_id'],row['traj_id']]).fetchone()
    if not got: raise ValueError('Atomic source identity mismatch')
    messages=json.loads(got[3]); user=next(m for m in messages if m['role']=='user')['content']
    if isinstance(user,list): user='\n'.join(x.get('text','') for x in user if isinstance(x,dict))
    match=re.search(r'<pr_description>(.*?)</pr_description>',user,re.S)
    issue=match.group(1).strip() if match else user
    actions=[]
    for mi,m in enumerate(messages):
        calls=list(m.get('tool_calls',[]) or [])
        content=m.get('content','')
        if isinstance(content,str):
            for name,body in re.findall(r'<function=([^>]+)>(.*?)</function>',content,re.S):
                params={key:value for key,value in re.findall(r'<parameter=([^>]+)>(.*?)</parameter>',body,re.S)}
                calls.append({'function':{'name':name,'arguments':params}})
        for call in calls:
            f=call.get('function',{}); raw=f.get('arguments',{})
            try: a=json.loads(raw) if isinstance(raw,str) else raw
            except (ValueError,TypeError): continue
            if a.get('command')!='str_replace': continue
            path=a.get('path','').removeprefix('/testbed/')
            if not production(path): continue
            if not isinstance(a.get('old_str'),str) or not isinstance(a.get('new_str'),str): continue
            if not a['old_str'] or a['old_str']==a['new_str']: continue
            actions.append({'source_message_index':mi,'filepath':path,'old_string':a['old_str'],'new_string':a['new_str'],'allow_multiple':False})
    if not actions: return dict(row,status='no_structured_production_replacement')
    if len(actions)>20: return dict(row,status='more_than_20_source_edits',edits=len(actions))
    out=HERE/'outputs'/row['instance_id'];out.mkdir(parents=True,exist_ok=True)
    dump(out/'source.json',dict(row,messages_sha256=sha(got[3]),issue=issue,actions=actions,patch_column_used=False))
    mirror='swesmith/'+row['instance_id'].split('.pr_')[0]
    ref=subprocess.run(['git','ls-remote','https://github.com/'+mirror+'.git','refs/heads/'+row['instance_id']],text=True,capture_output=True,check=True,timeout=90).stdout
    commit=ref.split()[0]
    if not re.fullmatch('[0-9a-f]{40}',commit): raise ValueError('Exact buggy commit missing')
    source=out/'source';files=[]
    def fetch(path):
        url='https://raw.githubusercontent.com/'+mirror+'/'+commit+'/'+path
        with urllib.request.urlopen(url,timeout=45) as response: data=response.read(2_000_001)
        if len(data)>2_000_000: raise ValueError('Oversized source file')
        dest=source/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        files.append({'path':path,'url':url,'bytes':len(data),'sha256':sha(data)})
    for path in sorted({a['filepath'] for a in actions}): fetch(path)
    licenses=[]
    for name in ('LICENSE','LICENSE.md','LICENSE.txt','COPYING'):
        try: fetch(name);licenses.append(files[-1]);break
        except urllib.error.HTTPError as e:
            if e.code!=404: raise
    if not licenses: return dict(row,status='license_not_found',buggy_commit=commit)
    provenance=dict(row,mirror=mirror,buggy_commit=commit,git_ls_remote_stdout=ref,files=files,licenses=licenses,
                    state_scope='Fixed-commit production source subset. No source tests/dependencies/shell executed.',
                    dataset_license='MIT; repository code license retained separately',patch_column_used=False)
    dump(out/'provenance.json',provenance)
    with tempfile.TemporaryDirectory(prefix='p011-replay-') as tmp:
        ws=Path(tmp)/'workspace';shutil.copytree(source,ws);R.baseline(ws)
        transport=R.LocalTransport(ws);ctx=R.Context(transport,row['repo']);ctx.budget.tool_calls=100
        rec=R.Recorder(ctx,TOOLS,'Implement the following requested change in the repository.\n\n'+issue+
            '\n\nScope of this editing session: update the following production files and submit the resulting local patch: '+
            ', '.join(sorted({a['filepath'] for a in actions}))+'. No full task validation has been performed in this session.')
        targets=[];failure=None
        for a in actions:
            path=a['filepath']; text=(ws/path).read_text()
            offset=text.find(a['old_string']);line=text[:max(offset,0)].count('\n')+1
            read=rec.call('read_file',{'filepath':path,'start_line':max(1,line-12),'end_line':line+a['old_string'].count('\n')+12})
            if read['status']!='ok': failure='read_file failed';break
            if read.get('is_truncated') or a['old_string'].rstrip('\n') not in read.get('content',''):
                failure='Selected read did not expose the complete exact edit context';break
            args={k:v for k,v in a.items() if k!='source_message_index'}
            ordinal=len(rec.events);result=rec.call('edit_file',args)
            if result['status']!='ok': failure='source edit did not apply to actual current state';break
            targets.append(ordinal)
        if not failure:
            for path in sorted({a['filepath'] for a in actions}):
                try: ast.parse((ws/path).read_text(),filename=path)
                except SyntaxError as e: failure='Edited production file has syntax error: '+str(e);break
        if not failure:
            ordinal=len(rec.events)
            result=rec.call('submit_patch',{})
            if result['status']!='ok' or not ctx.submitted_patch: failure='Official submission was empty or failed'
            else: targets.append(ordinal)
        dump(out/'events.json',rec.events);dump(out/'commands.json',transport.commands)
        patch=R.git(ws,'diff','_swegemma_baseline');(out/'actual.patch').write_text(patch)
        state={p:{'before_sha256':sha((source/p).read_bytes()),'after_sha256':sha((ws/p).read_bytes())} for p in sorted({a['filepath'] for a in actions})}
        dump(out/'states.json',state)
        if failure:return dict(row,status='replay_failed',reason=failure,buggy_commit=commit)
        tx={'sample_id':'p011-'+row['instance_id'],'transaction_id':'p011-'+row['instance_id'],
            'producer_revision':'p011-local-2',
            'synthetic':False,'origin_kind':'cpu_replayed_source_edit_subset',
            'group':{'issue':row['instance_id'],'repo':row['repo'],'commit':commit,'split':'train'},
            'source':{'uri':f'https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/tree/{REV}',
                      'license':'MIT dataset; repository license in provenance.json','sha256':sha(got[3]),
                      'teacher':row['teacher_source'],'traj_id':row['traj_id']},
            'protocol_id':'p005-harness-tools-v1','messages':rec.messages,'target_assistant_steps':targets,
            'behavior':'local_edit','needs_replay':False,'attempt_status':'local_edits_applied',
            'observed_outcome':{'action_succeeded':True,'task_resolved':None,'behavior_verified':False,
                                'syntax_parse_passed':True,'official_patch_nonempty':True,
                                'scope':'All selected structured production str_replace actions replayed from fixed buggy snapshot; source shell/create/insert and tests excluded. This is local edit supervision, not a resolved task or full trajectory.'},
            'source_resolved_label':True,'source_label_not_revalidated':True,
            'replay_boundary':'Unchanged official v31 tools with macOS local transport, production subset. Read locations derived from source edit targets; this is edit/action SFT, not search-policy supervision.',
            'continuation_policy':'New read/edit/submit tool observations only; no old assistant future or tool observations reused.'}
        dump(out/'transaction.json',tx)
        return dict(row,status='local_edits_applied',buggy_commit=commit,edits=len(targets),transaction=str((out/'transaction.json').relative_to(HERE)))

def main():
    if (HERE/'dataset-manifest.json').exists():
        raise RuntimeError('This dataset version is frozen; continue in a new version directory')
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--all-shards',action='store_true');ap.add_argument('--limit',type=int,default=10);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    paths=[ROOT/'training/artifacts/external/swe-smith/tool-00004-of-00008.parquet']
    if args.all_shards:paths+=sorted((HERE/'shards').glob('*.parquet'))
    name='data-usage-full.json' if args.all_shards else 'data-usage.json'
    usage=freeze(paths,name) if args.freeze else json.loads((HERE/name).read_text())
    if args.limit==0:return
    rows=usage['selected'][:args.limit];results=[];start=time.monotonic()
    def run(row):
        if row['repo'] in usage['excluded_repositories']:raise ValueError('Reserved repo')
        existing=HERE/'outputs'/row['instance_id']/'transaction.json'
        if existing.exists():
            provenance=json.loads((existing.parent/'provenance.json').read_text())
            if all(provenance.get(k)==row.get(k) for k in ('source_file','row_index','traj_id')):
                tx=json.loads(existing.read_text())
                if tx.get('producer_revision')=='p011-local-2':
                    return dict(row,status='local_edits_applied',transaction=str(existing.relative_to(HERE)),reused=True)
        try:return prepare(row)
        except Exception as e:return dict(row,status='source_or_replay_error',reason=type(e).__name__+': '+str(e))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run,rows):
            results.append(result);dump(HERE/'production-results.json',results)
            print(json.dumps({k:result[k] for k in ('instance_id','repo','status','reason','edits') if k in result}),flush=True)
    print(json.dumps({'attempted':len(results),'applied':sum(r['status']=='local_edits_applied' for r in results),'seconds':time.monotonic()-start}),flush=True)

if __name__=='__main__':main()
