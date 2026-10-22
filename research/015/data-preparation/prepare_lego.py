"""Pinned train-only SWE-Lego source census and actual local-tool edit replay.

This preserves source observations separately. Replay never executes source shell,
never copies old reasoning/observations, and does not claim whole-task resolution.
"""
import ast, concurrent.futures, hashlib, importlib.util, json, re, shutil, sys, tempfile, urllib.request
from pathlib import Path, PurePosixPath
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=HERE.parent/'data'
sys.path.insert(0,str(ROOT))
from training.encoding.gemma_tool_chat import load_tokenizer,count_template_tokens,encode_transaction
spec=importlib.util.spec_from_file_location('replay',ROOT/'research/006/data-replay-20261013/replay.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)
TOOLS=R.official_tools()
PREVIOUS={r['instance_id']:r for r in json.loads((HERE/'replay-results.json').read_text())} if (HERE/'replay-results.json').exists() else {}
def sha(x):return hashlib.sha256(x if isinstance(x,bytes) else x.encode()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def prod(p):
 q=PurePosixPath(p)
 return not q.is_absolute() and '..' not in q.parts and q.suffix=='.py' and not any(t.lower() in ('tests','test','docs','examples') for t in q.parts) and not q.name.startswith(('test','setup','check','verify','reproduce'))
def actions(row):
 out=[]
 for i,m in enumerate(row['messages']):
  if m['role']!='assistant':continue
  for name,body in re.findall(r'<function=([^>]+)>(.*?)</function>',m['content'],re.S):
   a={k:v.strip('\n') for k,v in re.findall(r'<parameter=([^>]+)>(.*?)</parameter>',body,re.S)}
   if name!='str_replace_editor' or a.get('command')!='str_replace':continue
   path=a.get('path','');path=path.split('/',3)[3] if path.startswith('/workspace/') and len(path.split('/',3))==4 else path.removeprefix('/testbed/')
   if prod(path) and a.get('old_str') and 'new_str' in a:
    out.append(dict(source_message_index=i,filepath=path,old_string=a['old_str'],new_string=a['new_str'],allow_multiple=False))
 return out

def replay(item):
 iid=item['instance_id']
 if iid in PREVIOUS:return PREVIOUS[iid]
 row=json.load(open(DATA/'source-rows'/f'{iid}.json'))['rows'][0]['row'];aa=actions(row)
 result=dict(instance_id=iid,repo=item['repo'],source_edits=len(aa),source_resolved=row['resolved'],license=row['license_name'])
 if not aa:return result|{'reason':'No supported structured production str_replace actions'}
 out=DATA/'replay'/iid;source=out/'source';files=[]
 try:
  for path in sorted({a['filepath'] for a in aa}):
   url=f"https://raw.githubusercontent.com/{row['repo']}/{row['base_commit']}/{path}"
   with urllib.request.urlopen(url,timeout=45) as response:b=response.read(2000001)
   if len(b)>2000000:raise ValueError('Source file too large')
   p=source/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);files.append(dict(path=path,url=url,sha256=sha(b)))
  with tempfile.TemporaryDirectory(prefix='p015-lego-') as tmp:
   ws=Path(tmp)/'workspace';shutil.copytree(source,ws);R.baseline(ws)
   ctx=R.Context(R.LocalTransport(ws),row['repo']);ctx.budget.tool_calls=200
   rec=R.Recorder(ctx,TOOLS,'Implement the requested change.\n\n'+row['problem_statement']+'\n\nLocal editing scope: '+', '.join(x['path'] for x in files)+'. This session has no full task validation.')
   targets=[];failures=[];success=0;recovered=0
   for a in aa:
    text=(ws/a['filepath']).read_text();off=text.find(a['old_string']);line=text[:max(0,off)].count('\n')+1
    read=rec.call('read_file',dict(filepath=a['filepath'],start_line=max(1,line-8),end_line=line+a['old_string'].count('\n')+8))
    # For successful targets the complete anchor must really have been observed.
    visible=not read.get('is_truncated') and a['old_string'] in read.get('content','')
    ordinal=len(rec.events);res=rec.call('edit_file',{k:v for k,v in a.items() if k!='source_message_index'})
    if res['status']=='ok':
     if not visible:raise ValueError('Successful edit lacks complete observed anchor')
     targets.append(ordinal);success+=1
     if failures:recovered+=1;failures=[]
    else:failures.append(ordinal)
   if not success:raise ValueError('No successful production edit')
   for p in files:ast.parse((ws/p['path']).read_text(),filename=p['path'])
   ordinal=len(rec.events);sub=rec.call('submit_patch',{})
   if sub['status']!='ok' or not ctx.submitted_patch:raise ValueError('No official patch')
   targets.append(ordinal)
   tx=dict(sample_id='p015-lego-'+iid,transaction_id='p015-lego-'+iid,synthetic=False,origin_kind='cpu_replayed_source_edit_subset',group=dict(issue=iid,repo=row['repo'].lower(),commit=row['base_commit'],split='train'),source=dict(uri='https://huggingface.co/datasets/Lego-X/SWE-Lego-Real-Data/tree/879c96b91dfa321b4d1c621d149a9200f8a1d118',license=row['license_name'],sha256=sha(json.dumps(row['messages'])),teacher='SWE-Lego public OpenHands trajectory; identity not established by row metadata'),protocol_id='p005-harness-tools-v1',messages=rec.messages,target_assistant_steps=targets,needs_replay=False,observed_outcome=dict(action_succeeded=True,task_resolved=None,behavior_verified=False,syntax_parse_passed=True,official_patch_nonempty=True),behavior='tool_recovery' if recovered else 'local_edit',source_resolved_label=row['resolved'],source_label_not_revalidated=True,replay_boundary='Pinned base_commit production Python subset; unchanged official v31 tools, macOS local transport; no source shell/tests. Source-derived edit locations; not autonomous search or full trajectory replay.',continuation_policy='All observations newly executed; failed edit actions masked; no old assistant reasoning or future observations reused.')
   dump(out/'transaction.json',tx);dump(out/'events.json',rec.events);dump(out/'commands.json',ctx.docker.commands);dump(out/'provenance.json',dict(files=files,base_commit=row['base_commit'],source_row_sha256=sha((DATA/'source-rows'/f'{iid}.json').read_bytes())))
   (out/'actual.patch').write_text(R.git(ws,'diff','_swegemma_baseline'))
   return result|dict(replay_succeeded=True,successful_edits=success,recovery_count=recovered,failed_edit_count=sum(e['tool']=='edit_file' and e['response']['status']!='ok' for e in rec.events),transaction=str(out/'transaction.json'))
 except Exception as e:return result|dict(replay_succeeded=False,reason=f'{type(e).__name__}: {e}')

def main():
 freeze=json.load(open(HERE/'train-identity-freeze.json'));tokpath=HERE.parent/'tokenizer-assets/official-31b-ct-v2-tokenizer'
 tok=load_tokenizer(str(tokpath));template=(tokpath/'chat_template.jinja').read_text();schema=json.load(open(ROOT/'training/configs/harness_tools.json'))
 census=[]
 for item in freeze['selected']:
  row=json.load(open(DATA/'source-rows'/f"{item['instance_id']}.json"))['rows'][0]['row']
  ms=[dict(role=m['role'],content=m['content']) for m in row['messages']]
  ids=tok.apply_chat_template(ms,tools=schema,chat_template=template,tokenize=True,add_generation_prompt=False,enable_thinking=True,preserve_thinking=True)
  census.append(dict(instance_id=item['instance_id'],source_full_template_tokens=len(ids["input_ids"]) if hasattr(ids,"keys") else len(ids),source_messages=len(ms),source_structured_production_edits=len(actions(row)),format='Teacher-native OpenHands XML as text in actual 31B CTv2 template; diagnostic length only, not competition-protocol training data'))
 dump(HERE/'source-length-census.json',census)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(replay,freeze['selected']))
 encoded=[]
 for r in results:
  if not r.get('replay_succeeded'):continue
  tx=json.load(open(r['transaction']));r['replay_full_template_tokens']=count_template_tokens(tx,tok,schema,chat_template=template)
  if r['replay_full_template_tokens']<=8192:
   enc=encode_transaction(tx,tok,schema,max_seq_len=8192,chat_template=template,tokenizer_revision='official-kaggle-mounted-gemma4-31b-ct-v2');encoded.append(enc);r['encoded_tokens']=len(enc['input_ids']);r['supervised_tokens']=sum(t!=-100 for t in enc['labels'])
 dump(HERE/'replay-results.json',results)
 (DATA/'encoded-lego-31b.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in encoded))
 print(json.dumps(dict(selected=len(census),replayed=sum(bool(x.get('replay_succeeded')) for x in results),encoded=len(encoded),encoded_file=str(DATA/'encoded-lego-31b.jsonl')),indent=2))
if __name__=='__main__':main()
