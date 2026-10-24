"""Materialize specifically reviewed public train task/trajectory joins; no task commands are run."""
from pathlib import Path
import json,urllib.parse,urllib.request,concurrent.futures,time,hashlib
ROOT=Path(__file__).resolve().parent
IDS={
'S2':['adrienverge__yamllint.8513d9b9.func_pm_op_change_const__l4zuvyrl','hukkin__tomli.443a0c1b.pr_238','scanny__python-pptx.278b47b1.lm_rewrite__s5vm6lrb','pdfminer__pdfminer.six.1a8bd2f7.lm_rewrite__63n4v0qf'],
'S3':['graphql-python__graphene.82903263.lm_rewrite__w72ly9k3','adrienverge__yamllint.8513d9b9.pr_614'],
'S5':['graphql-python__graphene.82903263.func_pm_ctrl_invert_if__fyzvhfb5','kayak__pypika.1c9646f0.lm_rewrite__x1l7vmxq'],
'F1':['facebookresearch__hydra.0f03eb60.pr_2361'],
'F2':['mahmoud__boltons.3bfcfdd0.pr_375','facebookresearch__hydra.0f03eb60.pr_3009','sloria__environs.73c372df.combine_file__hl27l2aa'],
'F3':['mahmoud__boltons.3bfcfdd0.func_pm_ctrl_invert_if__48b5m32l','pallets__jinja.ada0a9a6.lm_rewrite__5sk0hj9z','pylint-dev__astroid.b114f6b5.pr_2001'],
'F4':['paramiko__paramiko.23f92003.lm_rewrite__5omeuvpt','tornadoweb__tornado.d5ac65c1.lm_rewrite__cnp2i98g','python-trio__trio.cfbbe2c1.lm_rewrite__o9uv20ij'],
'F5':['facebookresearch__hydra.0f03eb60.pr_1560']}
REV='77cab9055d42ab4a5c25c89a8f937096db13558e';DS='SWE-bench/SWE-smith-py'
INDEX={r['instance_id']:r for r in map(json.loads,open('competition/vfarl/acquisition/general/smith-task-index.jsonl'))}
CACHE={r['instance_id']:r for r in json.load(open('research/016/data-preparation/smith-next-1000-acquisition.json'))['rows']}
def fetch(item):
 mech,iid=item;m=INDEX[iid];cached=CACHE[iid];idx=m['row_idx']
 url='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode({'dataset':DS,'config':'default','split':'train','offset':idx,'length':1,'revision':REV})
 for retry in range(3):
  try:
   with urllib.request.urlopen(url,timeout=35) as r:b=r.read(1000001);rev=r.headers.get('x-revision')
   if len(b)>1000000 or rev!=REV:raise RuntimeError('response size or revision mismatch')
   row=json.loads(b)['rows'][0]['row']
   if row['instance_id']!=iid or row['repo']!=m['repo']:raise RuntimeError('ID/repo mismatch')
   record={'source':DS,'revision':REV,'physical_split':'train','row_idx':idx,'source_url':url,'response_sha256':hashlib.sha256(b).hexdigest(),'row':row}
   taskpath=ROOT/(iid+'.task.json');taskpath.write_text(json.dumps(record,indent=2,ensure_ascii=False))
   raw=Path(cached['local_path']).read_bytes()
   if hashlib.sha256(raw).hexdigest()!=cached['sha256']:raise RuntimeError('source raw cache hash mismatch')
   traj=ROOT/(iid+'.trajectory.json');traj.write_bytes(raw)
   rec={'instance_id':iid,'primary_mechanism_proposed':mech,'canonical_repo':cached['repo'],'task_path':str(taskpath),'task_sha256':hashlib.sha256(taskpath.read_bytes()).hexdigest(),'trajectory_path':str(traj),'trajectory_sha256':cached['sha256'],'trajectory_dataset':cached['dataset'],'trajectory_revision':cached['revision'],'trajectory_physical_split':'tool','trajectory_id':cached['traj_id'],'publisher_resolved':cached['resolved'],'cached_full_buggy_commit':cached['full_buggy_commit_from_cached_provenance'],'cached_license_provenance_path':cached['cached_license_provenance'],'source_cache_meta':cached,'target_replay_completed':False}
   print('acquired',iid,flush=True);return rec
  except Exception as e:
   if retry==2:return {'instance_id':iid,'primary_mechanism_proposed':mech,'error':str(e)}
   time.sleep(2**retry)
if __name__=='__main__':
 items=[(k,i) for k,v in IDS.items() for i in v]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:out=list(ex.map(fetch,items))
 (ROOT/'acquisition-manifest.json').write_text(json.dumps({'date':'2026-10-24','task_dataset':DS,'task_revision':REV,'task_split':'train','trajectory_dataset':'SWE-bench/SWE-smith-trajectories','trajectory_revision':'08e109b4a59eaeebf80e4675cd125d42e7ac99a4','trajectory_physical_split':'tool','source_commands_executed':False,'rows':out},indent=2,ensure_ascii=False))
 print('success',sum('error' not in x for x in out),'count',len(out),flush=True)
