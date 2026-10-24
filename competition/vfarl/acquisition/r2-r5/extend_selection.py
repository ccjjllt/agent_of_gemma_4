"""Materialize the R curriculum's reviewed public-source identity selection."""
import json,hashlib,importlib.util,concurrent.futures
from pathlib import Path
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fetch',OUT/'fetch_selected.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
GROUPS={
'R1':['agronholm__typeguard.b6a7e438.func_pm_remove_loop__uosegp28','agronholm__typeguard.b6a7e438.func_pm_remove_wrapper__q7zhg0en','pylint-dev__astroid.b114f6b5.pr_2515','pydantic__pydantic.acb0f10f.func_pm_class_rm_base__qy6wl2ce'],
'R2':['encode__starlette.db5063c2.lm_rewrite__12xv5ppk','facebookresearch__hydra.0f03eb60.pr_1915','jd__tenacity.0d40e76f.pr_406','cool-RR__PySnooper.57472b46.lm_rewrite__urhyaplv'],
'R3':['dask__dask.5f61e423.pr_8185','benoitc__gunicorn.bacbf8aa.func_pm_ctrl_shuffle__wr94o292','pydantic__pydantic.acb0f10f.pr_10242','pylint-dev__astroid.b114f6b5.lm_rewrite__kxxx3vrz'],
'R4':['pylint-dev__astroid.b114f6b5.pr_2328','seperman__deepdiff.ed252022.pr_422','marshmallow-code__marshmallow.9716fc62.pr_2789','pydantic__pydantic.acb0f10f.pr_6456'],
'R5':['dask__dask.5f61e423.pr_9573','iterative__dvc.1d6ea681.pr_7628','encode__starlette.db5063c2.pr_2733','python-trio__trio.cfbbe2c1.pr_2932']}
CACHE=json.load(open(OUT.parents[3]/'research/016/data-preparation/smith-next-1000-acquisition.json'))
LOOKUP={r['instance_id']:r for r in CACHE['rows']}
def flat(c):
 if isinstance(c,str):return c
 if isinstance(c,list):return '\n'.join(flat(x) for x in c)
 if isinstance(c,dict):return c.get('text','')
 return ''
SELECT=[]
for sub,ids in GROUPS.items():
 for iid in ids:
  r=LOOKUP[iid];source=OUT.parents[3]/r['local_path'];b=source.read_bytes();d=json.loads(b)
  assert hashlib.sha256(b).hexdigest()==r['sha256']
  assert d['instance_id']==iid and d['traj_id']==r['traj_id']
  assert r['revision']=='08e109b4a59eaeebf80e4675cd125d42e7ac99a4' and r['source_split']=='tool'
  assert iid in a.INDEX
  target=OUT/'artifacts/native'/f'{iid}.json';target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b)
  row=dict(r,primary_adapter='R',submechanism=sub,physical_source_split='tool',usage='train-candidate',native_artifact_path=str(target),native_artifact_sha256=hashlib.sha256(b).hexdigest())
  SELECT.append(row)
(OUT/'selection.json').write_text(json.dumps({'source_dataset':CACHE['source_dataset'],'source_revision':CACHE['source_revision'],'source_physical_split':'tool','purpose':'train-candidate','groups':GROUPS,'rows':SELECT},indent=2))
if __name__=='__main__':
 wanted=[r['instance_id'] for r in SELECT if not (a.RAW/(r['instance_id']+'.json')).exists()]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:
  for r in p.map(a.get,wanted):print('fetched',r['instance_id'],r['response_bytes'],flush=True)
 print('Selected identities materialized. Resolve environments, retain licenses, then run build_reviews.py to write the unified collector artifact; transport receipts remain separate.',flush=True)
