"""Obtain exact public TRAIN definitions for the remaining S/F pilot slots."""
import json,hashlib,sys,concurrent.futures
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fetch_selected_tasks as f
ROOT=f.ROOT
# A source-native row's patch is not used as a solution or task join; task definitions come from SWE-smith-py.
for rec in map(json.loads,open('/private/tmp/gemma4-vfarl-seed/SWE-bench__SWE-smith-trajectories__raw.jsonl')):
 row=rec['row'];iid=row['instance_id']
 if iid in f.CACHE:continue
 repo=iid.split('.')[0].replace('__','/',1);p=ROOT/(iid+'.cached-native-source.json');b=json.dumps(row,ensure_ascii=False,sort_keys=True).encode();p.write_bytes(b)
 f.CACHE[iid]={'instance_id':iid,'repo':repo,'local_path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'dataset':'SWE-bench/SWE-smith-trajectories','revision':'08e109b4a59eaeebf80e4675cd125d42e7ac99a4','source_split':'tool','traj_id':row['traj_id'],'resolved':bool(row['resolved']),'full_buggy_commit_from_cached_provenance':None,'cached_license_provenance':None,'native_source_patch_quarantined':True,'original_response_sha256':rec['response_sha256'],'row_idx':rec['row_idx']}
ITEMS={
'S1':['marshmallow-code__apispec.8b421526.func_pm_op_swap__53fevqi2','martinblech__xmltodict.0952f382.pr_345','marshmallow-code__marshmallow.9716fc62.lm_rewrite__nqk1hyjp','pydantic__pydantic.acb0f10f.pr_8316'],
'S3':['mido__mido.a0158ff9.func_pm_ctrl_shuffle__cyg5oo56','agronholm__typeguard.b6a7e438.func_pm_ctrl_shuffle__rw2zp0ga'],
'S4':['agronholm__exceptiongroup.0b4f4937.func_pm_remove_loop__5rijazp0','hukkin__tomli.443a0c1b.func_pm_remove_cond__i5pq0aey','seperman__deepdiff.ed252022.lm_rewrite__c4ilq2zb','matthewwithanm__python-markdownify.6258f5c3.lm_rewrite__iu00e4fc'],
'S5':['andialbrecht__sqlparse.e57923b3.pr_746','graphql-python__graphene.82903263.pr_1504'],
'F1':['marshmallow-code__apispec.8b421526.pr_851','encode__starlette.db5063c2.pr_2280','pydantic__pydantic.acb0f10f.pr_8965'],
'F3':['pyupio__safety.7654596b.func_pm_remove_cond__rm170rw2','pudo__dataset.5c2dc8d3.pr_423'],
'F5':['python-hyper__h11.bed0dd4a.func_pm_remove_cond__h6ghpwe2','encode__starlette.db5063c2.pr_2061'],
}
if __name__=='__main__':
 items=[(k,x) for k,v in ITEMS.items() for x in v]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:out=list(ex.map(f.fetch,items))
 (ROOT/'remainder-acquisition.json').write_text(json.dumps({'rows':out},indent=2,ensure_ascii=False));print('acquired',sum('error' not in x for x in out),'of',len(out))
