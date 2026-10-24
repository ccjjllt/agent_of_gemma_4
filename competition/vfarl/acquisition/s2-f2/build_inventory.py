"""Assemble the manually reviewed S/F pilot inventory from exact public task records.

This packages source data only. It does not execute task commands, run tests, replay tools,
or promote publisher outcomes to target-protocol verification.
"""
from pathlib import Path
import json,hashlib,collections,re
ROOT=Path(__file__).resolve().parent

def sha(b):return hashlib.sha256(b).hexdigest()

def assemble():
 notes={**json.load(open(ROOT/'curation-notes.json')),**json.load(open(ROOT/'remainder-curation-notes.json'))}
 cache={}
 for p in [ROOT/'acquisition-manifest.json',ROOT/'additional-acquisition.json',ROOT/'second-additional-acquisition.json',ROOT/'third-additional-acquisition.json',ROOT/'remainder-acquisition.json',ROOT/'final-replacement-acquisition.json',ROOT/'slugify-replacement-acquisition.json',ROOT/'environment-replacement-acquisition.json']:
  if not p.exists():continue
  for x in json.load(open(p))['rows']:
   if 'source_cache_meta' in x:cache[x['instance_id']]=x['source_cache_meta']
 evidence={}
 for fn in ['upstream-license-provenance.json','raw-upstream-license-provenance.json','remainder-upstream-license-provenance.json']:
  for x in json.load(open(ROOT/fn)):
   if 'error' not in x:evidence[x['repo']]=x
 licenses={'adrienverge/yamllint':'GPL-3.0','hukkin/tomli':'MIT','pdfminer/pdfminer.six':'MIT','graphql-python/graphene':'MIT','spulec/freezegun':'Apache-2.0','kayak/pypika':'Apache-2.0','facebookresearch/hydra':'MIT','mahmoud/boltons':'BSD-3-Clause','tox-dev/pipdeptree':'MIT','pallets/jinja':'BSD-3-Clause','seperman/deepdiff':'MIT','benoitc/gunicorn':'MIT','tornadoweb/tornado':'Apache-2.0','python-trio/trio':'MIT OR Apache-2.0','laspy/laspy':'BSD-2-Clause','PennyLaneAI/pennylane':'Apache-2.0','marshmallow-code/apispec':'MIT','marshmallow-code/marshmallow':'MIT','pydantic/pydantic':'MIT','agronholm/typeguard':'MIT','agronholm/exceptiongroup':'MIT','matthewwithanm/python-markdownify':'MIT','andialbrecht/sqlparse':'BSD-3-Clause','encode/starlette':'BSD-3-Clause','pyupio/safety':'MIT','python-hyper/h11':'MIT','un33k/python-slugify':'MIT'}
 pins_path=ROOT/'environment-identities.json'
 pins={x['instance_id']:x for x in json.load(open(pins_path))['results']} if pins_path.exists() else {}
 exact_license_path=ROOT/'environment-replacement-license-provenance.json'
 exact_licenses={x['instance_id']:x for x in json.load(open(exact_license_path))['rows']} if exact_license_path.exists() else {}
 out=[]
 for iid,n in notes.items():
  taskpath=ROOT/(iid+'.task.json');raw=taskpath.read_bytes();d=json.loads(raw);r=d['row'];m=cache.get(iid)
  is_smith='SWE-smith' in d.get('source',d.get('dataset',''))
  repo=m['repo'] if m else r['repo']
  snapshot_match=re.search(r'\.([0-9a-f]{8})(?:\.|$)',iid) if is_smith else None
  base=r.get('base_commit') or (snapshot_match.group(1) if snapshot_match else None)
  if not base:raise ValueError('Missing explicit source snapshot identity: '+iid)
  family=repo+'::'+n[3].split(';')[0]
  record={'instance_id':iid,'canonical_repo':repo,'primary_adapter':n[0][0],'primary_mechanism':n[0],'repair_family_id':family,'snapshot_group_id':repo+'::'+base,'use_partition':'train_only','mechanism_status':'statically_supported_task_definition','agent_visible_trigger':n[1],'base_wrong_behavior':n[2],'production_symbols':n[3],'causal_change':n[4],'discriminating_observation':n[5],'review_notes':n[6],
   'source_task':{'dataset':d.get('source',d.get('dataset')),'revision':d['revision'],'physical_split':d.get('physical_split',d.get('split')),'row_idx':d['row_idx'],'source_url':d['source_url'],'path':str(taskpath),'sha256':sha(raw),'patch_direction':'bug_introducing' if is_smith else 'solution','patch_sha256':sha(r['patch'].encode()),'agent_problem_statement_field':'row.problem_statement','privileged_fields_excluded_from_policy':['patch','test_patch','FAIL_TO_PASS','PASS_TO_PASS','pr_description','interface','meta']},
   'environment':{'source_repo':r['repo'],'image_name':r['image_name'],'image_digest':None,'repo_snapshot_prefix':base if is_smith else None,'upstream_base_commit':r.get('base_commit'),'bug_branch':iid if is_smith else None,'buggy_commit':pins.get(iid,{}).get('buggy_commit'),'execution_commit':pins.get(iid,{}).get('buggy_commit') if is_smith else r.get('base_commit'),'fresh_remote_pin_evidence':str(pins_path) if is_smith else None,'fresh_remote_matches_cached_commit':pins.get(iid,{}).get('branch_matches_cached_commit'),'repo_snapshot_prefix_is_not_execution_commit':is_smith,'cached_buggy_commit':m.get('full_buggy_commit_from_cached_provenance') if m else None,'pin_status':'Fresh complete Smith bug-branch commit (or full V2 base commit) pins source execution identity. Image digest and a running target sandbox are not claimed.'},
   'license':{'dataset':'MIT' if is_smith else 'CC-BY-4.0','upstream':licenses[repo],'upstream_evidence':exact_licenses.get(iid,evidence.get(repo,{'path':str(ROOT/'laspy-license.txt'),'provenance':str(ROOT/'laspy-license-provenance.json')}))},
   'source_verification':{'FAIL_TO_PASS':r['FAIL_TO_PASS'],'PASS_TO_PASS_count':len(r['PASS_TO_PASS']),'contract_status':'exact publisher verification lists retained in full source task record; source data is not a locally measured pass'},
   'source_trajectory':None,'target_protocol_status':'not_executed_by_this_collector','six_tool_training_admitted':False,'verified_critical_pairs':0}
  if repo=='python-trio/trio':record['license']['dual_text_provenance']=str(ROOT/'trio-dual-license-provenance.json')
  if m:
   p=ROOT/(iid+'.trajectory.json');b=p.read_bytes()
   native=json.loads(b)
   patch_present=any(k in native for k in ['patch','model_patch'])
   record['source_trajectory']={'dataset':m['dataset'],'revision':m['revision'],'physical_split':'tool','trajectory_id':m['traj_id'],'path':str(p),'sha256':sha(b),'publisher_resolved':m['resolved'],'patch_column_present':patch_present,'patch_column_used':False,'patch_column_policy':'All source-native patch/model_patch columns are globally excluded as solutions; absent cached projections remain explicitly absent. Authoritative task diffs are separate.','review_scope':'Exact task causal diff is manually adjudicated. Selected source-native causal spans cross-read; no exhaustive assistant loss-span admission.','positive_loss_spans':[]}
   if sha(b)!=m['sha256']:raise ValueError('Cached source bytes changed while assembling '+iid)
  if iid.startswith('pdfminer__'):
   record['source_trajectory']['eligibility']='excluded: cached source teacher inserted filename-specific hardcoded answers and imports from tests'
  if iid.startswith('graphql-python__graphene.82903263.lm_rewrite__w72'):
   record['source_trajectory']['eligibility']='not admitted: teacher sort=False edit fails to establish the complete source mapping/creation-order contract'
  out.append(record)
 report={'schema_version':'vfarl-sf-pilot-task-inventory-v1','date':'2026-10-24','scope':'40 exact public training task definitions, each statically reviewed from its source problem and exact causal task diff. Replaces trace-only public-source seeds rather than guessing environment joins.','unique_tasks':len(out),'unique_repair_families':len(set(x['repair_family_id'] for x in out)),'primary_adapter_counts':dict(collections.Counter(x['primary_adapter'] for x in out)),'submechanism_counts':dict(collections.Counter(x['primary_mechanism'] for x in out)),'source_dataset_counts':dict(collections.Counter(x['source_task']['dataset'] for x in out)),'source_native_trajectory_count':sum(x['source_trajectory'] is not None for x in out),'source_native_publisher_resolved_count':sum(bool(x['source_trajectory'] and x['source_trajectory']['publisher_resolved']) for x in out),'target_replay_note':'This collection does not assert target replay. Root-run official tool/verifier evidence for any selected task must be joined separately.','heldout_or_competition_task_ingestion':False,'competition_repositories_in_selected_pool':False,'source_commands_executed_by_this_collector':False,'gpu_or_paid_sampling_started':False,'tasks':out}
 (ROOT/'reviewed-sf40.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 (ROOT/'reviewed-sf40.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out))
 return report
if __name__=='__main__':
 r=assemble();print(json.dumps({k:v for k,v in r.items() if k!='tasks'},indent=2,ensure_ascii=False))
