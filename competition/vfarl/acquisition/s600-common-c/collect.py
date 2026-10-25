"""Acquire and replay manually specified Addict contracts for S600 common data.

This collector does not author tests or change the shared admission catalog.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import argparse,difflib,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
ACQ=Path(__file__).resolve().parent
REPLAY=ROOT/'competition/vfarl/replay'
ART=REPLAY/'artifacts/s600-common-c'
REV='77cab9055d42ab4a5c25c89a8f937096db13558e'
PAGE=ROOT/'competition/vfarl/acquisition/expand-objects/train-page-46787.json'
PAGE_SHA='4799f722e1fa2b6d26c51e1e4281ce85a2a112e1af3d375d294d4d1fe25c250a'
UPSTREAM=REPLAY/'artifacts/expand-objects/addict-upstream'
UPSTREAM_SHA='75284f9593dfb929cadd900aff9e35e7c7aec54b'
SOURCE_FILE='addict/addict.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def label(p):return str(Path(p).relative_to(ROOT))
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def helper():
 s=importlib.util.spec_from_file_location('source_prep',REPLAY/'prepare_sfr.py')
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.ART=ART;return m

# Explicit source-contract review, not a keyword classifier. Same-function
# variants and freeze/unfreeze variants are not counted as separate tasks.
SPECS=[
('constructor','func_pm_ctrl_invert_if__1bxgjdlm','S','S1','Dict.__init__','mapping_pair_iterable_shape_dispatch',4,26,
 'Construct from individual key/value tuples, an empty argument and another tuple.',
 'Pair and iterable branches are exchanged; a scalar pair is iterated as a sequence of pairs.',
 'Restore direct pair assignment to the shape-specific branch and iterate only general pair iterables.',
"""from addict import Dict
d=Dict((0, {'inner': 1}), [], (2, 3))
print('constructor_pairs',dict(d),type(d[0]).__name__)
assert d[0].inner==1 and d[2]==3 and isinstance(d[0],Dict)
other=Dict([('x',{'y':4})],z=5)
assert other.x.y==4 and other.z==5
print('constructor_shapes_and_nested_values',True)"""),
('hook','func_basic__e4m4u8rc','S','S5','Dict._hook','nested_value_conversion_and_leaf_identity',56,67,
 'Convert mappings nested in lists/tuples while preserving arbitrary scalar leaves.',
 'Mappings stay plain dictionaries, recursive calls use the whole container and scalar leaves become None.',
 'Construct the Dict class for mappings, recurse on each element and retain ordinary leaves.',
"""from addict import Dict
marker=object()
value={'leaf':1,'seq':[{'x':2},marker],'pair':({'y':3},)}
converted=Dict._hook(value)
print('hook_root_type',type(converted).__name__)
assert isinstance(converted,Dict) and converted.leaf==1
assert isinstance(converted.seq,list) and isinstance(converted.seq[0],Dict)
assert converted.seq[0].x==2 and converted.seq[1] is marker
assert isinstance(converted.pair,tuple) and converted.pair[0].y==3
print('nested_conversion_and_leaf_identity',True)"""),
('to-dict','func_basic__tuhagvw6','S','S1','Dict.to_dict','plain_projection_mixed_value_and_container_fidelity',75,90,
 'Export nested mappings, mixed sequence elements and string leaves as ordinary dictionaries.',
 'Scalar sequence elements are dropped, nested mappings are not projected and strings become lengths.',
 'Project nested Dict objects to plain dict while retaining all other values and sequence types.',
"""from addict import Dict
value={'items':[{'n':2},3,'text'],'pair':({'k':4},),'label':'word'}
plain=Dict(value).to_dict()
print('plain_projection',plain)
assert plain==value and type(plain) is dict
assert type(plain['items'][0]) is dict and type(plain['pair'][0]) is dict
assert type(plain['pair']) is tuple
print('projection_value_and_container_fidelity',True)"""),
('shallow-copy','func_basic__ide4hfpv','F','F2','Dict.copy','shallow_owner_copy_preserves_nested_reference',87,101,
 'Copy a Dict holding a mutable list.',
 'copy() deep-copies and silently detaches the nested value.',
 'Use the shallow copy protocol for copy() while keeping deepcopy() separate.',
"""from addict import Dict
shared=[]
d=Dict()
d['payload']=shared
copied=d.copy()
print('copy_identity',copied is d,'nested_shared',copied['payload'] is shared)
assert copied is not d and isinstance(copied,Dict)
assert copied['payload'] is shared
copied['payload'].append(7)
assert d['payload']==[7]
copied['only_copy']=8
assert 'only_copy' not in d
print('shallow_ownership_boundary',True)"""),
('freeze','func_basic__06sj084c','F','F1','Dict.freeze','freeze_argument_propagates_to_nested_mapping',142,164,
 'Freeze a nested Dict, update existing keys, then unfreeze.',
 'The requested flag is inverted both on the owner and during recursive propagation.',
 'Store and propagate the same shouldFreeze argument; retain existing-key replacement.',
"""from addict import Dict
d=Dict({'inner':{'old':1},'old':2})
d.freeze()
blocked=[]
for obj in (d,d.inner):
 try:
  obj.new=9
  blocked.append(False)
 except KeyError:
  blocked.append(True)
print('frozen_new_key_blocks',blocked)
assert blocked==[True,True]
d.old=3
d.inner.old=4
assert d.old==3 and d.inner.old==4
d.unfreeze()
d.new=5
d.inner.new=6
assert d.new==5 and d.inner.new==6
print('freeze_argument_and_nested_policy',True)"""),
('setdefault','func_basic__9sg9rq7f','G','G-mapping_default_contract','Dict.setdefault','presence_sensitive_single_default_insertion',140,156,
 'Request defaults for an existing and missing key with an identity-bearing default.',
 'Existing keys return the fallback and missing keys store their key name instead.',
 'Return existing values unchanged; otherwise insert and return the exact default object.',
"""from addict import Dict
fallback=[]
d=Dict(k='existing')
old=d.setdefault('k',fallback)
new=d.setdefault('new',fallback)
print('default_results',old,new,dict(d))
assert old=='existing' and d.k=='existing'
assert new is fallback and d['new'] is fallback
assert d.setdefault('new',object()) is fallback
print('presence_and_default_identity',True)"""),
('addition','func_basic__9tjra5yk','R','R3','Dict.__add__','empty_addition_identity_nonempty_type_error',46,60,
 'Add an arbitrary object to an empty Dict and add to a populated Dict.',
 'The operator asks its right operand for keys and may return TypeError instead of raising.',
 'Return the exact right operand for empty Dict; raise original TypeError for nonempty unsupported addition.',
"""from addict import Dict
right=object()
value=Dict()+right
print('empty_addition_identity',value is right)
assert value is right
try:
 Dict(a=1)+2
 raise AssertionError('nonempty addition did not raise')
except TypeError as exc:
 assert "'Dict' and 'int'" in str(exc)
print('empty_identity_nonempty_exception_protocol',True)"""),
('missing','func_basic__90wh3hv9','F','F2','Dict.__missing__','lazy_child_parent_binding_commits_only_on_write',64,77,
 'Read a missing child, write through it, then access a missing key under freeze.',
 'An unfrozen read raises and a returned child is detached from the parent.',
 'Create missing children bound to parent/key only when unfrozen; reject missing keys when frozen.',
"""from addict import Dict
d=Dict()
child=d.later
assert d=={}
child.value=8
print('lazy_parent_identity',d.get('later') is child,dict(d))
assert d['later'] is child and d.later.value==8
d.freeze()
try:
 d.missing
 raise AssertionError('frozen missing access accepted')
except KeyError:
 pass
print('lazy_parent_binding_frozen_missing',True)"""),
('update','lm_rewrite__lmshg2zv','S','S5','Dict.update','recursive_merge_preserves_unmentioned_subtree',98,129,
 'Merge overlapping nested mappings plus top-level keyword values.',
 'Constructor-style replacement loses unmentioned nested entries.',
 'Accumulate positional mapping and overriding keywords, recursing only when old/new values are mappings.',
"""from addict import Dict
d=Dict({'branch':{'a':1,'keep':2},'leaf':0})
child=d.branch
d.update({'branch':{'a':3,'new':4}},leaf=5)
print('recursive_merge',d.to_dict(),'subtree_identity',d.branch is child)
assert d.to_dict()=={'branch':{'a':3,'keep':2,'new':4},'leaf':5}
assert d.branch is child
d.update({'leaf':6},leaf=7)
assert d.leaf==7
print('recursive_merge_and_keyword_override',True)"""),
('union','func_basic__isj642oc','R','R3','Dict.__or__','mapping_union_new_result_and_notimplemented_dispatch',121,142,
 'Union two Dict operands and explicitly reject an unsupported right operand.',
 'Supported mappings return NotImplemented, inverting the operand acceptance contract.',
 'Reject nonmappings and return a new merged Dict for supported operands.',
"""from addict import Dict
left=Dict(a=1)
right=Dict(b=2)
merged=left|right
print('mapping_union',merged,type(merged).__name__)
assert isinstance(merged,Dict) and merged=={'a':1,'b':2}
assert left=={'a':1} and right=={'b':2}
assert merged is not left and merged is not right
assert Dict.__or__(left,object()) is NotImplemented
print('union_dispatch_and_new_result',True)""")]

def inverse_edits(patch):
 edits=[];old=[];new=[]
 for line in patch.splitlines(keepends=True):
  if line.startswith('@@ '):
   if old or new:edits.append((''.join(old),''.join(new)))
   old=[];new=[]
  elif line.startswith(('diff --git ','index ','--- ','+++ ','\\')):continue
  elif line.startswith(' '):old.append(line[1:]);new.append(line[1:])
  elif line.startswith('+'):old.append(line[1:])
  elif line.startswith('-'):new.append(line[1:])
 if old or new:edits.append((''.join(old),''.join(new)))
 return edits

def acquire_one(spec):
 name,suffix,adapter,sub,symbol,family,start,end,trigger,wrong,causal,observation=spec
 h=helper();page=json.loads(PAGE.read_text())
 raw=next(x for x in page['rows'] if x['row']['instance_id'].endswith(suffix));row=raw['row']
 index=ROOT/'competition/vfarl/acquisition/general/smith-task-index.jsonl'
 identity=next(json.loads(line) for line in index.open() if json.loads(line)['instance_id']==row['instance_id'])
 assert identity['row_idx']==raw['row_idx'] and identity['repo']==row['repo']
 assert identity['split']=='train' and identity['revision']==REV
 assert row['problem_statement'].strip() and row.get('test_patch','')==''
 key='s600-c-addict-'+name
 original=ACQ/(row['instance_id']+'.task.json')
 save(original,{'row':row,'source':identity,'saved_page':label(PAGE),'saved_page_sha256':PAGE_SHA})
 remote=subprocess.check_output(['git','ls-remote','https://github.com/'+row['repo']+'.git','refs/heads/'+row['instance_id']],text=True).strip()
 commit,ref=remote.split()
 assert len(commit)==40 and ref=='refs/heads/'+row['instance_id']
 source=ART/(key+'-source');acquired=h.acquire(row['repo'],commit,source)
 reconstructed=(source/SOURCE_FILE).read_text();edits=inverse_edits(row['patch'])
 for old,new in edits:
  assert reconstructed.count(old)==1,(key,old)
  reconstructed=reconstructed.replace(old,new)
 assert reconstructed==(UPSTREAM/SOURCE_FILE).read_text(),key
 targets=sorted({n.split('::',1)[0] for n in row['FAIL_TO_PASS']+row['PASS_TO_PASS']})
 assert targets==['test_addict.py']
 original_test=UPSTREAM/targets[0]
 assert sha(original_test)=='175c9238dea0a952ca84968b25b87b9331ecbb49a9abad764b4588d5c484c28f'
 if (source/targets[0]).exists():
  assert (source/targets[0]).read_bytes()==original_test.read_bytes();test_patch=''
 else:test_patch=''.join(difflib.unified_diff([],original_test.read_text().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+targets[0]))
 task={k:row[k] for k in ['instance_id','repo','problem_statement','FAIL_TO_PASS','PASS_TO_PASS']}
 task.update(base_commit=commit,patch='',test_patch=test_patch,verification_target_files=targets)
 adapted=ART/(key+'-task.json');save(adapted,task)
 review=' '.join([trigger,wrong,causal])
 recipe=h.write_reviewed_recipe(key,SOURCE_FILE,'addict.addict',start,end,edits,observation,review,
  causal+' The source-bound inline observation now passes. Patch submitted; the harness separately runs clean full-file task verification.')
 source_license={'license':'MIT','path':'competition/vfarl/acquisition/expand-objects/addict.LICENSE','source_commit':UPSTREAM_SHA}
 source_license['sha256']=sha(ROOT/source_license['path'])
 dataset_license={'license':'MIT','path':'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt','revision':REV}
 dataset_license['sha256']=sha(ROOT/dataset_license['path'])
 prep={'instance_id':row['instance_id'],'key':key,'original_task':label(original),'original_task_sha256':sha(original),
  'adapted_task':label(adapted),'workspace':label(source),'source_dataset':'SWE-bench/SWE-smith-py',
  'source_train_revision':REV,'physical_split':'train','course_membership':'S600_common','primary_adapter':adapter,
  'submechanism':sub,'canonical_repo':'mewwts/addict','repair_family_id':'addict::'+symbol+'::'+family,
  'snapshot_group_id':'mewwts/addict::'+UPSTREAM_SHA,'execution_commit':commit,'git_ls_remote':remote,'buggy_source':acquired,
  'upstream_source':{'repo':'mewwts/addict','resolved_revision':UPSTREAM_SHA,
   'archive_path':'competition/vfarl/replay/artifacts/expand-objects/addict-upstream.tgz','archive_sha256':sha(REPLAY/'artifacts/expand-objects/addict-upstream.tgz')},
  'recipe':label(recipe),'recipe_sha256':sha(recipe),'F2P_count':len(row['FAIL_TO_PASS']),'P2P_count':len(row['PASS_TO_PASS']),
  'required_node_identifiers_verbatim_preserved':True,'source_requirements_relaxed':False,'original_patch_semantics':'bug_introducing',
  'verification_target_files':targets,'verification_assets':[{'path':targets[0],'sha256':sha(original_test),'source_revision':UPSTREAM_SHA,'source_path':label(original_test)}],
  'mechanism_evidence':{'trigger':trigger,'base_wrong_behavior':wrong,'production_symbols':symbol,'causal_change':causal,
   'discriminating_observation':{'kind':'explicitly_reviewed_source_bound_inline_execution','description':observation},
   'bug_patch_direction':'introduces_defect_not_a_repair_target'},'source_license':source_license,'dataset_license':dataset_license}
 save(REPLAY/(key+'-source-preparation.json'),prep)
 print(json.dumps({'prepared':key,'commit':commit,'F2P':prep['F2P_count'],'P2P':prep['P2P_count']}),flush=True)
 return prep

def prepare_all():
 assert sha(PAGE)==PAGE_SHA
 ART.mkdir(parents=True,exist_ok=True)
 with ThreadPoolExecutor(max_workers=3) as pool:prepared=list(pool.map(acquire_one,SPECS))
 save(ACQ/'source-preparation.json',{'tasks':prepared})

def run_all():
 prepared=json.loads((ACQ/'source-preparation.json').read_text())['tasks']
 def run_one(prep):
  output=ART/(prep['key']+'-run-001');log=ART/(prep['key']+'-run-001.log')
  cmd=[sys.executable,str(REPLAY/'run_official.py'),'--task',str(ROOT/prep['adapted_task']),'--workspace',str(ROOT/prep['workspace']),
   '--recipe',str(ROOT/prep['recipe']),'--output',str(output),'--verify','--verify-baseline','--source-layout','.',
   '--task-site-packages',str(REPLAY/'.venv-sfr/lib/python3.12/site-packages')]
  if not (output/'run-summary.json').exists():
   with log.open('w') as stream:result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
   if result.returncode:
    print(json.dumps({'run':prep['key'],'returncode':result.returncode,'log':label(log)}),flush=True);return
  h=helper();target=h.record_result(prep['key'],output.name);report=json.loads(target.read_text())
  report.update(source_license=prep['source_license'],dataset_license=prep['dataset_license'],course_membership='S600_common',
   source_limitations='Publisher description preserved verbatim; no native-agent or on-policy claim.',
   runtime_equivalence_limit='Official local SubprocessManager CPU backend; not competition Docker/Linux equivalence.')
  save(target,report)
  print(json.dumps({'run':prep['key'],'resolved':report['resolved_by_local_official_verify_task'],
   'baseline_failures':report['reports']['baseline']['failures'],'repaired_tests':report['reports']['repaired']['tests'],'report':label(target)}),flush=True)
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run_one,prepared))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['prepare','run']);a=p.parse_args()
 prepare_all() if a.operation=='prepare' else run_all()

