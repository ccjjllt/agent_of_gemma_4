"""Acquire and replay explicitly reviewed S3/S4 public-train contracts.

The plan below is a curator-reviewed list of distinct production functions, not
an automatic classifier or a new test suite. Observations run through the real
competition tools; original upstream files provide the unchanged verification.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REPLAY = ROOT / 'competition/vfarl/replay'
ART = REPLAY / 'artifacts/s600-specialist-a'
REVISION = '77cab9055d42ab4a5c25c89a8f937096db13558e'
UPSTREAM = REPLAY / 'artifacts/expand-cache/funcy-ttl-upstream'
UPSTREAM_SHA = '207a7810c216c7408596d463d3f429686e83b871'

# suffix, primary contract, actual changed symbol, path, observable contract.
PLANS = [
 ('877jp032','S3','flip','colls','dict_and_pair_key_value_inversion',
  "from funcy import flip\nr=flip({'a':1,'b':2}); print('flipped',r)\nassert r=={1:'a',2:'b'}"),
 ('9lv9kz05','S3','select_keys','colls','mapping_key_predicate_inclusion',
  "from funcy import select_keys\nr=select_keys(lambda key:key.startswith('a'), {'a':1,'b':2}); print('selected',r)\nassert r=={'a':1}"),
 ('dd6c7obd','S3','project','colls','requested_existing_key_projection',
  "from funcy import project\nr=project({'a':1,'b':2}, ['a','missing']); print('projected',r)\nassert r=={'a':1}"),
 ('dwzm0qqo','S3','zip_dicts','colls','common_key_intersection_and_single_input',
  "from funcy import zip_dicts\nr=dict(zip_dicts({'a':1,'b':2},{'a':3,'c':4})); print('joined',r)\nassert r=={'a':(1,3)}\nassert dict(zip_dicts({'a':1}))=={'a':(1,)}"),
 ('hjs3d8y2','S3','group_by','seqs','classification_preserves_item_payload_and_order',
  "from funcy import group_by\nr=group_by(lambda value:value%2,[1,2,3,4]);print('grouped',dict(r))\nassert r=={0:[2,4],1:[1,3]}"),
 ('ku01y3x1','S3','group_values','seqs','pair_grouping_preserves_multiplicity',
  "from funcy import group_values\nr=group_values([('a',1),('a',1),('b',3),('a',2)]);print('grouped_values',dict(r))\nassert r=={'a':[1,1,2],'b':[3]}"),
 ('lnpujkau','S3','count_reps','seqs','occurrence_count_includes_duplicate_items',
  "from funcy import count_reps\nr=count_reps('ababa');print('counts',dict(r))\nassert r=={'a':3,'b':2}"),
 ('q2ws66ac','S3','distinct','seqs','stable_first_occurrence_and_keyed_deduplication',
  "from funcy import distinct\nr=list(distinct([1,1,2,1]));print('distinct',r)\nassert r==[1,2]\nassert list(distinct(['A','a','B','b'],key=str.lower))==['A','B']"),
 ('zv5wyb4y','S3','join_with','colls','per_key_value_aggregation_contract',
  "from funcy import join_with\nr=join_with(list,[{'a':1},{'a':2,'b':3}],strict=True);print('aggregated',r)\nassert r=={'a':[1,2],'b':[3]}"),
 ('4zw03ae8','S3','without','seqs','excluded_item_filter_preserves_survivor_order',
  "from funcy import without\nr=list(without([1,2,1,3],1));print('survivors',r)\nassert r==[2,3]"),
 ('7c1z033f','S3','lpluck','colls','one_requested_key_value_per_input_mapping',
  "from funcy import lpluck\nr=lpluck('id',[{'id':1},{'id':2}]);print('plucked',r)\nassert r==[1,2]"),
 ('cd2paqnl','S3','one','colls','exactly_one_matching_item_cardinality',
  "from funcy import one\nr=one(lambda x:x>1,[0,2]);print('exactly_one',r)\nassert r is True\nassert one(lambda x:x>1,[2,3]) is False"),
 ('ybawrzcz','S3','walk','colls','collection_pair_transform_preserves_container_protocol',
  "from funcy import walk\nr=walk(lambda pair:(pair[0].upper(),pair[1]+1),{'a':1,'b':2});print('walked',r)\nassert r=={'A':2,'B':3}"),
 ('4udfa765','S4','tree_nodes','tree','depth_first_node_traversal_resumes_parent_iterator',
  "from funcy import tree_nodes\nroot=[1,[2,[3]],4];r=list(tree_nodes(root));print('nodes',r)\nassert r==[root,1,[2,[3]],2,[3],3,4]"),
 ('jtj6rhfx','S4','tree_leaves','tree','depth_first_leaf_traversal_preserves_leaf_values',
  "from funcy import tree_leaves\nr=list(tree_leaves([1,[2,[3]],4]));print('leaves',r)\nassert r==[1,2,3,4]"),
 ('f6o2umxg','S4','reductions','seqs','prefix_fold_preserves_input_order_and_seed',
  "from funcy import reductions\nimport operator\nr=list(reductions(operator.add,[1,2,3],10));print('prefix_folds',r)\nassert r==[11,13,16]"),
 ('ew5ofxg0','S4','split_by','seqs','prefix_predicate_split_and_remainder_partition',
  "from funcy import split_by\na,b=split_by(lambda x:x<3,[1,2,3,0]);r=(list(a),list(b));print('prefix_and_rest',r)\nassert r==([1,2],[3,0])"),
 ('ylep5fzu','S4','takewhile','seqs','take_until_first_predicate_failure',
  "from funcy import takewhile\nr=list(takewhile(lambda x:x<3,[1,2,3,0]));print('prefix',r)\nassert r==[1,2]\nassert list(takewhile([1,2,0,3]))==[1,2]"),
 ('39c59v7o','S4','dropwhile','seqs','drop_only_initial_matching_prefix',
  "from funcy import dropwhile\nr=list(dropwhile(lambda x:x<3,[1,2,3,0]));print('suffix',r)\nassert r==[3,0]\nassert list(dropwhile([1,2,0,3]))==[0,3]"),
 ('wwxu5wgx','S4','_cut_seq','seqs','window_start_stride_and_tail_boundary',
  "from funcy import partition,chunks\nr=list(partition(2,[0,1,2,3,4]));print('full_windows',r)\nassert r==[[0,1],[2,3]]\nassert list(chunks(2,[0,1,2,3,4]))==[[0,1],[2,3],[4]]"),
 ('z66fs5y4','S4','nth','seqs','zero_based_indexed_or_iterator_lookup',
  "from funcy import nth\nr=nth(1,iter([10,20,30]));print('iterator_nth',r)\nassert r==20\nassert nth(-4,[10,20,30]) is None"),
 ('zkx8dyhw','S4','interpose','seqs','separator_insertion_without_leading_or_trailing_item',
  "from funcy import interpose\nr=list(interpose('-', [1,2,3]));print('interposed',r)\nassert r==[1,'-',2,'-',3]"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def label(path):
    return str(path.resolve().relative_to(ROOT))


def load_helper():
    spec=importlib.util.spec_from_file_location('sfr_acquisition',REPLAY/'prepare_sfr.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prepare():
    helper=load_helper();ART.mkdir(parents=True,exist_ok=True)
    index={row['instance_id']:row for row in map(json.loads,(ROOT/'competition/vfarl/acquisition/general/smith-task-index.jsonl').read_text().splitlines())}
    pages={}
    for page in sorted((ROOT/'competition/vfarl/acquisition/expand-cache/artifacts/pages').glob('Suor__funcy-*.json')):
        if page.name.endswith('.receipt.json'):continue
        receipt=json.loads(page.with_suffix('.receipt.json').read_text())
        if receipt['revision']!=REVISION or receipt['split']!='train' or receipt['sha256']!=sha(page):raise ValueError('Source page receipt mismatch')
        for entry in json.loads(page.read_text())['rows']:
            row=entry['row'];known=index[row['instance_id']]
            if known['row_idx']!=entry['row_idx'] or known['repo']!=row['repo']:raise ValueError('Allowed-train identity mismatch')
            pages[row['instance_id']]=(page,entry)
    ids=['Suor__funcy.207a7810.func_basic__'+p[0] for p in PLANS]
    output=subprocess.check_output(['git','ls-remote','https://github.com/swesmith/Suor__funcy.207a7810.git',*['refs/heads/'+iid for iid in ids]],text=True)
    refs={ref.removeprefix('refs/heads/'):commit for commit,ref in (line.split() for line in output.splitlines())}
    if set(refs)!=set(ids):raise ValueError('Not all immutable bug refs resolved')
    save(HERE/'bug-ref-resolution.json',{'url':'https://github.com/swesmith/Suor__funcy.207a7810.git','output':output,'response_sha256':hashlib.sha256(output.encode()).hexdigest()})
    tasks=[]
    for suffix,sub,symbol,mod,contract,observation in PLANS:
        iid='Suor__funcy.207a7810.func_basic__'+suffix;page,envelope=pages[iid];row=envelope['row']
        if not row['problem_statement'].strip():raise ValueError('This batch only uses nonempty publisher descriptions')
        original=HERE/(iid+'.task.json');save(original,{'row':row})
        key='s600-a-funcy-'+symbol.strip('_').replace('_','-');source=ART/(key+'-source')
        acquired=helper.acquire(row['repo'],refs[iid],source)
        required=row['FAIL_TO_PASS']+row['PASS_TO_PASS'];targets=sorted({x.split('::')[0] for x in required})
        assets=[];test_patch=[]
        for original_asset in sorted(UPSTREAM.rglob('*')):
            if not original_asset.is_file() or original_asset.is_symlink():continue
            relative=original_asset.relative_to(UPSTREAM)
            if not any(part in {'tests','test','testing'} for part in relative.parts) and str(relative)!='conftest.py':continue
            raw=original_asset.read_bytes();existing=source/relative
            if existing.exists() and existing.read_bytes()!=raw:raise ValueError('Original verification asset differs '+str(relative))
            assets.append({'path':str(relative),'sha256':hashlib.sha256(raw).hexdigest(),'already_present':existing.exists()})
            if not existing.exists():test_patch.extend(difflib.unified_diff([],raw.decode().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+str(relative)))
        if not all(any(a['path']==target for a in assets) for target in targets):raise ValueError('Missing original full target file')
        task={'instance_id':iid,'repo':row['repo'],'base_commit':refs[iid],'problem_statement':row['problem_statement'],'patch':'','test_patch':''.join(test_patch),'FAIL_TO_PASS':row['FAIL_TO_PASS'],'PASS_TO_PASS':row['PASS_TO_PASS'],'verification_target_files':targets}
        task_path=ART/(key+'-task.json');save(task_path,task)
        path='funcy/'+mod+'.py';old_source=(source/path).read_text();new_source=(UPSTREAM/path).read_text()
        def body(text):
            node=next(node for node in ast.parse(text).body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==symbol)
            return ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno]),node.lineno,node.end_lineno
        old,start,end=body(old_source);new,_,_=body(new_source)
        if old==new or old_source.count(old)!=1 or old_source.replace(old,new)!=new_source:raise ValueError('Single reviewed function does not completely explain source diff '+iid)
        command="python3 - <<'PYREPLAY'\nfrom pathlib import Path\nimport hashlib,importlib\nm=importlib.import_module('funcy."+mod+"')\na=Path(m.__file__).resolve();e=(Path.cwd()/"+repr(path)+").resolve()\nassert a==e,(a,e)\nprint('source_binding',a,'sha256',hashlib.sha256(a.read_bytes()).hexdigest())\n"+observation+"\nPYREPLAY"
        def action(tool,args,status='ok'):
            a={'tool':tool,'arguments':args,'expected_status':status}
            if tool=='run_command':a['reviewed_command_sha256']=hashlib.sha256(args['command'].encode()).hexdigest()
            return a
        recipe={'evidence_kind':'human_reviewed_off_policy_demonstration','reviewed_by':'Codex explicit S3/S4 contract, source diff and real observation review2026-10-25','review_notes':f'{symbol}: {contract}. Reviewed source defect and complete upstream function; no test/requirement edits, no mutation-ID inflation. The expected diagnostic failure is a reproduction action, not a policy error.','actions':[action('get_status',{}),action('read_file',{'filepath':path,'start_line':max(1,start-2),'end_line':end+2}),action('run_command',{'command':command},'error'),action('edit_file',{'filepath':path,'old_string':old,'new_string':new}),action('run_command',{'command':command}),action('get_status',{}),action('submit_patch',{})],'final_response':f'Restored {symbol}: {contract.replace("_"," ")}. The actual source-bound observation completed and the patch was submitted.'}
        recipe_path=REPLAY/(key+'-reviewed-recipe.json');save(recipe_path,recipe)
        tasks.append({'instance_id':iid,'key':key,'canonical_repo':'Suor/funcy','primary_adapter':'S','submechanism':sub,'repair_family_id':'funcy.'+symbol+'::'+contract,'snapshot_group_id':'Suor/funcy::'+UPSTREAM_SHA,'original_task':label(original),'original_task_sha256':sha(original),'source_dataset':'SWE-bench/SWE-smith-py','source_revision':REVISION,'physical_split':'train','row_idx':envelope['row_idx'],'source_page':label(page),'source_page_sha256':sha(page),'buggy_source':acquired,'execution_commit':refs[iid],'upstream_source':{'repo':'Suor/funcy','resolved_revision':UPSTREAM_SHA,'path':label(UPSTREAM)},'workspace':label(source),'adapted_task':label(task_path),'recipe':label(recipe_path),'production_symbol':symbol,'production_path':path,'mechanism_evidence':{'trigger':observation,'base_wrong_behavior':'Reviewed single-function source mutation violates '+contract,'causal_change':'Restore the original complete production function from the fixed upstream snapshot, keeping other bytes identical','discriminating_observation':'The same actual source-bound public API observation is run before and after the edit','classification_basis':contract,'source_bug_patch_is_solution':False},'original_FAIL_TO_PASS_count':len(row['FAIL_TO_PASS']),'original_PASS_TO_PASS_count':len(row['PASS_TO_PASS']),'verification_target_files':targets,'verification_assets':assets,'original_requirements_verbatim':True,'original_patch_semantics':'bug_introducing','license_evidence':{'source':{'path':label(UPSTREAM/'LICENSE'),'sha256':sha(UPSTREAM/'LICENSE'),'revision':UPSTREAM_SHA},'dataset':{'license':'MIT','path':'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt','sha256':sha(ROOT/'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt'),'revision':REVISION}}})
        print('prepared',iid,sub,symbol,flush=True)
    review={'source_dataset':'SWE-bench/SWE-smith-py','source_revision':REVISION,'physical_split':'train','scope':'Explicitly reviewed distinct S3/S4 production contracts; prepared candidates are not admitted successes. Shared snapshot remains grouped. Original publisher descriptions and all original required nodes/files retained.','tasks':tasks}
    save(HERE/'reviewed-candidates.json',review)


def prepare_boltons(plans=None,review_filename='reviewed-boltons-candidates.json',curation_prompts=None,primary_by_suffix=None,course_membership=None):
    helper=load_helper();ART.mkdir(parents=True,exist_ok=True)
    up=REPLAY/'artifacts/sfr/upstream/mahmoud__boltons.3bfcfdd0'
    upstream_sha='3bfcfdd04395b6cc74a5c0cdc72c8f64cc4ac01f'
    plans=plans or [
        ('2bbqxvei','S3','subdict','dictutils','subdict_default_keep_and_explicit_drop',"from boltons.dictutils import subdict\nd={'a':1,'b':2,'c':3};r=subdict(d);print('default_subset',r)\nassert r==d\nassert subdict(d,keep=['a','b'],drop=['b'])=={'a':1}"),
        ('4qcfel1l','S3','OrderedMultiDict.items','dictutils','ordered_multivalue_item_projection',"from boltons.dictutils import OrderedMultiDict\nd=OrderedMultiDict([('a',1),('b',2),('a',3)]);r=d.items(multi=True);print('multi_items',r)\nassert r==[('a',1),('b',2),('a',3)]\nassert dict(d.items())=={'a':3,'b':2}"),
        ('5vq1fa1x','S3','IndexedSet.__init__','setutils','ordered_unique_collection_accepts_general_iterable',"from boltons.setutils import IndexedSet\nr=IndexedSet(x for x in [3,1,3,2]);print('iterable_items',list(r))\nassert list(r)==[3,1,2]\nassert r.index(1)==1"),
        ('x0it4etp','S3','_ComplementSet.__and__','setutils','finite_and_complement_set_intersection',"from boltons.setutils import complement\nr=complement({1}) & {1,2};print('intersection',r)\nassert r=={2}\nr=complement({1}) & complement({2});assert 1 not in r and 2 not in r and 3 in r"),
        ('zrn1muyg','S4','pairwise_iter','iterutils','adjacent_pair_window_and_optional_terminal_fill',"from boltons.iterutils import pairwise_iter\nr=list(pairwise_iter([1,2,3]));print('pairs',r)\nassert r==[(1,2),(2,3)]\nassert list(pairwise_iter([1,2,3],end=9))==[(1,2),(2,3),(3,9)]"),
    ]
    index={row['instance_id']:row for row in map(json.loads,(ROOT/'competition/vfarl/acquisition/general/smith-task-index.jsonl').read_text().splitlines())}
    pages={}
    for page in sorted((ROOT/'competition/vfarl/acquisition/expand-objects').glob('train-page-*.json')):
        if page.name.endswith('.receipt.json'):continue
        receipt=json.loads(page.with_suffix('.receipt.json').read_text())
        if receipt['revision']!=REVISION or receipt['split']!='train' or receipt['sha256']!=sha(page):raise ValueError('Boltons page receipt mismatch')
        for entry in json.loads(page.read_text())['rows']:
            row=entry['row'];known=index[row['instance_id']]
            if known['row_idx']!=entry['row_idx'] or known['repo']!=row['repo']:raise ValueError('Boltons allowed train identity mismatch')
            pages[row['instance_id']]=(page,entry)
    ids=['mahmoud__boltons.3bfcfdd0.func_basic__'+p[0] for p in plans]
    output=subprocess.check_output(['git','ls-remote','https://github.com/swesmith/mahmoud__boltons.3bfcfdd0.git',*['refs/heads/'+iid for iid in ids]],text=True)
    refs={ref.removeprefix('refs/heads/'):commit for commit,ref in (line.split() for line in output.splitlines())}
    if set(refs)!=set(ids):raise ValueError('Boltons immutable ref resolution incomplete')
    resolution_name=review_filename.removesuffix('.json')+'-bug-refs.json'
    save(HERE/resolution_name,{'url':'https://github.com/swesmith/mahmoud__boltons.3bfcfdd0.git','output':output,'response_sha256':hashlib.sha256(output.encode()).hexdigest()})
    tasks=[]
    for suffix,sub,symbol,mod,contract,observation in plans:
        primary=(primary_by_suffix or {}).get(suffix,'S')
        iid='mahmoud__boltons.3bfcfdd0.func_basic__'+suffix;page,envelope=pages[iid];row=envelope['row']
        key='s600-a-boltons-'+symbol.replace('.','-').strip('_').replace('_','-');source=ART/(key+'-source')
        original=HERE/(iid+'.task.json');save(original,{'row':row});acquired=helper.acquire(row['repo'],refs[iid],source)
        required=row['FAIL_TO_PASS']+row['PASS_TO_PASS'];targets=sorted({x.split('::')[0] for x in required})
        assets=[];test_patch=[]
        for asset in sorted(up.rglob('*')):
            if not asset.is_file() or asset.is_symlink():continue
            rel=asset.relative_to(up)
            if not any(part in {'tests','test','testing'} for part in rel.parts) and str(rel)!='conftest.py':continue
            raw=asset.read_bytes();existing=source/rel
            if existing.exists() and existing.read_bytes()!=raw:raise ValueError('Original Boltons asset differs '+str(rel))
            assets.append({'path':str(rel),'sha256':hashlib.sha256(raw).hexdigest(),'already_present':existing.exists()})
            if not existing.exists():test_patch.extend(difflib.unified_diff([],raw.decode().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+str(rel)))
        if not all(any(a['path']==target for a in assets) for target in targets):raise ValueError('Missing original Boltons target')
        task={'instance_id':iid,'repo':row['repo'],'base_commit':refs[iid],'problem_statement':row['problem_statement'],'patch':'','test_patch':''.join(test_patch),'FAIL_TO_PASS':row['FAIL_TO_PASS'],'PASS_TO_PASS':row['PASS_TO_PASS'],'verification_target_files':targets}
        task_path=ART/(key+'-task.json');save(task_path,task)
        visible_profile=None
        if not row['problem_statement'].strip():
            prompt=(curation_prompts or {}).get(suffix)
            if not prompt or not prompt.strip():raise ValueError('Original empty description needs an explicitly reviewed behavioral prompt')
            original_adapted=task_path;task_path=ART/(key+'-task-v2.json');task={**task,'problem_statement':prompt};save(task_path,task)
            visible_profile={'kind':'curator_authored_behavioral_clarification','original_problem_statement_sha256':hashlib.sha256(row['problem_statement'].encode()).hexdigest(),'curated_problem_statement_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'reason':'publisher statement empty','reviewed_by':'Codex explicit fixed-source public API contract and mutation review2026-10-25','curated_prompt':prompt,'source_basis':['Original fixed upstream public function docstring and implementation contract','Reviewed production-function mutation in the allowed train row'],'original_empty_adapted_task_preserved':label(original_adapted),'original_empty_description_was_not_replayed':True,'actual_new_replay_under_curated_prompt':label(ART/(key+'-run-001')),'reference_patch_or_required_list_in_prompt':False}
        path='boltons/'+mod+'.py';old_source=(source/path).read_text();new_source=(up/path).read_text()
        def body(text):
            scope=ast.parse(text).body
            parts=symbol.split('.')
            if len(parts)==2:scope=next(node for node in scope if isinstance(node,ast.ClassDef) and node.name==parts[0]).body
            node=next(node for node in scope if isinstance(node,ast.FunctionDef) and node.name==parts[-1])
            return ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno]),node.lineno,node.end_lineno
        old,start,end=body(old_source);new,_,_=body(new_source)
        if old==new or old_source.count(old)!=1 or old_source.replace(old,new)!=new_source:raise ValueError('Boltons reviewed function does not explain exact source diff '+iid)
        command="python3 - <<'PYREPLAY'\nfrom pathlib import Path\nimport hashlib,importlib\nm=importlib.import_module('boltons."+mod+"')\na=Path(m.__file__).resolve();e=(Path.cwd()/"+repr(path)+").resolve()\nassert a==e,(a,e)\nprint('source_binding',a,'sha256',hashlib.sha256(a.read_bytes()).hexdigest())\n"+observation+"\nPYREPLAY"
        def action(tool,args,status='ok'):
            value={'tool':tool,'arguments':args,'expected_status':status}
            if tool=='run_command':value['reviewed_command_sha256']=hashlib.sha256(args['command'].encode()).hexdigest()
            return value
        recipe_path=REPLAY/(key+'-reviewed-recipe.json');save(recipe_path,{'evidence_kind':'human_reviewed_off_policy_demonstration','reviewed_by':'Codex explicit collection/iteration contract and source-command review2026-10-25','review_notes':contract+'; original upstream function and complete source diff reviewed; original requirements and full target files intact. Short alias preloads the real package to preserve original __file__-based source lookup, no fake module.','actions':[action('get_status',{}),action('read_file',{'filepath':path,'start_line':max(1,start-2),'end_line':end+2}),action('run_command',{'command':command},'error'),action('edit_file',{'filepath':path,'old_string':old,'new_string':new}),action('run_command',{'command':command}),action('get_status',{}),action('submit_patch',{})],'final_response':f'Restored {symbol}: {contract.replace("_"," ")}. The actual public API observation completed and the patch was submitted.'})
        tasks.append({'instance_id':iid,'key':key,'canonical_repo':'mahmoud/boltons','primary_adapter':primary,'submechanism':sub,'repair_family_id':'boltons.'+symbol+'::'+contract,'snapshot_group_id':'mahmoud/boltons::'+upstream_sha,'original_task':label(original),'original_task_sha256':sha(original),'source_dataset':'SWE-bench/SWE-smith-py','source_revision':REVISION,'physical_split':'train','row_idx':envelope['row_idx'],'source_page':label(page),'source_page_sha256':sha(page),'buggy_source':acquired,'execution_commit':refs[iid],'upstream_source':{'repo':'mahmoud/boltons','resolved_revision':upstream_sha,'path':label(up)},'workspace':label(source),'adapted_task':label(task_path),'recipe':label(recipe_path),'production_symbol':symbol,'production_path':path,'mechanism_evidence':{'trigger':observation,'base_wrong_behavior':'Reviewed source mutation violates '+contract,'causal_change':'Restore the fixed upstream production function only','discriminating_observation':'Actual source-bound observation is run both before and after the edit','classification_basis':contract,'source_bug_patch_is_solution':False},'original_FAIL_TO_PASS_count':len(row['FAIL_TO_PASS']),'original_PASS_TO_PASS_count':len(row['PASS_TO_PASS']),'verification_target_files':targets,'verification_assets':assets,'original_requirements_verbatim':True,'original_patch_semantics':'bug_introducing','visible_prompt_profile':visible_profile,'license_evidence':{'source':{'path':label(up/'LICENSE'),'sha256':sha(up/'LICENSE'),'revision':upstream_sha},'dataset':{'license':'MIT','path':'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt','sha256':sha(ROOT/'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt'),'revision':REVISION}}})
        if course_membership:tasks[-1]['course_membership']=course_membership
        print('prepared',iid,sub,symbol,flush=True)
    save(HERE/review_filename,{'source_dataset':'SWE-bench/SWE-smith-py','source_revision':REVISION,'physical_split':'train','scope':'Explicitly reviewed distinct production contracts. Candidate preparation is not training admission. Full original target files and all original required nodes retained. Empty publisher prompts require source-reviewed behavior-only clarification and actual replay with those task bytes.','tasks':tasks})


def run(start=0,limit=None,batch='funcy',attempt=1,review_override=None):
    spec=importlib.util.spec_from_file_location('canonical_replay_helpers',REPLAY/'normalize_results.py');n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
    review_path=Path(review_override) if review_override else HERE/{'funcy':'reviewed-candidates.json','boltons':'reviewed-boltons-candidates.json','textdistance':'reviewed-textdistance-candidates.json','boltons_more':'reviewed-boltons-more-candidates.json','boltons_common':'reviewed-boltons-common-candidates.json','strsimpy':'reviewed-strsimpy-candidates.json','isodate':'reviewed-isodate-candidates.json','textfsm':'reviewed-textfsm-candidates.json','qrcode':'reviewed-qrcode-candidates.json','tomli':'reviewed-tomli-candidates.json','click':'reviewed-click-candidates.json'}[batch];review=json.loads(review_path.read_text())
    tasks=review['tasks'][start:None if limit is None else start+limit]
    for t in tasks:
        key=t['key'];output=ART/(key+f'-run-{attempt:03d}')
        report_path=REPLAY/(key+('' if attempt==1 else f'-attempt{attempt:03d}')+'-replay-result.json')
        if report_path.exists():print('retained_existing_result',t['instance_id'],flush=True);continue
        profile={'funcy':'.venv-cache-task','strsimpy':'.venv-cache-task','isodate':'.venv-cache-task','textfsm':'.venv-cache-task','tomli':'.venv-cache-task','qrcode':'.venv-qrcode-task','click':'.venv-click-task','boltons':'.venv-sfr','boltons_more':'.venv-sfr','boltons_common':'.venv-sfr','textdistance':'.venv-textdistance-task'}[batch]
        version='3.11' if batch=='textdistance' else '3.12'
        args=[sys.executable,str(REPLAY/'run_official.py'),'--task',str(ROOT/t['adapted_task']),'--workspace',str(ROOT/t['workspace']),'--recipe',str(ROOT/t['recipe']),'--output',str(output),'--verify','--verify-baseline','--source-layout',t.get('source_layout','.'),'--task-site-packages',str(REPLAY/(profile+'/lib/python'+version+'/site-packages')),'--task-site-packages-mode','copy','--time-minutes','30','--verification-timeout','300']
        if batch=='textdistance':args.extend(['--task-python',str(REPLAY/'.venv-textdistance-task/bin/python')])
        if batch.startswith('boltons'):args.extend(['--source-import-alias-root','/private/tmp/bq20261025s600a'+(t.get('source_import_alias_group') or ('more' if batch=='boltons_more' else 'common' if batch=='boltons_common' else ''))+str(start+review['tasks'].index(t))+'q'+str(attempt),'--source-preload-module','boltons'])
        if (output/'run-summary.json').exists():
            # Resume export after a collector serialization failure without
            # replacing the actual run or spending another execution attempt.
            result=None
            print('record_existing_actual_run',t['instance_id'],flush=True)
        else:
            with (ART/(key+'.log')).open('w') as log:
                result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        if not (output/'run-summary.json').exists():
            print('actual_run_failed_without_summary',t['instance_id'],result.returncode,flush=True);continue
        report=json.loads((output/'run-summary.json').read_text());original=json.loads((ROOT/t['original_task']).read_text())['row'];adapted=json.loads((ROOT/t['adapted_task']).read_text())
        if report['source_task_sha256']!=sha(ROOT/t['adapted_task']) or sha(output/'recipe.json')!=sha(ROOT/t['recipe']):
            raise ValueError('Existing actual run task or recipe differs from the reviewed bytes')
        reports={}
        for phase in ('baseline','repaired'):
            files=list(output.glob(phase+'-*_junit_*.xml'))
            if len(files)==1:reports[phase]=n.junit_report(files[0],original,adapted)
        report.update({'original_task':t['original_task'],'adapted_task':t['adapted_task'],'recipe':t['recipe'],'source_review':label(review_path),'source_review_sha256':sha(review_path),'replay_output':label(output),'reports':reports,'original_verifier_requirements_relaxed':False,'required_node_identifiers_verbatim_preserved':True,'source_FAIL_TO_PASS_count':t['original_FAIL_TO_PASS_count'],'source_PASS_TO_PASS_count':t['original_PASS_TO_PASS_count'],'source_train_revision':REVISION,'bug_branch_revision':t['execution_commit'],'primary_adapter':t['primary_adapter'],'submechanism':t['submechanism'],'repair_family_id':t['repair_family_id'],'snapshot_group_id':t['snapshot_group_id'],'mechanism':t['mechanism_evidence'],'training_eligibility':{'human_off_policy_tool_demonstration':bool(report.get('resolved_by_local_official_verify_task') and reports.get('baseline',{}).get('all_original_F2P_observed_failed') and reports.get('repaired',{}).get('original_official_junit_validation',[False])[0]),'competition_Docker_equivalence':False,'GPU_training_started':False}})
        if t.get('course_membership'):report['course_membership']=t['course_membership']
        if t.get('visible_prompt_profile'):report['visible_prompt_profile']=t['visible_prompt_profile']
        save(report_path,report)
        print(json.dumps({'instance_id':t['instance_id'],'resolved':report.get('resolved_by_local_official_verify_task'),'baseline_all_F2P_failed':reports.get('baseline',{}).get('all_original_F2P_observed_failed'),'repaired_counts':{k:reports.get('repaired',{}).get(k) for k in ('tests','failures','errors','skipped')}}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','run'));parser.add_argument('--start',type=int,default=0);parser.add_argument('--limit',type=int);parser.add_argument('--batch',choices=('funcy','boltons'),default='funcy');parser.add_argument('--attempt',type=int,default=1)
    options=parser.parse_args()
    if options.stage=='prepare':prepare() if options.batch=='funcy' else prepare_boltons()
    else:run(options.start,options.limit,options.batch,options.attempt)
