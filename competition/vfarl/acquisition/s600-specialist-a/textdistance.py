"""Prepare explicit distance-algorithm contracts from fixed allowed train rows.

Each listed algorithm has a different production recurrence or multiset formula.
Other mutations of the same algorithm are deliberately excluded. Preparation is
not an assertion of admission; actual official-tool replay determines outcomes.
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
REPLAY=ROOT/'competition/vfarl/replay'
ART=REPLAY/'artifacts/s600-specialist-a'
UP=ART/'textdistance-upstream-fixed'
REVISION='77cab9055d42ab4a5c25c89a8f937096db13558e'
UPSTREAM_SHA='c3aca916bd756a8cb71114688b469ec90ef5b232'
PLANS=[
 ('2tydf112','S3','Jaccard.__call__','token_based','multiset_intersection_over_union','Jaccard',"('abb','abc')",'0.5','{}'),
 ('i2oqnvnu','S3','Sorensen.__call__','token_based','twice_intersection_over_total_multiplicity','Sorensen',"('abb','abc')",'2/3','{}'),
 ('14t35hv0','S3','Overlap.__call__','token_based','intersection_over_smaller_multiset','Overlap',"('abb','abc')",'2/3','{}'),
 ('f6p6gdxf','S3','Cosine.__call__','token_based','intersection_over_geometric_mean_cardinality','Cosine',"('ab','abc')",'2/(6**0.5)','{}'),
 ('yctrkamb','S3','Bag.__call__','token_based','maximum_unmatched_multiset_multiplicity','Bag',"('abb','abc')",'1','{}'),
 ('r9i419am','S3','MongeElkan._calc','token_based','per_token_best_match_then_source_normalization','MongeElkan',"('ab','ac')",'0.25',"{'algorithm': t.Identity(), 'symmetric': False}"),
 ('9exwa2qy','S4','Hamming.__call__','edit_based','position_mismatch_count_and_unequal_length_padding','Hamming',"('abc','ax')",'2','{}'),
 ('yb1ifrdn','S4','Levenshtein._cycled','edit_based','minimum_insertion_deletion_substitution_recurrence','Levenshtein',"('kitten','sitting')",'3','{}'),
 ('hfubveqw','S4','DamerauLevenshtein._pure_python_restricted','edit_based','restricted_adjacent_transposition_dynamic_program','DamerauLevenshtein',"('ab','ba')",'1',"{'restricted': True}"),
 ('tm83e4m4','S4','NeedlemanWunsch.__call__','edit_based','global_alignment_score_and_gap_boundary','NeedlemanWunsch',"('abc','axc')",'2','{}'),
 ('ce7flcwl','S4','SmithWaterman.__call__','edit_based','local_alignment_zero_floor_and_predecessor_cells','SmithWaterman',"('abc','axc')",'2','{}'),
 ('i8swb4w6','S4','Gotoh.__call__','edit_based','affine_gap_open_extension_state_recurrences','Gotoh',"('abc','axc')",'2','{}'),
 ('t6zl87l9','S4','LCSSeq._dynamic','sequence_based','longest_common_subsequence_dp_and_traceback','LCSSeq',"('abcd','ac')",repr('ac'),'{}'),
 ('k6vlmwq0','S4','LCSStr._custom','sequence_based','longest_shared_contiguous_substring_across_inputs','LCSStr',"('abcdef','zbcdefq','ybcdefx')",repr('bcdef'),'{}'),
 ('2xev6i2v','S4','JaroWinkler.__call__','edit_based','match_window_transposition_and_prefix_boost','JaroWinkler',"('MARTHA','MARHTA')",'0.9611111111111111','{}'),
 ('np8new2x','S4','Editex.__call__','phonetic','phonetic_group_edit_recurrence_with_case_expansion','Editex',"('cat','hat')",'2','{}'),
 ('2j0fpx6j','S4','RatcliffObershelp.__call__','sequence_based','recursive_shared_match_normalized_by_total_length','RatcliffObershelp',"('abc','axc')",'2/3','{}'),
]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def label(path):return str(path.relative_to(ROOT))
def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def prepare():
    base=load('s600a_data_helpers',HERE/'collect.py');helper=base.load_helper()
    pin=json.loads((HERE/'textdistance-upstream-commit.json').read_text());assert pin['sha']==UPSTREAM_SHA
    index={r['instance_id']:r for r in map(json.loads,(ROOT/'competition/vfarl/acquisition/general/smith-task-index.jsonl').read_text().splitlines())}
    rows={}
    for page in sorted((HERE/'artifacts/pages').glob('textdistance-*.json')):
        if page.name.endswith('.receipt.json'):continue
        receipt=json.loads(page.with_suffix('.receipt.json').read_text());assert receipt['revision']==receipt['returned_revision']==REVISION and receipt['split']=='train' and receipt['sha256']==sha(page)
        for e in json.loads(page.read_text())['rows']:
            t=e['row'];i=index[t['instance_id']];assert i['row_idx']==e['row_idx'] and i['repo']==t['repo'];rows[t['instance_id']]=(page,e)
    ids=['life4__textdistance.c3aca916.func_basic__'+p[0] for p in PLANS]
    output=subprocess.check_output(['git','ls-remote','https://github.com/swesmith/life4__textdistance.c3aca916.git',*['refs/heads/'+i for i in ids]],text=True)
    refs={r.removeprefix('refs/heads/'):s for s,r in (line.split() for line in output.splitlines())};assert set(refs)==set(ids)
    save(HERE/'textdistance-bug-ref-resolution.json',{'url':'https://github.com/swesmith/life4__textdistance.c3aca916.git','output':output,'response_sha256':hashlib.sha256(output.encode()).hexdigest()})
    records=[]
    for suffix,sub,symbol,mod,contract,algorithm,args,expected,kwargs in PLANS:
        iid='life4__textdistance.c3aca916.func_basic__'+suffix;page,envelope=rows[iid];row=envelope['row'];assert row['problem_statement'].strip()
        key='s600-a-textdistance-'+algorithm.lower();source=ART/(key+'-source');acquired=helper.acquire(row['repo'],refs[iid],source)
        original=HERE/(iid+'.task.json');save(original,{'row':row})
        targets=sorted({n.split('::')[0] for n in row['FAIL_TO_PASS']+row['PASS_TO_PASS']});assets=[];patch=[]
        for p in sorted(UP.rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            rel=p.relative_to(UP)
            if not any(x in {'tests','test','testing'} for x in rel.parts) and str(rel)!='conftest.py':continue
            raw=p.read_bytes();destination=source/rel
            if destination.exists():assert destination.read_bytes()==raw,rel
            else:patch.extend(difflib.unified_diff([],raw.decode().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+str(rel)))
            assets.append({'path':str(rel),'sha256':hashlib.sha256(raw).hexdigest(),'already_present':destination.exists()})
        assert all(any(a['path']==target for a in assets) for target in targets)
        task_path=ART/(key+'-task.json');save(task_path,{'instance_id':iid,'repo':row['repo'],'base_commit':refs[iid],'problem_statement':row['problem_statement'],'patch':'','test_patch':''.join(patch),'FAIL_TO_PASS':row['FAIL_TO_PASS'],'PASS_TO_PASS':row['PASS_TO_PASS'],'verification_target_files':targets})
        relative='textdistance/algorithms/'+mod+'.py';before=(source/relative).read_text();after=(UP/relative).read_text()
        def body(text):
            cls,method=symbol.split('.');node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name==cls);fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==method)
            return ''.join(text.splitlines(keepends=True)[fn.lineno-1:fn.end_lineno]),fn.lineno,fn.end_lineno
        old,start,end=body(before);new,_,_=body(after);assert before.count(old)==1 and old!=new and before.replace(old,new)==after
        observation=("from pathlib import Path\nimport importlib,hashlib,math\nm=importlib.import_module('textdistance.algorithms."+mod+"')\na=Path(m.__file__).resolve();e=(Path.cwd()/"+repr(relative)+").resolve()\nassert a==e,(a,e)\nprint('source_binding',a,'sha256',hashlib.sha256(a.read_bytes()).hexdigest())\nimport textdistance as t\nobj=t."+algorithm+"(external=False,**"+kwargs+")\nvalue=obj(*"+args+")\nprint('algorithm_result',repr(value))\nexpected="+expected+"\nassert (math.isclose(value,expected) if isinstance(expected,(int,float)) else value==expected),(value,expected)\n")
        command="python3 - <<'PYREPLAY'\n"+observation+"PYREPLAY"
        def action(tool,args,status='ok'):
            value={'tool':tool,'arguments':args,'expected_status':status}
            if tool=='run_command':value['reviewed_command_sha256']=hashlib.sha256(args['command'].encode()).hexdigest()
            return value
        recipe_path=REPLAY/(key+'-reviewed-recipe.json');save(recipe_path,{'evidence_kind':'human_reviewed_off_policy_demonstration','reviewed_by':'Codex explicit multiset/DP algorithm source and numerical contract review2026-10-25','review_notes':symbol+': '+contract+'. The algorithm is forced through its real internal implementation for diagnostic observation; full original verification still retains every external-comparator case. Exactly one reviewed function explains the full production-file diff. No constant-score shortcuts or same-algorithm mutation duplicates.','actions':[action('get_status',{}),action('read_file',{'filepath':relative,'start_line':max(1,start-2),'end_line':end+2}),action('run_command',{'command':command},'error'),action('edit_file',{'filepath':relative,'old_string':old,'new_string':new}),action('run_command',{'command':command}),action('get_status',{}),action('submit_patch',{})],'final_response':'Restored '+algorithm+' '+contract.replace('_',' ')+'. The real internal numerical observation completed and the patch was submitted.'})
        records.append({'instance_id':iid,'key':key,'canonical_repo':'life4/textdistance','primary_adapter':'S','submechanism':sub,'repair_family_id':'textdistance.'+symbol+'::'+contract,'snapshot_group_id':'life4/textdistance::'+UPSTREAM_SHA,'original_task':label(original),'original_task_sha256':sha(original),'source_dataset':'SWE-bench/SWE-smith-py','source_revision':REVISION,'physical_split':'train','row_idx':envelope['row_idx'],'source_page':label(page),'source_page_sha256':sha(page),'buggy_source':acquired,'execution_commit':refs[iid],'upstream_source':json.loads((HERE/'textdistance-upstream-source.json').read_text()),'workspace':label(source),'adapted_task':label(task_path),'recipe':label(recipe_path),'production_symbol':symbol,'production_path':relative,'mechanism_evidence':{'trigger':observation,'base_wrong_behavior':'Reviewed single-function algorithm mutation violates '+contract,'causal_change':'Restore the fixed upstream recurrence or multiset formula in the exact production function','discriminating_observation':'Source-bound actual internal algorithm result before/after; original full verifier retains external comparisons','classification_basis':contract,'source_bug_patch_is_solution':False},'original_FAIL_TO_PASS_count':len(row['FAIL_TO_PASS']),'original_PASS_TO_PASS_count':len(row['PASS_TO_PASS']),'verification_target_files':targets,'verification_assets':assets,'original_requirements_verbatim':True,'original_patch_semantics':'bug_introducing','task_runtime_profile':{'python':'3.11.15','source_ci':'Original workflow includesPython3.8–3.11; exact upstream taskfile external-test mode requires test+registered comparator dependencies','unused_optional_benchmark_py_stringmatching':'Build failed under modern isolation due missing pkg_resources; no active comparator imports it at this fixed source. All actual registered comparator dependencies installed.','parent_ADK_isolation':'Parent remains original3.12 and NumPy2.5.4/Pydantic2.14.1; task NumPy1.26.4 and Python3.11 copied into task sandbox only'},'license_evidence':{'source':{'path':label(UP/'LICENSE'),'sha256':sha(UP/'LICENSE'),'revision':UPSTREAM_SHA},'dataset':{'license':'MIT','path':'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt','sha256':sha(ROOT/'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt'),'revision':REVISION}}})
        print('prepared',iid,sub,symbol,flush=True)
    save(HERE/'reviewed-textdistance-candidates.json',{'source_dataset':'SWE-bench/SWE-smith-py','source_revision':REVISION,'physical_split':'train','scope':'Explicit different algorithm production contracts from one grouped snapshot; no duplicate mutations of the same algorithm. Preparation does not mean admission. Original publisher prompts and all427 original requirements, full29 source files and all external-comparator verification retained.','tasks':records})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','run'));parser.add_argument('--start',type=int,default=0);parser.add_argument('--limit',type=int);parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args()
    if args.stage=='prepare':prepare()
    else:
        review=HERE/'reviewed-textdistance-candidates-v2.json'
        load('s600a_run_helpers',HERE/'collect.py').run(args.start,args.limit,'textdistance',args.attempt,review_override=review if review.exists() else None)
