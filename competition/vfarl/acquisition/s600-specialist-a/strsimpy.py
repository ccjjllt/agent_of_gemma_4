"""Replay source-reviewed independent string algorithm contracts.

Several Smith rows mutate the publisher's *_test.py files. Those rows are
excluded here: only reviewed production algorithms and original tests enter.
"""
from __future__ import annotations
import argparse,ast,difflib,hashlib,importlib.util,json,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];REPLAY=ROOT/'competition/vfarl/replay';ART=REPLAY/'artifacts/s600-specialist-a';UP=ART/'strsimpy-upstream'
REVISION='77cab9055d42ab4a5c25c89a8f937096db13558e';UPSTREAM_SHA='115acaacf926b41a15664bd34e763d074682bda3';PREFIX='luozhouyang__python-string-similarity.115acaac.'
PLANS=[
 ('func_basic__0w9pdflo','S3','ShingleBased.get_profile','shingle_based','function','overlapping_shingle_multiplicity_after_space_normalization',"from strsimpy.shingle_based import ShingleBased\nr=ShingleBased(2).get_profile('ababa');print('shingle_counts',r)\nassert r=={'ab':2,'ba':2}"),
 ('func_basic__3yir7e7s','S3','OverlapCoefficient.similarity','overlap_coefficient','function','shingle_set_overlap_normalized_by_smaller_set',"from strsimpy.overlap_coefficient import OverlapCoefficient\nr=OverlapCoefficient(1).similarity('abc','ab');print('overlap',r)\nassert r==1.0"),
 ('lm_rewrite__7v4jlejg','S3','SorensenDice.similarity','sorensen_dice','function','distinct_shingle_set_dice_coefficient',"from strsimpy.sorensen_dice import SorensenDice\nr=SorensenDice(1).similarity('abb','abc');print('dice',r)\nassert r==0.8"),
 ('lm_rewrite__iphd5rii','S3','Jaccard.similarity','jaccard','function','distinct_shingle_set_intersection_over_union',"from strsimpy.jaccard import Jaccard\nimport math\nr=Jaccard(1).similarity('abb','abc');print('jaccard',r)\nassert math.isclose(r,2/3)"),
 ('func_pm_class_rm_funcs__1zezezqe','S3','QGram','qgram','class','sparse_shingle_profile_L1_distance',"from strsimpy.qgram import QGram\nr=QGram(2).distance('ababa','abaca');print('qgram_L1',r)\nassert r==4\nassert QGram.distance_profile({'a':2},{'a':1,'b':1})==2"),
 ('func_basic__1ouyfaee','S4','LongestCommonSubsequence.length','longest_common_subsequence','function','maximal_subsequence_length_dynamic_program',"from strsimpy.longest_common_subsequence import LongestCommonSubsequence\nr=LongestCommonSubsequence.length('abc','ac');print('LCS_length',r)\nassert r==2"),
 ('func_basic__az8mpkt3','S4','SIFT4.distance','sift4','function','offset_matching_and_transposition_accounting',"from strsimpy.sift4 import SIFT4\nr=SIFT4().distance('MARTHA','MARHTA');print('SIFT4_distance',r)\nassert r==1"),
 ('func_basic__mu1u39zx','S4','JaroWinkler.matches','jaro_winkler','function','bounded_match_window_flags_and_transposition_count',"from strsimpy.jaro_winkler import JaroWinkler\nr=JaroWinkler.matches('MARTHA','MARHTA');print('Jaro_matches',r)\nassert r==[6,1,3,6]"),
 ('func_pm_remove_assign__r6sanfog','S4','NGram.distance','ngram','function','normalized_ngram_edit_matrix_and_padding_state',"from strsimpy.ngram import NGram\nr=NGram(2).distance('ABCD','ABCF');print('NGram_distance',r)\nassert r==0.125"),
 ('func_pm_remove_assign__w2a8939s','S4','WeightedLevenshtein.distance','weighted_levenshtein','function','weighted_edit_cost_recurrence_and_row_boundary',"from strsimpy.weighted_levenshtein import WeightedLevenshtein\nmetric=WeightedLevenshtein(substitution_cost_fn=lambda a,b:4,insertion_cost_fn=lambda c:2,deletion_cost_fn=lambda c:3)\nr=metric.distance('ab','ac');print('weighted_distance',r)\nassert r==4"),
 ('func_pm_remove_assign__qpoeyfeh','S4','Damerau.distance','damerau','function','unrestricted_transposition_matrix_and_last_seen_state',"from strsimpy.damerau import Damerau\nr=Damerau().distance('ab','ba');print('Damerau_distance',r)\nassert r==1"),
 ('func_pm_remove_assign__nelpa62b','S4','Levenshtein.distance','levenshtein','function','rolling_edit_distance_rows_and_unit_costs',"from strsimpy.levenshtein import Levenshtein\nr=Levenshtein().distance('kitten','sitting');print('Levenshtein_distance',r)\nassert r==3"),
 ('func_pm_class_rm_funcs__96brcjir','S4','OptimalStringAlignment','optimal_string_alignment','class','restricted_edit_distance_with_adjacent_transposition',"from strsimpy.optimal_string_alignment import OptimalStringAlignment\nr=OptimalStringAlignment().distance('ab','ba');print('OSA_distance',r)\nassert r==1"),
]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def label(path):return str(path.relative_to(ROOT))
def save(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def prepare():
    helper=load('s600a_strsimpy_helpers',HERE/'collect.py').load_helper();page=HERE/'artifacts/pages/python-string-similarity-5624.json';receipt=json.loads(page.with_suffix('.receipt.json').read_text());assert receipt['revision']==receipt['returned_revision']==REVISION and receipt['sha256']==sha(page)
    rows={r['row']['instance_id']:r for r in json.loads(page.read_text())['rows']};index={r['instance_id']:r for r in map(json.loads,(ROOT/'competition/vfarl/acquisition/general/smith-task-index.jsonl').read_text().splitlines())}
    ids=[PREFIX+p[0] for p in PLANS];ref_output=subprocess.check_output(['git','ls-remote','https://github.com/swesmith/luozhouyang__python-string-similarity.115acaac.git',*['refs/heads/'+i for i in ids]],text=True);refs={ref.removeprefix('refs/heads/'):commit for commit,ref in (line.split() for line in ref_output.splitlines())};assert set(ids)==set(refs)
    save(HERE/'strsimpy-bug-ref-resolution.json',{'url':'https://github.com/swesmith/luozhouyang__python-string-similarity.115acaac.git','response':ref_output,'sha256':hashlib.sha256(ref_output.encode()).hexdigest()})
    records=[]
    for suffix,sub,symbol,mod,scope,contract,observation in PLANS:
        ast.parse(observation);iid=PREFIX+suffix;envelope=rows[iid];row=envelope['row'];known=index[iid];assert known['repo']==row['repo'] and known['row_idx']==envelope['row_idx']
        relative='strsimpy/'+mod+'.py'
        changed_paths=[line[6:] for line in row['patch'].splitlines() if line.startswith('--- a/')]
        assert changed_paths==[relative] and not relative.endswith('_test.py')
        original=HERE/(iid+'.task.json');save(original,{'row':row});key='s600-a-strsimpy-'+mod.replace('_','-');source=ART/(key+'-source');acquired=helper.acquire(row['repo'],refs[iid],source)
        targets=sorted({node.split('::')[0] for node in row['FAIL_TO_PASS']+row['PASS_TO_PASS']});assets=[];test_patch=[]
        for path in sorted(UP.rglob('*')):
            if not path.is_file() or path.is_symlink():continue
            rel=path.relative_to(UP)
            if str(rel) not in targets and not path.name.endswith('_test.py') and not any(p in {'tests','test','testing'} for p in rel.parts):continue
            raw=path.read_bytes();actual=source/rel
            if actual.exists():assert actual.read_bytes()==raw,rel
            else:test_patch.extend(difflib.unified_diff([],raw.decode().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+str(rel)))
            assets.append({'path':str(rel),'sha256':hashlib.sha256(raw).hexdigest(),'already_present':actual.exists()})
        assert all(any(a['path']==target for a in assets) for target in targets)
        task={'instance_id':iid,'repo':row['repo'],'base_commit':refs[iid],'problem_statement':row['problem_statement'],'patch':'','test_patch':''.join(test_patch),'FAIL_TO_PASS':row['FAIL_TO_PASS'],'PASS_TO_PASS':row['PASS_TO_PASS'],'verification_target_files':targets};task_path=ART/(key+'-task.json');save(task_path,task);profile=None
        if not row['problem_statement'].strip():
            prompt='Fix strsimpy.levenshtein.Levenshtein.distance. Return the minimum number of unit-cost insertions, deletions, and substitutions needed to transform one string into another. Identical strings have distance zero and an empty string has distance equal to the other string length. Preserve the existing public input and error contracts.\n';old_task=task_path;task_path=ART/(key+'-task-v2.json');task={**task,'problem_statement':prompt};save(task_path,task)
            profile={'kind':'curator_authored_behavioral_clarification','original_problem_statement_sha256':hashlib.sha256(row['problem_statement'].encode()).hexdigest(),'curated_problem_statement_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'reason':'publisher statement empty','reviewed_by':'Codex fixed public unit-edit metric and source recurrence review2026-10-25','curated_prompt':prompt,'source_basis':['Original public Levenshtein API, fixed upstream unit edit-cost recurrence, and original source input/error contracts'],'original_empty_adapted_task_preserved':label(old_task),'original_empty_description_was_not_replayed':True,'actual_new_replay_under_curated_prompt':label(ART/(key+'-run-001')),'reference_patch_or_required_list_in_prompt':False}
        before=(source/relative).read_text();after=(UP/relative).read_text()
        def body(text):
            cls=symbol.split('.')[0];node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name==cls)
            if scope=='function':node=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==symbol.split('.')[1])
            return ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno]),node.lineno,node.end_lineno
        old,start,end=body(before);new,_,_=body(after);assert old!=new and before.count(old)==1 and before.replace(old,new)==after,(iid,symbol)
        binding="from pathlib import Path\nimport importlib,hashlib\nm=importlib.import_module('strsimpy."+mod+"')\na=Path(m.__file__).resolve();e=(Path.cwd()/"+repr(relative)+").resolve()\nassert a==e,(a,e)\nprint('source_binding',a,'sha256',hashlib.sha256(a.read_bytes()).hexdigest())\n";code=binding+observation+'\n';ast.parse(code);command="python3 - <<'PYREPLAY'\n"+code+'PYREPLAY'
        def action(tool,args,status='ok'):
            value={'tool':tool,'arguments':args,'expected_status':status}
            if tool=='run_command':value['reviewed_command_sha256']=hashlib.sha256(args['command'].encode()).hexdigest()
            return value
        recipe_path=REPLAY/(key+'-reviewed-recipe.json');save(recipe_path,{'evidence_kind':'human_reviewed_off_policy_demonstration','reviewed_by':'Codex source algorithm, original public tests and explicit numerical command review2026-10-25','review_notes':symbol+': '+contract+'. Original publisher tests under strsimpy/*_test.py are restored byte-identically and remain unedited. Other source rows that mutate tests are excluded. The exact reviewed method/class replacement explains the entire production-file diff; deleted methods are restored from fixed public upstream, not invented test stubs.','actions':[action('get_status',{}),action('read_file',{'filepath':relative,'start_line':max(1,start-2),'end_line':end+2}),action('run_command',{'command':command},'error'),action('edit_file',{'filepath':relative,'old_string':old,'new_string':new}),action('run_command',{'command':command}),action('get_status',{}),action('submit_patch',{})],'final_response':'Restored '+symbol+' '+contract.replace('_',' ')+'. The real source-bound numerical observation completed and the patch was submitted.'})
        records.append({'instance_id':iid,'key':key,'canonical_repo':'luozhouyang/python-string-similarity','primary_adapter':'S','submechanism':sub,'repair_family_id':'strsimpy.'+symbol+'::'+contract,'snapshot_group_id':'luozhouyang/python-string-similarity::'+UPSTREAM_SHA,'original_task':label(original),'original_task_sha256':sha(original),'source_dataset':'SWE-bench/SWE-smith-py','source_revision':REVISION,'physical_split':'train','row_idx':envelope['row_idx'],'source_page':label(page),'source_page_sha256':sha(page),'buggy_source':acquired,'execution_commit':refs[iid],'upstream_source':json.loads((HERE/'strsimpy-upstream-source.json').read_text()),'workspace':label(source),'adapted_task':label(task_path),'recipe':label(recipe_path),'production_symbol':symbol,'production_path':relative,'mechanism_evidence':{'trigger':observation,'base_wrong_behavior':'Reviewed production algorithm mutation violates '+contract,'causal_change':'Restore the exact fixed public method, or complete algorithm class when the target method was deleted','classification_basis':contract,'discriminating_observation':'Actual source-bound numerical API observation before/after; all original18 required cases across15 complete upstream files retained','source_bug_patch_is_solution':False},'original_FAIL_TO_PASS_count':len(row['FAIL_TO_PASS']),'original_PASS_TO_PASS_count':len(row['PASS_TO_PASS']),'verification_target_files':targets,'verification_assets':assets,'original_requirements_verbatim':True,'original_patch_semantics':'bug_introducing','visible_prompt_profile':profile,'license_evidence':{'source':{'path':label(UP/'LICENSE'),'sha256':sha(UP/'LICENSE'),'revision':UPSTREAM_SHA},'dataset':{'license':'MIT','path':'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt','sha256':sha(ROOT/'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt'),'revision':REVISION}}})
        print('prepared',iid,sub,symbol,flush=True)
    save(HERE/'reviewed-strsimpy-candidates.json',{'source_dataset':'SWE-bench/SWE-smith-py','source_revision':REVISION,'physical_split':'train','scope':'Thirteen distinct production algorithm contracts, not all59 source mutations. Publisher rows mutating tests are excluded. Full15 original files/all18 original required cases retained. One empty description uses an explicit curator task version, actual replay required. Prepared candidates are not automatically training admitted.','tasks':records})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','run'));parser.add_argument('--start',type=int,default=0);parser.add_argument('--limit',type=int);parser.add_argument('--attempt',type=int,default=1);a=parser.parse_args()
    if a.stage=='prepare':prepare()
    else:load('s600a_strsimpy_run',HERE/'collect.py').run(a.start,a.limit,'strsimpy',a.attempt)
