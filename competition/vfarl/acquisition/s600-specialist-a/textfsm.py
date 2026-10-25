"""Collect explicit TextFSM representation, multivalue and common contracts.

Parser module names do not imply specialist ownership: cache/reset/constructor
and required-guard repairs below are kept in the shared course explicitly.
"""
from __future__ import annotations
import argparse,ast,difflib,hashlib,importlib.util,json,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];REPLAY=ROOT/'competition/vfarl/replay';ART=REPLAY/'artifacts/s600-specialist-a';UP=ART/'textfsm-upstream';REV='77cab9055d42ab4a5c25c89a8f937096db13558e';UPSHA='c31b600743895f018e7583f93405a3738a9f4d55';PREFIX='google__textfsm.c31b6007.'


def template(lines,text,expected):
    return "from io import StringIO\nimport textfsm\ntemplate=chr(10).join("+repr(lines)+")+chr(10)\nfsm=textfsm.TextFSM(StringIO(template))\nresult=fsm.ParseText(chr(10).join("+repr(text)+")+chr(10))\nprint('parsed_records',result)\nassert result=="+repr(expected)


PLANS=[
 ('func_basic__97x3noow','S','S4','fillup-scan','parser','TextFSMOptions.Fillup.OnAssignVar','backward_fill_stops_at_existing_column_value',template([r'Value Fillup LABEL (\w+)',r'Value Required ID (\d+)','','Start',r'  ^label ${LABEL}',r'  ^row ${ID} -> Record'],['label old','row 1','row 2','label new'],[['old','1'],['new','2']])),
 ('func_basic__yioj3ypt','S','S1','list-captures','parser','TextFSMOptions.List.OnAssignVar','named_capture_dict_or_literal_value_decoding',template([r'Value List ITEM ((?P<name>\w+):(?P<num>\d+))',r'Value Required END (\w+)','','Start',r'  ^item ${ITEM}',r'  ^end ${END} -> Record'],['item a:1','item b:2','end done'],[[[{'name':'a','num':'1'},{'name':'b','num':'2'}],'done']])),
 ('func_basic__sbxjl844','S','S3','list-order','parser','TextFSMOptions.List.OnSaveRecord','multivalue_record_preserves_capture_order',template([r'Value List ITEM (\w+)',r'Value Required END (\w+)','','Start',r'  ^item ${ITEM}',r'  ^end ${END} -> Record'],['item a','item b','end done'],[[['a','b'],'done']])),
 ('func_basic__y81pci40','S','S1','value-serialization','parser','TextFSMValue.__str__','template_value_statement_option_and_pattern_serialization',"from textfsm.parser import TextFSMValue,TextFSMOptions\nv=TextFSMValue(options_class=TextFSMOptions);v.Parse(r'Value List ITEM (\\w+)');r=str(v);print('value_statement',r)\nassert r==r'Value List ITEM (\\w+)'"),
 ('func_basic__h7plffpk','S','S1','rule-serialization','parser','TextFSMRule.__str__','template_rule_operator_separator_and_state_serialization',"from textfsm.parser import TextFSMRule\nr=str(TextFSMRule('^foo -> Next.Record End',line_num=1));print('rule_statement',r)\nassert r=='  ^foo -> Next.Record End'"),
 ('func_basic__t7w766w6','S','S3','attribute-projection','parser','TextFSM.GetValuesByAttrib','ordered_field_projection_by_declared_option',"from io import StringIO\nimport textfsm\nt=chr(10).join([r'Value Required ID (\\d+)',r'Value List ITEM (\\w+)','','Start',r'  ^row ${ID} -> Record'])+chr(10)\nf=textfsm.TextFSM(StringIO(t));r=f.GetValuesByAttrib('List');print('list_fields',r)\nassert r==['ITEM']\nassert f.GetValuesByAttrib('Required')==['ID']"),
 ('func_basic__j4r5n7be','S','S3','table-projection','texttable','TextTable.FormattedTable','requested_column_projection_in_formatted_table',"from textfsm.texttable import TextTable\nt=TextTable();t.header=['KEEP','DROP'];t.Append(['v_keep','v_drop']);r=t.FormattedTable(columns=['KEEP'],color=False);print('projected_table',r)\nassert 'KEEP' in r and 'v_keep' in r and 'DROP' not in r and 'v_drop' not in r"),
 ('func_basic__b6lzflpn','S','S4','numeric-sort','texttable','TextTable.sort','numeric_then_lexicographic_table_row_order',"from textfsm.texttable import TextTable\nt=TextTable();t.header=['n'];t.Append(['10.5']);t.Append(['2.5']);t.sort();r=[row['n'] for row in t];print('sorted_values',r)\nassert r==['2.5','10.5']"),
 ('func_basic__c99ey74q','S','S1','row-values','texttable','Row._SetValues','scalar_or_list_cell_string_representation',"from textfsm.texttable import Row\nr=Row();r.header=['a','b'];r.values=[7,[1,2]];print('cell_representations',r.values)\nassert r.values==['7',['1','2']]"),
 ('func_basic__9gqqodxk','S','S1','ansi-enclosure','terminal','EncloseAnsiText','ansi_sequence_delimiter_representation_order',"from textfsm import terminal\ns=chr(27)+'[31mred'+chr(27)+'[0m';r=terminal.EncloseAnsiText(s);print('enclosed_ansi',repr(r))\nassert r.startswith(terminal.ANSI_START+chr(27)+'[31m'+terminal.ANSI_END)"),
 ('func_basic__hurhoefv','S','S3','index-match','clitable','IndexTable.GetRowMatch','first_row_satisfying_known_attribute_regex_constraints',"from textfsm.clitable import IndexTable\nt=IndexTable(file_path='testdata/parseindex_index');r=t.GetRowMatch({'Hostname':'abc','Vendor':'VendorB'});print('matched_index_row',r)\nassert r==2\nassert t.GetRowMatch({'Hostname':'abc'})==1"),
 ('func_basic__45cxv7us','F','F3','filldown-state','parser','TextFSMOptions.Filldown.OnClearVar','record_clear_retains_last_assigned_filldown_value',template([r'Value Filldown LABEL (\w+)',r'Value Required ID (\d+)','','Start',r'  ^label ${LABEL}',r'  ^row ${ID} -> Record'],['label one','row 1','row 2'],[['one','1'],['one','2']])),
 ('func_basic__nlrioivs','F','F3','list-record-reset','parser','TextFSMOptions.List.OnClearVar','record_scoped_multivalue_cache_reset_without_filldown',template([r'Value List ITEM (\w+)',r'Value Required END (\w+)','','Start',r'  ^item ${ITEM}',r'  ^end ${END} -> Record'],['item a','end one','item b','end two'],[[['a'],'one'],[['b'],'two']])),
 ('func_basic__ajsqn8d7','G','G-default_value','list-initial-state','parser','TextFSMOptions.List.OnClearAllVar','empty_list_initial_state_contains_no_placeholder_value',template([r'Value List ITEM (\w+)',r'Value Required END (\w+)','','Start',r'  ^item ${ITEM}',r'  ^end ${END} -> Record'],['item a','end one'],[[['a'],'one']])),
 ('func_basic__mwrbdvlu','F','F1','option-owner','parser','TextFSMOptions.OptionBase.__init__','option_capture_preserves_actual_value_owner_object',template([r'Value Required ID (\d+)','','Start',r'  ^row ${ID} -> Record'],['row 7'],[['7']])),
 ('func_basic__h6yd28ss','G','G-request_precondition','required-guard','parser','TextFSMOptions.Required.OnSaveRecord','missing_required_value_prevents_record_commit',template([r'Value Required ID (\d+)','','Start',r'  ^row ${ID} -> Record'],['row 7'],[['7']])),
]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def label(path):return str(path.relative_to(ROOT))
def save(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def function_text(path,qualified):
    text=path.read_text();tree=ast.parse(text);node=tree
    for component in qualified.split('.'):
        nodes=[n for n in node.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==component]
        if len(nodes)!=1:
            # These selected Smith rows explicitly unindent their last method.
            # Read the actual top-level old function; replacement must still
            # explain the exact entire-file delta to restore class ownership.
            nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==qualified.split('.')[-1]]
            if len(nodes)!=1:raise ValueError('Actual old function ownership is ambiguous '+qualified)
            node=nodes[0];break
        node=nodes[0]
    return ''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno]),node.lineno,node.end_lineno


def prepare():
    helper=load('s600a_textfsm_data',HERE/'collect.py').load_helper();page=HERE/'artifacts/pages/textfsm-9646.json';receipt=json.loads(page.with_suffix('.receipt.json').read_text());assert receipt['revision']==receipt['returned_revision']==REV and receipt['sha256']==sha(page);rows={e['row']['instance_id']:e for e in json.loads(page.read_text())['rows']};index={r['instance_id']:r for r in map(json.loads,(ROOT/'competition/vfarl/acquisition/general/smith-task-index.jsonl').read_text().splitlines())}
    ids=[PREFIX+p[0] for p in PLANS];output=subprocess.check_output(['git','ls-remote','https://github.com/swesmith/google__textfsm.c31b6007.git',*['refs/heads/'+i for i in ids]],text=True);refs={ref.removeprefix('refs/heads/'):commit for commit,ref in (line.split() for line in output.splitlines())};assert set(refs)==set(ids);save(HERE/'textfsm-bug-ref-resolution.json',{'url':'https://github.com/swesmith/google__textfsm.c31b6007.git','output':output,'sha256':hashlib.sha256(output.encode()).hexdigest()})
    tasks=[]
    for suffix,primary,sub,key,mod,symbol,contract,observation in PLANS:
        ast.parse(observation);iid=PREFIX+suffix;entry=rows[iid];row=entry['row'];assert index[iid]['repo']==row['repo'] and index[iid]['row_idx']==entry['row_idx'];key='s600-a-textfsm-'+key;source=ART/(key+'-source');acquired=helper.acquire(row['repo'],refs[iid],source);original=HERE/(iid+'.task.json');save(original,{'row':row});targets=sorted({node.split('::')[0] for node in row['FAIL_TO_PASS']+row['PASS_TO_PASS']});assets=[];patch=[]
        for path in sorted(UP.rglob('*')):
            if not path.is_file() or path.is_symlink():continue
            rel=path.relative_to(UP)
            if not any(p in {'test','tests','testing','testdata'} for p in rel.parts):continue
            raw=path.read_bytes();actual=source/rel
            if actual.exists():assert actual.read_bytes()==raw,rel
            else:patch.extend(difflib.unified_diff([],raw.decode().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+str(rel)))
            assets.append({'path':str(rel),'sha256':hashlib.sha256(raw).hexdigest(),'already_present':actual.exists()})
        assert all(any(a['path']==target for a in assets) for target in targets)
        task={'instance_id':iid,'repo':row['repo'],'base_commit':refs[iid],'problem_statement':row['problem_statement'],'patch':'','test_patch':''.join(patch),'FAIL_TO_PASS':row['FAIL_TO_PASS'],'PASS_TO_PASS':row['PASS_TO_PASS'],'verification_target_files':targets};task_path=ART/(key+'-task.json');save(task_path,task);profile=None
        if not row['problem_statement'].strip():
            prompt='Fix textfsm.TextFSM.GetValuesByAttrib. Given an option name declared by the template parser, return the names of fields carrying that option in their original field order. Fields without the option should not be included, and an unsupported option should retain the existing error behavior.\n';old_task=task_path;task_path=ART/(key+'-task-v2.json');task={**task,'problem_statement':prompt};save(task_path,task);profile={'kind':'curator_authored_behavioral_clarification','original_problem_statement_sha256':hashlib.sha256(row['problem_statement'].encode()).hexdigest(),'curated_problem_statement_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'reason':'publisher statement empty','reviewed_by':'Codex fixed public template field option projection source review2026-10-25','curated_prompt':prompt,'source_basis':['Original GetValuesByAttrib public docstring and declared options/field ordering contract'],'original_empty_adapted_task_preserved':label(old_task),'original_empty_description_was_not_replayed':True,'actual_new_replay_under_curated_prompt':label(ART/(key+'-run-001')),'reference_patch_or_required_list_in_prompt':False}
        relative='textfsm/'+mod+'.py';old,start,end=function_text(source/relative,symbol);new,_,_=function_text(UP/relative,symbol);before=(source/relative).read_text();assert old!=new and before.count(old)==1 and before.replace(old,new)==(UP/relative).read_text(),(iid,symbol)
        binding="from pathlib import Path\nimport importlib,hashlib\nm=importlib.import_module('textfsm."+mod+"')\na=Path(m.__file__).resolve();e=(Path.cwd()/"+repr(relative)+").resolve()\nassert a==e,(a,e)\nprint('source_binding',a,'sha256',hashlib.sha256(a.read_bytes()).hexdigest())\n";code=binding+observation+'\n';ast.parse(code);command="python3 - <<'PYREPLAY'\n"+code+'PYREPLAY'
        def action(tool,args,status='ok'):
            value={'tool':tool,'arguments':args,'expected_status':status}
            if tool=='run_command':value['reviewed_command_sha256']=hashlib.sha256(args['command'].encode()).hexdigest()
            return value
        recipe_path=REPLAY/(key+'-reviewed-recipe.json');save(recipe_path,{'evidence_kind':'human_reviewed_off_policy_demonstration','reviewed_by':'Codex explicit template/multivalue/state contract and source-command review2026-10-25','review_notes':symbol+': '+contract+'. Class-scope regressions are restored together with the reviewed method content only when their replacement exactly explains the full production-file delta. All original full files and fixed testdata retained, no rewritten assertions. Shared owner/reset/guard tasks do not occupy S specialist quotas.','actions':[action('get_status',{}),action('read_file',{'filepath':relative,'start_line':max(1,start-2),'end_line':end+2}),action('run_command',{'command':command},'error'),action('edit_file',{'filepath':relative,'old_string':old,'new_string':new}),action('run_command',{'command':command}),action('get_status',{}),action('submit_patch',{})],'final_response':contract.replace('_',' ')+'. The actual source-bound public observation completed and the patch was submitted.'})
        tasks.append({'instance_id':iid,'key':key,'canonical_repo':'google/textfsm','primary_adapter':primary,'submechanism':sub,'course_membership':'S600_specialist' if primary=='S' else 'S600_common','repair_family_id':'textfsm.'+symbol+'::'+contract,'snapshot_group_id':'google/textfsm::'+UPSHA,'original_task':label(original),'original_task_sha256':sha(original),'source_dataset':'SWE-bench/SWE-smith-py','source_revision':REV,'physical_split':'train','row_idx':entry['row_idx'],'source_page':label(page),'source_page_sha256':sha(page),'buggy_source':acquired,'execution_commit':refs[iid],'upstream_source':json.loads((HERE/'textfsm-upstream-source.json').read_text()),'workspace':label(source),'adapted_task':label(task_path),'recipe':label(recipe_path),'production_symbol':symbol,'production_path':relative,'mechanism_evidence':{'trigger':observation,'base_wrong_behavior':'Reviewed production mutation violates '+contract,'causal_change':'Restore the exact fixed-source production method, including its class indentation where explicitly mutated','classification_basis':contract,'discriminating_observation':'Actual source-bound public template/table operation, full original96 requirements/four whole source files retained','source_bug_patch_is_solution':False},'original_FAIL_TO_PASS_count':len(row['FAIL_TO_PASS']),'original_PASS_TO_PASS_count':len(row['PASS_TO_PASS']),'verification_target_files':targets,'verification_assets':assets,'original_requirements_verbatim':True,'original_patch_semantics':'bug_introducing','visible_prompt_profile':profile,'license_evidence':{'source':{'path':label(UP/'LICENSE.txt'),'sha256':sha(UP/'LICENSE.txt'),'revision':UPSHA,'declared_license':'Apache-2.0'},'dataset':{'license':'MIT','path':'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt','sha256':sha(ROOT/'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt'),'revision':REV}}})
        print('prepared',iid,primary,sub,symbol,flush=True)
    save(HERE/'reviewed-textfsm-candidates.json',{'source_dataset':'SWE-bench/SWE-smith-py','source_revision':REV,'physical_split':'train','scope':'Eleven specialist representation/projection/traversal contracts and five shared state/owner/guard contracts. One empty source statement uses a disclosed curator task. Preserve all96 original required nodes and four complete original files plus upstream testdata. Prepared candidates are not admitted until actual full verification.','tasks':tasks})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','run'));parser.add_argument('--start',type=int,default=0);parser.add_argument('--limit',type=int);parser.add_argument('--attempt',type=int,default=1);a=parser.parse_args()
    if a.stage=='prepare':prepare()
    else:load('s600a_textfsm_runner',HERE/'collect.py').run(a.start,a.limit,'textfsm',a.attempt)
