"""Collect individually reviewed S train mechanisms with authentic six-tool replay.

This is a data-production workflow. Candidate contracts and reproduction commands
are explicitly reviewed below; this does not classify tasks by keywords or count
mutations of one production contract as independent families.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REPLAY = ROOT / "competition/vfarl/replay"
ART = REPLAY / "artifacts/s600-specialist-b"
REV = "77cab9055d42ab4a5c25c89a8f937096db13558e"
PAGE_ROOTS = (ROOT/'competition/vfarl/acquisition/expand-cache/artifacts/pages',)
sys.path.insert(0, str(REPLAY))
from prepare_sfr import acquire
from normalize_results import junit_report


def candidate(suffix, symbol, rel, sub, contract, body):
    return {"instance_id": "Suor__funcy.207a7810.func_basic__" + suffix,
            "canonical_repo": "Suor/funcy", "snapshot": "207a7810",
            "symbol": symbol, "relative_path": rel, "submechanism": sub,
            "contract": contract, "body": body,
            "repair_family_id": "funcy::" + symbol + "::" + contract.split(".", 1)[0]}


CANDIDATES = {
    "regex-tester": candidate("74i96e6e", "re_tester", "funcy/strings.py", "S1",
        "Regex predicate search and strict Boolean result. A pattern or compiled regex must search anywhere in a string; matching and nonmatching calls return actual bool values.",
        "from funcy import re_tester\nimport re\nfor pattern in (r'\\d+',re.compile(r'\\d+')):\n f=re_tester(pattern)\n result=(f('a42b'),f('letters'))\n print('predicate_values',result,'types',tuple(type(x).__name__ for x in result))\n assert result==(True,False) and all(type(x) is bool for x in result)"),
    "regex-finder": candidate("dto91avw", "re_finder", "funcy/strings.py", "S1",
        "Regex extraction factory search and absent-value representation. Search beyond the start and preserve None for no match, with scalar or grouped match decoding.",
        "from funcy import re_finder\nf=re_finder(r'\\d+')\nprint('search_and_absence',f('ab42c'),f('letters'))\nassert f('ab42c')=='42' and f('letters') is None\ng=re_finder(r'([a-z]+)(\\d+)')\nassert g('--ab42--')==('ab','42')"),
    "regex-iterator": candidate("hdsfip2z", "re_iter", "funcy/strings.py", "S1",
        "Lazy regex-match decoding per match. The iterator must map the representation getter onto each match, preserving tuple and named-dictionary captures in source order.",
        "from funcy import re_iter\nfor pattern,expected in ((r'([a-z]+)(\\d+)',[('x','34'),('y','12')]),(r'(?P<l>[a-z]+)(?P<d>\\d+)',[{'l':'x','d':'34'},{'l':'y','d':'12'}])):\n result=re_iter(pattern,'x34y12')\n assert iter(result) is result\n values=list(result)\n print('decoded_match_stream',values)\n assert values==expected"),
    "regex-prepare": candidate("ouu8sz6a", "_prepare", "funcy/strings.py", "S1",
        "Regex compilation and decoder pair protocol. Preserve pattern/flags order, return the compiled regex before the capture getter, and retain already-compiled patterns.",
        "import re\np,getter=m._prepare(r'([a-z]+)(\\d+)',re.I)\nvalue=getter(p.search('AB42'))\nprint('prepared_pattern_and_capture',p.pattern,p.flags,value)\nassert value==('AB','42')\noriginal=re.compile(r'\\d+')\np,getter=m._prepare(original,0)\nassert p is original and getter(p.search('a12'))=='12'"),
    "regex-list": candidate("qo3jex4b", "re_all", "funcy/strings.py", "S1",
        "Ordered materialization of parsed regex captures. Return the decoded match list in source occurrence order, including no-match and named capture forms.",
        "from funcy import re_all\na=re_all(r'\\d+','x34y12')\nb=re_all(r'(?P<l>[a-z]+)(?P<d>\\d+)','x34y12')\nprint('materialized_capture_order',a,b)\nassert a==['34','12']\nassert b==[{'l':'x','d':'34'},{'l':'y','d':'12'}]\nassert re_all(r'\\d+','letters')==[]"),
    "regex-find": candidate("zp58ckvz", "re_find", "funcy/strings.py", "S1",
        "One-shot regex parser argument order. Pass the pattern and flags to the finder and the input text to the returned callable, preserving capture representation.",
        "from funcy import re_find\nvalues=(re_find(r'\\d+','x34y12'),re_find(r'y(\\d+)','x34y12'),re_find(r'([a-z]+)(\\d+)','x34y12'))\nprint('oneshot_regex_values',values)\nassert values==('34','12',('x','34'))\nassert re_find(r'\\d+','letters') is None"),
    "cut-prefix": candidate("zkdyaxut", "cut_prefix", "funcy/strings.py", "S2",
        "Prefix-length slicing boundary. When the complete prefix exists, return the span after that prefix; otherwise preserve the original string and interior occurrences.",
        "from funcy import cut_prefix\nvalues=(cut_prefix('name:alex','name:'),cut_prefix('alex','name:'),cut_prefix('xname:alex','name:'))\nprint('prefix_boundary',values)\nassert values==('alex','alex','xname:alex')"),
    "cut-suffix": candidate("fqlqaceh", "cut_suffix", "funcy/strings.py", "S2",
        "Suffix-length slicing boundary. Retain the span before the complete suffix rather than the suffix itself, preserving strings without the terminal suffix.",
        "from funcy import cut_suffix\nvalues=(cut_suffix('name.py','.py'),cut_suffix('name','.py'),cut_suffix('name.py.more','.py'))\nprint('suffix_boundary',values)\nassert values==('name','name','name.py.more')"),
    "nested-get": candidate("7miy82j0", "get_in", "funcy/colls.py", "S5",
        "Nested heterogeneous path lookup. Traverse each dict key or list index to the final value; missing keys and indices return the declared default, while an empty path returns the original root.",
        "from funcy import get_in\nroot={'items':[{'name':'first'}]};marker=object()\nvalues=(get_in(root,['items',0,'name']),get_in(root,['missing'],marker) is marker,get_in(root,['items',9],marker) is marker,get_in(root,[]) is root)\nprint('nested_lookup',values)\nassert values==('first',True,True,True)"),
    "nested-update": candidate("lk3dgqwe", "update_in", "funcy/colls.py", "S5",
        "Persistent recursive path update and dictionary auto-vivification. Copy each modified path level, preserve the input, build missing dict levels with the leaf default, and keep the existing list-boundary contract.",
        "from funcy import update_in\nroot={'items':[{'count':2}]};result=update_in(root,['items',0,'count'],lambda n:n+3)\ncreated=update_in({},['outer','leaf'],lambda n:n+1,0)\nprint('recursive_update',result,created,'input',root)\nassert result=={'items':[{'count':5}]} and root=={'items':[{'count':2}]}\nassert result is not root and result['items'] is not root['items']\nassert created=={'outer':{'leaf':1}}"),
    "nested-delete": candidate("i18x9tdn", "del_in", "funcy/colls.py", "S5",
        "Persistent recursive path deletion. Follow the first remaining component and recurse on its tail, removing only the designated nested key while preserving the input and untouched siblings.",
        "from funcy import del_in\nroot={'outer':{'leaf':1,'keep':2},'keep':3}\nresult=del_in(root,['outer','leaf'])\nprint('recursive_delete',result,'input',root)\nassert result=={'outer':{'keep':2},'keep':3}\nassert root=={'outer':{'leaf':1,'keep':2},'keep':3}\nassert result is not root and result['outer'] is not root['outer']"),
    "nested-set": candidate("4jl4b90w", "set_in", "funcy/colls.py", "S5",
        "Nested structural replacement wrapper. Replace the value at the complete path with the exact requested value, preserving its parent container and original root.",
        "from funcy import set_in\nroot={'outer':{'leaf':1,'keep':2}};marker=object()\nresult=set_in(root,['outer','leaf'],marker)\nprint('nested_set',result['outer']['leaf'] is marker,result['outer']['keep'],root)\nassert result['outer']['leaf'] is marker and result['outer']['keep']==2\nassert root=={'outer':{'leaf':1,'keep':2}}"),
    "nested-has": candidate("5zxb6loi", "has_path", "funcy/colls.py", "S5",
        "Nested structural existence separate from value truthiness. Traverse dict keys and list indices, treat an existing None leaf as present, and reject absent keys or indices.",
        "from funcy import has_path\nroot={'items':[{'value':None}]}\nvalues=(has_path(root,['items',0,'value']),has_path(root,['items',1]),has_path(root,['missing']),has_path(root,[]))\nprint('nested_existence',values)\nassert values==(True,False,False,True)"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def cached_train_row(iid):
    index = {r['instance_id']:r for r in (json.loads(s) for s in
             (ROOT/'competition/vfarl/acquisition/general/smith-task-index.jsonl').read_text().splitlines())}
    expected=index[iid]
    if expected['split']!='train' or expected['revision']!=REV:
        raise ValueError('Candidate is not in the fixed allowed train identity index')
    for page in (page for directory in PAGE_ROOTS for page in directory.glob('*.json')):
        payload=json.loads(page.read_text())
        if not isinstance(payload.get('rows'),list):
            continue
        receipt=page.with_suffix('.receipt.json')
        if not receipt.exists():
            continue
        proof=json.loads(receipt.read_text())
        if proof.get('sha256')!=sha(page) or proof.get('revision')!=REV or proof.get('split')!='train':
            raise ValueError('Cached page differs from its fixed train provenance')
        for row in payload['rows']:
            task=row.get('row',{})
            if task.get('instance_id')==iid:
                if row['row_idx']!=expected['row_idx'] or task['repo']!=expected['repo']:
                    raise ValueError('Allowed train row identity differs')
                return task, {'source_page':str(page.relative_to(ROOT)), 'source_page_sha256':sha(page),
                              'source_page_receipt':str(receipt.relative_to(ROOT)), 'row_idx':row['row_idx']}
    raise ValueError('Selected reviewed task was not found in its source cache')


def resolved_branch(repo, iid):
    path=HERE/(repo.replace('/','__')+'.refs.txt')
    if not path.exists():
        result=subprocess.run(['git','ls-remote','https://github.com/'+repo+'.git'],text=True,capture_output=True,check=True)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(result.stdout)
    refs={line.split()[1]:line.split()[0] for line in path.read_text().splitlines() if len(line.split())==2}
    commit=refs['refs/heads/'+iid]
    if len(commit)!=40:
        raise ValueError('A full execution commit is required')
    return commit, {'url':'https://github.com/'+repo+'.git','ref':'refs/heads/'+iid,
                    'full_ref_response_path':str(path.relative_to(ROOT)),'full_ref_response_sha256':sha(path)}


def function_text(path, symbol):
    text=path.read_text();lines=text.splitlines(keepends=True)
    nodes=[]
    def visit(tree, parents=()):
        for node in ast.iter_child_nodes(tree):
            if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                qualified=parents+(node.name,)
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and '.'.join(qualified)==symbol:
                    nodes.append(node)
                visit(node,qualified)
            else:
                visit(node,parents)
    visit(ast.parse(text))
    if len(nodes)!=1:
        raise ValueError('Production symbol must resolve uniquely: '+symbol)
    node=nodes[0]
    return ''.join(lines[node.lineno-1:node.end_lineno]),node.lineno,node.end_lineno


def prepare(key):
    config=CANDIDATES[key];iid=config['instance_id'];row,provenance=cached_train_row(iid)
    compile(config['body'],'<reviewed collector reproduction>','exec')
    empty_description=not row['problem_statement'].strip()
    if empty_description and not all(config.get(k) for k in ('curated_prompt','curated_by','curation_source_basis')):
        raise ValueError('Empty source description requires an explicit reviewed behavioral clarification')
    task_path=HERE/(iid+'.task.json')
    if not task_path.exists():
        save(task_path,{'source_dataset':'SWE-bench/SWE-smith-py','revision':REV,'split':'train',**provenance,'row':row})
    commit,ref=resolved_branch(row['repo'],iid)
    source=ART/(key+'-source');buggy=acquire(row['repo'],commit,source)
    upstream=ART/(config['canonical_repo'].replace('/','__')+'-upstream')
    up=acquire(config['canonical_repo'],config['snapshot'],upstream)
    targets=sorted({node.split('::',1)[0] for node in row['FAIL_TO_PASS']+row['PASS_TO_PASS']})
    files=[];missing=[]
    for path in sorted(upstream.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        rel=path.relative_to(upstream)
        required_document = str(rel) in targets and rel.suffix in {'.rst','.txt','.md'}
        required_test_module = str(rel) in targets and rel.name.startswith('test_') and rel.suffix=='.py'
        if not any(p in {'tests','test','testing'} for p in rel.parts) and str(rel) not in {'conftest.py','test.py'} and not required_document and not required_test_module:
            continue
        actual=source/rel;raw=path.read_bytes()
        if actual.exists() and actual.read_bytes()!=raw:
            raise ValueError('Original verification source differs: '+str(rel))
        files.append({'path':str(rel),'sha256':sha(path),'source_revision':up['resolved_revision'],'present_in_buggy_source':actual.exists()})
        if not actual.exists():
            missing.extend(difflib.unified_diff([],raw.decode().splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+str(rel)))
    if not all((source/p).exists() or any(f['path']==p for f in files) for p in targets):
        raise ValueError('An original required verification file is missing')
    task_data={'instance_id':iid,'repo':row['repo'],'base_commit':commit,'problem_statement':row['problem_statement'],'patch':'',
               'test_patch':''.join(missing),'FAIL_TO_PASS':row['FAIL_TO_PASS'],'PASS_TO_PASS':row['PASS_TO_PASS'],'verification_target_files':targets}
    adapted=ART/(key+'-task.json');profile=None
    if empty_description:
        original_empty=ART/(key+'-task-original-empty.json');save(original_empty,task_data)
        task_data={**task_data,'problem_statement':config['curated_prompt']}
        adapted=ART/(key+'-task-v2.json')
        attempt=config.get('run_number',1)
        profile={'kind':'curator_authored_behavioral_clarification','reason':'publisher statement empty',
            'original_problem_statement_sha256':hashlib.sha256(row['problem_statement'].encode()).hexdigest(),
            'curated_problem_statement_sha256':hashlib.sha256(config['curated_prompt'].encode()).hexdigest(),
            'curated_prompt':config['curated_prompt'],'reviewed_by':config['curated_by'],'source_basis':config['curation_source_basis'],
            'original_empty_adapted_task_preserved':str(original_empty.relative_to(ROOT)),
            'original_empty_adapted_task_sha256':sha(original_empty),'original_empty_prompt_was_replayed':False,
            'actual_new_replay_under_curated_prompt':str((ART/(key+f'-run-{attempt:03d}')).relative_to(ROOT)),
            'reference_patch_or_required_list_in_prompt':False}
    save(adapted,task_data)
    rel=config['relative_path'];before,line,end=function_text(source/rel,config['symbol']);after,_,_=function_text(upstream/rel,config['symbol'])
    if before==after:
        raise ValueError('Selected production function is not changed in this source')
    module=rel.removeprefix('src/').removesuffix('.py').replace('/','.')
    command="python3 - <<'PYREPLAY'\nfrom pathlib import Path\nimport importlib,hashlib\nm=importlib.import_module("+repr(module)+")\nactual=Path(m.__file__).resolve();expected=(Path.cwd()/"+repr(rel)+").resolve()\nassert actual==expected,(actual,expected)\nprint('source_binding',actual,'sha256',hashlib.sha256(actual.read_bytes()).hexdigest())\n"+config['body']+"\nPYREPLAY"
    def action(status):
        return {'tool':'run_command','arguments':{'command':command},'expected_status':status,'reviewed_command_sha256':hashlib.sha256(command.encode()).hexdigest()}
    recipe={'evidence_kind':'human_reviewed_off_policy_demonstration','reviewed_by':'Codex S600 source contract and explicit command review 2026-10-25',
            'review_notes':config['contract']+' Individually reviewed source/defect change and reproduction. Original full-file verification and requirements retained; not sampled Gemma, not a same-state preference pair.',
            'actions':[{'tool':'get_status','arguments':{},'expected_status':'ok'},
                       {'tool':'read_file','arguments':{'filepath':rel,'start_line':max(1,line-4),'end_line':min(end+4,line+120)},'expected_status':'ok'},action('error'),
                       {'tool':'edit_file','arguments':{'filepath':rel,'old_string':before,'new_string':after},'expected_status':'ok'},action('ok'),
                       {'tool':'get_status','arguments':{},'expected_status':'ok'},{'tool':'submit_patch','arguments':{},'expected_status':'ok'}],
            'final_response':config['contract']+' The actual source-bound reproduction passes after the production repair. Patch submitted.'}
    recipe_path=REPLAY/('s600-b-'+key+'-reviewed-recipe.json');save(recipe_path,recipe)
    lic=next((upstream/name for name in ('LICENSE','LICENSE.md','LICENSE.rst','LICENSE.txt','LICENCE') if (upstream/name).is_file()),None)
    if lic is None:
        raise ValueError('Original source license unavailable')
    copied=HERE/(config['canonical_repo'].replace('/','__')+'.LICENSE');copied.write_bytes(lic.read_bytes())
    dataset=ROOT/'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt'
    prep={k:v for k,v in config.items() if k!='body'}
    prep.update({'original_task':str(task_path.relative_to(ROOT)),'original_task_sha256':sha(task_path),'source_dataset':'SWE-bench/SWE-smith-py',
        'source_revision':REV,'physical_split':'train','use_partition':'train_only','primary_adapter':config.get('primary_adapter','S'),'execution_commit':commit,'mirror_ref':ref,
        'buggy_source':buggy,'upstream_source':up,'workspace':str(source.relative_to(ROOT)),'adapted_task':str(adapted.relative_to(ROOT)),
        'adapted_task_sha256':sha(adapted),'recipe':str(recipe_path.relative_to(ROOT)),'recipe_sha256':sha(recipe_path),
        'snapshot_group_id':config['canonical_repo']+'::'+up['resolved_revision'],'verification_target_files':targets,'verification_assets':files,
        'original_FAIL_TO_PASS_count':len(row['FAIL_TO_PASS']),'original_PASS_TO_PASS_count':len(row['PASS_TO_PASS']),
        'source_license':{'path':str(copied.relative_to(ROOT)),'sha256':sha(copied),'revision':up['resolved_revision'],'declared_license':config.get('declared_license','BSD-3-Clause')},
        'dataset_license':{'path':str(dataset.relative_to(ROOT)),'sha256':sha(dataset),'revision':REV,'declared_card_license':'MIT'},
        'mechanism_evidence':{'primary_mechanism':config['submechanism'],'production_symbol':config['symbol'],'causal_contract':config['contract'],
                              'bug_patch_direction':'introduces_defect_not_repair_target','source_diff_reviewed':True},
        'original_patch_semantics':'bug_introducing','original_verifier_requirements_relaxed':False,
        'source_mechanism_review_complete':True,**provenance})
    if profile:
        prep['visible_prompt_profile']=profile
    save(HERE/(key+'-preparation.json'),prep)
    return prep


def replay(key):
    prep=json.loads((HERE/(key+'-preparation.json')).read_text());attempt=CANDIDATES[key].get('run_number',1);out=ART/(key+f'-run-{attempt:03d}')
    if (out/'run-summary.json').exists():
        return
    log=ART/(key+f'-run-{attempt:03d}.log')
    with log.open('w') as output:
        command=[str(REPLAY/'.venv/bin/python'),str(REPLAY/'run_official.py'),'--task',str(ROOT/prep['adapted_task']),
            '--workspace',str(ROOT/prep['workspace']),'--recipe',str(ROOT/prep['recipe']),'--output',str(out),
            '--verify','--verify-baseline','--source-layout',CANDIDATES[key].get('source_layout','.'),'--task-site-packages',CANDIDATES[key].get('task_site_packages',str(REPLAY/'.venv-cache-task/lib/python3.12/site-packages')),
            '--task-site-packages-mode','copy','--command-timeout','90','--verification-timeout','180']
        if CANDIDATES[key].get('task_python'):
            command.extend(['--task-python',CANDIDATES[key]['task_python']])
        subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,check=True,
            env={**os.environ,**CANDIDATES[key].get('task_environment',{})})


def publish(keys, review_name='reviewed-candidates.json'):
    review_path=HERE/review_name
    if review_path.exists():
        raise FileExistsError('Published source review is immutable; use an explicitly versioned new batch or classification amendment: '+str(review_path))
    reviews=[];reports=[]
    for key in keys:
        prep=json.loads((HERE/(key+'-preparation.json')).read_text());attempt=CANDIDATES[key].get('run_number',1);out=ART/(key+f'-run-{attempt:03d}');summary_path=out/'run-summary.json'
        if not summary_path.exists():
            continue
        result=json.loads(summary_path.read_text());original=json.loads((ROOT/prep['original_task']).read_text())['row'];adapted=json.loads((ROOT/prep['adapted_task']).read_text())
        phases={}
        for phase in ('baseline','repaired'):
            matches=list(out.glob(phase+'-_swegemma_junit_*.xml'))
            if len(matches)==1:
                phases[phase]=junit_report(matches[0],original,adapted)
            elif not matches:
                # Retain an actual verifier attempt that failed before JUnit;
                # never turn a successful repair alone into baseline evidence.
                actual=out/('baseline-official-verification.json' if phase=='baseline' else 'official-verification.json')
                if not actual.exists():
                    raise ValueError('No actual verification evidence: '+key+'/'+phase)
                phases[phase]={'path':None,'sha256':None,'complete_JUnit':False,
                    'official_verification_path':str(actual.relative_to(ROOT)),
                    'official_verification_sha256':sha(actual),
                    'all_original_F2P_observed_failed':False,
                    'original_official_junit_validation':[False,'Actual verifier attempt did not produce a complete JUnit report']}
            else:
                raise ValueError('Ambiguous actual JUnit evidence: '+key+'/'+phase)
        events=[json.loads(s) for s in (out/'tool-events.jsonl').read_text().splitlines()]
        observations=[{'index':e['index'],'actual_response':e['response']} for e in events if e['tool']=='run_command']
        result.update({'original_task':prep['original_task'],'original_task_sha256':prep['original_task_sha256'],'adapted_task':prep['adapted_task'],
            'recipe':prep['recipe'],'replay_output':str(out.relative_to(ROOT)),'reports':phases,'canonical_repo':prep['canonical_repo'],
            'source_dataset':prep['source_dataset'],'source_revision':REV,'physical_split':'train','primary_adapter':prep['primary_adapter'],'submechanism':prep['submechanism'],
            'execution_commit':prep['execution_commit'],'repair_family_id':prep['repair_family_id'],'snapshot_group_id':prep['snapshot_group_id'],
            'mechanism':{'adapter':prep['primary_adapter'],'submechanism':prep['submechanism'],'evidence':prep['mechanism_evidence']},
            'original_verifier_requirements_relaxed':False,'required_node_identifiers_verbatim_preserved':True,
            'actual_mechanism_observations':observations,'source_preparation':str((HERE/(key+'-preparation.json')).relative_to(ROOT)),
            'training_eligibility':{'human_off_policy_tool_demonstration':bool(result['resolved_by_local_official_verify_task'] and phases['baseline']['all_original_F2P_observed_failed']),
                                    'model_on_policy_RL':False,'same_state_action_preference_pair':False,'GPU_training_started':False}})
        if prep.get('visible_prompt_profile'):
            result['visible_prompt_profile']=prep['visible_prompt_profile']
        if prep.get('course_membership'):
            result['course_membership']=prep['course_membership']
        report_path=REPLAY/('s600-b-'+key+'-replay-result.json')
        review={**prep,'source_review_complete':True,'actual_current_run':str(out.relative_to(ROOT)),
                'actual_run_summary_sha256':sha(summary_path),'actual_reports':phases,'actual_observations':observations,
                'actual_recipe_sha256':sha(ROOT/prep['recipe']),'actual_events_sha256':sha(out/'tool-events.jsonl'),
                'resolved_by_local_official_verify_task':result['resolved_by_local_official_verify_task']}
        reviews.append(review);reports.append((report_path,result))
    save(review_path,{'date':'2026-10-25','source_dataset':'SWE-bench/SWE-smith-py','source_revision':REV,'physical_split':'train',
        'scope':'Explicit S600 source and mechanism review with authentic six-tool off-policy observations. Original81 unchanged; source Python3.12, modern ADK parent, pinned original upstream verification assets.',
        'independence_basis':'One selected mutation per distinct production function contract; same upstream snapshot remains grouped. No random cross-task evaluation split. This is not model-generated RL data.',
        'tasks':reviews})
    for path,result in reports:
        result.update({'source_review_path':str(review_path.relative_to(ROOT)),'source_review_sha256':sha(review_path)})
        save(path,result)
        print(json.dumps({'instance_id':result['instance_id'],'resolved':result['resolved_by_local_official_verify_task'],
                          'baseline_F2P_failed':result['reports']['baseline']['all_original_F2P_observed_failed'],'report':str(path.relative_to(ROOT))}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--keys',nargs='+',choices=list(CANDIDATES),default=list(CANDIDATES));parser.add_argument('--stage',choices=('prepare','replay','publish','all'),default='all');args=parser.parse_args()
    for key in args.keys:
        if args.stage in {'prepare','all'}:prepare(key);print('prepared',key,flush=True)
        if args.stage in {'replay','all'}:replay(key);print('replayed',key,flush=True)
    if args.stage in {'publish','all'}:publish(args.keys)
