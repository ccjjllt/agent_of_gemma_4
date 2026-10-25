"""Acquire/replay the explicitly reviewed second common-course contract batch.

No testing suite is generated. Only byte-identical original verification assets
are restored, and task success comes from the unmodified official verifier.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import argparse,difflib,hashlib,importlib.util,json,re,subprocess,sys,tempfile,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
ACQ=Path(__file__).resolve().parent
REPLAY=ROOT/'competition/vfarl/replay'
ART=REPLAY/'artifacts/s600-common-c'
REV='77cab9055d42ab4a5c25c89a8f937096db13558e'
BATCH='batch02'
SPEC=ACQ/(BATCH+'-contracts.json')
UPSTREAM_BUILD_LOCK=Lock()
REPOS={'tenacity':{
 'canonical':'jd/tenacity','mirror':'swesmith/jd__tenacity.0d40e76f',
 'upstream':REPLAY/'artifacts/sfr/upstream/jd__tenacity.0d40e76f',
 'upstream_sha':'0d40e76f7d06d631fb127e1ec58c8bd776e70d49',
 'archive':REPLAY/'artifacts/sfr/upstream/jd__tenacity.tgz',
 'layout':'.','license':'Apache-2.0','license_file':'LICENSE',
 'site':REPLAY/'.venv-sfr/lib/python3.12/site-packages'},
 'tomli':{
 'canonical':'hukkin/tomli','mirror':'swesmith/hukkin__tomli.443a0c1b',
 'upstream':REPLAY/'artifacts/sfr/upstream/hukkin__tomli.443a0c1b',
 'upstream_sha':'443a0c1bc5da39b7ed84306912ee1900e6b72e2f',
 'archive':REPLAY/'artifacts/sfr/upstream/hukkin__tomli.tgz',
 'layout':'src','license':'MIT','license_file':'LICENSE',
 'site':REPLAY/'.venv-sfr/lib/python3.12/site-packages'},
 'flake8':{
 'canonical':'PyCQA/flake8','mirror':'swesmith/PyCQA__flake8.cf1542ce',
 'upstream':REPLAY/'artifacts/recover-semantics/flake8-upstream',
 'upstream_sha':'cf1542cefa3e766670b2066dd75c4571d682a649',
 'archive':REPLAY/'artifacts/recover-semantics/flake8-upstream.tgz',
 'layout':'src','license':'MIT; retain all bundled component notices','license_file':'LICENSE',
 'site':REPLAY/'.venv-semantics-task/lib/python3.12/site-packages'},
 'gunicorn':{
 'canonical':'benoitc/gunicorn','mirror':'swesmith/benoitc__gunicorn.bacbf8aa',
 'upstream':REPLAY/'artifacts/dynamic/gunicorn-upstream-source',
 'upstream_sha':'bacbf8aa5152b94e44aa5d2a94aeaf0318a85248',
 'archive':REPLAY/'artifacts/dynamic/gunicorn-upstream.tgz',
 'layout':'.','license':'MIT','license_file':'LICENSE',
 'site':REPLAY/'.venv-dynamic-gunicorn/lib/python3.12/site-packages'},
 'tinydb':{
 'canonical':'msiemens/tinydb','mirror':'swesmith/msiemens__tinydb.10644a0e',
 'upstream':REPLAY/'artifacts/expand-cache/tinydb-invalidation-upstream',
 'upstream_sha':'10644a0e07ad180c5b756aba272ee6b0dbd12df8',
 'archive':REPLAY/'artifacts/expand-cache/tinydb-invalidation-upstream.tgz',
 'layout':'.','license':'MIT','license_file':'LICENSE',
 'site':REPLAY/'.venv-cache-task/lib/python3.12/site-packages'},
 'bottle':{
 'canonical':'bottlepy/bottle','mirror':'swesmith/bottlepy__bottle.a8dfef30',
 'upstream':ART/'bottle-upstream',
 'upstream_sha':'a8dfef301dec35f13e7578306002c40796651629',
 'archive':ART/'bottle-upstream.tgz',
 'layout':'.','license':'MIT','license_file':'LICENSE',
 'site':REPLAY/'.venv-sfr/lib/python3.12/site-packages'},
 'apispec':{
 'canonical':'marshmallow-code/apispec','mirror':'swesmith/marshmallow-code__apispec.8b421526',
 'upstream':REPLAY/'artifacts/expand-compat/marshmallow-code__apispec-upstream',
 'upstream_sha':'8b421526ea1015046de42599dd93da6a3473fe44',
 'archive':REPLAY/'artifacts/expand-compat/marshmallow-code__apispec-upstream.tgz',
 'layout':'src','license':'MIT','license_file':'LICENSE',
 'site':REPLAY/'.venv-sfr/lib/python3.12/site-packages'},
 'prettytable':{
 'canonical':'prettytable/prettytable','mirror':'swesmith/prettytable__prettytable.ca90b055',
 'upstream':ART/'prettytable-upstream',
 'upstream_sha':'ca90b055f20a6e8a06dcc46c2e3afe8ff1e8d0f1',
 'archive':ART/'prettytable-upstream.tgz',
 'layout':'src','license':'BSD-3-Clause','license_file':'LICENSE',
 'site':REPLAY/'.venv-prettytable-task/lib/python3.12/site-packages'}}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def label(p):return str(Path(p).relative_to(ROOT))
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def load_module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def helper():
 m=load_module('batch02_preparation',REPLAY/'prepare_sfr.py');m.ART=ART;return m
def index_pages():
 result={}
 receipt_file=ACQ/(BATCH+'-source-receipts.json')
 if not receipt_file.exists():receipt_file=ACQ/'batch02-source-receipts.json'
 receipts=json.loads(receipt_file.read_text())['pages']
 for receipt in receipts:
  p=ROOT/receipt['path'];assert sha(p)==receipt['sha256']
  assert receipt['revision']==REV and receipt['split']=='train'
  for raw in json.loads(p.read_text())['rows']:
   result[raw['row']['instance_id']]=(raw,receipt)
 return result
def build_prettytable_version(source,repo,commit):
 configuration=source/'pyproject.toml'
 cfg=tomllib.loads(configuration.read_text())['tool']['hatch']
 generated=source/cfg['build']['hooks']['vcs']['version-file']
 assert cfg['version']['source']=='vcs'
 assert str(generated.relative_to(source))=='src/prettytable/_version.py'
 # Preserve the real pinned commit as the build backend's VCS input. No future
 # head, release tag, pretend-version environment variable or hand-written
 # module is used. This explicit shallow checkout is not a release-version
 # equivalence claim; the build backend may infer a local development version.
 if not (source/'.git').exists():
  subprocess.run(['git','init','-q',str(source)],check=True)
  subprocess.run(['git','-C',str(source),'fetch','--no-tags','--depth=1','https://github.com/'+repo+'.git',commit],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  subprocess.run(['git','-C',str(source),'reset','--mixed','FETCH_HEAD'],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
 assert head==commit
 subprocess.run(['git','-C',str(source),'diff','--exit-code'],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 with tempfile.TemporaryDirectory(prefix='s600-prettytable-build-') as wheel_dir:
  command=[str(REPLAY/'.venv-sfr/bin/python'),'-m','hatchling','build','-t','wheel','-d',wheel_dir]
  result=subprocess.run(command,cwd=source,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  log=ART/(source.name+'-hatch-build.log');log.write_text(result.stdout)
  if result.returncode:raise RuntimeError('Original hatch-vcs build failed: '+label(log))
  wheels=list(Path(wheel_dir).glob('*.whl'));assert len(wheels)==1
  wheel_sha=sha(wheels[0]);wheel_name=wheels[0].name
 assert generated.is_file()
 return {'kind':'original_hatch_vcs_build_from_fixed_shallow_git_commit',
  'configuration':label(configuration),'configuration_sha256':sha(configuration),
  'generator':'hatchling wheel build with original hatch-vcs version source/build hook',
  'git_head':head,'git_history_scope':'one explicitly fetched pinned commit; no release tags or future branch head',
  'pretend_version_used':False,'generated_file':label(generated),'generated_file_sha256':sha(generated),
  'original_version_config':cfg['version'],'original_build_hook_config':cfg['build']['hooks']['vcs'],
  'actual_built_wheel_name':wheel_name,'actual_built_wheel_sha256':wheel_sha,
  'build_log':label(log),'build_log_sha256':sha(log),
  'boundary':'Actual local development version from shallow pinned Git metadata; not a claim of the original tagged release version.'}
def acquire_one(spec):
 cfg=REPOS[spec['repo_key']];h=helper()
 matches=[(iid,v) for iid,v in index_pages().items() if iid.endswith(spec['suffix']) and v[0]['row']['repo']==cfg['mirror']]
 assert len(matches)==1,matches
 iid,(raw,receipt)=matches[0];row=raw['row'];key='s600-c-'+spec['key']
 assert row['problem_statement'].strip() and row.get('test_patch','')==''
 original=ACQ/(iid+'.task.json')
 save(original,{'row':row,'source':{'dataset':'SWE-bench/SWE-smith-py','revision':REV,'split':'train',
  'row_idx':raw['row_idx'],'source_page':receipt['path'],'source_page_sha256':receipt['sha256'],'source_url':receipt['url']}})
 remote=subprocess.check_output(['git','ls-remote','https://github.com/'+row['repo']+'.git','refs/heads/'+iid],text=True).strip()
 commit,ref=remote.split()
 assert len(commit)==40 and ref=='refs/heads/'+iid
 source=ART/(key+'-source');acquired=h.acquire(row['repo'],commit,source)
 paths=re.findall(r'^--- a/(.+)$',row['patch'],re.M)
 assert set(paths)=={spec['path']},(iid,paths)
 old_module=load_module('batch01_edits',ACQ/'collect.py')
 edits=old_module.inverse_edits(row['patch'])
 reconstructed=(source/spec['path']).read_text()
 for old,new in edits:
  assert reconstructed.count(old)==1,(iid,old)
  reconstructed=reconstructed.replace(old,new)
 assert reconstructed==(cfg['upstream']/spec['path']).read_text(),iid
 build_profile=None
 if spec['repo_key']=='prettytable':
  with UPSTREAM_BUILD_LOCK:
   upstream_profile_path=ACQ/'prettytable-upstream-build-profile.json'
   if upstream_profile_path.exists():
    upstream_profile=json.loads(upstream_profile_path.read_text())
    for pk,hk in [('configuration','configuration_sha256'),('generated_file','generated_file_sha256'),('build_log','build_log_sha256')]:assert sha(ROOT/upstream_profile[pk])==upstream_profile[hk]
    assert upstream_profile['git_head']==cfg['upstream_sha']
   else:
    upstream_profile=build_prettytable_version(cfg['upstream'],cfg['canonical'],cfg['upstream_sha'])
    save(upstream_profile_path,upstream_profile)
  prior=REPLAY/(key+'-source-preparation.json')
  old_profile=json.loads(prior.read_text()).get('source_build_profile',{}).get('task') if prior.exists() else None
  if old_profile:
   for pk,hk in [('configuration','configuration_sha256'),('generated_file','generated_file_sha256'),('build_log','build_log_sha256')]:assert sha(ROOT/old_profile[pk])==old_profile[hk]
   assert old_profile['git_head']==commit
  build_profile={'upstream':upstream_profile,'task':old_profile or build_prettytable_version(source,row['repo'],commit)}
 targets=sorted({n.split('::',1)[0] for n in row['FAIL_TO_PASS']+row['PASS_TO_PASS']})
 patches=[];assets=[]
 for p in sorted(cfg['upstream'].rglob('*')):
  rel=p.relative_to(cfg['upstream'])
  if not p.is_file() or p.is_symlink() or '__pycache__' in rel.parts or p.suffix=='.pyc':continue
  if not any(part in {'tests','test','testing'} for part in rel.parts) and str(rel) not in {'conftest.py','test.py'}:continue
  data=p.read_bytes();exists=(source/rel).exists()
  if exists:assert (source/rel).read_bytes()==data,(iid,str(rel))
  else:
   text=data.decode()
   patches.extend(difflib.unified_diff([],text.splitlines(keepends=True),fromfile='/dev/null',tofile='b/'+str(rel)))
  assets.append({'path':str(rel),'sha256':sha(p),'source_revision':cfg['upstream_sha'],'already_present':exists,'source_path':label(p)})
 assert all((source/t).is_file() or any(a['path']==t for a in assets) for t in targets)
 task={k:row[k] for k in ['instance_id','repo','problem_statement','FAIL_TO_PASS','PASS_TO_PASS']}
 task.update(base_commit=commit,patch='',test_patch=''.join(patches),verification_target_files=targets)
 adapted=ART/(key+'-task.json');save(adapted,task)
 starts=[int(x) for x in re.findall(r'^@@ -(\d+)',row['patch'],re.M)]
 ends=[int(a)+int(b or 1) for a,b in re.findall(r'^@@ -(\d+)(?:,(\d+))?',row['patch'],re.M)]
 ends.extend(int(a)+int(b or 1) for a,b in re.findall(r'^@@ -[^\n]*?\+(\d+)(?:,(\d+))?',row['patch'],re.M))
 start=max(1,min(starts)-3);end=max(ends)+8
 review=' '.join(spec[k] for k in ['trigger','wrong','causal'])
 recipe=h.write_reviewed_recipe(key,spec['path'],spec['module'],start,min(end,start+149),edits,spec['observation'],review,
  spec['causal']+' Source-bound inline observations now pass. Patch submitted; clean full-file task verification is performed separately by the harness.')
 if end>start+149:
  recipe_data=json.loads(recipe.read_text())
  more_reads=[{'tool':'read_file','arguments':{'filepath':spec['path'],'start_line':line,'end_line':min(end,line+149)},'expected_status':'ok'} for line in range(start+150,end+1,150)]
  recipe_data['actions'][2:2]=more_reads
  save(recipe,recipe_data)
 lic=ACQ/(spec['repo_key']+'.LICENSE')
 if lic.exists():assert lic.read_bytes()==(cfg['upstream']/cfg['license_file']).read_bytes()
 else:lic.write_bytes((cfg['upstream']/cfg['license_file']).read_bytes())
 source_license={'license':cfg['license'],'path':label(lic),'sha256':sha(lic),'source_commit':cfg['upstream_sha']}
 dp=ROOT/'competition/vfarl/acquisition/recover-semantics/sources/SWE-smith-py-README.txt'
 dataset_license={'license':'MIT','path':label(dp),'sha256':sha(dp),'revision':REV}
 prep={'instance_id':iid,'key':key,'original_task':label(original),'original_task_sha256':sha(original),
  'adapted_task':label(adapted),'workspace':label(source),'source_dataset':'SWE-bench/SWE-smith-py',
  'source_train_revision':REV,'physical_split':'train','course_membership':'S600_common',
  'primary_adapter':spec['adapter'],'submechanism':spec['submechanism'],'canonical_repo':cfg['canonical'],
  'repair_family_id':cfg['canonical']+'::'+spec['symbol']+'::'+spec['family'],
  'snapshot_group_id':cfg['canonical']+'::'+cfg['upstream_sha'],'execution_commit':commit,
  'git_ls_remote':remote,'buggy_source':acquired,'upstream_source':{'repo':cfg['canonical'],'resolved_revision':cfg['upstream_sha'],
   'archive_path':label(cfg['archive']),'archive_sha256':sha(cfg['archive'])},
  'recipe':label(recipe),'recipe_sha256':sha(recipe),'F2P_count':len(row['FAIL_TO_PASS']),'P2P_count':len(row['PASS_TO_PASS']),
  'required_node_identifiers_verbatim_preserved':True,'source_requirements_relaxed':False,'original_patch_semantics':'bug_introducing',
  'verification_target_files':targets,'verification_assets':assets,'source_license':source_license,'dataset_license':dataset_license,
  'mechanism_evidence':{'trigger':spec['trigger'],'base_wrong_behavior':spec['wrong'],'production_symbols':spec['symbol'],
   'causal_change':spec['causal'],'discriminating_observation':{'kind':'explicitly_reviewed_source_bound_inline_execution','description':spec['observation']},
   'bug_patch_direction':'introduces_defect_not_a_repair_target'},'repo_key':spec['repo_key'],
  'explicit_contract_spec':label(SPEC),'explicit_contract_spec_sha256':sha(SPEC)}
 if build_profile:prep['source_build_profile']=build_profile
 save(REPLAY/(key+'-source-preparation.json'),prep)
 print(json.dumps({'prepared':key,'commit':commit,'F2P':prep['F2P_count']}),flush=True)
 return prep
def prepare_all():
 ART.mkdir(parents=True,exist_ok=True)
 specs=json.loads(SPEC.read_text())['tasks']
 with ThreadPoolExecutor(max_workers=3) as pool:prepared=list(pool.map(acquire_one,specs))
 save(ACQ/(BATCH+'-source-preparation.json'),{'explicit_contract_spec':label(SPEC),'explicit_contract_spec_sha256':sha(SPEC),'tasks':prepared})
def run_all():
 preps=json.loads((ACQ/(BATCH+'-source-preparation.json')).read_text())['tasks']
 def run_one(prep):
  cfg=REPOS[prep['repo_key']];out=ART/(prep['key']+'-run-001');log=ART/(prep['key']+'-run-001.log')
  cmd=[sys.executable,str(REPLAY/'run_official.py'),'--task',str(ROOT/prep['adapted_task']),'--workspace',str(ROOT/prep['workspace']),
   '--recipe',str(ROOT/prep['recipe']),'--output',str(out),'--verify','--verify-baseline','--source-layout',cfg['layout'],
   '--task-site-packages',str(cfg['site'])]
  if not (out/'run-summary.json').exists():
   with log.open('w') as stream:result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
   if result.returncode:
    print(json.dumps({'run':prep['key'],'returncode':result.returncode,'log':label(log)}),flush=True);return
  h=helper();target=h.record_result(prep['key'],out.name);report=json.loads(target.read_text())
  report.update(source_license=prep['source_license'],dataset_license=prep['dataset_license'],course_membership='S600_common',
   source_limitations='Publisher description retained verbatim; no native-agent or on-policy claim.',
   runtime_equivalence_limit='Official local SubprocessManager CPU only; not competition Docker/Linux equivalence.')
  save(target,report)
  print(json.dumps({'run':prep['key'],'resolved':report['resolved_by_local_official_verify_task'],
   'baseline_failures':report['reports']['baseline']['failures'],'repaired_tests':report['reports']['repaired']['tests']}),flush=True)
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run_one,preps))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['prepare','run']);parser.add_argument('--batch',default='batch02');args=parser.parse_args()
 if not re.fullmatch(r'batch\d+',args.batch):parser.error('batch must use a numeric batch identifier')
 BATCH=args.batch;SPEC=ACQ/(BATCH+'-contracts.json')
 prepare_all() if args.operation=='prepare' else run_all()
