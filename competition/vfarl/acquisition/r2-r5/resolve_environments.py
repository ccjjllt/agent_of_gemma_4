"""Resolve selected public task branches to immutable commits; do not execute source."""
import json,hashlib,concurrent.futures,subprocess,re
from pathlib import Path
OUT=Path(__file__).resolve().parent
SELECT=json.load(open(OUT/'selection.json'))['rows']

def resolve(r):
 iid=r['instance_id'];p=OUT/'artifacts/tasks'/f'{iid}.json';d=json.load(open(p))['rows'][0]['row'];mirror=d['repo']
 if not re.fullmatch(r'swesmith/[A-Za-z0-9_.-]+',mirror) or not re.fullmatch(r'[A-Za-z0-9_.-]+',iid):raise ValueError('unexpected public identity')
 args=['git','ls-remote','https://github.com/'+mirror+'.git','refs/heads/'+iid]
 q=subprocess.run(args,capture_output=True,text=True,timeout=75)
 lines=q.stdout.strip().splitlines();commit=None
 if q.returncode==0 and len(lines)==1:
  sha,ref=lines[0].split('\t')
  if re.fullmatch(r'[0-9a-f]{40}',sha) and ref=='refs/heads/'+iid:commit=sha
 previous=r.get('full_buggy_commit_from_cached_provenance')
 if previous and commit and previous!=commit:raise ValueError('branch changed since cached provenance '+iid)
 return {'instance_id':iid,'task_dataset_revision':'77cab9055d42ab4a5c25c89a8f937096db13558e','mirror_repo':mirror,'bug_branch':iid,'buggy_commit':commit,'cached_buggy_commit':previous,'branch_matches_cached_commit':commit==previous if previous and commit else None,'image_name':d['image_name'],'repo_snapshot_prefix_in_task_id_is_not_execution_commit':True,'git_command':args,'git_return_code':q.returncode,'git_stdout':q.stdout,'git_stderr':q.stderr,'upstream_repo':r['repo'],'dataset_license':'MIT','upstream_license_provenance_path':r.get('cached_license_provenance'),'source_task_artifact':str(p),'source_task_artifact_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
if __name__=='__main__':
 rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as p:
  for r in p.map(resolve,SELECT):rows.append(r);print(r['instance_id'],r['buggy_commit'],flush=True)
 (OUT/'environment-identities.json').write_text(json.dumps({'scope':'Public git ls-remote only; no clone, source execution, GPU or task verification. Exact execution branch commits resolved independently from shared repo snapshot IDs.','rows':rows},indent=2))
