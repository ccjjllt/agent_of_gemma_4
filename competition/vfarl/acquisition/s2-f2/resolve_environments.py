"""Resolve selected public Smith bug branches to complete immutable execution commits.

Only git ls-remote is used. Source code and task commands are not executed.
"""
import json,subprocess,re,concurrent.futures,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def resolve_group(group):
 remote,items=group
 if not re.fullmatch(r'swesmith/[A-Za-z0-9_.-]+',remote):raise ValueError('Unexpected mirror repository')
 refs=['refs/heads/'+t['instance_id'] for t in items]
 if any(not re.fullmatch(r'refs/heads/[A-Za-z0-9_.-]+',r) for r in refs):raise ValueError('Unexpected task branch')
 command=['git','ls-remote','https://github.com/'+remote+'.git']+refs
 p=subprocess.run(command,capture_output=True,text=True,timeout=70)
 found={}
 for line in p.stdout.strip().splitlines():
  parts=line.split('\t')
  if len(parts)==2 and re.fullmatch(r'[0-9a-f]{40}',parts[0]):found[parts[1]]=parts[0]
 out=[]
 for t,ref in zip(items,refs):
  commit=found.get(ref);previous=t['environment'].get('cached_buggy_commit')
  out.append({'instance_id':t['instance_id'],'source_task_revision':t['source_task']['revision'],'mirror_repo':remote,'bug_branch':t['instance_id'],'buggy_commit':commit,'cached_buggy_commit':previous,'branch_matches_cached_commit':commit==previous if commit and previous else None,'image_name':t['environment']['image_name'],'upstream_repo':t['canonical_repo'],'shared_snapshot_prefix_is_not_execution_commit':True,'command':command,'return_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
  print('resolved',t['instance_id'],commit,'cached_match',commit==previous if previous else 'no_cached_pin',flush=True)
 return out
if __name__=='__main__':
 tasks=json.load(open(ROOT/'reviewed-sf40.json'))['tasks'];groups={}
 for t in tasks:
  if t['source_task']['dataset']!='SWE-bench/SWE-smith-py':continue
  groups.setdefault(t['environment']['source_repo'],[]).append(t)
 rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
  for result in pool.map(resolve_group,groups.items()):rows.extend(result)
 report={'date':'2026-10-24','read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Fresh public git ls-remote reads of exactly selected Smith task branch refs; no clone, source execution or task verification.','requested_tasks':len(rows),'complete_buggy_commits':sum(bool(x['buggy_commit']) for x in rows),'results':rows}
 (ROOT/'environment-identities.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print('resolved_count',report['complete_buggy_commits'],'of',report['requested_tasks'],flush=True)
