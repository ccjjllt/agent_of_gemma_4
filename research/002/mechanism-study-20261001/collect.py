#!/usr/bin/env python3
"""Offline mechanism evidence. Reads exposed tasks only; never calls a model.

Gold is only used for retrospective file/anchor annotations. Generated evidence
must not be supplied to candidate agents. Original traces/CSVs remain unchanged.
"""
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TASKS = Path('/private/tmp/gemma4-research/tasks.jsonl')
RUNS = [
    ROOT/'results/003/recovered-20260930/results/003/run-003-a0-smoke',
    ROOT/'results/003/recovered-20260930/results/003/run-003-joint-explore-20260929T124817Z-0d571a9b',
    ROOT/'results/003/recovered-20260930/results/003/run-003-joint-panel-20260929T151450Z-32b18764',
    ROOT/'results/002/run-002-cap-20260930T082350Z-6dedcd65',
]


def relative(path):
    return str(path.relative_to(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clipped(value, n=500):
    return str(value)[:n] if value is not None else None


def is_prod(path, repo):
    prefix = {'fastapi/fastapi':'fastapi/', 'Textualize/rich':'rich/', 'psf/requests':'src/requests/'}[repo]
    return isinstance(path,str) and (path.startswith(prefix) or '/'+prefix in path)


def task_data():
    allowed = {r['task_id'] for file in ['research/003/analysis_003_panel.csv','research/003/analysis_003_explore.csv'] for r in csv.DictReader((ROOT/file).open())}
    split = json.loads((ROOT/'research/task_splits.json').read_text())
    heldout = {r['instance_id'] for r in split['records'] if r['split']=='heldout'}
    assert not allowed & heldout and len(allowed)==27
    tasks = {}
    with TASKS.open() as source:
        for line in source:
            match = re.search(r'"instance_id"\s*:\s*"([^"]+)"',line)
            if not match or match.group(1) not in allowed:
                continue  # No JSON parsing, inspection, or export of heldout contents.
            row = json.loads(line)
            patch = row['patch']
            files = re.findall(r'^\+\+\+ b/(.+)$',patch,re.M)
            anchors = [l[1:].strip() for l in patch.splitlines() if l.startswith('-') and not l.startswith('---') and len(l[1:].strip())>=20]
            tasks[row['instance_id']] = {'task_id':row['instance_id'],'repo':row['repo'],'base_commit':row['base_commit'],'problem_title':row['problem_statement'].splitlines()[0],'problem_statement':row['problem_statement'],'gold_files':files,'gold_anchors':anchors,'gold_hunk_headers':re.findall(r'^@@.*$',patch,re.M),'gold_patch_sha256':hashlib.sha256(patch.encode()).hexdigest()}
    assert tasks.keys()==allowed
    return tasks


def result_locations():
    results=[]
    for run in RUNS:
        for path in run.glob('*/r1/attempts/*/result.json'):
            row=json.loads(path.read_text());results.append((run,path.parent,row,path))
        if run.name=='run-003-a0-smoke':
            path=run/'a0/r1/task_results.jsonl'
            for line in path.open():results.append((run,path.parent,json.loads(line),path))
    assert len(results)==93, len(results)
    return results


def collect(tasks):
    attempts=[]
    for run,directory,result,result_path in result_locations():
        tid=result['instance_id']; task=tasks[tid]; variant=directory.relative_to(run).parts[0]
        attempt=directory.name if directory.name!='r1' else 'smoke-'+tid+'-a0'
        trace_path=directory/'traces'/('trace_'+tid+'.json')
        patch_path=directory/'patches'/(tid+'.patch')
        item={'key':run.name+'/'+attempt,'run_id':run.name,'attempt_id':attempt,'task_id':tid,'variant':variant,'resolved':result['resolved'],'exit_code':result['test_exit_code'],'error':result.get('error'),'result_path':relative(result_path),'result_sha256':sha(result_path),'trace_path':relative(trace_path),'trace_present':trace_path.exists(),'patch_path':relative(patch_path),'patch_present':patch_path.exists(),'gold_files':task['gold_files']}
        if patch_path.exists():
            patch=patch_path.read_text();item['final_changed_files']=re.findall(r'^\+\+\+ b/(.+)$',patch,re.M)
            item['final_production_files']=[p for p in item['final_changed_files'] if is_prod(p,task['repo'])]
            item['final_patch_sha256']=sha(patch_path)
        if not trace_path.exists():attempts.append(item);continue
        trace=json.loads(trace_path.read_text());item['trace_sha256']=sha(trace_path)
        events=[];signatures=[]
        for step in trace['steps']:
            calls=step.get('tool_calls',[])
            if len(calls)!=1:continue
            call=calls[0];args=call.get('arguments',{}); name=call.get('function_name')
            if not isinstance(args,dict):continue
            try:obs=json.loads(step.get('observation',{}).get('content','null'))
            except (ValueError,TypeError):obs=None
            if not isinstance(obs,dict):continue
            status=obs.get('status')
            if status is None and obs.get('error'):status='error'
            filepath=args.get('filepath');content=obs.get('content','')
            gold_file=next((p for p in task['gold_files'] if filepath==p or (isinstance(filepath,str) and filepath.endswith('/'+p))),None)
            event={'step':step['step_id'],'time_s':call.get('extra',{}).get('elapsed_s'),'observation_time_s':step.get('observation',{}).get('extra',{}).get('elapsed_s'),'tool':name,'filepath':filepath,'status':status,'error_type':obs.get('error_type') or ('InvocationErrorField' if obs.get('error') else None),'error_message':clipped(obs.get('error_message',obs.get('error')),300),'exit_code':obs.get('exit_code'),'gold_file_read_ok':bool(name=='read_file' and status=='ok' and gold_file),'gold_old_anchor_seen':bool(name=='read_file' and status=='ok' and gold_file and isinstance(content,str) and any(a in content for a in task['gold_anchors'])),'source_edit_attempt':name in ('edit_file','write_file') and is_prod(filepath,task['repo']),'source_edit_ok':name in ('edit_file','write_file') and is_prod(filepath,task['repo']) and status=='ok'}
            if name=='read_file':event.update({'start_line':args.get('start_line'),'end_line':args.get('end_line'),'returned_start':obs.get('start_line'),'returned_end':obs.get('end_line')})
            if name=='edit_file':event.update({'old_excerpt':clipped(args.get('old_string'),900),'new_excerpt':clipped(args.get('new_string'),900),'old_chars':len(args.get('old_string','')),'new_chars':len(args.get('new_string',''))})
            if name=='write_file':event['written_chars']=len(args.get('content',''))
            if name=='run_command':
                command=args.get('command','');event.update({'command':clipped(command,1400),'stdout_excerpt':clipped(obs.get('stdout'),600),'stderr_excerpt':clipped(obs.get('stderr'),300),'verification_proxy':any(x in command for x in ('pytest','repro','test_','check_')),'gold_file_command_mention':any(p in command for p in task['gold_files'])})
            if name in ('get_status','submit_patch'):event['response_excerpt']=clipped(json.dumps(obs,ensure_ascii=False),650)
            signature=json.dumps([name,args],sort_keys=True)
            signatures.append(signature);events.append(event)
        def first(predicate):
            return next(({'step':e['step'],'time_s':e['time_s'],'tool':e['tool'],'filepath':e['filepath']} for e in events if predicate(e)),None)
        streaks=[];start=0
        for end in range(1,len(signatures)+1):
            if end==len(signatures) or signatures[end]!=signatures[start]:
                if end-start>=2:streaks.append({'count':end-start,'step_start':events[start]['step'],'step_end':events[end-1]['step'],'start_s':events[start]['time_s'],'end_s':events[end-1]['time_s'],'tool':events[start]['tool'],'filepath':events[start]['filepath'],'status':events[start]['status'],'error_type':events[start]['error_type']})
                start=end
        submits=[i for i,e in enumerate(events) if e['tool']=='submit_patch']
        item.update({'events':events,'event_count':len(events),'first_gold_file_read':first(lambda e:e['gold_file_read_ok']),'first_gold_anchor_read':first(lambda e:e['gold_old_anchor_seen']),'first_source_edit_attempt':first(lambda e:e['source_edit_attempt']),'first_source_edit_ok':first(lambda e:e['source_edit_ok']),'source_edit_ok_count':sum(e['source_edit_ok'] for e in events),'source_edit_errors':sum(e['source_edit_attempt'] and e['status']=='error' for e in events),'error_types':dict(Counter(e['error_type'] for e in events if e['error_type'])),'verification_command_proxy_count':sum(e.get('verification_proxy',False) for e in events),'identical_streaks':streaks,'submit_steps':[events[i]['step'] for i in submits],'post_first_submit_events':events[submits[0]+1:] if submits else []})
        attempts.append(item)
    return attempts


def main():
    tasks=task_data();attempts=collect(tasks)
    exposed=[{k:v for k,v in row.items() if k not in ('problem_statement','gold_anchors')} for row in tasks.values()]
    by_task={t:{'attempts':len(rows),'successes':sum(r['resolved'] for r in rows),'successful_keys':[r['key'] for r in rows if r['resolved']],'failed_keys':[r['key'] for r in rows if not r['resolved']]} for t in tasks for rows in [[r for r in attempts if r['task_id']==t]]}
    eligibility=json.loads((ROOT/'research/002/verification_eligibility.json').read_text())
    later_eligible_ids={r['task_id'] for r in eligibility['tasks'] if r['eligibility']=='eligible'}
    def funnel(rows):
        have=[r for r in rows if r['trace_present']]
        hit=[r for r in have if r['first_gold_file_read']]
        edited=[r for r in hit if r['first_source_edit_ok']]
        deltas=[r['first_source_edit_ok']['time_s']-r['first_gold_file_read']['time_s'] for r in edited]
        return {'attempts':len(rows),'distinct_tasks':len({r['task_id'] for r in rows}),'trace_available':len(have),'successful_gold_file_reads':len(hit),'distinct_tasks_with_gold_file_read':len({r['task_id'] for r in hit}),'gold_anchor_seen':sum(bool(r['first_gold_anchor_read']) for r in have),'gold_file_read_then_no_successful_explicit_source_edit':sum(not r['first_source_edit_ok'] for r in hit),'successful_explicit_source_edits':sum(bool(r['first_source_edit_ok']) for r in have),'hit_to_first_successful_edit_seconds':{'n':len(deltas),'median':statistics.median(deltas) if deltas else None,'mean':statistics.mean(deltas) if deltas else None,'deltas':deltas},'first_gold_file_read_seconds':{'n':len(hit),'median':statistics.median(r['first_gold_file_read']['time_s'] for r in hit) if hit else None,'at_or_before_seconds':{str(t):sum(r['first_gold_file_read']['time_s']<=t for r in hit) for t in (30,60,120,240)}}}
    failed=[r for r in attempts if not r['resolved']]
    later=[r for r in failed if r['task_id'] in later_eligible_ids]
    panel_ids=set(json.loads((ROOT/'research/task_splits.json').read_text())['dev_panel_ids'])
    result={'date':'2026-10-01','scope':'P003 smoke3+explore18+panel48 and P002 0C24; 93 result attempts, 92 available traces','gold_boundary':'Only 27 already exposed task IDs are parsed from source; no heldout contents exported. Gold files/anchors are retrospective diagnostic annotations and MUST NOT enter candidate agent inputs. Correct-file/anchor access is a proxy, not proof of causal understanding.','metric_boundaries':'Source edit means explicit edit_file/write_file status=ok, not shell writes or semantic correctness. Verification is a command-text proxy. Streaks mean consecutive identical arguments, not necessarily lack of progress. Missing trace is not failure or zero. Cross-run any-success is a retrospective multi-draw oracle, never a tested new policy.','gold_anchor_method':'Any stripped deleted gold line of length >=20 found as a substring in successful read_file output from a gold production file. Lines may be generic; this is neither correct-function coverage nor understanding. Gold patches without qualifying removed lines have no anchors; not seeing an anchor is not failure.','task_source_sha256':sha(TASKS),'allowed_tasks':sorted(tasks),'task_annotations':exposed,'by_task':by_task,'counts':{'results':len(attempts),'traces':sum(r['trace_present'] for r in attempts),'raw_success_attempts':sum(r['resolved'] for r in attempts),'tasks_with_any_raw_success':sum(x['successes']>0 for x in by_task.values())},'funnel':{'raw_failures':funnel(failed),'raw_successes':funnel([r for r in attempts if r['resolved']]),'failures_by_run':{run.name:funnel([r for r in failed if r['run_id']==run.name]) for run in RUNS},'later_eligible_id_sensitivity':{'warning':'Only a task-ID sensitivity grouping using a LATER 0B environment. This does NOT establish verification eligibility of earlier runs and is not an adjusted historical score.','ids':sorted(later_eligible_ids),'overall':funnel(later),'by_run':{run.name:funnel([r for r in later if r['run_id']==run.name]) for run in RUNS}}},'historical_oracle':{'all_exposed_ids':{'n':len(tasks),'ever_raw_success':sum(x['successes']>0 for x in by_task.values()),'ids':sorted(t for t,x in by_task.items() if x['successes']>0)},'same_24_panel_ids_all_history':{'n':len(panel_ids),'ever_raw_success':sum(by_task[t]['successes']>0 for t in panel_ids),'ids':sorted(t for t in panel_ids if by_task[t]['successes']>0)},'warning':'Retrospective best-of-observed attempts across different strategies, runs and repeated draws; extra compute and outcome knowledge. Not a deployable selection strategy, no claimed new score or global model capability ceiling. requests_6644 is already-pass in later calibration; raw success is not established repair gain.'},'attempts':attempts}
    (OUT/'evidence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'counts':result['counts'],'by_task':{k:{x:v[x] for x in ('attempts','successes')} for k,v in by_task.items()}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
