"""Same-server paired dev weight experiment; no training or historical control."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
import zipfile


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def stop(process):
    if process.poll() is None:
        try:os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=5)
        except (ProcessLookupError,subprocess.TimeoutExpired):
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait(timeout=10)


def stage_and_prove(agents, discover, extensions):
    pin=SETTINGS['adapter']
    matches=[p for p in Path('/kaggle/input').rglob('training_manifest.json') if digest(p)==pin['manifest_sha256']]
    if len(matches)!=1:raise RuntimeError('Real RFT training manifest missing/ambiguous')
    training=json.loads(matches[0].read_text())
    if (training['signature']!=pin['signature'] or training['status']!='training_completed'
            or training['adapter']['files']!=pin['files']):
        raise RuntimeError('Mounted RFT manifest differs from frozen real result')
    source=matches[0].parent/'adapter'
    # Large weights stay outside results/zip, but discovery sees real copied files.
    destination=agents/'adapter'/'adapters'/pin['name']
    destination.mkdir(parents=True,exist_ok=False)
    for name,expected in pin['files'].items():
        path=source/name
        if path.is_symlink() or path.stat().st_size!=expected['bytes'] or digest(path)!=expected['sha256']:
            raise RuntimeError('Final RFT file identity mismatch: '+name)
        shutil.copyfile(path,destination/name)
        if digest(destination/name)!=expected['sha256']:raise RuntimeError('RFT copy changed')
    adapters=discover(str(agents/'adapter'),adapter_extensions=extensions)
    if set(adapters.adapters)!={pin['name']} or adapters.adapters[pin['name']].rank!=8:
        raise RuntimeError('Expected exactly one real rank8 RFT adapter')
    if discover(str(agents/'base'),adapter_extensions=extensions).adapters:
        raise RuntimeError('Base Agent must select no adapter')
    import yaml
    class Loader(yaml.SafeLoader):pass
    Loader.add_constructor('!include',lambda loader,node:loader.construct_scalar(node))
    a=yaml.load((agents/'base/agent.yaml').read_text(),Loader=Loader)
    b=yaml.load((agents/'adapter/agent.yaml').read_text(),Loader=Loader)
    if a.get('adapter') or b.pop('adapter',None)!=pin['name'] or a!=b or a['model']!=SETTINGS['model_alias']:
        raise RuntimeError('Agent arms differ beyond the official adapter routing field')
    for relative in BUNDLE['base']:
        if relative!='agent.yaml' and (agents/'base'/relative).read_bytes()!=(agents/'adapter'/relative).read_bytes():
            raise RuntimeError('Agent arms have different prompts/tools/sampling configuration')
    if 'fast_base' in BUNDLE:
        if discover(str(agents/'fast_base'),adapter_extensions=extensions).adapters:
            raise RuntimeError('Stage C must not select an adapter')
        for relative in BUNDLE['base']:
            if relative!='configs/sampling.yaml' and (agents/'base'/relative).read_bytes()!=(agents/'fast_base'/relative).read_bytes():
                raise RuntimeError('Stage C changed more than sampling thinking budget')
        ordinary=yaml.safe_load((agents/'base/configs/sampling.yaml').read_text())
        fast=yaml.safe_load((agents/'fast_base/configs/sampling.yaml').read_text())
        if ordinary['thinking_config']['thinking_budget']!=4096 or fast['thinking_config']['thinking_budget']!=1024:
            raise RuntimeError('Stage C must change only4096 to1024')
        fast['thinking_config']['thinking_budget']=4096
        if fast!=ordinary:raise RuntimeError('Stage C sampling differs beyond thinking budget')
    return adapters,{'source_manifest':str(matches[0]),'source_adapter':str(source),
        'actual_served_path':str(destination),'files':pin['files'],
        'official_route_field':'adapter','expected_adapter_alias':pin['name']}


def route_evidence(attempt,expected,expected_budget):
    result={'expected_response_model':expected,'expected_thinking_budget':expected_budget,'verified':False}
    try:
        records=[json.loads(line) for line in (attempt/'raw-wire.jsonl').read_text().splitlines()]
        requests=[r for r in records if r['kind']=='wire_request']
        responses=[r for r in records if r['kind']=='wire_response']
        requested=sorted({r['request']['model'] for r in requests})
        served=sorted({r['response']['model'] for r in responses})
        budgets=[(r['request'].get('extra_body') or {}).get('thinking_token_budget') for r in requests]
        result.update(request_count=len(requests),response_count=len(responses),
                      actual_requested_models=requested,actual_response_models=served,
                      actual_thinking_budgets=sorted(set(budgets),key=str),
                      thinking_budget_verified=bool(budgets and all(value==expected_budget for value in budgets)))
        result['verified']=bool(requests and responses and requested==['openai/'+expected] and served==[expected]
                                and result['thinking_budget_verified'])
    except Exception:result['error']=traceback.format_exc()
    return result


def paired_summary(rows,tasks):
    lookup={(r['task_id'],r['variant']):r for r in rows}
    pairs=[]
    for task in tasks:
        iid=task['instance_id'];a=lookup.get((iid,'base'));b=lookup.get((iid,'adapter'))
        def valid(row):
            return bool(row and row.get('worker_exit_code')==0 and row.get('judge_valid') is True
                        and type(row.get('resolved')) is bool and row.get('route_evidence',{}).get('verified'))
        if not valid(a) or not valid(b):category='incomplete_or_invalid'
        elif b['resolved'] and not a['resolved']:category='win'
        elif a['resolved'] and not b['resolved']:category='regression'
        else:category='unchanged_success' if a['resolved'] else 'unchanged_failure'
        pairs.append({'task_id':iid,'category':category,
            'base_resolved':a.get('resolved') if a else None,'adapter_resolved':b.get('resolved') if b else None,
            'base_route_verified':bool(a and a.get('route_evidence',{}).get('verified')),
            'adapter_route_verified':bool(b and b.get('route_evidence',{}).get('verified'))})
    return {'split':'dev','independent_issues':len(tasks),'planned_attempts':2*len(tasks),
        'returned_attempts':len(rows),'paired_outcomes':pairs,
        'paired_counts':{label:sum(p['category']==label for p in pairs) for label in
            ('win','regression','unchanged_success','unchanged_failure','incomplete_or_invalid')},
        'by_arm':{arm:{'attempts':sum(r['variant']==arm for r in rows),
            'resolved':sum(r['variant']==arm and r.get('resolved') is True and r.get('worker_exit_code')==0
                and r.get('judge_valid') is True and r.get('route_evidence',{}).get('verified') for r in rows),
            'raw_reported_resolved':sum(r['variant']==arm and r.get('resolved') is True for r in rows),
            'route_verified_attempts':sum(r['variant']==arm and r.get('route_evidence',{}).get('verified') for r in rows)}
            for arm in ('base','adapter')},
        'historical5of12_used_as_control':False,'capacity_adapter_mounted':False}


def main():
    started=time.monotonic();deadline=started+SETTINGS['wall_seconds']
    out=Path('/kaggle/working/results/015')/('weight-ab-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    agents=Path('/tmp')/('p015-weight-ab-agents-'+out.name)
    for arm,files in BUNDLE.items():
        for relative,text in files.items():save(agents/arm/relative,text)
    panel=Path(MATERIAL_ROOT);document=json.loads((panel/'agent-tasks.json').read_text())
    tasks=document if isinstance(document,list) else document['tasks']
    tasks={t.get('instance_id',t.get('iid')):t for t in tasks}
    selected=[tasks[iid] for iid in SETTINGS['task_ids']]
    selected=[{**t,'instance_id':t.get('instance_id',t.get('iid')),
               'problem_statement':t.get('problem_statement',t.get('issue'))} for t in selected]
    criteria=json.loads((panel/'host-criteria.json').read_text())
    criteria=criteria if isinstance(criteria,list) else criteria['tasks']
    criteria={item.get('instance_id',item.get('iid')):item for item in criteria}
    if len(selected)!=12 or any(criteria[t['instance_id']]['split']!='dev' for t in selected):
        raise RuntimeError('The weight A/B runs only twelve frozen dev issues')
    import torch
    from adk_submission import VllmConfig,VllmServer,discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    hardware=[{'name':torch.cuda.get_device_name(i),'bytes':torch.cuda.get_device_properties(i).total_memory}
              for i in range(torch.cuda.device_count())]
    if len(hardware)!=4 or any('L4' not in x['name'] for x in hardware):raise RuntimeError('Requires four L4s')
    models=[p.parent for p in Path('/kaggle/input').rglob('tokenizer.json') if SETTINGS['model_path_pattern'] in str(p)]
    if len(models)!=1:raise RuntimeError('Official31B CTv2 mount missing/ambiguous')
    model=models[0]
    adapters,stage=stage_and_prove(agents,discover_adapters,ALLOWED_ADAPTER_EXTENSIONS)
    config=VllmConfig(model=str(model),port=8000,host='127.0.0.1',max_model_len=32768,
        dtype='bfloat16',gpu_memory_utilization=0.90,tensor_parallel_size=4,
        tool_call_parser='gemma4',reasoning_parser='gemma4',enable_auto_tool_choice=True,
        enable_lora=True,max_lora_rank=8,max_loras=1,startup_timeout=1200,
        extra_args=['--max-num-seqs','4'])
    server=VllmServer(config,adapter_manifest=adapters)
    command=server.build_cmd()
    for flag,value in (('--max-loras','1'),('--max-lora-rank','8'),('--tensor-parallel-size','4'),('--max-num-seqs','4')):
        if command[command.index(flag)+1]!=value:raise RuntimeError('Actual server command differs: '+flag)
    if '--enable-lora' not in command:raise RuntimeError('Actual command did not enable LoRA')
    manifest={'settings':SETTINGS,'adapter_stage':stage,'server_command':command,'hardware':hardware,
        'serving_conditions':{'enable_lora':True,'mounted_adapters':[SETTINGS['adapter']['name']],
            'actual_base_model_path':str(model),'actual_adapter_path':stage['actual_served_path'],
            'same_server_for_both_arms':True,'capacity_diagnostic_mounted':False},
        'versions':{name:importlib.metadata.version(name) for name in ('swegemma','adk-submission','vllm','torch')},
        'agent_files':{arm:{rel:hashlib.sha256(text.encode()).hexdigest() for rel,text in files.items()} for arm,files in BUNDLE.items()}}
    save(out/'experiment-manifest.json',manifest)
    worker=out/'attempt_worker.py';save(worker,WORKER_SOURCE)
    schedule=[]
    for slot,task in enumerate(selected):
        for arm in (('base','adapter') if slot%2==0 else ('adapter','base')):
            ordinal=len(schedule)+1;iid=task['instance_id']
            entry={'ordinal':ordinal,'task_slot':slot+1,'task_id':iid,'variant':arm,'split':'dev','stage':'AB',
                   'attempt_id':f'{ordinal:02d}-{iid}-{arm}'}
            entry['results_dir']=str(out/'attempts'/entry['attempt_id']);schedule.append((entry,task))
    c_schedule=[]
    if SETTINGS.get('fast_thinking_budget') is not None:
        for slot,task in enumerate(selected):
            ordinal=len(schedule)+len(c_schedule)+1;iid=task['instance_id']
            entry={'ordinal':ordinal,'task_slot':slot+1,'task_id':iid,'variant':'fast_base',
                'split':'dev','stage':'C','attempt_id':f'{ordinal:02d}-{iid}-fast_base'}
            entry['results_dir']=str(out/'attempts'/entry['attempt_id']);c_schedule.append((entry,task))
    save(out/'schedule.json',[entry for entry,task in schedule+c_schedule])
    c_state={'requested':bool(c_schedule),'status':'not_requested' if not c_schedule else 'awaiting_AB_barrier',
        'scope':'mechanism exploration only; not a paired weight-causal comparison or a historical5/12 gain',
        'thinking_budget':SETTINGS.get('fast_thinking_budget'),'planned_attempts':len(c_schedule)}
    active=[];rows=[]
    def collect(record,forced=None):
        entry,process,log,begin=record;log.close();attempt=Path(entry['results_dir'])
        result=json.loads((attempt/'result.json').read_text()) if (attempt/'result.json').exists() else {'resolved':None,'error':'worker output missing'}
        expected=SETTINGS['adapter']['name'] if entry['variant']=='adapter' else str(model)
        budget=1024 if entry['variant']=='fast_base' else 4096
        routing=route_evidence(attempt,expected,budget);save(attempt/'route-evidence.json',routing)
        row={**entry,**result,'worker_exit_code':process.returncode,'outer_seconds':time.monotonic()-begin,'route_evidence':routing}
        if (attempt/'context_metrics.json').is_file():
            row['context_metrics']=json.loads((attempt/'context_metrics.json').read_text())
        if forced:row['error']=forced
        rows.append(row);save(out/'analysis-rows.json',rows)
        print('RESULT '+json.dumps(row,ensure_ascii=False),flush=True)
    def run_stage(plan):
        # Called serially for whole stages, never backfilled across the barrier.
        if active:raise RuntimeError('Stage boundary reached with active workers')
        queue=iter(plan);exhausted=False
        while active or not exhausted:
            while not exhausted and len(active)<4 and time.monotonic()<deadline-150:
                try:entry,task=next(queue)
                except StopIteration:exhausted=True;break
                attempt=Path(entry['results_dir']);attempt.mkdir(parents=True)
                specification={'entry':entry,'agent_dir':str(agents/entry['variant']),'task':task,
                    'snapshot':str(panel/'snapshots'/(task['instance_id']+'.tar.gz')),
                    'base_url':str(server.base_url),'served_model':str(model),'model_alias':SETTINGS['model_alias'],
                    'budgets':SETTINGS['budgets'],'judge':criteria[task['instance_id']],
                    'panel_root':str(panel),'total_tasks':len(schedule)+len(c_schedule)}
                save(attempt/'spec.json',specification);log=(attempt/'worker.log').open('w')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
                process=subprocess.Popen([sys.executable,'-u',str(worker),str(attempt/'spec.json')],env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active.append((entry,process,log,time.monotonic()))
            if time.monotonic()>=deadline-150:exhausted=True
            for record in active[:]:
                entry,process,log,begin=record;forced=None
                if process.poll() is None and (time.monotonic()-begin>SETTINGS['attempt_seconds'] or time.monotonic()>=deadline-60):
                    forced='experiment_wall_timeout';stop(process)
                if process.poll() is not None:collect(record,forced);active.remove(record)
            if active:time.sleep(1)
    try:
        server.start();manifest['startup_seconds']=time.monotonic()-started;save(out/'experiment-manifest.json',manifest)
        print('WEIGHT_AB_SERVER '+json.dumps(manifest['serving_conditions']),flush=True)
        run_stage(schedule)
        ab_finished=sum(row.get('stage')=='AB' for row in rows)
        manifest['AB_stage_barrier']={'finished_attempts':ab_finished,'required_attempts':24,
            'active_workers':len(active),'wall_seconds':time.monotonic()-started}
        save(out/'experiment-manifest.json',manifest)
        if c_schedule:
            remaining=deadline-time.monotonic()
            # Admission estimate includes three four-worker waves at the full
            # Agent budget plus host-judge timeout and cleanup reserve. The
            # unchanged hard experiment deadline still bounds actual work.
            judge_seconds=max(float(criteria[t['instance_id']].get('timeout_seconds',120)) for t in selected)
            minimum_remaining=math.ceil(len(c_schedule)/4)*(60*float(SETTINGS['budgets']['max_time_minutes'])+judge_seconds)+150
            c_state.update(remaining_seconds_at_barrier=remaining,
                           admission_estimate_seconds=minimum_remaining)
            if ab_finished!=24 or active:
                c_state.update(status='not_executed',reason='All24 A/B attempts did not finish inside the existing wall budget')
            elif remaining<minimum_remaining:
                c_state.update(status='not_executed',reason='Insufficient remaining existing wall budget for Stage C; no added session or retry')
            else:
                c_state.update(status='running',started_wall_seconds=time.monotonic()-started)
                print('STAGE_C '+json.dumps(c_state),flush=True)
                run_stage(c_schedule)
                c_state['status']='finished' if sum(r.get('stage')=='C' for r in rows)==12 else 'partial'
    except BaseException:save(out/'failure.txt',traceback.format_exc());raise
    finally:
        for record in active:stop(record[1]);collect(record,'interrupted')
        log=getattr(server,'log_path',None)
        if log and Path(log).exists():shutil.copyfile(log,out/'vllm-server.log')
        server.stop(timeout=10)
        ab_rows=[r for r in rows if r.get('stage')=='AB']
        c_rows=[r for r in rows if r.get('stage')=='C']
        if c_state['status'] in ('awaiting_AB_barrier','running'):
            c_state.update(status='partial' if c_rows else 'not_executed',
                reason='Experiment ended before optional Stage C completed; no additional budget or retry')
        c_state.update(returned_attempts=len(c_rows),
            resolved=sum(r.get('resolved') is True and r.get('judge_valid') is True
                and r.get('worker_exit_code')==0 and r.get('route_evidence',{}).get('verified') for r in c_rows),
            agent_elapsed_seconds=[r.get('context_metrics',{}).get('agent_elapsed_seconds') for r in c_rows],
            outer_seconds=[r.get('outer_seconds') for r in c_rows],
            outcomes=[{'task_id':r['task_id'],'resolved':r.get('resolved'),
                'route_evidence':r.get('route_evidence'),'outer_seconds':r.get('outer_seconds')} for r in c_rows])
        summary=paired_summary(ab_rows,selected);summary['wall_seconds']=time.monotonic()-started
        summary['optional_stage_C']=c_state;summary['total_returned_attempts_including_C']=len(rows)
        save(out/'summary.json',summary);print('FINAL '+json.dumps(summary),flush=True)
        with zipfile.ZipFile('/kaggle/working/p015-weight-ab-output.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in out.rglob('*'):
                if path.is_file():archive.write(path,path.relative_to(out.parent))


if __name__=='__main__':main()
