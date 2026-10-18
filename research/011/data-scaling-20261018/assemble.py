"""Encode complete replay conversations and freeze the first 128 eligible issues."""
import hashlib, json, subprocess, sys
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def write_rows(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows))

def main():
    final=HERE/'encoded-train-12b.jsonl'
    if final.exists():raise RuntimeError('Frozen output exists; create a new version instead of overwriting')
    usage=json.loads((HERE/'data-usage-full.json').read_text());ordered=usage['selected']
    results=json.loads((HERE/'production-results.json').read_text());processed={r['instance_id']:r for r in results}
    transactions=[]
    for row in ordered:
        result=processed.get(row['instance_id'])
        if not result or result['status']!='local_edits_applied':continue
        tx=json.loads((HERE/result['transaction']).read_text())
        if tx.get('producer_revision')!='p011-local-2':raise ValueError('Mixed producer revision')
        transactions.append(tx)
    inputs=HERE/'transactions-all-applied.jsonl';write_rows(inputs,transactions)
    encoded=HERE/'encoded-all-applied-12b.jsonl'
    subprocess.run([sys.executable,str(ROOT/'training/encoding/gemma_tool_chat.py'),'encode',
        '--input',str(inputs),'--output',str(encoded),
        '--tokenizer',str(ROOT/'training/artifacts/tokenizers/gemma4-12b-ct-v2'),
        '--tools',str(ROOT/'training/configs/harness_tools.json'),
        '--chat-template',str(ROOT/'training/configs/chat_templates/gemma4_12b_ct_v2.jinja'),
        '--thinking','on','--thought-loss','mask','--max-seq-len','8192','--on-oversize','skip'],check=True)
    available=[json.loads(x) for x in encoded.read_text().splitlines()]
    if len(available)<128:
        print(json.dumps({'available':len(available),'needed':128,'frozen':False}));return
    fresh=available[:128];new_ids={r['meta']['group']['issue'] for r in fresh}
    newfile=HERE/'encoded-new128-12b.jsonl';write_rows(newfile,fresh)
    write_rows(HERE/'transactions-new128.jsonl',[t for t in transactions if t['group']['issue'] in new_ids])
    oldfiles=[ROOT/'research/009/training-data-20261018/encoded-train-12b.jsonl',ROOT/'research/009/training-data-v2-20261018/encoded-v2-12b.jsonl']
    old=[];previous=[]
    for path in oldfiles:
        rows=[json.loads(x) for x in path.read_text().splitlines()];take=[r for r in rows if len(r['input_ids'])<=8192]
        old.extend(take);previous.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'selected_rows':len(take),'excluded_oversize':[r['meta']['group']['issue'] for r in rows if len(r['input_ids'])>8192]})
    combined=fresh+old;write_rows(final,combined)
    issues=[r['meta']['group']['issue'] for r in combined]
    if len(issues)!=len(set(issues)):raise ValueError('Duplicate issue across new/previous data')
    provenance={key:combined[0]['meta'][key] for key in ('tokenizer_hash','chat_template_hash','tools_hash','protocol_hash','protocol_id')}
    for row in combined:
        if any(row['meta'][key]!=value for key,value in provenance.items()):raise ValueError('Protocol provenance differs')
    stop=max(i for i,r in enumerate(ordered) if r['instance_id'] in new_ids)+1
    entries=[]
    for iid in [r['meta']['group']['issue'] for r in fresh]:
        folder=HERE/'outputs'/iid
        assets=[p for p in folder.rglob('*') if p.is_file()]
        entries.append({'issue':iid,'repo':next(r['repo'] for r in ordered if r['instance_id']==iid),
                        'tier':'local_edit_application','behavior_verified':False,'task_resolved':None,
                        'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(assets)]})
    dump(HERE/'admitted-source-manifest.json',entries)
    lengths=sorted(len(r['input_ids']) for r in combined)
    manifest={'status':'frozen','split':'train','encoded':{'path':str(final.relative_to(ROOT)),'sha256':sha(final)},
              'rows':len(combined),'independent_issues':len(set(issues)),'independent_repos':len({r['meta']['group']['repo'] for r in combined}),
              'max_sequence_length':max(lengths),'training_sequence_limit':8192,'provenance':provenance,
              'input_tokens':sum(lengths),'supervised_tokens':sum(r['meta']['supervised_tokens'] for r in combined),
              'length_distribution':{'min':lengths[0],'median':lengths[len(lengths)//2],'p90':lengths[int(.9*(len(lengths)-1))],'max':lengths[-1],'le6144':sum(n<=6144 for n in lengths),'le8192':len(lengths)},
              'layers':{'new_local_edit':{'rows':128,'independent_repos':len({r['meta']['group']['repo'] for r in fresh}),'encoded':{'path':str(newfile.relative_to(ROOT)),'sha256':sha(newfile)},'behavior_verified':False,'task_resolved':None},
                        'previous_behavior_replay':{'rows':len(old),'files':previous,'task_resolved':None}},
              'selection':{'order':'data-usage-full.json selected order; first 128 successfully replayed and whole-transaction encoded <=8192, never concurrent completion order',
                           'metadata_candidates':len(ordered),'processed_candidates':len(results),'selected_prefix_end_one_based':stop,'selected_prefix_final_issue':ordered[stop-1]['instance_id'],
                           'processed_status_counts':dict(Counter(r['status'] for r in results))},
              'truncated':False,'gpu_or_teacher_calls_for_data':0,'training_executed_by_this_producer':False,
              'source_manifest':{'path':str((HERE/'admitted-source-manifest.json').relative_to(ROOT)),'sha256':sha(HERE/'admitted-source-manifest.json')},
              'frozen_policy':'Do not overwrite this encoded file or its source transactions after training is launched. New work requires another version.'}
    dump(HERE/'dataset-manifest.json',manifest)
    dump(HERE/'data-usage.json',dict(usage,admitted_new_issues=[r['meta']['group']['issue'] for r in fresh],
         additional_prior_behavior_issues=[r['meta']['group']['issue'] for r in old],training_file=manifest['encoded'],
         training_role='Train only; all assigned repositories excluded from any future selection replacements',automatic_training=False))
    print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
