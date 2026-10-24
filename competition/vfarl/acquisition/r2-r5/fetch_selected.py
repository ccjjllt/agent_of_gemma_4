"""Acquire exact selected public train tasks using pinned metadata row joins."""
import json,hashlib,urllib.parse,urllib.request,concurrent.futures,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
REV='77cab9055d42ab4a5c25c89a8f937096db13558e';DS='SWE-bench/SWE-smith-py'
IDS=[r['instance_id'] for r in json.load(open(OUT/'selection.json'))['rows']] if (OUT/'selection.json').exists() else []
INDEX={r['instance_id']:r for r in map(json.loads,(OUT.parent/'general/smith-task-index.jsonl').open())}
RAW=OUT/'artifacts'/'tasks';RAW.mkdir(parents=True,exist_ok=True)
def get(iid):
 meta=INDEX[iid];idx=meta.get('row_idx',meta.get('row_index'));u='https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode({'dataset':DS,'config':'default','split':'train','offset':idx,'length':1,'revision':REV})
 for attempt in range(4):
  try:
   with urllib.request.urlopen(u,timeout=75) as resp:
    b=resp.read(4_000_001);rev=resp.headers.get('X-Revision')
    if rev!=REV or len(b)>4_000_000:raise ValueError('revision/size mismatch '+str(rev))
    d=json.loads(b);r=d['rows'][0]
    if r['row_idx']!=idx or r['row']['instance_id']!=iid:raise ValueError('row identity mismatch')
    target=RAW/(iid+'.json');target.write_bytes(b)
    return {'instance_id':iid,'dataset':DS,'revision':REV,'split':'train','row_idx':idx,'source_url':u,'response_bytes':len(b),'response_sha256':hashlib.sha256(b).hexdigest(),'local_path':str(target),'source_row':r['row']}
  except Exception:
   if attempt==3:raise
   time.sleep(2**attempt)
if __name__=='__main__':
 out=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for r in pool.map(get,IDS):out.append(r);print('fetched',r['instance_id'],r['response_bytes'],flush=True)
 summary=[{k:v for k,v in r.items() if k!='source_row'} for r in out]
 (OUT/'direct-rows-acquisition-receipts.json').write_text(json.dumps({'scope':'Fixed SHA public train selection; no source commands executed; patch is bug introduction, never answer.','rows':summary,'total_response_bytes':sum(r['response_bytes'] for r in out)},indent=2))
 with (OUT/'artifacts/direct-rows-download.jsonl').open('w') as f:
  for r in out:f.write(json.dumps(r,ensure_ascii=False)+'\n')
