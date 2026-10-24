"""Read-only candidate mining of the pinned public TRAIN dataset; never execute task text."""
import urllib.request,urllib.parse,json,hashlib,time,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REV='10483de0f50fe5da545942705a76c6150171af7f';DS='nebius/SWE-rebench-V2'
QUERIES=['deep copy','text width','shared dictionary','coordinate offset']
def excluded(repo):
 s=repo.lower();tail=s.split('/')[-1]
 return any(x in s for x in ('fastapi','httpx')) or tail in ('rich','requests') or tail.startswith(('rich-','requests-')) or tail.endswith(('-rich','-requests'))
def fetch(q):
 url='https://datasets-server.huggingface.co/search?'+urllib.parse.urlencode({'dataset':DS,'config':'default','split':'train','query':q,'length':30,'revision':REV})
 for retry in range(2):
  try:
   with urllib.request.urlopen(url,timeout=30) as r:
    b=r.read(8000001);rev=r.headers.get('x-revision')
    if len(b)>8000000:raise RuntimeError('search response >8MB')
    if rev!=REV:raise RuntimeError('wrong revision')
    d=json.loads(b)
    rows=[{'dataset':DS,'revision':REV,'split':'train','row_idx':x['row_idx'],'source_url':url,'row':x['row'],'source_response_sha256':hashlib.sha256(b).hexdigest(),'canonical_row_sha256':hashlib.sha256(json.dumps(x['row'],sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()} for x in d['rows'] if x['row']['language']=='python' and not excluded(x['row']['repo'])]
    (ROOT/('search-'+q.replace(' ','-')+'.raw.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    print(q,'matching',d['num_rows_total'],'allowed_python',len(rows),'bytes',len(b),flush=True)
    return {'query':q,'url':url,'bytes':len(b),'response_sha256':hashlib.sha256(b).hexdigest(),'revision':rev,'search_total':d['num_rows_total'],'retained_allowed_python_rows':len(rows),'rows':[(x['row']['instance_id'],x['row_idx']) for x in rows]}
  except Exception as e:
   if retry==1:return {'query':q,'error':str(e)}
   time.sleep(2**retry)
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:results=list(ex.map(fetch,QUERIES))
 (ROOT/'search-provenance-second.json').write_text(json.dumps(results,indent=2));print('total_download_bytes',sum(x.get('bytes',0) for x in results),flush=True)
