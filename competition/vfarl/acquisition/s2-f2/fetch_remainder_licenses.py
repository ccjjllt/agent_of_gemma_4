"""Read public raw upstream licenses when the GitHub metadata API quota is exhausted."""
import json,urllib.request,hashlib,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCES=[('martinblech/xmltodict','0952f382',['LICENSE']),('marshmallow-code/marshmallow','9716fc62',['LICENSE']),('pydantic/pydantic','acb0f10f',['LICENSE']),('agronholm/typeguard','b6a7e438',['LICENSE']),('agronholm/exceptiongroup','0b4f4937',['LICENSE']),('matthewwithanm/python-markdownify','6258f5c3',['LICENSE']),('andialbrecht/sqlparse','e57923b3',['LICENSE']),('marshmallow-code/apispec','8b421526',['LICENSE']),('encode/starlette','db5063c2',['LICENSE.md','LICENSE']),('pyupio/safety','7654596b',['LICENSE']),('python-hyper/h11','bed0dd4a',['LICENSE.txt','LICENSE']),('un33k/python-slugify','872b3750',['LICENSE'])]
def fetch(item):
 repo,rev,filenames=item
 for fn in filenames:
  url='https://raw.githubusercontent.com/'+repo+'/'+rev+'/'+fn
  try:
   with urllib.request.urlopen(url,timeout=25) as r:b=r.read(100001)
   p=ROOT/(repo.replace('/','__')+'.license.txt');p.write_bytes(b)
   return {'repo':repo,'revision':rev,'url':url,'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'text_prefix':b[:250].decode(errors='replace')}
  except Exception as e:err=str(e)
 return {'repo':repo,'revision':rev,'error':err}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:out=list(ex.map(fetch,SOURCES))
 (ROOT/'remainder-upstream-license-provenance.json').write_text(json.dumps(out,indent=2,ensure_ascii=False));print([(x['repo'],x.get('text_prefix',x.get('error'))) for x in out],flush=True)
