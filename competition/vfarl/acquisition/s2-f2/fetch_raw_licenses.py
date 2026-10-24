"""Read public raw upstream licenses when the GitHub metadata API quota is exhausted."""
import json,urllib.request,hashlib,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCES=[('facebookresearch/hydra','0f03eb60',['LICENSE']),('pallets/jinja','ada0a9a6',['LICENSE.txt','LICENSE.rst','LICENSE']),('seperman/deepdiff','ed252022',['LICENSE']),('benoitc/gunicorn','bacbf8aa',['LICENSE']),('tornadoweb/tornado','d5ac65c1',['LICENSE']),('python-trio/trio','cfbbe2c1',['LICENSE','LICENSE.MIT','LICENSE.APACHE2']),('PennyLaneAI/pennylane','a9ad7a0df257970fe2edd32b6cccad25ff01cb88',['LICENSE.txt','LICENSE']),('adrienverge/yamllint','8513d9b9',['LICENSE'])]
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
 (ROOT/'raw-upstream-license-provenance.json').write_text(json.dumps(out,indent=2,ensure_ascii=False));print([(x['repo'],x.get('text_prefix',x.get('error'))) for x in out],flush=True)
