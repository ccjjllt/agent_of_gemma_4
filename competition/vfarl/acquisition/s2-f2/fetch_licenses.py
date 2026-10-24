"""Preserve upstream license evidence at dataset-specified source revisions."""
import json,urllib.request,urllib.parse,base64,hashlib,concurrent.futures,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCES={'hukkin/tomli':'443a0c1b','pdfminer/pdfminer.six':'1a8bd2f7','graphql-python/graphene':'82903263','spulec/freezegun':'5f171db0','kayak/pypika':'1c9646f0','facebookresearch/hydra':'0f03eb60','mahmoud/boltons':'3bfcfdd0','tox-dev/pipdeptree':'c31b6418','pallets/jinja':'ada0a9a6','seperman/deepdiff':'ed252022','benoitc/gunicorn':'bacbf8aa','tornadoweb/tornado':'d5ac65c1','python-trio/trio':'cfbbe2c1','python-cmd2/cmd2':'0aac6cee56a92bb6358106329f4f0c20e85bb7bc','PennyLaneAI/pennylane':'a9ad7a0df257970fe2edd32b6cccad25ff01cb88'}
def fetch(item):
 repo,rev=item;url='https://api.github.com/repos/'+repo+'/license?ref='+rev
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers={'Accept':'application/vnd.github+json','User-Agent':'gemma4-public-train-provenance'}),timeout=30) as r:b=r.read(100001)
  d=json.loads(b);text=base64.b64decode(d['content']);p=ROOT/(repo.replace('/','__')+'.license.txt');p.write_bytes(text)
  return {'repo':repo,'revision':rev,'api_url':url,'license_path':str(p),'license_text_sha256':hashlib.sha256(text).hexdigest(),'blob_sha':d['sha'],'html_url':d['html_url'],'api_license':d['license'],'response_sha256':hashlib.sha256(b).hexdigest()}
 except Exception as e:return {'repo':repo,'revision':rev,'api_url':url,'error':str(e)}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:out=list(ex.map(fetch,SOURCES.items()))
 (ROOT/'upstream-license-provenance.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
 print([(x['repo'],x.get('api_license',{}).get('spdx_id',x.get('error'))) for x in out],flush=True)
