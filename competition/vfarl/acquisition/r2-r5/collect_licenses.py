"""Retain primary license text at one selected immutable commit per source repo."""
import json,urllib.request,hashlib,concurrent.futures,urllib.error
from pathlib import Path
OUT=Path(__file__).resolve().parent
ENV=json.load(open(OUT/'environment-identities.json'))['rows'];chosen={}
for r in ENV:chosen.setdefault(r['mirror_repo'],r)
P=OUT/'artifacts/licenses';P.mkdir(exist_ok=True,parents=True)
def get(r):
 for path in ['LICENSE','COPYING','LICENSE.txt','LICENSE.rst','LICENSE.md']:
  u='https://raw.githubusercontent.com/'+r['mirror_repo']+'/'+r['buggy_commit']+'/'+path
  try:
   with urllib.request.urlopen(u,timeout=35) as resp:b=resp.read(150000)
   if len(b)>=150000:raise ValueError('license response too large')
   target=P/(r['mirror_repo'].split('/')[-1]+'__'+path);target.write_bytes(b)
   t=b.decode('utf8',errors='replace');license_id='MIT' if 'Permission is hereby granted, free of charge' in t else 'Apache-2.0' if 'Apache License' in t and 'Version 2.0' in t else 'LGPL-2.1-or-later' if 'GNU LESSER GENERAL PUBLIC LICENSE' in t and 'Version 2.1' in t else 'BSD-style' if 'Redistribution and use in source and binary forms' in t else 'see retained primary license text'
   supporting=[]
   if 'LICENSE.APACHE2 or LICENSE.MIT' in t:
    license_id='MIT OR Apache-2.0; dual-license declaration'
    for name in ['LICENSE.APACHE2','LICENSE.MIT']:
     url='https://raw.githubusercontent.com/'+r['mirror_repo']+'/'+r['buggy_commit']+'/'+name
     with urllib.request.urlopen(url,timeout=35) as resp:extra=resp.read(150000)
     dest=P/(r['mirror_repo'].split('/')[-1]+'__'+name);dest.write_bytes(extra)
     supporting.append({'license_path':name,'source_url':url,'sha256':hashlib.sha256(extra).hexdigest(),'bytes':len(extra),'local_path':str(dest)})
   return {'upstream_repo':r['upstream_repo'],'mirror_repo':r['mirror_repo'],'evidence_commit':r['buggy_commit'],'license_path':path,'source_url':u,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'local_path':str(target),'text_identification':license_id,'supporting_license_files':supporting,'scope':'Primary repository license at one selected immutable commit; not an audit of all files or dependencies.'}
  except urllib.error.HTTPError as e:
   if e.code!=404:raise
 raise RuntimeError('No standard root license path '+r['mirror_repo'])
if __name__=='__main__':
 rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:
  for r in p.map(get,chosen.values()):rows.append(r);print(r['upstream_repo'],r['text_identification'],flush=True)
 (OUT/'license-evidence.json').write_text(json.dumps({'dataset_license':'MIT','rows':rows},indent=2))
