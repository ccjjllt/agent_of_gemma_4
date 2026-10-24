"""Read only row groups needed for exact fixed-revision Smith train selections."""
import json,hashlib,importlib.util,concurrent.futures,collections
from pathlib import Path
import pyarrow.parquet as pq
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('idx',OUT.parent/'general/index_smith_train.py');x=importlib.util.module_from_spec(spec);spec.loader.exec_module(x)
IDX={r['instance_id']:r for r in map(json.loads,(OUT.parent/'general/smith-task-index.jsonl').open())}
TREE={r['path']:r for r in json.load(open(OUT.parent/'general/smith-tree.json')) if r['path'].endswith('.parquet')}
SEL=json.load(open(OUT/'selection.json'))['rows'];RAW=OUT/'artifacts/tasks';RAW.mkdir(exist_ok=True,parents=True)

def load_group(pair):
 file,wanted=pair;f=TREE[file];u='https://huggingface.co/datasets/'+x.DATASET+'/resolve/'+x.REV+'/'+file
 with x.RangeReader(u,f['size']) as rf:
  p=pq.ParquetFile(rf);bounds=[];off=0
  for i in range(p.num_row_groups):
   rg=p.metadata.row_group(i);n=rg.num_rows
   if any(off<=r['file_row_idx']<off+n for r in wanted):
    size=sum(rg.column(j).total_compressed_size for j in range(rg.num_columns));bounds.append((i,off,n,size))
   off+=n
  print('plan',file,'groups',[(i,n,size) for i,off,n,size in bounds],flush=True)
  if sum(z[3] for z in bounds)>35_000_000:raise ValueError('per-file selected compressed payload cap')
  out=[]
  for i,off,n,size in bounds:
   rows=p.read_row_group(i).to_pylist()
   for meta in wanted:
    j=meta['file_row_idx']-off
    if not 0<=j<n:continue
    row=rows[j];assert row['instance_id']==meta['instance_id']
    b=json.dumps({'dataset':x.DATASET,'source_revision':x.REV,'split':'train','transport':'fixed_revision_parquet_row_group','parquet_file':file,'parquet_row_idx':meta['file_row_idx'],'rows':[{'row_idx':meta['row_idx'],'row':row}]},ensure_ascii=False).encode()
    target=RAW/(row['instance_id']+'.json');target.write_bytes(b)
    out.append({'instance_id':row['instance_id'],'dataset':x.DATASET,'revision':x.REV,'split':'train','row_idx':meta['row_idx'],'file':file,'file_row_idx':meta['file_row_idx'],'transport':'fixed_revision_parquet_row_group','source_url':u,'response_sha256':None,'canonical_artifact_sha256':hashlib.sha256(b).hexdigest(),'artifact_bytes':len(b),'local_path':str(target)})
    print('acquired',row['instance_id'],len(b),flush=True)
 return out
if __name__=='__main__':
 groups=collections.defaultdict(list)
 for r in SEL:
  if not (RAW/(r['instance_id']+'.json')).exists():groups[IDX[r['instance_id']]['file']].append(IDX[r['instance_id']])
 results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  fs=[pool.submit(load_group,pair) for pair in groups.items()]
  for f in concurrent.futures.as_completed(fs):results.extend(f.result())
 target=OUT/'parquet-acquisition-manifest.json'
 previous=json.load(target.open()) if target.exists() else {'selected_rows':[],'requests':[]}
 combined={r['instance_id']:r for r in previous['selected_rows']}
 combined.update({r['instance_id']:r for r in results})
 requests=previous['requests']+x.RECEIPTS
 target.write_text(json.dumps({'dataset':x.DATASET,'revision':x.REV,'split':'train','scope':'Acquired task rows including subsequently rejected candidates; final membership is in task-acquisition-manifest.json.','selected_rows':list(combined.values()),'request_bytes':sum(r['bytes'] for r in requests),'requests':requests},indent=2))
