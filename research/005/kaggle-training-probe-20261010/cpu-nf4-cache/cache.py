"""CPU-only NF4 cache preparation; no GPU allocation or model downloads."""
import json, os, subprocess, sys, time, threading, signal, traceback
from pathlib import Path

OUT = Path('/kaggle/working')
started = time.monotonic()
report = {'purpose': 'CPU-only same-source NF4 cache; not a trained model', 'events': []}
def event(stage, **kw):
    report.update(kw)
    report['events'].append({'stage':stage, 'elapsed_seconds':time.monotonic()-started})
    (OUT/'cache-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'stage':stage, **kw}), flush=True)

CHILD = r'''
import gc, hashlib, json, os, sys, time
from pathlib import Path
import torch, transformers as tr, bitsandbytes as bnb, psutil
torch.set_num_threads(min(4, os.cpu_count() or 2))
out=Path('/kaggle/working')
def emit(stage, **data):
    data={'stage':stage, **data}
    (out/'cache-child-report.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data),flush=True)
assert not torch.cuda.is_available(), 'CPU stage must not consume GPU'
emit('cpu_runtime', torch=torch.__version__, bnb=bnb.__version__, transformers=tr.__version__, ram=psutil.virtual_memory()._asdict())
text=tr.Gemma4TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
 num_attention_heads=2,num_key_value_heads=2,head_dim=8,global_head_dim=8,num_global_key_value_heads=2,
 layer_types=['sliding_attention','full_attention'],sliding_window=16,max_position_embeddings=64,
 hidden_size_per_layer_input=0,vocab_size_per_layer_input=32,num_kv_shared_layers=0,attention_k_eq_v=True,
 pad_token_id=0,bos_token_id=1,eos_token_id=2,
 rope_parameters={k:{'rope_type':'default','rope_theta':10000.0} for k in ['sliding_attention','full_attention']})
vision=tr.Gemma4VisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,
 num_key_value_heads=2,head_dim=8,position_embedding_size=4)
cfg=tr.Gemma4Config(text_config=text,vision_config=vision,audio_config=None,image_token_id=29,boi_token_id=30,eoi_token_id=31)
torch.manual_seed(0)
model=tr.Gemma4ForConditionalGeneration(cfg).to(torch.bfloat16)
model.save_pretrained('/tmp/tiny-base')
del model
gc.collect()
quant=tr.BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
model=tr.AutoModelForImageTextToText.from_pretrained('/tmp/tiny-base',local_files_only=True,dtype=torch.bfloat16,
 quantization_config=quant,device_map={'':'cpu'},attn_implementation='sdpa')
quantized=[(n,p) for n,p in model.named_parameters() if p.__class__.__name__=='Params4bit']
assert quantized and all(p.bnb_quantized for n,p in quantized)
fingerprint={n:hashlib.sha256(p.detach().view(torch.uint8).numpy().tobytes()).hexdigest() for n,p in quantized}
model.save_pretrained('/tmp/tiny-nf4',safe_serialization=True)
del model,quantized
gc.collect()
model=tr.AutoModelForImageTextToText.from_pretrained('/tmp/tiny-nf4',local_files_only=True,dtype=torch.bfloat16,
 device_map={'':'cpu'},attn_implementation='sdpa')
reloaded={n:hashlib.sha256(p.detach().view(torch.uint8).numpy().tobytes()).hexdigest() for n,p in model.named_parameters() if p.__class__.__name__=='Params4bit'}
assert fingerprint==reloaded, 'NF4 packed bytes changed on save/reload'
emit('tiny_cpu_nf4_roundtrip_ok',quantized_tensors=len(fingerprint))
del model
gc.collect()
candidates=[]
for path in Path('/kaggle/input').rglob('config.json'):
    if 'gemma-4-31b-it-qat-q4_0-unquantized' not in str(path): continue
    c=json.loads(path.read_text())
    if c.get('model_type')=='gemma4' and not c.get('quantization_config') and (path.parent/'model.safetensors.index.json').exists(): candidates.append(path.parent)
assert len(candidates)==1
source=candidates[0]
assert psutil.virtual_memory().available > 23*1024**3, 'Insufficient CPU RAM for one bounded full cache attempt'
emit('full_cpu_nf4_begin',source=str(source),ram=psutil.virtual_memory()._asdict())
t=time.monotonic()
model=tr.AutoModelForImageTextToText.from_pretrained(str(source),local_files_only=True,dtype=torch.bfloat16,
 quantization_config=quant,device_map={'':'cpu'},attn_implementation='sdpa')
emit('full_cpu_nf4_loaded',load_seconds=time.monotonic()-t,footprint_bytes=model.get_memory_footprint())
dest=out/'qat-nf4'
model.save_pretrained(dest,safe_serialization=True,max_shard_size='2GB')
tr.AutoTokenizer.from_pretrained(str(source),local_files_only=True).save_pretrained(dest)
manifest={'source':'google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2',
 'source_config_sha256':hashlib.sha256((source/'config.json').read_bytes()).hexdigest(),
 'source_index_sha256':hashlib.sha256((source/'model.safetensors.index.json').read_bytes()).hexdigest(),
 'purpose':'Untrained QAT-derived NF4 cache for training only; not competition inference base',
 'versions':{'torch':torch.__version__,'transformers':tr.__version__,'bitsandbytes':bnb.__version__},
 'load_seconds':time.monotonic()-t,'files':[]}
for p in sorted(dest.iterdir()):
 if not p.is_file():continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 manifest['files'].append({'name':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
manifest['total_bytes']=sum(x['bytes'] for x in manifest['files'])
(dest/'cache-manifest.json').write_text(json.dumps(manifest,indent=2))
emit('complete',success=True,total_bytes=manifest['total_bytes'],quantized_tensors=len([p for p in model.parameters() if p.__class__.__name__=='Params4bit']))
'''

proc = None
try:
    stages=list(Path('/kaggle/input').rglob('cpu-stage-report.json'))
    assert len(stages)==1
    import hashlib
    stage=json.loads(stages[0].read_text())
    assert stage['success']
    wheels=[]
    for w in stage['wheels']:
        p=stages[0].parent/'training-wheels'/w['name']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==w['sha256']
        wheels.append(str(p))
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-index','--target','/tmp/p005-deps',*wheels],check=True,timeout=120)
    path=Path('/tmp/p005-cache-child.py');path.write_text(CHILD)
    env=dict(os.environ,PYTHONPATH='/tmp/p005-deps',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
    event('child_start')
    with (OUT/'cache-child.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,str(path)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        def relay():
            for line in proc.stdout:
                log.write(line);log.flush()
                if line.startswith('{'):print(line.rstrip(),flush=True)
        thread=threading.Thread(target=relay,daemon=True);thread.start()
        try: code=proc.wait(timeout=3300)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=124
        thread.join(timeout=10)
    event('finished',success=code==0,exit_code=code)
    if code:print((OUT/'cache-child.log').read_text()[-12000:],flush=True)
except Exception as exc:
    event('failed',success=False,error=repr(exc),traceback=traceback.format_exc())
finally:
    if proc is not None and proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
if not report.get('success'):raise SystemExit(2)
