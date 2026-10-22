#!/usr/bin/env python3
"""Build CPU-only calibration bundles; launch training only via explicit `run`.

No dependency installation, model download, credentials, or automatic GPU work.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2] if len(HERE.parents) >= 3 else HERE
CORE = ('common.py', 'train_sft.py', 'train_ddp.py', 'train_fsdp.py',
        'fsdp_qlora.py', 'fsdp_checkpoint.py', 'export_peft.py', 'memory.py', 'attention.py')
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                   'config.json', 'generation_config.json')
RUN_ENV = {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
           'TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1',
           'OMP_NUM_THREADS':'1','NCCL_DEBUG':'WARN','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1',
           'FSDP_TRANSFORMER_CLS_TO_WRAP':'Gemma4TextDecoderLayer'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def representative(rows, limit):
    """Cover length quantiles while preferring different repositories; no copies."""
    remaining = sorted(rows, key=lambda r: (len(r['input_ids']), r['meta']['group']['issue']))
    count = min(limit, len(remaining))
    if not count: return []
    lengths = [len(r['input_ids']) for r in remaining]
    targets = [lengths[round(i * (len(lengths)-1) / max(1, count-1))] for i in range(count)]
    selected, repos = [], set()
    # Start near the upper limit so the longest real sample is always included.
    for target in reversed(targets):
        unused_repo = [r for r in remaining if r['meta']['group']['repo'] not in repos]
        pool = unused_repo or remaining
        chosen = min(pool, key=lambda r: (abs(len(r['input_ids'])-target), r['meta']['group']['issue']))
        selected.append(chosen)
        repos.add(chosen['meta']['group']['repo'])
        remaining.remove(chosen)
    return sorted(selected, key=lambda r: (len(r['input_ids']), r['meta']['group']['issue']))


def build(args):
    sys.path.insert(0, str(REPO))
    from training.common import load_encoded_jsonl
    from training.encoding.gemma_tool_chat import load_tokenizer, tokenizer_fingerprint, normalize_tools, sha256_json
    from training.train_sft import validate_config
    if not args.name or '/' in args.name or '..' in args.name:
        raise ValueError('Use a simple new bundle name')
    if not 1 <= args.max_samples <= 16: raise ValueError('Each bucket accepts 1..16 unique samples')
    tools, tool_protocol = normalize_tools(read(args.tools))
    if {t['function']['name'] for t in tools} != {'bash','str_replace_editor','submit'}:
        raise ValueError('Calibration requires the frozen native three-tool protocol')
    tool_hash = sha256_json(tools)
    tokenizer = load_tokenizer(str(args.tokenizer))
    token_hash = tokenizer_fingerprint(tokenizer)
    template_hash = digest(args.tokenizer/'chat_template.jinja')
    admission_receipts=[]
    admitted_shards={}
    for path in args.admission_manifest:
        evidence=read(path)
        admission_receipts.append({'path':str(path.resolve()),'sha256':digest(path)})
        for name,facts in evidence.get('shards',{}).items():
            admitted_shards[str((path.parent/name).resolve())]=facts
    all_rows, input_receipts, provenance, seen = [], [], None, set()
    for path in args.data:
        facts=admitted_shards.get(str(path.resolve()))
        if facts is None or digest(path)!=facts['sha256'] or path.stat().st_size!=facts['bytes']:
            raise ValueError('Input shard is absent from or differs from the supplied admission manifest')
        rows, receipt = load_encoded_jsonl(path, 32768, require_real=True)
        if len(rows)!=facts['records']: raise ValueError('Admission manifest record count mismatch')
        if provenance is not None and provenance != receipt['provenance']:
            raise ValueError('Do not mix thought modes or protocol fingerprints')
        provenance = receipt['provenance']
        if (provenance['tools_hash'] != tool_hash or provenance['tokenizer_hash'] != token_hash
                or provenance['chat_template_hash'] != template_hash
                or (tool_protocol and provenance['protocol_id'] != tool_protocol)):
            raise ValueError('Encoded input differs from supplied tools/tokenizer/template')
        for row in rows:
            meta = row['meta']; group = meta['group']
            key = (group['repo'].lower(), group['issue'])
            if key in seen: raise ValueError('Repeated independent issue across input shards: ' + str(key))
            if meta.get('truncated') or meta.get('truncated_tool_block'):
                raise ValueError('Only whole untruncated admitted trajectories may be packaged')
            seen.add(key); all_rows.append(row)
        input_receipts.append({'path':str(path.resolve()), **receipt})
    reference_path = REPO/'research/015/capacity/output-fsdp-metadata/p015-fsdp/run/training_manifest.json'
    reference = read(reference_path)
    destination = REPO/'research/016/data/gpu-bundles'/args.name
    generated = REPO/'research/016/training/generated'/args.name
    if destination.exists() or generated.exists():
        raise ValueError('Bundle name already exists; use a new --name to freeze updated trainer bytes')
    destination.mkdir(parents=True); generated.mkdir(parents=True)
    for filename in CORE:
        target = destination/'training'/filename; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(REPO/'training'/filename,target)
    for source in (REPO/'training/encoding').glob('*.py'):
        target = destination/'training/encoding'/source.name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    shutil.copyfile(Path(__file__),destination/'prepare_gpu_run.py')
    for filename in TOKENIZER_FILES:
        source = args.tokenizer/filename
        if not source.is_file(): raise ValueError('Missing fixed tokenizer asset: '+filename)
        target = destination/'assets/tokenizer'/filename; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    shutil.copyfile(args.tools,destination/'assets/native-tools.json')
    for i,path in enumerate(args.admission_manifest):
        target=destination/'provenance'/('admission-%02d.json'%i)
        target.parent.mkdir(exist_ok=True);shutil.copyfile(path,target)
    runtime = {'source_manifest':str(reference_path.relative_to(REPO)),
               'source_manifest_sha256':digest(reference_path),
               'versions':reference['versions'],
               'evidence':'P015 actual 31B NF4 FSDP updates; not evidence of 16K/32K capacity or this new checkpoint implementation.'}
    write(destination/'reference-runtime.json',runtime)
    buckets = {}
    for label, lower, upper in [('16k',0,16384),('32k',16384,32768)]:
        pool = [r for r in all_rows if lower < len(r['input_ids']) <= upper]
        chosen = representative(pool,args.max_samples)
        if not chosen:
            buckets[label] = {'available':False,'reason':'No admitted full trajectory in this length bucket'}
            continue
        data_path = destination/'data'/('calibration-'+label+'.jsonl')
        data_path.parent.mkdir(exist_ok=True)
        with data_path.open('w') as f:
            for row in chosen: f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        cfg = dict(reference['config'])
        cfg.update(model_name_or_path=None,model_revision=None,tokenizer_name_or_path=None,
                   tokenizer_revision=None,chat_template_path=None,tool_schema_path='assets/native-tools.json',
                   protocol_id=provenance['protocol_id'],device='cuda',max_sequence_length=upper,
                   micro_batch_size=1,gradient_accumulation_steps=1,epochs=1,
                   max_steps=math.ceil(len(chosen)/4),checkpoint_every_n_steps=1,
                   source_version_note='P016 native Smith capacity calibration; existing official-QAT-derived NF4 cache; no GPU result yet')
        cfg = validate_config(cfg)
        config_path = destination/'configs'/('calibration-'+label+'.json')
        write(config_path,cfg); write(generated/config_path.name,cfg)
        loaded, data_receipt = load_encoded_jsonl(data_path,upper,require_real=True)
        buckets[label] = {'available':True,'data':str(data_path.relative_to(destination)),
           'config':str(config_path.relative_to(destination)),'data_receipt':data_receipt,
           'source_pool_rows':len(pool),'world_size':4,'micro_batch_size_per_rank':1,
           'gradient_accumulation_steps':1,'planned_total_optimizer_steps':cfg['max_steps'],
           'lengths':[len(r['input_ids']) for r in loaded],
           'samples':[{'issue':r['meta']['group']['issue'],'repo':r['meta']['group']['repo'],
                       'input_tokens':len(r['input_ids']),
                       'supervised_tokens_after_shift':sum(x!=-100 for x in r['labels'][1:]),
                       'raw_source_sha256':r['meta']['source']['sha256']} for r in loaded]}
    files = {str(p.relative_to(destination)):{'bytes':p.stat().st_size,'sha256':digest(p)}
             for p in sorted(destination.rglob('*')) if p.is_file()}
    manifest = {'kind':'p016-frozen-gpu-calibration-bundle','name':args.name,
        'prepared_on_cpu':True,'gpu_started':False,'contains_model_weights':False,
        'contains_credentials':False,'selection_rule':'Full admitted train trajectories; distinct issue; upper/lower length quantiles with repo coverage; no row duplication; <=16 rows per bucket.',
        'data_provenance':provenance,'input_sources':input_receipts,
        'admission_manifests':admission_receipts,'runtime_environment':RUN_ENV,'reference_runtime':runtime,
        'buckets':buckets,'files':files}
    write(destination/'bundle-manifest.json',manifest)
    archive = destination.parent/(destination.name+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar: tar.add(destination,arcname=destination.name)
    receipt = {'bundle_directory':str(destination),'archive':str(archive),
       'archive_bytes':archive.stat().st_size,'archive_sha256':digest(archive),
       'bundle_manifest_sha256':digest(destination/'bundle-manifest.json'),
       'buckets':{k:{x:v[x] for x in ('available','planned_total_optimizer_steps','lengths') if x in v} for k,v in buckets.items()},
       'trainer_files':{k:v for k,v in files.items() if k.startswith('training/')},'gpu_started':False}
    write(generated/'build-manifest.json',receipt)
    print(json.dumps(receipt,indent=2))


def process_snapshot():
    """Small process identities, never command lines or unrelated process signals."""
    result = {}
    if sys.platform.startswith('linux'):
        # The command name may itself contain spaces/parentheses. Fields after
        # its final ')' start at field 3; starttime (field 22) guards PID reuse.
        for path in Path('/proc').iterdir():
            if not path.name.isdigit(): continue
            try:
                fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
                result[int(path.name)] = (int(fields[1]), int(fields[2]), fields[19], fields[0])
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
    else:
        # CPU development hosts have no /proc. GPU hosts use Linux starttime
        # above; this fallback has the coarser lstart identity supplied by ps.
        listing = subprocess.run(['ps','-axo','pid=,ppid=,pgid=,stat=,lstart='],
                                 check=True, capture_output=True, text=True).stdout
        for line in listing.splitlines():
            fields = line.split(maxsplit=4)
            if len(fields) == 5:
                result[int(fields[0])] = (int(fields[1]), int(fields[2]), fields[4], fields[3][0])
    return result


class OwnedProcesses:
    """Remember observed descendants even after their parent exits/reparents them.

    This follows torchrun's ordinary worker tree, including workers in separate
    sessions. It is not a cgroup: an unobserved child which immediately daemonizes
    between snapshots cannot be recovered after its parent has disappeared.
    """
    def __init__(self, process):
        self.process, self.identities = process, {}
        snapshot = process_snapshot()
        if process.poll() is None and process.pid in snapshot:
            self.identities[process.pid] = snapshot[process.pid][2]
        self.refresh(snapshot)

    def refresh(self, snapshot=None):
        snapshot = process_snapshot() if snapshot is None else snapshot
        owned = {pid for pid, identity in self.identities.items()
                 if pid in snapshot and snapshot[pid][2] == identity}
        while True:
            children = {pid for pid, entry in snapshot.items()
                        if entry[0] in owned and pid not in owned}
            if not children: break
            for pid in children: self.identities[pid] = snapshot[pid][2]
            owned.update(children)
        return {pid for pid in owned if snapshot[pid][3] not in ('Z', 'X')}

    def signal(self, pids, signum):
        # Recheck identity immediately before signalling. Never signal a whole
        # NVIDIA workload, a name-matched process, or an unrelated process group.
        snapshot = process_snapshot()
        for pid in pids:
            if (pid not in snapshot or snapshot[pid][2] != self.identities.get(pid)
                    or snapshot[pid][3] in ('Z', 'X')): continue
            try: os.kill(pid, signum)
            except ProcessLookupError: pass


def stop_group(process, owned, grace_seconds=15):
    deadline = time.monotonic() + max(0, grace_seconds)
    alive = owned.refresh()
    # Give torchrun the first chance to forward TERM and reap its workers.
    owned.signal([process.pid], signal.SIGTERM)
    while alive and time.monotonic() < deadline:
        if process.poll() is not None:
            owned.signal(alive, signal.SIGTERM)
        time.sleep(min(.1, max(0, deadline-time.monotonic())))
        alive = owned.refresh()
    # torchrun workers can own separate sessions. Killing only its PGID leaves
    # those workers alive if torchrun exits first or its shutdown stalls.
    owned.signal(alive - {process.pid}, signal.SIGKILL)
    owned.signal([process.pid] if process.pid in alive else [], signal.SIGKILL)
    try: process.wait(timeout=1)
    except subprocess.TimeoutExpired: pass
    remaining = owned.refresh()
    if remaining:
        print(json.dumps({'cleanup_signalled_but_not_yet_exited_pids': sorted(remaining)}), flush=True)


def run(args):
    bundle = args.bundle.resolve(); manifest = read(bundle/'bundle-manifest.json')
    entry = manifest['buckets'][args.bucket]
    if not entry['available']: raise ValueError('No actual data in requested bucket')
    if not 60 <= args.wall_seconds <= 7200: raise ValueError('Set a wall limit from 60 to 7200 seconds')
    if args.resume and args.init_run: raise ValueError('--resume and --init-run are different operations')
    if args.stop_after_checkpoint is not None and not 1 <= args.stop_after_checkpoint < entry['planned_total_optimizer_steps']:
        raise ValueError('Controlled stop must be before the frozen final optimizer step')
    if args.resume and args.stop_after_checkpoint is not None:
        saved=read(args.resume/'complete.json')['progress']['global_step']
        if args.stop_after_checkpoint <= saved:
            raise ValueError('Controlled stop must follow the resumed committed step')
    for relative, facts in manifest['files'].items():
        p = bundle/relative
        if not p.is_file() or p.stat().st_size != facts['bytes'] or digest(p) != facts['sha256']:
            raise ValueError('Frozen bundle file changed: '+relative)
    model = args.model.expanduser().resolve(); tokenizer = args.tokenizer.expanduser().resolve()
    if not model.is_dir() or not tokenizer.is_dir(): raise ValueError('Bind existing local --model and --tokenizer directories')
    command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4','--max-restarts=0',
      '-m','training.train_fsdp','--config',str(bundle/entry['config']),
      '--data',str(bundle/entry['data']),'--out',str(args.out.resolve()),
      '--model',str(model),'--tokenizer',str(tokenizer),
      '--tools',str(bundle/'assets/native-tools.json'),
      '--chat-template',str(bundle/'assets/tokenizer/chat_template.jinja')]
    if args.resume: command += ['--resume',str(args.resume.resolve())]
    if args.init_run: command += ['--init-run',str(args.init_run.resolve())]
    python_paths=[str(bundle)]
    if os.environ.get('PYTHONPATH'): python_paths.append(os.environ['PYTHONPATH'])
    env = dict(os.environ,**RUN_ENV,PYTHONPATH=os.pathsep.join(python_paths))
    # No --max-steps override: resume must keep the original total schedule.
    print(json.dumps({'explicit_gpu_launch':command,'wall_seconds':args.wall_seconds,
       'shutdown_grace_reserved_seconds':15,
       'planned_total_steps':entry['planned_total_optimizer_steps']},indent=2),flush=True)
    # Fail before spawning if the host disallows process identity inspection.
    process_snapshot()
    started = time.monotonic(); process = subprocess.Popen(command,cwd=bundle,env=env,start_new_session=True)
    owned = OwnedProcesses(process)
    deadline = started + args.wall_seconds
    reason = 'process_exit'
    try:
        while process.poll() is None:
            owned.refresh()
            if time.monotonic()-started >= args.wall_seconds-15:
                # Let torchrun forward SIGTERM to rank workers before the hard
                # deadline; some launchers give workers their own process groups.
                reason='wall_limit'; stop_group(process,owned,grace_seconds=max(0,deadline-time.monotonic())); break
            if args.stop_after_checkpoint is not None:
                marker=args.out/('checkpoint-%06d'%args.stop_after_checkpoint)/'complete.json'
                if marker.is_file():
                    reason='explicit_committed_checkpoint_stop'; break
            time.sleep(.5)
    except KeyboardInterrupt:
        reason='user_interrupt'
    finally:
        # Also reap observed descendants after normal/abnormal launcher exit.
        # A parent's exit is not evidence that its separate worker sessions died.
        stop_group(process,owned,grace_seconds=max(0,min(15,deadline-time.monotonic())))
    print(json.dumps({'launcher_termination':reason,'child_returncode':process.returncode,
                     'wall_seconds':time.monotonic()-started}),flush=True)
    return process.returncode if reason=='process_exit' else 75 if reason=='explicit_committed_checkpoint_stop' else 124 if reason=='wall_limit' else 130


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='mode',required=True)
    b=sub.add_parser('build',help='CPU-only preparation; never starts torchrun')
    b.add_argument('--name',required=True); b.add_argument('--data',type=Path,action='append',required=True)
    b.add_argument('--admission-manifest',type=Path,action='append',required=True,
                   help='Explicit producer admission manifest containing the supplied shard hashes')
    b.add_argument('--tools',type=Path,required=True)
    b.add_argument('--tokenizer',type=Path,default=REPO/'research/015/tokenizer-assets/official-31b-ct-v2-tokenizer')
    b.add_argument('--max-samples',type=int,default=16)
    r=sub.add_parser('run',help='Explicitly launch one bounded four-rank GPU calibration')
    r.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    r.add_argument('--bucket',choices=['16k','32k'],required=True)
    r.add_argument('--model',type=Path,required=True); r.add_argument('--tokenizer',type=Path,required=True)
    r.add_argument('--out',type=Path,required=True); r.add_argument('--wall-seconds',type=int,required=True)
    r.add_argument('--resume',type=Path); r.add_argument('--init-run',type=Path)
    r.add_argument('--stop-after-checkpoint',type=int)
    args=parser.parse_args()
    return run(args) if args.mode=='run' else build(args) or 0


if __name__=='__main__': sys.exit(main())
