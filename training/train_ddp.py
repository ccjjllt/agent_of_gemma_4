#!/usr/bin/env python3
"""Rank-local Gemma4 LoRA training with DDP; frozen NF4 bases are replicated.

Launch with torchrun --standalone --nproc-per-node=4 -m training.train_ddp.
This is data parallel training, not weight sharding. Each rank must fit its base
and activations. Global selected-token normalization includes uneven last batches
without repeating examples in the training objective. No packages are installed.
"""
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from . import train_sft as single
from .common import TrainingError, atomic_json, load_config, load_encoded_jsonl
from .export_peft import inspect_adapter


def trainable_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run_ddp_loop(model, rows, cfg, output, torch, pad_token_id, signature, resume=None):
    dist = torch.distributed
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(cfg['device'])
    output = Path(output)
    # All ranks loaded the same immutable source and deterministic adapters.
    # Avoid broadcasting the ~17 GiB frozen base during DDP construction.
    hashes = [None] * world
    dist.all_gather_object(hashes, trainable_hash(model))
    if len(set(hashes)) != 1:
        raise TrainingError('DDP adapter initialization differs across ranks')
    ddp = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device.index] if device.type == 'cuda' else None,
        broadcast_buffers=False, find_unused_parameters=False,
        gradient_as_bucket_view=True, init_sync=False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    global_micro = cfg['micro_batch_size'] * world
    batches_per_epoch = math.ceil(len(rows) / global_micro)
    per_epoch = math.ceil(batches_per_epoch / cfg['gradient_accumulation_steps'])
    total_steps = per_epoch * cfg['epochs']
    if cfg['max_steps']:
        total_steps = min(total_steps, cfg['max_steps'])
    warmup = int(total_steps * cfg['warmup_ratio'])
    def schedule(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        return max(0., (total_steps-step)/max(1,total_steps-warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda' and cfg['compute_dtype']=='float16')
    epoch_start, batch_start, step_no = 0, 0, 0
    if resume:
        state = torch.load(Path(resume)/'training_state.pt', map_location='cpu', weights_only=True)
        if state['signature'] != signature or state['world_size'] != world:
            raise TrainingError('DDP resume requires identical config, data and world size')
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        rng = torch.load(Path(resume)/('rank-%03d-rng.pt'%rank), map_location='cpu', weights_only=True)
        torch.set_rng_state(rng['torch_rng'])
        if device.type=='cuda': torch.cuda.set_rng_state(rng['cuda_rng'], device)
        epoch_start, batch_start, step_no = state['epoch'],state['next_batch'],state['global_step']
    if step_no >= total_steps:
        raise TrainingError('DDP checkpoint already reached the configured step limit')
    collate = single.SelectedLabelCollator(pad_token_id, torch)
    def checkpoint(epoch, next_batch):
        dest=output/('checkpoint-%06d'%step_no)
        if rank==0:
            if dest.exists(): raise TrainingError('DDP checkpoint exists: '+str(dest))
            dest.mkdir()
            model.save_pretrained(dest/'adapter',safe_serialization=True)
            torch.save({'signature':signature,'world_size':world,'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,
                'next_batch':next_batch,'global_step':step_no},dest/'training_state.pt')
        dist.barrier()
        torch.save({'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state(device) if device.type=='cuda' else None},
            dest/('rank-%03d-rng.pt'%rank))
        dist.barrier()
        if rank==0: atomic_json(dest/'progress.json',{'world_size':world,'global_step':step_no,
                                                       'epoch':epoch,'next_batch':next_batch,'signature':signature})
        return dest
    model.train()
    started=time.monotonic()
    input_total, selected_total, last_loss, last_checkpoint = 0,0,None,None
    for epoch in range(epoch_start,cfg['epochs']):
        order=torch.randperm(len(rows),generator=torch.Generator().manual_seed(cfg['seed']+epoch)).tolist()
        batches=[order[i:i+global_micro] for i in range(0,len(order),global_micro)]
        offset=batch_start if epoch==epoch_start else 0
        if offset%cfg['gradient_accumulation_steps'] and offset!=len(batches):
            raise TrainingError('DDP resume is not at an optimizer boundary')
        for start in range(offset,len(batches),cfg['gradient_accumulation_steps']):
            group=batches[start:start+cfg['gradient_accumulation_steps']]
            denominator=sum(sum(x!=-100 for x in rows[i]['labels'][1:]) for batch in group for i in batch)
            if not denominator: raise TrainingError('Global batch has no selected next-token labels')
            optimizer.zero_grad(set_to_none=True)
            loss_contribution=torch.zeros((),device=device,dtype=torch.float64)
            step_started=time.monotonic()
            for micro_index,indices in enumerate(group):
                local=indices[rank*cfg['micro_batch_size']:(rank+1)*cfg['micro_batch_size']]
                count=sum(sum(x!=-100 for x in rows[i]['labels'][1:]) for i in local)
                # An empty rank still joins DDP collectives. Its duplicated row
                # has zero weight, so the global objective contains no padding.
                values=collate([rows[i] for i in local] if local else [rows[indices[0]]])
                batch={k:v.to(device) for k,v in values.items()}
                sync=ddp.no_sync() if micro_index<len(group)-1 else nullcontext()
                dtype=getattr(torch,cfg['compute_dtype'])
                with sync:
                    amp=torch.autocast(device_type=device.type,dtype=dtype) if dtype!=torch.float32 else nullcontext()
                    with amp: loss=ddp(**batch,use_cache=False).loss
                    if loss is None or not torch.isfinite(loss).all(): raise TrainingError('DDP non-finite loss')
                    # DDP averages rank gradients: undo that average, then
                    # normalize by selected tokens over the whole global window.
                    scaler.scale(loss*(world*count/denominator)).backward()
                loss_contribution.add_(loss.detach().double()*(count/denominator))
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(params,cfg['max_grad_norm'],error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();scheduler.step();step_no+=1
            dist.all_reduce(loss_contribution,op=dist.ReduceOp.SUM)
            last_loss=float(loss_contribution)
            tokens=sum(sum(rows[i]['attention_mask']) for batch in group for i in batch)
            input_total+=tokens;selected_total+=denominator
            if device.type=='cuda': torch.cuda.synchronize(device)
            seconds=time.monotonic()-step_started
            if rank==0:
                metric={'step':step_no,'epoch':epoch,'loss':last_loss,'supervised_tokens':denominator,
                    'input_tokens':tokens,'world_size':world,'seconds':seconds,
                    'selected_tokens_per_second':denominator/seconds,'grad_norm':float(norm),
                    'learning_rate':optimizer.param_groups[0]['lr']}
                with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metric)+'\n')
                print(json.dumps(metric),flush=True)
            following=start+len(group)
            next_epoch,next_batch=(epoch+1,0) if following==len(batches) else (epoch,following)
            if step_no%cfg['checkpoint_every_n_steps']==0 or step_no==total_steps:
                last_checkpoint=checkpoint(next_epoch,next_batch)
            if step_no>=total_steps:break
        if step_no>=total_steps:break
    dist.barrier()
    if rank==0:model.save_pretrained(output/'adapter',safe_serialization=True)
    dist.barrier()
    return {'optimizer_steps':step_no,'loss':last_loss,'global_input_tokens_this_run':input_total,
        'global_selected_tokens_this_run':selected_total,'wall_seconds_this_run':time.monotonic()-started,
        'world_size':world,'last_checkpoint':str(last_checkpoint),'trainable_sha256':trainable_hash(model)}


def main(argv=None):
    args=single.build_parser().parse_args(argv)
    dist=None
    try:
        raw=load_config(args.config)
        for attr,key in [('model','model_name_or_path'),('revision','model_revision'),('tokenizer','tokenizer_name_or_path'),
            ('tokenizer_revision','tokenizer_revision'),('tools','tool_schema_path'),('chat_template','chat_template_path'),
            ('device','device'),('max_steps','max_steps')]:
            if getattr(args,attr) is not None:raw[key]=getattr(args,attr)
        if args.allow_download:raw['local_files_only']=False
        cfg=single.validate_config(raw)
        rows,data=load_encoded_jsonl(args.data,cfg['max_sequence_length']) if args.data else (None,None)
        if args.dry_run:
            print(json.dumps({'dry_run':True,'mode':'ddp','config':cfg,'data':data,'weights_loaded':False},indent=2));return 0
        if not args.data or not args.out:raise TrainingError('DDP requires --data and --out')
        if data['provenance']['protocol_id']!=cfg['protocol_id']:raise TrainingError('Data protocol differs from config')
        stack=single.require_stack(cfg['quant_backend']);torch=stack['torch'];dist=torch.distributed
        if int(os.environ.get('WORLD_SIZE','1'))<2:raise TrainingError('Use torchrun with at least two ranks')
        local_rank=int(os.environ['LOCAL_RANK'])
        signature=single._signature(cfg,data)
        if cfg['device'].startswith('cuda'):
            torch.cuda.set_device(local_rank);cfg['device']='cuda:%d'%local_rank;backend='nccl'
        elif cfg['device']=='cpu':backend='gloo'
        else:raise TrainingError('DDP supports CUDA or CPU only')
        dist.init_process_group(backend=backend,timeout=timedelta(seconds=1800))
        rank=dist.get_rank();output=Path(args.out).resolve()
        if args.resume:
            if Path(args.resume).resolve().parent!=output or not (Path(args.resume)/'progress.json').is_file():
                raise TrainingError('DDP resume requires a complete checkpoint under --out')
        elif output.exists() and any(output.iterdir()):raise TrainingError('DDP output must be empty')
        random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
        if backend=='nccl':torch.cuda.manual_seed_all(cfg['seed'])
        print(json.dumps({'rank':rank,'stage':'model_load_begin','device':cfg['device']}),flush=True)
        t=time.monotonic()
        model,tok,device,info=single.load_model_and_tokenizer(cfg,stack,args.resume,expected_provenance=data['provenance'])
        print(json.dumps({'rank':rank,'stage':'model_load_ok','seconds':time.monotonic()-t,
            'allocated_bytes':torch.cuda.memory_allocated(device) if backend=='nccl' else None}),flush=True)
        if any(max(r['input_ids'])>=model.get_input_embeddings().num_embeddings for r in rows):
            raise TrainingError('Token id exceeds model vocabulary')
        if rank==0:
            output.mkdir(parents=True,exist_ok=True)
            atomic_json(output/'run_config.json',{'config':cfg,'data':data,'model':info,'signature':signature,'world_size':dist.get_world_size()})
        dist.barrier()
        if backend=='nccl':torch.cuda.reset_peak_memory_stats(device)
        result=run_ddp_loop(model,rows,cfg,output,torch,tok.pad_token_id,signature,args.resume)
        rank_result={**result,'rank':rank,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if backend=='nccl' else None}
        atomic_json(output/('rank-%03d-result.json'%rank),rank_result)
        if rank==0:
            atomic_json(output/'training_manifest.json',{'status':'training_completed','mode':'ddp','config':cfg,
                'data':data,'model':info,'metrics':result,'signature':signature,
                'adapter':inspect_adapter(output/'adapter',cfg['rank'],cfg['target_text_modules']),
                'runtime_compatibility_verified':False})
        dist.barrier();return 0
    except Exception as exc:
        print('ERROR: '+repr(exc),file=sys.stderr,flush=True);return 2
    finally:
        if dist is not None and dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':sys.exit(main())
