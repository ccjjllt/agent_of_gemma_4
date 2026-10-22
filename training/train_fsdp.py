#!/usr/bin/env python3
"""Bounded FSDP1 QLoRA experiment using the provenance-checked uint8 NF4 cache.

Launch: torchrun --standalone --nproc-per-node=4 -m training.train_fsdp ...
Each rank first loads the existing ~17 GiB cache, then shards packed BF16 byte
containers. This does not claim memory-efficient initialization. Step-boundary
checkpoints include rank-local optimizer/RNG state for fixed-topology resume.
"""
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import uuid

from . import train_sft as single
from .common import TrainingError, atomic_json, load_config, load_encoded_jsonl
from .export_peft import inspect_adapter
from .fsdp_qlora import pack_floating_nf4_storage, wrap_fsdp, save_adapter
from .fsdp_checkpoint import (batches_fingerprint, collective_call, load_checkpoint,
                              read_init_run, read_manifest, save_checkpoint, stage_signature)
from .train_ddp import length_grouped_batches, trainable_hash


def run_loop(wrapped, peft_model, rows, cfg, output, torch, pad_id, shapes,
             signature, resume=None):
    dist = torch.distributed
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(cfg['device'])
    optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],
                                 lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    global_micro = cfg['micro_batch_size'] * world
    batches_per_epoch = math.ceil(len(rows) / global_micro)
    total_steps = math.ceil(batches_per_epoch / cfg['gradient_accumulation_steps']) * cfg['epochs']
    if cfg['max_steps']:
        total_steps = min(total_steps, cfg['max_steps'])
    warmup = int(total_steps * cfg['warmup_ratio'])
    def schedule(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        return max(0., (total_steps-step)/max(1, total_steps-warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    def epoch_batches(epoch):
        if epoch == cfg['epochs']:
            return []
        order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(cfg['seed']+epoch)).tolist()
        return length_grouped_batches(rows, order, global_micro, cfg['seed']+epoch)
    collate = single.SelectedLabelCollator(pad_id, torch)
    wrapped.train()
    started = time.monotonic()
    step_no, input_total, selected_total, last_loss = 0, 0, 0, None
    epoch_start, batch_start, previous_wall = 0, 0, 0.
    last_checkpoint = None
    metrics_name = 'metrics.jsonl'
    if resume:
        progress = load_checkpoint(resume, wrapped, optimizer, scheduler, torch, device, signature)
        step_no = progress['global_step']
        epoch_start, batch_start = progress['epoch'], progress['next_batch']
        input_total, selected_total = progress['global_input_tokens'], progress['global_selected_tokens']
        last_loss, previous_wall = progress['loss'], progress['training_wall_seconds']
        def validate_cursor():
            if (progress['total_steps'] != total_steps or progress['data_rows'] != len(rows)
                    or not 0 <= epoch_start <= cfg['epochs']):
                raise TrainingError('FSDP resume schedule/data size mismatch')
            batches = epoch_batches(epoch_start)
            if (not 0 <= batch_start < max(1, len(batches))
                    or batch_start % cfg['gradient_accumulation_steps']
                    or progress['epoch_batches_sha256'] != batches_fingerprint(batches)):
                raise TrainingError('FSDP resume cursor/order is not the saved optimizer boundary')
            steps_per_epoch = math.ceil(batches_per_epoch / cfg['gradient_accumulation_steps'])
            if step_no != epoch_start * steps_per_epoch + batch_start // cfg['gradient_accumulation_steps']:
                raise TrainingError('FSDP global step disagrees with next-batch cursor')
            if min(input_total, selected_total, previous_wall) < 0 or step_no > total_steps:
                raise TrainingError('FSDP checkpoint has invalid counters or exceeds the step limit')
        collective_call(torch, validate_cursor)
        names = ['metrics-resume-%06d-%s.jsonl' % (step_no, uuid.uuid4().hex[:12]) if rank == 0 else None]
        dist.broadcast_object_list(names, src=0)
        metrics_name = names[0]
        last_checkpoint = Path(resume)
    start_input_total, start_selected_total = input_total, selected_total
    completed_at_resume = step_no == total_steps
    for epoch in range(epoch_start, cfg['epochs']):
        if step_no >= total_steps:
            break
        batches = epoch_batches(epoch)
        offset = batch_start if epoch == epoch_start else 0
        for start in range(offset, len(batches), cfg['gradient_accumulation_steps']):
            group = batches[start:start+cfg['gradient_accumulation_steps']]
            denominator = sum(sum(x != -100 for x in rows[i]['labels'][1:]) for batch in group for i in batch)
            if not denominator:
                raise TrainingError('Global batch has no selected next-token labels')
            optimizer.zero_grad(set_to_none=True)
            contribution = torch.zeros((), device=device, dtype=torch.float64)
            step_started = time.monotonic()
            for indices in group:
                local = indices[rank*cfg['micro_batch_size']:(rank+1)*cfg['micro_batch_size']]
                count = sum(sum(x != -100 for x in rows[i]['labels'][1:]) for i in local)
                values = collate([rows[i] for i in local] if local else [rows[indices[0]]])
                batch = {k: v.to(device) for k, v in values.items()}
                print(json.dumps({'rank': rank, 'step': step_no+1, 'phase': 'forward',
                                  'length': batch['input_ids'].shape[1],
                                  'selected_tokens_local': count,
                                  'allocated_bytes': torch.cuda.memory_allocated(device) if device.type == 'cuda' else None}), flush=True)
                # Synchronize every microbatch: no_sync retains FULL gradients
                # under FSDP and would erase much of the intended memory saving.
                dtype = getattr(torch, cfg['compute_dtype'])
                amp = torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()
                with amp:
                    loss = wrapped(**batch, use_cache=False).loss
                if loss is None or not bool(torch.isfinite(loss).all()):
                    raise TrainingError('FSDP non-finite loss')
                print(json.dumps({'rank': rank, 'step': step_no+1, 'phase': 'backward',
                                  'allocated_bytes': torch.cuda.memory_allocated(device) if device.type == 'cuda' else None}), flush=True)
                (loss * (world * count / denominator)).backward()
                contribution.add_(loss.detach().double() * (count / denominator))
            norm = wrapped.clip_grad_norm_(cfg['max_grad_norm'])
            if not bool(torch.isfinite(norm).all()):
                raise TrainingError('FSDP non-finite gradient norm')
            learning_rate = optimizer.param_groups[0]['lr']
            optimizer.step(); scheduler.step(); step_no += 1
            dist.all_reduce(contribution, op=dist.ReduceOp.SUM)
            last_loss = float(contribution)
            input_tokens = sum(sum(rows[i]['attention_mask']) for indices in group for i in indices)
            input_total += input_tokens; selected_total += denominator
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - step_started
            if rank == 0:
                metric = {'step': step_no, 'epoch': epoch, 'loss': last_loss,
                          'supervised_tokens': denominator, 'input_tokens': input_tokens,
                          'world_size': world, 'seconds': elapsed,
                          'selected_tokens_per_second': denominator/elapsed,
                          'grad_norm': float(norm), 'learning_rate_used': learning_rate,
                          'max_sequence_length_by_rank': [max(len(rows[i]['input_ids'])
                              for indices in group for i in
                              (indices[r*cfg['micro_batch_size']:(r+1)*cfg['micro_batch_size']] or [indices[0]]))
                              for r in range(world)]}
                with (output/metrics_name).open('a') as handle:
                    handle.write(json.dumps(metric)+'\n')
                print(json.dumps(metric), flush=True)
            if step_no % cfg['checkpoint_every_n_steps'] == 0 or step_no == total_steps:
                following = start + len(group)
                next_epoch, next_batch = (epoch + 1, 0) if following == len(batches) else (epoch, following)
                # All ranks commit the same logical progress; their wall clocks
                # differ and must not produce disagreeing rank payloads.
                progress = [{'global_step': step_no, 'epoch': next_epoch, 'next_batch': next_batch,
                    'epoch_batches_sha256': batches_fingerprint(epoch_batches(next_epoch)),
                    'total_steps': total_steps, 'data_rows': len(rows), 'loss': last_loss,
                    'global_input_tokens': input_total, 'global_selected_tokens': selected_total,
                    'training_wall_seconds': previous_wall + time.monotonic() - started} if rank == 0 else None]
                dist.broadcast_object_list(progress, src=0)
                last_checkpoint = save_checkpoint(wrapped, peft_model, optimizer, scheduler,
                    output, torch, device, shapes, signature, progress[0])
            if step_no >= total_steps:
                break
        if step_no >= total_steps:
            break
    if completed_at_resume:
        # A crash may happen after the final checkpoint was committed but before
        # final export/manifest. No dummy forward/update is needed to recover it.
        def recover_final_adapter():
            if rank == 0:
                (output/'adapter').mkdir(exist_ok=True)
                for name in ('adapter_config.json', 'adapter_model.safetensors'):
                    shutil.copyfile(Path(resume)/'adapter'/name, output/'adapter'/name)
        collective_call(torch, recover_final_adapter)
    else:
        save_adapter(wrapped, peft_model, output/'adapter', torch, shapes)
    return {'optimizer_steps': step_no, 'loss': last_loss,
            'global_input_tokens': input_total, 'global_selected_tokens': selected_total,
            'global_input_tokens_this_run': input_total-start_input_total,
            'global_selected_tokens_this_run': selected_total-start_selected_total,
            'wall_seconds': time.monotonic()-started,
            'wall_seconds_this_run': time.monotonic()-started,
            'training_wall_seconds': previous_wall+time.monotonic()-started,
            'world_size': world, 'last_checkpoint': str(last_checkpoint),
            'metrics_file_this_run': None if completed_at_resume else metrics_name,
            'resume_from': str(resume) if resume else None,
            'export_only_recovery': completed_at_resume}


def main(argv=None):
    parser = single.build_parser()
    parser.add_argument('--init-run', help='completed FSDP run providing adapter weights for a NEW phase; '
                        'fresh optimizer/scheduler/data cursor; mutually exclusive with --resume')
    args = parser.parse_args(argv)
    dist = None
    try:
        if args.init_run and args.resume:
            raise TrainingError('--init-run starts a new phase; it cannot be combined with --resume')
        raw = load_config(args.config)
        for attr, key in [('model','model_name_or_path'), ('revision','model_revision'),
                          ('tokenizer','tokenizer_name_or_path'), ('tokenizer_revision','tokenizer_revision'),
                          ('tools','tool_schema_path'), ('chat_template','chat_template_path'),
                          ('device','device'), ('max_steps','max_steps')]:
            if getattr(args, attr) is not None:
                raw[key] = getattr(args, attr)
        if args.allow_download:
            raise TrainingError('P015 FSDP requires the existing offline NF4 cache')
        cfg = single.validate_config(raw)
        if (cfg['quant_backend'] != 'nf4' or not cfg['allow_cached_nf4'] or not cfg['local_files_only']
                or cfg['compute_dtype'] != 'bfloat16' or cfg['kbit_prepare'] != 'norms_fp32'
                or cfg['loss_backend'] != 'selected_chunked'):
            raise TrainingError('FSDP requires offline cached NF4, BF16, norms_fp32 and selected_chunked loss')
        rows, data = load_encoded_jsonl(args.data, cfg['max_sequence_length']) if args.data else (None, None)
        if data and data['provenance']['protocol_id'] != cfg['protocol_id']:
            raise TrainingError('Data protocol differs from config')
        init_run = Path(args.init_run).resolve() if args.init_run else None
        init_from = read_init_run(init_run, cfg) if init_run else None
        if args.dry_run:
            print(json.dumps({'dry_run': True, 'mode': 'fsdp_qlora', 'config': cfg,
                              'data': data, 'init_from': init_from,
                              'current_base_identity_requires_model_load': bool(init_run),
                              'weights_loaded': False}, indent=2)); return 0
        if not args.data or not args.out:
            raise TrainingError('FSDP requires --data and --out')
        stack = single.require_stack(cfg['quant_backend']); torch = stack['torch']; dist = torch.distributed
        if int(os.environ.get('WORLD_SIZE', '1')) < 2:
            raise TrainingError('Use torchrun with at least two CUDA ranks')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank); cfg['device'] = 'cuda:%d' % local_rank
        dist.init_process_group(backend='nccl', timeout=timedelta(seconds=900))
        rank = dist.get_rank(); output = Path(args.out).resolve()
        resume = Path(args.resume).resolve() if args.resume else None
        def validate_output():
            if resume:
                if resume.parent != output:
                    raise TrainingError('FSDP --resume must name a committed checkpoint inside --out')
                manifest = read_manifest(resume, torch, torch.device(cfg['device']))
                saved_step = manifest['progress']['global_step']
                for checkpoint in output.glob('checkpoint-*'):
                    if (checkpoint/'complete.json').is_file():
                        later = json.loads((checkpoint/'complete.json').read_text())
                        if later['progress']['global_step'] > saved_step:
                            raise TrainingError('Resume the latest committed checkpoint; later states already exist')
            elif output.exists() and any(output.iterdir()):
                raise TrainingError('FSDP output must be empty unless using --resume')
        collective_call(torch, validate_output)
        if resume:
            init_from = collective_call(torch, lambda: json.loads(
                (output/'run_config.json').read_text()).get('init_from'))
        random.seed(cfg['seed']); torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
        started = time.monotonic()
        print(json.dumps({'rank': rank, 'stage': 'model_load_begin'}), flush=True)
        model, tok, device, info = single.load_model_and_tokenizer(
            cfg, stack, resume or init_run, expected_provenance=data['provenance'])
        if info['nf4_cache'] is None:
            raise TrainingError('FSDP requires validated cache metadata')
        if init_run:
            def validate_initial_adapter():
                if read_init_run(init_run, cfg, info) != init_from:
                    raise TrainingError('Initial run changed while loading its adapter')
            collective_call(torch, validate_initial_adapter)
        if any(max(row['input_ids']) >= model.get_input_embeddings().num_embeddings for row in rows):
            raise TrainingError('Token id exceeds vocabulary')
        shapes = {name: list(p.shape) for name, p in model.named_parameters() if p.requires_grad}
        hashes = [None] * dist.get_world_size()
        dist.all_gather_object(hashes, trainable_hash(model))
        if len(set(hashes)) != 1:
            raise TrainingError('Adapter initialization differs across ranks')
        storage = pack_floating_nf4_storage(model, torch, stack['bitsandbytes'], torch.bfloat16)
        wrapped, units = wrap_fsdp(model, torch, device)
        signature_cfg = dict(cfg, device='cuda')
        signature = stage_signature(single._signature(signature_cfg, data, info), init_from)
        if resume:
            collective_call(torch, lambda: read_manifest(resume, torch, device, signature))
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            config_path = output/('resume-config-%s.json' % uuid.uuid4().hex[:12] if resume else 'run_config.json')
            atomic_json(config_path, {'mode': 'fsdp_qlora', 'config': cfg, 'data': data,
                'model': info, 'signature': signature, 'storage': storage,
                'wrap_units_rank0': units, 'adapter_shapes': shapes,
                'mixed_precision': {'param_dtype': None, 'reduce_dtype': 'float32', 'autocast': 'bfloat16'},
                'initialization': 'replicated rank-local cache load before sharding',
                'checkpoint_kind': 'adapter plus rank-local optimizer/scheduler/RNG; fixed topology and versions',
                'resume_from': str(resume) if resume else None, 'init_from': init_from})
        dist.barrier()
        print(json.dumps({'rank': rank, 'stage': 'fsdp_wrapped', 'seconds': time.monotonic()-started,
                          'allocated_bytes': torch.cuda.memory_allocated(device)}), flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        metrics = run_loop(wrapped, model, rows, cfg, output, torch, tok.pad_token_id, shapes,
                           signature, resume)
        atomic_json(output/('rank-%03d-result.json' % rank), {**metrics, 'rank': rank,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device)})
        if rank == 0:
            atomic_json(output/'training_manifest.json', {'status': 'training_completed',
                'mode': 'fsdp_qlora', 'config': cfg, 'data': data, 'model': info,
                'signature': signature, 'metrics': metrics, 'init_from': init_from,
                'versions': {key: getattr(mod, '__version__', 'unknown') for key, mod in stack.items()},
                'adapter': inspect_adapter(output/'adapter', cfg['rank'], cfg['target_text_modules']),
                'runtime_compatibility_verified': False})
        dist.barrier(); return 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print('ERROR: '+repr(exc), file=sys.stderr, flush=True); return 2
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    sys.exit(main())
