"""Step-boundary checkpoints for the fixed-topology FSDP1 LoRA trainer.

The optimizer owns rank-local FlatParameters. Its native state is saved on
every rank and is only restored with identical parameter order, shard layout,
world size and runtime versions. This deliberately does not reshard optimizer
states or serialize the immutable NF4 base. LoRA weights use the existing
independent PEFT export and are loaded before constructing FSDP on restart.
"""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import tempfile

from .common import TrainingError, sha256_file
from .fsdp_qlora import save_adapter


FORMAT = 'p016-fsdp1-local-optimizer-v1'
SAMPLER = 'length-grouped-batches-v1'


def runtime_versions(torch):
    versions = {'python': platform.python_version(), 'torch': str(torch.__version__),
                'cuda_runtime': torch.version.cuda}
    for name in ('transformers', 'peft', 'bitsandbytes', 'safetensors', 'accelerate'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def batches_fingerprint(batches):
    return hashlib.sha256(json.dumps(batches, separators=(',', ':')).encode()).hexdigest()


def collective_call(torch, operation):
    """Propagate ordinary I/O/validation errors before the next collective."""
    result, error = None, None
    try:
        result = operation()
    except Exception as exc:
        error = '%s: %s' % (type(exc).__name__, exc)
    errors = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(errors, error)
    if any(errors):
        raise TrainingError('FSDP checkpoint: ' + '; '.join(
            'rank %d %s' % (rank, value) for rank, value in enumerate(errors) if value))
    return result


def optimizer_layout(wrapped, optimizer):
    """Bind positional optimizer IDs to actual names and FSDP shard metadata."""
    names = {id(parameter): name for name, parameter in wrapped.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        entries = []
        for parameter in group['params']:
            if id(parameter) not in names or not parameter.requires_grad:
                raise TrainingError('Optimizer contains an unknown/frozen parameter')
            if not hasattr(parameter, '_fqns'):
                raise TrainingError('Resume requires FSDP1 flattened trainable parameters')
            entries.append({'name': names[id(parameter)], 'shape': list(parameter.shape),
                'dtype': str(parameter.dtype), 'fqns': list(parameter._fqns),
                'full_shapes': [list(shape) for shape in parameter._shapes],
                'full_numels': list(parameter._numels),
                'shards': [list(info) for info in parameter._shard_param_infos]})
        groups.append(entries)
    return groups


def capture_rng(torch, device):
    result = {'python': random.getstate(), 'torch': torch.get_rng_state(),
              'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
              'numpy': None}
    try:
        import numpy as np
        state = np.random.get_state()
        result['numpy'] = (state[0], state[1].tolist(), state[2], state[3], state[4])
    except ImportError:
        pass
    return result


def restore_rng(state, torch, device):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        if state['cuda'] is None:
            raise TrainingError('CUDA checkpoint is missing this rank RNG')
        torch.cuda.set_rng_state(state['cuda'], device)
    elif state['cuda'] is not None:
        raise TrainingError('CUDA RNG cannot be restored by the CPU runner')
    if state['numpy'] is not None:
        import numpy as np
        value = state['numpy']
        np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))


def _write_json(path, value):
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_manifest(resume, torch, device, signature=None):
    """Validate completion and local payloads before loading adapter/optimizer."""
    resume = Path(resume)
    if resume.name.startswith('.') or not (resume/'complete.json').is_file():
        raise TrainingError('Resume requires a committed FSDP checkpoint, not an adapter-only/partial directory')
    manifest = json.loads((resume/'complete.json').read_text())
    if manifest.get('format') != FORMAT or manifest.get('sampler') != SAMPLER:
        raise TrainingError('Unsupported FSDP checkpoint/sampler format')
    if manifest.get('world_size') != torch.distributed.get_world_size():
        raise TrainingError('FSDP local optimizer resume requires identical world size')
    if manifest.get('versions') != runtime_versions(torch):
        raise TrainingError('FSDP local optimizer resume requires identical runtime versions')
    if manifest.get('device_type') != device.type or manifest.get('backend') != str(torch.distributed.get_backend()):
        raise TrainingError('FSDP resume requires the original device type and distributed backend')
    if signature is not None and manifest.get('signature') != signature:
        raise TrainingError('Resume base/config/data/protocol fingerprint mismatch')
    rank_file = 'rank-%03d-state.pt' % torch.distributed.get_rank()
    required = ['adapter/adapter_config.json', 'adapter/adapter_model.safetensors', rank_file]
    for relative in required:
        path = resume/relative
        recorded = manifest.get('files', {}).get(relative)
        if (not recorded or not path.is_file() or path.stat().st_size != recorded['bytes']
                or sha256_file(path) != recorded['sha256']):
            raise TrainingError('Incomplete/corrupt FSDP checkpoint payload: ' + relative)
    return manifest


def save_checkpoint(wrapped, peft_model, optimizer, scheduler, output, torch, device,
                    shapes, signature, progress):
    """All ranks participate. A directory becomes visible only after all writes."""
    dist = torch.distributed
    rank, world = dist.get_rank(), dist.get_world_size()
    output = Path(output)
    destination = output/('checkpoint-%06d' % progress['global_step'])
    rng = capture_rng(torch, device)
    temporary = [None]
    def prepare():
        if rank == 0:
            if destination.exists():
                raise TrainingError('Checkpoint already exists: ' + str(destination))
            temporary[0] = tempfile.mkdtemp(prefix='.'+destination.name+'-incomplete-', dir=output)
    collective_call(torch, prepare)
    dist.broadcast_object_list(temporary, src=0)
    staging = Path(temporary[0])
    save_adapter(wrapped, peft_model, staging/'adapter', torch, shapes)
    def save_local():
        state = {'format': FORMAT, 'rank': rank, 'world_size': world, 'signature': signature,
                 'layout': optimizer_layout(wrapped, optimizer), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'rng': rng, 'progress': progress}
        with (staging/('rank-%03d-state.pt' % rank)).open('wb') as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
    collective_call(torch, save_local)
    def commit():
        if rank != 0:
            return
        files = {}
        for relative in ['adapter/adapter_config.json', 'adapter/adapter_model.safetensors'] + [
                'rank-%03d-state.pt' % index for index in range(world)]:
            path = staging/relative
            with path.open('rb') as handle:
                os.fsync(handle.fileno())
            files[relative] = {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
        manifest = {'format': FORMAT, 'sampler': SAMPLER, 'world_size': world,
                    'versions': runtime_versions(torch), 'signature': signature,
                    'device_type': device.type, 'backend': str(dist.get_backend()),
                    'progress': progress, 'files': files,
                    'base_weights': 'immutable external cache; identity bound by signature'}
        _write_json(staging/'complete.json', manifest)
        _fsync_dir(staging/'adapter')
        _fsync_dir(staging)
        os.replace(staging, destination)
        _fsync_dir(output)
    collective_call(torch, commit)
    # Checkpoint work must not alter the next stochastic forward on either path.
    restore_rng(rng, torch, device)
    return destination


def load_checkpoint(resume, wrapped, optimizer, scheduler, torch, device, signature):
    """The caller must already have loaded this checkpoint's PEFT adapter."""
    def load_local():
        manifest = read_manifest(resume, torch, device, signature)
        rank = torch.distributed.get_rank()
        state = torch.load(Path(resume)/('rank-%03d-state.pt' % rank),
                           map_location='cpu', weights_only=True)
        if (state.get('format') != FORMAT or state.get('rank') != rank
                or state.get('world_size') != manifest['world_size']
                or state.get('signature') != signature or state.get('progress') != manifest['progress']):
            raise TrainingError('FSDP rank state disagrees with committed manifest')
        if state.get('layout') != optimizer_layout(wrapped, optimizer):
            raise TrainingError('FSDP optimizer parameter order/shard layout mismatch')
        scheduler.load_state_dict(state['scheduler'])
        optimizer.load_state_dict(state['optimizer'])
        return state
    state = collective_call(torch, load_local)
    restore_rng(state['rng'], torch, device)
    return state['progress']


def stage_signature(base_signature, initialization):
    """Bind a new phase to its starting adapter; mount paths are not identity."""
    if initialization is None:
        return base_signature
    identity = {key: initialization[key] for key in
                ('training_manifest_sha256', 'source_signature', 'adapter_files')}
    return hashlib.sha256(json.dumps({'base_config_data_signature': base_signature,
        'initial_adapter': identity}, sort_keys=True).encode()).hexdigest()


def read_init_run(run, cfg, current_model=None):
    """Accept only a completed, provenance-matched FSDP run as new phase input.

    This loads no model or optimizer. The optional second validation compares
    the manifest to the current loader's independently validated NF4 identity.
    A diagnostic run may supply weights; completion never implies repair gain.
    """
    from . import train_sft as single
    from .export_peft import inspect_adapter
    run = Path(run).resolve()
    path = run/'training_manifest.json'
    if not path.is_file():
        raise TrainingError('--init-run requires a completed run with training_manifest.json')
    manifest = json.loads(path.read_text())
    if (manifest.get('status') != 'training_completed' or manifest.get('mode') != 'fsdp_qlora'
            or type(manifest.get('metrics', {}).get('optimizer_steps')) is not int
            or manifest['metrics']['optimizer_steps'] <= 0):
        raise TrainingError('--init-run requires actual completed FSDP updates and an independent adapter')
    previous = manifest['config']
    for key in ('base_family', 'quant_backend', 'rank', 'alpha', 'dropout', 'bias'):
        if previous.get(key) != cfg.get(key):
            raise TrainingError('Initial adapter base/LoRA configuration differs: ' + key)
    if (cfg.get('quant_backend') != 'nf4' or cfg.get('is_fallback_reference')
            or previous.get('is_fallback_reference')
            or set(previous['target_text_modules']) != set(cfg['target_text_modules'])):
        raise TrainingError('Initial adapter requires the same official NF4 base and LoRA module types')
    model = manifest['model']
    cache = model.get('nf4_cache') or {}
    keys = ('source', 'manifest_sha256', 'config_sha256', 'source_config_sha256',
            'source_index_sha256', 'source_weights_sha256', 'source_weights_bytes')
    identity = {key: cache.get(key) for key in keys}
    if (not str(identity['source']).startswith('google/gemma-4/')
            or any(not isinstance(identity[key], str) or not re.fullmatch(r'[0-9a-f]{64}', identity[key])
                   for key in ('manifest_sha256', 'config_sha256', 'source_config_sha256'))):
        raise TrainingError('Initial run is missing its validated official NF4 base identity')
    old_signature = single._signature(dict(previous, device='cuda'), manifest['data'], model)
    old_signature = stage_signature(old_signature, manifest.get('init_from'))
    if old_signature != manifest.get('signature'):
        raise TrainingError('Initial run config/data/base signature is inconsistent')
    adapter = inspect_adapter(run/'adapter', cfg['rank'], cfg['target_text_modules'])
    if adapter['files'] != manifest.get('adapter', {}).get('files'):
        raise TrainingError('Initial adapter bytes differ from the completed run manifest')
    acfg = adapter['adapter_config']
    if (acfg.get('lora_alpha') != cfg['alpha'] or acfg.get('lora_dropout') != cfg['dropout']
            or acfg.get('use_rslora', False) or acfg.get('lora_bias', False)):
        raise TrainingError('Initial adapter uses different LoRA scaling/dropout/bias semantics')
    targets = set(model['target_modules'])
    if (set(acfg['target_modules']) != targets
            or {name.removeprefix('base_model.model.') for name in adapter['modules']} != targets):
        raise TrainingError('Initial adapter does not cover exactly the recorded text modules')
    if current_model is not None:
        current_cache = current_model.get('nf4_cache') or {}
        if (identity != {key: current_cache.get(key) for key in keys}
                or model.get('model_type') != current_model.get('model_type')
                or model.get('model_class') != current_model.get('model_class')
                or targets != set(current_model['target_modules'])):
            raise TrainingError('Initial run official base/cache/architecture differs from the current verified loader')
    return {'kind': 'adapter_warm_start_new_phase', 'source_run': str(run),
            'training_manifest_sha256': sha256_file(path), 'source_signature': manifest['signature'],
            'adapter_files': adapter['files'], 'official_nf4_base': identity,
            'source_optimizer_steps': manifest['metrics']['optimizer_steps'],
            'source_data_sha256': manifest['data']['sha256'],
            'source_protocol_id': manifest['data']['provenance']['protocol_id'],
            'reset_state': ['optimizer', 'scheduler', 'RNG', 'global_step', 'data_cursor', 'token_counters'],
            'source_completion_is_not_repair_gain': True}
