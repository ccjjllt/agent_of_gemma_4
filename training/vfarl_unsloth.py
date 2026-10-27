"""Unsloth execution arm for the existing exact-token VFARL collector.

This is an execution backend, not a replacement reward, trainer, quantization,
or chat format. The immutable QAT-derived NF4 cache is loaded as-is. Actual
loaded base tensors are compared with the serialized cache before sampling.
Unsloth and the historical native backend deliberately have different policy
identities; old behavior log probabilities must never be rewritten.

Primary API references:
https://unsloth.ai/docs/models/gemma-4/train
https://github.com/unslothai/unsloth/blob/main/unsloth/models/loader.py
https://github.com/unslothai/unsloth/blob/main/unsloth/models/vision.py
"""
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import re

from . import train_sft as single
from .common import TrainingError, sha256_file
from .encoding.gemma_tool_chat import tokenizer_fingerprint
from .train_ddp import trainable_hash
from .vfarl_policy import (NF4Policy, bind_template, digest, identities,
                          read_config, require_zero_lora, frozen_base_reference)


UNSLOTH_EXECUTION_PROFILE = {
    'backend': 'unsloth',
    'version': 1,
    'prompt_prefill': 'complete-prefix-contiguous-256-token-KV-chunks',
    'completion': 'one-token-cached-autoregressive-forward',
    'nf4_linear': 'Unsloth-loader-native-NF4-kernels-no-VFARL-fallback-override',
    'head': 'Unsloth-native-tied-head-no-VFARL-FP32-head-override',
    'logprob': 'actual-returned-logits-to-float32-log-softmax',
    'reference': 'independent-same-autoregressive-path-with-zero-LoRA-scaling',
    'forced_controls': 'preserve-in-prefix-and-mask-from-loss',
    'compute_dtype': 'bfloat16',
    'fast_inference': False,
    'quantization': 'same-prequantized-official-QAT-derived-NF4-cache',
    'adapter_initialization': 'copy-exact-native-PEFT-CPU-shadow-initialization',
}
NATIVE_SEED0_ADAPTER_SHA256 = '12b4e23d28fcc8ab2ea1581a9398edfd2105375e43f3fc795199f3d6b31b06dc'
CT_V2_TOKENIZER_SHA256 = 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f'
CT_V2_TEMPLATE_SHA256 = '94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413'


def _tensor_digest(tensor, torch, *, chunk_elements=4 * 1024 * 1024):
    """Bound host copies while comparing actual CUDA and on-disk tensors."""
    value = tensor.detach().reshape(-1)
    result = hashlib.sha256()
    for start in range(0, value.numel(), chunk_elements):
        piece = value[start:start + chunk_elements].contiguous().view(torch.uint8).cpu()
        result.update(piece.numpy().tobytes())
    return result.hexdigest()


def validate_loaded_cache(model, source, torch):
    """Compare actual base values and NF4 QuantState to the original archive.

    QuantState's packed JSON is compared semantically, allowing JSON key-order
    differences only. All numeric tensors, including packed NF4 bytes, are
    byte compared. The only permitted dtype conversion is a small RMSNorm
    vector's lossless BF16-to-FP32 cast. No weight is changed by this function.
    """
    from safetensors import safe_open
    source = Path(source)
    index_path = source / 'model.safetensors.index.json'
    if not index_path.is_file():
        raise TrainingError('Unsloth arm requires the original sharded NF4 cache index')
    weight_map = json.loads(index_path.read_text())['weight_map']
    actual = {}
    for name, tensor in model.state_dict().items():
        # PEFT wraps a frozen projection in ``base_layer`` without changing its
        # underlying value. Compare to the original pre-PEFT serialized key.
        canonical = name.replace('.base_layer.', '.')
        if canonical in actual:
            raise TrainingError('Ambiguous frozen base tensor after Unsloth/PEFT wrapping')
        actual[canonical] = tensor
    missing = set(weight_map) - set(actual)
    if missing:
        raise TrainingError('Unsloth changed serialized base keys: ' + repr(sorted(missing)[:8]))
    norm_keys = set()
    for name, module in model.named_modules():
        if type(module).__name__.endswith('RMSNorm') or isinstance(module, torch.nn.LayerNorm):
            norm_keys.update(name + '.' + key for key, _ in module.named_parameters(recurse=False))
    records, quantized, packed_metadata = [], 0, 0
    for filename in sorted(set(weight_map.values())):
        if Path(filename).name != filename:
            raise TrainingError('Invalid NF4 shard filename')
        with safe_open(str(source / filename), framework='pt', device='cpu') as shard:
            for name in sorted(key for key, value in weight_map.items() if value == filename):
                expected, loaded = shard.get_tensor(name), actual[name]
                if tuple(loaded.shape) != tuple(expected.shape):
                    raise TrainingError('Unsloth changed base tensor shape: ' + name)
                if name.endswith('.quant_state.bitsandbytes__nf4'):
                    def unpack(tensor):
                        return json.loads(bytes(tensor.detach().cpu().tolist()).decode('utf-8'))
                    if unpack(loaded) != unpack(expected):
                        raise TrainingError('Unsloth changed NF4 QuantState metadata: ' + name)
                    value_hash = digest(unpack(loaded))
                    packed_metadata += 1
                else:
                    if loaded.dtype != expected.dtype:
                        if (name not in norm_keys or loaded.numel() > 65536
                                or expected.dtype != torch.bfloat16 or loaded.dtype != torch.float32):
                            raise TrainingError('Unsloth changed frozen base dtype: ' + name)
                        expected = expected.to(torch.float32)
                    value_hash = _tensor_digest(loaded, torch)
                    if value_hash != _tensor_digest(expected, torch):
                        raise TrainingError('Unsloth changed frozen base or NF4 numeric bytes: ' + name)
                records.append((name, list(loaded.shape), str(loaded.dtype), value_hash))
                quantized += int(name.endswith('.weight') and
                    name + '.quant_state.bitsandbytes__nf4' in weight_map)
    del actual
    if not quantized or packed_metadata != quantized:
        raise TrainingError('No complete serialized NF4 weight/QuantState pairs were checked')
    return dict(serialized_tensors=len(records), nf4_weights=quantized,
        nf4_quantstates=packed_metadata, actual_base_values_sha256=digest(sorted(records)),
        source_index_sha256=sha256_file(index_path),
        validation='Actual loaded numeric bytes versus every serialized cache tensor; packed QuantState JSON semantic equality; only small RMSNorm BF16-to-FP32 casts allowed')


def native_cpu_adapter(model, targets, cfg, torch, peft):
    """Create native PEFT A/B using a storage-free copy of target topology.

    Each frozen shadow Linear has the real in/out shape but one shared storage
    element. It is never executed. Calling PEFT itself preserves constructor
    draws, reset draws, module ordering and dtype semantics instead of assuming
    Unsloth's random_state initializes an identical A matrix.
    """
    target_set = set(targets)
    ordered = [(name, module) for name, module in model.named_modules() if name in target_set]
    if len(ordered) != len(target_set):
        raise TrainingError('Unsloth arm target topology is incomplete')
    shadow = torch.nn.Module()
    for name, module in ordered:
        parent = shadow
        pieces = name.split('.')
        for part in pieces[:-1]:
            if part not in parent._modules:
                parent.add_module(part, torch.nn.Module())
            parent = parent._modules[part]
        with torch.device('meta'):
            linear = torch.nn.Linear(module.in_features, module.out_features, bias=False)
        # Base tensors are frozen, and their values are never read by PEFT's
        # plain LoRA initializer. Do not allocate the dense 31B shadow weights.
        storage = torch.empty(1, device='cpu', dtype=torch.float32)
        linear.weight = torch.nn.Parameter(storage.as_strided(
            (module.out_features, module.in_features), (0, 0)), requires_grad=False)
        parent.add_module(pieces[-1], linear)
    config = peft.LoraConfig(r=cfg['rank'], lora_alpha=cfg['alpha'],
        lora_dropout=cfg['dropout'], bias='none', target_modules=targets,
        task_type=None)
    random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    with torch.device('cpu'):
        shadow = peft.get_peft_model(shadow, config)
    return shadow


def copy_native_cpu_adapter(model, shadow, torch):
    expected = {name: value for name, value in shadow.named_parameters() if value.requires_grad}
    actual = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if set(expected) != set(actual) or not expected:
        raise TrainingError('Unsloth adapter parameter names differ from native PEFT initialization')
    casts = []
    with torch.no_grad():
        for name, value in expected.items():
            destination = actual[name]
            if destination.shape != value.shape:
                raise TrainingError('Unsloth adapter shape differs from native PEFT: ' + name)
            if destination.dtype != value.dtype:
                casts.append(dict(name=name, previous_dtype=str(destination.dtype),
                                  native_dtype=str(value.dtype)))
                destination.data = destination.data.to(value.dtype)
            destination.copy_(value.to(destination.device))
    wanted, observed = trainable_hash(shadow), trainable_hash(model)
    if wanted != observed:
        raise TrainingError('Unsloth adapter bytes/order differ from the actual native CPU initializer')
    return dict(boundary='Exact native PEFT CPU shadow topology; actual A/B copied by name',
        resumed_adapter=False, native_cpu_adapter_sha256=wanted,
        actual_adapter_sha256=observed, parameter_tensors=len(expected),
        adapter_dtype_casts=casts)


def _versions():
    result = {}
    for name in ('unsloth', 'unsloth_zoo', 'torch', 'transformers', 'peft',
                 'bitsandbytes', 'triton', 'accelerate'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise TrainingError('Unsloth runtime package missing: ' + name)
    return result


def validate_reference_scaling(model):
    """Confirm the installed fast-kernel accessor reads live LoRA scaling.

    No adapter is disabled or merged. Compiled forwards retain normal dynamo
    guards; actual reference forwards remain independent model computations.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer
    from unsloth.kernels.utils import get_lora_parameters
    layers = [module for module in model.modules() if isinstance(module, BaseTunerLayer)]
    if not layers:
        raise TrainingError('Unsloth backend has no plain LoRA layers for its frozen reference')
    before = [float(get_lora_parameters(layer)[4]) for layer in layers]
    with frozen_base_reference(model):
        zero = [float(get_lora_parameters(layer)[4]) for layer in layers]
    after = [float(get_lora_parameters(layer)[4]) for layer in layers]
    if any(value != 0 for value in zero) or after != before:
        raise TrainingError('Unsloth fast LoRA accessor cached or failed to restore reference scaling')
    return dict(layers=len(layers), active_scaling=sorted(set(before)),
        zero_scaling_readback=True, restored_scaling_readback=True,
        accessor_source_sha256=sha256_file(inspect.getfile(get_lora_parameters)))


class UnslothPolicy(NF4Policy):
    """Unsloth model execution with the original VFARL exact-token sampler."""

    def __init__(self, config, *, seed=None, adapter_run=None, device=None,
                 inference_device_count=1):
        if inference_device_count not in (1, 2) or type(inference_device_count) is not int:
            raise TrainingError('Unsloth policy supports one or two visible CUDA devices')
        self.cfg, self.rl = read_config(config)
        if device is not None:
            self.cfg['device'] = device
        if seed is not None and seed != self.cfg['seed']:
            raise TrainingError('Vary the sample seed, not the shared adapter initialization seed')
        # Import order is required by Unsloth's patching mechanism. The modules
        # imported at file scope above do not import transformers or torch.
        from unsloth import FastModel
        self.stack = single.require_stack('nf4')
        self.torch = torch = self.stack['torch']
        transformers, peft = self.stack['transformers'], self.stack['peft']
        if (not torch.cuda.is_available() or torch.version.hip is not None
                or not torch.cuda.is_bf16_supported()
                or torch.cuda.device_count() != inference_device_count
                or int(os.environ.get('WORLD_SIZE', '1')) != 1
                or torch.distributed.is_initialized()):
            raise TrainingError('Unsloth rollout requires standalone BF16 CUDA with the declared visible devices')
        operations = importlib.import_module('bitsandbytes.backends.cuda.ops')
        if getattr(operations, '_vfarl_uniform_probability_dispatch', None) is not None:
            raise TrainingError('Load the Unsloth backend in a fresh process, without native VFARL fallback patches')
        source, options = single.source_options(self.cfg['model_name_or_path'],
            self.cfg['model_revision'], self.cfg['local_files_only'])
        if not Path(source).is_dir():
            raise TrainingError('Unsloth arm only loads the locally staged immutable NF4 cache')
        original_config = transformers.AutoConfig.from_pretrained(source, **options)
        if getattr(original_config, 'model_type', None) != 'gemma4':
            raise TrainingError('This Unsloth arm requires the original full Gemma4 conditional-generation cache')
        cache_info = single.validate_nf4_cache(source, original_config, self.cfg)
        tok_source, tok_options = single.source_options(
            self.cfg['tokenizer_name_or_path'] or source,
            self.cfg['tokenizer_revision'] or self.cfg['model_revision'], True)
        placement = ({'device_map': 'balanced', 'max_memory': {0: '14GiB', 1: '14GiB'}}
                     if inference_device_count == 2 else {'device_map': {'': 0}})
        random.seed(self.cfg['seed']); torch.manual_seed(self.cfg['seed'])
        torch.cuda.manual_seed_all(self.cfg['seed'])
        self.model, _ = FastModel.from_pretrained(
            model_name=source, tokenizer_name=tok_source,
            max_seq_length=self.cfg['max_sequence_length'], dtype=torch.bfloat16,
            load_in_4bit=True, full_finetuning=False, fast_inference=False,
            use_exact_model_name=True, local_files_only=True,
            fix_tokenizer=False, trust_remote_code=False, offload_embedding=False,
            use_gradient_checkpointing=False, return_logits=True,
            attn_implementation=self.cfg['attn_implementation'], **placement)
        if type(self.model).__name__ != 'Gemma4ForConditionalGeneration':
            raise TrainingError('Unsloth changed the full cached model architecture')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tok_source, **tok_options)
        bind_template(self.tokenizer, self.cfg)
        if (tokenizer_fingerprint(self.tokenizer) != CT_V2_TOKENIZER_SHA256
                or hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest() != CT_V2_TEMPLATE_SHA256):
            raise TrainingError('Unsloth arm changed the pinned CT v2 tokenizer or chat template')
        self.model.requires_grad_(False)
        targets = single.resolve_target_modules(dict(self.model.named_modules()),
                                                self.cfg['target_text_modules'])
        if adapter_run is None:
            shadow = native_cpu_adapter(self.model, targets, self.cfg, torch, peft)
            if self.cfg['seed'] == 0 and trainable_hash(shadow) != NATIVE_SEED0_ADAPTER_SHA256:
                raise TrainingError('31B CPU shadow initialization differs from the actual native r32/alpha64 seed0 adapter; inspect target topology/order before sampling')
        random.seed(self.cfg['seed']); torch.manual_seed(self.cfg['seed'])
        # Unsloth's layer-family filter treats an explicit list as projection
        # leaf names. A regex of exact qualified paths bypasses that widening
        # and preserves precisely the native text projection set.
        target_pattern = '(?:' + '|'.join(re.escape(name) for name in targets) + ')'
        self.model = FastModel.get_peft_model(self.model,
            r=self.cfg['rank'], lora_alpha=self.cfg['alpha'],
            lora_dropout=0, bias='none', target_modules=target_pattern,
            finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            use_gradient_checkpointing=False, random_state=self.cfg['seed'])
        # The already attached set is exact; retain ordinary explicit PEFT
        # targets for compatible plain-adapter exports and reloads.
        self.model.peft_config['default'].target_modules = set(targets)
        if adapter_run is None:
            adapter_initialization = copy_native_cpu_adapter(self.model, shadow, torch)
            adapter_initialization['seed'] = self.cfg['seed']
            del shadow
            require_zero_lora(self.model, torch)
        else:
            from safetensors.torch import load_file
            from .export_peft import inspect_adapter
            adapter_path = Path(adapter_run) / 'adapter'
            inspect_adapter(adapter_path, self.cfg['rank'], self.cfg['target_text_modules'])
            weights = load_file(str(adapter_path / 'adapter_model.safetensors'), device='cpu')
            # Native VFARL exports FP32 trainable A/B. Do not silently round an
            # existing update into a backend's newly initialized BF16 adapter.
            for name, parameter in self.model.named_parameters():
                if 'lora_' in name and parameter.dtype != torch.float32:
                    parameter.data = parameter.data.float()
            result = peft.set_peft_model_state_dict(self.model, weights, adapter_name='default')
            unexpected = getattr(result, 'unexpected_keys', [])
            missing = [name for name in getattr(result, 'missing_keys', []) if 'lora_' in name]
            if unexpected or missing:
                raise TrainingError('Unsloth could not load the actual plain LoRA adapter')
            loaded = peft.get_peft_model_state_dict(self.model, adapter_name='default')
            if set(loaded) != set(weights) or any(
                    loaded[name].dtype != weights[name].dtype
                    or loaded[name].shape != weights[name].shape
                    or _tensor_digest(loaded[name], torch) != _tensor_digest(weights[name], torch)
                    for name in weights):
                raise TrainingError('Unsloth did not preserve every saved native LoRA tensor')
            adapter_initialization = dict(resumed_adapter=True,
                boundary='Actual saved plain PEFT adapter, loaded without merging into NF4 base',
                adapter_file_sha256=sha256_file(adapter_path / 'adapter_model.safetensors'),
                actual_adapter_sha256=trainable_hash(self.model),
                saved_parameter_tensors=len(weights), saved_tensors_byte_checked=True)
            del weights, loaded
        FastModel.for_inference(self.model)
        self.model.eval()
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_('lora_' in name)
        base = self.model.get_base_model()
        if getattr(base.lm_head, '_vfarl_probability_head', None) is not None:
            raise TrainingError('Native VFARL FP32-head override leaked into Unsloth execution')
        if base.lm_head.weight is not base.get_input_embeddings().weight:
            raise TrainingError('Unsloth changed the original tied vocabulary weights')
        if any(parameter.device.type != 'cuda' for parameter in self.model.parameters()):
            raise TrainingError('Unsloth arm refuses CPU/disk parameter offload')
        if inference_device_count == 2 and {parameter.device.index for parameter in self.model.parameters()} != {0, 1}:
            raise TrainingError('Unsloth did not actually place the model on both declared GPUs')
        # Check after adapter attachment/patching too: neither a backend loader
        # nor LoRA preparation may silently rewrite the frozen quantized base.
        actual_base = validate_loaded_cache(base, source, torch)
        reference_scaling = validate_reference_scaling(self.model)
        self.probability_profile = UNSLOTH_EXECUTION_PROFILE
        runtime_identity = dict(profile=self.probability_profile, packages=_versions(),
            model_type=type(base).__name__, base_values=actual_base,
            reference_scaling=reference_scaling,
            implementation_sha256=dict(
                unsloth_loader=sha256_file(inspect.getfile(FastModel)),
                bitsandbytes_cuda=sha256_file(operations.__file__)),
            math=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                allow_bf16_reduced_precision_reduction=bool(
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)))
        runtime = dict(identity=runtime_identity, identity_sha256=digest(runtime_identity))
        self.info = dict(source=source, model_type='gemma4',
            model_class=type(base).__name__, nf4_cache=cache_info,
            target_modules=targets, adapter_initialization=adapter_initialization,
            trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            probability_execution=runtime,
            inference_placement=dict(device_map={key: str(value) for key, value in
                getattr(base, 'hf_device_map', {}).items()}, requested=placement['device_map'],
                max_memory=placement.get('max_memory'), offload=False))
        self.device = self.model.get_input_embeddings().weight.device
        self.cuda_devices = [torch.device('cuda', i) for i in range(inference_device_count)]
        self.execution = dict(backend='unsloth', inference_device_count=inference_device_count,
            input_device=str(self.device), device_map=self.info['inference_placement']['device_map'],
            max_memory=placement.get('max_memory'), adapter_initialization=adapter_initialization,
            probability_execution_profile=self.probability_profile,
            probability_execution_sha256=digest(self.probability_profile),
            probability_runtime=runtime)
        self.identity = identities(self.model, self.tokenizer, self.info, self.cfg,
                                   profile=self.probability_profile)
        self.thought_open = self._special('<|channel>')
        self.thought_close = self._special('<channel|>')
        self.tool_close = self._special('<tool_call|>')
        self.turn_close = self._special('<turn|>')
