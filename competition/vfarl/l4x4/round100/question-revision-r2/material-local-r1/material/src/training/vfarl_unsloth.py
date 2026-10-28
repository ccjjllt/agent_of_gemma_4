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
import copy
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import re
import time

from . import train_sft as single
from .common import TrainingError, sha256_file
from .encoding.gemma_tool_chat import tokenizer_fingerprint
from .train_ddp import trainable_hash
from .vfarl_policy import (NF4Policy, bind_template, digest, identities,
                          read_config, require_zero_lora, frozen_base_reference,
                          cached_prefill)


UNSLOTH_EXECUTION_PROFILE = {
    'backend': 'unsloth',
    'version': 2,
    'prompt_prefill': 'exact-prefix-append-only-KV-reuse-with-256-token-delta-chunks',
    'completion': 'one-token-cached-autoregressive-forward',
    'nf4_linear': 'runtime-resolved-bitsandbytes-or-Unsloth-dispatch-no-VFARL-fallback-override',
    'head': 'Unsloth-native-tied-head-no-VFARL-FP32-head-override',
    'logprob': 'actual-returned-logits-to-float32-log-softmax',
    'reference': 'independent-same-autoregressive-path-with-zero-LoRA-scaling',
    'forced_controls': 'preserve-in-prefix-and-mask-from-loss',
    'compute_dtype': 'bfloat16',
    'fast_inference': False,
    'quantization': 'same-prequantized-official-QAT-derived-NF4-cache',
    'adapter_initialization': 'copy-exact-native-PEFT-CPU-shadow-initialization',
    'cross_tool_kv': True,
    'cache_boundary': 'explicit-episode-reset-or-exact-token-prefix-mismatch',
    'teacher_forced_scoring': 'independent-fresh-full-prefix-cache-no-actor-cache-reuse',
}
NATIVE_SEED0_ADAPTER_SHA256 = '12b4e23d28fcc8ab2ea1581a9398edfd2105375e43f3fc795199f3d6b31b06dc'
CT_V2_TOKENIZER_FILE_SHA256 = 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f'
CT_V2_TOKENIZER_FINGERPRINT = 'd001718145f4980b65e70b5ffa1dfce93b3eca5a5d8de2446b8c483f2b4b8927'
CT_V2_TEMPLATE_SHA256 = '94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413'


def ct_tokenizer_identity(tokenizer, source):
    """Keep serialized file bytes separate from the shared behavior hash."""
    return dict(tokenizer_class=type(tokenizer).__name__,
        tokenizer_file_sha256=sha256_file(Path(source) / 'tokenizer.json'),
        tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
        template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest())


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
        # PEFT first moves fresh A/B to a bnb projection's compute_dtype,
        # then its model wrapper promotes trainable adapters back to FP32.
        # Preserve that BF16 rounding as well as the RNG draws: a plain FP32
        # shadow otherwise has different A values from the native NF4 loader.
        linear.compute_dtype = module.compute_dtype
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
    for name in ('unsloth', 'unsloth_zoo', 'torchao', 'xformers', 'datasets', 'trl', 'torch', 'transformers', 'peft',
                 'bitsandbytes', 'triton', 'accelerate'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise TrainingError('Unsloth runtime package missing: ' + name)
    return result


def resolved_nf4_dispatch(model):
    """Record installed Python dispatch, without claiming measured kernel speed."""
    paths = {}
    for module in model.modules():
        if type(module).__name__ != 'Linear4bit' or hasattr(module, 'base_layer'):
            continue
        forward = type(module).forward
        name = type(module).__module__ + '.' + type(module).__qualname__
        try:
            source = Path(inspect.getfile(forward))
            source_hash = sha256_file(source) if source.is_file() else None
        except (TypeError, OSError):
            source_hash = None
        paths[name] = dict(forward_module=forward.__module__,
            forward_qualname=forward.__qualname__, forward_source_sha256=source_hash,
            unsloth_nf4_override=bool(getattr(forward, '_unsloth_nf4_override', False)),
            unsloth_block_wrapper=bool(getattr(forward, '_unsloth_block_nf4', False)))
    operations = importlib.import_module('bitsandbytes.backends.cuda.ops')
    return dict(linear4bit_dispatch=paths,
        bnb_gemm_4bit_custom_max_m=getattr(operations, '_gemm_4bit_custom_max_m', None),
        interpretation='Installed Python forward identity; no kernel-level profiling or speedup claim')


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


def inspect_plain_adapter_for_config(adapter_path, cfg):
    """Validate saved scaling before copying tensors into live plain LoRA.

    Unlike PEFT.from_pretrained, this backend creates layers from the current
    config and then copies A/B only. Matching tensor bytes therefore does not
    establish matching alpha, dropout or adapter semantics by itself.
    """
    from .export_peft import inspect_adapter
    report = inspect_adapter(adapter_path, cfg['rank'], cfg['target_text_modules'])
    saved = report['adapter_config']
    expected = dict(r=cfg['rank'], lora_alpha=cfg['alpha'], lora_dropout=cfg['dropout'])
    observed = {key: saved.get(key, 0.0 if key == 'lora_dropout' else None) for key in expected}
    differences = [key for key, value in observed.items()
        if type(value) not in (int, float) or value != expected[key]]
    # FastModel attachment is ordinary bias-free LoRA. Do not reinterpret
    # tensors exported with a different effective model as execution drift.
    variants = [key for key in (
        'use_rslora', 'use_dora', 'use_qalora', 'lora_bias', 'fan_in_fan_out',
        'rank_pattern', 'alpha_pattern', 'modules_to_save', 'layer_replication',
        'alora_invocation_tokens', 'velora_config', 'monteclora_config',
        'use_bdlora', 'arrow_config', 'kasa_config', 'target_parameters',
        'trainable_token_indices') if saved.get(key)]
    if saved.get('bias', 'none') != 'none':
        variants.append('bias')
    if differences or variants:
        raise TrainingError('Saved adapter differs from the live plain LoRA configuration: '
            + ', '.join(differences + variants))
    return dict(**observed, use_rslora=False, bias='none', plain_lora=True,
                scaling=observed['lora_alpha'] / observed['r'])


class UnslothPolicy(NF4Policy):
    """Unsloth model execution with the original VFARL exact-token sampler."""

    def __init__(self, config, *, seed=None, adapter_run=None, device=None,
                 inference_device_count=1, training=False):
        if inference_device_count not in (1, 2) or type(inference_device_count) is not int:
            raise TrainingError('Unsloth policy supports one or two visible CUDA devices')
        if type(training) is not bool:
            raise TrainingError('Unsloth training mode must be explicit')
        self.cfg, self.rl = read_config(config)
        checkpointing = ('unsloth' if training and self.cfg['gradient_checkpointing'] else False)
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
        # Check the staged protocol before allocating the 31B model. The raw
        # tokenizer.json SHA is not tokenizer_fingerprint(), which also binds
        # the loaded backend, tokenizer class and special-token map.
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tok_source, **tok_options)
        bind_template(self.tokenizer, self.cfg)
        tokenizer_identity = ct_tokenizer_identity(self.tokenizer, tok_source)
        if (tokenizer_identity['tokenizer_file_sha256'] != CT_V2_TOKENIZER_FILE_SHA256
                or tokenizer_identity['tokenizer_fingerprint'] != CT_V2_TOKENIZER_FINGERPRINT
                or tokenizer_identity['template_sha256'] != CT_V2_TEMPLATE_SHA256):
            raise TrainingError('Pinned CT v2 tokenizer/template differs before model load: '
                                + json.dumps(tokenizer_identity, sort_keys=True))
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
            use_gradient_checkpointing=checkpointing, return_logits=True,
            attn_implementation=self.cfg['attn_implementation'], **placement)
        if type(self.model).__name__ != 'Gemma4ForConditionalGeneration':
            raise TrainingError('Unsloth changed the full cached model architecture')
        loaded_tokenizer_identity = ct_tokenizer_identity(self.tokenizer, tok_source)
        if loaded_tokenizer_identity != tokenizer_identity:
            raise TrainingError('Unsloth changed the pinned CT v2 tokenizer/template during model load: '
                                + json.dumps(loaded_tokenizer_identity, sort_keys=True))
        self.model.requires_grad_(False)
        targets = single.resolve_target_modules(dict(self.model.named_modules()),
                                                self.cfg['target_text_modules'])
        if adapter_run is None:
            shadow = native_cpu_adapter(self.model, targets, self.cfg, torch, peft)
            shadow_hash = trainable_hash(shadow)
            print(json.dumps(dict(phase='native_adapter_initialization',
                actual_adapter_sha256=shadow_hash,
                expected_adapter_sha256=NATIVE_SEED0_ADAPTER_SHA256,
                target_count=len(targets), default_dtype=str(torch.get_default_dtype()),
                projection_compute_dtypes=sorted({str(dict(self.model.named_modules())[name].compute_dtype)
                                                  for name in targets}))), flush=True)
            if self.cfg['seed'] == 0 and shadow_hash != NATIVE_SEED0_ADAPTER_SHA256:
                raise TrainingError('31B CPU shadow initialization differs from the actual native r32/alpha64 seed0 adapter: '
                                    + shadow_hash + '; inspect target topology/order/dtypes before sampling')
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
            use_gradient_checkpointing=checkpointing, random_state=self.cfg['seed'])
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
            adapter_path = Path(adapter_run) / 'adapter'
            saved_semantics = inspect_plain_adapter_for_config(adapter_path, self.cfg)
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
                saved_configuration_semantics=saved_semantics,
                adapter_file_sha256=sha256_file(adapter_path / 'adapter_model.safetensors'),
                actual_adapter_sha256=trainable_hash(self.model),
                saved_parameter_tensors=len(weights), saved_tensors_byte_checked=True)
            del weights, loaded
        if training:
            FastModel.for_training(self.model, use_gradient_checkpointing=checkpointing)
        else:
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
        self.probability_profile = (dict(
            backend='unsloth', version=1, mode='vfarl-focal-training',
            backbone='complete-untruncated-sequence-use_cache=False',
            logprob='selected-hidden-states-Unsloth-fused-linear-logprobs',
            head='same-tied-frozen-head-native-softcap-temperature1',
            reference='separate-full-forward-zero-live-LoRA-scaling',
            compute_dtype='bfloat16', trainable_dtype='float32',
            gradient_checkpointing=checkpointing, cross_tool_kv=False,
            quantization='same-prequantized-official-QAT-derived-NF4-cache')
            if training else UNSLOTH_EXECUTION_PROFILE)
        runtime_identity = dict(profile=self.probability_profile, packages=_versions(),
            model_type=type(base).__name__, base_values=actual_base,
            nf4_dispatch=resolved_nf4_dispatch(self.model),
            reference_scaling=reference_scaling,
            implementation_sha256=dict(
                unsloth_loader=sha256_file(inspect.getfile(FastModel)),
                bitsandbytes_cuda=sha256_file(operations.__file__)),
            math=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                allow_bf16_reduced_precision_reduction=bool(
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)))
        runtime = dict(identity=runtime_identity, identity_sha256=digest(runtime_identity))
        self._runtime_template = copy.deepcopy(runtime_identity)
        self._training_checkpointing = checkpointing
        self._training_profile = dict(
            backend='unsloth', version=1, mode='vfarl-focal-training',
            backbone='complete-untruncated-sequence-use_cache=False',
            logprob='selected-hidden-states-Unsloth-fused-linear-logprobs',
            head='same-tied-frozen-head-native-softcap-temperature1',
            reference='separate-full-forward-zero-live-LoRA-scaling',
            compute_dtype='bfloat16', trainable_dtype='float32',
            gradient_checkpointing=checkpointing, cross_tool_kv=False,
            quantization='same-prequantized-official-QAT-derived-NF4-cache')
        self.info = dict(source=source, model_type='gemma4',
            model_class=type(base).__name__, nf4_cache=cache_info,
            target_modules=targets, adapter_initialization=adapter_initialization,
            execution_mode='training' if training else 'inference',
            gradient_checkpointing=checkpointing,
            trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            probability_execution=runtime,
            inference_placement=dict(device_map={key: str(value) for key, value in
                getattr(base, 'hf_device_map', {}).items()}, requested=placement['device_map'],
                max_memory=placement.get('max_memory'), offload=False))
        self.device = self.model.get_input_embeddings().weight.device
        self.cuda_devices = [torch.device('cuda', i) for i in range(inference_device_count)]
        self.execution = dict(backend='unsloth', inference_device_count=inference_device_count,
            execution_mode='training' if training else 'inference',
            gradient_checkpointing=checkpointing,
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

    def switch_execution_mode(self, mode):
        """Switch a resident model without changing base or adapter tensors."""
        if mode not in ('inference', 'training'):
            raise TrainingError('Unknown resident policy execution mode')
        from unsloth import FastModel
        self.clear_generation_cache(reason='execution-mode-change')
        if mode == 'training':
            if self.cfg['gradient_checkpointing'] and not self._training_checkpointing:
                raise TrainingError('Resident training checkpointing must be installed during loading')
            FastModel.for_training(self.model, use_gradient_checkpointing=self._training_checkpointing)
            profile = self._training_profile
        else:
            FastModel.for_inference(self.model)
            self.model.eval()
            profile = getattr(self, '_inference_profile', UNSLOTH_EXECUTION_PROFILE)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_('lora_' in name)
        # The learner adds its fused-head implementation hash. Rebuild the
        # actor runtime from its loaded, weight-independent template so both
        # workers bind the same inference identity after synchronization.
        runtime_identity = copy.deepcopy(self._runtime_template)
        runtime_identity['profile'] = profile
        runtime_identity['implementation_sha256'].pop('fused_linear_logprobs', None)
        if mode == 'inference' and getattr(self, '_inference_runtime', None):
            runtime_identity['batched_sampling'] = dict(self._inference_runtime)
            runtime_identity['implementation_sha256']['batched_sampler'] = self._inference_runtime['sampler_sha256']
            runtime_identity['implementation_sha256']['batched_reference_scorer'] = self._inference_runtime['reference_scorer_sha256']
        else:
            runtime_identity.pop('batched_sampling', None)
            runtime_identity['implementation_sha256'].pop('batched_sampler', None)
            runtime_identity['implementation_sha256'].pop('batched_reference_scorer', None)
        math_mode = runtime_identity['math']
        torch = self.torch
        torch.set_float32_matmul_precision(math_mode['float32_matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32 = math_mode['allow_tf32']
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = math_mode['allow_bf16_reduced_precision_reduction']
        runtime_identity['nf4_dispatch'] = resolved_nf4_dispatch(self.model)
        runtime = dict(identity=runtime_identity, identity_sha256=digest(runtime_identity))
        self.probability_profile = profile
        self.info.update(execution_mode=mode, probability_execution=runtime,
                         gradient_checkpointing=self._training_checkpointing if mode == 'training' else False)
        self.execution.update(execution_mode=mode, probability_runtime=runtime,
            gradient_checkpointing=self.info['gradient_checkpointing'],
            probability_execution_profile=profile, probability_execution_sha256=digest(profile))
        self.identity = identities(self.model, self.tokenizer, self.info, self.cfg, profile=profile)
        return self.identity

    def load_adapter_in_place(self, adapter_run):
        """Copy one exported plain adapter into existing Parameter objects."""
        from safetensors.torch import load_file
        adapter_path = Path(adapter_run) / 'adapter'
        saved_semantics = inspect_plain_adapter_for_config(adapter_path, self.cfg)
        peft, torch = self.stack['peft'], self.torch
        weights = load_file(str(adapter_path / 'adapter_model.safetensors'), device='cpu')
        current = peft.get_peft_model_state_dict(self.model, adapter_name='default')
        if set(current) != set(weights) or any(current[k].shape != weights[k].shape or
                current[k].dtype != weights[k].dtype for k in current):
            raise TrainingError('Resident adapter synchronization differs in keys/shapes/dtypes')
        result = peft.set_peft_model_state_dict(self.model, weights, adapter_name='default')
        if getattr(result, 'unexpected_keys', []) or any('lora_' in key for key in getattr(result, 'missing_keys', [])):
            raise TrainingError('Resident adapter synchronization is incomplete')
        loaded = peft.get_peft_model_state_dict(self.model, adapter_name='default')
        if any(_tensor_digest(loaded[k], torch) != _tensor_digest(weights[k], torch) for k in weights):
            raise TrainingError('Resident adapter synchronization changed saved values')
        self.info['adapter_initialization'] = dict(resumed_adapter=True,
            saved_configuration_semantics=saved_semantics,
            adapter_file_sha256=sha256_file(adapter_path / 'adapter_model.safetensors'),
            actual_adapter_sha256=trainable_hash(self.model),
            boundary='Resident in-place synchronization from committed adapter')
        self.execution['adapter_initialization'] = self.info['adapter_initialization']
        del weights, current, loaded
        return self.switch_execution_mode('inference')

    def clear_generation_cache(self, *, reason='explicit-reset'):
        self._generation_cache = None
        self._generation_cache_reset_reason = reason

    def _sample_prefill(self, prompt, input_ids):
        """Reuse actor KV only for a strictly longer, byte-exact token prefix."""
        retained = getattr(self, '_generation_cache', None)
        self._generation_cache = None  # transfer ownership before any forward
        past, prefix_length = None, 0
        reason = getattr(self, '_generation_cache_reset_reason', 'empty-cache')
        if retained is not None:
            tokens, candidate = retained
            if len(tokens) < len(prompt) and tuple(prompt[:len(tokens)]) == tokens:
                if int(candidate.get_seq_length()) != len(tokens):
                    raise TrainingError('Retained actor KV/token length mismatch')
                past, prefix_length, reason = candidate, len(tokens), None
            else:
                reason = 'prompt-is-not-a-strict-extension-of-cached-tokens'
            del candidate, tokens, retained
        started = time.monotonic()
        next_logits, past = cached_prefill(self.model, input_ids,
            past_key_values=past, prefix_length=prefix_length)
        for device in self.cuda_devices:
            self.torch.cuda.synchronize(device)
        report = dict(enabled=True, hit=prefix_length > 0,
            reused_tokens=prefix_length, new_prefill_tokens=len(prompt) - prefix_length,
            reset_reason=reason, prefill_seconds=time.monotonic() - started)
        return next_logits, past, report

    def _retain_sample_cache(self, prompt, completion, past):
        # The final selected token has not yet been forwarded. It will be read
        # with the tool suffix on the next call. Reference scoring allocates a
        # separate fresh cache and never receives this actor object.
        tokens = tuple(prompt) + tuple(completion[:-1])
        if not completion or int(past.get_seq_length()) != len(tokens):
            self.clear_generation_cache(reason='unexpected-sampled-cache-length')
            raise TrainingError('Sampled actor KV does not cover the exact forwarded tokens')
        self._generation_cache = (tokens, past)
        self._generation_cache_reset_reason = None
        return len(tokens)
