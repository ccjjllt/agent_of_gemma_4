"""Exact-token NF4 policy sampling for the bounded VFARL experiment.

The collector supplies the real six-tool history. This module never executes a
tool, fabricates a reward, or reconstructs behavior probabilities from a CT
server. Sampling and updating use the same QAT-derived NF4 cache.
"""
from contextlib import contextmanager, nullcontext
import functools
import hashlib
import importlib
import json
import math
import random
from pathlib import Path
import time
import types

from . import train_sft as single
from .common import TrainingError, sha256_file
from .encoding.gemma_tool_chat import tokenizer_fingerprint
from .train_ddp import trainable_hash


PREFILL_CHUNK_SIZE = 256
HEAD_VOCAB_CHUNK_SIZE = 16384
PROBABILITY_EXECUTION_PROFILE = {
    'version': 3,
    'prompt_prefill': 'complete-prefix-contiguous-256-token-KV-chunks',
    'completion': 'one-token-cached-autoregressive-forward',
    'nf4_linear': 'bitsandbytes-0.50.2-CUDA-dequant-linear-fallback-for-all-M',
    'head': 'tied-BF16-parameter-FP32-linear-vocab-blocks-16384',
    'head_backward': 'shared-FP32-input-with-per-vocab-block-nonreentrant-checkpoint',
    'float32_matmul_precision': 'highest',
    'allow_tf32': False,
    'allow_bf16_reduced_precision_reduction': False,
    'logprob': 'FP32-head-and-native-FP32-softcap-to-float32-log-softmax',
    'reference': 'independent-same-autoregressive-path-with-zero-LoRA-scaling',
    'forced_controls': 'preserve-in-prefix-and-mask-from-loss',
    'compute_dtype': 'bfloat16',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _install_fp32_blocked_head(model, torch):
    """Keep the tied BF16 parameter while evaluating its projection in FP32.

    This is installed on the Linear module's existing execution boundary, so
    Accelerate/FSDP hooks still run. Every vocabulary block checkpoints its
    own cast/matmul when gradients are required; the complete FP32 weight is
    never materialized or retained by this helper.
    """
    from torch.utils.checkpoint import checkpoint
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if type(base).__name__ not in ('Gemma4ForConditionalGeneration',
                                   'Gemma4UnifiedForConditionalGeneration'):
        raise TrainingError('VFARL FP32 probability head requires the supported Gemma4 base')
    head = base.lm_head
    embedding = base.get_input_embeddings()
    if (not isinstance(head, torch.nn.Linear) or head.bias is not None
            or head.weight.requires_grad or head.weight.dtype != torch.bfloat16
            or head.weight is not embedding.weight):
        raise TrainingError('VFARL probability head requires the original frozen tied BF16 Linear weight')
    active = getattr(head, '_vfarl_probability_head', None)
    if active is not None:
        if active.get('vocab_chunk_size') != HEAD_VOCAB_CHUNK_SIZE:
            raise TrainingError('A different VFARL probability head is already installed')
        return dict(active)
    # dispatch_model wraps forward with a device-alignment hook and keeps its
    # actual implementation in _old_forward. Preserve that outer hook.
    target = '_old_forward' if hasattr(head, '_hf_hook') else 'forward'
    original = getattr(head, target, None)
    if not callable(original):
        raise TrainingError('The probability head device-alignment hook has no callable implementation')

    @functools.wraps(original)
    def forward(self, input):
        if input.device != self.weight.device:
            raise TrainingError('Probability head hook did not align hidden states and the tied weight')
        with torch.autocast(device_type=input.device.type, enabled=False):
            # Share one cast so all vocabulary-block gradients accumulate in
            # FP32 before the single conversion back to the backbone dtype.
            values = input.float()

            def project(hidden, weight):
                with torch.autocast(device_type=hidden.device.type, enabled=False):
                    return torch.nn.functional.linear(hidden, weight.float())

            pieces = []
            for start in range(0, self.weight.shape[0], HEAD_VOCAB_CHUNK_SIZE):
                weight = self.weight[start:start + HEAD_VOCAB_CHUNK_SIZE]
                pieces.append(checkpoint(project, values, weight,
                    use_reentrant=False, preserve_rng_state=False)
                    if torch.is_grad_enabled() and values.requires_grad else project(values, weight))
            return torch.cat(pieces, dim=-1)

    setattr(head, target, types.MethodType(forward, head))
    report = dict(vocab_chunk_size=HEAD_VOCAB_CHUNK_SIZE,
        weight_shape=list(head.weight.shape), weight_dtype=str(head.weight.dtype),
        output_dtype='torch.float32', tied_parameter_preserved=True,
        per_vocab_block_checkpoint=True, shared_input_cast=True,
        patched_implementation=target)
    head._vfarl_probability_head = report
    return dict(report)


def install_probability_execution(model, stack):
    """Install this experiment's explicit NF4 and probability arithmetic.

    The private bitsandbytes control is restricted to the inspected 0.50.2
    CUDA implementation. This function is called only by VFARL, after model
    loading and before FSDP wrapping. Generic SFT loading stays unchanged.
    """
    torch, bnb = stack['torch'], stack['bitsandbytes']
    if (not torch.cuda.is_available() or torch.version.hip is not None
            or getattr(bnb, '__version__', None) != '0.50.2'):
        raise TrainingError('VFARL uniform probability execution requires CUDA and bitsandbytes==0.50.2')
    operations = importlib.import_module('bitsandbytes.backends.cuda.ops')
    source_path = Path(operations.__file__).resolve()
    source = source_path.read_text()
    if (not callable(getattr(operations, '_dequant_linear_fallback', None))
            or type(getattr(operations, '_gemm_4bit_custom_max_m', None)) is not int
            or 'if M > _gemm_4bit_custom_max_m:' not in source
            or 'return _dequant_linear_fallback(' not in source):
        raise TrainingError('Unsupported bitsandbytes 0.50.2 CUDA GEMM dispatch; cannot bind the uniform fallback')
    previous = operations._gemm_4bit_custom_max_m
    owned = getattr(operations, '_vfarl_uniform_probability_dispatch', None)
    backend_sha = sha256_file(source_path)
    if previous != 1536 and not (previous == 0 and owned == backend_sha):
        raise TrainingError('Unexpected existing bitsandbytes CUDA threshold; refuse to overwrite another backend policy')
    if any(p.device.type != 'cuda' for p in model.parameters()):
        raise TrainingError('VFARL probability execution does not support CPU/disk model offload')
    head = _install_fp32_blocked_head(model, torch)
    torch.set_float32_matmul_precision('highest')
    if torch.get_float32_matmul_precision() != 'highest' or torch.backends.cuda.matmul.allow_tf32:
        raise TrainingError('VFARL FP32 probability projection requires highest precision with TF32 disabled')
    try:
        reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    except AttributeError as exc:
        raise TrainingError('This torch lacks the required BF16 matmul reduction control') from exc
    if type(reduced) is not bool:
        raise TrainingError('Unsupported BF16 matmul reduction control type')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
        raise TrainingError('VFARL BF16 matmuls require full-precision accumulation')
    operations._gemm_4bit_custom_max_m = 0
    operations._vfarl_uniform_probability_dispatch = backend_sha
    active = dict(profile=PROBABILITY_EXECUTION_PROFILE,
        bitsandbytes_version=bnb.__version__, cuda_backend_sha256=backend_sha,
        nf4_gemm_custom_max_m=operations._gemm_4bit_custom_max_m,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        allow_bf16_reduced_precision_reduction=bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        head={key: value for key, value in head.items() if key != 'patched_implementation'})
    return dict(identity=active, identity_sha256=digest(active),
                installation=dict(cuda_backend_path=str(source_path),
                    previous_nf4_gemm_custom_max_m=previous,
                    head_implementation_boundary=head['patched_implementation']))


def read_config(path):
    if isinstance(path, (str, Path)):
        text = Path(path).read_text()
        if text.lstrip().startswith('{'):
            raw = json.loads(text)
        else:
            import yaml
            raw = yaml.safe_load(text)
    else:
        raw = dict(path)
    cfg = single.validate_config(raw['model'])
    if (cfg['quant_backend'] != 'nf4' or not cfg['allow_cached_nf4']
            or not cfg['local_files_only'] or cfg['compute_dtype'] != 'bfloat16'
            or cfg['rank'] != 32 or cfg['dropout'] != 0
            or set(cfg['target_text_modules']) != {
                'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}):
        raise TrainingError('VFARL requires offline NF4/BF16, r32 seven text projections and zero dropout')
    if cfg['base_family'] != 'gemma-4-31b-it-qat-q4_0-unquantized':
        raise TrainingError('This experiment is restricted to the provenance-checked 31B QAT-derived cache')
    rl = dict(clip_epsilon=0.2, kl_beta=0.02, logprob_chunk_size=16,
              max_old_logprob_error=0.08, rollout_correction='none', rollout_is_max=2.0)
    if set(raw.get('rl', {})) - set(rl):
        raise TrainingError('Unsupported VFARL objective setting')
    rl.update(raw.get('rl', {}))
    if (not all(isinstance(rl[key], (int, float)) and math.isfinite(rl[key])
                for key in ('clip_epsilon', 'kl_beta', 'max_old_logprob_error'))
            or not 0 < rl['clip_epsilon'] < 1 or rl['kl_beta'] < 0
            or rl['max_old_logprob_error'] <= 0):
        raise TrainingError('Invalid PPO clipping or KL coefficient')
    if not isinstance(rl['logprob_chunk_size'], int) or rl['logprob_chunk_size'] < 1:
        raise TrainingError('logprob_chunk_size must be positive')
    if (rl['rollout_correction'] not in ('none','sequence_tis')
            or type(rl['rollout_is_max']) not in (int,float)
            or not math.isfinite(rl['rollout_is_max']) or rl['rollout_is_max'] < 1):
        raise TrainingError('Unsupported rollout correction or invalid sequence IS cap')
    return cfg, rl


def bind_template(tokenizer, cfg):
    if cfg.get('chat_template_path'):
        tokenizer.chat_template = Path(cfg['chat_template_path']).read_text()
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        raise TrainingError('VFARL requires one explicit deployment chat template')


def identities(model, tokenizer, info, cfg):
    reference = dict(nf4_cache_manifest_sha256=info['nf4_cache']['manifest_sha256'],
                     tokenizer_sha256=tokenizer_fingerprint(tokenizer),
                     template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                     tool_schema_sha256=sha256_file(Path(cfg['tool_schema_path'])),
                     reference='frozen-NF4-base-equivalent-to-zero-initialized-LoRA',
                     probability_execution_sha256=digest(PROBABILITY_EXECUTION_PROFILE),
                     probability_runtime_sha256=info['probability_execution']['identity_sha256'])
    policy = dict(reference, adapter_sha256=trainable_hash(model),
                  temperature=1.0, top_p=1.0, top_k=0, dropout=0.0)
    return dict(policy_id=digest(policy), reference_id=digest(reference),
                policy_identity=policy, reference_identity=reference)


def require_zero_lora(model, torch):
    matrices = [p for name, p in model.named_parameters() if '.lora_B.' in name]
    if not matrices or any(bool(torch.count_nonzero(p).item()) for p in matrices):
        raise TrainingError('First VFARL update requires genuinely zero-initialized LoRA B matrices')


@contextmanager
def frozen_base_reference(model):
    """Evaluate the fixed base without changing FSDP trainability or traversal.

    PEFT disable_adapter() changes requires_grad and skips the LoRA A/B leaves.
    Zero scaling preserves those FSDP collectives and evaluates base + 0 on the
    plain finite LoRA path. Call under no_grad/inference_mode on every rank.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer
    layers = []
    for module in model.modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        if (not hasattr(module, 'scaling') or set(module.scaling) != {'default'}
                or module.merged or getattr(module, 'lora_variant', None)):
            raise TrainingError('Fixed VFARL reference requires one unmerged plain default LoRA')
        layers.append((module, dict(module.scaling)))
    if not layers:
        raise TrainingError('No plain LoRA layers found for fixed base reference')
    try:
        for module, _ in layers:
            module.scaling['default'] = 0.0
        yield
    finally:
        for module, scaling in layers:
            module.scaling.update(scaling)


def install_eval_checkpointing(model):
    """Keep eval/dropout semantics while checkpointing real decoder layers.

    HF's built-in checkpoint path is conditional on training mode. These
    wrappers run inside the later FSDP decoder units, preserving their hooks.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    count = 0
    for module in model.modules():
        if type(module).__name__ not in ('Gemma4TextDecoderLayer', 'Gemma4UnifiedTextDecoderLayer'):
            continue
        if hasattr(module, '_vfarl_eval_checkpoint'):
            count += 1
            continue
        original = module.forward

        def make_forward(bound):
            @functools.wraps(bound)
            def forward(*args, **kwargs):
                if torch.is_grad_enabled():
                    return checkpoint(bound, *args, use_reentrant=False, **kwargs)
                return bound(*args, **kwargs)
            return forward

        module.forward = make_forward(original)
        module._vfarl_eval_checkpoint = True
        count += 1
    if not count:
        raise TrainingError('No supported Gemma4 decoder layers found for eval checkpointing')
    return count


def selected_logprobs(base, input_ids, attention_mask, selected_mask, *, chunk_size=16):
    """Differentiable log P(token[t] | tokens[:t]), projecting selected positions.

    Return order is the row-major order of selected_mask[:, 1:]. No vocab-sized
    tensor escapes a checkpointed chunk; Gemma's native softcap is preserved.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    if input_ids.ndim != 2 or input_ids.shape != selected_mask.shape:
        raise TrainingError('Token IDs and selection mask must have equal [batch, sequence] shape')
    selection = selected_mask[:, 1:].bool()
    batch_indices, positions = selection.nonzero(as_tuple=True)
    if not positions.numel():
        raise TrainingError('No sampled completion tokens selected')
    targets = input_ids[:, 1:][selection]
    outputs = base.model(input_ids=input_ids, attention_mask=attention_mask,
                         use_cache=False, return_dict=True)
    hidden = outputs.last_hidden_state
    cap = base.config.get_text_config().final_logit_softcapping

    def project(values, token_ids):
        logits = base.lm_head(values)
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return -torch.nn.functional.cross_entropy(
            logits.float(), token_ids.to(logits.device), reduction='none')

    pieces = []
    for start in range(0, targets.numel(), chunk_size):
        end = start + chunk_size
        values = hidden[batch_indices[start:end], positions[start:end]]
        token_ids = targets[start:end]
        pieces.append(checkpoint(project, values, token_ids, use_reentrant=False)
                      if torch.is_grad_enabled() else project(values, token_ids))
    return torch.cat(pieces)


def cached_prefill(model, input_ids, *, chunk_size=PREFILL_CHUNK_SIZE):
    """Read the entire unpadded prefix in bounded chunks and return its next logits.

    Gemma's DynamicCache retains absolute positions and handles sliding/shared
    KV state. Each chunk receives the attention mask for every token seen so
    far; chunking changes computation scheduling, not the visible history.
    """
    import torch
    if (torch.is_grad_enabled() or input_ids.ndim != 2 or input_ids.shape[0] != 1
            or not input_ids.shape[1] or chunk_size < 1):
        raise TrainingError('Cached prefill requires one nonempty prefix, no gradients and a positive chunk size')
    past = None
    for start in range(0, input_ids.shape[1], chunk_size):
        end = min(start + chunk_size, input_ids.shape[1])
        result = model(input_ids=input_ids[:, start:end],
                       attention_mask=torch.ones((1, end), dtype=torch.long, device=input_ids.device),
                       past_key_values=past, use_cache=True, logits_to_keep=1, return_dict=True)
        past = result.past_key_values
        if past is None:
            raise TrainingError('Chunked prefill requested use_cache=True but returned no KV cache')
        if end == input_ids.shape[1]:
            return result.logits[:, -1, :], past
        del result


def cached_selected_logprobs(base, input_ids, selected_mask, *,
                            prefill_chunk_size=PREFILL_CHUNK_SIZE, projection_chunk_size=16):
    """Actual reference scoring with bounded backbone and vocabulary projections.

    Feed all predictor tokens, including forced controls, through the same KV
    cache. Only selected next-token targets enter the returned log probabilities.
    This inference-only path is separate from the differentiable FSDP scorer.
    """
    import torch
    if (torch.is_grad_enabled() or input_ids.ndim != 2 or input_ids.shape[0] != 1
            or input_ids.shape != selected_mask.shape or input_ids.shape[1] < 2
            or prefill_chunk_size < 1 or projection_chunk_size < 1):
        raise TrainingError('Cached scoring requires matching single-sequence tokens/mask and no gradients')
    if not bool(selected_mask[:, 1:].any()):
        raise TrainingError('No sampled completion tokens selected for cached scoring')
    cap = base.config.get_text_config().final_logit_softcapping
    pieces, past = [], None
    # Input position t predicts token t+1. The final target need not be cached.
    for start in range(0, input_ids.shape[1] - 1, prefill_chunk_size):
        end = min(start + prefill_chunk_size, input_ids.shape[1] - 1)
        result = base.model(input_ids=input_ids[:, start:end],
                            attention_mask=torch.ones((1, end), dtype=torch.long, device=input_ids.device),
                            past_key_values=past, use_cache=True, return_dict=True)
        past = result.past_key_values
        if past is None:
            raise TrainingError('Chunked reference scoring requested use_cache=True but returned no KV cache')
        selection = selected_mask[:, start + 1:end + 1].bool()
        batch_indices, positions = selection.nonzero(as_tuple=True)
        targets = input_ids[:, start + 1:end + 1][selection]
        for offset in range(0, targets.numel(), projection_chunk_size):
            stop = offset + projection_chunk_size
            hidden_device = result.last_hidden_state.device
            hidden = result.last_hidden_state[batch_indices[offset:stop].to(hidden_device),
                                              positions[offset:stop].to(hidden_device)]
            logits = base.lm_head(hidden)
            if cap is not None:
                logits = torch.tanh(logits / cap) * cap
            pieces.append(-torch.nn.functional.cross_entropy(
                logits.float(), targets[offset:stop].to(logits.device), reduction='none'))
            del hidden, logits
        del result, selection, batch_indices, positions, targets
    del past
    return torch.cat(pieces)


def cached_teacher_forced_logprobs(model, prompt_ids, completion_token_ids, loss_mask, *,
                                  prefill_chunk_size=PREFILL_CHUNK_SIZE):
    """Score actual next tokens using the sampler's exact forward schedule.

    The prompt is read in the same 256-token chunks as generation. Completion
    tokens then use one-token KV forwards and the native last-position head,
    including every forced control in the prefix. No token is sampled here.
    Return one log probability per completion position, zero for mask=0.
    """
    import torch
    if (torch.is_grad_enabled() or prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1
            or not prompt_ids.shape[1] or not completion_token_ids
            or len(completion_token_ids) != len(loss_mask)
            or any(type(token) is not int or token < 0 for token in completion_token_ids)
            or any(type(keep) is not int or keep not in (0, 1) for keep in loss_mask)):
        raise TrainingError('Teacher-forced cached scoring needs one real prefix, completion/mask and no gradients')
    next_logits, past = cached_prefill(model, prompt_ids, chunk_size=prefill_chunk_size)
    values = []
    for index, (token, keep) in enumerate(zip(completion_token_ids, loss_mask)):
        if index:
            current = torch.tensor([[completion_token_ids[index - 1]]], dtype=torch.long,
                                   device=prompt_ids.device)
            attention = torch.ones((1, prompt_ids.shape[1] + index), dtype=torch.long,
                                   device=prompt_ids.device)
            result = model(input_ids=current, attention_mask=attention,
                           past_key_values=past, use_cache=True, logits_to_keep=1,
                           return_dict=True)
            past = result.past_key_values
            if past is None:
                raise TrainingError('Teacher-forced scoring requested use_cache=True but returned no KV cache')
            next_logits = result.logits[:, -1, :]
            del result, current, attention
        logits = next_logits[0].float()
        del next_logits
        logprobs = torch.log_softmax(logits, dim=-1)
        values.append(logprobs[token].clone() if keep else logprobs.new_zeros(()))
        del logits, logprobs
    del past
    return torch.stack(values)


def install_logprob_forward(model, chunk_size=16):
    """Route selected-token forwards through PEFT and the outer FSDP hooks."""
    base = model.get_base_model()
    original = base.forward

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        mask = kwargs.pop('vfarl_selected_mask', None)
        if mask is None:
            return original(*args, **kwargs)
        if args:
            raise TrainingError('VFARL selected forwards require named arguments')
        return selected_logprobs(self, kwargs['input_ids'], kwargs.get('attention_mask'),
                                 mask, chunk_size=chunk_size)

    base.forward = types.MethodType(forward, base)


class NF4Policy:
    def __init__(self, config, *, seed=None, adapter_run=None, device=None,
                 inference_device_count=1):
        if type(inference_device_count) is not int or inference_device_count not in (1, 2):
            raise TrainingError('VFARL sampling supports one or two GPUs per policy')
        self.cfg, self.rl = read_config(config)
        if device is not None:
            self.cfg['device'] = device
        self.stack = single.require_stack('nf4')
        self.torch = torch = self.stack['torch']
        if seed is not None and seed != self.cfg['seed']:
            raise TrainingError('All workers must initialize LoRA with config seed; vary sample(seed=...) only')
        seed = self.cfg['seed']
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        placement = ({'inference_device_map': 'balanced',
                      'inference_max_memory': {0: '14GiB', 1: '14GiB'}}
                     if inference_device_count == 2 else {})
        self.model, self.tokenizer, self.device, self.info = single.load_model_and_tokenizer(
            self.cfg, self.stack, resume=adapter_run, adapter_init_seed=seed, **placement)
        self.info['probability_execution'] = install_probability_execution(self.model, self.stack)
        self.device = self.model.get_input_embeddings().weight.device
        self.cuda_devices = ([torch.device('cuda', index) for index in range(2)]
                             if inference_device_count == 2 else [self.device])
        self.execution = dict(inference_device_count=inference_device_count,
            input_device=str(self.device),
            device_map=(self.info.get('inference_placement', {}).get('device_map')
                        or {'': str(self.device)}),
            max_memory=placement.get('inference_max_memory'),
            adapter_initialization=self.info['adapter_initialization'],
            probability_execution_profile=PROBABILITY_EXECUTION_PROFILE,
            probability_execution_sha256=digest(PROBABILITY_EXECUTION_PROFILE),
            probability_runtime=self.info['probability_execution'])
        if self.info.get('nf4_cache') is None:
            raise TrainingError('VFARL sampling requires the same validated cache as training')
        bind_template(self.tokenizer, self.cfg)
        self.model.eval()
        if adapter_run is None:
            require_zero_lora(self.model, torch)
        self.identity = identities(self.model, self.tokenizer, self.info, self.cfg)
        self.thought_open = self._special('<|channel>')
        self.thought_close = self._special('<channel|>')
        self.tool_close = self._special('<tool_call|>')
        self.turn_close = self._special('<turn|>')

    def _special(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1 or self.tokenizer.decode(ids, skip_special_tokens=False) != text:
            raise TrainingError('Expected one actual Gemma control token: ' + text)
        return ids[0]

    def render_prompt(self, messages, tools):
        ids = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=True, add_generation_prompt=True,
            enable_thinking=True, preserve_thinking=True, return_dict=False)
        # Transformers 5 defaults to BatchEncoding. A state must bind the
        # actual flat token sequence, never the mapping's string keys.
        if not isinstance(ids, list) or not ids or any(type(token) is not int for token in ids):
            raise TrainingError('Expected one flat sequence of exact prompt token IDs')
        return ids

    def _score_record(self, prompt_token_ids, completion_token_ids, loss_mask, *,
                      frozen_reference=False, cached_reference=False):
        torch = self.torch
        prompt, completion, mask = (list(prompt_token_ids), list(completion_token_ids),
                                    list(loss_mask))
        if (not prompt or not completion or len(completion) != len(mask)
                or any(type(token) is not int or token < 0 for token in prompt + completion)
                or any(type(keep) is not int or keep not in (0, 1) for keep in mask)
                or len(prompt) + len(completion) > self.cfg['max_sequence_length']):
            raise TrainingError('Invalid exact-token teacher-forced sequence or full context exceeds the run limit')
        if cached_reference and not frozen_reference:
            raise TrainingError('The legacy cached comparison is restricted to the frozen reference')
        started = time.monotonic()
        for device in self.cuda_devices:
            torch.cuda.reset_peak_memory_stats(device)
        self.model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            with frozen_base_reference(self.model) if frozen_reference else nullcontext():
                if cached_reference:
                    ids = torch.tensor([prompt + completion], dtype=torch.long, device=self.device)
                    selected = torch.tensor([[0] * len(prompt) + mask], dtype=torch.bool,
                                            device=self.device)
                    selected_values = cached_selected_logprobs(self.model.get_base_model(), ids,
                        selected, projection_chunk_size=self.rl['logprob_chunk_size'])
                    values, cursor = [], 0
                    for keep in mask:
                        values.append(float(selected_values[cursor].item()) if keep else 0.0)
                        cursor += keep
                    del ids, selected, selected_values
                else:
                    ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
                    scored = cached_teacher_forced_logprobs(self.model, ids, completion, mask)
                    values = scored.float().cpu().tolist()
                    del ids, scored
            for device in self.cuda_devices:
                torch.cuda.synchronize(device)
        memory = [dict(device=str(device),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                  for device in self.cuda_devices]
        return dict(prompt_token_ids=prompt, completion_token_ids=completion,
                    loss_mask=mask, logprobs=values,
                    scoring=('legacy-cached-256-backbone-chunks-selected-head-chunks'
                             if cached_reference else 'sampler-style-teacher-forced'),
                    frozen_reference=frozen_reference,
                    probability_execution_sha256=digest(PROBABILITY_EXECUTION_PROFILE),
                    execution=self.execution, memory_by_device=memory,
                    seconds=time.monotonic() - started, **self.identity)

    def score_teacher_forced(self, prompt_token_ids, completion_token_ids, loss_mask, *,
                            frozen_reference=False):
        return self._score_record(prompt_token_ids, completion_token_ids, loss_mask,
                                  frozen_reference=frozen_reference)

    def score_cached_reference(self, prompt_token_ids, completion_token_ids, loss_mask):
        """Keep the former reference schedule available for explicit diagnostics."""
        return self._score_record(prompt_token_ids, completion_token_ids, loss_mask,
                                  frozen_reference=True, cached_reference=True)

    def sample(self, prompt_token_ids, *, max_new_tokens, seed,
               thinking_budget=1024, stop_token_ids=None, compute_reference=True):
        torch = self.torch
        if type(compute_reference) is not bool:
            raise TrainingError('compute_reference must explicitly be True or False')
        prompt = list(prompt_token_ids)
        if not prompt or max_new_tokens < 1 or thinking_budget < 1:
            raise TrainingError('Nonempty prompt and positive generation/thinking budgets required')
        available = self.cfg['max_sequence_length'] - len(prompt)
        if available < 1:
            raise TrainingError('Full real prompt leaves no completion room; truncation is forbidden')
        limit = min(max_new_tokens, available)
        stop = set(stop_token_ids or [self.tool_close, self.turn_close])
        generator = None
        generated, old, mask = [], [], []
        in_thought = False
        thought_count = 0
        started = time.monotonic()
        for device in self.cuda_devices:
            torch.cuda.reset_peak_memory_stats(device)
        reason = 'context_limit' if available < max_new_tokens else 'length'
        self.model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
            next_logits, past = cached_prefill(self.model, ids)
            del ids
            for index in range(limit):
                if index:
                    ids = torch.tensor([[generated[-1]]], dtype=torch.long, device=self.device)
                    attention = torch.ones((1, len(prompt) + index), dtype=torch.long, device=self.device)
                    result = self.model(input_ids=ids, attention_mask=attention,
                                        past_key_values=past, use_cache=True, logits_to_keep=1,
                                        return_dict=True)
                    past = result.past_key_values
                    if past is None:
                        raise TrainingError('Generation requested use_cache=True but returned no KV cache')
                    next_logits = result.logits[:, -1, :]
                    del result, ids, attention
                logits = next_logits[0].float()
                del next_logits
                if generator is None:
                    generator = torch.Generator(device=logits.device).manual_seed(seed)
                elif generator.device != logits.device:
                    raise TrainingError('Autoregressive logits changed devices during one sampled action')
                logprobs = torch.log_softmax(logits, dim=-1)
                forced = in_thought and thought_count >= thinking_budget
                token = (self.thought_close if forced else
                         int(torch.multinomial(logprobs.exp(), 1, generator=generator).item()))
                generated.append(token)
                old.append(0.0 if forced else float(logprobs[token].item()))
                mask.append(0 if forced else 1)
                if token == self.thought_open:
                    in_thought = True
                elif token == self.thought_close:
                    in_thought = False
                elif in_thought:
                    thought_count += 1
                del logits, logprobs
                # A quoted control token inside actual thinking is not an
                # emitted tool action or terminal assistant response.
                if token in stop and not in_thought:
                    reason = 'tool_calls' if token == self.tool_close else 'stop'
                    break
            del past
            reference = None
            if compute_reference:
                # Only focal actions need a separate reference forward. A
                # skipped score is absent, never behavior data or fake zeros.
                ids = torch.tensor([prompt], device=self.device, dtype=torch.long)
                with frozen_base_reference(self.model):
                    values = cached_teacher_forced_logprobs(self.model, ids, generated, mask)
                reference = values.float().cpu().tolist()
                del ids, values
            for device in self.cuda_devices:
                torch.cuda.synchronize(device)
        memory_by_device = [dict(device=str(device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            allocated_bytes=torch.cuda.memory_allocated(device),
            reserved_bytes=torch.cuda.memory_reserved(device)) for device in self.cuda_devices]
        return dict(prompt_token_ids=prompt, completion_token_ids=generated,
                    behavior_logprobs=old, reference_logprobs=reference, loss_mask=mask,
                    raw_text=self.tokenizer.decode(generated, skip_special_tokens=False,
                                                  clean_up_tokenization_spaces=False),
                    finish_reason=reason, seed=seed, thinking_budget=thinking_budget,
                    thought_tokens=thought_count, forced_control_tokens=mask.count(0),
                    sampling=dict(temperature=1.0, top_p=1.0, top_k=0),
                    prefill_chunk_size=PREFILL_CHUNK_SIZE,
                    reference_scoring=('actual-independent-sampler-style-teacher-forced'
                        if compute_reference else 'not-computed-nonfocal-action'),
                    probability_execution_sha256=digest(PROBABILITY_EXECUTION_PROFILE),
                    reference_forward_performed=compute_reference,
                    seconds=time.monotonic() - started,
                    prompt_tokens=len(prompt), completion_tokens=len(generated),
                    execution=dict(self.execution,
                        reference_scope=('current-sampled-focal-action'
                            if compute_reference else 'not-computed-nonfocal-action')),
                    sampling_device=str(generator.device),
                    memory_by_device=memory_by_device,
                    peak_memory_scope='maximum_of_model_cuda_devices',
                    peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in memory_by_device),
                    peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in memory_by_device),
                    **self.identity)

    def close(self):
        del self.model
        import gc
        gc.collect()
        for device in self.cuda_devices:
            with self.torch.cuda.device(device):
                self.torch.cuda.empty_cache()
