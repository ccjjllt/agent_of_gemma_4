"""Exact-token NF4 policy sampling for the bounded VFARL experiment.

The collector supplies the real six-tool history. This module never executes a
tool, fabricates a reward, or reconstructs behavior probabilities from a CT
server. Sampling and updating use the same QAT-derived NF4 cache.
"""
from contextlib import contextmanager
import functools
import hashlib
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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


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
              max_old_logprob_error=0.08)
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
                     reference='frozen-NF4-base-equivalent-to-zero-initialized-LoRA')
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
        self.device = self.model.get_input_embeddings().weight.device
        self.cuda_devices = ([torch.device('cuda', index) for index in range(2)]
                             if inference_device_count == 2 else [self.device])
        self.execution = dict(inference_device_count=inference_device_count,
            input_device=str(self.device),
            device_map=(self.info.get('inference_placement', {}).get('device_map')
                        or {'': str(self.device)}),
            max_memory=placement.get('inference_max_memory'),
            adapter_initialization=self.info['adapter_initialization'])
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

    def sample(self, prompt_token_ids, *, max_new_tokens, seed,
               thinking_budget=1024, stop_token_ids=None):
        torch = self.torch
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
            # A separate, actual reference forward, not a copy of behavior data.
            ids = torch.tensor([prompt + generated], device=self.device, dtype=torch.long)
            selected = torch.tensor([[0] * len(prompt) + mask], device=self.device, dtype=torch.bool)
            with frozen_base_reference(self.model):
                values = cached_selected_logprobs(self.model.get_base_model(), ids, selected,
                    projection_chunk_size=self.rl['logprob_chunk_size'])
            reference, cursor = [], 0
            for chosen in mask:
                reference.append(float(values[cursor].item()) if chosen else 0.0)
                cursor += chosen
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
                    reference_scoring='actual-cached-chunked-forward',
                    reference_forward_performed=True, seconds=time.monotonic() - started,
                    prompt_tokens=len(prompt), completion_tokens=len(generated),
                    execution=self.execution, sampling_device=str(generator.device),
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
