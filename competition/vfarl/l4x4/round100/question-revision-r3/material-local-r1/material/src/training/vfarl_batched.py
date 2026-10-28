"""Single-owner batched actions for the frozen Unsloth VFARL policy.

The default v3 profile has fresh KV for each action batch. Optional v4 preserves
private per-session actor KV, incrementally prefills each row, and merges ready
rows for real batched decode. A stopping row is extracted before dummy updates
can evict its sliding history. Histories are never truncated; every request has
independent RNG and explicit logical positions.
"""
import asyncio
from collections import Counter, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import inspect
import copy
import threading
import time

from .common import TrainingError, sha256_file
from .vfarl_policy import digest, frozen_base_reference, cached_selected_logprobs
from .vfarl_unsloth import UNSLOTH_EXECUTION_PROFILE


BATCHED_EXECUTION_PROFILE = dict(UNSLOTH_EXECUTION_PROFILE,
    version=3, mode='batched-padded-action-decode',
    prompt_prefill='fresh-left-padded-batch-with-256-token-chunks-and-logical-per-row-positions',
    completion='batched-one-token-cached-forward-with-independent-request-generators',
    reference='independent-unpadded-zero-LoRA-256-token-backbone-and-16-selected-head-scoring',
    teacher_forced_scoring='fresh-reference-KV-with-chunked-backbone-and-selected-head-projection',
    cross_tool_kv=False,
    cache_boundary='fresh-private-KV-per-action-batch-no-cross-session-retention',
    padding='left-pad-keys-masked-logical-position-ids-completed-rows-masked',
    host_transfer='one-batched-token-and-logprob-transfer-per-decode-step')
BATCHED_UNSLOTH_EXECUTION_PROFILE = BATCHED_EXECUTION_PROFILE

SESSION_BATCHED_EXECUTION_PROFILE = dict(BATCHED_EXECUTION_PROFILE,
    version=4, mode='session-incremental-prefill-batched-decode',
    prompt_prefill='exact-session-prefix-KV-reuse-single-row-256-token-incremental-prefill',
    completion='right-aligned-cache-merge-batched-cached-forward-independent-request-generators',
    cross_tool_kv=True,
    cache_boundary='private-session-and-policy-bound-KV; independent-reference-cache; LRU-token-budget',
    cache_factory='explicit-Gemma4-DynamicLayer-or-DynamicSlidingWindowLayer-from-text-config',
    cache_retirement='extract-completed-row-before-any-masked-dummy-tail-update',
    padding='full-KV-left-pad; sliding-KV-right-align; explicit-logical-position-ids')

REWIND_SESSION_BATCHED_EXECUTION_PROFILE = dict(SESSION_BATCHED_EXECUTION_PROFILE,
    version=5, mode='session-prefix-rewind-incremental-prefill-batched-decode',
    cache_boundary='private-policy-and-epoch-bound-raw-KV-plus-pre-action-prefix-anchor; exact-prefix-rewind-only',
    cache_retirement='stop-time-B1-extraction; full-layer-crop-plus-owned-sliding-prefix-backup',
    prefix_rewind='strict-raw-extension-first; otherwise-strict-pre-action-prefix-extension; fresh-on-earlier-change',
    rewind_storage='clone-sliding-tail-only; full-crop-retains-backing-storage-until-next-update')


def configure_batched_policy(policy, *, max_batch_size=2, max_batch_tokens=32768,
                             prefill_chunk_size=256, session_cache_tokens=0,
                             session_prefix_rewind=False):
    """Bind the actual batch implementation/config before queue policy binding."""
    if (type(max_batch_size) is not int or max_batch_size < 1
            or type(max_batch_tokens) is not int or max_batch_tokens < 1
            or prefill_chunk_size != 256 or type(session_cache_tokens) is not int
            or session_cache_tokens < 0):
        raise TrainingError('Batched sampler needs positive batch/token budgets and its declared 256-token chunks')
    if type(session_prefix_rewind) is not bool or (session_prefix_rewind and not session_cache_tokens):
        raise TrainingError('Session prefix rewind requires an explicit boolean and positive session cache budget')
    runtime = dict(max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens, prefill_chunk_size=prefill_chunk_size,
        sampler_sha256=sha256_file(__file__),
        reference_scorer_sha256=sha256_file(inspect.getfile(cached_selected_logprobs)))
    profile = SESSION_BATCHED_EXECUTION_PROFILE if session_cache_tokens else BATCHED_EXECUTION_PROFILE
    if session_cache_tokens:
        runtime['session_cache_tokens'] = session_cache_tokens
    if session_prefix_rewind:
        runtime['session_prefix_rewind'] = True
        profile = REWIND_SESSION_BATCHED_EXECUTION_PROFILE
    if (getattr(policy, '_inference_profile', None) == profile
            and getattr(policy, '_inference_runtime', None) == runtime
            and policy.probability_profile == profile
            and policy.execution.get('execution_mode') == 'inference'):
        return policy.identity
    policy._inference_profile = dict(profile)
    policy._inference_runtime = runtime
    policy.clear_generation_cache(reason='batched-action-profile-installed')
    policy.switch_execution_mode('inference')
    return policy.identity


def _request(policy, prompt_token_ids, *, max_new_tokens, seed,
             thinking_budget=1024, stop_token_ids=None, compute_reference=True,
             session_id=None):
    prompt = list(prompt_token_ids)
    if (not prompt or any(type(token) is not int or token < 0 for token in prompt)
            or type(max_new_tokens) is not int or max_new_tokens < 1
            or type(thinking_budget) is not int or thinking_budget < 1
            or type(compute_reference) is not bool or type(seed) is not int):
        raise TrainingError('Batched actions require exact tokens, positive budgets and an explicit seed/reference flag')
    available = policy.cfg['max_sequence_length'] - len(prompt)
    if available < 1:
        raise TrainingError('Full real prompt leaves no completion room; truncation is forbidden')
    stop = set(stop_token_ids or [policy.tool_close, policy.turn_close])
    return dict(prompt=prompt, limit=min(max_new_tokens, available), seed=seed,
        max_new_tokens=max_new_tokens, thinking_budget=thinking_budget,
        compute_reference=compute_reference, session_id=session_id,
        stop=stop, generated=[], old=[], mask=[], in_thought=False, thought_count=0,
        done=False, generator=None,
        reason='context_limit' if available < max_new_tokens else 'length')


def _batch_token_slots(requests):
    # This bounds padded KV allocation, not just the sum of unpadded lengths.
    return len(requests) * (max(len(row['prompt']) for row in requests)
                            + max(row['limit'] for row in requests))


def _batch_span(requests):
    return max(len(row['prompt']) for row in requests) + max(row['limit'] for row in requests)


class SessionCacheStore:
    """LRU of inactive, exact-prefix actor caches; checked-out caches have one owner.

    The lock protects ownership transfers only. A released in-flight session
    changes epoch, so its eventual result cannot resurrect a retired cache.
    The token budget applies to retained entries, not the separate active batch.
    """
    def __init__(self, token_budget, *, session_prefix_rewind=False):
        if type(session_prefix_rewind) is not bool or (session_prefix_rewind and token_budget <= 0):
            raise TrainingError('Session prefix rewind requires a positive retained-cache budget')
        self.token_budget = token_budget
        self.session_prefix_rewind = session_prefix_rewind
        self._entries, self._epochs = OrderedDict(), {}
        self._lock = threading.RLock()
        self.tokens, self.hits, self.misses, self.evictions = 0, 0, 0, 0
        self.anchor_tensor_bytes, self.rewind_hits = 0, 0

    def epoch(self, session_id):
        with self._lock:
            if self.session_prefix_rewind:
                self._epochs.setdefault(session_id, 0)
            return self._epochs.get(session_id, 0)

    def _detach_anchor(self, entry):
        anchor = entry.pop('anchor', None)
        if anchor is not None:
            self.anchor_tensor_bytes -= anchor['tensor_bytes']
        return anchor

    def take(self, session_id, policy_id, prompt, epoch):
        with self._lock:
            entry = self._entries.pop(session_id, None) if session_id is not None else None
            if entry is not None:
                self.tokens -= len(entry['prefix'])
                anchor = self._detach_anchor(entry)
            else:
                anchor = None
            if self.session_prefix_rewind:
                self._epochs.setdefault(session_id, 0)
            reason = 'session-cache-empty'
            if epoch != self._epochs.get(session_id, 0):
                reason = 'session-retired'
            elif entry is not None:
                prefix = entry['prefix']
                if entry['policy_id'] != policy_id:
                    reason = 'session-policy-changed'
                elif len(prefix) >= len(prompt) or tuple(prompt[:len(prefix)]) != prefix:
                    if (self.session_prefix_rewind and anchor is not None
                            and len(anchor['prefix']) < len(prompt)
                            and tuple(prompt[:len(anchor['prefix'])]) == anchor['prefix']):
                        try:
                            _restore_session_prefix(entry['cache'], anchor, prefix, policy_id, epoch)
                        finally:
                            _release_session_anchor(anchor)
                        self.hits += 1
                        self.rewind_hits += 1
                        return entry['cache'], len(anchor['prefix']), 'session-prefix-rewind'
                    reason = 'prompt-is-not-a-strict-extension-of-session-prefix'
                elif int(entry['cache'].get_seq_length()) != len(prefix):
                    _release_session_anchor(anchor)
                    raise TrainingError('Session KV length differs from its exact token prefix')
                else:
                    _release_session_anchor(anchor)
                    self.hits += 1
                    return entry['cache'], len(prefix), None
            _release_session_anchor(anchor)
            self.misses += 1
            return None, 0, reason

    def put(self, session_id, policy_id, prefix, cache, epoch, *, anchor=None):
        if session_id is None:
            return False
        prefix = tuple(prefix)
        if int(cache.get_seq_length()) != len(prefix):
            raise TrainingError('Refusing to retain a session cache with a false token prefix')
        if anchor is not None:
            if not self.session_prefix_rewind:
                raise TrainingError('Prefix anchors require the explicit rewind session profile')
            _validate_session_anchor(cache, anchor, prefix, policy_id, epoch)
        with self._lock:
            if epoch != self._epochs.get(session_id, 0) or len(prefix) > self.token_budget:
                return False
            old = self._entries.pop(session_id, None)
            if old is not None:
                self.tokens -= len(old['prefix'])
                _release_session_anchor(self._detach_anchor(old))
            while self._entries and self.tokens + len(prefix) > self.token_budget:
                _, evicted = self._entries.popitem(last=False)
                self.tokens -= len(evicted['prefix'])
                _release_session_anchor(self._detach_anchor(evicted))
                self.evictions += 1
            self._entries[session_id] = dict(policy_id=policy_id, prefix=prefix, cache=cache)
            if anchor is not None:
                self._entries[session_id]['anchor'] = anchor
                self.anchor_tensor_bytes += anchor['tensor_bytes']
            self.tokens += len(prefix)
            return True

    def release(self, session_id):
        with self._lock:
            self._epochs[session_id] = self._epochs.get(session_id, 0) + 1
            entry = self._entries.pop(session_id, None)
            if entry is not None:
                self.tokens -= len(entry['prefix'])
                _release_session_anchor(self._detach_anchor(entry))

    def clear(self):
        with self._lock:
            for entry in self._entries.values():
                _release_session_anchor(self._detach_anchor(entry))
            self._entries.clear()
            self.tokens = 0
            self.anchor_tensor_bytes = 0
            if self.session_prefix_rewind:
                # Include epochs registered by checked-out/queued requests.
                for session_id in self._epochs:
                    self._epochs[session_id] += 1

    def stats(self):
        with self._lock:
            result = dict(retained_sessions=len(self._entries), retained_tokens=self.tokens,
                token_budget=self.token_budget, hits=self.hits, misses=self.misses,
                evictions=self.evictions)
            if self.session_prefix_rewind:
                result.update(session_prefix_rewind=True, rewind_hits=self.rewind_hits,
                    anchor_tensor_bytes=self.anchor_tensor_bytes,
                    anchor_storage='Sliding-only owned clones; full crop retains backing until update')
            return result


def _cache_from_layers(layers):
    # Preserve the DynamicCache type without re-running a constructor that
    # would infer/truncate layer types or lose sliding cumulative positions.
    from transformers.cache_utils import Cache, DynamicCache
    cache = DynamicCache.__new__(DynamicCache)
    Cache.__init__(cache, layers=layers)
    return cache


def _fresh_session_cache(policy):
    from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer
    config = policy.model.get_base_model().config.get_text_config()
    if getattr(config, 'num_kv_shared_layers', 0):
        raise TrainingError('Session cache cloning currently supports the non-shared 31B Gemma4 configuration')
    layers = []
    for kind in config.layer_types:
        if kind == 'sliding_attention':
            layers.append(DynamicSlidingWindowLayer(config=config))
        elif kind == 'full_attention':
            layers.append(DynamicLayer(config=config))
        else:
            raise TrainingError('Unsupported session cache layer type: ' + kind)
    return _cache_from_layers(layers)


def _check_cache_layer(layer):
    if type(layer).__name__ not in {'DynamicLayer', 'DynamicSlidingWindowLayer'}:
        raise TrainingError('Unsupported session cache implementation: ' + type(layer).__name__)
    if not layer.is_initialized or layer.keys.ndim != 4 or layer.values.ndim != 4:
        raise TrainingError('Session cache layers must contain actual four-dimensional KV states')


def _single_session_cache(cache, logical_length):
    """Validate a uniquely owned, unpadded B1 cache without copying its KV.

    The caller has taken this cache out of the session store. No other request
    may own it until it is returned to that store, or released after decode.
    """
    if not cache.layers:
        raise TrainingError('Single-session transfer requires actual cache layers')
    for layer in cache.layers:
        _check_cache_layer(layer)
        if int(layer.get_seq_length()) != logical_length:
            raise TrainingError('Single-session cache has a different logical length')
        expected = (min(logical_length, layer.sliding_window - 1)
                    if layer.is_sliding else logical_length)
        if any(value.shape[0] != 1 or value.shape[-2] != expected
               for value in (layer.keys, layer.values)):
            raise TrainingError('Single-session transfer requires one unpadded KV row')
    return cache


def _anchor_storage_bytes(layers):
    storages = {}
    for layer in layers:
        for name in ('keys', 'values'):
            value = layer.get(name)
            if value is not None:
                storage = value.untyped_storage()
                storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


def _release_session_anchor(anchor):
    if anchor is not None:
        for saved in anchor['layers']:
            saved.pop('keys', None)
            saved.pop('values', None)


def _capture_session_prefix(cache, prefix, policy_id, epoch):
    """Own only materialized sliding tails; full prefix KV stays in the cache."""
    prefix = tuple(prefix)
    _single_session_cache(cache, len(prefix))
    layers = []
    try:
        for layer in cache.layers:
            record = dict(type=type(layer).__name__, sliding=bool(layer.is_sliding),
                key_shape=tuple(layer.keys.shape), value_shape=tuple(layer.values.shape),
                dtype=str(layer.keys.dtype), device=str(layer.keys.device))
            layers.append(record)
            if layer.is_sliding:
                record['window'] = layer.sliding_window
                record['keys'], record['values'] = layer.keys.clone(), layer.values.clone()
    except BaseException:
        _release_session_anchor(dict(layers=layers))
        raise
    return dict(format='vfarl-session-prefix-anchor-v1', prefix=prefix,
        prefix_sha256=digest(list(prefix)), policy_id=policy_id, epoch=epoch,
        layers=layers, tensor_bytes=_anchor_storage_bytes(layers))


def _validate_session_anchor(cache, anchor, raw_prefix, policy_id, epoch):
    prefix = anchor.get('prefix', ())
    if (anchor.get('format') != 'vfarl-session-prefix-anchor-v1'
            or not prefix or anchor.get('prefix_sha256') != digest(list(prefix))
            or anchor.get('policy_id') != policy_id or anchor.get('epoch') != epoch
            or len(prefix) > len(raw_prefix) or tuple(raw_prefix[:len(prefix)]) != prefix
            or len(anchor.get('layers', [])) != len(cache.layers)
            or anchor.get('tensor_bytes') != _anchor_storage_bytes(anchor['layers'])):
        raise TrainingError('Session rewind anchor differs from its actual action/policy/epoch')
    _single_session_cache(cache, len(raw_prefix))
    for layer, saved in zip(cache.layers, anchor['layers']):
        if (saved['type'] != type(layer).__name__ or saved['sliding'] != bool(layer.is_sliding)
                or saved['dtype'] != str(layer.keys.dtype) or saved['device'] != str(layer.keys.device)
                or any(tuple(value.shape[:2]) != saved[shape][:2]
                       or value.shape[-1] != saved[shape][-1]
                       for value, shape in ((layer.keys, 'key_shape'), (layer.values, 'value_shape')))):
            raise TrainingError('Session rewind cache layer metadata changed')
        if layer.is_sliding:
            if (saved.get('window') != layer.sliding_window
                    or any(saved.get(name) is None or tuple(saved[name].shape) != saved[shape]
                           or saved[name].dtype != layer.keys.dtype or saved[name].device != layer.keys.device
                           for name, shape in (('keys', 'key_shape'), ('values', 'value_shape')))):
                raise TrainingError('Session rewind has no exact owned sliding prefix tail')


def _restore_session_prefix(cache, anchor, raw_prefix, policy_id, epoch):
    """Consume a B1 anchor after stop-time extraction, retaining full backing views."""
    _validate_session_anchor(cache, anchor, raw_prefix, policy_id, epoch)
    length = len(anchor['prefix'])
    for layer, saved in zip(cache.layers, anchor['layers']):
        if layer.is_sliding:
            layer.keys, layer.values = saved.pop('keys'), saved.pop('values')
            layer.cumulative_length = length
        else:
            # DynamicLayer.crop is a view, not an immediate backing-storage
            # release. The next update/cat materializes the canonical suffix.
            layer.crop(length)
    return _single_session_cache(cache, length)


@contextmanager
def _rewind_request_anchors(requests, enabled):
    try:
        yield
    finally:
        if enabled:
            # Stop-time put pops ownership into the store. Anything remaining
            # belongs to an unfinished/failed request and must be released.
            for row in requests:
                _release_session_anchor(row.pop('_prefix_rewind_anchor', None))


def _pack_session_caches(caches, lengths, torch):
    """Consume B1 caches into one right-aligned, physically padded cache."""
    width = max(lengths)
    if len(caches) == 1:
        # B1 needs neither concatenation nor padding. Transfer the cache object
        # itself; a single-element torch.cat would copy every layer's full KV.
        return _single_session_cache(caches[0], width), width
    if len({len(cache.layers) for cache in caches}) != 1:
        raise TrainingError('Session caches have different layer inventories')
    layers = []
    for index in range(len(caches[0].layers)):
        originals = [cache.layers[index] for cache in caches]
        for layer, length in zip(originals, lengths):
            _check_cache_layer(layer)
            if int(layer.get_seq_length()) != length or layer.keys.shape[0] != 1:
                raise TrainingError('Cannot pack a session with an inconsistent logical KV length')
        first = originals[0]
        if any(type(layer) is not type(first) or layer.keys.dtype != first.keys.dtype
                or layer.keys.device != first.keys.device for layer in originals):
            raise TrainingError('Cannot pack differing cache layer implementations/devices/dtypes')
        sliding = bool(first.is_sliding)
        if sliding and any(layer.sliding_window != first.sliding_window for layer in originals):
            raise TrainingError('Cannot pack different sliding windows')
        target = min(width, first.sliding_window - 1) if sliding else width
        values = []
        for attribute in ('keys', 'values'):
            rows = []
            for layer, length in zip(originals, lengths):
                value = getattr(layer, attribute)
                expected = min(length, layer.sliding_window - 1) if sliding else length
                if value.shape[-2] != expected:
                    raise TrainingError('Session KV retained tensor length differs from its cache type')
                padding = target - value.shape[-2]
                if padding:
                    shape = list(value.shape)
                    shape[-2] = padding
                    value = torch.cat((value.new_zeros(shape), value), dim=-2)
                rows.append(value)
            values.append(torch.cat(rows, dim=0))
        packed = copy.copy(first)
        packed.keys, packed.values = values
        if sliding:
            packed.cumulative_length = width
        layers.append(packed)
        # Ownership moved into the packed cache. Release source layer tensors
        # incrementally to avoid retaining two complete 32K caches at once.
        for layer in originals:
            layer.keys = layer.values = None
    return _cache_from_layers(layers), width


def _extract_session_row(cache, row_index, *, left_padding, logical_length):
    """Snapshot a stopping row before later dummy updates lose sliding keys."""
    layers = []
    for original in cache.layers:
        _check_cache_layer(original)
        layer = copy.copy(original)
        if original.is_sliding:
            keep = min(logical_length, original.sliding_window - 1)
            layer.keys = original.keys[row_index:row_index + 1, :, -keep:, :].clone()
            layer.values = original.values[row_index:row_index + 1, :, -keep:, :].clone()
            layer.cumulative_length = logical_length
        else:
            end = left_padding + logical_length
            if end > original.keys.shape[-2]:
                raise TrainingError('Stopping session claims tokens not present in its packed cache')
            layer.keys = original.keys[row_index:row_index + 1, :, left_padding:end, :].clone()
            layer.values = original.values[row_index:row_index + 1, :, left_padding:end, :].clone()
        layers.append(layer)
    result = _cache_from_layers(layers)
    if int(result.get_seq_length()) != logical_length:
        raise TrainingError('Extracted session cache did not preserve its logical length')
    return result


def _prefill_sessions(policy, requests, store, *, chunk_size=256):
    torch = policy.torch
    caches, logits, reports = [], [], []
    started = time.monotonic()
    for row in requests:
        cache, reused, reason = store.take(row['session_id'], policy.identity['policy_id'],
            row['prompt'], row.get('_session_epoch', 0))
        if cache is None:
            cache = _fresh_session_cache(policy)
        before = time.monotonic()
        prompt = row['prompt']
        for start in range(reused, len(prompt), chunk_size):
            end = min(start + chunk_size, len(prompt))
            ids = torch.tensor([prompt[start:end]], dtype=torch.long, device=policy.device)
            output = policy.model(input_ids=ids,
                attention_mask=torch.ones((1, end), dtype=torch.long, device=policy.device),
                position_ids=torch.arange(start, end, device=policy.device)[None, :],
                past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
            cache = output.past_key_values
            if cache is None:
                raise TrainingError('Incremental session prefill returned no requested cache')
            if end == len(prompt):
                logits.append(output.logits[:, -1, :])
            del output, ids
        for device in policy.cuda_devices:
            torch.cuda.synchronize(device)
        if store.session_prefix_rewind and row['session_id'] is not None:
            row['_prefix_rewind_anchor'] = _capture_session_prefix(cache, prompt,
                policy.identity['policy_id'], row.get('_session_epoch', 0))
        caches.append(cache)
        reports.append(dict(enabled=True, hit=reused > 0, reused_tokens=reused,
            new_prefill_tokens=len(prompt) - reused, reset_reason=reason,
            prefill_seconds=time.monotonic() - before))
        if store.session_prefix_rewind:
            reports[-1].update(reuse_kind=('prefix-rewind' if reason == 'session-prefix-rewind'
                else 'strict-extension' if reused else 'miss'),
                reset_reason=None if reused else reason,
                prefix_anchor_tensor_bytes=row.get('_prefix_rewind_anchor', {}).get('tensor_bytes', 0))
    lengths = [len(row['prompt']) for row in requests]
    past, width = _pack_session_caches(caches, lengths, torch)
    del caches
    attention = torch.tensor([[0] * (width - length) + [1] * length for length in lengths],
                             dtype=torch.long, device=policy.device)
    for device in policy.cuda_devices:
        torch.cuda.synchronize(device)
    return torch.cat(logits, dim=0), past, attention, width, time.monotonic() - started, reports


def _prefill(policy, requests, pad_token, *, chunk_size=256):
    torch = policy.torch
    width = max(len(row['prompt']) for row in requests)
    rows = [[pad_token] * (width - len(row['prompt'])) + row['prompt'] for row in requests]
    masks = [[0] * (width - len(row['prompt'])) + [1] * len(row['prompt']) for row in requests]
    ids = torch.tensor(rows, dtype=torch.long, device=policy.device)
    attention = torch.tensor(masks, dtype=torch.long, device=policy.device)
    positions = attention.cumsum(-1) - 1
    positions.masked_fill_(attention == 0, 0)
    past = None
    started = time.monotonic()
    for start in range(0, width, chunk_size):
        end = min(start + chunk_size, width)
        output = policy.model(input_ids=ids[:, start:end],
            attention_mask=attention[:, :end], position_ids=positions[:, start:end],
            past_key_values=past, use_cache=True, logits_to_keep=1, return_dict=True)
        past = output.past_key_values
        if past is None:
            raise TrainingError('Batched prefill returned no requested KV cache')
        if end == width:
            logits = output.logits[:, -1, :]
        del output
    for device in policy.cuda_devices:
        torch.cuda.synchronize(device)
    return logits, past, attention, width, time.monotonic() - started


def sample_batch(policy, requests, *, max_batch_tokens=32768, batch_index=0, session_cache=None):
    """Run actual shared tensor forwards; return one exact action per request."""
    if (not requests or _batch_token_slots(requests) > max_batch_tokens
            or _batch_span(requests) > policy.cfg['max_sequence_length']):
        raise TrainingError('Batched action exceeds its declared padded KV token budget')
    rewind_enabled = session_cache is not None and session_cache.session_prefix_rewind
    profile = (REWIND_SESSION_BATCHED_EXECUTION_PROFILE if rewind_enabled else
               SESSION_BATCHED_EXECUTION_PROFILE if session_cache is not None else BATCHED_EXECUTION_PROFILE)
    if policy.probability_profile != profile:
        raise TrainingError('Batched sampler must retain its distinct actor probability profile')
    sessions = [row['session_id'] for row in requests if row['session_id'] is not None]
    if session_cache is not None and len(sessions) != len(set(sessions)):
        raise TrainingError('One batch cannot contain two outstanding actions for the same session')
    torch = policy.torch
    pad_token = policy.tokenizer.pad_token_id
    if pad_token is None:
        pad_token = policy.tokenizer.eos_token_id
    if type(pad_token) is not int or pad_token < 0:
        raise TrainingError('Batched sampler needs a valid masked padding token')
    started = time.monotonic()
    for device in policy.cuda_devices:
        torch.cuda.reset_peak_memory_stats(device)
    policy.model.eval()
    with _rewind_request_anchors(requests, rewind_enabled), torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        if session_cache is None:
            next_logits, past, attention, width, prefill_seconds = _prefill(policy, requests, pad_token)
            prefill_reports = [dict(enabled=False, hit=False, reused_tokens=0,
                new_prefill_tokens=len(row['prompt']), reset_reason='fresh-private-batched-action-cache',
                prefill_seconds=prefill_seconds) for row in requests]
        else:
            next_logits, past, attention, width, prefill_seconds, prefill_reports = _prefill_sessions(
                policy, requests, session_cache)
        cache_layer_types = dict(Counter(type(layer).__name__ for layer in getattr(past, 'layers', [])))
        decode_started = time.monotonic()
        steps = 0
        for index in range(max(row['limit'] for row in requests)):
            if index:
                active = [not row['done'] for row in requests]
                ids = torch.tensor([[row['generated'][-1] if alive else pad_token]
                    for row, alive in zip(requests, active)], dtype=torch.long, device=policy.device)
                suffix_mask = torch.tensor([[int(alive)] for alive in active],
                    dtype=torch.long, device=policy.device)
                attention = torch.cat((attention, suffix_mask), dim=1)
                positions = torch.tensor([[len(row['prompt']) + index - 1 if alive else 0]
                    for row, alive in zip(requests, active)], dtype=torch.long, device=policy.device)
                output = policy.model(input_ids=ids, attention_mask=attention,
                    position_ids=positions, past_key_values=past, use_cache=True,
                    logits_to_keep=1, return_dict=True)
                past = output.past_key_values
                if past is None:
                    raise TrainingError('Batched decode returned no requested KV cache')
                next_logits = output.logits[:, -1, :]
                del output, ids, suffix_mask, positions
            logprobs = torch.log_softmax(next_logits.float(), dim=-1)
            del next_logits
            selected, forced_rows = [], []
            for row, values in zip(requests, logprobs):
                forced = row['in_thought'] and row['thought_count'] >= row['thinking_budget']
                forced_rows.append(forced)
                if row['done'] or forced:
                    selected.append(torch.tensor(pad_token if row['done'] else policy.thought_close,
                        dtype=torch.long, device=values.device))
                else:
                    if row['generator'] is None:
                        row['generator'] = torch.Generator(device=values.device).manual_seed(row['seed'])
                    elif row['generator'].device != values.device:
                        raise TrainingError('Batched autoregressive logits changed devices')
                    selected.append(torch.multinomial(values.exp(), 1, generator=row['generator']).reshape(()))
            token_ids = torch.stack(selected)
            probabilities = logprobs.gather(1, token_ids[:, None]).squeeze(1)
            # All vocabulary IDs are exactly representable in FP32. One
            # transfer replaces per-request token.item + logprob.item calls.
            observations = torch.stack((token_ids.float(), probabilities), dim=1).cpu().tolist()
            del logprobs, token_ids, probabilities, selected
            steps += 1
            for row_index, (row, observed, forced) in enumerate(zip(requests, observations, forced_rows)):
                if row['done']:
                    continue
                token, old = int(observed[0]), float(observed[1])
                row['generated'].append(token)
                row['old'].append(0.0 if forced else old)
                row['mask'].append(0 if forced else 1)
                if token == policy.thought_open:
                    row['in_thought'] = True
                elif token == policy.thought_close:
                    row['in_thought'] = False
                elif row['in_thought']:
                    row['thought_count'] += 1
                if token in row['stop'] and not row['in_thought']:
                    row['reason'] = 'tool_calls' if token == policy.tool_close else 'stop'
                    row['done'] = True
                elif len(row['generated']) >= row['limit']:
                    row['done'] = True
                if row['done'] and session_cache is not None:
                    # Stop-time extraction is essential: later masked dummy
                    # updates would evict valid states from sliding layers.
                    prefix = tuple(row['prompt']) + tuple(row['generated'][:-1])
                    if len(requests) == 1:
                        # No other row can continue with masked dummy tokens.
                        # Decode stops below, and reference scoring has its own
                        # fresh cache, so sole ownership can move to the store.
                        if row_index != 0 or width != len(row['prompt']):
                            raise TrainingError('B1 cache retirement cannot contain padding')
                        retained = _single_session_cache(past, len(prefix))
                    else:
                        retained = _extract_session_row(past, row_index,
                            left_padding=width - len(row['prompt']), logical_length=len(prefix))
                    if rewind_enabled:
                        anchor = row.pop('_prefix_rewind_anchor', None)
                        saved = False
                        try:
                            saved = session_cache.put(row['session_id'], policy.identity['policy_id'],
                                prefix, retained, row.get('_session_epoch', 0), anchor=anchor)
                        finally:
                            if not saved:
                                _release_session_anchor(anchor)
                            del anchor
                    else:
                        saved = session_cache.put(row['session_id'], policy.identity['policy_id'],
                            prefix, retained, row.get('_session_epoch', 0))
                    prefill_reports[row_index].update(retained_tokens_after=len(prefix),
                                                     retained_after_action=saved)
                    del retained
            if all(row['done'] for row in requests):
                break
        for device in policy.cuda_devices:
            torch.cuda.synchronize(device)
        decode_seconds = time.monotonic() - decode_started
        del past, attention
        generation_seconds = time.monotonic() - started
        references, reference_seconds = [], []
        for row in requests:
            reference_started = time.monotonic()
            values = None
            if row['compute_reference']:
                ids = torch.tensor([row['prompt'] + row['generated']], dtype=torch.long, device=policy.device)
                selected = torch.tensor([[0] * len(row['prompt']) + row['mask']],
                    dtype=torch.bool, device=policy.device)
                with frozen_base_reference(policy.model):
                    scored = cached_selected_logprobs(policy.model.get_base_model(), ids, selected,
                        prefill_chunk_size=256, projection_chunk_size=16)
                actual = iter(scored.float().cpu().tolist())
                values = [next(actual) if keep else 0.0 for keep in row['mask']]
                del ids, selected, scored
            references.append(values)
            reference_seconds.append(time.monotonic() - reference_started)
        for device in policy.cuda_devices:
            torch.cuda.synchronize(device)
    batch_seconds = time.monotonic() - started
    memory = [dict(device=str(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device)) for device in policy.cuda_devices]
    results = []
    for row, reference, ref_seconds, prefill_report in zip(requests, references, reference_seconds, prefill_reports):
        generated, mask = row['generated'], row['mask']
        batching = dict(batch_index=batch_index, actual_batch_size=len(requests),
            padded_prompt_tokens=width, pad_tokens=width-len(row['prompt']),
            padded_kv_token_budget=_batch_token_slots(requests), session_id=row['session_id'],
            prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
            generation_seconds=generation_seconds, reference_seconds=ref_seconds,
            all_reference_seconds=sum(reference_seconds), batch_seconds=batch_seconds,
            batch_decode_steps=steps, batch_generated_tokens=sum(len(value['generated']) for value in requests),
            timing_unit='Batch wall times overlap across its returned action rows; do not sum rows as GPU wall time')
        if session_cache is not None:
            batching.update(session_cache_hit=prefill_report['hit'],
                reused_session_tokens=prefill_report['reused_tokens'],
                new_session_prefill_tokens=prefill_report['new_prefill_tokens'],
                session_cache_transfer=('single-row-ownership' if len(requests) == 1
                                        else 'packed-row-stop-time-copy'),
                session_cache_stats=session_cache.stats())
        results.append(dict(prompt_token_ids=row['prompt'], completion_token_ids=generated,
            behavior_logprobs=row['old'], reference_logprobs=reference, loss_mask=mask,
            raw_text=policy.tokenizer.decode(generated, skip_special_tokens=False,
                clean_up_tokenization_spaces=False),
            finish_reason=row['reason'], seed=row['seed'], thinking_budget=row['thinking_budget'],
            requested_max_new_tokens=row['max_new_tokens'], effective_max_new_tokens=row['limit'],
            thought_tokens=row['thought_count'], forced_control_tokens=mask.count(0),
            sampling=dict(temperature=1.0, top_p=1.0, top_k=0), prefill_chunk_size=256,
            reference_scoring='actual-independent-unpadded-chunked-backbone-selected-head'
                if row['compute_reference'] else 'not-computed-nonfocal-action',
            probability_execution_sha256=digest(policy.probability_profile),
            reference_forward_performed=row['compute_reference'], seconds=batch_seconds,
            prompt_tokens=len(row['prompt']), completion_tokens=len(generated),
            execution=dict(policy.execution, batching=batching,
                reference_scope='current-sampled-focal-action' if row['compute_reference'] else 'not-computed-nonfocal-action'),
            batching=batching, sampling_device=str(row['generator'].device) if row['generator'] is not None else None,
            prefill_cache=prefill_report, memory_by_device=memory,
            peak_memory_scope='maximum_of_batch_model_cuda_devices',
            peak_allocated_bytes=max(value['peak_allocated_bytes'] for value in memory),
            peak_reserved_bytes=max(value['peak_reserved_bytes'] for value in memory), **policy.identity))
    print(json.dumps(dict(phase='batched_action_completed', batch_index=batch_index,
        actual_batch_size=len(requests), padded_prompt_tokens=width,
        cache_layer_types=cache_layer_types,
        generated_tokens=sum(len(row['generated']) for row in requests),
        prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
        reference_seconds=sum(reference_seconds), batch_seconds=batch_seconds,
        **(dict(session_cache_hits=sum(int(report['hit']) for report in prefill_reports),
            reused_session_tokens=sum(report['reused_tokens'] for report in prefill_reports),
            new_session_prefill_tokens=sum(report['new_prefill_tokens'] for report in prefill_reports),
            session_cache_stats=session_cache.stats()) if session_cache is not None else {}))), flush=True)
    return results


@dataclass
class _Queued:
    row: dict
    future: object
    enqueued_at: float


class BatchedPolicyService:
    """Async request queue with exactly one owner of the live CUDA model."""
    def __init__(self, policy, *, max_batch_size=2, wait_ms=50, max_batch_tokens=32768,
                 session_cache_tokens=0, session_prefix_rewind=False):
        if type(wait_ms) not in (int, float) or not math.isfinite(wait_ms) or wait_ms < 0:
            raise TrainingError('Batch collection delay must be nonnegative')
        self.policy = policy
        self.max_batch_size, self.max_batch_tokens = max_batch_size, max_batch_tokens
        self.wait_seconds = wait_ms / 1000
        configure_batched_policy(policy, max_batch_size=max_batch_size,
                                 max_batch_tokens=max_batch_tokens,
                                 session_cache_tokens=session_cache_tokens,
                                 session_prefix_rewind=session_prefix_rewind)
        self._session_cache = (SessionCacheStore(session_cache_tokens,
            session_prefix_rewind=session_prefix_rewind) if session_cache_tokens else None)
        self._queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='vfarl-batch-owner')
        self._task, self._failure, self._closed = None, None, False
        self._backlog = deque()
        self.batch_histogram = Counter()
        self.batch_count = 0

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def clear_generation_cache(self, *, reason='explicit-reset'):
        # Legacy callbacks have no session ID and may arrive while a different
        # agent is generating. Retirement must use release_session(session_id).
        return None

    def release_session(self, session_id):
        if self._session_cache is not None:
            self._session_cache.release(session_id)

    def sample(self, *args, **kwargs):
        raise TrainingError('Batched policy service requires await sample_async(...)')

    async def sample_async(self, prompt_token_ids, **kwargs):
        if self._closed:
            raise TrainingError('Batched policy service is closed')
        if self._failure is not None:
            raise TrainingError('The batch owner failed; preserve prior behavior') from self._failure
        row = _request(self.policy, prompt_token_ids, **kwargs)
        if self._session_cache is not None:
            row['_session_epoch'] = self._session_cache.epoch(row['session_id'])
        if _batch_token_slots([row]) > self.max_batch_tokens:
            raise TrainingError('One complete action exceeds the declared batch KV budget')
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._queue.put_nowait(_Queued(row, future, time.monotonic()))
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        try:
            return await future
        except asyncio.CancelledError:
            if (self._session_cache is not None and self._session_cache.session_prefix_rewind
                    and row['session_id'] is not None):
                # Retire an already-dispatched request too: its eventual put
                # must not resurrect the caller's cancelled prefix backup.
                self._session_cache.release(row['session_id'])
            raise

    async def _run(self):
        closing = False
        while not closing:
            first = self._backlog.popleft() if self._backlog else await self._queue.get()
            if first is None:
                self._queue.task_done()
                break
            if first.future.cancelled():
                self._queue.task_done()
                continue
            batch = [first]
            deadline = time.monotonic() + self.wait_seconds
            while len(batch) < self.max_batch_size:
                try:
                    if self._backlog:
                        item = self._backlog.popleft()
                    elif not self._queue.empty():
                        item = self._queue.get_nowait()
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        item = await asyncio.wait_for(self._queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    self._queue.task_done()
                    closing = True
                    break
                if item.future.cancelled():
                    self._queue.task_done()
                    continue
                candidate = [value.row for value in batch] + [item.row]
                if (_batch_token_slots(candidate) > self.max_batch_tokens
                        or _batch_span(candidate) > self.policy.cfg['max_sequence_length']):
                    self._backlog.appendleft(item)
                    break
                batch.append(item)
            dispatched_at = time.monotonic()
            self.batch_count += 1
            self.batch_histogram[len(batch)] += 1
            try:
                loop = asyncio.get_running_loop()
                values = await loop.run_in_executor(self._executor,
                    lambda: sample_batch(self.policy, [value.row for value in batch],
                        max_batch_tokens=self.max_batch_tokens, batch_index=self.batch_count,
                        session_cache=self._session_cache))
                if len(values) != len(batch):
                    raise TrainingError('Batch owner returned a different number of actions')
                for queued, value in zip(batch, values):
                    value['batching']['queue_wait_seconds'] = dispatched_at - queued.enqueued_at
                    value['batching']['request_elapsed_seconds'] = time.monotonic() - queued.enqueued_at
                    if not queued.future.done():
                        queued.future.set_result(value)
            except BaseException as error:
                self._failure = error
                if self._session_cache is not None and self._session_cache.session_prefix_rewind:
                    self._session_cache.clear()
                for queued in batch:
                    if not queued.future.done():
                        queued.future.set_exception(error)
                while self._backlog or not self._queue.empty():
                    queued = self._backlog.popleft() if self._backlog else self._queue.get_nowait()
                    if queued is not None and not queued.future.done():
                        queued.future.set_exception(error)
                    self._queue.task_done()
                return
            finally:
                for _ in batch:
                    self._queue.task_done()

    async def drain(self):
        await self._queue.join()
        if self._failure is not None:
            raise TrainingError('The batch owner failed; preserve prior behavior') from self._failure

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        if self._task is not None and not self._task.done():
            self._queue.put_nowait(None)
            await self._task
        self._executor.shutdown(wait=True, cancel_futures=False)
        cache_stats = self._session_cache.stats() if self._session_cache is not None else None
        if self._session_cache is not None:
            self._session_cache.clear()
        print(json.dumps(dict(phase='batched_service_closed', batches=self.batch_count,
            batch_histogram=dict(self.batch_histogram), session_cache_before_clear=cache_stats)), flush=True)
