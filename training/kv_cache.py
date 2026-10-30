"""Batched dynamic KV state and bounded exact-prefix reuse for one policy.

Gemma4 has heterogeneous full/sliding layers. Its ordinary DynamicCache keeps
each layer's own tensor shape and device; the Transformers 5.17 paged cache
does not. These helpers retain that native representation. Padding exists only
in a batched forward and is removed before a cache enters the prefix store.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import dataclass
from typing import Any


def _layers(cache):
    if type(cache).__name__ != "DynamicCache":
        raise TypeError(f"Expected native DynamicCache, found {type(cache).__name__}")
    if getattr(cache, "offloading", False):
        raise ValueError("The batching cache does not support offloaded KV state")
    layers = getattr(cache, "layers", None)
    if not layers:
        raise ValueError("Expected an initialized native DynamicCache")
    for layer in layers:
        if type(layer).__name__ not in {"DynamicLayer", "DynamicSlidingWindowLayer"}:
            raise TypeError(f"Unsupported batching cache layer: {type(layer).__name__}")
        if not layer.is_initialized or layer.keys is None or layer.values is None:
            raise ValueError("Every non-shared KV layer must have been initialized")
        if getattr(layer, "record_past", False):
            raise ValueError("Cache with uncommitted speculative history cannot be batched")
        if layer.keys.ndim != 4 or layer.values.ndim != 4:
            raise ValueError("Expected KV tensors shaped [batch, heads, tokens, head_dim]")
        if layer.is_sliding and int(layer.sliding_window) < 2:
            raise ValueError("This Gemma4 cache requires a sliding window of at least two tokens")
    return layers


def _stored_length(layer, length):
    return min(length, int(layer.sliding_window) - 1) if layer.is_sliding else length


def _validate_unpadded(cache, length):
    if length < 1:
        raise ValueError("An individual prefix cache must contain at least one token")
    for layer in _layers(cache):
        expected = _stored_length(layer, length)
        if (int(layer.get_seq_length()) != length or
                layer.keys.shape[0] != 1 or layer.values.shape[0] != 1 or
                layer.keys.shape[-2] != expected or layer.values.shape[-2] != expected):
            raise ValueError("Individual cache must contain exactly its unpadded token prefix")


def _replace_layer(layer, keys, values, length):
    result = copy.copy(layer)
    result.keys, result.values = keys, values
    result.dtype, result.device = keys.dtype, keys.device
    result.is_initialized = True
    if result.is_sliding:
        result.cumulative_length = int(length)
    return result


def clone_cache(cache):
    """Clone tensors without changing per-layer devices or sliding counters."""
    result = copy.copy(cache)
    result.layers = [
        _replace_layer(layer, layer.keys.detach().clone(), layer.values.detach().clone(),
                       int(layer.get_seq_length()))
        for layer in _layers(cache)
    ]
    return result


def merge_caches(caches, lengths):
    """Left-pad unpadded single-row caches for one shared decode forward.

    Pass a matching left-padded attention mask and each row's true position_ids
    to the model. The common physical length drives causal/sliding mask offsets;
    the per-row position_ids preserve the token positions used by RoPE.
    """
    import torch

    caches, lengths = list(caches), [int(value) for value in lengths]
    if not caches or len(caches) != len(lengths):
        raise ValueError("Need one true prefix length for every cache")
    for cache, length in zip(caches, lengths):
        _validate_unpadded(cache, length)
    if len({len(cache.layers) for cache in caches}) != 1:
        raise ValueError("Cannot batch different cache architectures")
    physical_length = max(lengths)
    result = copy.copy(caches[0])
    result.layers = []
    for index, template in enumerate(caches[0].layers):
        layers = [cache.layers[index] for cache in caches]
        if any(type(layer) is not type(template) or
               getattr(layer, "sliding_window", None) != getattr(template, "sliding_window", None)
               for layer in layers):
            raise ValueError("Cannot batch different attention layer types/windows")
        stored = _stored_length(template, physical_length)
        tensors = []
        for name in ("keys", "values"):
            reference = getattr(template, name)
            rows = []
            for layer in layers:
                tensor = getattr(layer, name)
                if (tensor.device != reference.device or tensor.dtype != reference.dtype or
                        tensor.shape[1] != reference.shape[1] or tensor.shape[3] != reference.shape[3]):
                    raise ValueError("Per-layer KV head shapes, dtype and devices must agree")
                padding = stored - tensor.shape[-2]
                if padding:
                    zeros = tensor.new_zeros((1, tensor.shape[1], padding, tensor.shape[3]))
                    tensor = torch.cat((zeros, tensor), dim=-2)
                rows.append(tensor)
            tensors.append(torch.cat(rows, dim=0))
        result.layers.append(_replace_layer(template, *tensors, physical_length))
    return result


def select_cache(cache, row, actual_length, physical_length=None):
    """Remove batch padding and detach a row, without retaining batch storage.

    actual_length includes every token already consumed by forward, and excludes
    the most recently sampled token until a later forward has consumed it.
    """
    row, actual_length = int(row), int(actual_length)
    layers = _layers(cache)
    physical_length = int(layers[0].get_seq_length()) if physical_length is None else int(physical_length)
    if not 0 < actual_length <= physical_length or row < 0:
        raise ValueError("Invalid cache row or true/physical prefix length")
    result = copy.copy(cache)
    result.layers = []
    for layer in layers:
        if (row >= layer.keys.shape[0] or row >= layer.values.shape[0] or
                int(layer.get_seq_length()) != physical_length or
                layer.keys.shape[-2] != _stored_length(layer, physical_length) or
                layer.values.shape[-2] != _stored_length(layer, physical_length)):
            raise ValueError("Batched cache shape differs from its declared physical length")
        stored = _stored_length(layer, actual_length)
        # clone(), not a view: completed rows must release the other batch rows.
        keys = layer.keys[row:row + 1, :, -stored:, :].detach().clone()
        values = layer.values[row:row + 1, :, -stored:, :].detach().clone()
        result.layers.append(_replace_layer(layer, keys, values, actual_length))
    return result


def _cache_bytes(cache):
    return sum(tensor.numel() * tensor.element_size()
               for layer in _layers(cache) for tensor in (layer.keys, layer.values))


@dataclass(frozen=True)
class PrefixEntry:
    cache: Any
    cached_length: int
    next_logits: Any


class PrefixCache:
    """LRU snapshots shared across requests only within one policy version.

    A hit reuses a complete stored prefix. It never rolls a sliding layer back
    to an earlier point, because those old KV states may already be discarded.
    Exact prompt hits use next_logits; extensions forward only their new suffix.
    Caller access is serialized by the GPU scheduler.
    """

    def __init__(self, max_entries=8, max_bytes=2 * 1024**3):
        self.max_entries, self.max_bytes = int(max_entries), int(max_bytes)
        if self.max_entries < 0 or self.max_bytes < 0:
            raise ValueError("Prefix cache limits must be nonnegative")
        self._entries = OrderedDict()
        self._version = None
        self.used_bytes = 0
        self.hits = self.misses = self.evictions = 0

    def clear(self):
        self._entries.clear()
        self.used_bytes = 0
        self._version = None

    def _policy(self, version):
        if self._version != version:
            self.clear()
            self._version = version

    def get(self, prompt_ids, version):
        self._policy(version)
        prompt = tuple(prompt_ids)
        matches = [key for key in self._entries if len(key) <= len(prompt) and prompt[:len(key)] == key]
        if not matches:
            self.misses += 1
            return None
        key = max(matches, key=len)
        entry, _size = self._entries[key]
        self._entries.move_to_end(key)
        self.hits += 1
        return PrefixEntry(clone_cache(entry.cache), entry.cached_length, entry.next_logits.detach().clone())

    def put(self, cache, cached_ids, version, next_logits):
        self._policy(version)
        if not self.max_entries or not self.max_bytes:
            return False
        key = tuple(cached_ids)
        _validate_unpadded(cache, len(key))
        if next_logits.ndim != 1:
            raise ValueError("next_logits must be one vocabulary row for this exact cached prefix")
        size = _cache_bytes(cache) + next_logits.numel() * next_logits.element_size()
        if size > self.max_bytes:
            return False
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.used_bytes -= previous[1]
            del previous
        # Evict before cloning so replacement does not temporarily duplicate all
        # existing resident cache entries on a memory-constrained GPU.
        while self._entries and (len(self._entries) >= self.max_entries or self.used_bytes + size > self.max_bytes):
            _key, (_entry, removed_size) = self._entries.popitem(last=False)
            self.used_bytes -= removed_size
            self.evictions += 1
            del _entry
        entry = PrefixEntry(clone_cache(cache), len(key), next_logits.detach().clone())
        self._entries[key] = (entry, size)
        self.used_bytes += size
        return True

    def stats(self):
        return {"entries": len(self._entries), "bytes": self.used_bytes,
                "hits": self.hits, "misses": self.misses, "evictions": self.evictions}
