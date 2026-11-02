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


_CAPACITY_LAYER = None


def enable_capacity_cache(cache, increment, capacity_limit, counters=None, *, sliding=True):
    """Grow full-attention KV storage in blocks, retaining native mask semantics.

    Sliding storage can use a mirrored ring with native chronological views.
    Only this generation owner mutates caches; snapshots/membership helpers
    reset backing storage when replacing tensors. No cache enters training.
    """
    global _CAPACITY_LAYER
    increment, capacity_limit = int(increment), int(capacity_limit)
    if increment < 0 or capacity_limit < 1:
        raise ValueError("KV capacity increment must be nonnegative and limit positive")
    if not increment:
        return cache
    if _CAPACITY_LAYER is None:
        from transformers.cache_utils import DynamicLayer

        class CapacityDynamicLayer(DynamicLayer):
            def set_storage(self, keys, values, length=None):
                self._key_storage, self._value_storage = keys, values
                self._length = (keys.shape[-2] if keys.ndim == 4 else 0) if length is None else int(length)
                if length is not None and (keys.ndim != 4 or values.ndim != 4 or
                                           not 0 <= self._length <= min(keys.shape[-2], values.shape[-2])):
                    raise ValueError("Logical KV length must fit its backing storage")
                self.keys = keys[..., :self._length, :] if keys.ndim == 4 else keys
                self.values = values[..., :self._length, :] if values.ndim == 4 else values

            def update(self, key_states, value_states, *args, **kwargs):
                if not self.is_initialized:
                    self.lazy_initialization(key_states, value_states)
                    self.set_storage(self.keys, self.values)
                old_length = self._length
                new_length = old_length + key_states.shape[-2]
                if new_length > self.capacity_limit:
                    raise ValueError("Complete generation KV exceeds its declared context")
                capacity = self._key_storage.shape[-2] if self._key_storage.ndim == 4 else 0
                if new_length > capacity:
                    rounded = ((new_length + self.capacity_increment - 1) // self.capacity_increment) * self.capacity_increment
                    capacity = min(self.capacity_limit, max(capacity * 2, rounded))
                    key_shape, value_shape = list(key_states.shape), list(value_states.shape)
                    key_shape[-2] = value_shape[-2] = capacity
                    new_keys, new_values = key_states.new_empty(key_shape), value_states.new_empty(value_shape)
                    if old_length:
                        new_keys[..., :old_length, :].copy_(self.keys)
                        new_values[..., :old_length, :].copy_(self.values)
                        self.prefix_elements_copied += self.keys.numel() + self.values.numel()
                        if self.capacity_counters is not None:
                            self.capacity_counters["prefix_elements_copied"] += self.keys.numel() + self.values.numel()
                    self._key_storage, self._value_storage = new_keys, new_values
                    self.capacity_growths += 1
                    if self.capacity_counters is not None:
                        self.capacity_counters["growths"] += 1
                self._key_storage[..., old_length:new_length, :].copy_(key_states)
                self._value_storage[..., old_length:new_length, :].copy_(value_states)
                self._length = new_length
                if self.capacity_counters is not None:
                    self.capacity_counters["appended_elements"] += key_states.numel() + value_states.numel()
                self.keys = self._key_storage[..., :new_length, :]
                self.values = self._value_storage[..., :new_length, :]
                return self.keys, self.values

            def crop(self, tokens_to_remove):
                super().crop(tokens_to_remove)
                self._length = self.keys.shape[-2]

        _CAPACITY_LAYER = CapacityDynamicLayer
    converted = []
    for layer in cache.layers:
        if type(layer).__name__ == "DynamicLayer":
            replacement = _CAPACITY_LAYER()
            replacement.__dict__.update(layer.__dict__)
            replacement.capacity_increment, replacement.capacity_limit = increment, capacity_limit
            replacement.capacity_growths = replacement.prefix_elements_copied = 0
            replacement.capacity_counters = counters
            if layer.is_initialized:
                replacement.set_storage(layer.keys, layer.values)
            converted.append(replacement)
        elif type(layer).__name__ == "DynamicSlidingWindowLayer" and sliding:
            from .sliding_cache import enable_sliding_storage
            converted.append(enable_sliding_storage(layer, increment, capacity_limit, counters))
        elif type(layer).__name__ in {"DynamicSlidingWindowLayer", "CapacityDynamicLayer", "RingSlidingWindowLayer"}:
            converted.append(layer)
        else:
            raise TypeError("Capacity caching supports only native Gemma dynamic/full-sliding layers")
    cache.layers = converted
    return cache


def _layers(cache):
    if type(cache).__name__ != "DynamicCache":
        raise TypeError(f"Expected native DynamicCache, found {type(cache).__name__}")
    if getattr(cache, "offloading", False):
        raise ValueError("The batching cache does not support offloaded KV state")
    layers = getattr(cache, "layers", None)
    if not layers:
        raise ValueError("Expected an initialized native DynamicCache")
    for layer in layers:
        if type(layer).__name__ not in {"DynamicLayer", "DynamicSlidingWindowLayer", "CapacityDynamicLayer", "RingSlidingWindowLayer"}:
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
    if type(result).__name__ == "RingSlidingWindowLayer":
        result.set_storage(keys, values, length=length)
    elif result.is_sliding:
        result.cumulative_length = int(length)
    elif type(result).__name__ == "CapacityDynamicLayer":
        result.set_storage(keys, values, length=length)
    return result


def clone_cache(cache):
    """Clone tensors without changing per-layer devices or sliding counters."""
    result = copy.copy(cache)
    result.layers = [
        _replace_layer(layer, layer.keys.detach().clone(), layer.values.detach().clone(),
                       int(layer.get_seq_length()))
        for layer in _layers(cache)
    ]
    result._membership_view = False
    return result


def merge_caches(caches, lengths):
    """Left-pad unpadded single-row caches for one shared decode forward.

    Pass a matching left-padded attention mask and each row's true position_ids
    to the model. The common physical length drives causal/sliding mask offsets;
    the per-row position_ids preserve the token positions used by RoPE.

    Full-capacity layers write directly into final buffers with at most one
    capacity increment of headroom. Membership counters count K/V allocations,
    allocated elements (including headroom), copied source elements and zeroed
    padding separately; update-time growth/copy counters retain their meaning.
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
    result._membership_view = False
    for index, template in enumerate(caches[0].layers):
        layers = [cache.layers[index] for cache in caches]
        if any(type(layer) is not type(template) or
               getattr(layer, "sliding_window", None) != getattr(template, "sliding_window", None)
               for layer in layers):
            raise ValueError("Cannot batch different attention layer types/windows")
        stored = _stored_length(template, physical_length)
        capacity_layer = type(template).__name__ == "CapacityDynamicLayer"
        if capacity_layer:
            increment, limit = template.capacity_increment, template.capacity_limit
            if any((layer.capacity_increment, layer.capacity_limit) != (increment, limit) for layer in layers):
                raise ValueError("Cannot batch different KV capacity settings")
            if stored > limit:
                raise ValueError("Complete generation KV exceeds its declared context")
            # Even an exactly aligned logical length has room for its next
            # token, unless the declared context is already completely full.
            capacity = min(limit, ((min(stored + 1, limit) + increment - 1) // increment) * increment)
        tensors = []
        for name in ("keys", "values"):
            reference = getattr(template, name)
            if capacity_layer:
                storage = reference.new_empty((len(layers), reference.shape[1], capacity, reference.shape[3]))
                copied_elements = padding_elements = 0
            rows = []
            for row, layer in enumerate(layers):
                tensor = getattr(layer, name)
                if (tensor.device != reference.device or tensor.dtype != reference.dtype or
                        tensor.shape[1] != reference.shape[1] or tensor.shape[3] != reference.shape[3]):
                    raise ValueError("Per-layer KV head shapes, dtype and devices must agree")
                padding = stored - tensor.shape[-2]
                if capacity_layer:
                    destination = storage[row:row + 1, :, :stored, :]
                    if padding:
                        destination[..., :padding, :].zero_()
                        padding_elements += destination[..., :padding, :].numel()
                    destination[..., padding:, :].copy_(tensor)
                    copied_elements += tensor.numel()
                    continue
                if padding:
                    zeros = tensor.new_zeros((1, tensor.shape[1], padding, tensor.shape[3]))
                    tensor = torch.cat((zeros, tensor), dim=-2)
                rows.append(tensor)
            if capacity_layer:
                tensors.append(storage)
                counters = template.capacity_counters
                if counters is not None:
                    for key, value in (("membership_buffer_allocations", 1),
                                       ("membership_allocated_elements", storage.numel()),
                                       ("membership_copied_elements", copied_elements),
                                       ("membership_padding_elements", padding_elements)):
                        counters[key] = counters.get(key, 0) + value
            else:
                tensors.append(torch.cat(rows, dim=0))
        result.layers.append(_replace_layer(template, *tensors, physical_length))
    return result


def select_cache(cache, row, actual_length, physical_length=None, *, copy_tensors=True):
    """Remove batch padding; use a temporary view only for immediate rebuilding.

    actual_length includes every token already consumed by forward, and excludes
    the most recently sampled token until a later forward has consumed it.
    Default copies own their tensors. Views must not survive a membership
    transition; merge/clone or PrefixCache.put materializes their ownership.
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
        keys = layer.keys[row:row + 1, :, -stored:, :].detach()
        values = layer.values[row:row + 1, :, -stored:, :].detach()
        if copy_tensors:
            keys, values = keys.clone(), values.clone()
        result.layers.append(_replace_layer(layer, keys, values, actual_length))
    result._membership_view = not copy_tensors
    return result


def _cache_bytes(cache):
    return sum(tensor.numel() * tensor.element_size()
               for layer in _layers(cache) for tensor in (layer.keys, layer.values))


def _prefix_bytes_by_device(cache, next_logits):
    """Logical snapshot storage; cloning removes any backing-buffer headroom."""
    sizes = {}
    tensors = [tensor for layer in _layers(cache) for tensor in (layer.keys, layer.values)]
    for tensor in [*tensors, next_logits]:
        device = str(tensor.device)
        sizes[device] = sizes.get(device, 0) + tensor.numel() * tensor.element_size()
    return sizes


@dataclass(frozen=True)
class PrefixEntry:
    cache: Any
    cached_length: int
    next_logits: Any


@dataclass
class _StoredPrefix:
    entry: PrefixEntry
    bytes_by_device: dict
    owners: set
    anonymous: bool = False

    @property
    def size(self):
        return sum(self.bytes_by_device.values())


class PrefixCache:
    """LRU prompt snapshots shared only within one policy version.

    A hit reuses a complete stored prefix. It never rolls a sliding layer back
    to an earlier point, because those old KV states may already be discarded.
    Exact prompt hits use next_logits; extensions forward only their new suffix.
    Each rollout owns at most its latest cached prompt; identical prompts share
    one stored snapshot. Calls without an owner retain the legacy key-based LRU.
    Limits bound resident snapshots, not the independent clone returned by get;
    the generation scheduler must reserve clone/active-KV space before calling.
    Caller access is serialized by that scheduler.
    """

    def __init__(self, max_entries=8, max_bytes=2 * 1024**3):
        self.max_entries, self.max_bytes = int(max_entries), int(max_bytes)
        if self.max_entries < 0 or self.max_bytes < 0:
            raise ValueError("Prefix cache limits must be nonnegative")
        self._entries = OrderedDict()
        self._owners = {}
        self._version = None
        self.device_limits = None
        self.used_bytes = 0
        self.bytes_by_device = {}
        self.hits = self.misses = self.evictions = 0
        self.replacements = self.rejections = self.budget_evictions = 0
        self.peak_entries = self.peak_bytes = 0

    def clear(self):
        self._entries.clear()
        self._owners.clear()
        self.used_bytes = 0
        self.bytes_by_device.clear()
        self._version = None

    def _remove(self, key, *, eviction=False, budget=False):
        record = self._entries.pop(key)
        self.used_bytes -= record.size
        for device, size in record.bytes_by_device.items():
            remaining = self.bytes_by_device[device] - size
            if remaining:
                self.bytes_by_device[device] = remaining
            else:
                del self.bytes_by_device[device]
        for owner in record.owners:
            if self._owners.get(owner) == key:
                del self._owners[owner]
        self.evictions += int(eviction)
        self.budget_evictions += int(budget)

    def _over_limit(self, additional=None, extra_entries=0):
        additional = additional or {}
        return (len(self._entries) + extra_entries > self.max_entries or
                self.used_bytes + sum(additional.values()) > self.max_bytes or
                any(self.bytes_by_device.get(device, 0) + additional.get(device, 0) > limit
                    for device, limit in (self.device_limits or {}).items()))

    def _fits_with_other_owners(self, owner, sizes):
        """Would the candidate fit while retaining every other rollout's entry?"""
        count, resident = 1, dict(sizes)
        for record in self._entries.values():
            if any(other != owner for other in record.owners):
                count += 1
                for device, size in record.bytes_by_device.items():
                    resident[device] = resident.get(device, 0) + size
        return (count <= self.max_entries and sum(resident.values()) <= self.max_bytes and
                all(resident.get(device, 0) <= limit
                    for device, limit in (self.device_limits or {}).items()))

    def set_limits(self, max_bytes, device_limits=None):
        """Apply current resident-memory allowances and trim immediately by LRU.

        Device limits are optional caps on named devices; omitted devices have
        only the global limit. Passing None removes previous per-device caps.
        A zero global budget disables storage, including zero-sized snapshots.
        """
        max_bytes = int(max_bytes)
        limits = None if device_limits is None else {str(k): int(v) for k, v in device_limits.items()}
        if max_bytes < 0 or any(value < 0 for value in (limits or {}).values()):
            raise ValueError("Prefix cache limits must be nonnegative")
        self.max_bytes, self.device_limits = max_bytes, limits
        while self._entries and (not self.max_bytes or self._over_limit()):
            self._remove(next(iter(self._entries)), eviction=True, budget=True)

    def drop_owner(self, owner):
        """Release a completed rollout without deleting another owner's copy."""
        if owner is None:
            return False
        key = self._owners.pop(owner, None)
        if key is None:
            return False
        record = self._entries[key]
        record.owners.remove(owner)
        if not record.owners and not record.anonymous:
            self._remove(key)
        return True

    def _bind_owner(self, key, owner):
        if owner is None:
            self._entries[key].anonymous = True
            return
        previous = self._owners.get(owner)
        if previous == key:
            return
        if previous is not None:
            self.replacements += 1
            self.drop_owner(owner)
        self._owners[owner] = key
        self._entries[key].owners.add(owner)

    def _policy(self, version):
        if self._version != version:
            self.clear()
            self._version = version

    def get(self, prompt_ids, version, *, owner=None):
        self._policy(version)
        prompt = tuple(prompt_ids)
        matches = [key for key in self._entries if len(key) <= len(prompt) and prompt[:len(key)] == key]
        if not matches:
            self.misses += 1
            return None
        key = max(matches, key=len)
        entry = self._entries[key].entry
        result = PrefixEntry(clone_cache(entry.cache), entry.cached_length, entry.next_logits.detach().clone())
        self._entries.move_to_end(key)
        # A reused older prefix is only a starting state. Bind a new owner when
        # its complete prompt is present; a later put replaces partial reuse.
        if owner is not None and len(key) == len(prompt):
            self._bind_owner(key, owner)
        self.hits += 1
        return result

    def put(self, cache, cached_ids, version, next_logits, *, owner=None):
        """Store a complete prompt, releasing this owner's old prompt first."""
        self._policy(version)
        if not self.max_entries or not self.max_bytes:
            self.rejections += 1
            return False
        key = tuple(cached_ids)
        _validate_unpadded(cache, len(key))
        if next_logits.ndim != 1:
            raise ValueError("next_logits must be one vocabulary row for this exact cached prefix")
        sizes = _prefix_bytes_by_device(cache, next_logits)
        size = sum(sizes.values())
        if (size > self.max_bytes or
                any(sizes.get(device, 0) > limit for device, limit in (self.device_limits or {}).items())):
            self.rejections += 1
            return False
        if key in self._entries:
            self._bind_owner(key, owner)
            self._entries.move_to_end(key)
            return True
        previous = self._owners.get(owner) if owner is not None else None
        preserve_others = previous is not None and key[:len(previous)] == previous
        # A longer snapshot is optional when the old complete prefix still
        # works. Keep that prefix rather than sacrificing another rollout's
        # coverage. First admissions and incompatible prompts use ordinary LRU.
        if preserve_others and not self._fits_with_other_owners(owner, sizes):
            self.rejections += 1
            return False
        if previous is not None:
            self.replacements += 1
            self.drop_owner(owner)
        # Release superseded ownership and evict before cloning, so the old
        # owned snapshot cannot unnecessarily displace another rollout's prompt.
        while self._entries and self._over_limit(sizes, extra_entries=1):
            victim = (next(key for key, record in self._entries.items() if not record.owners)
                      if preserve_others else next(iter(self._entries)))
            self._remove(victim, eviction=True, budget=True)
        entry = PrefixEntry(clone_cache(cache), len(key), next_logits.detach().clone())
        self._entries[key] = _StoredPrefix(entry, sizes, set(), anonymous=owner is None)
        self.used_bytes += size
        self.peak_entries = max(self.peak_entries, len(self._entries))
        self.peak_bytes = max(self.peak_bytes, self.used_bytes)
        for device, amount in sizes.items():
            self.bytes_by_device[device] = self.bytes_by_device.get(device, 0) + amount
        self._bind_owner(key, owner)
        return True

    def stats(self):
        return {"entries": len(self._entries), "bytes": self.used_bytes,
                "hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "replacements": self.replacements, "rejections": self.rejections,
                "budget_evictions": self.budget_evictions,
                "peak_entries": self.peak_entries, "peak_bytes": self.peak_bytes,
                "bytes_by_device": dict(self.bytes_by_device)}
