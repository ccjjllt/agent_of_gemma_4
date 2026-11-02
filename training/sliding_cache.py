"""Mirrored sliding KV storage for the serialized inference cache owner.

The native Transformers mask and cumulative-position protocol is unchanged.
Returned attention views remain valid until the next mutation of this layer;
snapshots must own independent tensors before generation advances again.
"""
from __future__ import annotations


_RING_SLIDING_LAYER = None


def _ring_sliding_layer_type():
    """Keep importing cache utilities possible without importing Transformers."""
    global _RING_SLIDING_LAYER
    if _RING_SLIDING_LAYER is not None:
        return _RING_SLIDING_LAYER

    from transformers.cache_utils import DynamicSlidingWindowLayer

    class RingSlidingWindowLayer(DynamicSlidingWindowLayer):
        """Two mirrored ring halves expose chronological KV as one slice.

        A half holds at least W-1 previous tokens plus the entire query block.
        Only newly appended tokens are written twice during ordinary updates.
        On growth, the existing W-1 tail is copied into both new halves once.
        """

        def __init__(self, sliding_window, increment=256, capacity_limit=32768, counters=None):
            sliding_window, increment, capacity_limit = map(int, (sliding_window, increment, capacity_limit))
            if sliding_window < 2 or increment < 0 or capacity_limit < 1:
                raise ValueError("Ring sliding storage needs window >= 2 and nonnegative increment/positive context")
            super().__init__(sliding_window)
            self.capacity_increment = max(1, increment)
            self.capacity_limit = capacity_limit
            self.capacity_counters = counters
            for name in ("allocations", "allocated_elements", "old_elements_copied", "new_elements_written"):
                self._count(name, 0)
            self._clear_backing()

        def _clear_backing(self):
            # copy.copy followed by set_storage must never retain the source
            # layer's backing buffers, including when keys/values are compact.
            self._key_storage = self._value_storage = None
            self._ring_capacity = self._ring_start = 0

        def _count(self, name, amount):
            if self.capacity_counters is not None:
                key = "sliding_" + name
                self.capacity_counters[key] = self.capacity_counters.get(key, 0) + int(amount)

        def storage_capacity_tokens(self, query_chunk):
            """Worst physical token slots in one K or V buffer, including 2C.

            Multiply by each buffer's batch, heads, head dimension and element
            size when budgeting. Context limits constrain cumulative tokens;
            they must not truncate the old-tail-plus-query attention view.
            """
            query_chunk = int(query_chunk)
            if query_chunk < 0:
                raise ValueError("Query chunk must be nonnegative")
            needed = max(self.sliding_window, self.sliding_window - 1 + query_chunk)
            rounded = ((needed + self.capacity_increment - 1) // self.capacity_increment) * self.capacity_increment
            return 2 * max(self._ring_capacity, rounded)

        def set_storage(self, keys, values, length):
            """Bind a compact snapshot/view and discard all prior ring metadata.

            This does not clone caller-owned tensors. The next update allocates
            independent ring storage before writing, so a membership view is
            never modified in place through a copied layer's stale backing.
            """
            length = int(length)
            expected = min(length, self.sliding_window - 1)
            if not 0 <= length <= self.capacity_limit:
                raise ValueError("Sliding KV length exceeds its declared context")
            if (keys.ndim != 4 or values.ndim != 4 or
                    keys.shape[-2] != expected or values.shape[-2] != expected or
                    keys.shape[0] != values.shape[0]):
                raise ValueError("Sliding snapshot must contain exactly its chronological trailing window")
            self._clear_backing()
            self.keys, self.values = keys, values
            self.dtype, self.device = keys.dtype, keys.device
            self.is_initialized = True
            self.cumulative_length = length
            self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)

        def activate_past_recording(self):
            raise RuntimeError("Mirrored sliding storage does not support speculative/past-state recording")

        def _ensure_capacity(self, key_states, value_states, query_length):
            capacity = self.storage_capacity_tokens(query_length) // 2
            if self._key_storage is not None and self._ring_capacity >= capacity:
                return
            key_shape, value_shape = list(key_states.shape), list(value_states.shape)
            key_shape[-2] = value_shape[-2] = 2 * capacity
            keys = key_states.new_empty(key_shape)
            values = value_states.new_empty(value_shape)
            retained = min(self.cumulative_length, self.sliding_window - 1)
            if retained:
                for storage, old in ((keys, self.keys), (values, self.values)):
                    storage[..., :retained, :].copy_(old)
                    storage[..., capacity:capacity + retained, :].copy_(old)
                self._count("old_elements_copied", 2 * (self.keys.numel() + self.values.numel()))
            self._key_storage, self._value_storage = keys, values
            self._ring_capacity, self._ring_start = capacity, 0
            self._count("allocations", 2)
            self._count("allocated_elements", keys.numel() + values.numel())

        def _write(self, storage, states, position):
            capacity, count = self._ring_capacity, states.shape[-2]
            first = min(count, capacity - position)
            if first:
                chunk = states[..., :first, :]
                storage[..., position:position + first, :].copy_(chunk)
                storage[..., capacity + position:capacity + position + first, :].copy_(chunk)
            remainder = count - first
            if remainder:
                chunk = states[..., first:, :]
                storage[..., :remainder, :].copy_(chunk)
                storage[..., capacity:capacity + remainder, :].copy_(chunk)

        def update(self, key_states, value_states, *args, **kwargs):
            if self.record_past:
                raise RuntimeError("Mirrored sliding storage does not support speculative/past-state recording")
            if (key_states.ndim != 4 or value_states.ndim != 4 or
                    key_states.shape[-2] != value_states.shape[-2] or
                    key_states.shape[0] != value_states.shape[0]):
                raise ValueError("Expected matching [batch, heads, tokens, head_dim] query KV")
            query_length = key_states.shape[-2]
            if self.cumulative_length + query_length > self.capacity_limit:
                raise ValueError("Complete generation KV exceeds its declared context")
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            retained = min(self.cumulative_length, self.sliding_window - 1)
            if retained:
                for old, incoming in ((self.keys, key_states), (self.values, value_states)):
                    if (old.ndim != 4 or old.shape[-2] != retained or
                            old.shape[:2] != incoming.shape[:2] or old.shape[-1] != incoming.shape[-1] or
                            old.device != incoming.device or old.dtype != incoming.dtype):
                        raise ValueError("Sliding query KV shape, dtype or device differs from its existing tail")
            self._ensure_capacity(key_states, value_states, query_length)
            start, capacity = self._ring_start, self._ring_capacity
            position = (start + retained) % capacity
            self._write(self._key_storage, key_states, position)
            self._write(self._value_storage, value_states, position)
            self._count("new_elements_written", 2 * (key_states.numel() + value_states.numel()))
            full_length = retained + query_length
            full_keys = self._key_storage[..., start:start + full_length, :]
            full_values = self._value_storage[..., start:start + full_length, :]
            self.cumulative_length += query_length
            retained = min(self.cumulative_length, self.sliding_window - 1)
            self._ring_start = (start + full_length - retained) % capacity
            end = self._ring_start + retained
            self.keys = self._key_storage[..., self._ring_start:end, :]
            self.values = self._value_storage[..., self._ring_start:end, :]
            return full_keys, full_values

        def crop(self, tokens_to_remove):
            if self.record_past:
                raise RuntimeError("Mirrored sliding storage does not support speculative/past-state recording")
            if not self.is_initialized:
                return
            # Native rejects rollback once the unrecorded window was reached.
            super().crop(tokens_to_remove)
            keys, values = self.keys.clone(), self.values.clone()
            self._count("allocations", 2)
            self._count("allocated_elements", keys.numel() + values.numel())
            self._count("old_elements_copied", keys.numel() + values.numel())
            self.set_storage(keys, values, self.cumulative_length)

        def reset(self):
            self._clear_backing()
            self.keys = self.values = None
            self.cumulative_length = 0
            self.is_initialized = False

        def batch_repeat_interleave(self, repeats):
            super().batch_repeat_interleave(repeats)
            if self.is_initialized and self.keys.ndim == 4:
                self.set_storage(self.keys, self.values, self.cumulative_length)

        def batch_select_indices(self, indices):
            super().batch_select_indices(indices)
            if self.is_initialized and self.keys.ndim == 4:
                self.set_storage(self.keys, self.values, self.cumulative_length)

        def reorder_cache(self, beam_idx):
            super().reorder_cache(beam_idx)
            if self.is_initialized and self.keys.ndim == 4:
                self.set_storage(self.keys, self.values, self.cumulative_length)

        def offload(self):
            raise RuntimeError("Mirrored sliding storage does not support offloaded KV state")

        def prefetch(self):
            raise RuntimeError("Mirrored sliding storage does not support offloaded KV state")

    _RING_SLIDING_LAYER = RingSlidingWindowLayer
    return _RING_SLIDING_LAYER


def enable_sliding_storage(layer, increment, capacity_limit, counters=None):
    """Convert a pinned native sliding layer without copying its logical tail."""
    layer_type = _ring_sliding_layer_type()
    increment, capacity_limit = int(increment), int(capacity_limit)
    if increment < 0 or capacity_limit < 1:
        raise ValueError("KV increment must be nonnegative and context limit positive")
    if getattr(layer, "record_past", False):
        raise ValueError("Cannot convert sliding cache with speculative/past-state recording")
    if isinstance(layer, layer_type):
        if layer.get_seq_length() > capacity_limit:
            raise ValueError("Sliding KV length exceeds its declared context")
        layer.capacity_increment = max(1, increment)
        layer.capacity_limit, layer.capacity_counters = capacity_limit, counters
        for name in ("allocations", "allocated_elements", "old_elements_copied", "new_elements_written"):
            layer._count(name, 0)
        return layer
    if type(layer).__name__ != "DynamicSlidingWindowLayer":
        raise TypeError("Expected the native Transformers DynamicSlidingWindowLayer")
    replacement = layer_type(layer.sliding_window, increment, capacity_limit, counters)
    replacement.__dict__.update(layer.__dict__)
    if layer.is_initialized:
        if layer.keys.ndim == 1 and layer.keys.numel() == 0 and not layer.get_seq_length():
            # Native early/lazy initialization has no shaped logical KV yet.
            replacement.reset()
        else:
            replacement.set_storage(layer.keys, layer.values, int(layer.get_seq_length()))
    return replacement


def __getattr__(name):
    if name == "RingSlidingWindowLayer":
        return _ring_sliding_layer_type()
    raise AttributeError(name)
