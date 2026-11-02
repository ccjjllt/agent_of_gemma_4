"""Select surviving KV rows without re-padding their physical token axis."""
from __future__ import annotations

import copy

from .kv_cache import _layers, _replace_layer


def select_batch_rows(cache, indices, *, physical_length=None, counters=None):
    """Transfer active-cache ownership to selected rows in the given order.

    The caller must retire ``cache`` before mutating the returned cache and
    select the same attention-mask rows. ``physical_length`` may remove common
    left padding; the caller must also trim those mask columns and guarantee
    they contain no real tokens. True position IDs do not change. By default
    the old physical length is retained. No dead rows are computed.

    A contiguous selection uses only views, including capacity/ring backing
    storage. Other selections allocate only survivor rows: full-capacity layers
    retain their old capacity minus removed padding while copying only logical KV; sliding layers
    copy compact logical KV and let ``_replace_layer`` reset ring storage.

    ``batch_row_*_selections`` count calls, ``batch_row_selected_rows`` counts
    returned rows, and the remaining counters count KV buffer allocations,
    allocated elements (including full-KV headroom) and actually copied
    elements across all layers. View selections allocate/copy no KV elements.
    ``common_padding_tokens_removed`` is removed columns times selected rows,
    counted once per selection rather than once per layer.
    """
    layers = _layers(cache)
    indices = list(indices)
    batch_size = layers[0].keys.shape[0]
    if not indices or any(type(row) is not int or not 0 <= row < batch_size for row in indices):
        raise ValueError("Select at least one valid batch row")
    old_length = int(layers[0].get_seq_length())
    physical_length = old_length if physical_length is None else int(physical_length)
    if not 1 <= physical_length <= old_length:
        raise ValueError("Selected physical length must be within the existing KV length")
    trim = old_length - physical_length
    if any(layer.keys.shape[0] != batch_size or layer.values.shape[0] != batch_size
           or int(layer.get_seq_length()) != old_length for layer in layers):
        raise ValueError("KV layers must agree on batch size and physical length")

    contiguous = indices == list(range(indices[0], indices[0] + len(indices)))
    result = copy.copy(cache)
    result.layers = []
    result._membership_view = False
    allocations = allocated_elements = copied_elements = 0
    for layer in layers:
        if contiguous:
            first, end = indices[0], indices[-1] + 1
            selected = copy.copy(layer)
            if layer.is_sliding:
                retained = min(physical_length, int(layer.sliding_window) - 1)
                keys = layer.keys[first:end, :, -retained:, :]
                values = layer.values[first:end, :, -retained:, :]
                if (type(layer).__name__ == "RingSlidingWindowLayer"
                        and layer._key_storage is not None and layer._value_storage is not None):
                    selected._key_storage = layer._key_storage[first:end]
                    selected._value_storage = layer._value_storage[first:end]
                    removed = layer.keys.shape[-2] - retained
                    selected._ring_start = (layer._ring_start + removed) % layer._ring_capacity
                    end_token = selected._ring_start + retained
                    selected.keys = selected._key_storage[..., selected._ring_start:end_token, :]
                    selected.values = selected._value_storage[..., selected._ring_start:end_token, :]
                    selected.cumulative_length = physical_length
                else:
                    selected = _replace_layer(layer, keys, values, physical_length)
            else:
                selected.keys = layer.keys[first:end, :, trim:, :]
                selected.values = layer.values[first:end, :, trim:, :]
                for name in ("_key_storage", "_value_storage"):
                    storage = getattr(layer, name, None)
                    if storage is not None:
                        setattr(selected, name, storage[first:end, :, trim:, :])
                if type(layer).__name__ == "CapacityDynamicLayer":
                    selected._length = physical_length
        else:
            tensors = []
            full_capacity = type(layer).__name__ == "CapacityDynamicLayer"
            for name, backing_name in (("keys", "_key_storage"), ("values", "_value_storage")):
                source = getattr(layer, name)
                source = (source[..., -min(physical_length, int(layer.sliding_window) - 1):, :]
                          if layer.is_sliding else source[..., trim:old_length, :])
                shape = list(source.shape)
                shape[0] = len(indices)
                if full_capacity:
                    shape[-2] = getattr(layer, backing_name).shape[-2] - trim
                storage = source.new_empty(shape)
                logical_length = source.shape[-2]
                for destination_row, source_row in enumerate(indices):
                    storage[destination_row, :, :logical_length, :].copy_(source[source_row])
                tensors.append(storage)
                allocations += 1
                allocated_elements += storage.numel()
                copied_elements += len(indices) * source[0].numel()
            selected = _replace_layer(layer, *tensors, physical_length)
        result.layers.append(selected)

    if counters is not None:
        changes = {
            "batch_row_view_selections" if contiguous else "batch_row_copy_selections": 1,
            "batch_row_selected_rows": len(indices),
            "batch_row_buffer_allocations": allocations,
            "batch_row_allocated_elements": allocated_elements,
            "batch_row_copied_elements": copied_elements,
            "common_padding_tokens_removed": trim * len(indices),
        }
        for key, value in changes.items():
            counters[key] = counters.get(key, 0) + value
    return result
