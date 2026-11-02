"""Per-device snapshot allowance, separate from the model's active KV budget."""
from __future__ import annotations

import math


def _storage_bytes(tensors):
    """Count backing storage once, including capacity headroom and row views."""
    sizes, seen = {}, set()
    for tensor in tensors:
        if tensor is None:
            continue
        device = str(tensor.device)
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        identity = (device, storage.data_ptr(), size)
        if size and identity not in seen:
            seen.add(identity)
            sizes[device] = sizes.get(device, 0) + size
    return sizes


class PrefixCacheBudget:
    """Estimate conservative GPU headroom; never allocate memory here.

    Actual CUDA free/allocated memory already includes resident optimizer state.
    Additionally reserve gradients plus two FP32 Adam moments for trainable
    parameters, even before the first update. After Adam exists this deliberately
    remains conservative. Each collection constructs a fresh budget; snapshots
    are released before training. This is an estimate, not an OOM guarantee.
    """

    def __init__(self, torch, model, *, max_batch_size, max_context,
                 prefill_chunk_tokens, safety_bytes=2 * 1024**3,
                 max_total_fraction=0.15, hard_cap_bytes=None, cpu_fallback_bytes=0):
        self.torch = torch
        self.max_batch_size, self.max_context = int(max_batch_size), int(max_context)
        self.prefill_chunk_tokens, self.safety_bytes = int(prefill_chunk_tokens), int(safety_bytes)
        self.fraction = float(max_total_fraction)
        self.hard_cap = None if hard_cap_bytes is None else int(hard_cap_bytes)
        self.cpu_fallback = int(cpu_fallback_bytes)
        if (min(self.max_batch_size, self.max_context, self.prefill_chunk_tokens) < 1
                or self.safety_bytes < 0 or self.cpu_fallback < 0
                or self.hard_cap is not None and self.hard_cap < 0
                or not math.isfinite(self.fraction) or not 0 <= self.fraction <= 1):
            raise ValueError("Invalid prefix-cache memory budget")
        self.parameters = [p for p in model.parameters() if p.requires_grad] if model is not None else []
        self.training_bytes = {}
        for p in self.parameters:
            device = str(p.device)
            self.training_bytes[device] = self.training_bytes.get(device, 0) + p.numel() * max(4, p.element_size())
        self.geometry = None

    def observe(self, cache):
        """Learn the actual non-shared layer layout after a real model forward."""
        if cache is None or not getattr(cache, "layers", None):
            return
        geometry = []
        for layer in cache.layers:
            if not layer.is_initialized or layer.keys.ndim != 4 or layer.values.ndim != 4:
                return
            if hasattr(layer, "storage_capacity_tokens"):
                tokens = layer.storage_capacity_tokens(self.prefill_chunk_tokens)
            else:
                tokens = (min(self.max_context, int(layer.sliding_window) - 1 + self.prefill_chunk_tokens)
                          if layer.is_sliding else self.max_context)
            for tensor in (layer.keys, layer.values):
                geometry.append((str(tensor.device), tensor.shape[1] * tensor.shape[-1] * tensor.element_size(), tokens))
        if self.geometry is not None and geometry != self.geometry:
            raise ValueError("Generation KV device/head/dtype geometry changed")
        self.geometry = geometry

    def refresh(self, *, active_caches=(), prefix_bytes_by_device=None):
        resident = dict(prefix_bytes_by_device or {})
        if not self.torch.cuda.is_available():
            cap = self.cpu_fallback
            if self.hard_cap is not None:
                cap = min(cap, self.hard_cap)
            return {"max_bytes": cap, "max_bytes_by_device": {"cpu": cap},
                    "accounting": {"mode": "explicit_cpu_cap", "limit_bytes": cap}}
        devices = [f"cuda:{i}" for i in range(self.torch.cuda.device_count())]
        if self.geometry is None or self.hard_cap == 0:
            return {"max_bytes": 0, "max_bytes_by_device": dict.fromkeys(devices, 0),
                    "accounting": {"mode": "disabled" if self.hard_cap == 0 else "awaiting_kv_geometry"}}
        tensors = []
        for cache in active_caches:
            for layer in getattr(cache, "layers", ()):
                if layer.is_initialized:
                    tensors.extend((layer.keys, layer.values))
        active = _storage_bytes(tensors)
        gradients = _storage_bytes(p.grad for p in self.parameters)
        peak = dict.fromkeys(devices, 0)
        # Old/new maximum batches can coexist during rebuild/growth. Reserve
        # another prefill leader and a checkout clone. Sliding geometry includes
        # the full query chunk and, when present, both mirrored storage halves.
        rows = 2 * self.max_batch_size + 2
        for device, per_token, tokens in self.geometry:
            if device in peak:
                peak[device] += rows * per_token * tokens
        limits, accounting = {}, {}
        for i, device in enumerate(devices):
            free, total = self.torch.cuda.mem_get_info(i)
            allocated = self.torch.cuda.memory_allocated(i)
            reserved = self.torch.cuda.memory_reserved(i)
            available = int(free) + max(0, int(reserved) - int(allocated))
            remaining_active = max(0, peak[device] - active.get(device, 0))
            remaining_training = max(0, 3 * self.training_bytes.get(device, 0) - gradients.get(device, 0))
            headroom = (resident.get(device, 0) + available - remaining_active
                        - remaining_training - self.safety_bytes)
            limits[device] = max(0, min(int(total * self.fraction), headroom))
            accounting[device] = {
                "limit_bytes": limits[device], "reusable_free_bytes": available,
                "resident_prefix_bytes": resident.get(device, 0),
                "active_kv_peak_bytes": peak[device], "active_kv_storage_bytes": active.get(device, 0),
                "remaining_active_reserve_bytes": remaining_active,
                "remaining_gradient_adam_reserve_bytes": remaining_training,
                "workspace_reserve_bytes": self.safety_bytes,
            }
        total_limit = sum(limits.values())
        if self.hard_cap is not None:
            total_limit = min(total_limit, self.hard_cap)
        return {"max_bytes": total_limit, "max_bytes_by_device": limits,
                "accounting": {"mode": "cuda_headroom", "max_bytes": total_limit,
                               "memory_fraction": self.fraction, "devices": accounting}}
