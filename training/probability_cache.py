"""Bounded CPU storage for a fixed reference's measured token log-probabilities.

Each instance belongs to one trainer and one immutable reference identity. Keys
contain the exact, complete prompt and completion token sequences; task IDs and
hash-only matches are never used. Values are ordinary immutable Python floats,
not activations, tensors, KV state or autograd graphs. Nothing is persisted.
"""
from __future__ import annotations

from collections import OrderedDict
import math
from numbers import Real


class ReferenceProbabilityCache:
    """LRU for complete selected-token scores from one fixed reference.

    ``max_tokens`` counts every stored prompt token, completion token and score,
    so a long prefix is charged even when its completion is short. Entry and
    token limits jointly bound resident Python storage; token IDs must fit a
    signed int64. This is a capacity count, not a byte-memory measurement.

    ``hits`` counts successful lookups, not independently measured model work.
    The trainer records a skipped reference forward when it actually uses a hit.
    The object is deliberately local, single-owner and has no disk/global cache.
    """

    def __init__(self, namespace: str, max_entries: int = 128, max_tokens: int = 1_000_000):
        if not isinstance(namespace, str) or not namespace or len(namespace) > 1024:
            raise ValueError("A nonempty, bounded fixed-reference namespace is required")
        for name, value in (("max_entries", max_entries), ("max_tokens", max_tokens)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self._namespace = namespace
        self.max_entries = max_entries
        self.max_tokens = max_tokens
        self._entries = OrderedDict()
        self.used_tokens = 0
        self.hits = self.misses = self.insertions = self.replacements = 0
        self.evictions = self.oversized_skips = self.disabled_skips = 0

    @property
    def namespace(self):
        return self._namespace

    def _check_namespace(self, namespace):
        if namespace is not None and namespace != self.namespace:
            raise ValueError("Reference namespace differs; use a separate cache for a different reference")

    @staticmethod
    def _tokens(values, name):
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"{name} must be CPU token IDs in a list or tuple")
        result = tuple(values)
        if not result or any(type(value) is not int or not 0 <= value <= 2**63 - 1 for value in result):
            raise ValueError(f"{name} must contain nonempty, nonnegative int64 token IDs")
        return result

    @classmethod
    def _key(cls, prompt_ids, completion_ids):
        return cls._tokens(prompt_ids, "prompt_ids"), cls._tokens(completion_ids, "completion_ids")

    def get(self, prompt_ids, completion_ids, *, namespace=None):
        """Return an independent score list, or None for this exact call's miss."""
        self._check_namespace(namespace)
        key = self._key(prompt_ids, completion_ids)
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return list(entry)

    def put(self, prompt_ids, completion_ids, logprobs, *, namespace=None):
        """Store measured, finite CPU scores; return False when capacity rejects them.

        Plain real-number values are converted to floats. Tensor values are
        rejected rather than implicitly detached or copied from a GPU graph.
        Invalid inputs cannot evict an already valid reference entry.
        """
        self._check_namespace(namespace)
        key = self._key(prompt_ids, completion_ids)
        if not isinstance(logprobs, (list, tuple)) or len(logprobs) != len(key[1]):
            raise ValueError("Reference scores need exactly one value per completion token")
        values = []
        for value in logprobs:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("Reference scores must be CPU real scalars, never tensors")
            try:
                scalar = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError("Reference scores must be finite") from exc
            if not math.isfinite(scalar):
                raise ValueError("Reference scores must be finite")
            values.append(scalar)
        values = tuple(values)
        cost = len(key[0]) + len(key[1]) + len(values)
        if not self.max_entries or not self.max_tokens:
            self.disabled_skips += 1
            return False
        if cost > self.max_tokens:
            self.oversized_skips += 1
            return False

        previous = self._entries.pop(key, None)
        if previous is not None:
            self.used_tokens -= len(key[0]) + len(key[1]) + len(previous)
            self.replacements += 1
        else:
            self.insertions += 1
        while self._entries and (len(self._entries) >= self.max_entries or self.used_tokens + cost > self.max_tokens):
            (old_prompt, old_completion), old_values = self._entries.popitem(last=False)
            self.used_tokens -= len(old_prompt) + len(old_completion) + len(old_values)
            self.evictions += 1
        self._entries[key] = values
        self.used_tokens += cost
        return True

    def clear(self, *, reset_stats=False):
        """Discard scores while keeping this instance's fixed reference identity."""
        self._entries.clear()
        self.used_tokens = 0
        if reset_stats:
            self.hits = self.misses = self.insertions = self.replacements = 0
            self.evictions = self.oversized_skips = self.disabled_skips = 0

    def stats(self):
        """Numerical cache observations only; no tokens, probabilities or namespace."""
        return {
            "entries": len(self._entries), "token_units": self.used_tokens,
            "max_entries": self.max_entries, "max_tokens": self.max_tokens,
            "hits": self.hits, "misses": self.misses,
            "insertions": self.insertions, "replacements": self.replacements,
            "evictions": self.evictions,
            "oversized_skips": self.oversized_skips, "disabled_skips": self.disabled_skips,
        }
