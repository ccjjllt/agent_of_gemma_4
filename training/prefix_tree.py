"""Exact token-prefix plans and compact ancestor attention layouts for training.

The planner is CPU-only and records indices/counts, not duplicated trajectories.
No dense N-by-N mask, inference KV or detached prefix state is constructed here.
"""

from __future__ import annotations

from dataclasses import dataclass


ATTENTION_LAYOUT_VERSION = "segmented_tree_sdpa_v1"


@dataclass(frozen=True)
class TreeSegment:
    """One contiguous unique-child path, plus its preceding ancestor path."""

    start: int
    stop: int
    ancestors: tuple[int, ...]


@dataclass(frozen=True)
class TreeAttentionMask:
    """Attention recipe; the backend materializes only one query block at a time.

    Full and sliding attention share the same immutable CPU segment metadata.
    ``sliding_window=None`` selects full causal attention along each tree path.
    """

    token_count: int
    segments: tuple[TreeSegment, ...]
    query_chunk_size: int
    sliding_window: int | None = None
    layout_version: str = ATTENTION_LAYOUT_VERSION


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _call_input(call):
    prompt, completion = call.get("prompt_ids"), call.get("completion_ids")
    if not isinstance(prompt, (list, tuple)) or not isinstance(completion, (list, tuple)) or not prompt or not completion:
        raise ValueError("Each call needs its complete nonempty prompt_ids and completion_ids")
    if any(type(token) is not int or token < 0 for part in (prompt, completion) for token in part):
        raise ValueError("Token identities must be nonnegative integers")
    # The input ends one token before the last target. The final prompt token
    # predicts completion[0], including for a one-token completion.
    return tuple(prompt) + tuple(completion[:-1])


class _Trie:
    def __init__(self):
        self.input_ids = []
        self.parent_indices = []
        self.depths = []
        self.children = []
        self.roots = {}

    def add(self, tokens, *, return_path=False):
        parent = -1
        path = [] if return_path else None
        for depth, token in enumerate(tokens):
            children = self.roots if parent == -1 else self.children[parent]
            index = children.get(token)
            if index is None:
                index = len(self.input_ids)
                children[token] = index
                self.input_ids.append(token)
                self.parent_indices.append(parent)
                self.depths.append(depth)
                self.children.append({})
            if return_path:
                path.append(index)
            parent = index
        return path


def build_tree(calls):
    """Build a topologically ordered trie and each call's hidden-row mapping.

    A shared hidden row may predict different targets in different calls. It
    may also carry different advantages or behavior probabilities. The caller
    must retain those separate loss occurrences, not average their targets.
    """
    if not calls:
        raise ValueError("A training tree needs at least one complete call")
    tree = _Trie()
    score_indices = []
    for call in calls:
        tokens = _call_input(call)
        path = tree.add(tokens, return_path=True)
        start = len(call["prompt_ids"]) - 1
        score_indices.append(path[start:start + len(call["completion_ids"])])
    return {"input_ids": tree.input_ids, "parent_indices": tree.parent_indices,
            "depths": tree.depths, "score_indices": score_indices,
            "token_count": len(tree.input_ids)}


def mask_workspace_bytes(token_count, query_chunk_size, mask_element_size=4):
    """Conservative per-device mask/index workspace, not total attention memory.

    Three full-path int64 vectors plus two query-block-sized mask workspaces
    cover placement/position indices and construction of a rectangular mask.
    All segments, including roots, use query chunks. These buffers are transient
    and must be rebuilt inside attention checkpoint replay, not retained once
    per layer or query block. Q/K/V, attention-score/softmax workspaces, saved
    activations and optimizer storage are outside this bound. No N-by-N mask
    term occurs, including when SDPA selects its math attention backend.
    """
    count = _integer(token_count, "token_count", minimum=1)
    chunk = _integer(query_chunk_size, "query_chunk_size", minimum=1)
    size = _integer(mask_element_size, "mask_element_size", minimum=1)
    return 3 * count * 8 + 2 * min(chunk, count) * count * size


def plan_call_groups(calls, *, max_context, max_group_calls, max_tree_tokens,
                     max_mask_bytes, mask_element_size, min_shared_tokens,
                     attention_query_chunk_size=256):
    """Plan deterministic lexicographic groups, falling back to full calls.

    Attention work remains conservatively bounded by N^2, compared with the
    independent input-length squares. Mask memory is only Q_chunk-by-N plus
    linear indices; query chunks shrink to fit the explicit per-device budget.
    A singleton uses the original independent scorer and allocates no tree mask.
    """
    max_context = _integer(max_context, "max_context", minimum=1)
    max_group_calls = _integer(max_group_calls, "max_group_calls", minimum=1)
    max_tree_tokens = _integer(max_tree_tokens, "max_tree_tokens", minimum=1)
    max_mask_bytes = _integer(max_mask_bytes, "max_mask_bytes")
    mask_element_size = _integer(mask_element_size, "mask_element_size", minimum=1)
    min_shared_tokens = _integer(min_shared_tokens, "min_shared_tokens")
    attention_query_chunk_size = _integer(attention_query_chunk_size, "attention_query_chunk_size", minimum=1)
    inputs = [_call_input(call) for call in calls]
    for index, tokens in enumerate(inputs):
        if len(tokens) + 1 > max_context:
            raise ValueError(f"Call {index} exceeds complete-context limit; no truncation is allowed")
    order = sorted(range(len(calls)), key=inputs.__getitem__)
    groups = []
    cursor = 0
    while cursor < len(order):
        tree = _Trie()
        total_tokens = independent_attention = 0
        best = None
        reason = "group_call_limit" if max_group_calls == 1 else "no_partner"
        stop = min(len(order), cursor + max_group_calls)
        for end in range(cursor, stop):
            index = order[end]
            length = len(inputs[index])
            tree.add(inputs[index])
            total_tokens += length
            independent_attention += length * length
            unique = len(tree.input_ids)
            if unique > max_tree_tokens:
                reason = "tree_token_limit"
                break
            available_mask_rows = (max_mask_bytes - 3 * unique * 8) // (2 * unique * mask_element_size)
            chunk = min(attention_query_chunk_size, unique, available_mask_rows)
            if chunk < 1:
                reason = "mask_budget"
                break
            mask_bytes = mask_workspace_bytes(unique, chunk, mask_element_size)
            if end == cursor:
                continue
            saved = total_tokens - unique
            if unique * unique > independent_attention:
                reason = "dense_attention_work_limit"
                continue
            if saved < min_shared_tokens:
                reason = "insufficient_shared_tokens"
                continue
            best = {"call_indices": order[cursor:end + 1],
                    "input_tokens": total_tokens, "unique_tokens": unique,
                    "saved_tokens": saved, "mode": "tree", "reason": "exact_prefix_reuse",
                    "mask_bytes": mask_bytes, "tree_attention_elements": unique * unique,
                    "attention_layout_version": ATTENTION_LAYOUT_VERSION,
                    "attention_query_chunk_size": chunk,
                    "independent_attention_elements": independent_attention}
        if best is not None:
            groups.append(best)
            cursor += len(best["call_indices"])
        else:
            index = order[cursor]
            length = len(inputs[index])
            groups.append({"call_indices": [index], "input_tokens": length,
                           "unique_tokens": length, "saved_tokens": 0,
                           "mode": "independent", "reason": reason, "mask_bytes": 0,
                           "tree_attention_elements": 0, "independent_attention_elements": length * length})
            cursor += 1
    return groups


def tree_segments(tree):
    """Validate a trie and partition its physical rows into unique-child paths.

    Branch siblings must never attend each other. Each segment's ancestors are
    its exact root-to-parent path, not all preceding physical packing indices.
    With bounded group size, metadata is O(number_of_calls * token_count).
    """
    parents, depths = tree["parent_indices"], tree["depths"]
    count = len(tree["input_ids"])
    if (count < 1 or type(tree.get("token_count")) is not int or tree["token_count"] != count
            or len(parents) != count or len(depths) != count):
        raise ValueError("Malformed nonempty training tree")
    child_counts = [0] * count
    for index, parent in enumerate(parents):
        if type(parent) is not int or not -1 <= parent < index:
            raise ValueError("Tree parents must precede their children")
        expected_depth = 0 if parent == -1 else depths[parent] + 1
        if type(depths[index]) is not int or depths[index] != expected_depth:
            raise ValueError("Tree depth must equal original token-path position")
        if parent >= 0:
            child_counts[parent] += 1
    starts = [0] + [index for index in range(1, count)
                    if parents[index] != index - 1 or child_counts[parents[index]] != 1]
    segments = []
    for start, stop in zip(starts, starts[1:] + [count]):
        ancestors = []
        parent = parents[start]
        while parent >= 0:
            ancestors.append(parent)
            parent = parents[parent]
        ancestors.reverse()
        segments.append(TreeSegment(start, stop, tuple(ancestors)))
    return tuple(segments)


def make_tree_masks(tree, *, sliding_window, query_chunk_size=256, device=None, dtype=None):
    """Return shared immutable attention descriptions, without dense masks.

    ``device``/``dtype`` remain accepted for caller compatibility, but placement
    and transient mask construction belong to the native attention backend.
    Path depths still supply RoPE positions and sliding-window boundaries.
    """
    window = _integer(sliding_window, "sliding_window", minimum=1)
    chunk = _integer(query_chunk_size, "query_chunk_size", minimum=1)
    if dtype is not None and not dtype.is_floating_point:
        raise ValueError("Attention masks require a floating-point dtype")
    segments = tree_segments(tree)
    count = tree["token_count"]
    return {
        "full_attention": TreeAttentionMask(count, segments, chunk),
        "sliding_attention": TreeAttentionMask(count, segments, chunk, window),
    }
