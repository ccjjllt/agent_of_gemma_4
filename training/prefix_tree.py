"""Exact token-prefix plans and ancestor attention masks for training.

The planner is CPU-only and records indices/counts, not duplicated trajectories.
Torch is imported only when materializing masks. No inference KV is involved.
"""

from __future__ import annotations


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


def plan_call_groups(calls, *, max_context, max_group_calls, max_tree_tokens,
                     max_mask_bytes, mask_element_size, min_shared_tokens):
    """Plan deterministic lexicographic groups, falling back to full calls.

    Dense-attention work is conservatively N^2, compared with the sum of
    independent input-length squares. The mask budget covers both final dense
    additive masks; make_tree_masks uses only row-chunk boolean temporaries.
    A singleton uses the original independent scorer and allocates no tree mask.
    """
    max_context = _integer(max_context, "max_context", minimum=1)
    max_group_calls = _integer(max_group_calls, "max_group_calls", minimum=1)
    max_tree_tokens = _integer(max_tree_tokens, "max_tree_tokens", minimum=1)
    max_mask_bytes = _integer(max_mask_bytes, "max_mask_bytes")
    mask_element_size = _integer(mask_element_size, "mask_element_size", minimum=1)
    min_shared_tokens = _integer(min_shared_tokens, "min_shared_tokens")
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
            mask_bytes = 2 * unique * unique * mask_element_size
            if unique > max_tree_tokens:
                reason = "tree_token_limit"
                break
            if mask_bytes > max_mask_bytes:
                reason = "mask_budget"
                break
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


def make_tree_masks(tree, *, device, dtype, sliding_window):
    """Return [1,1,N,N] additive masks with only ancestors/self visible.

    Physical packing indices are not positions: siblings may be stored between
    ancestors and descendants. Euler intervals determine ancestry; original
    path depth determines the sliding window and the caller's position_ids.
    """
    import torch

    window = _integer(sliding_window, "sliding_window", minimum=1)
    parents, depths = tree["parent_indices"], tree["depths"]
    count = len(tree["input_ids"])
    if count < 1 or tree.get("token_count") != count or len(parents) != count or len(depths) != count:
        raise ValueError("Malformed nonempty training tree")
    if not dtype.is_floating_point:
        raise ValueError("Additive tree masks require a floating-point dtype")
    children = [[] for _ in range(count)]
    roots = []
    for index, parent in enumerate(parents):
        if type(parent) is not int or not -1 <= parent < index:
            raise ValueError("Tree parents must precede their children")
        expected_depth = 0 if parent == -1 else depths[parent] + 1
        if type(depths[index]) is not int or depths[index] != expected_depth:
            raise ValueError("Tree depth must equal original token-path position")
        (roots if parent == -1 else children[parent]).append(index)
    entry, exit_ = [0] * count, [0] * count
    clock = 0
    stack = [(root, False) for root in reversed(roots)]
    while stack:
        index, leaving = stack.pop()
        if leaving:
            exit_[index] = clock
            continue
        entry[index] = clock
        clock += 1
        stack.append((index, True))
        stack.extend((child, False) for child in reversed(children[index]))
    entry = torch.tensor(entry, device=device, dtype=torch.long)
    exit_ = torch.tensor(exit_, device=device, dtype=torch.long)
    depth = torch.tensor(depths, device=device, dtype=torch.long)
    full = torch.full((1, 1, count, count), -float("inf"), device=device, dtype=dtype)
    sliding = torch.full_like(full, -float("inf"))
    # Bound transient predicate storage rather than materializing another
    # N-by-N int64 depth-difference matrix alongside both additive masks.
    for start in range(0, count, 256):
        stop = min(count, start + 256)
        query_entry = entry[start:stop, None]
        visible = entry[None, :] <= query_entry
        visible.logical_and_(query_entry < exit_[None, :])
        full[0, 0, start:stop].masked_fill_(visible, 0.0)
        visible.logical_and_(depth[None, :] > depth[start:stop, None] - window)
        sliding[0, 0, start:stop].masked_fill_(visible, 0.0)
    return {"full_attention": full, "sliding_attention": sliding}
