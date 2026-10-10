#!/usr/bin/env python3
"""Register exact-content deduplication and leakage-safe train/selection splits."""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def near_dup_key(tx):
    # Exact normalized content; tool-name sequences alone are not duplicate tasks.
    payload = {"messages": tx.get("messages", []), "targets": tx.get("target_assistant_steps", [])}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def group_key(tx):
    g = tx.get("group", {})
    if not all(g.get(k) for k in ("issue", "repo", "commit")):
        raise ValueError("every transaction needs issue/repo/commit")
    return str(g["issue"]), str(g["repo"]), str(g["commit"])


def load_ids(path):
    return {x.strip() for x in Path(path).read_text().splitlines() if x.strip()} if path else set()


def register(input_path, max_per_issue=3):
    groups, seen_ids, seen_hashes, counts = {}, set(), set(), {}
    for line in Path(input_path).read_text().splitlines():
        if not line.strip():
            continue
        tx = json.loads(line)
        issue, repo, commit = group_key(tx)
        tid = tx.get("transaction_id")
        if not tid:
            raise ValueError("missing transaction_id")
        if tid in seen_ids:
            raise ValueError(f"duplicate transaction id {tid}")
        seen_ids.add(tid)
        digest = near_dup_key(tx)
        key = json.dumps([issue, repo, commit])
        issue_key = (repo, issue)
        if digest in seen_hashes or counts.get(issue_key, 0) >= max_per_issue:
            continue
        seen_hashes.add(digest)
        counts[issue_key] = counts.get(issue_key, 0) + 1
        group = groups.setdefault(key, {"issue": issue, "repo": repo, "commit": commit,
                                       "members": [], "content_hashes": [], "split": "unassigned"})
        group["members"].append(tid)
        group["content_hashes"].append(digest)
    return {"registry_version": 2, "groups": groups, "max_per_issue": max_per_issue,
            "deduplication": "exact canonical content; not semantic near-duplicate detection"}


def assign(registry, fraction, whole_repo=False, seed=20261010, heldout=None):
    if not 0 < fraction < 1:
        raise ValueError("select-fraction must be between 0 and 1")
    heldout = heldout or set()
    groups = registry["groups"]
    for group in groups.values():
        if group["issue"] in heldout:
            group["split"] = "heldout-excluded"
    keys = sorted(groups)
    parent = {k: k for k in keys}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    axes = {}
    for key in keys:
        g = groups[key]
        # Keep same issue AND same repository snapshot together even with distinct task IDs.
        links = [("issue", g["repo"], g["issue"]), ("snapshot", g["repo"], g["commit"])]
        if whole_repo:
            links.append(("repo", g["repo"]))
        for axis in links:
            if axis in axes:
                parent[find(key)] = find(axes[axis])
            else:
                axes[axis] = key
    units = {}
    for key in keys:
        units.setdefault(find(key), []).append(key)
    usable = []
    for unit in units.values():
        if any(groups[k]["split"] == "heldout-excluded" for k in unit):
            for k in unit:
                groups[k]["split"] = "heldout-excluded"
        else:
            usable.append(unit)
    if len(usable) < 2:
        raise ValueError("need at least two independent split groups")
    usable.sort(key=lambda unit: hashlib.sha256((str(seed)+min(unit)).encode()).hexdigest())
    n_select = min(len(usable)-1, max(1, round(len(usable)*fraction)))
    for i, unit in enumerate(usable):
        for k in unit:
            groups[k]["split"] = "select" if i < n_select else "train"
    registry["policy"] = {"select_fraction": fraction, "whole_repo": whole_repo, "seed": seed,
                          "group_axes": ["repo+issue", "repo+commit"], "heldout_propagated": True}
    return registry


def materialize(input_path, registry, train_path, select_path):
    memberships = {}
    for g in registry["groups"].values():
        for tid, digest in zip(g["members"], g["content_hashes"]):
            memberships[tid] = (g["split"], digest)
    counts = {"train": 0, "select": 0, "excluded": 0}
    for path in (train_path, select_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w") as train, open(select_path, "w") as select:
        for line in Path(input_path).read_text().splitlines():
            if not line.strip():
                continue
            tx = json.loads(line)
            member = memberships.get(tx.get("transaction_id"))
            if not member or member[0] not in ("train", "select"):
                counts["excluded"] += 1
                continue
            split, digest = member
            if near_dup_key(tx) != digest:
                raise ValueError(f"transaction changed after registration: {tx['transaction_id']}")
            tx["group"]["split"] = split
            (train if split == "train" else select).write(json.dumps(tx, ensure_ascii=False)+"\n")
            counts[split] += 1
    if not counts["train"] or not counts["select"]:
        raise ValueError("both train and select must contain data")
    return counts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--input", required=True)
    r.add_argument("--registry", required=True)
    r.add_argument("--max-per-issue", type=int, default=3)
    a = sub.add_parser("assign")
    a.add_argument("--registry", required=True)
    a.add_argument("--out")
    a.add_argument("--heldout-ids")
    a.add_argument("--select-fraction", type=float, default=.2)
    a.add_argument("--seed", type=int, default=20261010)
    a.add_argument("--whole-repo", action="store_true")
    m = sub.add_parser("materialize")
    m.add_argument("--input", required=True)
    m.add_argument("--registry", required=True)
    m.add_argument("--train-out", required=True)
    m.add_argument("--select-out", required=True)
    for cmd in ("report", "check"):
        s = sub.add_parser(cmd)
        s.add_argument("--registry", required=True)
        s.add_argument("--heldout-ids")
    args = p.parse_args(argv)
    if args.cmd == "register":
        if args.max_per_issue < 1:
            raise ValueError("max-per-issue must be positive")
        reg = register(args.input, args.max_per_issue)
        Path(args.registry).write_text(json.dumps(reg, indent=2)+"\n")
        print(json.dumps({"groups": len(reg["groups"])}))
        return 0
    reg = json.loads(Path(args.registry).read_text())
    if args.cmd == "assign":
        assign(reg, args.select_fraction, args.whole_repo, args.seed, load_ids(args.heldout_ids))
        Path(args.out or args.registry).write_text(json.dumps(reg, indent=2)+"\n")
    elif args.cmd == "materialize":
        print(json.dumps(materialize(args.input, reg, args.train_out, args.select_out)))
        return 0
    groups = reg["groups"].values()
    splits = {}
    for g in groups:
        splits[g["split"]] = splits.get(g["split"], 0)+len(g["members"])
    leaks = [g["issue"] for g in reg["groups"].values()
             if g["issue"] in load_ids(getattr(args, "heldout_ids", None))
             and g["split"] in ("train", "select")]
    print(json.dumps({"transactions_by_split": splits, "heldout_leaks": leaks}))
    return 1 if leaks else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as e:
        print(f"split_registry: {e}", file=sys.stderr)
        sys.exit(2)
