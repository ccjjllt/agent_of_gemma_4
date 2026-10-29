"""Metadata-only scheduling for complete groups of official public tasks.

Adaptive selection and zero-variance filtering deliberately change the training
distribution. The returned probability is conditional on the current eligible
set; it is recorded for audit, not presented as an unbiased uniform-task loss.
This module never downloads, rewrites, or returns question/patch/test contents.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re
import statistics


_FORBIDDEN = re.compile(r"(^|[/_.-])(heldout|holdout|reserved|secret|hidden)($|[/_.-])", re.I)


def load_tasks(path, include_ids=None, exclude_ids=None):
    """Retain only task identity, repository and base commit from public rows.

    Run this on the manually prepared server-side public tasks.jsonl. No source
    task rows or derived prompt/answer datasets are saved. Exclusions are applied
    before retaining any metadata, and never enter the sampler's task universe.
    """
    source = Path(path).expanduser().resolve(strict=True)
    if _FORBIDDEN.search(str(source)):
        raise ValueError("The scheduler accepts only the explicit public training tasks file")
    if isinstance(include_ids, str) or isinstance(exclude_ids, str):
        raise TypeError("include_ids and exclude_ids must be collections of task IDs, not a string")
    included = None if include_ids is None else set(include_ids)
    excluded = set(exclude_ids or [])
    if included is not None and included & excluded:
        raise ValueError("A task cannot be both included and reserved/excluded")
    records, seen = [], set()
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed public task metadata at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Public task row {line_number} is not an object")
            identifier = row.get("instance_id")
            if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", identifier):
                raise ValueError(f"Invalid public instance_id at line {line_number}")
            if ".." in identifier.split("/"):
                raise ValueError(f"Unsafe public instance_id at line {line_number}")
            if identifier in seen:
                raise ValueError(f"Duplicate public instance_id: {identifier}")
            seen.add(identifier)
            if identifier in excluded or (included is not None and identifier not in included):
                continue
            repo, commit = row.get("repo"), row.get("base_commit")
            if not isinstance(repo, str) or not repo or not isinstance(commit, str) or not commit:
                raise ValueError(f"Public task {identifier} is missing repo/base_commit metadata")
            records.append({"id": identifier, "repo": repo, "base_commit": commit})
    if included is not None and included - seen:
        raise ValueError(f"Requested public task IDs were not found: {sorted(included - seen)}")
    if not records:
        raise ValueError("No public training tasks remain after explicit selection/exclusions")
    return records


class TaskSampler:
    """Coverage first, then an explicit exploration/priority mixture.

    Only complete groups are observed. Infrastructure failures must be handled
    by the caller and must not be passed as zeros to observe(). The caller also
    caps replacement groups per collection round; this class never resamples a
    failed or zero-variance group in an internal loop.
    """

    def __init__(self, tasks, config=None, seed=3407):
        supplied = dict(config or {})
        self.config = {
            "uniform_mix": float(supplied.get("uniform_mix", 0.2)),
            "ema_decay": float(supplied.get("ema_decay", 0.8)),
            "min_group_std": float(supplied.get("min_group_std", 1e-6)),
            "variance_scale": float(supplied.get("variance_scale", 0.05)),
            "variance_weight": float(supplied.get("variance_weight", 1.0)),
            "success_weight": float(supplied.get("success_weight", 1.0)),
            "novelty_weight": float(supplied.get("novelty_weight", 0.2)),
            "priority_floor": float(supplied.get("priority_floor", 0.05)),
            "repo_repeat_penalty": float(supplied.get("repo_repeat_penalty", 0.5)),
        }
        if not all(math.isfinite(value) for value in self.config.values()):
            raise ValueError("Curriculum parameters must be finite")
        if not 0 < self.config["uniform_mix"] <= 1:
            raise ValueError("uniform_mix must be in (0, 1] to preserve exploration")
        if not 0 <= self.config["ema_decay"] < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.config["min_group_std"] < 0 or self.config["variance_scale"] <= 0:
            raise ValueError("Require min_group_std>=0 and variance_scale>0")
        if any(self.config[name] < 0 for name in ("variance_weight", "success_weight", "novelty_weight")):
            raise ValueError("Priority component weights must be nonnegative")
        if self.config["priority_floor"] <= 0 or not 0 < self.config["repo_repeat_penalty"] <= 1:
            raise ValueError("Require a positive priority floor and repo_repeat_penalty in (0, 1]")
        self.tasks = {}
        for task in tasks:
            item = {key: task[key] for key in ("id", "repo", "base_commit")}
            if item["id"] in self.tasks:
                raise ValueError(f"Duplicate task in sampler: {item['id']}")
            self.tasks[item["id"]] = item
        if not self.tasks:
            raise ValueError("TaskSampler needs at least one training task")
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.draws = 0
        self.stats = {identifier: {
            "groups": 0, "trajectories": 0, "successes": 0,
            "reward_mean_ema": 0.0, "reward_variance_ema": 0.0,
            "informative_groups": 0, "zero_variance_groups": 0,
            "last_selected_draw": None,
        } for identifier in self.tasks}

    def _priority(self, identifier, used_repos):
        item = self.stats[identifier]
        # Beta(1,1) posterior keeps estimates usable after a single 0/1 group.
        success_rate = (item["successes"] + 1.0) / (item["trajectories"] + 2.0)
        medium_difficulty = 4.0 * success_rate * (1.0 - success_rate)
        variance = item["reward_variance_ema"]
        reward_signal = variance / (variance + self.config["variance_scale"])
        novelty = 1.0 / math.sqrt(1.0 + item["groups"])
        priority = (self.config["priority_floor"] +
                    self.config["success_weight"] * medium_difficulty +
                    self.config["variance_weight"] * reward_signal +
                    self.config["novelty_weight"] * novelty)
        if self.tasks[identifier]["repo"] in used_repos:
            priority *= self.config["repo_repeat_penalty"]
        return priority

    def choose(self, exclude_ids=None, used_repos=None):
        """Draw once, with exact probability after caller's batch exclusions.

        Pass every task already attempted in this collection round in
        exclude_ids, including discarded zero-variance groups. This prevents
        retrying the same task until it happens to become informative.
        """
        excluded, repos = set(exclude_ids or []), set(used_repos or [])
        eligible = [identifier for identifier in self.tasks if identifier not in excluded]
        if not eligible:
            return None
        unseen = [identifier for identifier in eligible if self.stats[identifier]["groups"] == 0]
        phase = "coverage" if unseen else "adaptive"
        candidates = unseen or eligible
        priorities = [self._priority(identifier, repos) for identifier in candidates]
        total = sum(priorities)
        mix = self.config["uniform_mix"]
        probabilities = [mix / len(candidates) + (1.0 - mix) * weight / total for weight in priorities]
        index = self.rng.choices(range(len(candidates)), weights=probabilities, k=1)[0]
        identifier = candidates[index]
        self.draws += 1
        self.stats[identifier]["last_selected_draw"] = self.draws
        return {
            **self.tasks[identifier],
            "selection_probability": probabilities[index],
            "selection_phase": phase,
            "eligible_count": len(eligible),
            "candidate_count": len(candidates),
            "priority": priorities[index],
            "draw": self.draws,
        }

    def observe(self, task_id, rewards, resolved):
        """Record a complete group's scalar rewards and official binary results."""
        if task_id not in self.tasks:
            raise ValueError("Cannot observe a task outside the declared training universe")
        values, outcomes = list(rewards), list(resolved)
        if len(values) < 2 or len(values) != len(outcomes):
            raise ValueError("Observe a complete group of at least two trajectories with matching outcomes")
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise ValueError("Incomplete/non-finite rewards are infrastructure errors, not zero scores")
        if any(value not in (False, True, 0, 1) or value is None for value in outcomes):
            raise ValueError("resolved must contain complete official binary outcomes")
        values = [float(value) for value in values]
        mean = statistics.fmean(values)
        variance = statistics.pvariance(values)
        deviation = math.sqrt(variance)
        informative = deviation > self.config["min_group_std"]
        item = self.stats[task_id]
        decay = self.config["ema_decay"] if item["groups"] else 0.0
        item["reward_mean_ema"] = decay * item["reward_mean_ema"] + (1.0 - decay) * mean
        item["reward_variance_ema"] = decay * item["reward_variance_ema"] + (1.0 - decay) * variance
        item["groups"] += 1
        item["trajectories"] += len(values)
        item["successes"] += sum(int(value) for value in outcomes)
        item["informative_groups"] += int(informative)
        item["zero_variance_groups"] += int(not informative)
        return {
            "task_id": task_id, "group_size": len(values),
            "reward_mean": mean, "reward_std": deviation,
            "official_success_rate": sum(outcomes) / len(outcomes),
            "informative": informative,
            "cumulative_groups": item["groups"],
            "cumulative_success_rate": item["successes"] / item["trajectories"],
        }

    def state_dict(self):
        return {
            "distribution": "coverage_then_adaptive_mixture; not uniform-risk-unbiased",
            "seed": self.seed, "draws": self.draws,
            "config": dict(self.config),
            "tasks": [dict(item) for item in self.tasks.values()],
            "statistics": {identifier: dict(item) for identifier, item in self.stats.items()},
            "rng_state": self.rng.getstate(),
        }
