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
    """Mix coverage and feedback allocation without a full-catalog warmup.

    Only complete groups are observed. Infrastructure failures must be handled
    by the caller and must not be passed as zeros to observe(). The caller also
    caps replacement groups per collection round; this class never resamples a
    failed or zero-variance group in an internal loop.
    """

    def __init__(self, tasks, config=None, seed=3407):
        supplied = dict(config or {})
        self.config = {
            "uniform_mix": float(supplied.get("uniform_mix", 0.2)),
            "coverage_mix": float(supplied.get("coverage_mix", 0.3)),
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
        if not 0 < self.config["uniform_mix"] < 1:
            raise ValueError("uniform_mix must be in (0, 1) to preserve exploration and feedback allocation")
        if not 0 <= self.config["coverage_mix"] < 1 - self.config["uniform_mix"]:
            raise ValueError("coverage_mix must be nonnegative and leave positive adaptive probability mass")
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
        self.feedback_revision = 0
        self.policy_version = 0
        self.stats = {identifier: {
            "groups": 0, "trajectories": 0, "successes": 0,
            "success_rate_ema": 0.0,
            "reward_mean_ema": 0.0, "reward_variance_ema": 0.0,
            "reward_signature": None, "reward_signature_groups": 0,
            "informative_groups": 0, "zero_variance_groups": 0,
            "last_informative": None, "last_observed_policy_version": None,
            "last_selected_draw": None,
        } for identifier in self.tasks}

    def set_policy_version(self, version):
        """Advance after real optimizer updates, never after a zero-signal batch.

        Old feedback remains an explicitly dated prior for a nearby new policy.
        Only same-policy zero-signal suppression is lifted on an actual advance.
        """
        if type(version) is not int or version < self.policy_version:
            raise ValueError("policy_version must be a nonnegative, nondecreasing integer")
        self.policy_version = version

    def _priority(self, identifier, used_repos):
        item = self.stats[identifier]
        # Use the same recency scale as reward variation. Lifetime success
        # totals are audit data, not a stationary difficulty estimate.
        success_rate = item["success_rate_ema"]
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

    def selection_distribution(self, exclude_ids=None, used_repos=None):
        """Compute the actual marginal distribution without drawing or mutating.

        Pool masses, not the relative counts of visited/unvisited tasks, control
        feedback allocation. Uniform exploration is over ALL eligible tasks.
        Empty pools transfer their non-uniform mass to the other available pool;
        repository weighting never changes the uniform exploration floor.
        """
        excluded, repos = set(exclude_ids or []), set(used_repos or [])
        eligible = [identifier for identifier in self.tasks if identifier not in excluded]
        observed = [identifier for identifier in eligible if self.stats[identifier]["groups"] > 0]
        unseen = [identifier for identifier in eligible if self.stats[identifier]["groups"] == 0]
        suppressed = {identifier for identifier in observed
                      if self.stats[identifier]["last_informative"] is False
                      and self.stats[identifier]["last_observed_policy_version"] == self.policy_version}
        adaptive_candidates = [identifier for identifier in observed if identifier not in suppressed]
        uniform = self.config["uniform_mix"]
        coverage = self.config["coverage_mix"]
        adaptive = 1.0 - uniform - coverage
        if not adaptive_candidates and unseen:
            coverage += adaptive
            adaptive = 0.0
        elif adaptive_candidates and not unseen:
            adaptive += coverage
            coverage = 0.0
        elif not adaptive_candidates and not unseen:
            # No unvisited or currently useful observed candidate remains.
            # Uniform rechecks preserve a chance to detect stochastic signal;
            # they are not relabeled as adaptive exploitation.
            uniform, coverage, adaptive = 1.0, 0.0, 0.0
        if not eligible:
            return {"candidates": [], "selection_phase": "exhausted", "eligible_count": 0,
                    "candidate_count": 0, "observed_candidate_count": 0, "unseen_candidate_count": 0,
                    "adaptive_candidate_count": 0, "suppressed_zero_signal_count": 0,
                    "effective_pool_masses": {"uniform": 0.0, "coverage": 0.0, "adaptive": 0.0},
                    "observed_probability_mass": 0.0, "unseen_probability_mass": 0.0,
                    "feedback_revision": self.feedback_revision, "policy_version": self.policy_version}
        priorities = {identifier: self._priority(identifier, repos) for identifier in observed}
        # Unvisited tasks have no measured learning signal. Their coverage
        # weights express only the existing within-batch repository diversity.
        coverage_weights = {identifier: (self.config["repo_repeat_penalty"]
                                        if self.tasks[identifier]["repo"] in repos else 1.0)
                            for identifier in unseen}
        total_priority = math.fsum(priorities[identifier] for identifier in adaptive_candidates)
        total_coverage = math.fsum(coverage_weights.values())
        rows = []
        for identifier in eligible:
            seen = identifier in priorities
            components = {
                "uniform": uniform / len(eligible),
                "coverage": coverage * coverage_weights[identifier] / total_coverage if not seen else 0.0,
                "adaptive": (adaptive * priorities[identifier] / total_priority
                             if seen and identifier not in suppressed else 0.0),
            }
            last_version = self.stats[identifier]["last_observed_policy_version"]
            feedback_status = (
                "unseen" if not seen else "no_signal_current_policy" if identifier in suppressed
                else "reprobe_after_policy_update" if last_version < self.policy_version
                else "informative_current_policy"
            )
            rows.append({"id": identifier, "selected_pool": "observed" if seen else "unseen",
                         "priority": priorities[identifier] if seen else coverage_weights[identifier],
                         "adaptive_eligible": seen and identifier not in suppressed,
                         "feedback_status": feedback_status, "last_observed_policy_version": last_version,
                         "feedback_policy_age": self.policy_version - last_version if seen else None,
                         "selection_probability": math.fsum(components.values()),
                         "selection_probability_components": components})
        phase = ("unseen_only" if not observed else
                 "feedback_mixture" if adaptive_candidates and unseen else
                 "observed_only" if adaptive_candidates else
                 "coverage_with_uniform_recheck" if unseen else "uniform_recheck")
        return {
            "candidates": rows,
            "selection_phase": phase,
            "eligible_count": len(eligible), "candidate_count": len(eligible),
            "observed_candidate_count": len(observed), "unseen_candidate_count": len(unseen),
            "adaptive_candidate_count": len(adaptive_candidates), "suppressed_zero_signal_count": len(suppressed),
            "effective_pool_masses": {"uniform": uniform, "coverage": coverage, "adaptive": adaptive},
            "observed_probability_mass": math.fsum(row["selection_probability"] for row in rows if row["selected_pool"] == "observed"),
            "unseen_probability_mass": math.fsum(row["selection_probability"] for row in rows if row["selected_pool"] == "unseen"),
            "feedback_revision": self.feedback_revision, "policy_version": self.policy_version,
        }

    def choose(self, exclude_ids=None, used_repos=None):
        """Draw once from the recorded mixture after caller's batch exclusions.

        Pass every task already attempted in this collection round in
        exclude_ids, including discarded zero-variance groups. This prevents
        retrying the same task until it happens to become informative.
        """
        distribution = self.selection_distribution(exclude_ids, used_repos)
        candidates = distribution.pop("candidates")
        if not candidates:
            return None
        probabilities = [row["selection_probability"] for row in candidates]
        index = self.rng.choices(range(len(candidates)), weights=probabilities, k=1)[0]
        selected = candidates[index]
        identifier = selected["id"]
        self.draws += 1
        self.stats[identifier]["last_selected_draw"] = self.draws
        return {
            **self.tasks[identifier], **selected, **distribution,
            "draw": self.draws,
        }

    def observe(self, task_id, rewards, resolved, *, policy_version=None, reward_signature="default"):
        """Record a complete group's scalar rewards and official binary results."""
        if task_id not in self.tasks:
            raise ValueError("Cannot observe a task outside the declared training universe")
        version = self.policy_version if policy_version is None else policy_version
        if type(version) is not int or version != self.policy_version:
            raise ValueError("Feedback must come from the current collection policy")
        if not isinstance(reward_signature, str) or not reward_signature:
            raise ValueError("A nonempty reward signature is required")
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
        success_rate = sum(outcomes) / len(outcomes)
        signature_changed = item["reward_signature"] != reward_signature
        reward_decay = 0.0 if signature_changed else decay
        item["success_rate_ema"] = decay * item["success_rate_ema"] + (1.0 - decay) * success_rate
        item["reward_mean_ema"] = reward_decay * item["reward_mean_ema"] + (1.0 - reward_decay) * mean
        item["reward_variance_ema"] = reward_decay * item["reward_variance_ema"] + (1.0 - reward_decay) * variance
        item["reward_signature_groups"] = 1 if signature_changed else item["reward_signature_groups"] + 1
        item["reward_signature"] = reward_signature
        item["last_informative"] = informative
        item["last_observed_policy_version"] = version
        item["groups"] += 1
        item["trajectories"] += len(values)
        item["successes"] += sum(int(value) for value in outcomes)
        item["informative_groups"] += int(informative)
        item["zero_variance_groups"] += int(not informative)
        self.feedback_revision += 1
        return {
            "task_id": task_id, "group_size": len(values),
            "reward_mean": mean, "reward_std": deviation,
            "official_success_rate": success_rate,
            "informative": informative,
            "cumulative_groups": item["groups"],
            "cumulative_success_rate": item["successes"] / item["trajectories"],
            "success_rate_ema": item["success_rate_ema"],
            "reward_signature": reward_signature, "reward_statistics_reset": signature_changed,
            "policy_version": version,
            "feedback_revision": self.feedback_revision,
        }

    def state_dict(self):
        return {
            "distribution": "uniform_plus_unseen_coverage_plus_signal_aware_observed_priority; not uniform-risk-unbiased",
            "seed": self.seed, "draws": self.draws,
            "feedback_revision": self.feedback_revision, "policy_version": self.policy_version,
            "config": dict(self.config),
            "tasks": [dict(item) for item in self.tasks.values()],
            "statistics": {identifier: dict(item) for identifier, item in self.stats.items()},
            "rng_state": self.rng.getstate(),
        }
