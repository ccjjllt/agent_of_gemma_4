"""Metadata-only scheduling with separate capability and task-health feedback.

Adaptive selection and zero-variance filtering deliberately change the training
distribution. The returned probability is conditional on the current eligible
set; it is recorded for audit, not presented as an unbiased uniform-task loss.
This module never downloads, rewrites, or returns question/patch/test contents.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import re
import statistics
import time


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

    Only fully verified trajectories are observed. Infrastructure failures go
    to record_health(), never into observe() as zero rewards. The caller also
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
            "health_backoff_seconds": float(supplied.get("health_backoff_seconds", 60.0)),
            "health_backoff_max_seconds": float(supplied.get("health_backoff_max_seconds", 1800.0)),
        }
        if not all(math.isfinite(value) for value in self.config.values()):
            raise ValueError("Curriculum parameters must be finite")
        pure_uniform = self.config["uniform_mix"] == 1.0 and self.config["coverage_mix"] == 0.0
        if not 0 < self.config["uniform_mix"] <= 1:
            raise ValueError("uniform_mix must be in (0, 1]; 1 is the uniform-sampling comparison")
        if not pure_uniform and not 0 <= self.config["coverage_mix"] < 1 - self.config["uniform_mix"]:
            raise ValueError("coverage_mix must be nonnegative and leave positive adaptive probability mass")
        if not 0 <= self.config["ema_decay"] < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.config["min_group_std"] < 0 or self.config["variance_scale"] <= 0:
            raise ValueError("Require min_group_std>=0 and variance_scale>0")
        if any(self.config[name] < 0 for name in ("variance_weight", "success_weight", "novelty_weight")):
            raise ValueError("Priority component weights must be nonnegative")
        if self.config["priority_floor"] <= 0 or not 0 < self.config["repo_repeat_penalty"] <= 1:
            raise ValueError("Require a positive priority floor and repo_repeat_penalty in (0, 1]")
        if not 0 < self.config["health_backoff_seconds"] <= self.config["health_backoff_max_seconds"]:
            raise ValueError("Health backoff must be positive and no greater than its maximum")
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
            "learning_signal_second_moment_ema": 0.0,
            "reward_signature": None, "reward_signature_groups": 0,
            "informative_groups": 0, "zero_variance_groups": 0,
            "no_signal_groups": 0, "missing_evidence_groups": 0, "uncertainty_blocked_groups": 0,
            "last_signal_status": None,
            "last_informative": None, "last_observed_policy_version": None,
            "last_selected_draw": None,
        } for identifier in self.tasks}
        self.health = {identifier: self._initial_health() for identifier in self.tasks}
        self.health_records = {}

    @staticmethod
    def _initial_health(*, history_complete=True, ever_selected=False):
        # Counters describe the recorded history, not inferred pre-migration
        # health. Capability statistics remain in self.stats independently.
        return {"history_complete": history_complete, "ever_selected": ever_selected,
                "selection_attempts": 0, "healthy_groups": 0, "failed_groups": 0,
                "completed_slots": 0, "exhausted_slots": 0,
                "consecutive_failures": 0, "retry_after_unix": 0.0,
                "last_outcome": None, "last_group_id": None}

    def record_health(self, task_id, *, group_id, completed_slots, exhausted_slots):
        """Record one group's runtime outcome exactly once, without reward data.

        A partially usable group with an exhausted infrastructure slot backs
        off immediately. If collection resumes, its failure record may acquire
        more terminal slots without counting another failure or extending the
        cooldown. Budget cancellation alone must not call this method.
        """
        if task_id not in self.tasks:
            raise ValueError("Cannot record health outside the declared training universe")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("Health feedback requires a stable nonempty group id")
        if (any(type(value) is not int or value < 0 for value in (completed_slots, exhausted_slots))
                or completed_slots + exhausted_slots == 0):
            raise ValueError("Health feedback requires nonnegative counts and at least one terminal slot")
        identity = {"task_id": task_id, "completed_slots": completed_slots, "exhausted_slots": exhausted_slots}
        previous = self.health_records.get(group_id)
        if previous is not None:
            if all(previous[key] == value for key, value in identity.items()):
                return dict(previous)
            if (previous["task_id"] != task_id or not previous["exhausted_slots"]
                    or completed_slots < previous["completed_slots"]
                    or exhausted_slots < previous["exhausted_slots"]):
                raise ValueError("Health feedback changed for an already recorded group")
            item = self.health[task_id]
            item["completed_slots"] += completed_slots - previous["completed_slots"]
            item["exhausted_slots"] += exhausted_slots - previous["exhausted_slots"]
            previous.update(identity)
            # In particular, retain recorded_at, streak and retry_after, even
            # when this older group's additional slots arrive after newer work.
            return dict(previous)
        item = self.health[task_id]
        item["ever_selected"] = True
        item["completed_slots"] += completed_slots
        item["exhausted_slots"] += exhausted_slots
        now = time.time()
        if exhausted_slots:
            item["failed_groups"] += 1
            item["consecutive_failures"] += 1
            base, maximum = self.config["health_backoff_seconds"], self.config["health_backoff_max_seconds"]
            exponent = min(item["consecutive_failures"] - 1, max(0, math.ceil(math.log2(maximum / base))))
            delay = min(maximum, base * 2 ** exponent)
            item["retry_after_unix"] = now + delay
            item["last_outcome"] = "infrastructure_failure"
        else:
            item["healthy_groups"] += 1
            item["consecutive_failures"] = 0
            item["retry_after_unix"] = 0.0
            item["last_outcome"] = "healthy"
        item["last_group_id"] = group_id
        record = {**identity, "group_id": group_id, "recorded_at_unix": now,
                  "outcome": item["last_outcome"], "consecutive_failures": item["consecutive_failures"],
                  "retry_after_unix": item["retry_after_unix"]}
        self.health_records[group_id] = record
        return dict(record)

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
        # With interval rewards, true reward variance can be unknown. Rank the
        # observed usable learning signal instead of inventing a scalar reward.
        variance = item["learning_signal_second_moment_ema"]
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
        Coverage is for never-scheduled tasks. Previously scheduled tasks with
        no capability feedback get uniform rechecks after their health cooldown.
        Empty pools transfer their non-uniform mass to the other available pool;
        repository weighting never changes the eligible uniform floor.
        """
        excluded, repos = set(exclude_ids or []), set(used_repos or [])
        now = time.time()
        available = [identifier for identifier in self.tasks if identifier not in excluded]
        cooling = [identifier for identifier in available if self.health[identifier]["retry_after_unix"] > now]
        eligible = [identifier for identifier in available if identifier not in cooling]
        health_status = {
            "cooldown_blocked": bool(cooling and not eligible), "cooldown_count": len(cooling),
            "next_retry_unix": min((self.health[identifier]["retry_after_unix"] for identifier in cooling), default=None),
        }
        observed = [identifier for identifier in eligible if self.stats[identifier]["groups"] > 0]
        unseen = [identifier for identifier in eligible if self.stats[identifier]["groups"] == 0
                  and not self.health[identifier]["ever_selected"]]
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
            return {"candidates": [], "selection_phase": "health_cooldown" if cooling else "exhausted", "eligible_count": 0,
                    "candidate_count": 0, "observed_candidate_count": 0, "unseen_candidate_count": 0,
                    "adaptive_candidate_count": 0, "suppressed_zero_signal_count": 0,
                    "effective_pool_masses": {"uniform": 0.0, "coverage": 0.0, "adaptive": 0.0},
                    "observed_probability_mass": 0.0, "unseen_probability_mass": 0.0,
                    "health_reprobe_probability_mass": 0.0,
                    "feedback_revision": self.feedback_revision, "policy_version": self.policy_version, **health_status}
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
                "coverage": coverage * coverage_weights[identifier] / total_coverage if identifier in coverage_weights else 0.0,
                "adaptive": (adaptive * priorities[identifier] / total_priority
                             if seen and identifier not in suppressed else 0.0),
            }
            last_version = self.stats[identifier]["last_observed_policy_version"]
            feedback_status = (
                ("unseen" if identifier in coverage_weights else "awaiting_capability_feedback") if not seen else (
                    "uncertain_reward_current_policy" if self.stats[identifier]["last_signal_status"] == "no_provable_signal_with_missing"
                    else "no_signal_current_policy") if identifier in suppressed
                else "reprobe_after_policy_update" if last_version < self.policy_version
                else "informative_current_policy"
            )
            rows.append({"id": identifier, "selected_pool": "observed" if seen else (
                             "unseen" if identifier in coverage_weights else "health_reprobe"),
                         "priority": priorities[identifier] if seen else coverage_weights.get(identifier, 0.0),
                         "adaptive_eligible": seen and identifier not in suppressed,
                         "feedback_status": feedback_status, "last_observed_policy_version": last_version,
                         "feedback_policy_age": self.policy_version - last_version if seen else None,
                         "selection_probability": math.fsum(components.values()),
                         "selection_probability_components": components})
        phase = ("uniform_only" if self.config["uniform_mix"] == 1.0 else
                 "unseen_only" if len(unseen) == len(eligible) else
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
            "health_reprobe_probability_mass": math.fsum(row["selection_probability"] for row in rows if row["selected_pool"] == "health_reprobe"),
            "feedback_revision": self.feedback_revision, "policy_version": self.policy_version,
            **health_status,
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
        self.health[identifier]["ever_selected"] = True
        self.health[identifier]["selection_attempts"] += 1
        return {
            **self.tasks[identifier], **selected, **distribution,
            "draw": self.draws,
        }

    def observe(self, task_id, rewards, resolved, *, policy_version=None, reward_signature="default",
                learning_signal=None):
        """Record verified trajectories' rewards, never missing slots as zeros."""
        if task_id not in self.tasks:
            raise ValueError("Cannot observe a task outside the declared training universe")
        version = self.policy_version if policy_version is None else policy_version
        if type(version) is not int or version != self.policy_version:
            raise ValueError("Feedback must come from the current collection policy")
        if not isinstance(reward_signature, str) or not reward_signature:
            raise ValueError("A nonempty reward signature is required")
        values, outcomes = list(rewards), list(resolved)
        if len(values) < 2 or len(values) != len(outcomes):
            raise ValueError("Observe at least two verified trajectories with matching outcomes")
        if any(value is not None and not math.isfinite(float(value)) for value in values):
            raise ValueError("Non-finite rewards are not valid observations")
        if any(value not in (False, True, 0, 1) or value is None for value in outcomes):
            raise ValueError("resolved must contain complete official binary outcomes")
        values = [float(value) if value is not None else None for value in values]
        reward_complete = all(value is not None for value in values)
        mean = statistics.fmean(values) if reward_complete else None
        variance = statistics.pvariance(values) if reward_complete else None
        deviation = math.sqrt(variance) if reward_complete else None
        if learning_signal is None:
            if not reward_complete:
                raise ValueError("Unknown rewards require an explicit evidence-based learning signal")
            signal = [value - mean for value in values]
        else:
            signal = list(learning_signal)
            if (len(signal) != len(values) or
                    any(value is None or not math.isfinite(float(value)) for value in signal)):
                raise ValueError("Learning signal must be finite and cover all observed trajectories")
            signal = [float(value) for value in signal]
        second_moment = statistics.fmean(value * value for value in signal)
        if not math.isfinite(second_moment):
            raise ValueError("Learning signal magnitude must be finite")
        signal_rms = math.sqrt(second_moment)
        informative = signal_rms > self.config["min_group_std"]
        signal_status = ("complete_evidence_informative" if reward_complete else "informative_with_missing_evidence") if informative else (
            "complete_evidence_zero_signal" if reward_complete else "no_provable_signal_with_missing")
        item = self.stats[task_id]
        decay = self.config["ema_decay"] if item["groups"] else 0.0
        success_rate = sum(outcomes) / len(outcomes)
        signature_changed = item["reward_signature"] != reward_signature
        reward_decay = 0.0 if signature_changed else decay
        item["success_rate_ema"] = decay * item["success_rate_ema"] + (1.0 - decay) * success_rate
        item["reward_mean_ema"] = (None if mean is None else mean if item["reward_mean_ema"] is None
                                    else reward_decay * item["reward_mean_ema"] + (1.0 - reward_decay) * mean)
        item["reward_variance_ema"] = (None if variance is None else variance if item["reward_variance_ema"] is None
                                        else reward_decay * item["reward_variance_ema"] + (1.0 - reward_decay) * variance)
        item["learning_signal_second_moment_ema"] = (reward_decay * item["learning_signal_second_moment_ema"]
                                                    + (1.0 - reward_decay) * second_moment)
        item["reward_signature_groups"] = 1 if signature_changed else item["reward_signature_groups"] + 1
        item["reward_signature"] = reward_signature
        item["last_informative"] = informative
        item["last_signal_status"] = signal_status
        item["last_observed_policy_version"] = version
        item["groups"] += 1
        item["trajectories"] += len(values)
        item["successes"] += sum(int(value) for value in outcomes)
        item["informative_groups"] += int(informative)
        item["zero_variance_groups"] += int(reward_complete and not informative)
        item["no_signal_groups"] += int(not informative)
        item["missing_evidence_groups"] += int(not reward_complete)
        item["uncertainty_blocked_groups"] += int(not reward_complete and not informative)
        self.feedback_revision += 1
        return {
            "task_id": task_id, "group_size": len(values),
            "reward_mean": mean, "reward_std": deviation,
            "reward_complete": reward_complete, "learning_signal_rms": signal_rms,
            "learning_signal_second_moment": second_moment, "signal_status": signal_status,
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
            "distribution": "health_filtered_uniform_plus_never_selected_coverage_plus_signal_aware_priority; not uniform-risk-unbiased",
            "seed": self.seed, "draws": self.draws,
            "feedback_revision": self.feedback_revision, "policy_version": self.policy_version,
            "config": dict(self.config),
            "tasks": [dict(item) for item in self.tasks.values()],
            "statistics": {identifier: dict(item) for identifier, item in self.stats.items()},
            "health_schema_version": 1,
            "health": copy.deepcopy(self.health),
            "health_records": copy.deepcopy(self.health_records),
            "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state):
        """Restore only an identical public task universe and sampling recipe."""
        saved_config = dict(state.get("config") or {})
        legacy_health = "health_schema_version" not in state
        if legacy_health:
            if "health" in state or "health_records" in state:
                raise ValueError("Sampler health state has no supported schema version")
            # An older recipe had no cooldown policy. Upgrade only to the
            # documented defaults, not silently to a different supplied policy.
            saved_config.setdefault("health_backoff_seconds", 60.0)
            saved_config.setdefault("health_backoff_max_seconds", 1800.0)
        elif type(state["health_schema_version"]) is not int or state["health_schema_version"] != 1:
            raise ValueError("Unsupported sampler health schema")
        if saved_config != self.config or state.get("tasks") != list(self.tasks.values()):
            raise ValueError("Sampler checkpoint belongs to a different task universe or recipe")
        if set(state.get("statistics", {})) != set(self.tasks):
            raise ValueError("Sampler checkpoint is missing task statistics")
        for name in ("draws", "feedback_revision", "policy_version", "seed"):
            if type(state.get(name)) is not int or state[name] < 0:
                raise ValueError("Invalid sampler checkpoint counter")
        if state["seed"] != self.seed:
            raise ValueError("Sampler seed differs from the checkpoint")
        for identifier, item in state["statistics"].items():
            if set(item) != set(self.stats[identifier]):
                raise ValueError("Sampler checkpoint schema differs from this implementation")
        if legacy_health:
            health = {identifier: self._initial_health(history_complete=False, ever_selected=(
                item["last_selected_draw"] is not None or item["groups"] > 0))
                for identifier, item in state["statistics"].items()}
            health_records = {}
        else:
            health = copy.deepcopy(state.get("health"))
            health_records = copy.deepcopy(state.get("health_records"))
            if not isinstance(health, dict) or set(health) != set(self.tasks) or not isinstance(health_records, dict):
                raise ValueError("Sampler checkpoint is missing task health records")
            integer_fields = ("selection_attempts", "healthy_groups", "failed_groups", "completed_slots",
                              "exhausted_slots", "consecutive_failures")
            record_fields = {"task_id", "group_id", "completed_slots", "exhausted_slots", "recorded_at_unix",
                             "outcome", "consecutive_failures", "retry_after_unix"}

            def nonnegative_time(value):
                return type(value) in (float, int) and math.isfinite(value) and value >= 0

            for group_id, record in health_records.items():
                if (not isinstance(group_id, str) or not group_id or not isinstance(record, dict)
                        or set(record) != record_fields or record["group_id"] != group_id
                        or record["task_id"] not in self.tasks
                        or any(type(record[key]) is not int or record[key] < 0
                               for key in ("completed_slots", "exhausted_slots", "consecutive_failures"))
                        or not record["completed_slots"] + record["exhausted_slots"]
                        or not all(nonnegative_time(record[key]) for key in ("recorded_at_unix", "retry_after_unix"))):
                    raise ValueError("Invalid sampler group health record")
                failed = bool(record["exhausted_slots"])
                if (record["outcome"] != ("infrastructure_failure" if failed else "healthy")
                        or bool(record["consecutive_failures"]) != failed
                        or (failed and record["retry_after_unix"] <= record["recorded_at_unix"])
                        or (not failed and record["retry_after_unix"] != 0)):
                    raise ValueError("Sampler group health outcome contradicts its counts")
            for identifier, item in health.items():
                if (not isinstance(item, dict) or set(item) != set(self._initial_health())
                        or any(type(item[key]) is not int or item[key] < 0 for key in integer_fields)
                        or type(item["history_complete"]) is not bool or type(item["ever_selected"]) is not bool
                        or not nonnegative_time(item["retry_after_unix"])):
                    raise ValueError("Invalid sampler task health snapshot")
                records = [row for row in health_records.values() if row["task_id"] == identifier]
                expected = {"healthy_groups": sum(not row["exhausted_slots"] for row in records),
                            "failed_groups": sum(bool(row["exhausted_slots"]) for row in records),
                            "completed_slots": sum(row["completed_slots"] for row in records),
                            "exhausted_slots": sum(row["exhausted_slots"] for row in records)}
                if any(item[key] != value for key, value in expected.items()):
                    raise ValueError("Sampler task health counters differ from group evidence")
                last = health_records.get(item["last_group_id"])
                if records:
                    if (last is None or last["task_id"] != identifier or not item["ever_selected"]
                            or any(item[key] != last[key] for key in ("consecutive_failures", "retry_after_unix"))
                            or item["last_outcome"] != last["outcome"]):
                        raise ValueError("Sampler task health tail differs from group evidence")
                elif (item["last_group_id"] is not None or item["last_outcome"] is not None
                      or item["consecutive_failures"] or item["retry_after_unix"]):
                    raise ValueError("Sampler task health has no supporting group evidence")
                if item["selection_attempts"] and not item["ever_selected"]:
                    raise ValueError("Scheduled task cannot be unseen in health state")

        def tuples(value):
            return tuple(tuples(item) for item in value) if isinstance(value, (tuple, list)) else value

        restored_rng = random.Random()
        restored_rng.setstate(tuples(state["rng_state"]))
        # Validate the serializable snapshot before replacing live state.
        json.dumps(state, allow_nan=False)
        self.stats = copy.deepcopy(state["statistics"])
        self.health, self.health_records = health, health_records
        self.rng = restored_rng
        self.draws = state["draws"]
        self.feedback_revision = state["feedback_revision"]
        self.policy_version = state["policy_version"]
