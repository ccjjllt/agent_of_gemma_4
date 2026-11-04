"""Prepare one-factor recipes and compare recorded research evidence on CPU.

This command never launches training/evaluation or chooses a winning recipe.
Missing outcomes stay unknown. Actual run recipes must match the experiment
before any between-run differences are reported.
"""
from __future__ import annotations

import argparse
import copy
from datetime import date, datetime, time
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import tomllib

from .rewards import reward_objective
from .lr_schedule import SCHEDULE_VERSION, lr_multiplier, resolve_schedule


_REWARD_WEIGHTS = {"resolved_weight", "repair_weight", "regression_weight",
                   "efficiency_weight", "repetition_weight"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".ablation-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path):
    if path is None or not Path(path).is_file():
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _differences(before, after, path=()):
    changes = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(before.keys() | after.keys()):
            if key not in before or key not in after:
                changes.append({"path": ".".join((*path, key)), "keys": list((*path, key)),
                                "before_present": key in before, "after_present": key in after,
                                "before": before.get(key), "after": after.get(key)})
            else:
                changes.extend(_differences(before[key], after[key], (*path, key)))
    elif before != after:
        changes.append({"path": ".".join(path), "keys": list(path),
                        "before_present": True, "after_present": True,
                        "before": before, "after": after})
    return changes


def _positive_integer(value):
    if isinstance(value, bool) or not re.fullmatch(r"[1-9][0-9]*", str(value)):
        raise ValueError("Expected a positive integer")
    return int(value)


def _explicit_scheduler(recipe):
    result = copy.deepcopy(recipe)
    grpo = result["grpo"]
    effective = grpo.get("scheduler_total_steps")
    if effective is None:
        effective = (_positive_integer(result["run"]["max_batches"])
                     * _positive_integer(grpo["num_iterations"]))
    grpo["scheduler_total_steps"] = _positive_integer(effective)
    contract = resolve_schedule(grpo, total_steps=grpo["scheduler_total_steps"],
                                learning_rate=result["run"]["learning_rate"])
    grpo.update(warmup_steps=contract["warmup_steps"], lr_schedule_version=contract["version"],
                lr_scheduler_type=contract["type"])
    return result


def _schedule_preview(recipe):
    grpo = recipe["grpo"]
    contract = resolve_schedule(grpo, total_steps=grpo["scheduler_total_steps"],
                                learning_rate=recipe["run"]["learning_rate"])
    calls = _positive_integer(recipe["run"]["max_batches"]) * _positive_integer(grpo["num_iterations"])
    return {
        "contract": contract, "maximum_planned_optimizer_attempts": calls,
        "first_scheduled_update_lrs": [contract["base_lr"] * lr_multiplier(
            index, contract["total_steps"], contract["warmup_steps"]) for index in range(min(calls, 8))],
        "scope": "Preview assumes every planned cohort exists and each optimizer attempt is applied; skipped attempts do not advance this LR index. No updates are executed.",
    }


def make_candidate(parent, factor, value):
    """Return one semantic intervention, including explicitly coupled fields."""
    candidate = copy.deepcopy(parent)
    if factor == "rank":
        rank = _positive_integer(value)
        old_rank = _positive_integer(parent["policy"]["lora_rank"])
        old_alpha = _positive_integer(parent["policy"]["lora_alpha"])
        alpha = Fraction(old_alpha * rank, old_rank)
        if alpha.denominator != 1:
            raise ValueError("Preserving alpha/r requires an exact integer alpha")
        candidate["policy"].update(lora_rank=rank, lora_alpha=int(alpha))
    elif factor.startswith("reward.") and factor.split(".", 1)[1] in _REWARD_WEIGHTS:
        key = factor.split(".", 1)[1]
        if key not in parent["rewards"]:
            raise ValueError("The baseline must explicitly declare the reward weight")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Reward weights must be finite and nonnegative")
        candidate["rewards"][key] = weight
    elif factor == "sampling":
        if value != "uniform":
            raise ValueError("The sampling intervention is --value uniform")
        if not {"uniform_mix", "coverage_mix"} <= parent["curriculum"].keys():
            raise ValueError("The baseline must explicitly declare both sampling mixture weights")
        candidate["curriculum"].update(uniform_mix=1.0, coverage_mix=0.0)
    elif factor == "reuse":
        if _positive_integer(value) != 1 or parent["grpo"]["num_iterations"] != 2:
            raise ValueError("The reuse intervention changes num_iterations from 2 to 1")
        candidate["grpo"]["num_iterations"] = 1
    elif factor == "scale_rewards":
        if value not in ("group", "none"):
            raise ValueError("The scale_rewards intervention is --value group or --value none")
        if parent["grpo"].get("scale_rewards") not in ("group", "none"):
            raise ValueError("The baseline must explicitly declare grpo.scale_rewards as group or none")
        # Raw reward signals still determine selection and sampler feedback.
        # This intervention changes only the advantages used by the learner.
        candidate["grpo"]["scale_rewards"] = value
    elif factor == "prefix_reuse":
        if (value != "none" or parent["grpo"].get("training_prefix_reuse") != "tree"
                or int(parent["grpo"].get("training_prefix_max_calls", 1)) <= 1):
            raise ValueError("prefix_reuse=none requires a native tree baseline with multiple calls per group")
        # Independent groups use the original backend. This compares the two
        # execution paths, not sharing in isolation from kernel selection.
        candidate["grpo"]["training_prefix_max_calls"] = 1
    else:
        raise ValueError("Choose rank, reward.<one weight>, sampling, reuse, scale_rewards, or prefix_reuse")
    reward_objective(candidate["rewards"])
    if not _differences(parent, candidate):
        raise ValueError("The candidate must change the declared factor")
    return candidate


def _key(value):
    return value if re.fullmatch(r"[A-Za-z0-9_-]+", value) else json.dumps(value, ensure_ascii=False)


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_toml_value(v)}" for k, v in value.items()) + " }"
    raise ValueError(f"Unsupported recipe value: {type(value).__name__}")


def _render_toml(config):
    lines = ["# Single-factor research candidate; all other recipe values are preserved."]

    def table(values, path):
        if path:
            lines.extend(("", "[" + ".".join(map(_key, path)) + "]"))
        for key, value in values.items():
            if not isinstance(value, dict):
                lines.append(f"{_key(key)} = {_toml_value(value)}")
        for key, value in values.items():
            if isinstance(value, dict):
                table(value, (*path, key))

    table(config, ())
    return "\n".join(lines) + "\n"


def prepare_experiment(baseline, output_dir, factor, value):
    baseline, output_dir = Path(baseline), Path(output_dir)
    source = tomllib.loads(baseline.read_text(encoding="utf-8"))
    parent = _explicit_scheduler(source)
    parent_text = _render_toml(parent)
    candidate = make_candidate(parent, factor, value)
    schedules = {"baseline": _schedule_preview(parent), "candidate": _schedule_preview(candidate)}
    if schedules["baseline"]["contract"] != schedules["candidate"]["contract"]:
        raise ValueError("Comparison arms must share the same effective LR schedule")
    candidate_text = _render_toml(candidate)
    if tomllib.loads(candidate_text) != candidate:
        raise ValueError("Candidate serialization changed recipe semantics")
    record = {"schema_version": 1, "record_type": "single_factor_ablation",
              "factor": factor, "value": str(value), "changes": _differences(parent, candidate),
              "baseline_source": str(baseline.resolve()),
              "equivalent_explicitization": _differences(source, parent),
              "parent_recipe_sha256": _digest(parent), "candidate_recipe_sha256": _digest(candidate),
              "lr_schedules": schedules,
              "interpretation": "One candidate only; preparing a recipe does not run an experiment or establish a benefit."}
    if factor == "reuse":
        record["reuse_interpretation"] = (
            "Both arms start at the same LR and share the applied-step schedule, without dummy updates or changing warmup. "
            "Reuse 1 and 2 allow different update counts and computation; this is not an equal-update or equal-compute comparison.")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "parent.toml").write_text(parent_text, encoding="utf-8")
    (output_dir / "candidate.toml").write_text(candidate_text, encoding="utf-8")
    _save_json(output_dir / "change.json", record)
    return record


def _load_experiment(directory):
    directory = Path(directory)
    record = _read_json(directory / "change.json")
    if not record or record.get("record_type") != "single_factor_ablation" or record.get("schema_version") != 1:
        raise ValueError("Expected a prepared single-factor experiment")
    parent = tomllib.loads((directory / "parent.toml").read_text(encoding="utf-8"))
    candidate = tomllib.loads((directory / "candidate.toml").read_text(encoding="utf-8"))
    expected = make_candidate(parent, record["factor"], record["value"])
    if (candidate != expected or _differences(parent, candidate) != record["changes"]
            or _digest(parent) != record["parent_recipe_sha256"]
            or _digest(candidate) != record["candidate_recipe_sha256"]):
        raise ValueError("Experiment recipes no longer match the declared single factor")
    return record, parent, candidate


def _match_planned(actual, planned, path=()):
    for key, value in planned.items():
        if key not in actual:
            raise ValueError("Run recipe is missing " + ".".join((*path, key)))
        if isinstance(value, dict) and isinstance(actual[key], dict):
            _match_planned(actual[key], value, (*path, key))
        elif actual[key] != value:
            raise ValueError("Run differs from its planned recipe at " + ".".join((*path, key)))


def _training_record(summary_path, run_config_path, planned):
    summary = _read_json(summary_path)
    if summary is not None and (summary.get("record_type") != "training_evidence" or summary.get("schema_version") != 1):
        raise ValueError("Expected a training_summary.json evidence record")
    config_path = (Path(run_config_path) if run_config_path is not None else
                   Path(summary_path).parent / "run_config.json" if summary_path is not None else None)
    run = _read_json(config_path)
    recipe = None
    binding = None
    schedule = None
    if run is not None:
        raw_recipe = run.get("recipe")
        if not isinstance(raw_recipe, dict):
            raise ValueError("Actual run_config.json must contain its recipe")
        recipe = _explicit_scheduler(raw_recipe)
        # Planning may resolve the current default, but old run files without
        # a version do not prove that this positive-first-update schedule ran.
        if raw_recipe["grpo"].get("lr_schedule_version") == SCHEDULE_VERSION:
            schedule = _schedule_preview(recipe)["contract"]
        _match_planned(recipe, planned)
        if summary is not None and summary.get("config_sha256") not in (None, _digest(raw_recipe)):
            raise ValueError("Training evidence does not belong to the supplied run recipe")
        existing_binding = run.get("source_binding")
        if isinstance(existing_binding, dict) and all(existing_binding.get(key) is not None for key in
                ("code", "environments", "model", "public_tasks", "selected_tasks")):
            binding = {key: value for key, value in existing_binding.items() if key != "config"}
    report = {"status": "observed" if summary is not None else "unknown",
              "summary_path": str(summary_path) if summary_path is not None else None,
              "run_config_sha256": _file_digest(config_path) if run is not None else None,
              "history_coverage": summary.get("history_coverage") if summary is not None else None,
              "source_binding_available": binding is not None,
              "lr_schedule": schedule,
              "training_population": ({
                  "tasks_sha256": _digest(run["training_tasks"]) if "training_tasks" in run else None,
                  "public_tasks_sha256": run.get("public_tasks_sha256"),
                  "task_id": run.get("task_id"), "task_index": run.get("task_index"),
              } if run is not None else None),
              "collection": summary.get("collection") if summary is not None else None,
              "updates": ({key: summary.get("updates", {}).get(key) for key in
                           ("optimizer_steps", "checkpoint_optimizer_steps", "optimizer_steps_outside_reported_batches",
                            "nonzero_lr_optimizer_steps", "zero_lr_optimizer_steps", "skipped_optimizer_steps",
                            "update_wall_seconds", "time_coverage", "seconds_per_optimizer_step",
                            "seconds_per_nonzero_lr_step")} if summary is not None else None),
              "kl": summary.get("kl") if summary is not None else None,
              "rewards": copy.deepcopy(summary.get("rewards")) if summary is not None else None,
              "training_prefix_reuse": copy.deepcopy(summary.get("updates", {}).get("training_prefix_reuse"))
                  if summary is not None else None,
              "rollout_pre_update": summary.get("rollout_pre_update") if summary is not None else None,
              "post_update_evaluation": None}
    return report, recipe, summary, binding


def _evaluation_record(path, training, summary):
    evaluation = _read_json(path)
    if evaluation is None:
        return {"status": "unknown", "summary": None}
    if evaluation.get("record_type") != "paired_evaluation" or evaluation.get("schema_version") != 1:
        raise ValueError("Expected a paired_evaluation summary")
    declared = evaluation.get("training_run_config_sha256")
    actual = training["run_config_sha256"]
    if declared is not None and actual is not None and declared != actual:
        raise ValueError("Paired evaluation belongs to a different training run")
    known_adapters = {item["sha256"] for item in (summary or {}).get("adapters", []) if item.get("sha256")}
    adapter = evaluation.get("adapter_sha256")
    if adapter is not None and known_adapters and adapter not in known_adapters:
        raise ValueError("Paired evaluation used an adapter absent from the training evidence")
    bound = declared is not None and actual == declared and adapter is not None and adapter in known_adapters
    return {"status": "bound" if bound else "unbound", "summary": evaluation}


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _delta(left, right):
    left, right = _number(left), _number(right)
    return right - left if left is not None and right is not None else None


def _all_zero(values):
    values = [_number(value) for value in values]
    if any(value is not None and value != 0 for value in values):
        return False
    return True if all(value is not None for value in values) else None


def _nested(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _adapter_update_qualification(summary, evaluation):
    """Bind effective updates to the evaluated artifact, never later batches."""
    result = {"status": "unknown", "reason": "Bound evaluation and per-batch update evidence are required",
              "adapter_sha256": _nested(evaluation, "summary", "adapter_sha256"), "batch": None,
              "cumulative_nonzero_lr_optimizer_steps": None, "update_evidence_complete": False,
              "changed_from_initial": None, "initial_sha256": None}
    if summary is None or evaluation.get("status") != "bound":
        return result
    rows = summary.get("batches")
    coverage = summary.get("history_coverage") or {}
    if not isinstance(rows, list) or not rows or any(
            not isinstance(row, dict) or type(row.get("batch")) is not int or row["batch"] < 0 for row in rows):
        result["reason"] = "Per-batch training history is missing or invalid"
        return result
    rows = sorted(rows, key=lambda row: row["batch"])
    ids = [row["batch"] for row in rows]
    if (ids != list(range(len(rows))) or coverage.get("reported_batch_ids") != ids
            or coverage.get("reported_batches") != len(rows) or coverage.get("completed_batches") != len(rows)
            or coverage.get("unreported_completed_batches") != 0):
        result["reason"] = "Completed-batch history is not fully reported and contiguous from initialization"
        return result
    matches = [row for row in rows if row.get("adapter", {}).get("path")
               and row["adapter"].get("sha256") == result["adapter_sha256"]]
    if not matches:
        result["reason"] = "Evaluated adapter has no corresponding exported batch"
        return result
    # An identical artifact exported again cannot borrow the later batch's
    # update count. Hash-only evaluation identity binds conservatively to its
    # earliest export; the actual parameter artifact is the same.
    selected = matches[0]
    result["batch"] = selected["batch"]
    prefix = rows[:selected["batch"] + 1]
    counts = [row.get("updates", {}).get("nonzero_lr_optimizer_steps") for row in prefix]
    if any(type(count) is not int or count < 0 for count in counts):
        result["reason"] = "Effective update counts before this adapter are unknown"
        return result
    if any(type(row.get("updates", {}).get("optimizer_steps")) is int
           and count > row["updates"]["optimizer_steps"] for row, count in zip(prefix, counts)):
        result["reason"] = "Effective update counts exceed recorded optimizer attempts"
        return result
    result.update(cumulative_nonzero_lr_optimizer_steps=sum(counts), update_evidence_complete=True)
    adapter = dict(selected["adapter"])
    for item in summary.get("adapters", []):
        if item.get("batch") == selected["batch"] and item.get("sha256") == result["adapter_sha256"]:
            for key in ("initial_sha256", "changed_from_initial"):
                if adapter.get(key) is None:
                    adapter[key] = item.get(key)
    initial = adapter.get("initial_sha256")
    changed = adapter.get("changed_from_initial")
    if isinstance(initial, str) and initial:
        hash_changed = result["adapter_sha256"] != initial
        if isinstance(changed, bool) and changed != hash_changed:
            result["reason"] = "Adapter change flag contradicts its recorded initial artifact hash"
            return result
        changed = hash_changed
        result["initial_sha256"] = initial
    result["changed_from_initial"] = changed if isinstance(changed, bool) else None
    if sum(counts) == 0:
        result.update(status="invalid", reason="This adapter has no non-skipped nonzero-LR optimizer update; later updates cannot qualify it")
    elif changed is False:
        result.update(status="invalid", reason="The evaluated adapter is unchanged from the initial export")
    elif changed is not True:
        result["reason"] = "Independent adapter change from initialization is not recorded"
    else:
        result.update(status="qualified", reason=None)
    return result


def _reuse_gain_qualification(base, candidate, base_summary, candidate_summary, evaluations):
    arms = {"baseline": _adapter_update_qualification(base_summary, evaluations["baseline"]),
            "candidate": _adapter_update_qualification(candidate_summary, evaluations["candidate"])}
    schedules = [item.get("lr_schedule") for item in (base, candidate)]
    schedule_status, schedule_reason = "confirmed", None
    if any(item is None for item in schedules):
        schedule_status, schedule_reason = "unknown", "Actual run recipes do not both record the LR schedule version"
    elif schedules[0] != schedules[1]:
        schedule_status, schedule_reason = "invalid", "Actual runs have different effective LR schedules"
    elif schedules[0]["first_update_lr"] <= 0:
        schedule_status, schedule_reason = "invalid", "The common schedule does not start with a positive update LR"
    statuses = [arm["status"] for arm in arms.values()]
    status = ("invalid" if schedule_status == "invalid" or "invalid" in statuses else
              "qualified" if schedule_status == "confirmed" and all(value == "qualified" for value in statuses)
              else "unknown")
    reasons = ([schedule_reason] if schedule_reason else []) + [
        f"{name}: {arm['reason']}" for name, arm in arms.items() if arm["reason"]]
    return {"status": status, "reason": "; ".join(reasons) if reasons else None,
            "lr_schedule_status": schedule_status, **arms,
            "scope": "Eligibility to compare trained-adapter outcomes only; positive LR and changed exported weights do not establish learning benefit."}


def _evaluation_delta(base, candidate, comparable, qualification=None):
    result = {"status": "unknown", "reason": "Comparable complete paired evaluations are not available",
              "resolution_rate": None, "paired_resolution_gain": None,
              "repair_fraction": None, "regression_fraction": None, "costs": {}}
    if not comparable or base["status"] != "bound" or candidate["status"] != "bound":
        return result
    a, b = base["summary"], candidate["summary"]
    if any(a.get(key) is None or a.get(key) != b.get(key)
           for key in ("settings_sha256", "task_seed_set_sha256", "backend", "population")):
        result["reason"] = "Evaluation settings, task/seed population or backend differ"
        return result
    if any(not item.get("planned_pairs") or item.get("valid_pairs") != item["planned_pairs"] for item in (a, b)):
        result["reason"] = "Partial summaries do not identify a common complete comparison population"
        return result
    result.update(status="observed", reason=None,
                  resolution_rate=_delta(_nested(a, "paired", "adapter_resolution_rate"),
                                         _nested(b, "paired", "adapter_resolution_rate")),
                  paired_resolution_gain=_delta(_nested(a, "paired", "resolution_delta"),
                                                _nested(b, "paired", "resolution_delta")))
    for field in ("repair", "regression"):
        x, y = _nested(a, "arms", "adapter", field), _nested(b, "arms", "adapter", field)
        if _nested(x, "coverage") == _nested(y, "coverage") == 1:
            result[field + "_fraction"] = _delta(_nested(x, "mean"), _nested(y, "mean"))
    costs_a, costs_b = _nested(a, "arms", "adapter", "costs") or {}, _nested(b, "arms", "adapter", "costs") or {}
    for key in sorted(costs_a.keys() | costs_b.keys()):
        x, y = costs_a.get(key), costs_b.get(key)
        result["costs"][key] = (_delta(_nested(x, "mean"), _nested(y, "mean"))
                                if _nested(x, "coverage") == _nested(y, "coverage") == 1 else None)
    if qualification is not None and qualification["status"] != "qualified":
        result.update(status=qualification["status"], reason=qualification["reason"],
                      resolution_rate=None, paired_resolution_gain=None,
                      repair_fraction=None, regression_fraction=None)
    return result


def compare_experiment(experiment, *, baseline_summary=None, candidate_summary=None,
                       baseline_run_config=None, candidate_run_config=None,
                       baseline_evaluation=None, candidate_evaluation=None):
    record, parent, child = _load_experiment(experiment)
    base, base_recipe, base_raw, base_binding = _training_record(baseline_summary, baseline_run_config, parent)
    candidate, candidate_recipe, candidate_raw, candidate_binding = _training_record(candidate_summary, candidate_run_config, child)
    recipes_available = base_recipe is not None and candidate_recipe is not None
    if recipes_available and _differences(base_recipe, candidate_recipe) != record["changes"]:
        raise ValueError("Actual run recipes change more than the declared single research factor")
    differences = ([key for key in sorted(base_binding.keys() | candidate_binding.keys())
                    if base_binding.get(key) != candidate_binding.get(key)]
                   if base_binding is not None and candidate_binding is not None else None)
    if not recipes_available:
        reason = "Actual run_config recipes are missing"
    elif differences is None:
        reason = "Existing code/environment/model/public-task source binding is missing"
    elif differences:
        reason = "Existing source bindings differ: " + ", ".join(differences)
    elif base["training_population"] != candidate["training_population"]:
        reason = "Actual runs used different public task populations or task overrides"
    else:
        reason = None
    comparable = reason is None
    evaluations = {"baseline": _evaluation_record(baseline_evaluation, base, base_raw),
                   "candidate": _evaluation_record(candidate_evaluation, candidate, candidate_raw)}
    for name, training in (("baseline", base), ("candidate", candidate)):
        if evaluations[name]["status"] == "bound":
            training["post_update_evaluation"] = evaluations[name]["summary"]
    qualification = (_reuse_gain_qualification(base, candidate, base_raw, candidate_raw, evaluations)
                     if record["factor"] == "reuse" else None)
    deltas = {
        "training_scope": "Observed reported completed batches only; not complete-run training cost or gain",
        "complete_completed_batch_history": _all_zero(
            _nested(item, "history_coverage", "unreported_completed_batches") for item in (base, candidate)),
        "includes_all_checkpoint_optimizer_steps": _all_zero(
            _nested(item, "updates", "optimizer_steps_outside_reported_batches") for item in (base, candidate)),
    }
    for section, names in (("collection", ("effective_advantage_groups", "effective_advantage_group_fraction",
                                           "effective_advantage_yield", "elapsed_seconds", "effective_groups_per_collection_hour")),
                           ("updates", ("optimizer_steps", "nonzero_lr_optimizer_steps", "zero_lr_optimizer_steps",
                                        "skipped_optimizer_steps", "update_wall_seconds",
                                        "seconds_per_optimizer_step", "seconds_per_nonzero_lr_step"))):
        deltas[section] = {name: _delta(_nested(base, section, name), _nested(candidate, section, name))
                          if comparable else None for name in names}
    deltas["evaluation"] = _evaluation_delta(evaluations["baseline"], evaluations["candidate"], comparable, qualification)
    return {"schema_version": 1, "record_type": "single_factor_comparison",
            "factor": record["factor"], "value": record["value"], "changes": record["changes"],
            "comparability": {"status": "confirmed" if comparable else "unknown",
                              "recipe_status": "single_factor" if recipes_available else "unknown",
                              "source_binding_difference_fields": differences, "reason": reason},
            "baseline": base, "candidate": candidate, "paired_evaluations": evaluations,
            "training_gain_qualification": qualification,
            "candidate_minus_baseline": deltas, "recommendation": None,
            "interpretation": {
                "training": "Collection rewards precede updates; they are not post-update adapter gains.",
                "kl": "Preserve each metric's scope and fixed-call post-update probe; never subtract probes from different call populations.",
                "evaluation": "Official acceptance in the recorded backend, not necessarily official competition CT; retain population and claim.",
                "missing": "Null means unknown, not zero. No automatic winner is selected.",
                "comparison": "Single-factor recipe consistency does not by itself establish a causal or statistically reliable gain.",
                **({"reuse": "Both arms share the initial LR and applied-step LR sequence, without dummy updates or changing warmup. Reuse 1 and 2 permit different update counts and computation, so this is not an equal-update or equal-compute comparison. Trained-adapter outcome deltas require effective updates and independent weight-change evidence for each evaluated artifact; raw evaluations and descriptive costs remain available."}
                   if record["factor"] == "reuse" else {}),
                **({"scale_rewards": "Only grpo.scale_rewards changes. Acceptance and sampler feedback use unscaled raw signals with unchanged settings. Different updates can change later trajectories and adaptive task selection, so this between-run comparison is not a fixed-cohort causal comparison."}
                   if record["factor"] == "scale_rewards" else {}),
                **({"prefix_reuse": "Only maximum calls per training tree changes to one; native attention/checkpoint backend stays fixed. Token/attention position savings are not measured GPU speedups. Separate full-run training can change later rollouts; same frozen-call replay is needed for an exact numerical comparison."}
                   if record["factor"] == "prefix_reuse" else {}),
            }}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Save one candidate recipe and its semantic change")
    prepare.add_argument("--baseline", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--factor", required=True, help="rank, reward.<one weight>, sampling, reuse, scale_rewards, or prefix_reuse")
    prepare.add_argument("--value", required=True, help="Positive rank, nonnegative reward weight, uniform, 1, or group/none")
    compare = commands.add_parser("compare", help="Compare actual summaries without launching an experiment")
    compare.add_argument("--experiment", type=Path, required=True)
    for name in ("baseline-summary", "candidate-summary", "baseline-run-config", "candidate-run-config",
                 "baseline-evaluation", "candidate-evaluation"):
        compare.add_argument("--" + name, type=Path)
    compare.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_experiment(args.baseline, args.output_dir, args.factor, args.value)
        else:
            result = compare_experiment(args.experiment, **{name: getattr(args, name) for name in
                ("baseline_summary", "candidate_summary", "baseline_run_config", "candidate_run_config",
                 "baseline_evaluation", "candidate_evaluation")})
            if args.output is not None:
                _save_json(args.output, result)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
