"""Manifest-backed, officially verified SFT data and lazy masked CE.

The learner consumes aggregate judge evidence, never private node maps or model
narration.  Records that do not prove at least one supported SFT category are
rejected (and reported), rather than being assigned a synthetic RL reward.
Torch is imported only by :func:`masked_ce_loss`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..trajectory.schema import ActionCall, RewardRecord, Trajectory
from .curriculum import _digest, _public_path


_CATEGORY_ORDER = ("successful_repair", "correct_tool_use", "verified_recovery",
                   "multi_turn_improvement")
_REWARD_FIELDS = ("resolved", "P", "G", "progress", "regression", "validity",
                  "base_reward", "efficiency", "efficiency_weight", "total_reward",
                  "eligible", "exclusion_reason", "objective_version")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceError(ValueError):
    """The record is well-formed JSON but lacks admission evidence."""


@dataclass(frozen=True)
class LoadedRecord:
    payload: Mapping[str, Any]
    source: Mapping[str, Any] | None = None


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _check_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _manifest(path: str | Path, expected_dataset: str) -> tuple[list[LoadedRecord], dict[str, Any]]:
    manifest_path = _public_path(path)
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("dataset manifest is not valid JSON") from exc
    if not isinstance(document, Mapping) or document.get("schema_version") != 1:
        raise ValueError("dataset manifest schema_version must be 1")
    if document.get("dataset") != expected_dataset:
        raise ValueError(f"manifest dataset must be {expected_dataset!r}")
    sources = document.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)) or not sources:
        raise ValueError("manifest sources must be a non-empty array")
    records: list[LoadedRecord] = []
    normalized_sources: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(f"manifest source {index} is not an object")
        source_path = source.get("path")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError("every manifest source needs an explicit path")
        candidate = Path(source_path).expanduser()
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        data_path = _public_path(candidate)
        expected_hash = _check_sha(source.get("sha256"), f"sources[{index}].sha256")
        actual_hash = _digest(data_path)
        if actual_hash != expected_hash:
            raise ValueError(f"data source hash mismatch: {data_path}")
        role = source.get("source")
        if role not in {"student_rl", "verified_sft", "teacher_offline", "teacher_opd"}:
            raise ValueError(f"unsupported data source role: {role!r}")
        normalized = dict(source)
        normalized.update(path=str(data_path), sha256=actual_hash)
        if expected_dataset in {"teacher_offline", "teacher_opd"}:
            if not isinstance(source.get("origin"), str) or not source["origin"].strip():
                raise ValueError("teacher sources must record an explicit origin")
            if not isinstance(source.get("model"), str) or not source["model"].strip():
                raise ValueError("teacher sources must record the model or endpoint")
            _check_sha(source.get("config_sha256"), "teacher config_sha256")
        count = source.get("record_count")
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
            raise ValueError("record_count must be a non-negative integer")
        normalized_sources.append(normalized)
        source_records = 0
        with data_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed record {data_path}:{line_number}") from exc
                if not isinstance(row, Mapping):
                    raise ValueError(f"record {data_path}:{line_number} is not an object")
                if row.get("heldout") is True or _FORBIDDEN_SPLIT(row.get("split")):
                    raise ValueError("heldout records are forbidden in training data")
                records.append(LoadedRecord(row, normalized))
                source_records += 1
        if count is not None and count != source_records:
            raise ValueError(f"record_count mismatch for {data_path}")
    info = dict(document)
    info.update(path=str(manifest_path), manifest_sha256=_digest(manifest_path),
                sources=normalized_sources)
    return records, info


def _FORBIDDEN_SPLIT(value: Any) -> bool:
    return isinstance(value, str) and bool(re.search(
        r"(^|[/_.-])(heldout|holdout|reserved|secret|hidden)($|[/_.-])", value, re.I))


def _trajectory(value: Any) -> Trajectory:
    if isinstance(value, Trajectory):
        return value
    if isinstance(value, Mapping):
        return Trajectory.from_dict(value)
    raise ValueError("record trajectory must be a Trajectory or object")


def _verification(row: Mapping[str, Any]) -> Mapping[str, Any]:
    for name in ("verification", "verification_bundle", "official_verification"):
        value = row.get(name)
        if isinstance(value, Mapping):
            return value
    raise EvidenceError("official verification bundle is missing")


def _reward_record(verification: Mapping[str, Any]) -> tuple[RewardRecord, Mapping[str, Any]]:
    value: Any = verification
    for name in ("reward_record", "reward", "official_reward"):
        candidate = verification.get(name)
        if isinstance(candidate, Mapping):
            value = candidate
            break
    if not isinstance(value, Mapping) or not isinstance(value.get("audit"), Mapping):
        raise EvidenceError("aggregate RewardEngine audit evidence is required")
    try:
        record = RewardRecord.from_dict({name: value.get(name) for name in _REWARD_FIELDS})
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"invalid official reward evidence: {exc}") from exc
    return record, value


def _check_common(trajectory: Trajectory, row: Mapping[str, Any],
                  verification: Mapping[str, Any]) -> tuple[RewardRecord, Mapping[str, Any], Mapping[str, Any]]:
    # Private judge maps must remain on the judge side.  Only the aggregate
    # RewardEngine result is an admissible learner-side VerificationBundle.
    for name in ("private_targets", "judge_private_targets", "node_statuses", "nodes"):
        if name in verification:
            raise EvidenceError("private judge node/target content cannot enter learner data")
    record, reward = _reward_record(verification)
    if not record.eligible:
        raise EvidenceError(record.exclusion_reason or "official verification is ineligible")
    if record.resolved is None:
        raise EvidenceError("official resolved verdict is missing")
    identity = verification.get("trajectory_id", row.get("trajectory_id"))
    if identity != trajectory.trajectory_id:
        raise EvidenceError("verification trajectory identity does not match")
    if "task_key" in verification and verification["task_key"] != trajectory.task_key.to_dict():
        raise EvidenceError("verification task identity does not match")
    if trajectory.verification_ref is None:
        raise EvidenceError("trajectory has no official verification reference")
    if verification.get("verification_ref", trajectory.verification_ref) != trajectory.verification_ref:
        raise EvidenceError("verification reference does not match the trajectory")
    differential = reward["audit"].get("differential")
    if not isinstance(differential, Mapping) or differential.get("eligible") is not True:
        raise EvidenceError("official differential evidence is incomplete")
    coverage = differential.get("coverage")
    if not isinstance(coverage, Mapping) or any(
            not isinstance(coverage.get(name), Mapping) or coverage[name].get("complete") is not True
            for name in ("repair", "regression")):
        raise EvidenceError("official repair/regression coverage is incomplete")
    process = reward["audit"].get("process_evidence")
    if not isinstance(process, Mapping) or process.get("model_self_report_used") is not False:
        raise EvidenceError("structured official process evidence is required")
    return record, reward, process


def _verified_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and (
        value.get("verified") is True or value.get("status") in {"verified", "observed", "passed"}
    ) and value.get("model_self_report_used") is not True


def _positive_count(value: Mapping[str, Any], names: Sequence[str]) -> bool:
    for name in names:
        count = value.get(name)
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            return True
    return False


def official_categories(trajectory: Trajectory, verification: Mapping[str, Any],
                        row: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return only categories backed by structured official evidence.

    The returned metadata contains references and aggregate labels, not rewards.
    """
    row = row or {}
    record, reward, process = _check_common(trajectory, row, verification)
    ref = trajectory.verification_ref
    result: dict[str, dict[str, Any]] = {}
    differential = reward["audit"]["differential"]
    improvement = process.get("post_change_improvement", {})
    if record.resolved is True and isinstance(record.P, (int, float)) and record.P > 0 \
            and isinstance(improvement, Mapping) and improvement.get("status") == "observed":
        result["successful_repair"] = {"evidence_ref": ref, "P": record.P,
                                        "official_resolved": True}

    tool = next((value for value in (
        verification.get("tool_evidence"), verification.get("tool_use"),
        process.get("tool_use"), process.get("tool_call_validity"),
    ) if isinstance(value, Mapping)), None)
    if _verified_mapping(tool) and _positive_count(tool, (
            "correct_calls", "correct_tool_calls", "correct_call_count", "verified_calls")):
        result["correct_tool_use"] = {"evidence_ref": tool.get("evidence_ref", ref),
                                       "correct_calls": next(tool[name] for name in (
                                           "correct_calls", "correct_tool_calls", "correct_call_count",
                                           "verified_calls") if isinstance(tool.get(name), int) and tool[name] > 0)}

    recovery = next((value for value in (
        verification.get("recovery_evidence"), verification.get("recovery"),
        process.get("error_recovery"), process.get("recovery"),
    ) if isinstance(value, Mapping)), None)
    if _verified_mapping(recovery) and recovery.get("error_observed") is True \
            and recovery.get("recovery_observed") is True and _positive_count(recovery, (
                "recovered_calls", "recovered_call_count", "successful_recovery_count")):
        result["verified_recovery"] = {"evidence_ref": recovery.get("evidence_ref", ref),
                                        "error_observed": True, "recovery_observed": True}

    tests = process.get("failing_test_identification", {})
    executed = tests.get("executed_test_command_count") if isinstance(tests, Mapping) else None
    if len(trajectory.calls) >= 2 and isinstance(executed, int) and executed > 0 \
            and isinstance(record.P, (int, float)) and record.P > 0 \
            and isinstance(improvement, Mapping) and improvement.get("status") == "observed":
        result["multi_turn_improvement"] = {"evidence_ref": ref,
                                             "call_count": len(trajectory.calls),
                                             "executed_test_command_count": executed,
                                             "P": record.P}
    return {name: result[name] for name in _CATEGORY_ORDER if name in result}


@dataclass(frozen=True)
class SFTSample:
    trajectory: Trajectory
    categories: tuple[str, ...]
    evidence: Mapping[str, Mapping[str, Any]]
    manifest_sha256: str | None = None

    @property
    def source(self) -> str:
        return str(self.trajectory.provenance.get("source"))

    def to_dict(self) -> dict[str, Any]:
        # Deliberately no reward, advantage, V or A fields are emitted.
        return {"trajectory": self.trajectory.to_dict(), "source": self.source,
                "categories": list(self.categories),
                "evidence": {key: dict(value) for key, value in self.evidence.items()},
                "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True)
class SFTCall:
    sample: SFTSample
    call: ActionCall

    def __post_init__(self) -> None:
        if not isinstance(self.sample, SFTSample) or not isinstance(self.call, ActionCall):
            raise TypeError("SFTCall requires an SFTSample and ActionCall")
        if self.call.trajectory_id != self.sample.trajectory.trajectory_id:
            raise ValueError("SFT call identity differs from its sample")
        if not self.input_ids or len(self.input_ids) < 2:
            raise ValueError("SFT calls need a causal prefix and at least two tokens")
        if self.supervised_token_count <= 0:
            raise ValueError("SFT call has no supervised action token after causal shifting")

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.call.prompt_ids + self.call.completion_ids

    @property
    def target_mask(self) -> tuple[int, ...]:
        return (0,) * len(self.call.prompt_ids) + self.call.action_mask

    @property
    def supervised_token_count(self) -> int:
        return sum(self.target_mask[1:])


def collate_sft_calls(calls: Sequence[SFTCall], *, pad_id: int = 0,
                      max_length: int | None = None) -> dict[str, Any]:
    """Pad independent TITO calls without joining separate environment prefixes."""
    if isinstance(pad_id, bool) or not isinstance(pad_id, int) or pad_id < 0:
        raise ValueError("pad_id must be a non-negative integer")
    if not calls or any(not isinstance(call, SFTCall) for call in calls):
        raise ValueError("collate_sft_calls needs at least one SFTCall")
    sequences = [call.input_ids for call in calls]
    masks = [call.target_mask for call in calls]
    length = max(map(len, sequences))
    if max_length is not None:
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2:
            raise ValueError("max_length must be an integer >= 2")
        if length > max_length:
            raise ValueError("SFT sequence exceeds max_length; truncation is not automatic")
        length = max_length
    input_ids, attention, labels, action_mask = [], [], [], []
    for sequence, mask in zip(sequences, masks):
        padding = length - len(sequence)
        input_ids.append(list(sequence) + [pad_id] * padding)
        attention.append([1] * len(sequence) + [0] * padding)
        labels.append([token if selected else -100 for token, selected in zip(sequence, mask)]
                      + [-100] * padding)
        action_mask.append(list(mask) + [0] * padding)
    return {
        "input_ids": input_ids, "attention_mask": attention, "labels": labels,
        "action_mask": action_mask, "loss_mask": [list(row) for row in action_mask],
        "pad_id": pad_id, "action_tokens": sum(sum(row) for row in action_mask),
        "supervised_action_tokens": sum(sum(row[1:]) for row in action_mask),
        "trajectory_ids": [call.sample.trajectory.trajectory_id for call in calls],
        "call_indices": [call.call.call_index for call in calls],
        "request_ids": [call.call.request_id for call in calls],
        "sources": [call.sample.source for call in calls],
        "categories": [list(call.sample.categories) for call in calls],
        "objective": "masked_causal_cross_entropy",
        "reward_record": None, "raw_advantage": None,
    }


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError("masked CE requires torch only when the loss is evaluated") from exc


def masked_ce_loss(logits: Any, batch: Mapping[str, Any], *,
                   global_action_tokens: int | None = None) -> Any:
    """Causal masked CE; prompt/tool/environment positions are excluded.

    ``logits`` are unshifted model outputs for ``batch['input_ids']``.  The
    function shifts logits and labels internally, so an action at sequence index
    ``i`` is predicted by logits at ``i-1``.  No RL reward is constructed here.
    """
    torch = _torch()
    if not isinstance(batch, Mapping):
        raise TypeError("SFT batch must be a mapping")
    values = torch.as_tensor(logits)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.ndim != 3 or not values.is_floating_point():
        raise ValueError("logits must have shape [batch, sequence, vocabulary] and be floating point")
    labels = torch.as_tensor(batch["labels"], device=values.device, dtype=torch.long)
    mask = torch.as_tensor(batch.get("loss_mask", batch["action_mask"]),
                           device=values.device, dtype=values.dtype)
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if labels.shape != mask.shape or labels.shape != values.shape[:2]:
        raise ValueError("SFT labels, mask and logits sequence shapes must match")
    if bool(((mask != 0) & (mask != 1)).any().item()):
        raise ValueError("SFT action mask must be binary")
    target = labels[:, 1:]
    selected = mask[:, 1:]
    prediction = values[:, :-1, :]
    local = int(selected.sum().item())
    if global_action_tokens is None:
        global_action_tokens = int(batch.get("global_action_tokens", local))
    if isinstance(global_action_tokens, bool) or not isinstance(global_action_tokens, int) \
            or global_action_tokens <= 0 or local > global_action_tokens:
        raise ValueError("global_action_tokens must be positive and cover the local mask")
    if local == 0:
        return values.sum() * 0.0
    chosen_targets = target[selected != 0]
    if bool((chosen_targets < 0).any().item()) or bool((chosen_targets >= values.shape[-1]).any().item()):
        raise ValueError("action labels must be valid vocabulary IDs, not -100")
    if not bool(torch.isfinite(prediction[selected != 0]).all().item()):
        raise ValueError("non-finite logits at supervised action positions")
    per_token = torch.nn.functional.cross_entropy(
        prediction.reshape(-1, values.shape[-1]), target.reshape(-1),
        reduction="none", ignore_index=-100,
    ).reshape(target.shape)
    return (per_token * selected).sum() / float(global_action_tokens)


class SFTDataset:
    """Iterable verified trajectories with session-friendly batch/loss methods."""

    dataset_name = "sft"
    input_sources = frozenset({"student_rl", "verified_sft"})
    output_source = "verified_sft"
    require_resolved = False

    def __init__(self, records: Iterable[Any] | str | Path, *, manifest: Mapping[str, Any] | None = None,
                 strict: bool = False):
        self.manifest = dict(manifest or {})
        loaded: Iterable[Any]
        if isinstance(records, (str, Path)):
            loaded, self.manifest = _manifest(records, self.dataset_name)
        else:
            loaded = records
        self._samples: list[SFTSample] = []
        self.rejected: list[dict[str, Any]] = []
        for item in loaded:
            source = item.source if isinstance(item, LoadedRecord) else None
            row = item.payload if isinstance(item, LoadedRecord) else item
            if not isinstance(row, Mapping):
                raise ValueError("every SFT record must be an object")
            try:
                sample = self._admit(row, source)
            except EvidenceError as exc:
                identifier = row.get("trajectory_id")
                if isinstance(row.get("trajectory"), Mapping):
                    identifier = row["trajectory"].get("trajectory_id", identifier)
                rejection = {"trajectory_id": identifier, "reason": str(exc)}
                if strict:
                    raise
                self.rejected.append(rejection)
                continue
            self._samples.append(sample)
        self._calls = tuple(call for sample in self._samples for call in self._sample_calls(sample))

    def _admit(self, row: Mapping[str, Any], source: Mapping[str, Any] | None) -> SFTSample:
        trajectory = _trajectory(row.get("trajectory"))
        provenance = dict(trajectory.provenance)
        input_source = provenance.get("source")
        if input_source not in self.input_sources:
            raise EvidenceError(f"trajectory source must be one of {sorted(self.input_sources)}")
        if source is not None and source.get("source") != input_source:
            raise EvidenceError("manifest source and trajectory provenance disagree")
        verification = _verification(row)
        record, _, _ = _check_common(trajectory, row, verification)
        if self.require_resolved and record.resolved is not True:
            raise EvidenceError("teacher/offline admission requires official resolved=true")
        categories = official_categories(trajectory, verification, row)
        requested = row.get("categories")
        if requested is not None:
            if isinstance(requested, (str, bytes)) or not isinstance(requested, Sequence):
                raise EvidenceError("categories must be an array")
            unknown = set(requested) - set(_CATEGORY_ORDER)
            if unknown:
                raise EvidenceError(f"unknown SFT categories: {sorted(unknown)}")
            categories = {name: value for name, value in categories.items() if name in requested}
        if not categories:
            raise EvidenceError("no supported category has complete official evidence")
        if not any(call.action_token_count and len(call.prompt_ids) + len(call.completion_ids) >= 2
                   and sum(call.action_mask[1:] if not call.prompt_ids else call.action_mask) > 0
                   for call in trajectory.calls):
            raise EvidenceError("verified trajectory has no causally supervised action token")
        provenance.update({
            "source": self.output_source,
            "input_source": input_source,
            "official_verification_ref": trajectory.verification_ref,
            "sft_categories": list(categories),
        })
        if self.manifest.get("manifest_sha256"):
            provenance["manifest_sha256"] = self.manifest["manifest_sha256"]
        admitted = replace(trajectory, provenance=provenance)
        return SFTSample(admitted, tuple(categories), categories,
                         self.manifest.get("manifest_sha256"))

    @staticmethod
    def _sample_calls(sample: SFTSample) -> Iterator[SFTCall]:
        for call in sample.trajectory.calls:
            try:
                yield SFTCall(sample, call)
            except ValueError:
                # Calls with no action targets (for example tool-only messages)
                # remain in the trajectory but do not form an CE microbatch.
                continue

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[SFTSample]:
        return iter(self._samples)

    def __getitem__(self, index: int) -> SFTSample:
        return self._samples[index]

    @property
    def samples(self) -> tuple[SFTSample, ...]:
        return tuple(self._samples)

    def iter_calls(self) -> Iterator[SFTCall]:
        return iter(self._calls)

    def iter_batches(self, batch_size: int = 1, *, shuffle: bool = False,
                     seed: int = 3407, pad_id: int = 0,
                     max_length: int | None = None) -> Iterator[dict[str, Any]]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        calls = list(self._calls)
        if shuffle:
            random.Random(seed).shuffle(calls)
        for offset in range(0, len(calls), batch_size):
            yield collate_sft_calls(calls[offset:offset + batch_size], pad_id=pad_id,
                                    max_length=max_length)

    def loss(self, logits: Any, batch: Mapping[str, Any], *,
             global_action_tokens: int | None = None) -> Any:
        return masked_ce_loss(logits, batch, global_action_tokens=global_action_tokens)

    batch_loss = loss


masked_cross_entropy = masked_ce_loss
make_sft_batch = collate_sft_calls
