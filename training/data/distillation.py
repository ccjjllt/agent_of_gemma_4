"""Offline teacher trajectories and strictly aligned OPD top-k distillation.

No teacher service, downloader, endpoint call or GPU is started here.  Teacher
records must carry official aggregate acceptance and explicit origin/model/config
provenance.  Same-tokenizer OPD compares original token IDs only.  A different
tokenizer is accepted solely with a self-contained raw-byte span proof mapping
teacher top-k IDs into student IDs; ordinary text re-tokenization is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..trajectory.schema import ActionCall, Trajectory
from .sft import (
    EvidenceError, LoadedRecord, SFTCall, SFTDataset, SFTSample, _check_common,
    _check_sha, _hash_value, _manifest, _trajectory, _verification,
)


class TokenizerAlignmentError(EvidenceError):
    """Raised when OPD would otherwise invent a token correspondence."""


def _teacher_info(row: Mapping[str, Any], trajectory: Trajectory,
                  source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    supplied = row.get("teacher")
    if supplied is not None and not isinstance(supplied, Mapping):
        raise EvidenceError("teacher provenance must be an object")
    supplied = supplied or {}
    provenance = trajectory.provenance
    values: dict[str, Any] = {}
    for name in ("origin", "model", "endpoint", "config_sha256", "config_hash",
                 "tokenizer_sha256", "tokenizer_hash", "template_sha256", "template_hash"):
        value = supplied.get(name, provenance.get(name))
        if value is None and source is not None:
            value = source.get(name)
        if value is not None:
            values[name] = value
    for name in ("origin", "model"):
        if not isinstance(values.get(name), str) or not values[name].strip():
            raise EvidenceError(f"teacher {name} must be explicitly recorded")
    config = values.get("config_sha256", values.get("config_hash"))
    try:
        values["config_sha256"] = _check_sha(config, "teacher config_sha256")
    except ValueError as exc:
        raise EvidenceError(str(exc)) from exc
    values.pop("config_hash", None)
    for name in ("tokenizer_sha256", "template_sha256"):
        alias = name.replace("_sha256", "_hash")
        value = values.get(name, values.get(alias))
        if value is not None:
            try:
                values[name] = _check_sha(value, f"teacher {name}")
            except ValueError as exc:
                raise EvidenceError(str(exc)) from exc
        values.pop(alias, None)
    if source is not None:
        for name in ("origin", "model", "config_sha256"):
            if name in source and source[name] != values.get(name):
                raise EvidenceError(f"teacher {name} differs from its manifest")
    return values


def _tokenizer_info(row: Mapping[str, Any], side: str,
                    trajectory: Trajectory) -> dict[str, str]:
    containers: list[Mapping[str, Any]] = []
    for container in (row.get("tokenizers"), row.get("tokenizer_templates")):
        if isinstance(container, Mapping) and isinstance(container.get(side), Mapping):
            containers.append(container[side])
    direct = row.get(f"{side}_tokenizer")
    if isinstance(direct, Mapping):
        containers.append(direct)
    containers.append(trajectory.provenance)
    values: dict[str, Any] = {}
    for name in ("tokenizer_sha256", "tokenizer_hash", "template_sha256", "template_hash"):
        for container in containers:
            if name in container:
                values[name] = container[name]
                break
        prefixed = f"{side}_{name}"
        if prefixed in row:
            values[name] = row[prefixed]
    result: dict[str, str] = {}
    for name in ("tokenizer_sha256", "template_sha256"):
        value = values.get(name, values.get(name.replace("_sha256", "_hash")))
        try:
            result[name] = _check_sha(value, f"{side} {name}")
        except ValueError as exc:
            raise TokenizerAlignmentError(str(exc)) from exc
    return result


def _sequence(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvidenceError(f"{name} must be an array")
    return tuple(value)


def _topk(ids: Any, logprobs: Any, student_ids: Any = None) -> tuple[
        tuple[tuple[int, ...], ...], tuple[tuple[float, ...], ...], tuple[tuple[int, ...], ...] | None]:
    raw_ids = _sequence(ids, "teacher_topk_ids")
    raw_logprobs = _sequence(logprobs, "teacher_topk_logprobs")
    raw_student = None if student_ids is None else _sequence(student_ids, "student_topk_ids")
    if len(raw_ids) != len(raw_logprobs) or (raw_student is not None and len(raw_student) != len(raw_ids)):
        raise EvidenceError("OPD top-k positions and logprobs must have equal lengths")
    normalized_ids: list[tuple[int, ...]] = []
    normalized_logprobs: list[tuple[float, ...]] = []
    normalized_student: list[tuple[int, ...]] = []
    width: int | None = None
    for position, (position_ids, position_logprobs) in enumerate(zip(raw_ids, raw_logprobs)):
        ids_row = _sequence(position_ids, f"teacher_topk_ids[{position}]")
        log_row = _sequence(position_logprobs, f"teacher_topk_logprobs[{position}]")
        if not ids_row or len(ids_row) != len(log_row):
            raise EvidenceError("every OPD position needs non-empty matching top-k IDs/logprobs")
        if width is None:
            width = len(ids_row)
        if len(ids_row) != width:
            raise EvidenceError("OPD top-k width must be constant within a record")
        converted_ids: list[int] = []
        converted_logs: list[float] = []
        for token_id, logprob in zip(ids_row, log_row):
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise EvidenceError("OPD token IDs must be non-negative integers")
            if isinstance(logprob, bool) or not isinstance(logprob, (int, float)) \
                    or not math.isfinite(float(logprob)) or float(logprob) > 0:
                raise EvidenceError("OPD logprobs must be finite and <= 0")
            converted_ids.append(token_id)
            converted_logs.append(float(logprob))
        if len(set(converted_ids)) != len(converted_ids):
            raise EvidenceError("teacher top-k IDs must be unique at each position")
        if any(converted_logs[index] < converted_logs[index + 1]
               for index in range(len(converted_logs) - 1)):
            raise EvidenceError("teacher top-k logprobs must be sorted descending")
        probability_mass = math.fsum(math.exp(value) for value in converted_logs)
        if probability_mass <= 0 or probability_mass > 1.0 + 1e-5:
            raise EvidenceError("teacher top-k logprobs have invalid probability mass")
        normalized_ids.append(tuple(converted_ids))
        normalized_logprobs.append(tuple(converted_logs))
        if raw_student is not None:
            student_row = _sequence(raw_student[position], f"student_topk_ids[{position}]")
            if len(student_row) != len(ids_row):
                raise EvidenceError("mapped student top-k IDs must match teacher top-k width")
            converted_student: list[int] = []
            for token_id in student_row:
                if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                    raise EvidenceError("student top-k IDs must be non-negative integers")
                converted_student.append(token_id)
            if len(set(converted_student)) != len(converted_student):
                raise EvidenceError("student top-k mapping must be one-to-one at each position")
            normalized_student.append(tuple(converted_student))
    return (tuple(normalized_ids), tuple(normalized_logprobs),
            None if raw_student is None else tuple(normalized_student))


def _hex_bytes(value: Any, name: str, *, nonempty: bool = False) -> bytes:
    if not isinstance(value, str) or len(value) % 2 or re.fullmatch(r"[0-9a-fA-F]*", value) is None:
        raise TokenizerAlignmentError(f"{name} must be hexadecimal raw bytes")
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise TokenizerAlignmentError(f"{name} is not valid raw bytes") from exc
    if nonempty and not result:
        raise TokenizerAlignmentError(f"{name} must not be empty")
    return result


def _verify_span_proof(proof: Any, alignment: "OPDAlignment",
                       student_call: ActionCall, teacher_call: ActionCall) -> None:
    """Verify byte spans and every top-k ID mapping, without re-tokenizing text."""
    if not isinstance(proof, Mapping) or proof.get("schema_version") != "raw_span_opd_v1":
        raise TokenizerAlignmentError("cross-tokenizer OPD requires raw_span_opd_v1 evidence")
    student_context = _hex_bytes(proof.get("student_raw_context_hex"), "student raw context", nonempty=True)
    teacher_context = _hex_bytes(proof.get("teacher_raw_context_hex"), "teacher raw context", nonempty=True)
    if student_context != teacher_context:
        raise TokenizerAlignmentError("cross-tokenizer raw contexts differ")
    context_hash = hashlib.sha256(student_context).hexdigest()
    if proof.get("raw_context_sha256") != context_hash:
        raise TokenizerAlignmentError("raw context hash does not match its bytes")
    entries = _sequence(proof.get("entries"), "raw span entries")
    if len(entries) != len(alignment.student_completion_indices):
        raise TokenizerAlignmentError("raw span proof does not cover every student action")
    by_position = {entry.get("student_completion_index"): entry for entry in entries
                   if isinstance(entry, Mapping)}
    if len(by_position) != len(entries):
        raise TokenizerAlignmentError("raw span entries have duplicate/invalid positions")
    for position, (student_index, teacher_index, teacher_ids, student_ids) in enumerate(zip(
            alignment.student_completion_indices, alignment.teacher_completion_indices,
            alignment.teacher_topk_ids, alignment.student_topk_ids or ())):
        entry = by_position.get(student_index)
        if not isinstance(entry, Mapping) or entry.get("teacher_completion_index") != teacher_index:
            raise TokenizerAlignmentError("raw span entry does not match call positions")
        if student_index >= len(student_call.completion_ids) or teacher_index >= len(teacher_call.completion_ids):
            raise TokenizerAlignmentError("raw span completion index is out of range")
        if entry.get("student_token_id") != student_call.completion_ids[student_index] \
                or entry.get("teacher_token_id") != teacher_call.completion_ids[teacher_index]:
            raise TokenizerAlignmentError("raw span token identity differs from ActionCall")
        student_raw = _hex_bytes(entry.get("student_raw_hex"), "student action raw span", nonempty=True)
        teacher_raw = _hex_bytes(entry.get("teacher_raw_hex"), "teacher action raw span", nonempty=True)
        if student_raw != teacher_raw:
            raise TokenizerAlignmentError("student/teacher action raw spans differ")
        try:
            student_start, student_end = int(entry["student_start"]), int(entry["student_end"])
            teacher_start, teacher_end = int(entry["teacher_start"]), int(entry["teacher_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TokenizerAlignmentError("raw span offsets are missing/invalid") from exc
        if not (0 <= student_start < student_end <= len(student_context)
                and 0 <= teacher_start < teacher_end <= len(teacher_context)
                and student_context[student_start:student_end] == student_raw
                and teacher_context[teacher_start:teacher_end] == teacher_raw):
            raise TokenizerAlignmentError("raw span offsets do not prove the action bytes")
        if tuple(entry.get("teacher_topk_ids", ())) != teacher_ids \
                or tuple(entry.get("student_topk_ids", ())) != student_ids:
            raise TokenizerAlignmentError("raw span top-k IDs differ from the alignment")
        teacher_raw_topk = _sequence(entry.get("teacher_topk_raw_hex"), "teacher top-k raw spans")
        student_raw_topk = _sequence(entry.get("student_topk_raw_hex"), "student top-k raw spans")
        if len(teacher_raw_topk) != len(teacher_ids) or len(student_raw_topk) != len(student_ids):
            raise TokenizerAlignmentError("every cross-tokenizer top-k alternative needs a raw span")
        for teacher_raw_id, student_raw_id in zip(teacher_raw_topk, student_raw_topk):
            if _hex_bytes(teacher_raw_id, "teacher top-k raw span", nonempty=True) != _hex_bytes(
                    student_raw_id, "student top-k raw span", nonempty=True):
                raise TokenizerAlignmentError("teacher/student top-k raw alternatives differ")
    proof_body = {
        "schema_version": proof.get("schema_version"),
        "student_raw_context_hex": proof.get("student_raw_context_hex"),
        "teacher_raw_context_hex": proof.get("teacher_raw_context_hex"),
        "raw_context_sha256": proof.get("raw_context_sha256"),
        "entries": list(entries),
    }
    if proof.get("proof_sha256") != _hash_value(proof_body):
        raise TokenizerAlignmentError("raw span proof hash is invalid")


@dataclass(frozen=True)
class OPDAlignment:
    student_call_index: int
    teacher_call_index: int
    student_completion_indices: tuple[int, ...]
    teacher_completion_indices: tuple[int, ...]
    teacher_topk_ids: tuple[tuple[int, ...], ...]
    teacher_topk_logprobs: tuple[tuple[float, ...], ...]
    student_topk_ids: tuple[tuple[int, ...], ...] | None = None
    raw_span_alignment: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("student_call_index", "teacher_call_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        student_indices = _sequence(self.student_completion_indices, "student_completion_indices")
        teacher_indices = _sequence(self.teacher_completion_indices, "teacher_completion_indices")
        if len(student_indices) != len(teacher_indices) or not student_indices:
            raise ValueError("OPD alignment needs matching non-empty student/teacher positions")
        for values, name in ((student_indices, "student"), (teacher_indices, "teacher")):
            if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in values):
                raise ValueError(f"{name} completion indices must be non-negative integers")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} completion indices must be unique")
        ids, logs, student_ids = _topk(self.teacher_topk_ids, self.teacher_topk_logprobs,
                                       self.student_topk_ids)
        if len(ids) != len(student_indices):
            raise ValueError("OPD top-k positions do not cover the action indices")
        object.__setattr__(self, "student_completion_indices", tuple(student_indices))
        object.__setattr__(self, "teacher_completion_indices", tuple(teacher_indices))
        object.__setattr__(self, "teacher_topk_ids", ids)
        object.__setattr__(self, "teacher_topk_logprobs", logs)
        object.__setattr__(self, "student_topk_ids", student_ids)
        if self.raw_span_alignment is not None and not isinstance(self.raw_span_alignment, Mapping):
            raise ValueError("raw_span_alignment must be an object")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OPDAlignment":
        if not isinstance(value, Mapping):
            raise ValueError("OPD alignment must be an object")
        return cls(
            student_call_index=value.get("student_call_index"),
            teacher_call_index=value.get("teacher_call_index"),
            student_completion_indices=value.get("student_completion_indices",
                                                   value.get("student_action_indices")),
            teacher_completion_indices=value.get("teacher_completion_indices",
                                                   value.get("teacher_action_indices")),
            teacher_topk_ids=value.get("teacher_topk_ids", value.get("topk_ids")),
            teacher_topk_logprobs=value.get("teacher_topk_logprobs", value.get("topk_logprobs")),
            student_topk_ids=value.get("student_topk_ids"),
            raw_span_alignment=value.get("raw_span_alignment", value.get("span_alignment")),
        )

    @property
    def top_k(self) -> int:
        return len(self.teacher_topk_ids[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "student_call_index": self.student_call_index,
            "teacher_call_index": self.teacher_call_index,
            "student_completion_indices": list(self.student_completion_indices),
            "teacher_completion_indices": list(self.teacher_completion_indices),
            "teacher_topk_ids": [list(row) for row in self.teacher_topk_ids],
            "teacher_topk_logprobs": [list(row) for row in self.teacher_topk_logprobs],
            "student_topk_ids": None if self.student_topk_ids is None else [list(row) for row in self.student_topk_ids],
            "raw_span_alignment": self.raw_span_alignment,
        }


@dataclass(frozen=True)
class OPDSample:
    student: Trajectory
    teacher: Trajectory
    alignments: tuple[OPDAlignment, ...]
    teacher_provenance: Mapping[str, Any]
    student_tokenizer: Mapping[str, str]
    teacher_tokenizer: Mapping[str, str]
    manifest_sha256: str | None = None

    @property
    def source(self) -> str:
        return "teacher_opd"

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "student": self.student.to_dict(),
                "teacher": self.teacher.to_dict(),
                "alignments": [item.to_dict() for item in self.alignments],
                "teacher_provenance": dict(self.teacher_provenance),
                "student_tokenizer": dict(self.student_tokenizer),
                "teacher_tokenizer": dict(self.teacher_tokenizer),
                "manifest_sha256": self.manifest_sha256}


@dataclass(frozen=True)
class OPDCall:
    sample: OPDSample
    student_call: ActionCall
    teacher_call: ActionCall
    alignment: OPDAlignment

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.student_call.prompt_ids + self.student_call.completion_ids

    @property
    def target_mask(self) -> tuple[int, ...]:
        return (0,) * len(self.student_call.prompt_ids) + self.student_call.action_mask

    @property
    def supervised_token_count(self) -> int:
        return sum(self.target_mask[1:])

    @property
    def top_k(self) -> int:
        return self.alignment.top_k


class TeacherOfflineDataset(SFTDataset):
    """Officially accepted teacher trajectories with explicit provenance."""

    dataset_name = "teacher_offline"
    input_sources = frozenset({"teacher_offline"})
    output_source = "teacher_offline"
    require_resolved = True

    def _admit(self, row: Mapping[str, Any], source: Mapping[str, Any] | None) -> SFTSample:
        trajectory = _trajectory(row.get("trajectory"))
        if trajectory.provenance.get("source") != "teacher_offline":
            raise EvidenceError("teacher trajectory provenance.source must be teacher_offline")
        info = _teacher_info(row, trajectory, source)
        verification = _verification(row)
        record, _, _ = _check_common(trajectory, row, verification)
        if record.resolved is not True:
            raise EvidenceError("teacher trajectory did not pass official acceptance")
        # A teacher trajectory is admissible on official resolved verdict alone;
        # category labels are optional additional process evidence.
        from .sft import official_categories
        categories = official_categories(trajectory, verification, row)
        if not any(call.action_token_count and len(call.prompt_ids) + len(call.completion_ids) >= 2
                   for call in trajectory.calls):
            raise EvidenceError("teacher trajectory has no trainable action call")
        provenance = dict(trajectory.provenance)
        provenance.update(info)
        provenance.update({"source": "teacher_offline", "teacher": dict(info),
                           "official_verification_ref": trajectory.verification_ref})
        if self.manifest.get("manifest_sha256"):
            provenance["manifest_sha256"] = self.manifest["manifest_sha256"]
        admitted = replace(trajectory, provenance=provenance)
        return SFTSample(admitted, tuple(categories), categories,
                         self.manifest.get("manifest_sha256"))


class OPDDataset:
    """Student-action-aligned teacher top-k records and lazy forward-KL loss."""

    dataset_name = "teacher_opd"

    def __init__(self, records: Iterable[Any] | str | Path, *, strict: bool = False):
        self.manifest: dict[str, Any] = {}
        loaded: Iterable[Any]
        if isinstance(records, (str, Path)):
            loaded, self.manifest = _manifest(records, self.dataset_name)
        else:
            loaded = records
        self._samples: list[OPDSample] = []
        self.rejected: list[dict[str, Any]] = []
        for item in loaded:
            source = item.source if isinstance(item, LoadedRecord) else None
            row = item.payload if isinstance(item, LoadedRecord) else item
            if not isinstance(row, Mapping):
                raise ValueError("every OPD record must be an object")
            try:
                sample = self._admit(row, source)
            except EvidenceError as exc:
                if strict:
                    raise
                self.rejected.append({"trajectory_id": row.get("student_trajectory_id"),
                                      "reason": str(exc)})
                continue
            self._samples.append(sample)
        self._calls = tuple(call for sample in self._samples for call in self._sample_calls(sample))
        widths = {call.top_k for call in self._calls}
        if len(widths) > 1:
            raise ValueError("an OPD dataset must use one fixed top-k width")

    def _admit(self, row: Mapping[str, Any], source: Mapping[str, Any] | None) -> OPDSample:
        student = _trajectory(row.get("student_trajectory", row.get("student")))
        teacher = _trajectory(row.get("teacher_trajectory", row.get("teacher_trajectory_record")))
        if student.provenance.get("source") != "student_rl":
            raise EvidenceError("OPD student trajectory source must be student_rl")
        if teacher.provenance.get("source") not in {"teacher_offline", "teacher_opd"}:
            raise EvidenceError("OPD teacher trajectory source must be teacher_offline")
        if student.task_key != teacher.task_key:
            raise EvidenceError("OPD student and teacher task identities differ")
        verification = _verification(row)
        # Verification is for the teacher, not a fabricated student reward.
        teacher_row = dict(row)
        teacher_row["trajectory_id"] = teacher.trajectory_id
        teacher_row["task_key"] = teacher.task_key.to_dict()
        record, _, _ = _check_common(teacher, teacher_row, verification)
        if record.resolved is not True:
            raise EvidenceError("OPD teacher trajectory lacks official resolved=true")
        info = _teacher_info(row, teacher, source)
        student_tokenizer = _tokenizer_info(row, "student", student)
        teacher_tokenizer = _tokenizer_info(row, "teacher", teacher)
        same_tokenizer = student_tokenizer == teacher_tokenizer
        raw_alignments = row.get("alignments", row.get("opd_alignments"))
        if isinstance(raw_alignments, (str, bytes)) or not isinstance(raw_alignments, Sequence) \
                or not raw_alignments:
            raise EvidenceError("OPD record needs at least one alignment")
        alignments = tuple(OPDAlignment.from_dict(value) for value in raw_alignments)
        by_student_call = {item.student_call_index: item for item in alignments}
        if len(by_student_call) != len(alignments):
            raise EvidenceError("OPD has duplicate student call alignments")
        expected_calls = {call.call_index for call in student.calls if call.action_token_count}
        if set(by_student_call) != expected_calls:
            raise EvidenceError("OPD must cover every student action call exactly once")
        used_teacher_calls: set[int] = set()
        for alignment in alignments:
            if alignment.student_call_index >= len(student.calls) \
                    or alignment.teacher_call_index >= len(teacher.calls):
                raise TokenizerAlignmentError("OPD call index is out of range")
            student_call = student.calls[alignment.student_call_index]
            teacher_call = teacher.calls[alignment.teacher_call_index]
            if alignment.teacher_call_index in used_teacher_calls:
                raise TokenizerAlignmentError("OPD teacher call is reused for multiple student calls")
            used_teacher_calls.add(alignment.teacher_call_index)
            expected_indices = tuple(index for index, selected in enumerate(student_call.action_mask) if selected)
            if alignment.student_completion_indices != expected_indices:
                raise TokenizerAlignmentError("OPD positions are not exactly the student action_mask positions")
            if same_tokenizer:
                if student_call.prompt_ids != teacher_call.prompt_ids \
                        or student_call.completion_ids != teacher_call.completion_ids:
                    raise TokenizerAlignmentError(
                        "same-tokenizer OPD requires exact original prompt/completion token alignment")
                if alignment.teacher_completion_indices != alignment.student_completion_indices:
                    raise TokenizerAlignmentError("same-tokenizer OPD completion positions differ")
                if alignment.student_topk_ids is not None \
                        and alignment.student_topk_ids != alignment.teacher_topk_ids:
                    raise TokenizerAlignmentError("same-tokenizer teacher/student top-k IDs differ")
                if alignment.raw_span_alignment is not None:
                    raise TokenizerAlignmentError("same-tokenizer OPD must use token IDs, not span substitution")
                object.__setattr__(alignment, "student_topk_ids", alignment.teacher_topk_ids)
            else:
                if alignment.student_topk_ids is None or alignment.raw_span_alignment is None:
                    raise TokenizerAlignmentError(
                        "cross-tokenizer OPD without verified raw span alignment is explicitly rejected")
                _verify_span_proof(alignment.raw_span_alignment, alignment, student_call, teacher_call)
        provenance = dict(info)
        provenance.update({"source": "teacher_opd", "teacher": dict(info),
                           "student_tokenizer": dict(student_tokenizer),
                           "teacher_tokenizer": dict(teacher_tokenizer)})
        return OPDSample(student, teacher, alignments, provenance,
                         student_tokenizer, teacher_tokenizer,
                         self.manifest.get("manifest_sha256"))

    @staticmethod
    def _sample_calls(sample: OPDSample) -> Iterator[OPDCall]:
        for alignment in sample.alignments:
            student_call = sample.student.calls[alignment.student_call_index]
            teacher_call = sample.teacher.calls[alignment.teacher_call_index]
            call = OPDCall(sample, student_call, teacher_call, alignment)
            if call.supervised_token_count:
                yield call

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[OPDSample]:
        return iter(self._samples)

    def __getitem__(self, index: int) -> OPDSample:
        return self._samples[index]

    @property
    def samples(self) -> tuple[OPDSample, ...]:
        return tuple(self._samples)

    @property
    def top_k(self) -> int | None:
        return self._calls[0].top_k if self._calls else None

    def iter_calls(self) -> Iterator[OPDCall]:
        return iter(self._calls)

    def iter_batches(self, batch_size: int = 1, *, pad_id: int = 0,
                     max_length: int | None = None) -> Iterator[dict[str, Any]]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        for offset in range(0, len(self._calls), batch_size):
            yield collate_opd_calls(self._calls[offset:offset + batch_size], pad_id=pad_id,
                                    max_length=max_length)

    def loss(self, student_logits: Any, batch: Mapping[str, Any], *,
             global_action_tokens: int | None = None) -> Any:
        return opd_forward_kl(student_logits, batch, global_action_tokens=global_action_tokens)

    batch_loss = loss


def collate_opd_calls(calls: Sequence[OPDCall], *, pad_id: int = 0,
                      max_length: int | None = None) -> dict[str, Any]:
    if not calls or any(not isinstance(call, OPDCall) for call in calls):
        raise ValueError("collate_opd_calls needs at least one OPDCall")
    if isinstance(pad_id, bool) or not isinstance(pad_id, int) or pad_id < 0:
        raise ValueError("pad_id must be a non-negative integer")
    width = calls[0].top_k
    if any(call.top_k != width for call in calls):
        raise ValueError("OPD batches require a fixed top-k width")
    sequences = [call.input_ids for call in calls]
    masks = [call.target_mask for call in calls]
    length = max(map(len, sequences))
    if max_length is not None:
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2:
            raise ValueError("max_length must be an integer >= 2")
        if length > max_length:
            raise ValueError("OPD sequence exceeds max_length; truncation is not automatic")
        length = max_length
    input_ids, attention, masks_out = [], [], []
    student_topk, teacher_topk, teacher_logs = [], [], []
    for call, sequence, mask in zip(calls, sequences, masks):
        padding = length - len(sequence)
        input_ids.append(list(sequence) + [pad_id] * padding)
        attention.append([1] * len(sequence) + [0] * padding)
        masks_out.append(list(mask) + [0] * padding)
        student_rows = [[0] * width for _ in sequence]
        teacher_rows = [[0] * width for _ in sequence]
        log_rows = [[0.0] * width for _ in sequence]
        for student_index, teacher_ids, student_ids, logprobs in zip(
                call.alignment.student_completion_indices, call.alignment.teacher_topk_ids,
                call.alignment.student_topk_ids or (), call.alignment.teacher_topk_logprobs):
            sequence_index = len(call.student_call.prompt_ids) + student_index
            student_rows[sequence_index] = list(student_ids)
            teacher_rows[sequence_index] = list(teacher_ids)
            log_rows[sequence_index] = list(logprobs)
        student_topk.append(student_rows + [[0] * width for _ in range(padding)])
        teacher_topk.append(teacher_rows + [[0] * width for _ in range(padding)])
        teacher_logs.append(log_rows + [[0.0] * width for _ in range(padding)])
    return {
        "input_ids": input_ids, "attention_mask": attention,
        "action_mask": masks_out, "opd_mask": [list(row) for row in masks_out],
        "topk_ids": student_topk, "student_topk_ids": student_topk,
        "teacher_topk_ids": teacher_topk, "teacher_topk_logprobs": teacher_logs,
        "top_k": width, "pad_id": pad_id,
        "supervised_action_tokens": sum(sum(row[1:]) for row in masks_out),
        "trajectory_ids": [call.sample.student.trajectory_id for call in calls],
        "teacher_trajectory_ids": [call.sample.teacher.trajectory_id for call in calls],
        "call_indices": [call.student_call.call_index for call in calls],
        "student_tokenizer": [dict(call.sample.student_tokenizer) for call in calls],
        "teacher_tokenizer": [dict(call.sample.teacher_tokenizer) for call in calls],
        "objective": "masked_forward_kl_teacher_topk_renormalized",
        "teacher_distribution": "top-k teacher probabilities renormalized on their stored support",
        "spo_actor_loss_included": False,
    }


def opd_forward_kl(student_logits: Any, batch: Mapping[str, Any], *,
                   global_action_tokens: int | None = None) -> Any:
    """Masked teacher||student forward KL on stored top-k support.

    Teacher top-k logprobs are renormalized over their stored support.  This is
    explicitly a top-k OPD objective, not a claim about an unavailable full
    teacher vocabulary distribution.  IDs used for gather are student-space IDs;
    cross-tokenizer records can reach here only after raw-span verification.
    """
    from .sft import _torch
    torch = _torch()
    logits = torch.as_tensor(student_logits)
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    if logits.ndim != 3 or not logits.is_floating_point():
        raise ValueError("student_logits must be floating point [batch, sequence, vocabulary]")
    ids = torch.as_tensor(batch.get("student_topk_ids", batch["topk_ids"]),
                          dtype=torch.long, device=logits.device)
    teacher_logs = torch.as_tensor(batch["teacher_topk_logprobs"],
                                   dtype=logits.dtype, device=logits.device)
    mask = torch.as_tensor(batch.get("opd_mask", batch["action_mask"]),
                           dtype=logits.dtype, device=logits.device)
    if ids.ndim == 3:
        pass
    elif ids.ndim == 2:
        ids = ids.unsqueeze(0)
    else:
        raise ValueError("OPD top-k IDs must have shape [batch, sequence, k]")
    if teacher_logs.shape != ids.shape or mask.shape != ids.shape[:2] \
            or ids.shape[:2] != logits.shape[:2] or ids.shape[-1] <= 0:
        raise ValueError("OPD logits, top-k and mask shapes do not match")
    if bool(((mask != 0) & (mask != 1)).any().item()):
        raise ValueError("OPD mask must be binary")
    prediction = logits[:, :-1, :]
    ids = ids[:, 1:, :]
    teacher_logs = teacher_logs[:, 1:, :]
    selected = mask[:, 1:]
    local = int(selected.sum().item())
    if global_action_tokens is None:
        global_action_tokens = int(batch.get("global_action_tokens", local))
    if isinstance(global_action_tokens, bool) or not isinstance(global_action_tokens, int) \
            or global_action_tokens <= 0 or local > global_action_tokens:
        raise ValueError("global_action_tokens must be positive and cover the local mask")
    if local == 0:
        return logits.sum() * 0.0
    selected_ids = ids[selected != 0]
    if bool((selected_ids < 0).any().item()) or bool((selected_ids >= logits.shape[-1]).any().item()):
        raise ValueError("OPD student top-k IDs are outside the student vocabulary")
    if not bool(torch.isfinite(teacher_logs[selected != 0]).all().item()):
        raise ValueError("OPD teacher logprobs are non-finite at selected positions")
    if not bool(torch.isfinite(prediction[selected != 0]).all().item()):
        raise ValueError("OPD student logits are non-finite at selected positions")
    log_teacher = teacher_logs - torch.logsumexp(teacher_logs, dim=-1, keepdim=True)
    teacher_probability = torch.exp(log_teacher)
    student_log_probability = torch.log_softmax(prediction, dim=-1)
    selected_student = student_log_probability.gather(-1, ids)
    token_kl = (teacher_probability * (log_teacher - selected_student)).sum(dim=-1)
    return (token_kl * selected).sum() / float(global_action_tokens)


masked_forward_kl = opd_forward_kl
make_opd_batch = collate_opd_calls
