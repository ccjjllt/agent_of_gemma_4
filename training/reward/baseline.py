"""Private, content-keyed empty-patch baselines for official differential rewards.

The runner is supplied by the judge/official runtime.  This module never runs a
repository command itself and never exposes the private node map to the learner.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
from typing import Callable, Mapping
import uuid


BASELINE_SCHEMA_VERSION = "empty_patch_baseline_v1"
_COMPONENTS = ("repair", "regression")
_ALLOWED_STATUSES = {"passed", "failure", "error", "skipped", "nonpassed", "missing"}


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TaskKey:
    """Complete identity of an official task and its judge environment."""

    task_id: str
    repo: str
    base_commit: str
    test_patch_sha256: str
    judge_environment_hash: str

    def __post_init__(self) -> None:
        for field in ("task_id", "repo", "base_commit", "test_patch_sha256", "judge_environment_hash"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"TaskKey.{field} must be a nonempty string")
        for field in ("test_patch_sha256", "judge_environment_hash"):
            value = getattr(self, field)
            if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
                raise ValueError(f"TaskKey.{field} must be a SHA256 hex digest")
            object.__setattr__(self, field, value.lower())

    @classmethod
    def from_dict(cls, value: Mapping[str, object] | "TaskKey") -> "TaskKey":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("task_key must be a TaskKey or mapping")
        return cls(**{field: value.get(field) for field in (
            "task_id", "repo", "base_commit", "test_patch_sha256", "judge_environment_hash",
        )})

    def as_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "test_patch_sha256": self.test_patch_sha256,
            "judge_environment_hash": self.judge_environment_hash,
        }

    @property
    def cache_key(self) -> str:
        # The environment hash is both part of the payload and the cache digest;
        # equal task text under a changed judge can never reuse an old baseline.
        return _sha256_json(self.as_dict())


def _universes(value: object) -> dict[str, list[str]]:
    if not isinstance(value, Mapping) or any(name not in value for name in _COMPONENTS):
        raise ValueError("expected_universes must contain repair and regression sequences")
    result: dict[str, list[str]] = {}
    for name in _COMPONENTS:
        nodes = value[name]
        if isinstance(nodes, (str, bytes)) or not isinstance(nodes, (list, tuple)):
            raise ValueError(f"expected_universes.{name} must be a sequence")
        unique: list[str] = []
        for node in nodes:
            if not isinstance(node, str) or not node:
                raise ValueError("Official test node identifiers must be nonempty strings")
            if node not in unique:
                unique.append(node)
        result[name] = unique
    return result


def _node_statuses(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or any(name not in value for name in _COMPONENTS):
        raise ValueError("node_statuses must contain repair and regression maps")
    result: dict[str, dict[str, str]] = {}
    for name in _COMPONENTS:
        rows = value[name]
        if not isinstance(rows, Mapping):
            raise ValueError(f"node_statuses.{name} must be a map")
        component: dict[str, str] = {}
        for node, row in rows.items():
            if not isinstance(node, str) or not node:
                raise ValueError("Observed test node identifiers must be nonempty strings")
            status = row.get("status") if isinstance(row, Mapping) else row
            if status == "failed":
                status = "failure"
            if status not in _ALLOWED_STATUSES:
                raise ValueError(f"Unsupported official node status: {status!r}")
            component[node] = str(status)
        result[name] = component
    return result


def _artifact_hashes(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("artifact_hashes must be a map of artifact names to SHA256 digests")
    result: dict[str, str] = {}
    for name, digest in value.items():
        if not isinstance(name, str) or not name or not isinstance(digest, str):
            raise ValueError("Invalid artifact hash entry")
        if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
            raise ValueError("Artifact hashes must be SHA256 hex digests")
        result[name] = digest.lower()
    return result


def normalize_baseline(task_key: TaskKey | Mapping[str, object], report: Mapping[str, object]) -> dict:
    """Validate and retain exactly the private evidence needed for pairing."""
    key = TaskKey.from_dict(task_key)
    if not isinstance(report, Mapping):
        raise TypeError("Empty-patch baseline runner must return a mapping")
    expected = _universes(report.get("expected_universes"))
    nodes = _node_statuses(report.get("node_statuses", report.get("nodes")))
    complete = report.get("verification_complete") is True
    valid = report.get("verdict_valid") is True
    infrastructure_error = report.get("infrastructure_error")
    if infrastructure_error is not None and not isinstance(infrastructure_error, str):
        infrastructure_error = str(infrastructure_error)
    junit_available = report.get("junit_available")
    if junit_available is not None and type(junit_available) is not bool:
        raise ValueError("junit_available must be bool when present")
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "task_key": key.as_dict(),
        "cache_key": key.cache_key,
        "expected_universes": expected,
        "node_statuses": nodes,
        "artifact_hashes": _artifact_hashes(report.get("artifact_hashes")),
        "verification_complete": complete,
        "verdict_valid": valid,
        "junit_available": junit_available,
        "infrastructure_error": infrastructure_error,
        "status": "complete" if complete and valid and not infrastructure_error else "incomplete",
    }


class EmptyPatchBaselineStore:
    """Judge-owned baseline cache, atomic JSON files with mandatory mode 0600."""

    def __init__(self, root: str | os.PathLike[str], runner: Callable[[TaskKey], Mapping[str, object]] | None = None):
        path = Path(root).expanduser()
        if path.is_symlink():
            raise PermissionError("Baseline store root must not be a symlink")
        created = not path.exists()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if created:
            path.chmod(0o700)
        info = path.stat()
        if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700:
            raise PermissionError("Baseline store root must be a private mode 0700 directory")
        if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
            raise PermissionError("Baseline store root must be owned by the judge user")
        self.root = path.resolve(strict=True)
        self.runner = runner
        self._lock = threading.Lock()

    def path_for(self, task_key: TaskKey | Mapping[str, object]) -> Path:
        return self.root / f"{TaskKey.from_dict(task_key).cache_key}.json"

    def _read(self, key: TaskKey) -> dict | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
                raise PermissionError("Cached baseline must be a regular mode 0600 file")
            if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
                raise PermissionError("Cached baseline must be owned by the judge user")
            with os.fdopen(descriptor, "r", encoding="utf-8") as source:
                descriptor = -1
                record = json.load(source)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        normalized = normalize_baseline(key, record)
        if record.get("schema_version") != BASELINE_SCHEMA_VERSION or record.get("cache_key") != key.cache_key:
            raise ValueError("Baseline cache identity or schema mismatch")
        if TaskKey.from_dict(record.get("task_key")) != key:
            raise ValueError("Baseline cache TaskKey mismatch")
        return normalized

    def _write(self, key: TaskKey, record: Mapping[str, object]) -> None:
        path = self.path_for(key)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                descriptor = -1
                os.fchmod(output.fileno(), 0o600)
                json.dump(record, output, ensure_ascii=False, indent=2, allow_nan=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def put(self, task_key: TaskKey | Mapping[str, object], report: Mapping[str, object]) -> dict:
        key = TaskKey.from_dict(task_key)
        record = normalize_baseline(key, report)
        with self._lock:
            self._write(key, record)
        return record

    def get(
        self,
        task_key: TaskKey | Mapping[str, object],
        runner: Callable[[TaskKey], Mapping[str, object]] | None = None,
        *,
        refresh: bool = False,
    ) -> dict:
        """Return cached evidence, or execute the judge-supplied empty-patch runner."""
        key = TaskKey.from_dict(task_key)
        with self._lock:
            if not refresh:
                cached = self._read(key)
                if cached is not None:
                    return cached
            compute = runner or self.runner
            if compute is None:
                raise FileNotFoundError(f"No empty-patch baseline for TaskKey {key.cache_key}")
            report = compute(key)
            record = normalize_baseline(key, report)
            self._write(key, record)
            return record
