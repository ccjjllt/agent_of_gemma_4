"""Immutable policy identities and strict adapter metadata validation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
RANK = 128
ALPHA = 256
DROPOUT = 0.0


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
    return value


def _parent(version: int, parent: int | None) -> int | None:
    if version == 0:
        if parent is not None:
            raise ValueError("the genesis version must have parent=None")
        return None
    if type(parent) is not int or parent != version - 1:
        raise ValueError("parent must be exactly version - 1")
    return parent


def validate_lag(behavior_version: int, current_version: int, max_lag: int) -> int:
    """Return non-negative policy lag, rejecting future or stale behavior."""
    behavior_version = _integer(behavior_version, "behavior_version")
    current_version = _integer(current_version, "current_version")
    max_lag = _integer(max_lag, "max_lag")
    lag = current_version - behavior_version
    if lag < 0:
        raise ValueError("a trajectory cannot come from a future policy")
    if lag > max_lag:
        raise ValueError(f"policy lag {lag} exceeds max_lag {max_lag}")
    return lag


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PolicyVersion:
    version: int
    adapter_sha256: str
    parent: int | None
    optimizer_step: int
    config_hash: str

    def __post_init__(self) -> None:
        _integer(self.version, "version")
        _hash(self.adapter_sha256, "adapter_sha256")
        _parent(self.version, self.parent)
        _integer(self.optimizer_step, "optimizer_step")
        _hash(self.config_hash, "config_hash")

    @property
    def parent_version(self) -> int | None:
        return self.parent

    def lag(self, current_version: int) -> int:
        current_version = _integer(current_version, "current_version")
        lag = current_version - self.version
        if lag < 0:
            raise ValueError("a trajectory cannot come from a future policy")
        return lag

    def validate_lag(self, current_version: int, max_lag: int) -> int:
        return validate_lag(self.version, current_version, max_lag)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "adapter_sha256": self.adapter_sha256,
            "parent": self.parent,
            "optimizer_step": self.optimizer_step,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicyVersion":
        if not isinstance(value, Mapping):
            raise ValueError("PolicyVersion must be a JSON object")
        required = {"version", "adapter_sha256", "parent", "optimizer_step", "config_hash"}
        if set(value) != required:
            raise ValueError(f"PolicyVersion fields must be exactly {sorted(required)}")
        return cls(**dict(value))

    def to_manifest(self, *, base_revision: str, rank: int = RANK,
                    alpha: int = ALPHA, dropout: float = DROPOUT) -> "AdapterManifest":
        return AdapterManifest(
            version=self.version,
            parent_version=self.parent,
            adapter_sha256=self.adapter_sha256,
            base_revision=base_revision,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            optimizer_step=self.optimizer_step,
            config_hash=self.config_hash,
        )


@dataclass(frozen=True)
class AdapterManifest:
    version: int
    parent_version: int | None
    adapter_sha256: str
    base_revision: str
    rank: int
    alpha: int
    dropout: float
    optimizer_step: int
    config_hash: str

    def __post_init__(self) -> None:
        _integer(self.version, "version")
        _parent(self.version, self.parent_version)
        _hash(self.adapter_sha256, "adapter_sha256")
        if not isinstance(self.base_revision, str) or not self.base_revision.strip():
            raise ValueError("base_revision must be a non-empty string")
        if type(self.rank) is not int or self.rank != RANK:
            raise ValueError("adapter rank must be exactly 128")
        if type(self.alpha) is not int or self.alpha != ALPHA:
            raise ValueError("adapter alpha must be exactly 256")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) \
                or float(self.dropout) != DROPOUT:
            raise ValueError("adapter dropout must be exactly 0")
        _integer(self.optimizer_step, "optimizer_step")
        _hash(self.config_hash, "config_hash")

    @property
    def parent(self) -> int | None:
        return self.parent_version

    def lag(self, current_version: int) -> int:
        return self.to_policy_version().lag(current_version)

    def validate_lag(self, current_version: int, max_lag: int) -> int:
        return validate_lag(self.version, current_version, max_lag)

    def to_policy_version(self) -> PolicyVersion:
        return PolicyVersion(
            version=self.version,
            adapter_sha256=self.adapter_sha256,
            parent=self.parent_version,
            optimizer_step=self.optimizer_step,
            config_hash=self.config_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "adapter_sha256": self.adapter_sha256,
            "base_revision": self.base_revision,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "optimizer_step": self.optimizer_step,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdapterManifest":
        if not isinstance(value, Mapping):
            raise ValueError("AdapterManifest must be a JSON object")
        required = {
            "version", "parent_version", "adapter_sha256", "base_revision",
            "rank", "alpha", "dropout", "optimizer_step", "config_hash",
        }
        if set(value) != required:
            raise ValueError(f"AdapterManifest fields must be exactly {sorted(required)}")
        return cls(**dict(value))

    @classmethod
    def read(cls, path: str | Path) -> "AdapterManifest":
        source = Path(path)
        if source.is_dir():
            source = source / "manifest.json"
        value = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_dict(value)
