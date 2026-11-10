"""Disk-backed immutable adapter registry with durable reference counts."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any, Iterator
import uuid

from training.rollout.policy_version import AdapterManifest, PolicyVersion, sha256_file


MANIFEST_NAME = "manifest.json"
REGISTRY_NAME = "registry.json"


def _sync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".registry-", delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class AdapterManager:
    """Manage immutable PEFT directories and an atomically replaced registry.

    Registration never overwrites a version. Publishing and activation are
    separate so a DoubleBuffer can defer serving N+1 while old trajectories
    still require N. Unregistration never deletes weights unless explicitly
    requested by the caller.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.versions_dir = self.root / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / REGISTRY_NAME
        self._lock = threading.RLock()
        self._registry = self._read_registry()

    def _empty_registry(self) -> dict[str, Any]:
        return {"schema_version": 1, "active_version": None, "versions": {}}

    def _read_registry(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return self._empty_registry()
        value = json.loads(self.registry_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {"schema_version", "active_version", "versions"} \
                or value.get("schema_version") != 1 or not isinstance(value.get("versions"), dict):
            raise ValueError("invalid adapter registry schema")
        normalized = self._empty_registry()
        for key, row in value["versions"].items():
            if not isinstance(key, str) or not key.isdigit() or not isinstance(row, dict) \
                    or set(row) != {"manifest", "path", "references"}:
                raise ValueError("invalid adapter registry entry")
            manifest = AdapterManifest.from_dict(row["manifest"])
            if int(key) != manifest.version or type(row["references"]) is not int or row["references"] < 0:
                raise ValueError("invalid adapter version or reference count")
            relative = Path(row["path"])
            if not isinstance(row["path"], str) or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("adapter registry paths must remain inside the manager root")
            expected = self._relative_path(manifest)
            if relative != expected:
                raise ValueError("adapter registry path does not match its immutable identity")
            directory = self.root / relative
            if not directory.is_dir():
                raise ValueError(f"registered adapter directory is missing: {directory}")
            normalized["versions"][key] = {
                "manifest": manifest.to_dict(), "path": row["path"], "references": row["references"],
            }
        active = value["active_version"]
        if active is not None:
            if type(active) is not int or str(active) not in normalized["versions"]:
                raise ValueError("active adapter version is not registered")
            normalized["active_version"] = active
        return normalized

    def _write_registry(self) -> None:
        _atomic_json(self.registry_path, self._registry)

    @staticmethod
    def _directory_name(manifest: AdapterManifest) -> str:
        return f"{manifest.version:08d}-{manifest.adapter_sha256[:16]}"

    def _relative_path(self, manifest: AdapterManifest) -> Path:
        return Path("versions") / self._directory_name(manifest)

    @staticmethod
    def _verify_directory(manifest: AdapterManifest, directory: Path) -> None:
        if not directory.is_dir():
            raise ValueError("adapter source must be a directory")
        weights = directory / "adapter_model.safetensors"
        config_path = directory / "adapter_config.json"
        if not weights.is_file() or not config_path.is_file():
            raise ValueError("PEFT adapter weights and adapter_config.json are required")
        if sha256_file(weights) != manifest.adapter_sha256:
            raise ValueError("adapter SHA256 does not match adapter_model.safetensors")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("r") != 128 or config.get("lora_alpha") != 256 \
                or isinstance(config.get("r"), bool) or isinstance(config.get("lora_alpha"), bool) \
                or isinstance(config.get("lora_dropout"), bool) \
                or float(config.get("lora_dropout", -1)) != 0.0:
            raise ValueError("PEFT config must contain rank 128, alpha 256 and dropout 0")
        disk_manifest_path = directory / MANIFEST_NAME
        if disk_manifest_path.exists() and AdapterManifest.read(disk_manifest_path) != manifest:
            raise ValueError("on-disk manifest differs from the candidate manifest")

    def _resolve_version(self, adapter: int | str | AdapterManifest | PolicyVersion) -> int:
        if isinstance(adapter, (AdapterManifest, PolicyVersion)):
            row = self._registry["versions"].get(str(adapter.version))
            if row is None or row["manifest"]["adapter_sha256"] != adapter.adapter_sha256:
                raise KeyError("adapter identity is not registered")
            return adapter.version
        if type(adapter) is int:
            if str(adapter) not in self._registry["versions"]:
                raise KeyError(f"adapter version {adapter} is not registered")
            return adapter
        if isinstance(adapter, str):
            matches = [int(key) for key, row in self._registry["versions"].items()
                       if row["manifest"]["adapter_sha256"] == adapter]
            if len(matches) != 1:
                raise KeyError("adapter SHA256 is not uniquely registered")
            return matches[0]
        raise TypeError("adapter must be a version, SHA256, PolicyVersion, or AdapterManifest")

    def publish(self, candidate: AdapterManifest | str | Path,
                source: str | Path | None = None, *, activate: bool = False) -> AdapterManifest:
        """Install a candidate immutably; optionally move the active pointer.

        Both publish(path) and publish(manifest, path) are supported. The
        source manifest, when present, must exactly match the returned schema.
        """
        with self._lock:
            if source is None:
                if isinstance(candidate, AdapterManifest):
                    raise TypeError("publish(manifest) also requires a source directory")
                directory = Path(candidate).expanduser().resolve(strict=True)
                manifest = AdapterManifest.read(directory / MANIFEST_NAME)
            else:
                if not isinstance(candidate, AdapterManifest):
                    raise TypeError("publish(manifest, source) requires an AdapterManifest")
                manifest = candidate
                directory = Path(source).expanduser().resolve(strict=True)
            if str(manifest.version) in self._registry["versions"]:
                raise FileExistsError(f"adapter version {manifest.version} is immutable and already registered")
            if manifest.parent_version is not None \
                    and str(manifest.parent_version) not in self._registry["versions"]:
                raise ValueError("adapter parent must be registered before its child")
            self._verify_directory(manifest, directory)
            relative = self._relative_path(manifest)
            final = self.root / relative
            if final.exists():
                raise FileExistsError(f"immutable adapter destination already exists: {final}")
            staging = self.versions_dir / f".publish-{uuid.uuid4().hex}"
            try:
                shutil.copytree(directory, staging)
                _atomic_json(staging / MANIFEST_NAME, manifest.to_dict())
                _sync_directory(staging)
                staging.rename(final)
                _sync_directory(self.versions_dir)
                self._registry["versions"][str(manifest.version)] = {
                    "manifest": manifest.to_dict(),
                    "path": relative.as_posix(),
                    "references": 0,
                }
                if activate:
                    self._registry["active_version"] = manifest.version
                self._write_registry()
            except BaseException:
                # Only temporary publish artifacts may be removed here. An
                # already committed immutable version is never auto-deleted.
                if staging.exists():
                    shutil.rmtree(staging)
                raise
            return manifest

    register = publish

    def get(self, adapter: int | str | AdapterManifest | PolicyVersion) -> AdapterManifest:
        with self._lock:
            version = self._resolve_version(adapter)
            return AdapterManifest.from_dict(self._registry["versions"][str(version)]["manifest"])

    def path(self, adapter: int | str | AdapterManifest | PolicyVersion) -> Path:
        with self._lock:
            version = self._resolve_version(adapter)
            return self.root / self._registry["versions"][str(version)]["path"]

    @property
    def active(self) -> AdapterManifest | None:
        with self._lock:
            version = self._registry["active_version"]
            return None if version is None else self.get(version)

    def activate(self, adapter: int | str | AdapterManifest | PolicyVersion) -> AdapterManifest:
        with self._lock:
            version = self._resolve_version(adapter)
            self._registry["active_version"] = version
            self._write_registry()
            return self.get(version)

    def rollback(self, adapter: int | str | AdapterManifest | PolicyVersion | None = None) -> AdapterManifest:
        """Move only the serving pointer; weights and references are retained."""
        with self._lock:
            if adapter is None:
                active = self.active
                if active is None or active.parent_version is None:
                    raise ValueError("rollback requires a target or an active adapter with a parent")
                adapter = active.parent_version
            return self.activate(adapter)

    def acquire(self, adapter: int | str | AdapterManifest | PolicyVersion) -> AdapterManifest:
        with self._lock:
            version = self._resolve_version(adapter)
            row = self._registry["versions"][str(version)]
            row["references"] += 1
            self._write_registry()
            return AdapterManifest.from_dict(row["manifest"])

    def release(self, adapter: int | str | AdapterManifest | PolicyVersion) -> None:
        with self._lock:
            version = self._resolve_version(adapter)
            row = self._registry["versions"][str(version)]
            if row["references"] < 1:
                raise ValueError("adapter reference count cannot become negative")
            row["references"] -= 1
            self._write_registry()

    @contextmanager
    def reference(self, adapter: int | str | AdapterManifest | PolicyVersion) -> Iterator[AdapterManifest]:
        manifest = self.acquire(adapter)
        try:
            yield manifest
        finally:
            self.release(manifest)

    def refcount(self, adapter: int | str | AdapterManifest | PolicyVersion) -> int:
        with self._lock:
            version = self._resolve_version(adapter)
            return self._registry["versions"][str(version)]["references"]

    def unregister(self, adapter: int | str | AdapterManifest | PolicyVersion,
                   *, delete: bool = False) -> None:
        with self._lock:
            version = self._resolve_version(adapter)
            row = self._registry["versions"][str(version)]
            if self._registry["active_version"] == version:
                raise ValueError("the active adapter cannot be unregistered")
            if row["references"]:
                raise ValueError("an adapter with live references cannot be unregistered")
            children = [item for item in self._registry["versions"].values()
                        if item["manifest"]["parent_version"] == version]
            if children:
                raise ValueError("an adapter with registered children cannot be unregistered")
            path = self.root / row["path"]
            del self._registry["versions"][str(version)]
            self._write_registry()
            if delete:
                shutil.rmtree(path)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._registry))
