"""Disk-backed immutable adapter registry with durable reference counts."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Any, Iterator, Mapping
import uuid

from training.rollout.policy_version import AdapterManifest, PolicyVersion, sha256_file

from .retention import legacy_pruned_versions, read_pruned_versions


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
        # Deletion evidence is read before the registry so a missing version
        # directory can be told apart from a corrupted one. The marker lives
        # inside ``adapters/``; the legacy finished-run record lives one level
        # up in the run directory.
        self._pruned: dict[int, dict[str, Any]] = read_pruned_versions(self.root)
        self._legacy_pruned = legacy_pruned_versions(self.root.parent)
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
            if not directory.is_dir() and not self._accept_pruned(manifest, relative):
                raise ValueError(f"registered adapter directory is missing: {directory}")
            normalized["versions"][key] = {
                "manifest": manifest.to_dict(), "path": row["path"], "references": row["references"],
            }
        active = value["active_version"]
        if active is not None:
            if type(active) is not int or str(active) not in normalized["versions"]:
                raise ValueError("active adapter version is not registered")
            active_path = self.root / normalized["versions"][str(active)]["path"]
            if not active_path.is_dir():
                # Deletion evidence never makes the serving adapter optional.
                raise ValueError(f"active adapter directory is missing: {active_path}")
            normalized["active_version"] = active
        return normalized

    def _accept_pruned(self, manifest: AdapterManifest, relative: Path) -> bool:
        """Tolerate a missing directory only with exact deletion evidence.

        The authoritative marker must name this version and SHA256.  Runs
        pruned by an older launcher have only ``pruned_artifacts.json``; that
        record is accepted when its ``adapter vN`` label, immutable directory
        name and optional SHA256 all match.  Any other missing directory still
        raises in ``_read_registry``.
        """
        entry = self._pruned.get(manifest.version)
        if entry is not None and entry["adapter_sha256"] == manifest.adapter_sha256:
            return True
        legacy = self._legacy_pruned.get(manifest.version)
        if legacy is None or legacy.get("directory_name") != relative.name:
            return False
        recorded_sha = legacy.get("adapter_sha256")
        if recorded_sha is not None and recorded_sha != manifest.adapter_sha256:
            return False
        self._pruned[manifest.version] = {
            "version": manifest.version,
            "adapter_sha256": manifest.adapter_sha256,
            "bytes": int(legacy.get("bytes") or 0),
            "pruned_unix": legacy.get("pruned_unix"),
            "source": "legacy_pruned_artifacts",
        }
        return True

    def is_pruned(self, version: int) -> bool:
        """True when this version's weights were removed under the marker."""
        if type(version) is not int or version < 0:
            raise ValueError("adapter version must be a non-negative integer")
        with self._lock:
            return version in self._pruned

    def pruned_versions(self) -> dict[int, dict[str, Any]]:
        """Deletion evidence for versions whose directories are gone."""
        with self._lock:
            copied = json.loads(json.dumps(self._pruned))
            return {int(version): value for version, value in copied.items()}

    def _write_registry(self) -> None:
        _atomic_json(self.registry_path, self._registry)

    @staticmethod
    def _directory_name(manifest: AdapterManifest) -> str:
        return f"{manifest.version:08d}-{manifest.adapter_sha256[:16]}"

    def _relative_path(self, manifest: AdapterManifest) -> Path:
        return Path("versions") / self._directory_name(manifest)

    @staticmethod
    def _copy_tree(source: Path, destination: Path) -> None:
        """Install an immutable tree, preferring hardlinks on one filesystem.

        The exporter writes a full staging copy and the registry then needs the
        same bytes under ``versions/``. A byte-for-byte second copy would put
        two full rank-128 adapters on disk during every publish. Hardlinks keep
        one physical copy when source and destination share a filesystem and
        fall back to a normal copy across filesystems.
        """
        try:
            shutil.copytree(source, destination, copy_function=os.link)
        except OSError:
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination)

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
        if not isinstance(config, dict) or config.get("r") != manifest.rank or config.get("lora_alpha") != manifest.alpha \
                or type(config.get("r")) is not int or type(config.get("lora_alpha")) is not int \
                or isinstance(config.get("lora_dropout"), bool) \
                or float(config.get("lora_dropout", -1)) != manifest.dropout:
            raise ValueError("PEFT rank/alpha/dropout must match the adapter manifest")
        disk_manifest_path = directory / MANIFEST_NAME
        if disk_manifest_path.exists() and AdapterManifest.read(disk_manifest_path) != manifest:
            raise ValueError("on-disk manifest differs from the candidate manifest")

    @staticmethod
    def lora_name(manifest: AdapterManifest, *, prefix: str = "spo") -> str:
        """Stable vLLM server-side LoRA name derived from immutable identity."""
        if not isinstance(manifest, AdapterManifest):
            raise TypeError("lora_name requires an AdapterManifest")
        if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
            raise ValueError("LoRA name prefix must contain only letters, digits, _ or -")
        return f"{prefix}-v{manifest.version:08d}-{manifest.adapter_sha256[:16]}"

    def serving_binding(self, adapter: int | str | AdapterManifest | PolicyVersion,
                        *, prefix: str = "spo") -> dict[str, Any]:
        """Return the exact name/path/manifest tuple registered on vLLM."""
        manifest = self.get(adapter)
        return {
            "manifest": manifest,
            "lora_name": self.lora_name(manifest, prefix=prefix),
            "path": str(self.path(manifest)),
            "adapter_sha256": manifest.adapter_sha256,
            "version": manifest.version,
        }

    @staticmethod
    def _text_config(model_config: str | Path | Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if model_config is None:
            return None
        if isinstance(model_config, (str, Path)):
            raw = json.loads(Path(model_config).read_text(encoding="utf-8"))
        elif isinstance(model_config, Mapping):
            raw = model_config
        else:
            raise TypeError("model_config must be a path, mapping, or None")
        if not isinstance(raw, Mapping):
            raise ValueError("model config must be a JSON object")
        text = raw.get("text_config")
        return text if isinstance(text, Mapping) else raw

    @staticmethod
    def _target_matcher(target_modules: Any):
        if isinstance(target_modules, str) and target_modules.strip():
            try:
                pattern = re.compile(target_modules)
            except re.error as exc:
                raise ValueError(f"target_modules regex is invalid: {exc}") from exc
            return lambda name: pattern.search(name) is not None
        if isinstance(target_modules, (list, tuple)) and target_modules \
                and all(isinstance(item, str) and item for item in target_modules):
            names = tuple(target_modules)
            return lambda name: any(
                name == item or name.endswith("." + item) for item in names)
        raise ValueError("target_modules must be a nonempty regex string or name sequence")

    def audit_lora_compatibility(
        self,
        adapter: int | str | AdapterManifest | PolicyVersion,
        *,
        model_config: str | Path | Mapping[str, Any] | None = None,
        source: str | Path | None = None,
        expected_target_count: int | None = 400,
        allow_k_eq_v_lora: bool = False,
    ) -> dict[str, Any]:
        """Audit a PEFT adapter against the Gemma 4 k_eq_v target layout.

        The vLLM 0.32.0 stable wheel predates the upstream k_eq_v LoRA packing
        fix.  The production adapter deliberately omits ``k_proj`` on the ten
        ``full_attention`` layers, so it is loadable without that fix.  If a
        future adapter adds those targets, this audit fails closed instead of
        letting vLLM silently serve V without the adapter delta.
        """
        directory = Path(source).expanduser().resolve(strict=True) if source is not None \
            else self.path(adapter)
        config_path = directory / "adapter_config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"adapter_config.json is unreadable: {exc}") from exc
        if not isinstance(config, dict) or config.get("peft_type") != "LORA":
            raise ValueError("adapter_config.json is not a PEFT LORA adapter")
        manifest = adapter if isinstance(adapter, AdapterManifest) else self.get(adapter)
        reasons: list[str] = []
        if type(config.get("r")) is not int or config.get("r") != manifest.rank \
                or type(config.get("lora_alpha")) is not int or config.get("lora_alpha") != manifest.alpha:
            reasons.append("rank_or_alpha_mismatch")
        try:
            dropout = float(config.get("lora_dropout", -1))
        except (TypeError, ValueError):
            dropout = -1.0
        if dropout != 0.0:
            reasons.append("dropout_not_zero")
        if config.get("use_dora") or config.get("use_rslora"):
            reasons.append("unsupported_peft_variant")
        if config.get("modules_to_save"):
            reasons.append("modules_to_save_present")
        if config.get("bias") not in (None, "none"):
            reasons.append("bias_training_present")

        text = self._text_config(model_config)
        audit: dict[str, Any] = {
            "adapter_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "reasons": reasons,
            "target_count": None,
            "expected_target_count": expected_target_count,
            "k_eq_v_layers": [],
            "k_eq_v_targets": [],
            "kv_shared_targets": [],
            "target_audit_complete": text is not None,
        }
        if text is not None:
            layers = text.get("layer_types")
            num_layers = text.get("num_hidden_layers")
            if not isinstance(layers, list) or type(num_layers) is not int \
                    or len(layers) != num_layers:
                reasons.append("model_layer_types_unavailable")
            else:
                matcher = self._target_matcher(config.get("target_modules"))
                leaves = (
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                )
                matched = []
                num_kv_shared = text.get("num_kv_shared_layers", 0)
                kv_shared = set(range(num_layers - int(num_kv_shared), num_layers)) \
                    if type(num_kv_shared) is int and num_kv_shared > 0 else set()
                for index in range(num_layers):
                    for leaf in leaves:
                        name = f"model.language_model.layers.{index}.{leaf}"
                        if matcher(name):
                            matched.append(name)
                            if index in kv_shared and leaf in {
                                    "self_attn.k_proj", "self_attn.v_proj"}:
                                audit["kv_shared_targets"].append(name)
                            if leaf.endswith("k_proj") \
                                    and text.get("attention_k_eq_v") \
                                    and layers[index] == "full_attention" \
                                    and not allow_k_eq_v_lora:
                                audit["k_eq_v_targets"].append(name)
                            if leaf.endswith("v_proj") \
                                    and text.get("attention_k_eq_v") \
                                    and layers[index] == "full_attention":
                                reasons.append("v_proj_on_k_eq_v_layer")
                audit["target_count"] = len(matched)
                audit["k_eq_v_layers"] = [
                    index for index, kind in enumerate(layers) if kind == "full_attention"
                ]
                if audit["kv_shared_targets"]:
                    reasons.append("kv_shared_layer_k_or_v_target")
                if expected_target_count is not None and len(matched) != expected_target_count:
                    reasons.append("target_count_mismatch")
                if audit["k_eq_v_targets"]:
                    reasons.append("k_eq_v_k_proj_requires_vllm_pr_61267")
                if not matched:
                    reasons.append("no_lora_targets_matched")
        audit["reasons"] = list(dict.fromkeys(reasons))
        audit["compatible"] = not audit["reasons"]
        return audit

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

    def _install(self, manifest: AdapterManifest, directory: Path, *,
                 activate: bool, require_parent: bool) -> AdapterManifest:
        with self._lock:
            if str(manifest.version) in self._registry["versions"]:
                raise FileExistsError(f"adapter version {manifest.version} is immutable and already registered")
            if require_parent and manifest.parent_version is not None \
                    and str(manifest.parent_version) not in self._registry["versions"]:
                raise ValueError("adapter parent must be registered before its child")
            self._verify_directory(manifest, directory)
            relative = self._relative_path(manifest)
            final = self.root / relative
            if final.exists():
                raise FileExistsError(f"immutable adapter destination already exists: {final}")
            staging = self.versions_dir / f".publish-{uuid.uuid4().hex}"
            try:
                self._copy_tree(directory, staging)
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

    def import_adapter(self, manifest: AdapterManifest, source: str | Path, *,
                       activate: bool = False) -> AdapterManifest:
        """Register an adapter produced by an earlier run as this run's base.

        The manifest keeps its published version and sha; only the parent row
        is allowed to be absent because a new continuation run intentionally
        starts from one version without importing its whole history. The caller
        records the source path/version in the run metadata.
        """
        if not isinstance(manifest, AdapterManifest):
            raise TypeError("import_adapter requires an AdapterManifest")
        directory = Path(source).expanduser().resolve(strict=True)
        if not directory.is_dir():
            raise ValueError(f"adapter source is not a directory: {directory}")
        return self._install(manifest, directory, activate=activate,
                             require_parent=False)

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
            return self._install(manifest, directory, activate=activate,
                                 require_parent=True)

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
