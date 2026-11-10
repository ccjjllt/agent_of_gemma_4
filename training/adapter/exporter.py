"""Atomic BF16 PEFT export built on an existing Policy instance.

This module never imports a model framework or loads a checkpoint. The caller
owns the already initialized Policy; save_adapter is its only model operation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any
import uuid

from training.rollout.policy_version import AdapterManifest, canonical_hash, sha256_file


MANIFEST_NAME = "manifest.json"


def _sync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_manifest(path: Path, manifest: AdapterManifest) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(manifest.to_dict(), output, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_bf16_safetensors(path: str | Path) -> dict[str, Any]:
    """Validate the safetensors header without importing torch or numpy."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > size - 8 or header_size > 100 * 1024**2:
            raise ValueError("invalid safetensors header size")
        try:
            header = json.loads(source.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    if not tensors:
        raise ValueError("PEFT export contains no tensors")
    for name, value in tensors.items():
        if not isinstance(value, dict) or value.get("dtype") != "BF16":
            raise ValueError(f"adapter tensor {name} was not serialized as BF16")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape) \
                or not isinstance(offsets, list) or len(offsets) != 2 \
                or any(type(offset) is not int or offset < 0 for offset in offsets) \
                or offsets[0] > offsets[1] or offsets[1] > size - 8 - header_size:
            raise ValueError(f"invalid safetensors tensor layout: {name}")
    return {"tensor_count": len(tensors), "dtypes": ["BF16"]}


class AdapterExporter:
    """Export one immutable candidate directory and its checksummed manifest."""

    def export(self, policy: Any, destination: str | Path, *, version: int,
               parent_version: int | None, optimizer_step: int,
               base_revision: str | None = None, config_hash: str | None = None) -> AdapterManifest:
        save_adapter = getattr(policy, "save_adapter", None)
        if not callable(save_adapter):
            raise TypeError("policy must provide save_adapter(path)")
        config = getattr(policy, "config", {})
        if not isinstance(config, dict):
            raise TypeError("policy.config must be a dictionary")
        if config.get("lora_rank", 128) != 128 or config.get("lora_alpha", 256) != 256 \
                or isinstance(config.get("lora_dropout"), bool) \
                or float(config.get("lora_dropout", 0.0)) != 0.0:
            raise ValueError("export requires the fixed rank 128, alpha 256, dropout 0 adapter")
        if config.get("adapter_export_dtype", "bfloat16") != "bfloat16":
            raise ValueError("adapter_export_dtype must be bfloat16 for candidate export")
        revision = base_revision if base_revision is not None else config.get("model_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("base_revision is required directly or as policy.config['model_revision']")
        destination = Path(destination).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"adapter export destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
        try:
            save_adapter(staging)
            weights = staging / "adapter_model.safetensors"
            adapter_config_path = staging / "adapter_config.json"
            if not weights.is_file() or not adapter_config_path.is_file():
                raise ValueError("save_adapter did not produce PEFT weights and adapter_config.json")
            validate_bf16_safetensors(weights)
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            if not isinstance(adapter_config, dict) or type(adapter_config.get("r")) is not int \
                    or adapter_config["r"] != 128 or type(adapter_config.get("lora_alpha")) is not int \
                    or adapter_config["lora_alpha"] != 256 \
                    or isinstance(adapter_config.get("lora_dropout"), bool) \
                    or float(adapter_config.get("lora_dropout", -1)) != 0.0:
                raise ValueError("serialized PEFT config is not rank 128, alpha 256, dropout 0")
            digest = sha256_file(weights)
            training_config_hash = canonical_hash(adapter_config) if config_hash is None else config_hash
            manifest = AdapterManifest(
                version=version,
                parent_version=parent_version,
                adapter_sha256=digest,
                base_revision=revision,
                rank=128,
                alpha=256,
                dropout=0.0,
                optimizer_step=optimizer_step,
                config_hash=training_config_hash,
            )
            _write_manifest(staging / MANIFEST_NAME, manifest)
            for item in staging.iterdir():
                if item.is_file():
                    with item.open("rb") as source:
                        os.fsync(source.fileno())
            _sync_directory(staging)
            staging.rename(destination)
            _sync_directory(destination.parent)
            return manifest
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    export_adapter = export


def export_adapter(policy: Any, destination: str | Path, **metadata: Any) -> AdapterManifest:
    return AdapterExporter().export(policy, destination, **metadata)
