#!/usr/bin/env python3
"""Validate/copy an independent PEFT LoRA package; never merge a base model.
Requires safetensors, but no torch/model download. Invalid or missing weights fail.
https://huggingface.co/docs/peft/developer_guides/checkpoint
"""
import argparse
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
try:
    from .common import TrainingError, atomic_json, integer, load_config, sha256_file
except ImportError:
    from common import TrainingError, atomic_json, integer, load_config, sha256_file

TEXT_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
FORBIDDEN = ("vision", "visual", "audio", "projector", "embed", "lm_head", "norm")
TRAINING_RELATIONS = {
    "bf16": "train W_QAT + dW; serve dequant(Q_CT(W_QAT)) + dW",
    "nf4": "train Q_NF4(W_QAT) + dW; serve dequant(Q_CT(W_QAT)) + dW",
}


def inspect_adapter(adapter_dir, expected_rank=None, expected_targets=None, max_rank=128):
    root = Path(adapter_dir)
    paths = {n: root / n for n in ("adapter_config.json", "adapter_model.safetensors")}
    for path in paths.values():
        if not path.is_file() or not path.stat().st_size:
            raise TrainingError("required adapter file missing/empty: " + str(path))
    cfg = json.loads(paths["adapter_config.json"].read_text(encoding="utf-8"))
    if cfg.get("peft_type") != "LORA" or cfg.get("task_type") != "CAUSAL_LM":
        raise TrainingError("only PEFT LORA / CAUSAL_LM packages are accepted")
    rank = integer(cfg.get("r"), "adapter r")
    if rank > max_rank or (expected_rank is not None and rank != expected_rank):
        raise TrainingError("adapter rank exceeds limit or differs from training config")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise TrainingError("per-layer rank/alpha patterns are outside this exporter contract")
    if cfg.get("modules_to_save") or cfg.get("use_dora") or cfg.get("bias", "none") != "none":
        raise TrainingError("full modules, DoRA and trainable bias are outside this exporter contract")
    if cfg.get("fan_in_fan_out", False):
        raise TrainingError("fan_in_fan_out adapters are unsupported")
    alpha = cfg.get("lora_alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise TrainingError("adapter lora_alpha must be positive")
    if not isinstance(cfg.get("base_model_name_or_path"), str) or not cfg["base_model_name_or_path"]:
        raise TrainingError("adapter must preserve its actual base_model_name_or_path")
    targets = cfg.get("target_modules")
    if not isinstance(targets, list) or not targets or any(not isinstance(t, str) for t in targets):
        raise TrainingError("target_modules must be an explicit nonempty list")
    for target in targets:
        if target.rsplit(".", 1)[-1] not in TEXT_TARGETS or any(p in target.lower() for p in FORBIDDEN):
            raise TrainingError("non-text/unsupported target: " + target)
    try:
        safe_open = importlib.import_module("safetensors").safe_open
    except ImportError as exc:
        raise TrainingError("safetensors is required; no weights were exported") from exc
    tensors, pairs = {}, {}
    try:
        with safe_open(str(paths["adapter_model.safetensors"]), framework="numpy") as handle:
            for key in handle.keys():
                match = re.fullmatch(r"(base_model\.model\..+)\.lora_([AB])\.weight", key)
                if not match:
                    raise TrainingError("unsupported PEFT key; use save_pretrained format: " + key)
                module, side = match.groups()
                if module.rsplit(".", 1)[-1] not in TEXT_TARGETS or any(p in module.lower() for p in FORBIDDEN):
                    raise TrainingError("forbidden/non-text adapter tensor: " + key)
                if not any(module.endswith("." + t) or module == t for t in targets):
                    raise TrainingError("tensor absent from target_modules: " + key)
                view = handle.get_slice(key)
                shape, dtype = list(view.get_shape()), view.get_dtype()
                if len(shape) != 2 or any(type(s) is not int or s <= 0 for s in shape):
                    raise TrainingError("LoRA tensor must be a nonempty matrix: " + key)
                if dtype not in ("F16", "BF16", "F32") or shape[0 if side == "A" else 1] != rank:
                    raise TrainingError("tensor dtype/rank mismatch: " + key)
                tensors[key] = {"shape": shape, "dtype": dtype}
                pairs.setdefault(module, {})[side] = shape
    except TrainingError:
        raise
    except Exception as exc:
        raise TrainingError("invalid safetensors adapter: " + str(exc)) from exc
    if not pairs or any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise TrainingError("every LoRA module needs both A and B tensors")
    if expected_targets and not set(expected_targets).issubset({m.rsplit(".", 1)[-1] for m in pairs}):
        raise TrainingError("adapter is missing requested target types")
    for target in targets:
        if not any(m.endswith("." + target) or m == target for m in pairs):
            raise TrainingError("declared target has no A/B tensors: " + target)
    files = {n: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for n, p in paths.items()}
    return {"adapter_config": cfg, "rank": rank, "modules": sorted(pairs), "tensors": tensors,
            "tensor_parameters": sum(math.prod(t["shape"]) for t in tensors.values()), "files": files,
            "runtime_compatibility_verified": False}


def export_adapter(adapter_dir, destination=None, config=None, training_manifest=None):
    """Validate first, then atomically copy only independent PEFT files."""
    config, training_manifest = config or {}, training_manifest or {}
    report = inspect_adapter(adapter_dir, config.get("rank"), config.get("target_text_modules"))
    recorded = training_manifest.get("adapter", {}).get("files")
    if recorded and recorded != report["files"]:
        raise TrainingError("adapter bytes differ from supplied training manifest")
    source, dest = Path(adapter_dir).resolve(), Path(adapter_dir).resolve()
    if destination is not None:
        dest = Path(destination).resolve()
        if dest == source or source in dest.parents or dest.exists():
            raise TrainingError("destination must be a new directory separate from source")
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="." + dest.name + "-", dir=str(dest.parent)))
        try:
            for name in report["files"]:
                shutil.copyfile(source / name, staging / name)
            if inspect_adapter(staging)["files"] != report["files"]:
                raise TrainingError("adapter changed during copy")
            os.replace(staging, dest)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    quant = config.get("quant_backend", training_manifest.get("quant_backend", "unknown"))
    return {"manifest_version": 2, "source_adapter": str(source), "adapter_dir": str(dest),
            "adapter": report, "training": training_manifest or None, "quant_backend": quant,
            "training_relation": TRAINING_RELATIONS.get(quant, "unknown"),
            "base_model_name_or_path": report["adapter_config"]["base_model_name_or_path"],
            "kept_independent": True, "runtime_compatibility_verified": False}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--destination", help="new independent adapter directory")
    p.add_argument("--config", help="optional training config for rank/target comparison")
    p.add_argument("--training-manifest", help="actual training manifest with provenance")
    p.add_argument("--data-fingerprint", help="optional declared data SHA256")
    p.add_argument("--out", required=True, help="output manifest JSON")
    p.add_argument("--manifest-only", action="store_true", help="validate real weights without copying")
    a = p.parse_args(argv)
    try:
        if a.manifest_only == bool(a.destination):
            raise TrainingError("choose exactly one: --destination or --manifest-only")
        if a.data_fingerprint and not re.fullmatch(r"[0-9a-f]{64}", a.data_fingerprint):
            raise TrainingError("--data-fingerprint must be SHA256")
        cfg = load_config(a.config) if a.config else {}
        run = json.loads(Path(a.training_manifest).read_text(encoding="utf-8")) if a.training_manifest else None
        result = export_adapter(a.adapter_dir, a.destination, cfg, run)
        if a.data_fingerprint:
            result["declared_data_sha256"] = a.data_fingerprint
        atomic_json(a.out, result)
        print(json.dumps({"ok": True, "manifest": str(Path(a.out).resolve()),
                          "adapter_dir": result["adapter_dir"], "rank": result["adapter"]["rank"]}))
        return 0
    except (TrainingError, OSError, ValueError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
