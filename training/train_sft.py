#!/usr/bin/env python3
"""Single-device Gemma 4 text LoRA SFT over pre-encoded, selectively labelled JSONL.

Default is local_files_only=true. --dry-run imports no ML packages and never
downloads weights. Remote HF IDs require an explicit revision; --allow-download
is the only CLI switch that permits network loading. Kaggle version labels are
metadata, not HF revisions. CT checkpoints are rejected before weight loading.

Real dependencies: torch, transformers with Gemma4 support, peft, safetensors.
NF4 additionally requires bitsandbytes and accelerate and a CUDA device.
No dependency is installed by this program.

Example:
  python training/train_sft.py --config training/configs/sft_qat_r8.yaml \
    --model /models/gemma4-qat-unquantized --tokenizer /models/gemma4-qat-unquantized \
    --data /data/encoded.jsonl --out /runs/repair-r8
Sources: huggingface.co/docs/transformers/model_doc/gemma4
         huggingface.co/docs/peft/developer_guides/quantization
"""
import argparse
from contextlib import nullcontext
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import hashlib

try:
    from .common import (TrainingError, atomic_json, integer, load_config,
                         load_encoded_jsonl, pad_encoded_batch, require_bool, sha256_file)
    from .export_peft import inspect_adapter, TEXT_TARGETS, FORBIDDEN
except ImportError:
    from common import (TrainingError, atomic_json, integer, load_config,
                        load_encoded_jsonl, pad_encoded_batch, require_bool, sha256_file)
    from export_peft import inspect_adapter, TEXT_TARGETS, FORBIDDEN

TARGET_SUBSTRINGS = ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CT_MARKERS = ("w4a16-ct", "compressed-tensors", "compressed_tensors")
DEFAULTS = {
    "model_name_or_path": None, "model_revision": None,
    "tokenizer_name_or_path": None, "tokenizer_revision": None,
    "chat_template_path": None, "tool_schema_path": "training/configs/harness_tools.json",
    "base_family": "gemma-4-31b-it-qat-q4_0-unquantized",
    "source_version_note": None, "is_fallback_reference": False,
    "model_source_manifest": None,
    "local_files_only": True, "quant_backend": "bf16", "compute_dtype": "bfloat16",
    "target_text_modules": list(TARGET_SUBSTRINGS), "rank": 8, "alpha": 16,
    "dropout": 0.0, "bias": "none", "learning_rate": 5e-5,
    "max_sequence_length": 4096, "micro_batch_size": 1, "gradient_accumulation_steps": 8,
    "epochs": 1, "warmup_ratio": 0.03, "max_grad_norm": 1.0,
    "gradient_checkpointing": True, "weight_decay": 0.0,
    "loss": "selected_assistant_tokens", "protocol_id": "p005-harness-tools-v1",
    "checkpoint_every_n_steps": 100, "seed": 0, "max_steps": 0,
    "device": "cuda", "attn_implementation": "sdpa",
    "kbit_prepare": "norms_fp32", "loss_backend": "model", "loss_chunk_size": 32,
    "allow_cached_nf4": False,
}


def check_base_family(name):
    if any(marker in str(name).lower() for marker in CT_MARKERS):
        raise TrainingError("CT is an inference format: use QAT unquantized weights, not " + str(name))


def validate_config(raw):
    # Optional DDP-only batching choice. Omission preserves old configuration
    # identities, including checkpoints created before length grouping existed.
    unknown = set(raw) - set(DEFAULTS) - {'ddp_length_grouped_batches'}
    if unknown:
        raise TrainingError("unknown config keys: " + ", ".join(sorted(unknown)))
    cfg = dict(DEFAULTS, **raw)
    if 'ddp_length_grouped_batches' in cfg:
        require_bool(cfg['ddp_length_grouped_batches'], 'ddp_length_grouped_batches')
    for key in ("rank", "alpha", "max_sequence_length", "micro_batch_size",
                "gradient_accumulation_steps", "epochs", "checkpoint_every_n_steps", "loss_chunk_size"):
        integer(cfg[key], key)
    for key in ("seed", "max_steps"):
        integer(cfg[key], key, minimum=0)
    for key in ("local_files_only", "gradient_checkpointing", "is_fallback_reference", "allow_cached_nf4"):
        require_bool(cfg[key], key)
    if cfg["rank"] > 128 or cfg["max_sequence_length"] < 2:
        raise TrainingError("rank must be <=128 and sequence length >=2")
    for key in ("learning_rate", "warmup_ratio", "max_grad_norm", "weight_decay", "dropout"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise TrainingError(key + " must be a finite nonnegative number")
    if cfg["learning_rate"] == 0 or cfg["max_grad_norm"] == 0 or cfg["dropout"] >= 1 or cfg["warmup_ratio"] >= 1:
        raise TrainingError("invalid learning rate/gradient norm/dropout/warmup")
    if cfg["quant_backend"] not in ("bf16", "nf4"):
        raise TrainingError("quant_backend must be bf16 or nf4")
    if cfg["kbit_prepare"] not in ("norms_fp32", "reference_fp32"):
        raise TrainingError("kbit_prepare must be norms_fp32 or reference_fp32")
    if cfg["loss_backend"] not in ("model", "selected_chunked"):
        raise TrainingError("loss_backend must be model or selected_chunked")
    if cfg["allow_cached_nf4"] and cfg["quant_backend"] != "nf4":
        raise TrainingError("allow_cached_nf4 requires quant_backend=nf4")
    if cfg["compute_dtype"] not in ("bfloat16", "float16", "float32"):
        raise TrainingError("unsupported compute_dtype")
    if cfg["quant_backend"] == "bf16" and cfg["compute_dtype"] != "bfloat16":
        raise TrainingError("BF16 LoRA requires compute_dtype=bfloat16")
    if cfg["loss"] != "selected_assistant_tokens" or cfg["bias"] != "none":
        raise TrainingError("only selected_assistant_tokens loss and bias=none are supported")
    targets = cfg["target_text_modules"]
    if not isinstance(targets, list) or not targets or not set(targets).issubset(TEXT_TARGETS):
        raise TrainingError("target_text_modules must be explicit supported text projection suffixes")
    check_base_family(cfg["base_family"])
    if "qat" not in cfg["base_family"].lower() and not cfg["is_fallback_reference"]:
        raise TrainingError("ordinary IT requires is_fallback_reference=true")
    if cfg["model_name_or_path"]:
        check_base_family(cfg["model_name_or_path"])
    # A CT directory is a valid tokenizer source, but never training weights.
    known_plain_it = str(cfg["model_name_or_path"] or "").lower().rstrip("/")
    if re.fullmatch(r"(?:google|unsloth)/gemma-4-(?:31b|12b)-it", known_plain_it) and not cfg["is_fallback_reference"]:
        raise TrainingError("ordinary IT weights require is_fallback_reference=true; do not relabel them QAT")
    if cfg["base_family"] == "gemma-4-12b-it-qat-q4_0-unquantized":
        if not cfg["model_source_manifest"] or not cfg["chat_template_path"]:
            raise TrainingError("12B QAT requires an explicit model_source_manifest and deployment CT chat_template_path")
    for key in ("model_revision", "tokenizer_revision"):
        if cfg[key] and str(cfg[key]).lower().startswith("kaggle"):
            raise TrainingError("Kaggle version is not an HF revision; use a local path or HF commit/tag")
    return cfg


def source_options(source, revision, local_files_only):
    if not source:
        raise TrainingError("supply --model/local model_name_or_path or an explicit HF source")
    local = Path(source).expanduser().is_dir()
    if not local and (str(source).startswith(("/", "./", "../", "~")) or "/" not in str(source)):
        raise TrainingError("local model/tokenizer directory does not exist: " + str(source))
    if not local and not revision:
        raise TrainingError("remote or cached HF source requires an explicit revision: " + str(source))
    return str(Path(source).expanduser()) if local else source, {
        "local_files_only": local_files_only, "trust_remote_code": False,
        **({"revision": revision} if not local else {}),
    }


def resolve_target_modules(named_modules, requested=None):
    """Exact projection suffixes, actual text stack only, no vision/LM-head."""
    requested = set(requested or TARGET_SUBSTRINGS)
    names = named_modules.keys() if isinstance(named_modules, dict) else named_modules
    result = []
    for name in names:
        if any(part in name.lower() for part in FORBIDDEN):
            continue
        if name.rsplit(".", 1)[-1] not in requested:
            continue
        if not re.search(r"(?:^|\.)(?:language_model|text_model)(?:\.|$)", name):
            continue
        if isinstance(named_modules, dict):
            module = named_modules[name]
            if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
                continue
        result.append(name)
    present = {name.rsplit(".", 1)[-1] for name in result}
    if present != requested:
        raise TrainingError("actual text modules missing requested projections: " + str(sorted(requested - present)))
    return sorted(result)


def require_stack(quant_backend):
    names = ["torch", "transformers", "peft", "safetensors"]
    if quant_backend == "nf4":
        names += ["bitsandbytes", "accelerate"]
    modules, failures = {}, []
    for name in names:
        try:
            modules[name] = importlib.import_module(name)
        except Exception as exc:
            failures.append(name + ": " + str(exc))
    if failures:
        raise TrainingError("training dependencies unavailable; install a compatible stack explicitly: " + "; ".join(failures))
    return modules


class SelectedLabelCollator:
    def __init__(self, pad_token_id, torch_module):
        self.pad_token_id, self.torch = pad_token_id, torch_module

    def __call__(self, rows):
        values = pad_encoded_batch(rows, self.pad_token_id)
        return {key: self.torch.tensor(value, dtype=self.torch.long) for key, value in values.items()}


def validate_tokenizer_protocol(tokenizer, provenance, cfg):
    """Match encoded tokens, deployment template and complete tools before loading weights."""
    try:
        from .encoding.gemma_tool_chat import tokenizer_fingerprint, normalize_tools, sha256_json
    except ImportError:
        from encoding.gemma_tool_chat import tokenizer_fingerprint, normalize_tools, sha256_json
    if tokenizer_fingerprint(tokenizer) != provenance["tokenizer_hash"]:
        raise TrainingError("encoded tokenizer hash differs from loaded tokenizer; no model weights loaded")
    if not cfg.get("tool_schema_path"):
        raise TrainingError("real training requires the deployment tool schema via --tools/tool_schema_path")
    tools, protocol_id = normalize_tools(json.loads(Path(cfg["tool_schema_path"]).read_text(encoding="utf-8")))
    if sha256_json(tools) != provenance["tools_hash"]:
        raise TrainingError("encoded tool schema hash differs from deployment tools")
    if protocol_id and protocol_id != provenance["protocol_id"]:
        raise TrainingError("deployment tools protocol_id differs from encoded data")
    if cfg.get("chat_template_path"):
        template = Path(cfg["chat_template_path"]).read_text(encoding="utf-8")
        template_source = str(Path(cfg["chat_template_path"]).resolve())
    else:
        template = tokenizer.get_chat_template(tools=tools)
        template_source = "loaded-tokenizer-default-tool-template"
    if not isinstance(template, str) or not template:
        raise TrainingError("no unambiguous deployment chat template; use explicit --chat-template")
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if digest != provenance["chat_template_hash"]:
        raise TrainingError("encoded chat template differs from deployment template; pass matching explicit --chat-template if intentional")
    return {"chat_template_source": template_source, "chat_template_hash": digest,
            "tool_schema_path": str(Path(cfg["tool_schema_path"]).resolve()),
            "tools_hash": provenance["tools_hash"], "protocol_hash": provenance["protocol_hash"]}


def validate_nf4_cache(source, model_config, cfg):
    """Accept only an explicitly selected, provenance-labelled QAT NF4 cache.

    Verify its small manifest and file sizes without spending GPU time rehashing
    17 GB. The CPU producer hashes every saved file; mounted Kaggle versions are
    immutable. This does not claim NF4 equals the competition CT quantization.
    """
    quant = getattr(model_config, "quantization_config", None)
    if not quant:
        return None
    if not cfg["allow_cached_nf4"] or not Path(source).is_dir():
        raise TrainingError("prequantized base requires an explicit local QAT NF4 cache and allow_cached_nf4=true")
    quant = quant.to_dict() if hasattr(quant, "to_dict") else dict(quant)
    if (quant.get("quant_method") != "bitsandbytes" or
        not quant.get("load_in_4bit", quant.get("_load_in_4bit")) or
        quant.get("bnb_4bit_quant_type") != "nf4" or not quant.get("bnb_4bit_use_double_quant")):
        raise TrainingError("cached base is not the supported double-quantized bitsandbytes NF4")
    if str(quant.get("bnb_4bit_compute_dtype")).removeprefix("torch.") != cfg["compute_dtype"]:
        raise TrainingError("cached NF4 compute dtype differs from the requested training dtype")
    path = Path(source) / "cache-manifest.json"
    if not path.is_file():
        raise TrainingError("NF4 cache is missing its official-source provenance manifest")
    manifest = json.loads(path.read_text())
    if cfg["base_family"] == "gemma-4-31b-it-qat-q4_0-unquantized":
        expected = "google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"
        if manifest.get("source") != expected:
            raise TrainingError("NF4 cache source is not official 31B QAT-unquantized v2")
    elif cfg["base_family"] == "gemma-4-12b-it-qat-q4_0-unquantized":
        if not cfg.get("model_source_manifest"):
            raise TrainingError("12B NF4 cache requires model_source_manifest")
        src_manifest = json.loads(Path(cfg["model_source_manifest"]).read_text(encoding="utf-8"))
        training = src_manifest["training_source"]
        if (manifest.get("base_family") != cfg["base_family"]
                or manifest.get("source_repo_id") != training["repo_id"]
                or manifest.get("source_revision") != training["revision"]
                or manifest.get("source_config_sha256") != training["config_sha256"]
                or manifest.get("source_weights_sha256") != training["weights"]["sha256"]
                or manifest.get("source_weights_bytes") != training["weights"]["bytes"]):
            raise TrainingError("12B NF4 cache provenance does not match pinned 12B QAT source manifest; 31B or plain IT caches cannot be reused")
    else:
        raise TrainingError("unsupported base_family for NF4 cache: " + str(cfg["base_family"]))
    files = manifest.get("files", [])
    if not files or not any(x["name"].endswith(".safetensors") for x in files):
        raise TrainingError("NF4 cache manifest has no weights")
    for item in files:
        name = item["name"]
        if Path(name).name != name or not (Path(source)/name).is_file() or (Path(source)/name).stat().st_size != item["bytes"]:
            raise TrainingError("NF4 cache file is missing or has changed size: " + name)
    config_entry = next((item for item in files if item["name"] == "config.json"), None)
    config_sha256 = sha256_file(Path(source) / "config.json")
    if config_entry is None or config_entry.get("sha256") != config_sha256:
        raise TrainingError("NF4 cache config differs from its creation manifest")
    return {"source": manifest.get("source") or f"{manifest.get('source_repo_id')}@{manifest.get('source_revision')}",
            "manifest_sha256": sha256_file(path),
            "config_sha256": config_sha256,
            "source_config_sha256": manifest.get("source_config_sha256"),
            "source_index_sha256": manifest.get("source_index_sha256"),
            "source_weights_sha256": manifest.get("source_weights_sha256"),
            "source_weights_bytes": manifest.get("source_weights_bytes"),
            "validation": "manifest source + actual config SHA256 + saved file sizes; full weight hashes recorded at CPU creation"}


def validate_pinned_model_source(source, options, tok_source, tok_options, cfg, transformers):
    """Check explicit QAT source and deployment tokenizer/template before weights.

    A local mirror must also match the published weight hash (or a verified
    same-source NF4 cache manifest when allow_cached_nf4=true), so renaming plain
    IT or a 31B directory is not sufficient. Remote HF sources require the exact
    recorded commit. Existing 31B recipes retain their cache validation path.
    """
    manifest_path = cfg.get("model_source_manifest")
    if not manifest_path:
        return None
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("base_family") != cfg["base_family"]:
        raise TrainingError("model source manifest does not match base_family")
    training = manifest["training_source"]
    local = Path(source).is_dir()
    if not local and (source != training["repo_id"] or options.get("revision") != training["revision"]):
        raise TrainingError("model source/revision differs from the pinned QAT manifest")

    def source_file(root, load_options, name):
        if Path(root).is_dir():
            path = Path(root) / name
        else:
            path = Path(transformers.utils.hub.cached_file(root, name, **load_options))
        if not path.is_file():
            raise TrainingError("pinned source file missing: " + str(path))
        return path

    cache_manifest_path = Path(source) / "cache-manifest.json" if local else None
    cached_nf4 = bool(local and cfg.get("allow_cached_nf4") and cache_manifest_path and cache_manifest_path.is_file())
    if cached_nf4:
        cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        if (cache_manifest.get("base_family") != cfg["base_family"]
                or cache_manifest.get("source_repo_id") != training["repo_id"]
                or cache_manifest.get("source_revision") != training["revision"]
                or cache_manifest.get("source_config_sha256") != training["config_sha256"]
                or cache_manifest.get("source_weights_sha256") != training["weights"]["sha256"]
                or cache_manifest.get("source_weights_bytes") != training["weights"]["bytes"]):
            raise TrainingError("local NF4 cache manifest differs from pinned QAT source; do not relabel another base")
    else:
        config_path = source_file(source, options, "config.json")
        if sha256_file(config_path) != training["config_sha256"]:
            raise TrainingError("model config differs from the pinned unquantized QAT source")
        if local:
            weight = training["weights"]
            path = source_file(source, options, weight["name"])
            if path.stat().st_size != weight["bytes"] or sha256_file(path) != weight["sha256"]:
                raise TrainingError("local QAT weight file differs from its published source; do not relabel another base")
    deployment = manifest["deployment_source"]
    for name, digest in deployment["tokenizer_sha256"].items():
        if sha256_file(source_file(tok_source, tok_options, name)) != digest:
            raise TrainingError("tokenizer differs from pinned deployment CT file: " + name)
    if not cfg["chat_template_path"] or sha256_file(cfg["chat_template_path"]) != deployment["chat_template_sha256"]:
        raise TrainingError("chat template differs from pinned deployment CT template")
    return {"manifest_path": str(Path(manifest_path).resolve()),
            "manifest_sha256": sha256_file(manifest_path), **manifest,
            "local_weights_hash_checked": local and not cached_nf4,
            "local_nf4_cache_manifest_checked": cached_nf4}


def load_model_and_tokenizer(cfg, stack, resume=None, expected_provenance=None):
    torch, transformers, peft = stack["torch"], stack["transformers"], stack["peft"]
    source, options = source_options(cfg["model_name_or_path"], cfg["model_revision"], cfg["local_files_only"])
    tok_source, tok_options = source_options(
        cfg["tokenizer_name_or_path"] or source,
        cfg["tokenizer_revision"] or cfg["model_revision"], cfg["local_files_only"])
    pinned_source = validate_pinned_model_source(source, options, tok_source, tok_options, cfg, transformers)
    model_config = transformers.AutoConfig.from_pretrained(source, **options)
    model_type = getattr(model_config, "model_type", None)
    model_classes = {"gemma4": "Gemma4ForConditionalGeneration",
                     "gemma4_unified": "Gemma4UnifiedForConditionalGeneration"}
    if model_type not in model_classes:
        raise TrainingError("expected full gemma4 or gemma4_unified conditional-generation config")
    if model_type == "gemma4_unified" and not cfg["is_fallback_reference"]:
        if cfg["base_family"] != "gemma-4-12b-it-qat-q4_0-unquantized" or not pinned_source:
            raise TrainingError("production unified training requires the pinned 12B QAT recipe; do not relabel a 31B config")
    if pinned_source and model_type != pinned_source["training_source"]["model_type"]:
        raise TrainingError("model architecture differs from the pinned source")
    cache_info = validate_nf4_cache(source, model_config, cfg)
    model_class = getattr(transformers, model_classes[model_type], None)
    if model_class is None:
        raise TrainingError("installed transformers lacks " + model_classes[model_type])
    tokenizer = transformers.AutoTokenizer.from_pretrained(tok_source, **tok_options)
    if tokenizer.pad_token_id is None:
        raise TrainingError("tokenizer has no pad token; do not silently add new tokens")
    protocol_info = validate_tokenizer_protocol(tokenizer, expected_provenance, cfg) if expected_provenance else None
    device = torch.device(cfg["device"])
    if device.type not in ("cuda", "cpu"):
        raise TrainingError("this training runner supports explicit single CUDA or CPU devices only")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 and not torch.distributed.is_initialized():
        raise TrainingError("use training.train_ddp for a distributed launch; rank-local initialization is required")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA requested but unavailable")
    dtype = getattr(torch, cfg["compute_dtype"])
    if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise TrainingError("GPU does not support BF16; use NF4 compute_dtype=float16 on compatible hardware")
    kwargs = dict(options, dtype=dtype, attn_implementation=cfg["attn_implementation"])
    if cfg["quant_backend"] == "nf4":
        if device.type != "cuda":
            raise TrainingError("NF4 runner requires a CUDA device")
        if cache_info is None:
            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = {"": device.index if device.index is not None else torch.cuda.current_device()}
    model = model_class.from_pretrained(source, **kwargs)
    if cfg["quant_backend"] == "nf4":
        if cfg["kbit_prepare"] == "norms_fp32":
            try:
                from .memory import prepare_model_for_memory_efficient_kbit_training
            except ImportError:
                from memory import prepare_model_for_memory_efficient_kbit_training
            model = prepare_model_for_memory_efficient_kbit_training(
                model, use_gradient_checkpointing=cfg["gradient_checkpointing"])
        else:
            model = peft.prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg["gradient_checkpointing"],
                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.to(device)
        model.requires_grad_(False)
        if cfg["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    targets = resolve_target_modules(dict(model.named_modules()), cfg["target_text_modules"])
    if resume:
        inspect_adapter(Path(resume) / "adapter", cfg["rank"], cfg["target_text_modules"])
        # PEFT's bare "cuda" default can deserialize onto GPU 0 in every rank.
        # CPU loading copies into each rank's existing parameters without P2P.
        model = peft.PeftModel.from_pretrained(model, str(Path(resume) / "adapter"),
                                             is_trainable=True, local_files_only=True,
                                             torch_device="cpu")
    else:
        adapter = peft.LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"],
                                 lora_dropout=cfg["dropout"], bias="none",
                                 target_modules=targets, task_type=peft.TaskType.CAUSAL_LM)
        model = peft.get_peft_model(model, adapter)
    if cfg["loss_backend"] == "selected_chunked":
        try:
            from .memory import install_selected_token_loss
        except ImportError:
            from memory import install_selected_token_loss
        install_selected_token_loss(model, chunk_size=cfg["loss_chunk_size"])
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name or any(x in name.lower() for x in FORBIDDEN)
                            for name, _ in trainable):
        raise TrainingError("unexpected trainable parameters outside language LoRA")
    provenance = {"source": source, "requested_revision": cfg["model_revision"],
                  "model_type": model_type, "model_class": model_class.__name__, "pinned_source": pinned_source,
                  "resolved_commit": getattr(model_config, "_commit_hash", None),
                  "config_sha256": hashlib.sha256(model_config.to_json_string().encode()).hexdigest(),
                  "tokenizer_source": tok_source, "target_modules": targets, "encoding_protocol": protocol_info,
                  "nf4_cache": cache_info, "kbit_prepare": cfg["kbit_prepare"], "loss_backend": cfg["loss_backend"],
                  "trainable_parameters": sum(p.numel() for _, p in trainable)}
    return model, tokenizer, device, provenance


def _signature(cfg, data, model_info=None):
    """Bind pinned runs to verified base bytes, independently of mount paths.

    Runs without an NF4 cache retain their historical signature format.
    Cached runs additionally bind the effective quantized base, independently
    of its mount path. Old cache signatures cannot prove that base identity.
    """
    config = dict(cfg)
    payload = {"config": config, "data_sha256": data["sha256"],
               "provenance": data["provenance"]}
    if cfg.get("model_source_manifest"):
        pinned = (model_info or {}).get("pinned_source")
        if not pinned:
            raise TrainingError("checkpoint identity requires the verified pinned model source")
        training, deployment = pinned["training_source"], pinned["deployment_source"]
        payload["base_identity"] = {
            "manifest_sha256": pinned["manifest_sha256"],
            "base_family": pinned["base_family"],
            "repo_id": training["repo_id"], "revision": training["revision"],
            "model_type": training["model_type"],
            "config_sha256": training["config_sha256"],
            "weights_sha256": training["weights"]["sha256"],
            "weights_bytes": training["weights"]["bytes"],
        }
        config.update({
            "model_name_or_path": training["repo_id"], "model_revision": training["revision"],
            "tokenizer_name_or_path": deployment["kaggle_handle"], "tokenizer_revision": None,
            "model_source_manifest": pinned["manifest_sha256"],
            "chat_template_path": deployment["chat_template_sha256"],
            "tool_schema_path": data["provenance"]["tools_hash"],
        })
    else:
        # This optional key did not exist in historical 31B run configs.
        config.pop("model_source_manifest", None)
    cache = (model_info or {}).get("nf4_cache")
    if cache is not None:
        identity = {key: cache.get(key) for key in ("manifest_sha256", "config_sha256")}
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
               for value in identity.values()):
            raise TrainingError("NF4 checkpoint identity requires verified cache manifest and actual config hashes; old cache metadata is insufficient")
        payload["effective_nf4_base"] = identity
        config.update({
            "model_name_or_path": "nf4-cache:" + identity["manifest_sha256"],
            "tokenizer_name_or_path": data["provenance"]["tokenizer_hash"],
            "chat_template_path": data["provenance"]["chat_template_hash"],
            "tool_schema_path": data["provenance"]["tools_hash"],
        })
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_resume_state(resume, signature, torch, world_size=None):
    """Reject incompatible resumes before optimizer restore or metadata writes."""
    state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=True)
    if state["signature"] != signature:
        raise TrainingError("resume base/config/data/protocol fingerprint mismatch")
    if world_size is not None and state.get("world_size") != world_size:
        raise TrainingError("DDP resume requires identical world size")
    return state


def run_training_loop(model, rows, cfg, output_dir, torch, pad_token_id, signature, resume=None):
    """Actual selected-token SFT. Normalize over each full accumulation window.

    Checkpoints only occur at optimizer boundaries, including final short windows.
    Resume restores weights (loader), optimizer/scheduler/RNG and next batch.
    """
    output_dir = Path(output_dir)
    device = torch.device(cfg["device"])
    collate = SelectedLabelCollator(pad_token_id, torch)
    batches_per_epoch = math.ceil(len(rows) / cfg["micro_batch_size"])
    per_epoch = math.ceil(batches_per_epoch / cfg["gradient_accumulation_steps"])
    total_steps = per_epoch * cfg["epochs"]
    if cfg["max_steps"]:
        total_steps = min(total_steps, cfg["max_steps"])
    warmup = int(total_steps * cfg["warmup_ratio"])
    def schedule(step):
        if warmup and step < warmup:
            return float(step + 1) / warmup
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    use_scaler = device.type == "cuda" and cfg["compute_dtype"] == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    start_epoch, next_batch, global_step = 0, 0, 0
    if resume:
        state = load_resume_state(resume, signature, torch)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch, next_batch, global_step = state["epoch"], state["next_batch"], state["global_step"]
    if global_step >= total_steps:
        raise TrainingError("resume checkpoint has already reached configured optimizer step limit")
    def save_checkpoint(epoch, batch_index):
        dest = output_dir / ("checkpoint-%06d" % global_step)
        if dest.exists():
            raise TrainingError("checkpoint already exists: " + str(dest))
        dest.mkdir(parents=True)
        model.save_pretrained(dest / "adapter", safe_serialization=True)
        torch.save({"signature": signature, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                    "epoch": epoch, "next_batch": batch_index, "global_step": global_step},
                   dest / "training_state.pt")
        atomic_json(dest / "progress.json", {"global_step": global_step, "epoch": epoch,
                                            "next_batch": batch_index, "signature": signature})
        return dest
    model.train()
    started = time.monotonic()
    last_loss, processed_tokens, last_saved = None, 0, None
    for epoch in range(start_epoch, cfg["epochs"]):
        generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
        order = torch.randperm(len(rows), generator=generator).tolist()
        batches = [order[i:i + cfg["micro_batch_size"]] for i in range(0, len(order), cfg["micro_batch_size"])]
        offset = next_batch if epoch == start_epoch else 0
        if offset % cfg["gradient_accumulation_steps"] and offset != len(batches):
            raise TrainingError("resume is not at an optimizer boundary")
        for start in range(offset, len(batches), cfg["gradient_accumulation_steps"]):
            group = batches[start:start + cfg["gradient_accumulation_steps"]]
            denominator = sum(sum(label != -100 for label in rows[i]["labels"][1:])
                              for batch in group for i in batch)
            if denominator == 0:
                raise TrainingError("accumulation window has no supervised next-token labels")
            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            for indices in group:
                batch = {k: v.to(device) for k, v in collate([rows[i] for i in indices]).items()}
                count = int(batch["labels"][:, 1:].ne(-100).sum().item())
                dtype = getattr(torch, cfg["compute_dtype"])
                amp = torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()
                with amp:
                    result = model(**batch, use_cache=False)
                    loss = result.loss
                if loss is None or not torch.isfinite(loss).all():
                    raise TrainingError("non-finite/absent training loss")
                scaler.scale(loss * (count / denominator)).backward()
                weighted_loss += float(loss.detach()) * (count / denominator)
                processed_tokens += int(batch["attention_mask"].sum().item())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            last_loss = weighted_loss
            following = start + len(group)
            resume_epoch, resume_batch = (epoch + 1, 0) if following == len(batches) else (epoch, following)
            with open(output_dir / "metrics.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": global_step, "epoch": epoch, "loss": weighted_loss,
                                         "supervised_tokens": denominator,
                                         "learning_rate": optimizer.param_groups[0]["lr"]}) + "\n")
            print(json.dumps({"step": global_step, "planned_steps": total_steps,
                              "selected_token_loss": weighted_loss}), flush=True)
            if global_step % cfg["checkpoint_every_n_steps"] == 0 or global_step == total_steps:
                last_saved = save_checkpoint(resume_epoch, resume_batch)
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break
    model.save_pretrained(output_dir / "adapter", safe_serialization=True)
    return {"optimizer_steps": global_step, "loss": last_loss, "processed_tokens_this_run": processed_tokens,
            "wall_seconds_this_run": time.monotonic() - started, "last_checkpoint": str(last_saved)}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data", help="encoded transaction JSONL")
    p.add_argument("--out", help="new run directory")
    p.add_argument("--model", help="local unquantized QAT directory or explicit HF ID")
    p.add_argument("--revision", help="HF commit/tag, never a Kaggle version label")
    p.add_argument("--tokenizer")
    p.add_argument("--tokenizer-revision")
    p.add_argument("--tools", help="deployment tool-schema manifest matching encoded data")
    p.add_argument("--chat-template", help="explicit deployment Jinja file; must match encoded hash")
    p.add_argument("--allow-download", action="store_true", help="explicitly allow HF network loading")
    p.add_argument("--device", help="single cuda[:index] or cpu")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--resume", help="checkpoint-N directory; output must be its existing run")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        raw = load_config(a.config)
        for attr, key in (("model", "model_name_or_path"), ("revision", "model_revision"),
                          ("tokenizer", "tokenizer_name_or_path"), ("tokenizer_revision", "tokenizer_revision"),
                          ("tools", "tool_schema_path"), ("chat_template", "chat_template_path"),
                          ("device", "device"), ("max_steps", "max_steps")):
            if getattr(a, attr) is not None:
                raw[key] = getattr(a, attr)
        if a.allow_download:
            raw["local_files_only"] = False
        cfg = validate_config(raw)
        rows, data = load_encoded_jsonl(a.data, cfg["max_sequence_length"]) if a.data else (None, None)
        if data and data["provenance"]["protocol_id"] != cfg["protocol_id"]:
            raise TrainingError("data protocol_id differs from training config")
        if a.dry_run:
            print(json.dumps({"dry_run": True, "config": cfg, "data": data,
                              "weights_loaded": False, "dependencies_imported": False,
                              "status": "plan_only; not a training result"}, ensure_ascii=False, indent=2))
            return 0
        if not a.data or not a.out:
            raise TrainingError("real training requires --data and --out")
        source_options(cfg["model_name_or_path"], cfg["model_revision"], cfg["local_files_only"])
        stack = require_stack(cfg["quant_backend"])
        output = Path(a.out).resolve()
        if a.resume:
            resume = Path(a.resume).resolve()
            if resume.parent != output or not (resume / "training_state.pt").is_file():
                raise TrainingError("--resume must name a complete checkpoint inside --out")
        elif output.exists() and any(output.iterdir()):
            raise TrainingError("output directory is not empty; use a new run or --resume")
        torch = stack["torch"]
        random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg["seed"])
        model, tokenizer, device, model_info = load_model_and_tokenizer(
            cfg, stack, a.resume, expected_provenance=data["provenance"])
        vocabulary_size = model.get_input_embeddings().num_embeddings
        if any(max(row["input_ids"]) >= vocabulary_size for row in rows):
            raise TrainingError("encoded token id exceeds the model embedding vocabulary")
        output.mkdir(parents=True, exist_ok=True)
        signature = _signature(cfg, data, model_info)
        if a.resume:
            load_resume_state(a.resume, signature, torch)
        atomic_json(output / "run_config.json", {"config": cfg, "data": data, "model": model_info,
                                               "signature": signature})
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = run_training_loop(model, rows, cfg, output, torch, tokenizer.pad_token_id, signature, a.resume)
        adapter = inspect_adapter(output / "adapter", cfg["rank"], cfg["target_text_modules"])
        manifest = {"manifest_version": 2, "status": "training_completed", "config": cfg,
                    "model": model_info, "data": data, "quant_backend": cfg["quant_backend"],
                    "signature": signature, "metrics": result, "adapter": adapter,
                    "versions": {key: getattr(mod, "__version__", "unknown") for key, mod in stack.items()},
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    "runtime_compatibility_verified": False}
        atomic_json(output / "training_manifest.json", manifest)
        print(json.dumps({"ok": True, "adapter": str(output / "adapter"),
                          "manifest": str(output / "training_manifest.json")}))
        return 0
    except Exception as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
