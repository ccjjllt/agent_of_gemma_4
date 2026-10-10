#!/usr/bin/env python3
"""Bounded, single-device diagnostic for Gemma4 QAT-unquantized -> NF4 LoRA.

Local file only: this worker never installs packages, downloads weights, launches
notebooks or starts other workers. The caller must enforce a 900-second process
timeout. Report defaults to /kaggle/working/gpu-worker-report.json. Input is synthetic,
for numerical/capacity diagnosis only, not training quality or benchmark scoring.
"""
import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import traceback

EXPECTED_SLUG = "gemma-4-31b-it-qat-q4_0-unquantized"
EXPECTED_SOURCE = "google/gemma-4/Transformers/" + EXPECTED_SLUG + "/2"
TRAINER_REFERENCE_SHA256 = "7c50f65f2db10291de84be5ea6ce0215b442866f360f7dccf2119112a5d92112"
TARGETS = {"q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
FORBIDDEN = ("vision", "visual", "audio", "projector", "embed", "lm_head", "norm")
GIB = 1024 ** 3


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Recorder:
    def __init__(self, root, budget, report_path):
        self.root, self.started, self.budget = root, time.monotonic(), budget
        self.report_path = report_path
        self.report = {"schema_version": 1, "status": "running",
                       "purpose": "diagnostic synthetic FP/BP, not model-quality training",
                       "external_timeout_required_seconds": 900, "internal_budget_seconds": budget,
                       "expected_source": EXPECTED_SOURCE, "verified_optimizer_steps": 0,
                       "events": [], "trainer_reference_sha256": TRAINER_REFERENCE_SHA256,
                       "loader_equivalence": {
                           "model_class": "AutoModelForImageTextToText",
                           "dtype": "bfloat16", "attention": "sdpa", "device_map": {"": 0},
                           "quantization": "NF4, double quant, bfloat16 compute",
                           "gradient_checkpointing": True, "use_reentrant": False,
                           "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0,
                           "bias": "none", "targets": sorted(TARGETS),
                           "offload": False, "retries": 0,
                           "note": "Equivalent original single-device loading path; no memory workaround."}}
        write_json(self.report_path, self.report)

    def event(self, stage, **fields):
        event = {"stage": stage, "elapsed_seconds": round(time.monotonic() - self.started, 3), **fields}
        self.report["events"].append(event)
        self.report["last_stage"] = stage
        write_json(self.report_path, self.report)
        print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)

    def remaining(self):
        return self.budget - (time.monotonic() - self.started)


def check_model(path):
    config_path, index_path = path / "config.json", path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise ValueError("model requires config.json and model.safetensors.index.json")
    cfg = json.loads(config_path.read_text())
    text = cfg.get("text_config") or {}
    if cfg.get("model_type") != "gemma4" or cfg.get("quantization_config"):
        raise ValueError("expected unquantized gemma4 config, not CT or prequantized bnb")
    if text.get("hidden_size") != 5376 or text.get("num_hidden_layers") != 60:
        raise ValueError("config is not the expected dense Gemma4 31B architecture")
    if EXPECTED_SLUG not in str(path).lower():
        raise ValueError("mounted model path must identify the QAT-unquantized family")
    parts = path.parts
    for i, part in enumerate(parts[:-1]):
        if part.lower() == EXPECTED_SLUG and parts[i + 1].isdigit() and parts[i + 1] != "2":
            raise ValueError("mounted QAT model is not expected version 2")
    index = json.loads(index_path.read_text())
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("empty/invalid model shard index")
    shards = sorted(set(mapping.values()))
    for name in shards:
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("invalid model shard filename")
        if not (path / name).is_file() or (path / name).stat().st_size == 0:
            raise ValueError("missing/empty model shard: " + name)
    return {"path": str(path.resolve()), "model_type": cfg["model_type"],
            "text_hidden_size": text["hidden_size"], "text_layers": text["num_hidden_layers"],
            "config_sha256": sha256(config_path), "index_sha256": sha256(index_path),
            "shards": {name: (path / name).stat().st_size for name in shards},
            "shard_total_bytes": sum((path / name).stat().st_size for name in shards),
            "index_total_size": index.get("metadata", {}).get("total_size"),
            "source_verification": "Kaggle Input family/version path + config/index; full weights not rehashed"}


def discover_model(input_root, explicit=None):
    root = Path(input_root).resolve()
    if explicit:
        candidate = Path(explicit).resolve()
        candidate.relative_to(root)
        return candidate, check_model(candidate)
    found = []
    for config in root.rglob("config.json"):
        if EXPECTED_SLUG not in str(config.parent).lower():
            continue
        try:
            info = check_model(config.parent)
        except (ValueError, OSError):
            continue
        found.append((config.parent, info))
    if len(found) != 1:
        raise ValueError("expected exactly one mounted QAT-unquantized v2 model; found %d" % len(found))
    return found[0]


def memory(torch):
    free, total = torch.cuda.mem_get_info(0)
    return {"free_bytes": free, "total_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(0),
            "reserved_bytes": torch.cuda.memory_reserved(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}


def parameter_hash(parameters):
    # Stream one LoRA tensor at a time; never copy all base weights.
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_adapter(model, out, safe_open):
    staging = Path(tempfile.mkdtemp(prefix=".adapter-", dir=str(out)))
    try:
        model.save_pretrained(staging, safe_serialization=True)
        cfg = json.loads((staging / "adapter_config.json").read_text())
        if cfg.get("r") != 8 or cfg.get("peft_type") != "LORA" or cfg.get("task_type") != "CAUSAL_LM":
            raise ValueError("unexpected saved adapter config")
        pairs = {}
        with safe_open(str(staging / "adapter_model.safetensors"), framework="numpy") as tensors:
            for name in tensors.keys():
                match = re.fullmatch(r"(base_model\.model\..+)\.lora_([AB])\.weight", name)
                if not match:
                    raise ValueError("unexpected saved tensor: " + name)
                module, side = match.groups()
                shape = tensors.get_slice(name).get_shape()
                if module.rsplit(".", 1)[-1] not in TARGETS or any(x in module.lower() for x in FORBIDDEN):
                    raise ValueError("unexpected adapter target")
                if len(shape) != 2 or shape[0 if side == "A" else 1] != 8:
                    raise ValueError("saved adapter rank/shape mismatch")
                pairs.setdefault(module, set()).add(side)
        if not pairs or any(sides != {"A", "B"} for sides in pairs.values()):
            raise ValueError("saved adapter has incomplete A/B pairs")
        dest = out / "adapter"
        dest.mkdir(exist_ok=True)
        files = {}
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            os.replace(staging / name, dest / name)
            files[name] = {"bytes": (dest / name).stat().st_size, "sha256": sha256(dest / name)}
        return {"files": files, "modules": len(pairs), "rank": 8}
    finally:
        # Any PEFT README generated during save is a disposable staging artifact.
        shutil.rmtree(staging, ignore_errors=True)


def run(args, rec):
    rec.event("model_discovery_begin")
    model_path, model_info = discover_model(args.input_root, args.model)
    rec.report["model"] = model_info
    rec.event("model_discovery_ok", shard_count=len(model_info["shards"]),
              shard_total_bytes=model_info["shard_total_bytes"])
    rec.event("dependencies_begin")
    modules = {name: importlib.import_module(name) for name in
               ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "safetensors", "numpy")}
    torch, tr, peft, bnb = (modules[n] for n in ("torch", "transformers", "peft", "bitsandbytes"))
    rec.report["versions"] = {n: getattr(m, "__version__", "unknown") for n, m in modules.items()}
    rec.report["python"] = sys.version
    rec.report["torch_cuda_build"] = torch.version.cuda
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; not a GPU training result")
    torch.cuda.set_device(0)
    devices = [{"index": i, "name": torch.cuda.get_device_name(i),
                "bytes": torch.cuda.get_device_properties(i).total_memory}
               for i in range(torch.cuda.device_count())]
    rec.report["allocated_devices"] = devices
    rec.report["used_device"] = 0
    if "L4" not in devices[0]["name"].upper():
        raise RuntimeError("this original-path probe requires device 0 to be L4")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unavailable on the allocated device")
    torch.manual_seed(20261010)
    torch.cuda.manual_seed_all(20261010)
    rec.event("dependencies_and_gpu_ok", devices=devices, memory=memory(torch))

    rec.event("small_bnb_backward_begin")
    started = time.monotonic()
    layer = bnb.nn.Linear4bit(32, 16, bias=False, compute_dtype=torch.bfloat16,
                             quant_type="nf4", compress_statistics=True).to("cuda:0")
    x = torch.randn(2, 32, device="cuda:0", dtype=torch.bfloat16, requires_grad=True)
    loss = layer(x).float().square().mean()
    loss.backward()
    torch.cuda.synchronize()
    norm = float(x.grad.float().norm())
    if not math.isfinite(float(loss)) or not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("small bnb forward/backward produced invalid loss/gradient")
    rec.event("small_bnb_backward_ok", seconds=time.monotonic() - started,
              finite_loss=float(loss.detach()), nonzero_input_grad_norm=norm, memory=memory(torch))
    del loss, x, layer
    gc.collect()
    torch.cuda.empty_cache()

    rec.event("tokenizer_and_config_begin")
    config = tr.AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    tokenizer = tr.AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    if config.model_type != "gemma4" or getattr(config, "quantization_config", None):
        raise RuntimeError("base config changed or is already quantized")
    rec.event("tokenizer_and_config_ok", tokenizer_class=type(tokenizer).__name__)

    rec.event("original_nf4_load_begin", memory=memory(torch))
    started = time.monotonic()
    model = tr.AutoModelForImageTextToText.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa",
        quantization_config=tr.BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={"": 0})
    torch.cuda.synchronize()
    rec.event("original_nf4_load_ok", seconds=time.monotonic() - started, memory=memory(torch))
    rec.event("kbit_prepare_begin")
    model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                 gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    rec.event("kbit_prepare_ok", memory=memory(torch))
    targets = sorted(name for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in TARGETS
        and not any(x in name.lower() for x in FORBIDDEN)
        and re.search(r"(?:^|\.)(?:language_model|text_model)(?:\.|$)", name)
        and hasattr(module, "in_features") and hasattr(module, "out_features"))
    if {name.rsplit(".", 1)[-1] for name in targets} != TARGETS:
        raise RuntimeError("actual model text projection targets are incomplete")
    rec.event("lora_injection_begin", target_count=len(targets))
    model = peft.get_peft_model(model, peft.LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=targets, task_type=peft.TaskType.CAUSAL_LM))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n or any(x in n.lower() for x in FORBIDDEN) for n, _ in trainable):
        raise RuntimeError("trainable parameters escaped the language LoRA")
    rec.report["trainable_parameters"] = sum(p.numel() for _, p in trainable)
    rec.report["base_parameters_frozen"] = all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)
    before = parameter_hash(trainable)
    rec.event("lora_injection_ok", trainable_parameters=rec.report["trainable_parameters"],
              initial_lora_sha256=before, memory=memory(torch))
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=5e-5, weight_decay=0.0)
    vocabulary = model.get_input_embeddings().num_embeddings
    model.train()

    def step(length, index):
        rec.event("optimizer_step_begin", step=index, sequence_length=length, memory=memory(torch))
        optimizer.zero_grad(set_to_none=True)
        ids = torch.randint(100, min(10000, vocabulary), (1, length), device="cuda:0")
        labels = ids.clone()
        labels[:, :-64] = -100
        torch.cuda.reset_peak_memory_stats(0)
        started = time.monotonic()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           labels=labels, use_cache=False)
            loss = output.loss
        torch.cuda.synchronize()
        forward_seconds = time.monotonic() - started
        if loss is None or not torch.isfinite(loss).all():
            raise RuntimeError("non-finite forward loss")
        rec.event("forward_ok", step=index, sequence_length=length,
                  finite_loss=float(loss.detach()), seconds=forward_seconds, memory=memory(torch))
        rec.event("backward_begin", step=index, sequence_length=length)
        started_backward = time.monotonic()
        loss.backward()
        torch.cuda.synchronize()
        backward_seconds = time.monotonic() - started_backward
        rec.event("backward_ok", step=index, seconds=backward_seconds, memory=memory(torch))
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [p for _, p in trainable], 1.0, error_if_nonfinite=True))
        if not math.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError("gradient is non-finite or identically zero")
        rec.event("optimizer_update_begin", step=index, nonzero_grad_norm=grad_norm, memory=memory(torch))
        optimizer.step()
        torch.cuda.synchronize()
        rec.report["verified_optimizer_steps"] = index
        rec.event("optimizer_step_ok", step=index, sequence_length=length,
                  supervised_tokens=64, finite_loss=float(loss.detach()),
                  nonzero_grad_norm=grad_norm, forward_seconds=forward_seconds,
                  backward_seconds=backward_seconds,
                  step_seconds=time.monotonic() - started, memory=memory(torch))
        del output, loss, ids, labels
        optimizer.zero_grad(set_to_none=True)

    for i in range(1, 3):
        step(256, i)
    after = parameter_hash(trainable)
    if before == after:
        raise RuntimeError("two optimizer updates left all LoRA tensors unchanged")
    rec.report["lora_changed_after_256"] = True
    rec.event("adapter_save_begin", completed_steps=2)
    rec.report["adapter"] = save_adapter(model, rec.root, modules["safetensors"].safe_open)
    rec.report["adapter"]["saved_after_optimizer_step"] = 2
    rec.event("adapter_save_ok", lora_sha256=after, adapter=rec.report["adapter"])

    gc.collect()
    torch.cuda.empty_cache()
    mem = memory(torch)
    eligible = rec.remaining() >= 180 and mem["free_bytes"] >= 2 * GIB and not args.skip_1024
    rec.event("optional_1024_decision", run=eligible, remaining_seconds=rec.remaining(),
              minimum_remaining_seconds=180, minimum_free_bytes=2 * GIB, memory=mem)
    if eligible:
        step(1024, 3)
        rec.report["adapter"] = save_adapter(model, rec.root, modules["safetensors"].safe_open)
        rec.report["adapter"]["saved_after_optimizer_step"] = 3
        rec.report["status"] = "success_256x2_and_1024x1"
    else:
        rec.report["status"] = "success_256x2_1024_not_attempted"
    rec.event("complete", status=rec.report["status"], adapter=rec.report["adapter"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--model", help="explicit mounted model directory under --input-root")
    parser.add_argument("--out", default="/kaggle/working/gpu-worker-report.json", help="stage JSON report file")
    parser.add_argument("--artifact-dir", default="/kaggle/working/gpu-probe-artifacts", help="new/empty directory for adapter only")
    parser.add_argument("--budget-seconds", type=int, default=840,
                        help="soft accounting budget; caller must enforce external 900-second kill")
    parser.add_argument("--skip-1024", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.artifact_dir)
    report_path = Path(args.out)
    if report_path.exists() or (out.exists() and any(out.iterdir())):
        print("ERROR: report must be new and artifact directory must be empty", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rec = Recorder(out, args.budget_seconds, report_path)
    try:
        run(args, rec)
        return 0
    except Exception as exc:
        rec.report["status"] = "failed_original_path"
        if rec.report["verified_optimizer_steps"] >= 2 and rec.report.get("adapter"):
            rec.report["status"] = "success_256x2_later_stage_failed"
        try:
            torch = sys.modules.get("torch")
            if torch is not None and torch.cuda.is_available():
                rec.report["failure_memory"] = memory(torch)
        except Exception:
            pass
        rec.event("failure", failed_stage=rec.report.get("last_stage"),
                  exception_type=type(exc).__name__, message=str(exc),
                  traceback=traceback.format_exc())
        return 2


if __name__ == "__main__":
    sys.exit(main())
