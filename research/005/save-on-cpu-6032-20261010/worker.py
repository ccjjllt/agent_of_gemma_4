"""Four-L4 one-step diagnostic on the 6032-token complete transaction.

One arm per process. Baseline OOM must not leave a live DDP reducer for the
save_on_cpu arm. Capacity adapter updates are discarded. No 16K retry.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

SCHEMA = "p005-save-on-cpu-6032"
CACHE_KERNEL = "ccjjllt/p005-cpu-nf4-cache"
SOURCE = "google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"
WORLD = 4
RESERVED = {
    "fastapi_13713", "requests_6592", "requests_6644", "requests_7309",
    "requests_7505", "rich_3130", "rich_3296", "rich_3469",
}
TX = "fastapi_11194-3f51fb716edd-tx01"
SEQ = 6032
ENCODED_SHA256 = "38cd361b7c1f15d78492963eddc226c17699ae11ec1d0c850071a7d02b585ce5"


def write_json(path, value):
    if not isinstance(value, dict):
        raise TypeError("Report must be a JSON object")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def find_cache(input_root):
    matches = sorted(p for p in Path(input_root).rglob("cache-manifest.json") if p.parent.name == "qat-nf4")
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one mounted qat-nf4/cache-manifest.json")
    return matches[0].parent


def verify_cache(cache):
    manifest_path = Path(cache) / "cache-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("source") != SOURCE:
        raise RuntimeError("Cache source is not official QAT-unquantized v2")
    producer = json.loads((Path(cache).parent / "cache-report.json").read_text())
    child = json.loads((Path(cache).parent / "cache-child-report.json").read_text())
    if not isinstance(producer, dict) or producer.get("success") is not True:
        raise RuntimeError("CPU cache producer did not report success")
    if not isinstance(child, dict) or child.get("success") is not True or child.get("stage") != "complete":
        raise RuntimeError("CPU NF4 cache did not complete")
    tok = Path(cache) / "tokenizer.json"
    if not tok.is_file() or tok.stat().st_size != 32169626:
        raise RuntimeError("NF4 cache tokenizer.json is missing or the wrong size")
    return {"kernel": CACHE_KERNEL, "path": str(cache), "source": SOURCE,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "cache_total_bytes": manifest.get("total_bytes"),
            "cpu_cache_verified_complete": True}


def cuda_memory(torch, device):
    if torch is None or device is None or device.type != "cuda":
        return None
    try:
        free, total = torch.cuda.mem_get_info(device)
        return {"allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "free_bytes": free, "total_bytes": total}
    except Exception as exc:
        return {"error": repr(exc)}


def is_oom(exc):
    text = (repr(exc) + " " + str(exc)).lower()
    return "out of memory" in text or "cuda oom" in text


def gib(nbytes):
    if not isinstance(nbytes, int):
        return None
    return round(nbytes / (1024 ** 3), 3)


def load_rows(path):
    from training.common import load_encoded_jsonl, TrainingError
    rows, data = load_encoded_jsonl(path, 8192, require_real=False)
    if data.get("sha256") != ENCODED_SHA256:
        raise TrainingError("encoded capacity SHA256 drifted")
    if len(rows) != 1:
        raise TrainingError("capacity JSONL must contain exactly one row")
    row, meta = rows[0], rows[0]["meta"]
    if meta.get("transaction_id") != TX:
        raise TrainingError("unexpected transaction_id: " + str(meta.get("transaction_id")))
    if len(row["input_ids"]) != SEQ:
        raise TrainingError("expected 6032 tokens, got " + str(len(row["input_ids"])))
    if meta.get("synthetic") is not False or meta.get("truncated") or meta.get("truncated_tool_block"):
        raise TrainingError("refusing synthetic or truncated capacity rows")
    group = meta.get("group") or {}
    if group.get("split") != "select":
        raise TrainingError("capacity row must stay on the select split")
    if group.get("issue") in RESERVED:
        raise TrainingError("reserved pairing issue leaked")
    return rows, data


def decoder_layers(model):
    base = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    language = getattr(getattr(base, "model", None), "language_model", None)
    layers = getattr(language, "layers", None)
    return list(layers) if layers is not None else []


def attach_layer_hooks(layers, tracker):
    handles = []

    def pre(index, name):
        def hook(_module, _args):
            tracker["last_decoder_layer"] = index
            tracker["last_decoder_name"] = name
            tracker["last_decoder_phase"] = tracker.get("phase")
        return hook

    for index, layer in enumerate(layers):
        name = "language_model.layers.%d" % index
        handles.append(layer.register_forward_pre_hook(pre(index, name)))
    return handles


def adam_state_bytes(optimizer, torch):
    tensors, nbytes = 0, 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                tensors += 1
                nbytes += int(value.numel() * value.element_size())
    return {"tensors": tensors, "bytes": nbytes, "gib": gib(nbytes)}


def run_one_step(model, rows, cfg, torch, dist, device, tokenizer, tracker, emit, report):
    from training import train_ddp as ddp
    from training import train_sft as single

    rank, world = dist.get_rank(), dist.get_world_size()
    layers = decoder_layers(model)
    tracker["n_decoder_layers"] = len(layers)
    handles = attach_layer_hooks(layers, tracker)
    report["n_decoder_layers"] = len(layers)
    phases = report["phases"]

    def begin(name):
        tracker["phase"] = name
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        entry = cuda_memory(torch, device)
        emit(name + "_begin", memory=entry, last_decoder_layer=tracker.get("last_decoder_layer"))
        return started, entry

    def end(name, started, extra=None):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        memory = cuda_memory(torch, device)
        payload = {
            "stage": name, "seconds": time.monotonic() - started, "memory": memory,
            "peak_allocated_bytes": None if not memory else memory.get("peak_allocated_bytes"),
            "peak_allocated_gib": gib((memory or {}).get("peak_allocated_bytes")),
            "peak_reserved_bytes": None if not memory else memory.get("peak_reserved_bytes"),
            "peak_reserved_gib": gib((memory or {}).get("peak_reserved_bytes")),
            "last_decoder_layer": tracker.get("last_decoder_layer"),
            "last_decoder_name": tracker.get("last_decoder_name"),
            "last_decoder_phase": tracker.get("last_decoder_phase"),
        }
        if extra:
            payload.update(extra)
        phases.append(payload)
        emit(name + "_end", **{k: payload[k] for k in (
            "seconds", "peak_allocated_gib", "last_decoder_layer") if k in payload})
        return payload

    try:
        hashes = [None] * world
        dist.all_gather_object(hashes, ddp.trainable_hash(model))
        if len(set(hashes)) != 1:
            raise RuntimeError("DDP adapter initialization differs across ranks")
        started, _ = begin("ddp_wrap")
        wrapped = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False, find_unused_parameters=False,
            gradient_as_bucket_view=True, init_sync=False)
        end("ddp_wrap", started)

        started, _ = begin("optimizer_construct")
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["compute_dtype"] == "float16")
        end("optimizer_construct", started, extra={"trainable_tensors": len(params)})

        collate = single.SelectedLabelCollator(tokenizer.pad_token_id, torch)
        indices = [0]
        local = indices[rank * cfg["micro_batch_size"]:(rank + 1) * cfg["micro_batch_size"]]
        count = sum(sum(x != -100 for x in rows[i]["labels"][1:]) for i in local)
        denominator = sum(sum(x != -100 for x in rows[i]["labels"][1:]) for i in indices)
        if not denominator:
            raise RuntimeError("capacity row has no supervised next-token labels")
        values = collate([rows[i] for i in local] if local else [rows[indices[0]]])
        batch = {k: v.to(device) for k, v in values.items()}
        dtype = getattr(torch, cfg["compute_dtype"])
        amp = torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()
        optimizer.zero_grad(set_to_none=True)

        started, _ = begin("forward")
        with amp:
            loss = wrapped(**batch, use_cache=False).loss
        if loss is None or not torch.isfinite(loss).all():
            raise RuntimeError("non-finite loss")
        end("forward", started, extra={"loss": float(loss.detach()), "local_selected_tokens": count,
                                       "global_selected_tokens": denominator})

        started, _ = begin("backward")
        scaler.scale(loss * (world * count / denominator)).backward()
        end("backward", started)

        started, _ = begin("optimizer")
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"], error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        state = adam_state_bytes(optimizer, torch)
        end("optimizer", started, extra={"grad_norm": float(norm), "adam_state": state})
        report["optimizer_reached"] = True
        report["adam_state"] = state
        report["loss"] = float(loss.detach())
        report["grad_norm"] = float(norm)
        return True
    finally:
        for handle in handles:
            handle.remove()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--arm", choices=("baseline", "save_on_cpu"), required=True)
    parser.add_argument("--save-on-cpu", type=int, choices=(0, 1), required=True)
    args = parser.parse_args(argv)
    save_on_cpu = bool(args.save_on_cpu)
    if (args.arm == "save_on_cpu") != save_on_cpu:
        raise SystemExit("arm and --save-on-cpu disagree")
    rank, local_rank, world = (int(os.environ.get(key, default)) for key, default in
                               (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    started = time.monotonic()
    bundle = args.bundle or Path(__file__).resolve().parent
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / ("rank-%03d-report.json" % rank)
    tracker = {"phase": "start", "last_decoder_layer": None, "last_decoder_name": None,
               "last_decoder_phase": None}
    report = {"schema": SCHEMA, "status": "started", "rank": rank, "local_rank": local_rank,
              "world_size": world, "arm": args.arm, "save_on_cpu": save_on_cpu,
              "transaction_id": TX, "sequence_length": SEQ, "synthetic": False,
              "did_not_truncate": True, "did_not_retry_16k": True,
              "loss_is_not_repair_gain": True, "adapter_discarded": True,
              "optimizer_reached": False, "oom": False, "phases": [], "events": []}

    def emit(stage, **values):
        report["stage"] = stage
        report["elapsed_seconds"] = time.monotonic() - started
        event = {"stage": stage, "elapsed_seconds": report["elapsed_seconds"],
                 "phase": tracker.get("phase"),
                 "last_decoder_layer": tracker.get("last_decoder_layer"), **values}
        report["events"].append(event)
        write_json(report_path, report)
        printable = {k: values[k] for k in values if k != "traceback"}
        print(json.dumps({"rank": rank, "arm": args.arm, "stage": stage, **printable},
                         allow_nan=False, default=str), flush=True)

    torch = dist = device = None
    try:
        emit("rank_start", python=sys.version, bundle=str(bundle))
        if world != WORLD:
            raise RuntimeError("This job requires exactly four torchrun ranks")
        cache = args.cache or find_cache(args.input_root)
        report["cache"] = verify_cache(cache)
        from training import train_sft as single
        from training.common import load_config
        from training.memory import install_selected_token_loss
        stack = single.require_stack("nf4")
        torch = stack["torch"]
        dist = torch.distributed
        report["versions"] = {name: getattr(module, "__version__", None) for name, module in stack.items()}
        report["has_save_on_cpu"] = hasattr(torch.autograd.graph, "save_on_cpu")
        if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD:
            raise RuntimeError("Exactly four visible CUDA devices are required")
        if any("L4" not in torch.cuda.get_device_name(index) for index in range(WORLD)):
            raise RuntimeError("Expected the requested four-L4 accelerator allocation")
        torch.set_num_threads(1)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
        random.seed(20261010)
        torch.manual_seed(20261010)
        torch.cuda.manual_seed_all(20261010)

        yaml_path = bundle / "training" / "configs" / "sft_qat_nf4_ddp.yaml"
        tools_path = bundle / "training" / "configs" / "harness_tools.json"
        template_path = bundle / "training" / "configs" / "chat_template.jinja"
        data_path = bundle / "data" / "encoded-capacity-8192.jsonl"
        raw = load_config(yaml_path)
        raw.update({
            "model_name_or_path": str(cache), "tokenizer_name_or_path": str(cache),
            "tool_schema_path": str(tools_path), "chat_template_path": str(template_path),
            "device": str(device), "max_sequence_length": 8192, "max_steps": 1,
            "gradient_accumulation_steps": 1, "warmup_ratio": 0.0,
            "checkpoint_every_n_steps": 1, "epochs": 1,
        })
        cfg = single.validate_config(raw)
        if cfg["rank"] != 8 or cfg["alpha"] != 16 or cfg["loss_chunk_size"] != 32:
            raise RuntimeError("recipe drifted from rank8/alpha16/chunk32")
        if cfg["kbit_prepare"] != "norms_fp32" or cfg["loss_backend"] != "selected_chunked":
            raise RuntimeError("recipe must keep norms_fp32 and selected_chunked")
        if cfg["attn_implementation"] != "sdpa" or cfg["quant_backend"] != "nf4":
            raise RuntimeError("recipe must keep NF4 and SDPA")
        rows, data = load_rows(data_path)
        report["data"] = data
        report["config"] = {k: cfg[k] for k in (
            "quant_backend", "compute_dtype", "kbit_prepare", "loss_backend",
            "loss_chunk_size", "rank", "alpha", "attn_implementation",
            "gradient_checkpointing", "max_sequence_length", "micro_batch_size",
            "gradient_accumulation_steps", "learning_rate")}
        emit("model_load_begin", device=str(device), gpu=torch.cuda.get_device_name(device))
        load_started = time.monotonic()
        model, tokenizer, device, info = single.load_model_and_tokenizer(
            cfg, stack, expected_provenance=data["provenance"])
        install_selected_token_loss(model, chunk_size=cfg["loss_chunk_size"], save_on_cpu=save_on_cpu)
        torch.cuda.synchronize(device)
        report["load_seconds"] = time.monotonic() - load_started
        report["model"] = info
        report["load_memory"] = cuda_memory(torch, device)
        report["save_on_cpu_installed"] = bool(model.get_base_model()._p005_save_on_cpu)
        if report["save_on_cpu_installed"] != save_on_cpu:
            raise RuntimeError("save_on_cpu flag did not stick on the Gemma4 base")
        emit("model_load_ok", load_seconds=report["load_seconds"],
             memory=report["load_memory"], save_on_cpu_installed=report["save_on_cpu_installed"])
        run_one_step(model, rows, cfg, torch, dist, device, tokenizer, tracker, emit, report)
        report["status"] = "completed"
        report["backprop_ok"] = True
        emit("completed", memory=cuda_memory(torch, device),
             optimizer_reached=True, last_decoder_layer=tracker.get("last_decoder_layer"))
        if dist.is_initialized():
            dist.destroy_process_group()
        return 0
    except Exception as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        report["failure_phase"] = tracker.get("phase")
        report["last_decoder_layer"] = tracker.get("last_decoder_layer")
        report["last_decoder_name"] = tracker.get("last_decoder_name")
        report["last_decoder_phase"] = tracker.get("last_decoder_phase")
        if torch is not None and device is not None:
            report["failure_memory"] = cuda_memory(torch, device)
        if is_oom(exc):
            report["status"] = "oom"
            report["oom"] = True
            report["backprop_ok"] = False
            report["optimizer_reached"] = bool(report.get("optimizer_reached"))
            emit("oom", error=repr(exc), failure_phase=tracker.get("phase"),
                 last_decoder_layer=tracker.get("last_decoder_layer"),
                 traceback=report["traceback"])
            os._exit(0)
        report["status"] = "failed"
        emit("failed", error=repr(exc), traceback=report["traceback"])
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
