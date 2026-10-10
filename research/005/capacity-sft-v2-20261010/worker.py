"""Four-L4 capacity on complete transactions, then one rank8 SFT if 16K backprop works.

No downloads, installs, truncation, or diagnostic synthetic 2K. Temporary capacity
updates are reset before the SFT candidate. Loss is not repair gain.
"""
import argparse
from datetime import timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

SCHEMA = "p005-capacity-sft-v2"
CACHE_KERNEL = "ccjjllt/p005-cpu-nf4-cache"
SOURCE = "google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"
WORLD = 4
RESERVED = {
    "fastapi_13713", "requests_6592", "requests_6644", "requests_7309",
    "requests_7505", "rich_3130", "rich_3296", "rich_3469",
}
STEP_CAP = 100


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
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "free_bytes": free, "total_bytes": total}


def is_oom(exc):
    text = (repr(exc) + " " + str(exc)).lower()
    return "out of memory" in text or "cuda oom" in text


def summarize_rows(rows):
    items = []
    for row in rows:
        meta = row["meta"]
        group = meta.get("group") or {}
        items.append({
            "transaction_id": meta.get("transaction_id"),
            "issue": group.get("issue"),
            "split": group.get("split"),
            "behavior": meta.get("behavior"),
            "sequence_length": len(row["input_ids"]),
            "supervised_tokens_after_shift": sum(label != -100 for label in row["labels"][1:]),
            "synthetic": meta.get("synthetic"),
            "training_eligible": meta.get("training_eligible"),
            "truncated": bool(meta.get("truncated")),
        })
    return items


def load_rows(path, max_length, require_real):
    from training.common import load_encoded_jsonl, TrainingError
    rows, data = load_encoded_jsonl(path, max_length, require_real=require_real)
    for row in rows:
        meta = row["meta"]
        if meta.get("synthetic") is not False:
            raise TrainingError("refusing synthetic encoded rows")
        if meta.get("truncated") or meta.get("truncated_tool_block"):
            raise TrainingError("refusing truncated encoded rows")
        issue = (meta.get("group") or {}).get("issue")
        if issue in RESERVED:
            raise TrainingError("reserved pairing issue leaked: " + str(issue))
    return rows, data


def reset_lora(model):
    reset = 0
    for module in model.modules():
        fn = getattr(module, "reset_lora_parameters", None)
        if callable(fn) and getattr(module, "lora_A", None) is not None:
            fn("default", True)
            reset += 1
    if reset == 0:
        raise RuntimeError("LoRA reset found no adapters; refusing to keep capacity weights")
    return reset


def clone_cfg(base, **overrides):
    from training import train_sft as single
    cfg = dict(base)
    cfg.update(overrides)
    return single.validate_config(cfg)


def run_phase(name, model, rows, cfg, output, torch, dist, device, tokenizer, data, emit):
    from training import train_ddp as ddp
    from training import train_sft as single
    rank = dist.get_rank()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    rows = list(rows)
    if name == "sft":
        rows.sort(key=lambda row: len(row["input_ids"]))
    signature = single._signature(dict(cfg, device="cuda:rank-local"), data)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    emit(name + "_begin", n_rows=len(rows), max_sequence_length=cfg["max_sequence_length"],
         row_lengths=[len(r["input_ids"]) for r in rows], memory=cuda_memory(torch, device))
    started = time.monotonic()
    result = ddp.run_ddp_loop(model, rows, cfg, output, torch, tokenizer.pad_token_id, signature)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    memory = cuda_memory(torch, device)
    payload = {
        "phase": name, "backprop_ok": True, "seconds": time.monotonic() - started,
        "result": result, "memory": memory, "rows": summarize_rows(rows),
        "peak_allocated_bytes": memory["peak_allocated_bytes"],
        "selected_tokens_per_second": (
            result["global_selected_tokens_this_run"] / result["wall_seconds_this_run"]
            if result.get("wall_seconds_this_run") else None),
        "loss_is_not_repair_gain": True,
        "signature": signature,
    }
    emit(name + "_ok", **{k: payload[k] for k in (
        "seconds", "peak_allocated_bytes", "selected_tokens_per_second")})
    if rank == 0:
        write_json(output / "phase-report.json", payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/capacity-sft-v2"))
    parser.add_argument("--bundle", type=Path, default=None)
    args = parser.parse_args(argv)
    rank, local_rank, world = (int(os.environ.get(key, default)) for key, default in
                               (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    started = time.monotonic()
    bundle = args.bundle or Path(__file__).resolve().parent
    report_path = args.output_root / ("rank-%03d-report.json" % rank)
    report = {"schema": SCHEMA, "status": "started", "rank": rank, "local_rank": local_rank,
              "world_size": world, "synthetic": False, "training_eligible": True,
              "gpu_sft": False, "sft_skip_reason": None, "capacity": [], "events": [],
              "loss_is_not_repair_gain": True, "did_not_truncate": True,
              "did_not_use_diagnostic_synthetic_2k": True}

    def emit(stage, **values):
        report["stage"] = stage
        report["elapsed_seconds"] = time.monotonic() - started
        event = {"stage": stage, "elapsed_seconds": report["elapsed_seconds"], **values}
        report["events"].append(event)
        write_json(report_path, report)
        print(json.dumps({"rank": rank, **event}, allow_nan=False, default=str), flush=True)

    torch = dist = device = None
    try:
        emit("rank_start", python=sys.version, bundle=str(bundle))
        if world != WORLD:
            raise RuntimeError("This job requires exactly four torchrun ranks")
        cache = args.cache or find_cache(args.input_root)
        report["cache"] = verify_cache(cache)
        from training import train_sft as single
        from training.common import load_config
        from training.export_peft import inspect_adapter
        stack = single.require_stack("nf4")
        torch = stack["torch"]
        dist = torch.distributed
        if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD:
            raise RuntimeError("Exactly four visible CUDA devices are required")
        if any("L4" not in torch.cuda.get_device_name(index) for index in range(WORLD)):
            raise RuntimeError("Expected the requested four-L4 accelerator allocation")
        torch.set_num_threads(1)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", timeout=timedelta(seconds=300))
        random.seed(20261010)
        torch.manual_seed(20261010)
        torch.cuda.manual_seed_all(20261010)

        train_path = bundle / "data" / "encoded-train-16384.jsonl"
        cap8_path = bundle / "data" / "encoded-capacity-8192.jsonl"
        yaml_path = bundle / "training" / "configs" / "sft_qat_nf4_ddp_16384.yaml"
        tools_path = bundle / "training" / "configs" / "harness_tools.json"
        template_path = bundle / "training" / "configs" / "chat_template.jinja"
        raw = load_config(yaml_path)
        raw.update({
            "model_name_or_path": str(cache),
            "tokenizer_name_or_path": str(cache),
            "tool_schema_path": str(tools_path),
            "chat_template_path": str(template_path),
            "device": str(device),
        })
        cfg = single.validate_config(raw)
        if cfg["max_sequence_length"] != 16384 or cfg["rank"] != 8 or cfg["alpha"] != 16:
            raise RuntimeError("SFT recipe drifted from rank8/alpha16/16384")
        if cfg["kbit_prepare"] != "norms_fp32" or cfg["loss_backend"] != "selected_chunked":
            raise RuntimeError("SFT recipe must keep norms_fp32 and selected_chunked")
        train_rows, train_data = load_rows(train_path, cfg["max_sequence_length"], True)
        if any((row["meta"].get("group") or {}).get("split") != "train" for row in train_rows):
            raise RuntimeError("SFT JSONL contains a non-train split")
        cap8_rows, cap8_data = load_rows(cap8_path, cfg["max_sequence_length"], False)
        if len(cap8_rows) != 1 or (cap8_rows[0]["meta"].get("group") or {}).get("split") != "select":
            raise RuntimeError("8K capacity JSONL must be the one complete select transaction")
        longest = max(train_rows, key=lambda row: len(row["input_ids"]))
        report["data"] = {
            "train": {**train_data, "rows_detail": summarize_rows(train_rows)},
            "capacity_8192": {**cap8_data, "rows_detail": summarize_rows(cap8_rows)},
            "capacity_16384_probe": summarize_rows([longest]),
        }
        report["config"] = cfg
        emit("model_load_begin", device=str(device), gpu=torch.cuda.get_device_name(device))
        load_started = time.monotonic()
        model, tokenizer, device, info = single.load_model_and_tokenizer(
            cfg, stack, expected_provenance=train_data["provenance"])
        torch.cuda.synchronize(device)
        report["load_seconds"] = time.monotonic() - load_started
        report["model"] = info
        report["load_memory"] = cuda_memory(torch, device)
        emit("model_load_ok", load_seconds=report["load_seconds"], memory=report["load_memory"])

        def record_capacity(window, payload, oom=None):
            entry = {"window": window, "backprop_ok": oom is None, "oom": None if oom is None else repr(oom)}
            if payload:
                entry.update({k: payload.get(k) for k in (
                    "seconds", "peak_allocated_bytes", "selected_tokens_per_second",
                    "rows", "result", "loss_is_not_repair_gain")})
            report["capacity"].append(entry)
            write_json(report_path, report)
            return entry

        # 8K: one complete transaction (select). Weights are discarded after the probe.
        cap8_cfg = clone_cfg(cfg, max_steps=1, epochs=1, gradient_accumulation_steps=1,
                             checkpoint_every_n_steps=1, warmup_ratio=0.0)
        try:
            payload8 = run_phase("capacity_8192", model, cap8_rows, cap8_cfg,
                                 args.output_root / "capacity-8192", torch, dist, device,
                                 tokenizer, cap8_data, emit)
            record_capacity(8192, payload8)
        except Exception as exc:
            report["failure_memory"] = cuda_memory(torch, device) if device is not None else None
            entry = record_capacity(8192, None, exc)
            entry["traceback"] = traceback.format_exc()
            if not is_oom(exc):
                raise
            report["gpu_sft"] = False
            report["sft_skip_reason"] = "8K complete-transaction backprop OOM; not retrying 16K, not truncating"
            report["activation_vs_weight"] = {
                "load_allocated_bytes": (report.get("load_memory") or {}).get("allocated_bytes"),
                "failure_peak_allocated_bytes": (report.get("failure_memory") or {}).get("peak_allocated_bytes"),
                "note": "Each rank loaded the NF4 base, then exceeded available memory during the capacity step. The exact failing operation is not classified; use the traceback before attributing it to activations or optimizer state. This job does not auto-retry.",
            }
            report["status"] = "capacity_oom"
            emit("capacity_oom", window=8192, error=repr(exc))
            return 0

        reset_lora(model)
        gc.collect()
        torch.cuda.empty_cache()

        # 16K: longest complete train transaction that fits 16384.
        cap16_cfg = clone_cfg(cfg, max_steps=1, epochs=1, gradient_accumulation_steps=1,
                              checkpoint_every_n_steps=1, warmup_ratio=0.0)
        cap16_data = dict(train_data)
        cap16_bytes = json.dumps([longest["input_ids"], longest["labels"]], separators=(",", ":")).encode()
        cap16_data = dict(train_data, rows=1, sha256=hashlib.sha256(cap16_bytes).hexdigest())
        try:
            payload16 = run_phase("capacity_16384", model, [longest], cap16_cfg,
                                  args.output_root / "capacity-16384", torch, dist, device,
                                  tokenizer, cap16_data, emit)
            record_capacity(16384, payload16)
        except Exception as exc:
            report["failure_memory"] = cuda_memory(torch, device) if device is not None else None
            entry = record_capacity(16384, None, exc)
            entry["traceback"] = traceback.format_exc()
            if not is_oom(exc):
                raise
            report["gpu_sft"] = False
            report["sft_skip_reason"] = "16K complete-transaction backprop OOM; no truncation, no synthetic 2K"
            report["activation_vs_weight"] = {
                "load_allocated_bytes": (report.get("load_memory") or {}).get("allocated_bytes"),
                "failure_peak_allocated_bytes": (report.get("failure_memory") or {}).get("peak_allocated_bytes"),
                "note": "8K capacity succeeded; the 16K capacity step exceeded available memory. The exact failing operation is not classified; use the traceback before attributing it to activations or optimizer state. Train has 0 complete 8K rows, so formal SFT stops without an automatic retry.",
            }
            report["status"] = "capacity_oom"
            emit("capacity_oom", window=16384, error=repr(exc))
            return 0

        if len(train_rows) == 0:
            report["sft_skip_reason"] = "no complete train transactions fit 16384"
            report["status"] = "capacity_only"
            emit("sft_skipped", reason=report["sft_skip_reason"])
            return 0

        reset_lora(model)
        gc.collect()
        torch.cuda.empty_cache()
        n_steps = 1  # 4 rows, world 4, microbatch 1, accum 1, one epoch
        sft_cfg = clone_cfg(cfg, max_steps=min(STEP_CAP, max(n_steps, 1)), epochs=1,
                            gradient_accumulation_steps=1, warmup_ratio=0.0)
        sft_payload = run_phase("sft", model, train_rows, sft_cfg,
                                args.output_root / "sft", torch, dist, device,
                                tokenizer, train_data, emit)
        report["sft"] = sft_payload
        if rank == 0:
            adapter = args.output_root / "sft" / "adapter"
            inspection = inspect_adapter(adapter, sft_cfg["rank"], sft_cfg["target_text_modules"])
            report["adapter"] = inspection
            from training.common import atomic_json
            atomic_json(args.output_root / "sft" / "training_manifest.json", {
                "status": "training_completed", "mode": "ddp", "gpu_sft": True,
                "config": sft_cfg, "data": train_data, "model": info,
                "metrics": sft_payload["result"], "adapter": inspection,
                "capacity": report["capacity"],
                "loss_is_not_repair_gain": True,
                "runtime_compatibility_verified": False,
                "independent_issues": sorted({(r["meta"].get("group") or {}).get("issue") for r in train_rows}),
                "note": "Method exploration on four complete 16384-fitting train rows. Not a medal claim.",
            })
        dist.barrier()
        report["gpu_sft"] = True
        report["status"] = "completed"
        emit("completed", gpu_sft=True, memory=cuda_memory(torch, device))
        dist.destroy_process_group()
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        if is_oom(exc):
            report["gpu_sft"] = False
            report["sft_skip_reason"] = report.get("sft_skip_reason") or "OOM during load or training"
        if torch is not None and device is not None:
            try:
                report["failure_memory"] = cuda_memory(torch, device)
            except Exception as memory_error:
                report["memory_read_error"] = repr(memory_error)
        emit("failed", error=repr(exc))
        # Only the handled capacity OOM branches above are successful probe outcomes.
        # Load/SFT failures must remain nonzero even after prior capacity checks passed.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
