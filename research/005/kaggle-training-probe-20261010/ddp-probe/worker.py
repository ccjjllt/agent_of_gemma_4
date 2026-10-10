"""Bounded four-L4 training diagnostic, launched only by an external wrapper.

No downloads, installs, retries, offload, or real training-data eligibility
claims. The cached source is untrained QAT-unquantized-v2 -> CPU NF4. Training
uses the repository's real loader and DDP loop. Only AdamW.step is temporarily
wrapped to observe gradients/updates; its original implementation still runs.
"""
import argparse
from contextlib import contextmanager
from datetime import timedelta
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

SCHEMA = "p005-ddp-probe-v1"
CACHE_KERNEL = "ccjjllt/p005-cpu-nf4-cache"
SOURCE = "google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized/2"
LENGTHS = (256, 256, 1024, 2048)
WORLD = 4
SELECTED = 64


def write_json(path, value):
    if not isinstance(value, dict):
        raise TypeError("Probe report must be a JSON object")
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
    return {"kernel": CACHE_KERNEL, "path": str(cache), "source": SOURCE,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "cache_total_bytes": manifest.get("total_bytes"),
            "cpu_versions": manifest.get("versions"),
            "cpu_cache_verified_complete": True}


def diagnostic_rows(tokenizer, torch, seed=0, world=WORLD, lengths=LENGTHS, selected=SELECTED):
    """Invert the trainer's deterministic shuffle to schedule length stages.

    These deliberately bypass encoded-data ingestion, and are marked synthetic
    / ineligible. They test tensor capacity and gradients, not coding quality.
    """
    intended = []
    for step, length in enumerate(lengths, 1):
        if length <= selected:
            raise ValueError("The diagnostic must include unlabelled prefix context")
        for rank in range(world):
            text = f"def diagnostic_{step}_{rank}(value):\n    return value + {rank + 1}\n\n"
            unit = list(tokenizer.encode(text, add_special_tokens=False))
            if not unit:
                raise ValueError("Tokenizer returned no diagnostic text tokens")
            ids = (unit * math.ceil(length / len(unit)))[:length]
            intended.append({"input_ids": ids, "attention_mask": [1] * length,
                             "labels": [-100] * (length - selected) + ids[-selected:],
                             "meta": {"synthetic": True, "training_eligible": False,
                                      "purpose": "memory_and_gradient_diagnostic_only",
                                      "planned_step": step, "planned_rank": rank}})
    order = torch.randperm(len(intended), generator=torch.Generator().manual_seed(seed)).tolist()
    rows = [None] * len(intended)
    for slot, shuffled_index in enumerate(order):
        rows[shuffled_index] = intended[slot]
    return rows


def cuda_memory(torch, device):
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "free_bytes": free, "total_bytes": total}


@contextmanager
def observe_adamw(torch, model, dist, device, initial_hash, hash_model, emit, step_records):
    """Observe the unchanged optimizer path and synchronize diagnostic checks."""
    original = torch.optim.AdamW.step
    previous_hash = initial_hash
    rank, world = dist.get_rank(), dist.get_world_size()

    @functools.wraps(original)
    def step(optimizer, *args, **kwargs):
        nonlocal previous_hash
        step_no = len(step_records) + 1
        if step_no > len(LENGTHS):
            raise RuntimeError("Unexpected optimizer step beyond the fixed probe plan")
        diagnostic_start = time.monotonic()
        params = [p for group in optimizer.param_groups for p in group["params"]]
        norms = [p.grad.detach().float().norm() for p in params if p.grad is not None]
        stacked = torch.stack(norms) if norms else torch.zeros(1, device=device)
        finite = bool(torch.isfinite(stacked).all().item())
        grad_norm = float(stacked.norm().item())
        gradient = {"rank": rank, "finite": finite,
                    "gradient_norm_after_clipping": grad_norm if math.isfinite(grad_norm) else None,
                    "nonzero_gradient_tensors": int((stacked > 0).sum().item()),
                    "gradient_tensors": len(norms), "trainable_tensors": len(params)}
        gradient_seconds = time.monotonic() - diagnostic_start
        gathering = time.monotonic()
        gradients = [None] * world
        dist.all_gather_object(gradients, gradient)
        pre_gather_seconds = time.monotonic() - gathering
        if any(not value["finite"] or value["nonzero_gradient_tensors"] == 0
               or value["gradient_tensors"] != value["trainable_tensors"] for value in gradients):
            emit("gradient_verification_failed", all_rank_gradients=gradients, step=step_no)
            raise RuntimeError("At least one rank has non-finite, zero, or missing adapter gradients")
        before_update_diagnostics = time.monotonic() - diagnostic_start
        update_started = time.monotonic()
        result = original(optimizer, *args, **kwargs)
        torch.cuda.synchronize(device)
        optimizer_seconds = time.monotonic() - update_started
        after_update_diagnostics = time.monotonic()
        hashing = time.monotonic()
        current_hash = hash_model(model)
        hash_seconds = time.monotonic() - hashing
        local = {"rank": rank, "trainable_sha256": current_hash,
                 "changed_from_previous_step": current_hash != previous_hash}
        gathering = time.monotonic()
        updates = [None] * world
        dist.all_gather_object(updates, local)
        post_gather_seconds = time.monotonic() - gathering
        consistent = len({value["trainable_sha256"] for value in updates}) == 1
        changed = all(value["changed_from_previous_step"] for value in updates)
        timings = {"gradient_check_seconds": gradient_seconds,
                   "pre_step_all_gather_seconds": pre_gather_seconds,
                   "adapter_hash_seconds": hash_seconds,
                   "post_step_all_gather_seconds": post_gather_seconds,
                   "total_seconds": before_update_diagnostics + time.monotonic() - after_update_diagnostics}
        record = {"step": step_no, "sequence_length": LENGTHS[step_no - 1],
                  "selected_tokens_per_rank": SELECTED, "all_rank_gradients": gradients,
                  "all_rank_updates": updates, "all_rank_hashes_equal": consistent,
                  "all_rank_adapters_changed": changed, "diagnostic_timing": timings,
                  "optimizer_step_seconds": optimizer_seconds, "memory": cuda_memory(torch, device)}
        step_records.append(record)
        emit("optimizer_step_verified" if consistent and changed else "optimizer_step_verification_failed", **record)
        if not consistent or not changed:
            raise RuntimeError("Adapter updates differ across ranks or did not change")
        previous_hash = current_hash
        return result

    torch.optim.AdamW.step = step
    try:
        yield
    finally:
        torch.optim.AdamW.step = original


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/ddp-probe"))
    args = parser.parse_args(argv)
    rank, local_rank, world = (int(os.environ.get(key, default)) for key, default in
                               (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    started = time.monotonic()
    report_path = args.output_root / f"rank-{rank:03d}-report.json"
    source_path = Path(__file__).parent / "bundle-manifest.json"
    report = {"schema": SCHEMA, "status": "started", "rank": rank, "local_rank": local_rank,
              "world_size": world, "synthetic": True, "training_eligible": False,
              "purpose": "capacity_and_gradient_diagnostic_not_quality_evaluation",
              "cache_kernel": CACHE_KERNEL, "length_plan": list(LENGTHS),
              "selected_tokens_per_rank_per_step": SELECTED,
              "source_manifest": json.loads(source_path.read_text()) if source_path.is_file() else None,
              "events": [], "steps": []}

    def emit(stage, **values):
        report["stage"] = stage
        report["elapsed_seconds"] = time.monotonic() - started
        event = {"stage": stage, "elapsed_seconds": report["elapsed_seconds"], **values}
        report["events"].append(event)
        write_json(report_path, report)
        print(json.dumps({"rank": rank, **event}, allow_nan=False), flush=True)

    torch = dist = device = None
    load_started = None
    training_started = None
    try:
        emit("rank_start", python=sys.version)
        if world != WORLD:
            raise RuntimeError("This diagnostic requires exactly four torchrun ranks")
        cache = args.cache or find_cache(args.input_root)
        report["cache"] = verify_cache(cache)
        from training import train_sft as single
        from training import train_ddp as ddp
        stack = single.require_stack("nf4")
        torch = stack["torch"]
        dist = torch.distributed
        if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD:
            raise RuntimeError("Exactly four visible CUDA devices are required")
        if any("L4" not in torch.cuda.get_device_name(index).split() for index in range(WORLD)):
            raise RuntimeError("Expected the requested four-L4 accelerator allocation")
        torch.set_num_threads(1)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group("nccl", timeout=timedelta(seconds=180))
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        cfg = single.validate_config({"model_name_or_path": str(cache), "tokenizer_name_or_path": str(cache),
            "source_version_note": SOURCE + " -> CPU NF4 cache", "allow_cached_nf4": True,
            "quant_backend": "nf4", "compute_dtype": "bfloat16", "kbit_prepare": "norms_fp32",
            "loss_backend": "selected_chunked", "loss_chunk_size": 32, "rank": 8, "alpha": 16,
            "target_text_modules": ["q_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "gradient_checkpointing": True, "local_files_only": True, "max_sequence_length": max(LENGTHS),
            "micro_batch_size": 1, "gradient_accumulation_steps": 1, "epochs": 1,
            "max_steps": len(LENGTHS), "warmup_ratio": 0.0, "checkpoint_every_n_steps": 1,
            "learning_rate": 5e-5, "dropout": 0.0, "seed": 0, "device": str(device)})
        report["versions"] = {name: getattr(module, "__version__", None) for name, module in stack.items()}
        report["config"] = cfg
        emit("model_load_begin", device=str(device), gpu=torch.cuda.get_device_name(device))
        load_started = time.monotonic()
        model, tokenizer, device, info = single.load_model_and_tokenizer(cfg, stack)
        torch.cuda.synchronize(device)
        report["load_seconds"] = time.monotonic() - load_started
        report["model"] = info
        report["load_memory"] = cuda_memory(torch, device)
        quantized = [p for p in model.parameters()
                     if any(cls.__name__ == "Params4bit" for cls in type(p).__mro__)]
        states = [getattr(parameter, "quant_state", None) for parameter in quantized]
        cache_format = {
            "params4bit_tensors": len(quantized),
            "all_packed_and_on_rank_cuda": bool(quantized) and all(
                getattr(p, "bnb_quantized", False) and p.device == device for p in quantized),
            "all_nf4_with_nested_state": bool(states) and all(
                state is not None and getattr(state, "quant_type", None) == "nf4"
                and getattr(state, "state2", None) is not None for state in states),
            "packed_storage_dtypes": sorted({str(p.dtype) for p in quantized}),
            "quant_state_original_dtypes": sorted({str(getattr(state, "dtype", None)) for state in states}),
            "linear4bit_compute_dtypes": sorted({str(getattr(module, "compute_dtype", None))
                for module in model.modules()
                if any(cls.__name__ == "Linear4bit" for cls in type(module).__mro__)}),
            "embedding_dtype": str(model.get_input_embeddings().weight.dtype),
            "validation": "Actual CUDA Params4bit state inspected after cached load; packed bytes not rehashed",
        }
        report["cpu_to_cuda_cache_format"] = cache_format
        if not cache_format["all_packed_and_on_rank_cuda"] or not cache_format["all_nf4_with_nested_state"]:
            raise RuntimeError("CPU NF4 cache did not load as packed, nested NF4 on the rank CUDA device")
        del quantized, states
        emit("model_load_ok", load_seconds=report["load_seconds"], memory=report["load_memory"])
        rows = diagnostic_rows(tokenizer, torch, cfg["seed"])
        row_bytes = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        data = {"sha256": hashlib.sha256(row_bytes).hexdigest(),
                "provenance": {"synthetic": True, "training_eligible": False,
                               "origin": "probe_generated_code_tokens", "lengths": list(LENGTHS)}}
        report["diagnostic_data"] = dict(data, samples=len(rows), generation="inverse-fixed-shuffle")
        train_output = args.output_root / "training"
        if rank == 0:
            train_output.mkdir(parents=True, exist_ok=False)
            write_json(args.output_root / "diagnostic-plan.json", {"schema": SCHEMA, "config": cfg, "data": data})
        dist.barrier()
        hash_started = time.monotonic()
        initial_hash = ddp.trainable_hash(model)
        report["initial_adapter_hash_seconds"] = time.monotonic() - hash_started
        report["initial_trainable_sha256"] = initial_hash
        # Rank-independent source signature. Physical CUDA index is not a data
        # or optimization difference and is excluded from this probe signature.
        signature = single._signature(dict(cfg, device="cuda:rank-local"), data)
        torch.cuda.reset_peak_memory_stats(device)
        emit("training_begin", initial_trainable_sha256=initial_hash)
        training_started = time.monotonic()
        forward_clock = {}

        def forward_begin(module, positional, named):
            ids = named.get("input_ids", positional[0] if positional else None)
            emit("forward_begin", step=len(report["steps"]) + 1,
                 sequence_length=int(ids.shape[1]), memory=cuda_memory(torch, device))
            forward_clock["started"] = time.monotonic()

        def forward_complete(module, positional, named, output):
            loss = float(output.loss.detach())
            emit("forward_complete", step=len(report["steps"]) + 1,
                 loss=loss if math.isfinite(loss) else None, finite_loss=math.isfinite(loss),
                 forward_seconds=time.monotonic() - forward_clock["started"])

        handles = [model.register_forward_pre_hook(forward_begin, with_kwargs=True),
                   model.register_forward_hook(forward_complete, with_kwargs=True)]
        try:
            with observe_adamw(torch, model, dist, device, initial_hash, ddp.trainable_hash, emit, report["steps"]):
                result = ddp.run_ddp_loop(model, rows, cfg, train_output, torch, tokenizer.pad_token_id, signature)
        finally:
            for handle in handles:
                handle.remove()
        report["training_seconds_including_diagnostics_and_checkpoint_io"] = time.monotonic() - training_started
        report["training_result"] = result
        report["training_memory"] = cuda_memory(torch, device)
        if len(report["steps"]) != len(LENGTHS):
            raise RuntimeError("The fixed probe did not finish all four optimizer steps")
        report["status"] = "completed"
        emit("completed", memory=report["training_memory"], optimizer_steps=len(report["steps"]))
        dist.destroy_process_group()
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        if load_started is not None and "load_seconds" not in report:
            report["load_seconds_to_failure"] = time.monotonic() - load_started
        if training_started is not None:
            report["training_seconds_including_diagnostics_and_checkpoint_io"] = time.monotonic() - training_started
        if torch is not None and device is not None:
            try:
                report["failure_memory"] = cuda_memory(torch, device)
            except Exception as memory_error:
                report["memory_read_error"] = repr(memory_error)
        emit("failed", error=repr(exc), successful_optimizer_steps=len(report["steps"]))
        # No cleanup collective after a rank failure: torchrun/the outer hard
        # timeout terminates peers, avoiding a destroy-process-group deadlock.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
