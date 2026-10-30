"""Replay complete recorded model calls to measure same-policy GPU batching.

Run on the authorized Linux GPU server. Prompts/completions stay in memory on
that server; only numerical summaries, source fingerprints and optional profiler
traces are written. This teacher-forced replay does not execute a task, perform
an update, or measure repair success. Historical behavior logprobs are ignored.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import tomllib

from .generation import GenerationEngine


PROJECT = Path(__file__).resolve().parents[1]
_RESERVED = re.compile(r"(^|[/_.-])(heldout|holdout|reserved|secret|hidden)($|[/_.-])", re.I)


def _write(path, value):
    temporary = path.with_name(".writing-" + path.name)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_calls(paths, maximum, context):
    """Read explicit recorded calls, never a task dataset or reward asset."""
    records, sources = [], []
    resolved = [path.expanduser().resolve(strict=True) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Each --calls source must be distinct")
    for path in resolved:
        if path.name != "model_calls.jsonl" or _RESERVED.search(str(path)):
            raise ValueError("--calls must name an explicit public-training run's model_calls.jsonl")
        if not path.is_file():
            raise ValueError("Model-call source is not a regular file")
        # Hash and parse the same immutable in-memory snapshot, even when the
        # source is an append-only log. No source snapshot is written locally.
        data = path.read_bytes()
        source_index = len(sources)
        selected = []
        incomplete_tail = False
        lines = data.splitlines(keepends=True)
        for line_number, line in enumerate(lines, 1):
            if len(records) >= maximum:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if line_number == len(lines) and not line.endswith(b"\n"):
                    incomplete_tail = True
                    break
                raise ValueError(f"Malformed model-call record at source {source_index}, line {line_number}") from None
            if not isinstance(row, dict):
                raise ValueError("Every model-call record must be an object")
            for name in ("prompt_ids", "completion_ids"):
                values = row.get(name)
                if not isinstance(values, list) or not values or any(type(value) is not int or value < 0 for value in values):
                    raise ValueError(f"Source {source_index}, line {line_number}: invalid {name}")
            if len(row["prompt_ids"]) + len(row["completion_ids"]) > context:
                raise ValueError("A recorded complete call exceeds this policy's context; it will not be truncated")
            identity = f"call-{len(records):04d}"
            records.append({"id": identity, "prompt_ids": row["prompt_ids"],
                            "completion_ids": row["completion_ids"],
                            "source_index": source_index, "source_line": line_number})
            selected.append({"id": identity, "line": line_number,
                             "prompt_tokens": len(row["prompt_ids"]),
                             "completion_tokens": len(row["completion_ids"])})
        sources.append({"path": str(path), "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "selected_calls": selected, "incomplete_tail_ignored": incomplete_tail})
        del data, lines
    if not records:
        raise ValueError("No complete model-call records were found")
    return records, sources


def _synchronize(torch):
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)


def _old_generation_config(policy, completion_length):
    """Mirror Policy.generate's categorical recipe, without changing Policy."""
    generation = copy.deepcopy(policy.generation_config)
    settings = {
        "do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "typical_p": 1.0, "min_p": None, "epsilon_cutoff": 0.0, "eta_cutoff": 0.0,
        "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "min_length": 0,
        "min_new_tokens": None, "forced_bos_token_id": None, "forced_eos_token_id": None,
        "suppress_tokens": None, "begin_suppress_tokens": None, "bad_words_ids": None,
        "sequence_bias": None, "num_beams": 1, "num_return_sequences": 1,
        "max_new_tokens": completion_length, "return_dict_in_generate": False,
        "output_scores": False, "output_logits": False, "use_cache": True,
    }
    for name, value in settings.items():
        setattr(generation, name, value)
    return generation


def _old_serial_replay(policy, records, destination):
    """Run the original HF/Unsloth generate loop with recorded continuations.

    The processor observes untouched categorical scores, then forces the next
    recorded token. generate still invokes its usual multinomial once per step,
    but on the forced distribution. Consequently sampling timings are not an
    exact measure of free-running categorical generation.
    """
    from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    torch = policy.torch
    policy._for_inference()
    _synchronize(torch)
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)
    probabilities, calls = {}, []
    all_started = time.perf_counter()
    for record in records:
        prompt, completion = record["prompt_ids"], record["completion_ids"]
        started = time.perf_counter()

        class ObserveAndForce(LogitsProcessor):
            def __init__(self):
                self.values = []
                self.first_logits_seconds = None

            def __call__(self, ids, scores):
                index = ids.shape[-1] - len(prompt)
                if ids.shape[0] != 1 or index != len(self.values) or index >= len(completion):
                    raise RuntimeError("Old serial replay requires one ordinary generation step per recorded token")
                if self.first_logits_seconds is None:
                    _synchronize(torch)
                    self.first_logits_seconds = time.perf_counter() - started
                distribution = torch.log_softmax(scores.float(), dim=-1)
                self.values.append(distribution[0, completion[index]].detach())
                forced = torch.full_like(scores, -float("inf"))
                forced[0, completion[index]] = 0.0
                return forced

        class StopOnAbort(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full((input_ids.shape[0],), policy.abort_generation.is_set(),
                                  device=input_ids.device, dtype=torch.bool)

        observer = ObserveAndForce()
        inputs = torch.tensor([prompt], dtype=torch.long, device=policy.input_device)
        with torch.inference_mode():
            sequences = policy.model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                generation_config=_old_generation_config(policy, len(completion)),
                # In Transformers 5.17 a None in GenerationConfig is refilled
                # from the model defaults. Explicit kwargs disable these stops.
                eos_token_id=None, forced_eos_token_id=None, stop_strings=None, max_time=None,
                logits_processor=LogitsProcessorList([observer]),
                stopping_criteria=StoppingCriteriaList([StopOnAbort()]),
            )
            actual = sequences[0, len(prompt):].cpu().tolist()
            values = torch.stack(observer.values).float().cpu().tolist() if observer.values else []
        _synchronize(torch)
        elapsed = time.perf_counter() - started
        if policy.abort_generation.is_set():
            raise RuntimeError("Old serial replay was interrupted; unfinished calls are not completed work")
        if actual != completion or len(values) != len(completion) or not all(map(math.isfinite, values)):
            raise RuntimeError("Old serial replay did not retain every recorded token and its finite categorical logprob")
        probabilities[record["id"]] = values
        calls.append({"id": record["id"], "prompt_tokens": len(prompt), "completion_tokens": len(completion),
                      "generation_elapsed_seconds": elapsed, "first_logits_seconds": observer.first_logits_seconds,
                      "tokens_per_second": len(completion) / elapsed})
        _write(destination / "progress.json", {"completed_calls": len(calls),
                                                "completed_tokens": sum(item["completion_tokens"] for item in calls),
                                                "elapsed_seconds": time.perf_counter() - all_started})
    elapsed = time.perf_counter() - all_started
    tokens = sum(item["completion_tokens"] for item in calls)
    return {
        "backend": "old_serial_hf_unsloth_generate", "batch_size_limit": 1,
        "profiled_pass": False, "calls": len(calls), "completion_tokens": tokens,
        "elapsed_seconds": elapsed, "tokens_per_second": tokens / elapsed,
        "prefill_boundary_synchronization": True, "per_call": calls,
        "peak_memory_allocated_gib": [torch.cuda.max_memory_allocated(i) / 2**30
                                      for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "measurement_difference": (
            "Original model.generate forward/cache/decoding path, with a logits processor that records the original "
            "full categorical logprob then forces the recorded token. generate samples this forced distribution "
            "once per step. New-engine replay instead samples a full-distribution proposal and discards it. "
            "Neither replay is free generation; forcing, observation and first-logits synchronization add overhead. "
            "No prompt tokenization or tool-response parsing is timed in either path."
        ),
    }, probabilities


class _ReplayPolicy:
    """Use the real policy model while keeping replay output out of the parser."""

    def __init__(self, policy):
        self.policy = policy

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def finish_generation(self, prepared, completion_ids, behavior, **timing):
        if completion_ids != prepared["replay_tokens"]:
            raise RuntimeError("Replay did not consume the complete recorded completion")
        if len(behavior) != len(completion_ids) or not all(math.isfinite(value) for value in behavior):
            raise RuntimeError("Replay requires one finite probability per recorded token")
        # Returned only to this process, never serialized by this program.
        return {"behavior_logprobs": behavior, "completion_tokens": len(completion_ids), **timing}


class _ReplayEngine(GenerationEngine):
    """Keep real cache scheduling while forcing the same recorded token paths."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_logits_seconds = {}
        self._prefill_request = None

    def _prefill(self, request):
        self._prefill_request = request
        try:
            result = super()._prefill(request)
            if request.request_id not in self.first_logits_seconds:
                # Exact-prefix hits have no model forward; include cache access.
                _synchronize(self.torch)
                self.first_logits_seconds[request.request_id] = time.perf_counter() - request.started_at
            return result
        finally:
            self._prefill_request = None

    def _forward(self, ids, mask, positions, cache, phase):
        result = super()._forward(ids, mask, positions, cache, phase)
        if phase == "prefill":
            # The extra prefill boundary sync makes this a readiness measurement,
            # not CPU enqueue time. Ordinary decode steps still use the engine.
            _synchronize(self.torch)
            request = self._prefill_request
            self.first_logits_seconds[request.request_id] = time.perf_counter() - request.started_at
        return result

    def _sample(self, requests, logits, profiled):
        torch = self.torch
        with self.profiler.region("sample_logprobs"):
            log_distribution = torch.log_softmax(logits.float(), dim=-1)
            # Retain categorical-sampling cost, then discard its proposals. The
            # recorded path, not a new sampled continuation, is what is measured.
            torch.multinomial(log_distribution.exp(), 1)
            targets = torch.tensor([[request.prepared["replay_tokens"][len(request.completion)]]
                                    for request in requests], device=logits.device, dtype=torch.long)
            selected = log_distribution.gather(-1, targets).squeeze(-1)
            token_ids = targets.squeeze(-1).cpu().tolist()
        ready = time.perf_counter()
        for row, (request, token) in enumerate(zip(requests, token_ids)):
            request.completion.append(token)
            request.logprobs.append(selected[row])
            request.next_logits = logits[row].detach()
            request.profiled_tokens += int(profiled)
            if request.first_token_seconds is None:
                request.first_token_seconds = ready - request.started_at
        self.tokens += len(requests)
        self.profiled_tokens += len(requests) if profiled else 0

    def _done(self, request):
        # Include the entire recorded completion even if it contains an EOS.
        return len(request.completion) == len(request.prepared["replay_tokens"])


def _replay(policy, records, batch_size, generation_config, destination, *, profiled=False):
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    torch = policy.torch
    config = {**generation_config, "max_batch_size": batch_size,
              "max_batch_tokens": batch_size * policy.max_context,
              "profiling": {**generation_config.get("profiling", {}), "enabled": profiled}}
    _synchronize(torch)
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)
    engine = _ReplayEngine(_ReplayPolicy(policy), config, destination)
    pending, results = deque(records), {}
    queued_at = time.perf_counter()
    started = queued_at
    try:
        while pending or engine.active_count:
            incoming = []
            while pending and len(incoming) < engine.capacity:
                record = pending[0]
                prepared = {"prompt_ids": record["prompt_ids"], "replay_tokens": record["completion_ids"],
                            "effective_max_tokens": len(record["completion_ids"])}
                if not engine.can_admit(prepared, incoming):
                    break
                pending.popleft()
                incoming.append({"request_id": record["id"], "prepared": prepared, "queued_at": queued_at})
            if not incoming and not engine.active_count:
                raise RuntimeError("A full recorded call cannot fit in the configured batching capacity")
            for identity, result in engine.step(incoming):
                if identity in results:
                    raise RuntimeError("Replay returned a request twice")
                results[identity] = result
        _synchronize(torch)
        elapsed = time.perf_counter() - started
        if set(results) != {record["id"] for record in records}:
            raise RuntimeError("Replay did not finish the complete fixed workload")
        tokens = sum(len(record["completion_ids"]) for record in records)
        if engine.tokens != tokens:
            raise RuntimeError("Observed replay tokens differ from the complete recorded workload")
        summary = {
            "batch_size_limit": batch_size, "max_observed_batch_size": engine.max_observed_batch,
            "forward_batch_histograms": engine.actual_forward_batches,
            "max_actual_decode_batch_size": max(engine.actual_forward_batches["decode"], default=0),
            "profiled_pass": profiled, "calls": len(records), "completion_tokens": tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens / elapsed if not profiled else None,
            "profiling_elapsed_seconds_not_for_throughput": elapsed if profiled else None,
            "prefill_boundary_synchronization": True,
            "prefix_cache": engine.prefixes.stats(),
            "peak_memory_allocated_gib": [torch.cuda.max_memory_allocated(i) / 2**30
                                          for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            "per_call": [{"id": record["id"], "prompt_tokens": len(record["prompt_ids"]),
                          "completion_tokens": len(record["completion_ids"]),
                          "first_logits_seconds": engine.first_logits_seconds[record["id"]],
                          **{key: results[record["id"]].get(key) for key in
                             ("first_token_seconds", "queue_wait_seconds", "generation_elapsed_seconds",
                              "reused_prefix_tokens")}}
                         for record in records],
        }
    finally:
        engine.close()
    summary["profile_artifacts"] = sorted(path.relative_to(destination).as_posix()
                                          for path in (destination / "profiles").rglob("*") if path.is_file())
    return summary, {identity: result["behavior_logprobs"] for identity, result in results.items()}


def _probability_difference(reference, measured, atol, *, reference_description=None):
    differences, by_call = [], []
    if set(reference) != set(measured):
        raise RuntimeError("Probability comparisons require the same complete calls")
    for identity in reference:
        expected, actual = reference[identity], measured[identity]
        if len(expected) != len(actual) or not expected:
            raise RuntimeError("Probability comparisons require every recorded token")
        delta = [float(right) - float(left) for left, right in zip(expected, actual)]
        if not all(math.isfinite(value) for value in delta):
            raise RuntimeError("Nonfinite per-token probability difference")
        differences.extend(delta)
        by_call.append({"id": identity, "tokens": len(delta), "max_abs_logprob_delta": max(map(abs, delta)),
                        "mean_abs_logprob_delta": sum(map(abs, delta)) / len(delta)})
    absolute = sorted(map(abs, differences))
    return {"reference": reference_description or "same model and fresh LoRA, this run's unprofiled batch-size-1 replay",
            "tokens": len(differences), "reporting_atol": atol,
            "tokens_exceeding_atol": sum(value > atol for value in absolute),
            "max_abs_logprob_delta": absolute[-1],
            "mean_abs_logprob_delta": sum(absolute) / len(absolute),
            "p95_abs_logprob_delta": absolute[math.ceil(0.95 * len(absolute)) - 1],
            "rms_logprob_delta": math.sqrt(sum(value**2 for value in differences) / len(differences)),
            "per_call": by_call}


def _worker(args, cfg):
    worker_started = time.monotonic()
    output = args.output_dir.resolve(strict=True)
    records, sources = _load_calls(args.calls, args.max_calls, int(cfg["policy"].get("max_context", 32768)))
    if len(records) < max(args.batch_sizes):
        raise ValueError("Fewer complete calls than the largest requested batch; supply additional --calls sources")
    model_path = args.model_path.expanduser().resolve(strict=True)
    model_config = {**cfg["policy"], "model_path": str(model_path)}
    metadata = {name: _sha256(model_path / name) for name in
                ("config.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                 "model.safetensors.index.json") if (model_path / name).is_file()}
    response_template_metadata = None
    if model_config.get("response_template_path"):
        sidecar = Path(model_config["response_template_path"]).expanduser().resolve(strict=True)
        response_template_metadata = {"path": str(sidecar), "sha256": _sha256(sidecar),
                                      "purpose": "Policy parser metadata; replay does not parse responses"}
    manifest = {
        "experiment": "complete-call teacher-forced generation replay", "sources": sources,
        "config_path": str(args.config.resolve()), "config_sha256": _sha256(args.config),
        "model_path": str(model_path), "declared_model_name": model_config.get("model_name"),
        "declared_model_revision": model_config.get("model_revision"), "model_metadata_sha256": metadata,
        "response_template_metadata": response_template_metadata,
        "policy_initialization": "Policy's fresh zero-B LoRA; no trained adapter is loaded",
        "batch_sizes": args.batch_sizes, "walltime_seconds": args.walltime_seconds,
        "prefix_cache_enabled": args.prefix_cache, "separate_profiled_passes": args.profile,
        "historical_behavior_logprobs_used": False,
        "workload": (
            "One complete first-call warmup for old serial generate, its complete fixed-workload pass, "
            "one complete first-call warmup for the new engine, then that same complete call list for B1/B2/B4. "
            "Optional profiling repeats only the new-engine passes after all unprofiled passes."
        ),
        "interpretation": (
            "Recorded completions are forced; original full categorical logprobs are recorded before forcing. "
            "This is a decoder performance/probability experiment, not a new trajectory, training step, "
            "official verification or repair-success measurement. Device mapping is not tensor parallelism. "
            "Reduced-precision batch differences are reported, not asserted to be zero or unbiased. "
            "Prefill boundaries synchronize every visible GPU; profiled passes are excluded from throughput."
            " Warmup contains one complete call per backend; compilation for additional batch shapes remains measured. "
            "Old model.generate samples a forced distribution, whereas new-engine replay discards a full-category "
            "proposal, so sampling overheads and free-generation speed are not identical."
        ),
        "privacy": "No prompt text, token IDs, completion text or per-token probabilities are exported.",
    }
    _write(output / "experiment.json", manifest)
    from .policy import Policy

    policy = Policy(model_config)
    torch = policy.torch
    manifest["environment"] = {"torch": str(torch.__version__), "cuda": str(torch.version.cuda),
                               "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    _write(output / "experiment.json", manifest)
    generation = dict(cfg.get("generation", {}))
    if not args.prefix_cache:
        generation.update(prefix_cache_entries=0, prefix_cache_bytes=0)
    reference, total_tokens = None, 0
    summaries, warmups = [], []

    def save_progress(complete=False):
        _write(output / "summary.json", {"complete": complete,
                                         "elapsed_worker_seconds": time.monotonic() - worker_started,
                                         "replayed_tokens_including_warmup": total_tokens,
                                         "token_count_scope": "completed passes only; an interrupted partial pass is not included",
                                         "warmups": warmups, "passes": summaries})

    old_warmup, _ = _old_serial_replay(policy, records[:1], output / "warmup-old-serial")
    _write(output / "warmup-old-serial" / "summary.json", {**old_warmup, "warmup_only": True})
    warmups.append({"pass": "warmup-old-serial", "completion_tokens": old_warmup["completion_tokens"]})
    total_tokens += old_warmup["completion_tokens"]
    save_progress()
    old_summary, old_probabilities = _old_serial_replay(policy, records, output / "old-serial")
    old_summary["pass"] = "old-serial"
    _write(output / "old-serial" / "summary.json", old_summary)
    summaries.append(old_summary)
    total_tokens += old_summary["completion_tokens"]
    save_progress()
    warmup, _ = _replay(policy, records[:1], 1, generation, output / "warmup-b1")
    _write(output / "warmup-b1" / "summary.json", {**warmup, "warmup_only": True})
    warmups.append({"pass": "warmup-b1", "completion_tokens": warmup["completion_tokens"]})
    total_tokens += warmup["completion_tokens"]
    save_progress()
    for profiled in ([False, True] if args.profile else [False]):
        for batch_size in args.batch_sizes:
            label = ("profiled" if profiled else "unprofiled") + f"-b{batch_size}"
            summary, probabilities = _replay(policy, records, batch_size, generation, output / label, profiled=profiled)
            if reference is None:
                reference = probabilities
                old_summary["probability_comparison"] = _probability_difference(reference, old_probabilities, args.logprob_atol)
                _write(output / "old-serial" / "summary.json", old_summary)
            summary["probability_comparison"] = _probability_difference(reference, probabilities, args.logprob_atol)
            summary["probability_vs_old_serial"] = _probability_difference(
                old_probabilities, probabilities, args.logprob_atol,
                reference_description="same model and fresh LoRA, this run's old serial HF/Unsloth generate replay",
            )
            summary["throughput_ratio_vs_old_serial_replay"] = (
                summary["tokens_per_second"] / old_summary["tokens_per_second"] if not profiled else None
            )
            summary["pass"] = label
            _write(output / label / "summary.json", summary)
            summaries.append(summary)
            total_tokens += summary["completion_tokens"]
            save_progress()
    save_progress(complete=True)


def _known_progress(output):
    path = output / "summary.json"
    if not path.is_file():
        return {"completed_pass_tokens_recorded": None, "completed_passes_recorded": 0}
    summary = json.loads(path.read_text())
    return {"completed_pass_tokens_recorded": summary.get("replayed_tokens_including_warmup"),
            "completed_passes_recorded": len(summary.get("passes", [])),
            "partial_pass_tokens_not_counted": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/first_batch.toml")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--calls", type=Path, action="append", required=True)
    parser.add_argument("--max-calls", type=int, default=4)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--walltime-seconds", type=float, required=True, help="Explicit whole-process GPU budget")
    parser.add_argument("--prefix-cache", action="store_true", help="Measure prefix reuse in an additional experiment")
    parser.add_argument("--profile", action="store_true", help="Run separate profiled passes after all throughput passes")
    parser.add_argument("--logprob-atol", type=float, default=0.02, help="Reporting threshold; exceeding it is reported")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        args.batch_sizes = sorted(set(int(value) for value in args.batch_sizes.split(",")))
    except ValueError:
        parser.error("--batch-sizes must be a comma-separated list of positive integers")
    if not args.batch_sizes or args.batch_sizes[0] != 1 or args.max_calls < max(args.batch_sizes):
        parser.error("Include batch size 1 and enough --max-calls for every requested batch size")
    if not math.isfinite(args.walltime_seconds) or args.walltime_seconds <= 0:
        parser.error("--walltime-seconds must be positive and finite")
    if not math.isfinite(args.logprob_atol) or args.logprob_atol < 0:
        parser.error("--logprob-atol must be nonnegative and finite")
    if sys.platform != "linux":
        parser.error("Run this experiment on the authorized Linux GPU server")
    cfg = tomllib.loads(args.config.read_text())
    os.umask(0o077)
    if not args._worker:
        args.output_dir = args.output_dir.expanduser().resolve()
        args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        command_args = ["--config", str(args.config.expanduser().resolve()),
                        "--model-path", str(args.model_path.expanduser().resolve()),
                        "--output-dir", str(args.output_dir), "--max-calls", str(args.max_calls),
                        "--batch-sizes", ",".join(map(str, args.batch_sizes)),
                        "--walltime-seconds", str(args.walltime_seconds),
                        "--logprob-atol", str(args.logprob_atol), "--_worker"]
        for path in args.calls:
            command_args.extend(("--calls", str(path.expanduser().resolve())))
        for enabled, flag in ((args.prefix_cache, "--prefix-cache"), (args.profile, "--profile")):
            if enabled:
                command_args.append(flag)
        parent_started = time.monotonic()
        child = subprocess.Popen([sys.executable, "-m", "training.profile_generation", *command_args],
                                 cwd=PROJECT, start_new_session=True)
        try:
            return child.wait(timeout=args.walltime_seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                pass
            _write(args.output_dir / "interruption.json", {"complete": False, "reason": type(exc).__name__,
                                                          "walltime_seconds": args.walltime_seconds,
                                                          "elapsed_parent_seconds": time.monotonic() - parent_started,
                                                          **_known_progress(args.output_dir),
                                                          "note": "Keep completed passes; do not infer missing outcomes."})
            return 124
    def terminate(*_):
        raise KeyboardInterrupt("Run budget terminated replay")

    signal.signal(signal.SIGTERM, terminate)
    child_started = time.monotonic()
    try:
        _worker(args, cfg)
    except BaseException as exc:
        _write(args.output_dir / "failure.json", {"complete": False, "error_type": type(exc).__name__,
                                                 "error": str(exc), "elapsed_worker_seconds": time.monotonic() - child_started,
                                                 **_known_progress(args.output_dir)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
