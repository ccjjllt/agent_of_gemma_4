"""Bounded CPU/CUDA profiling windows for the generation worker.

This module does not import torch or inspect prompts. The caller supplies its
already loaded torch module and places ``step`` around a scheduler iteration,
with ``region`` around prefill, decode, sampling and cache work. Only counts,
operator names, tensor shapes and timing/memory statistics are exported.

Profiler setup, shape recording and boundary synchronization perturb execution.
Profiled windows must not be used as unprofiled throughput measurements.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import threading
import time


class GenerationProfiler:
    """Profile a finite number of windows on one generation thread.

    ``wait_steps`` skips initial iterations, then each window has
    ``warmup_steps`` followed by ``active_steps`` recorded iterations.
    ``interval_steps`` skips iterations between windows, and ``repeat`` is
    finite and positive. ``step`` yields whether profiling affects this
    iteration (including warmup). No CUDA synchronization occurs on ordinary
    iterations; all visible GPUs synchronize at window boundaries only.
    """

    PHASES = frozenset(("generation", "prefill", "decode", "prefill_decode"))
    REGIONS = frozenset(("prefill", "decode", "sample_logprobs", "cache",
                         "admission", "finalize", "transfer"))

    def __init__(self, torch, config: dict | None, output_dir: str | Path):
        config = dict(config or {})
        self.torch = torch
        self.output_dir = Path(output_dir)
        self.enabled = self._boolean(config, "enabled", False)
        self.record_shapes = self._boolean(config, "record_shapes", True)
        self.profile_memory = self._boolean(config, "profile_memory", True)
        self.wait_steps = self._integer(config, "wait_steps", 16, minimum=0)
        self.warmup_steps = self._integer(config, "warmup_steps", 2, minimum=0)
        self.active_steps = self._integer(config, "active_steps", 8, minimum=1)
        self.interval_steps = self._integer(config, "interval_steps", 64, minimum=0)
        self.repeat = self._integer(config, "repeat", 1, minimum=1)
        self.row_limit = self._integer(config, "row_limit", 50, minimum=1)
        self._iteration = 0
        self._next_start = self.wait_steps
        self._windows = 0
        self._profile = None
        self._window = None
        self._thread = None
        self._inside_step = False
        self._closed = False
        self._cuda = False

    @staticmethod
    def _integer(config, name, default, *, minimum):
        value = config.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"profiling.{name} must be an integer >= {minimum}")
        return value

    @staticmethod
    def _boolean(config, name, default):
        value = config.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"profiling.{name} must be a boolean")
        return value

    @property
    def is_profiling(self):
        return self._profile is not None

    def _assert_owner(self):
        owner = threading.get_ident()
        if self._thread is None:
            self._thread = owner
        elif self._thread != owner:
            raise RuntimeError("GenerationProfiler must stay on its generation thread")

    def _synchronize(self):
        started = time.perf_counter()
        if self._cuda:
            for device in range(self.torch.cuda.device_count()):
                self.torch.cuda.synchronize(device)
        return time.perf_counter() - started

    def _start_window(self):
        self.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._cuda = bool(self.torch.cuda.is_available())
        activities = [self.torch.profiler.ProfilerActivity.CPU]
        if self._cuda:
            activities.append(self.torch.profiler.ProfilerActivity.CUDA)
        start_sync = self._synchronize()
        started = time.perf_counter()
        profile = self.torch.profiler.profile(
            activities=activities,
            schedule=self.torch.profiler.schedule(
                wait=0, warmup=self.warmup_steps, active=self.active_steps, repeat=1,
            ),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=False,
            with_flops=False,
        )
        profile.__enter__()
        self._profile = profile
        self._window = {
            "window": self._windows,
            "first_iteration": self._iteration,
            "activities": ["CPU", "CUDA"] if self._cuda else ["CPU"],
            "torch_version": str(self.torch.__version__),
            "record_shapes": self.record_shapes,
            "profile_memory": self.profile_memory,
            "warmup_steps": self.warmup_steps,
            "planned_active_steps": self.active_steps,
            "start_synchronize_seconds": start_sync,
            "profiler_start_seconds": time.perf_counter() - started,
            "iterations": [],
        }
        print(json.dumps({"event": "generation_profile_started", "window": self._windows,
                          "first_iteration": self._iteration,
                          "warmup_steps": self.warmup_steps,
                          "active_steps": self.active_steps,
                          "activities": self._window["activities"]}), flush=True)

    def region(self, name: str):
        """Return a named range without accepting arbitrary prompt-bearing labels."""
        if name not in self.REGIONS:
            raise ValueError(f"Unknown generation profiling region: {name!r}")
        if not self.is_profiling:
            return nullcontext()
        self._assert_owner()
        return self.torch.profiler.record_function("generation/" + name)

    @contextmanager
    def step(self, *, phase: str = "decode", batch_size: int = 1):
        if phase not in self.PHASES:
            raise ValueError(f"Unknown generation profiling phase: {phase!r}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Profile batch_size must be a positive integer")
        if self._closed:
            raise RuntimeError("GenerationProfiler is closed")
        if not self.enabled:
            yield False
            return
        self._assert_owner()
        if self._inside_step:
            raise RuntimeError("Generation profiling steps cannot nest")
        if self._profile is None and self._windows < self.repeat and self._iteration >= self._next_start:
            self._start_window()
        self._inside_step = True
        started = time.perf_counter()
        error = None
        try:
            scope = (self.torch.profiler.record_function("generation/iteration/" + phase)
                     if self.is_profiling else nullcontext())
            with scope:
                yield self.is_profiling
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._inside_step = False
            try:
                if self._profile is not None:
                    window_index = len(self._window["iterations"])
                    self._window["iterations"].append({
                        "iteration": self._iteration,
                        "phase": phase,
                        "batch_size": batch_size,
                        "warmup": window_index < self.warmup_steps,
                        "host_elapsed_seconds": time.perf_counter() - started,
                        "completed": error is None,
                    })
                    end_window = window_index + 1 >= self.warmup_steps + self.active_steps
                    if end_window or error is not None:
                        self._finish_window(complete=end_window and error is None,
                                            error_type=type(error).__name__ if error else None)
                    else:
                        self._profile.step()
            except BaseException as profile_error:
                if error is None:
                    raise
                # Keep a real model failure (including OOM) as the primary error.
                error.add_note(f"Profiler cleanup also failed: {type(profile_error).__name__}")
            finally:
                self._iteration += 1

    def _finish_window(self, *, complete: bool, error_type: str | None = None):
        profile, window = self._profile, self._window
        if profile is None:
            return
        self._profile = None
        self._window = None
        window["finish_synchronize_seconds"] = self._synchronize()
        finished = time.perf_counter()
        profile.__exit__(None, None, None)
        window["profiler_finish_seconds"] = time.perf_counter() - finished
        window["complete"] = complete
        window["error_type"] = error_type
        active = [item for item in window["iterations"] if not item["warmup"]]
        window["observed_active_steps"] = len(active)
        window["profiled_iteration_host_seconds"] = sum(
            item["host_elapsed_seconds"] for item in window["iterations"]
        )
        window["timing_interpretation"] = (
            "Host iteration times can include profiler overhead and asynchronous CUDA enqueue time. "
            "Only window boundaries synchronize all visible CUDA devices. CPU and device operator "
            "times overlap and must not be summed as elapsed time. Shape/memory recording and "
            "profiler setup alter execution; compare throughput in a separate run with profiling "
            "disabled. No slowdown ratio can be inferred from this trace alone."
        )
        window["privacy"] = "No prompt text, token IDs, completions or tensor values are recorded."
        basename = f"generation-{self._windows:03d}-step-{window['first_iteration']:06d}"
        exported = time.perf_counter()
        if active:
            trace = self.output_dir / (basename + ".trace.json")
            profile.export_chrome_trace(str(trace))
            averages = profile.key_averages(group_by_input_shape=self.record_shapes)
            table = averages.table(sort_by="self_cpu_time_total", row_limit=self.row_limit)
            if self._cuda:
                table += "\n\n" + averages.table(sort_by="self_device_time_total", row_limit=self.row_limit)
            (self.output_dir / (basename + ".operators.txt")).write_text(table + "\n")
            window["operators"] = [{
                "name": item.key,
                "count": item.count,
                "input_shapes": item.input_shapes if self.record_shapes else None,
                "self_cpu_time_us": item.self_cpu_time_total,
                "cpu_time_us": item.cpu_time_total,
                "self_device_time_us": item.self_device_time_total,
                "device_time_us": item.device_time_total,
                "self_cpu_memory_bytes": item.self_cpu_memory_usage,
                "self_device_memory_bytes": item.self_device_memory_usage,
            } for item in averages]
            window["trace"] = trace.name
        else:
            window["trace"] = None
            window["note"] = "Generation stopped during profiler warmup; no active trace was captured."
        window["export_seconds"] = time.perf_counter() - exported
        metadata = self.output_dir / (basename + ".json")
        metadata.write_text(json.dumps(window, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"event": "generation_profile_finished", "window": self._windows,
                          "complete": complete, "active_steps": len(active),
                          "metadata": str(metadata),
                          "throughput_includes_profiler_overhead": True}), flush=True)
        self._windows += 1
        self._next_start = self._iteration + 1 + self.interval_steps

    def close(self):
        """Finish a partial capture on the generation thread before it exits."""
        if self._closed:
            return
        if self.enabled:
            self._assert_owner()
        if self._inside_step:
            raise RuntimeError("Close GenerationProfiler outside its iteration context")
        try:
            self._finish_window(complete=False)
        finally:
            self._closed = True
