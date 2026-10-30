"""Continuous decode batching on the learner's existing BF16/LoRA model.

One worker owns the GPU. Requests join between decode steps, finished rows leave
immediately, and every remaining row advances in one shared model forward.
This is dynamic batching, not tensor parallelism. Prefills are bounded by each
request's original context and currently run one at a time between decode steps.
"""
from __future__ import annotations

from contextlib import nullcontext
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import uuid

from .kv_cache import PrefixCache, merge_caches, select_cache
from .policy import GenerationAborted
from .profiling import GenerationProfiler


@dataclass
class _Request:
    request_id: str
    prepared: dict
    queued_at: float
    started_at: float = field(default_factory=time.perf_counter)
    completion: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    reused_tokens: int = 0
    first_token_seconds: float | None = None
    profiled_tokens: int = 0
    next_logits: object = None

    @property
    def cached_length(self):
        # The last sampled token is consumed by the following decode forward.
        return len(self.prepared["prompt_ids"]) + max(0, len(self.completion) - 1)

    @property
    def reserved_length(self):
        return len(self.prepared["prompt_ids"]) + self.prepared["effective_max_tokens"]


class GenerationEngine:
    def __init__(self, policy, config, run_dir):
        self.policy, self.torch = policy, policy.torch
        self.config = dict(config or {})
        self.max_batch_size = int(self.config.get("max_batch_size", 4))
        self.max_batch_tokens = int(self.config.get("max_batch_tokens", self.max_batch_size * policy.max_context))
        self.version = int(self.config.get("policy_version", 0))
        if self.max_batch_size < 1 or self.max_batch_tokens < policy.max_context:
            raise ValueError("Need positive max_batch_size and max_batch_tokens >= a complete context")
        self.prefixes = PrefixCache(
            int(self.config.get("prefix_cache_entries", 8)),
            int(self.config.get("prefix_cache_bytes", 2 * 1024**3)),
        )
        self.run_dir = Path(run_dir)
        self.collection_id = f"policy-{self.version:04d}-{uuid.uuid4().hex[:12]}"
        self.profiler = GenerationProfiler(
            self.torch, self.config.get("profiling"), self.run_dir / "profiles" / self.collection_id,
        )
        self.active = []
        self.cache = self.mask = None
        self.closed = False
        eos = policy.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.steps = self.tokens = self.completed = self.profiled_tokens = 0
        self.busy_seconds = self.unprofiled_seconds = 0.0
        self.max_observed_batch = 0
        self.actual_forward_batches = {"prefill": Counter(), "decode": Counter()}
        self.last_profiled = False
        self.started_at = time.perf_counter()

    @property
    def active_count(self):
        return len(self.active)

    @property
    def capacity(self):
        return self.max_batch_size - self.active_count

    def can_admit(self, prepared, incoming=()):
        reserved = [item.reserved_length for item in self.active]
        candidates = [item["prepared"] for item in incoming] + [prepared]
        reserved.extend(len(item["prompt_ids"]) + item["effective_max_tokens"] for item in candidates)
        return (len(reserved) <= self.max_batch_size and
                max(reserved) * len(reserved) <= self.max_batch_tokens)

    def _check_abort(self):
        if self.policy.abort_generation.is_set():
            raise GenerationAborted("Run cancelled; unfinished requests have no task reward")

    def _autocast(self):
        if self.policy.input_device.type == "cuda":
            return self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
        return nullcontext()

    def _forward(self, ids, mask, positions, cache, phase):
        self._check_abort()
        self.actual_forward_batches[phase][int(ids.shape[0])] += 1
        with self.profiler.region(phase):
            logits, cache = self.policy.forward_generation(ids, mask, positions, cache)
        if logits.ndim != 2 or logits.shape[0] != ids.shape[0]:
            raise RuntimeError("Expected one vocabulary distribution per active row")
        return logits, cache

    def _prefill(self, request):
        torch, ids = self.torch, request.prepared["prompt_ids"]
        with self.profiler.region("cache"):
            entry = self.prefixes.get(ids, self.version)
        offset = entry.cached_length if entry is not None else 0
        request.reused_tokens = offset
        if offset == len(ids):
            return entry.cache, entry.next_logits.unsqueeze(0)
        cache = entry.cache if entry is not None else self.policy.new_generation_cache()
        device = self.policy.input_device
        suffix = torch.tensor([ids[offset:]], dtype=torch.long, device=device)
        mask = torch.ones((1, len(ids)), dtype=torch.long, device=device)
        positions = torch.arange(offset, len(ids), device=device).unsqueeze(0)
        logits, cache = self._forward(suffix, mask, positions, cache, "prefill")
        with self.profiler.region("cache"):
            self.prefixes.put(cache, ids, self.version, logits[0])
        return cache, logits

    def _sample(self, requests, logits, profiled):
        torch = self.torch
        with self.profiler.region("sample_logprobs"):
            # Exactly the declared full categorical behavior, including EOS.
            # No top-k/p, forced tokens, penalty or separate thinking cap.
            log_distribution = torch.log_softmax(logits.float(), dim=-1)
            sampled = torch.multinomial(log_distribution.exp(), 1)
            selected = log_distribution.gather(-1, sampled).squeeze(-1)
            # One small device->CPU transfer for all batch token IDs. Keep the
            # selected logprobs on device until a request finishes.
            token_ids = sampled.squeeze(-1).cpu().tolist()
        ready_at = time.perf_counter()
        for row, (request, token) in enumerate(zip(requests, token_ids)):
            request.completion.append(token)
            request.logprobs.append(selected[row])
            request.next_logits = logits[row].detach()
            request.profiled_tokens += int(profiled)
            if request.first_token_seconds is None:
                request.first_token_seconds = ready_at - request.started_at
        self.tokens += len(requests)
        self.profiled_tokens += len(requests) if profiled else 0

    def _done(self, request):
        return (request.completion[-1] in self.eos or
                len(request.completion) == request.prepared["effective_max_tokens"])

    def _finish(self, request, cache):
        self._check_abort()
        with self.profiler.region("cache"):
            cached_ids = request.prepared["prompt_ids"] + request.completion[:-1]
            self.prefixes.put(cache, cached_ids, self.version, request.next_logits)
        with self.profiler.region("finalize"):
            behavior = self.torch.stack(request.logprobs).float().cpu().tolist()
            result = self.policy.finish_generation(
                request.prepared, request.completion, behavior,
                generation_elapsed_seconds=time.perf_counter() - request.started_at,
                first_token_seconds=request.first_token_seconds,
                queue_wait_seconds=request.started_at - request.queued_at,
                reused_prefix_tokens=request.reused_tokens,
                profiled_tokens=request.profiled_tokens,
                generation_backend="continuous_native_cache",
                timing_semantics="per-request wall time overlaps other requests; use generation_summary for throughput",
            )
        self.completed += 1
        return request.request_id, result

    def _rebuild(self, requests, caches):
        self.active = requests
        if not requests:
            self.cache = self.mask = None
            return
        lengths = [request.cached_length for request in requests]
        with self.profiler.region("cache"):
            self.cache = caches[0] if len(caches) == 1 else merge_caches(caches, lengths)
            width = max(lengths)
            device = self.policy.input_device
            columns = self.torch.arange(width, device=device).unsqueeze(0)
            pads = self.torch.tensor([width - n for n in lengths], device=device).unsqueeze(1)
            self.mask = (columns >= pads).long()

    def step(self, incoming):
        """Admit requests, then advance all live rows by one batched decode.

        Cache tensors are split/merged only when membership changes, never for
        ordinary decode steps. A long prefill can delay a step; chunked prefill
        and CUDA graphs are intentionally not claimed by this implementation.
        """
        if self.closed:
            raise RuntimeError("Generation engine is closed")
        self._check_abort()
        for index, item in enumerate(incoming):
            if not self.can_admit(item["prepared"], incoming[:index]):
                raise ValueError("Scheduler exceeded the reserved batch context capacity")
        if not incoming and not self.active:
            return []
        started = time.perf_counter()
        before = self.tokens
        finished = []
        batch_size = len(self.active) + len(incoming)
        self.max_observed_batch = max(self.max_observed_batch, batch_size)
        with self.torch.inference_mode(), self._autocast():
            with self.profiler.step(phase="prefill_decode" if incoming else "decode", batch_size=batch_size) as profiled:
                if incoming:
                    with self.profiler.region("cache"):
                        caches = [select_cache(self.cache, row, request.cached_length)
                                  for row, request in enumerate(self.active)]
                    requests = list(self.active)
                    for item in incoming:
                        request = _Request(**item)
                        cache, logits = self._prefill(request)
                        self._sample([request], logits, profiled)
                        if self._done(request):
                            finished.append(self._finish(request, cache))
                        else:
                            requests.append(request)
                            caches.append(cache)
                    self._rebuild(requests, caches)
                if self.active:
                    torch, device = self.torch, self.policy.input_device
                    tokens = torch.tensor([[item.completion[-1]] for item in self.active], device=device)
                    positions = torch.tensor([[item.cached_length] for item in self.active], device=device)
                    self.mask = torch.cat((self.mask, torch.ones((len(self.active), 1), device=device,
                                                               dtype=self.mask.dtype)), dim=-1)
                    logits, self.cache = self._forward(tokens, self.mask, positions, self.cache, "decode")
                    self._sample(self.active, logits, profiled)
                    done = [self._done(item) for item in self.active]
                    if any(done):
                        requests, caches = [], []
                        for row, (request, ended) in enumerate(zip(self.active, done)):
                            with self.profiler.region("cache"):
                                cache = select_cache(self.cache, row, request.cached_length)
                            if ended:
                                finished.append(self._finish(request, cache))
                            else:
                                requests.append(request)
                                caches.append(cache)
                        self._rebuild(requests, caches)
        elapsed = time.perf_counter() - started
        self.steps += 1
        self.last_profiled = profiled
        self.busy_seconds += elapsed
        self.unprofiled_seconds += 0 if profiled else elapsed
        if incoming or finished or self.steps % 128 == 0:
            print(json.dumps({"event": "generation_batch_step", "policy_version": self.version,
                              "step": self.steps, "batch_size": batch_size,
                              "admitted": len(incoming), "finished": len(finished),
                              "active": len(self.active), "sampled_tokens": self.tokens - before,
                              "step_seconds": elapsed, "profiled": profiled,
                              "prefix_cache": self.prefixes.stats()}), flush=True)
        return finished

    def close(self):
        if self.closed:
            return
        try:
            self.profiler.close()
            drain_started = time.perf_counter()
            # Cache clones may still be queued on either model partition after
            # the last token transfer. Drain once at the collection boundary,
            # before another thread may score or update this model.
            if self.torch.cuda.is_available():
                for device in range(self.torch.cuda.device_count()):
                    self.torch.cuda.synchronize(device)
            drain_seconds = time.perf_counter() - drain_started
            self.busy_seconds += drain_seconds
            self.unprofiled_seconds += 0 if self.last_profiled else drain_seconds
        except BaseException:
            self.active.clear()
            self.cache = self.mask = None
            self.prefixes.clear()
            self.closed = True
            raise
        unprofiled = self.tokens - self.profiled_tokens
        summary = {"event": "generation_summary", "policy_version": self.version,
                   "collection_id": self.collection_id,
                   "completed_requests": self.completed, "unfinished_requests": len(self.active),
                   "decode_iterations": self.steps, "sampled_tokens": self.tokens,
                   "max_observed_batch_size": self.max_observed_batch,
                   "forward_batch_histograms": self.actual_forward_batches,
                   "max_actual_decode_batch_size": max(self.actual_forward_batches["decode"], default=0),
                   "engine_busy_seconds": self.busy_seconds,
                   "collection_drain_seconds": drain_seconds,
                   "collection_wall_seconds": time.perf_counter() - self.started_at,
                   "profiled_tokens": self.profiled_tokens, "unprofiled_tokens": unprofiled,
                   "unprofiled_busy_seconds": self.unprofiled_seconds,
                   "unprofiled_tokens_per_busy_second": (unprofiled / self.unprofiled_seconds
                                                         if self.unprofiled_seconds else None),
                   "prefix_cache": self.prefixes.stats()}
        self.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.run_dir / "generation_metrics.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(summary) + "\n")
        print(json.dumps(summary), flush=True)
        self.active.clear()
        self.cache = self.mask = None
        self.prefixes.clear()
        self.closed = True
