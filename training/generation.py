"""Continuous decode batching on the learner's existing BF16/LoRA model.

One worker owns the GPU. Requests join between decode steps, finished rows leave
immediately, and every remaining row advances in one shared model forward.
This is dynamic batching, not tensor parallelism. Complete prompts advance in
bounded chunks between decode steps; identical pending prompts share prefill.
"""
from __future__ import annotations

from contextlib import nullcontext
from collections import Counter, deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import uuid

from .kv_cache import PrefixCache, clone_cache, enable_capacity_cache, merge_caches, select_cache
from .policy import GenerationAborted
from .profiling import GenerationProfiler


@dataclass
class _Request:
    request_id: str
    prepared: dict
    queued_at: float
    started_at: float | None = None
    prefill_offset: int = 0
    prefill_cache: object = None
    prefill_leader: str | None = None
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
    def __init__(self, policy, config, run_dir, on_request_started=None):
        self.policy, self.torch = policy, policy.torch
        self.config = dict(config or {})
        self.max_batch_size = int(self.config.get("max_batch_size", 4))
        self.max_batch_tokens = int(self.config.get("max_batch_tokens", self.max_batch_size * policy.max_context))
        self.prefill_chunk_tokens = int(self.config.get("prefill_chunk_tokens", 512))
        self.max_prefill_tokens_per_step = int(self.config.get("max_prefill_tokens_per_step", 1024))
        self.kv_capacity_increment = int(self.config.get("kv_capacity_increment", 256))
        if min(self.prefill_chunk_tokens, self.max_prefill_tokens_per_step) < 1 or self.kv_capacity_increment < 0:
            raise ValueError("Prefill token budgets must be positive and KV increment nonnegative")
        self.on_request_started = on_request_started
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
        self.prefilling = deque()
        self.prefill_leaders = {}
        self.cache = self.mask = None
        self.closed = False
        eos = policy.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.steps = self.tokens = self.completed = self.profiled_tokens = 0
        self.cancelled_requests = self.cancelled_generated_tokens = 0
        self.busy_seconds = self.unprofiled_seconds = 0.0
        self.max_observed_batch = 0
        self.prefill_forward_tokens = self.prefill_chunks = self.intermediate_head_skips = 0
        self.deduplicated_prefill_requests = self.deduplicated_prefill_tokens = 0
        self.kv_capacity_counters = {"growths": 0, "prefix_elements_copied": 0, "appended_elements": 0}
        self.actual_forward_batches = {"prefill": Counter(), "decode": Counter()}
        self.last_profiled = False
        self.started_at = time.perf_counter()

    @property
    def active_count(self):
        return len(self.active) + len(self.prefilling)

    @property
    def capacity(self):
        return self.max_batch_size - self.active_count

    def can_admit(self, prepared, incoming=()):
        reserved = [item.reserved_length for item in [*self.active, *self.prefilling]]
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

    def _forward(self, ids, mask, positions, cache, phase, *, project_logits=True):
        self._check_abort()
        self.actual_forward_batches[phase][int(ids.shape[0])] += 1
        if phase == "prefill":
            self.prefill_forward_tokens += ids.numel()
            self.prefill_chunks += 1
            self.intermediate_head_skips += int(not project_logits)
        with self.profiler.region(phase):
            if project_logits:
                logits, cache = self.policy.forward_generation(ids, mask, positions, cache)
            else:
                logits, cache = self.policy.forward_generation(ids, mask, positions, cache, project_logits=False)
        if project_logits and (logits.ndim != 2 or logits.shape[0] != ids.shape[0]):
            raise RuntimeError("Expected one vocabulary distribution per active row")
        return logits, cache

    def _start_request(self, request):
        if request.started_at is None:
            request.started_at = time.perf_counter()
            if self.on_request_started is not None:
                self.on_request_started(request.request_id, request.started_at)

    def _prefill(self, request, token_limit):
        """Advance one complete prompt without sampling intermediate chunks."""
        torch, ids = self.torch, request.prepared["prompt_ids"]
        self._start_request(request)
        if request.prefill_cache is None:
            with self.profiler.region("cache"):
                entry = self.prefixes.get(ids, self.version)
            offset = entry.cached_length if entry is not None else 0
            request.reused_tokens = request.prefill_offset = offset
            if entry is not None:
                request.prefill_cache = entry.cache
                if offset == len(ids):
                    return entry.cache, entry.next_logits.unsqueeze(0), 0, True
            else:
                request.prefill_cache = enable_capacity_cache(
                    self.policy.new_generation_cache(), self.kv_capacity_increment, self.policy.max_context,
                    self.kv_capacity_counters)
        offset = request.prefill_offset
        end = min(len(ids), offset + self.prefill_chunk_tokens, offset + token_limit)
        if end <= offset:
            raise ValueError("Prefill needs a positive remaining token budget")
        complete = end == len(ids)
        device = self.policy.input_device
        suffix = torch.tensor([ids[offset:end]], dtype=torch.long, device=device)
        mask = torch.ones((1, end), dtype=torch.long, device=device)
        positions = torch.arange(offset, end, device=device).unsqueeze(0)
        logits, cache = self._forward(suffix, mask, positions, request.prefill_cache, "prefill",
                                      project_logits=complete)
        request.prefill_cache, request.prefill_offset = cache, end
        if complete:
            with self.profiler.region("cache"):
                self.prefixes.put(cache, ids, self.version, logits[0])
        return cache, logits, end - offset, complete

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
            try:
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
            except (ValueError, KeyError, TypeError) as error:
                # A response parser failure belongs to this invocation. CUDA,
                # cache-integrity and model-runtime errors still escape as fatal.
                return request.request_id, error
        self.completed += 1
        return request.request_id, result

    def _rebuild(self, requests, caches):
        self.active = requests
        if not requests:
            self.cache = self.mask = None
            return
        lengths = [request.cached_length for request in requests]
        with self.profiler.region("cache"):
            self.cache = (clone_cache(caches[0]) if getattr(caches[0], "_membership_view", False) else caches[0]) if len(caches) == 1 else merge_caches(caches, lengths)
            width = max(lengths)
            device = self.policy.input_device
            columns = self.torch.arange(width, device=device).unsqueeze(0)
            pads = self.torch.tensor([width - n for n in lengths], device=device).unsqueeze(1)
            self.mask = (columns >= pads).long()

    def cancel(self, request_ids):
        """Cancel queued-prefill or decoding rows at the owner's step boundary."""
        identities = set(request_ids)
        removed = [item.request_id for item in [*self.active, *self.prefilling] if item.request_id in identities]
        if not removed:
            return []
        with self.torch.inference_mode():
            requests, caches = [], []
            for row, item in enumerate(self.active):
                if item.request_id in identities:
                    self.cancelled_requests += 1
                    self.cancelled_generated_tokens += len(item.completion)
                else:
                    requests.append(item)
                    caches.append(select_cache(self.cache, row, item.cached_length, copy_tensors=False))
            if len(requests) != len(self.active):
                self._rebuild(requests, caches)
            remaining = deque()
            for item in self.prefilling:
                if item.request_id in identities:
                    self.cancelled_requests += 1
                    if item.prefill_leader is None:
                        self.prefill_leaders.pop(tuple(item.prepared["prompt_ids"]), None)
                else:
                    remaining.append(item)
            self.prefilling = remaining
            # A cancelled leader has produced no completion token. Promote a
            # remaining follower; it will safely rebuild/reuse the full prefix.
            for item in self.prefilling:
                if item.prefill_leader in identities:
                    key = tuple(item.prepared["prompt_ids"])
                    leader = self.prefill_leaders.get(key)
                    item.prefill_leader = leader.request_id if leader is not None else None
                    if leader is None:
                        self.prefill_leaders[key] = item
        return removed

    def _decode(self, profiled):
        if not self.active:
            return []
        torch, device = self.torch, self.policy.input_device
        tokens = torch.tensor([[item.completion[-1]] for item in self.active], device=device)
        positions = torch.tensor([[item.cached_length] for item in self.active], device=device)
        self.mask = torch.cat((self.mask, torch.ones((len(self.active), 1), device=device,
                                                   dtype=self.mask.dtype)), dim=-1)
        logits, self.cache = self._forward(tokens, self.mask, positions, self.cache, "decode")
        self._sample(self.active, logits, profiled)
        done = [self._done(item) for item in self.active]
        finished = []
        if any(done):
            requests, caches = [], []
            for row, (request, ended) in enumerate(zip(self.active, done)):
                with self.profiler.region("cache"):
                    cache = select_cache(self.cache, row, request.cached_length, copy_tensors=False)
                if ended:
                    finished.append(self._finish(request, cache))
                else:
                    requests.append(request)
                    caches.append(cache)
            self._rebuild(requests, caches)
        return finished

    def step(self, incoming):
        """Advance old decoders, then bounded round-robin complete-prompt work.

        In-flight identical prompts have one prefill leader. Their first tokens
        are independently sampled after that full prefix is ready. Temporary
        shared read views are materialized into independent batch rows before
        any subsequent decode can mutate KV state.
        """
        if self.closed:
            raise RuntimeError("Generation engine is closed")
        self._check_abort()
        for index, item in enumerate(incoming):
            if not self.can_admit(item["prepared"], incoming[:index]):
                raise ValueError("Scheduler exceeded the reserved batch context capacity")
        if not incoming and not self.active_count:
            return []
        started, before = time.perf_counter(), self.tokens
        for item in incoming:
            request = _Request(**item)
            key = tuple(request.prepared["prompt_ids"])
            leader = self.prefill_leaders.get(key)
            if leader is not None:
                request.prefill_leader = leader.request_id
            else:
                self.prefill_leaders[key] = request
            self.prefilling.append(request)
        batch_size = self.active_count
        self.max_observed_batch = max(self.max_observed_batch, batch_size)
        with self.torch.inference_mode(), self._autocast():
            with self.profiler.step(phase="prefill_decode" if self.prefilling else "decode", batch_size=batch_size) as profiled:
                finished = self._decode(profiled)
                ready = []
                remaining_tokens = self.max_prefill_tokens_per_step
                skipped = 0
                while self.prefilling and remaining_tokens > 0:
                    request = self.prefilling.popleft()
                    if request.prefill_leader is not None:
                        self.prefilling.append(request)
                        skipped += 1
                        if skipped >= len(self.prefilling):
                            break
                        continue
                    skipped = 0
                    cache, logits, consumed, complete = self._prefill(request, remaining_tokens)
                    remaining_tokens -= consumed
                    if not complete:
                        self.prefilling.append(request)
                        continue
                    self.prefill_leaders.pop(tuple(request.prepared["prompt_ids"]), None)
                    ready.append((request, cache, logits))
                    waiting = deque()
                    for follower in self.prefilling:
                        if follower.prefill_leader == request.request_id:
                            follower.prefill_offset = follower.reused_tokens = len(follower.prepared["prompt_ids"])
                            follower.prefill_leader = None
                            ready.append((follower, cache, logits))
                            self.deduplicated_prefill_requests += 1
                            self.deduplicated_prefill_tokens += follower.reused_tokens
                        else:
                            waiting.append(follower)
                    self.prefilling = waiting
                if ready:
                    # Start followers at their first actual GPU sampling/copy,
                    # rather than charging their leader's earlier prefill time.
                    for request, _cache, _logits in ready:
                        self._start_request(request)
                    logits = self.torch.cat([item[2] for item in ready], dim=0)
                    self._sample([item[0] for item in ready], logits, profiled)
                    with self.profiler.region("cache"):
                        requests = list(self.active)
                        caches = [select_cache(self.cache, row, request.cached_length, copy_tensors=False)
                                  for row, request in enumerate(self.active)]
                    for request, cache, _logits in ready:
                        request.prefill_cache = None
                        if self._done(request):
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
                              "active": self.active_count, "decoding": len(self.active),
                              "prefilling": len(self.prefilling), "sampled_tokens": self.tokens - before,
                              "prefill_tokens": self.max_prefill_tokens_per_step - remaining_tokens,
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
            self.prefilling.clear()
            self.prefill_leaders.clear()
            self.cache = self.mask = None
            self.prefixes.clear()
            self.closed = True
            raise
        unprofiled = self.tokens - self.profiled_tokens
        summary = {"event": "generation_summary", "policy_version": self.version,
                   "collection_id": self.collection_id,
                   "completed_requests": self.completed, "unfinished_requests": self.active_count,
                   "cancelled_requests": self.cancelled_requests,
                   "cancelled_generated_tokens": self.cancelled_generated_tokens,
                   "decode_iterations": self.steps, "sampled_tokens": self.tokens,
                   "max_observed_batch_size": self.max_observed_batch,
                   "forward_batch_histograms": self.actual_forward_batches,
                   "prefill_forward_tokens": self.prefill_forward_tokens,
                   "prefill_chunks": self.prefill_chunks,
                   "intermediate_prefill_head_skips": self.intermediate_head_skips,
                   "deduplicated_prefill_requests": self.deduplicated_prefill_requests,
                   "deduplicated_prefill_tokens": self.deduplicated_prefill_tokens,
                   "prefill_chunk_tokens": self.prefill_chunk_tokens,
                   "max_prefill_tokens_per_step": self.max_prefill_tokens_per_step,
                   "kv_capacity_increment": self.kv_capacity_increment,
                   "full_kv_capacity_updates": dict(self.kv_capacity_counters),
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
        self.prefilling.clear()
        self.prefill_leaders.clear()
        self.cache = self.mask = None
        self.prefixes.clear()
        self.closed = True
