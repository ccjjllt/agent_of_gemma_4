"""Single-owner batched actions for the frozen Unsloth VFARL policy.

This is a new numerical execution profile. Each action batch has fresh KV;
left padding is masked and receives per-row logical positions. Histories are
never truncated. Completed rows remain masked dummy rows until the batch
finishes, so one request cannot mutate another request's cache or RNG.
"""
import asyncio
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import inspect
import time

from .common import TrainingError, sha256_file
from .vfarl_policy import digest, frozen_base_reference, cached_selected_logprobs
from .vfarl_unsloth import UNSLOTH_EXECUTION_PROFILE


BATCHED_EXECUTION_PROFILE = dict(UNSLOTH_EXECUTION_PROFILE,
    version=3, mode='batched-padded-action-decode',
    prompt_prefill='fresh-left-padded-batch-with-256-token-chunks-and-logical-per-row-positions',
    completion='batched-one-token-cached-forward-with-independent-request-generators',
    reference='independent-unpadded-zero-LoRA-256-token-backbone-and-16-selected-head-scoring',
    teacher_forced_scoring='fresh-reference-KV-with-chunked-backbone-and-selected-head-projection',
    cross_tool_kv=False,
    cache_boundary='fresh-private-KV-per-action-batch-no-cross-session-retention',
    padding='left-pad-keys-masked-logical-position-ids-completed-rows-masked',
    host_transfer='one-batched-token-and-logprob-transfer-per-decode-step')
BATCHED_UNSLOTH_EXECUTION_PROFILE = BATCHED_EXECUTION_PROFILE


def configure_batched_policy(policy, *, max_batch_size=2, max_batch_tokens=32768,
                             prefill_chunk_size=256):
    """Bind the actual batch implementation/config before queue policy binding."""
    if (type(max_batch_size) is not int or max_batch_size < 1
            or type(max_batch_tokens) is not int or max_batch_tokens < 1
            or prefill_chunk_size != 256):
        raise TrainingError('Batched sampler needs positive batch/token budgets and its declared 256-token chunks')
    runtime = dict(max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens, prefill_chunk_size=prefill_chunk_size,
        sampler_sha256=sha256_file(__file__),
        reference_scorer_sha256=sha256_file(inspect.getfile(cached_selected_logprobs)))
    if (getattr(policy, '_inference_profile', None) == BATCHED_EXECUTION_PROFILE
            and getattr(policy, '_inference_runtime', None) == runtime
            and policy.probability_profile == BATCHED_EXECUTION_PROFILE
            and policy.execution.get('execution_mode') == 'inference'):
        return policy.identity
    policy._inference_profile = dict(BATCHED_EXECUTION_PROFILE)
    policy._inference_runtime = runtime
    policy.clear_generation_cache(reason='batched-action-profile-installed')
    policy.switch_execution_mode('inference')
    return policy.identity


def _request(policy, prompt_token_ids, *, max_new_tokens, seed,
             thinking_budget=1024, stop_token_ids=None, compute_reference=True,
             session_id=None):
    prompt = list(prompt_token_ids)
    if (not prompt or any(type(token) is not int or token < 0 for token in prompt)
            or type(max_new_tokens) is not int or max_new_tokens < 1
            or type(thinking_budget) is not int or thinking_budget < 1
            or type(compute_reference) is not bool or type(seed) is not int):
        raise TrainingError('Batched actions require exact tokens, positive budgets and an explicit seed/reference flag')
    available = policy.cfg['max_sequence_length'] - len(prompt)
    if available < 1:
        raise TrainingError('Full real prompt leaves no completion room; truncation is forbidden')
    stop = set(stop_token_ids or [policy.tool_close, policy.turn_close])
    return dict(prompt=prompt, limit=min(max_new_tokens, available), seed=seed,
        max_new_tokens=max_new_tokens, thinking_budget=thinking_budget,
        compute_reference=compute_reference, session_id=session_id,
        stop=stop, generated=[], old=[], mask=[], in_thought=False, thought_count=0,
        done=False, generator=None,
        reason='context_limit' if available < max_new_tokens else 'length')


def _batch_token_slots(requests):
    # This bounds padded KV allocation, not just the sum of unpadded lengths.
    return len(requests) * (max(len(row['prompt']) for row in requests)
                            + max(row['limit'] for row in requests))


def _batch_span(requests):
    return max(len(row['prompt']) for row in requests) + max(row['limit'] for row in requests)


def _prefill(policy, requests, pad_token, *, chunk_size=256):
    torch = policy.torch
    width = max(len(row['prompt']) for row in requests)
    rows = [[pad_token] * (width - len(row['prompt'])) + row['prompt'] for row in requests]
    masks = [[0] * (width - len(row['prompt'])) + [1] * len(row['prompt']) for row in requests]
    ids = torch.tensor(rows, dtype=torch.long, device=policy.device)
    attention = torch.tensor(masks, dtype=torch.long, device=policy.device)
    positions = attention.cumsum(-1) - 1
    positions.masked_fill_(attention == 0, 0)
    past = None
    started = time.monotonic()
    for start in range(0, width, chunk_size):
        end = min(start + chunk_size, width)
        output = policy.model(input_ids=ids[:, start:end],
            attention_mask=attention[:, :end], position_ids=positions[:, start:end],
            past_key_values=past, use_cache=True, logits_to_keep=1, return_dict=True)
        past = output.past_key_values
        if past is None:
            raise TrainingError('Batched prefill returned no requested KV cache')
        if end == width:
            logits = output.logits[:, -1, :]
        del output
    for device in policy.cuda_devices:
        torch.cuda.synchronize(device)
    return logits, past, attention, width, time.monotonic() - started


def sample_batch(policy, requests, *, max_batch_tokens=32768, batch_index=0):
    """Run actual shared tensor forwards; return one exact action per request."""
    if (not requests or _batch_token_slots(requests) > max_batch_tokens
            or _batch_span(requests) > policy.cfg['max_sequence_length']):
        raise TrainingError('Batched action exceeds its declared padded KV token budget')
    if policy.probability_profile != BATCHED_EXECUTION_PROFILE:
        raise TrainingError('Batched sampler must retain its distinct actor probability profile')
    torch = policy.torch
    pad_token = policy.tokenizer.pad_token_id
    if pad_token is None:
        pad_token = policy.tokenizer.eos_token_id
    if type(pad_token) is not int or pad_token < 0:
        raise TrainingError('Batched sampler needs a valid masked padding token')
    started = time.monotonic()
    for device in policy.cuda_devices:
        torch.cuda.reset_peak_memory_stats(device)
    policy.model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        next_logits, past, attention, width, prefill_seconds = _prefill(policy, requests, pad_token)
        decode_started = time.monotonic()
        steps = 0
        for index in range(max(row['limit'] for row in requests)):
            if index:
                active = [not row['done'] for row in requests]
                ids = torch.tensor([[row['generated'][-1] if alive else pad_token]
                    for row, alive in zip(requests, active)], dtype=torch.long, device=policy.device)
                suffix_mask = torch.tensor([[int(alive)] for alive in active],
                    dtype=torch.long, device=policy.device)
                attention = torch.cat((attention, suffix_mask), dim=1)
                positions = torch.tensor([[len(row['prompt']) + index - 1 if alive else 0]
                    for row, alive in zip(requests, active)], dtype=torch.long, device=policy.device)
                output = policy.model(input_ids=ids, attention_mask=attention,
                    position_ids=positions, past_key_values=past, use_cache=True,
                    logits_to_keep=1, return_dict=True)
                past = output.past_key_values
                if past is None:
                    raise TrainingError('Batched decode returned no requested KV cache')
                next_logits = output.logits[:, -1, :]
                del output, ids, suffix_mask, positions
            logprobs = torch.log_softmax(next_logits.float(), dim=-1)
            del next_logits
            selected, forced_rows = [], []
            for row, values in zip(requests, logprobs):
                forced = row['in_thought'] and row['thought_count'] >= row['thinking_budget']
                forced_rows.append(forced)
                if row['done'] or forced:
                    selected.append(torch.tensor(pad_token if row['done'] else policy.thought_close,
                        dtype=torch.long, device=values.device))
                else:
                    if row['generator'] is None:
                        row['generator'] = torch.Generator(device=values.device).manual_seed(row['seed'])
                    elif row['generator'].device != values.device:
                        raise TrainingError('Batched autoregressive logits changed devices')
                    selected.append(torch.multinomial(values.exp(), 1, generator=row['generator']).reshape(()))
            token_ids = torch.stack(selected)
            probabilities = logprobs.gather(1, token_ids[:, None]).squeeze(1)
            # All vocabulary IDs are exactly representable in FP32. One
            # transfer replaces per-request token.item + logprob.item calls.
            observations = torch.stack((token_ids.float(), probabilities), dim=1).cpu().tolist()
            del logprobs, token_ids, probabilities, selected
            steps += 1
            for row, observed, forced in zip(requests, observations, forced_rows):
                if row['done']:
                    continue
                token, old = int(observed[0]), float(observed[1])
                row['generated'].append(token)
                row['old'].append(0.0 if forced else old)
                row['mask'].append(0 if forced else 1)
                if token == policy.thought_open:
                    row['in_thought'] = True
                elif token == policy.thought_close:
                    row['in_thought'] = False
                elif row['in_thought']:
                    row['thought_count'] += 1
                if token in row['stop'] and not row['in_thought']:
                    row['reason'] = 'tool_calls' if token == policy.tool_close else 'stop'
                    row['done'] = True
                elif len(row['generated']) >= row['limit']:
                    row['done'] = True
            if all(row['done'] for row in requests):
                break
        for device in policy.cuda_devices:
            torch.cuda.synchronize(device)
        decode_seconds = time.monotonic() - decode_started
        del past, attention
        generation_seconds = time.monotonic() - started
        references, reference_seconds = [], []
        for row in requests:
            reference_started = time.monotonic()
            values = None
            if row['compute_reference']:
                ids = torch.tensor([row['prompt'] + row['generated']], dtype=torch.long, device=policy.device)
                selected = torch.tensor([[0] * len(row['prompt']) + row['mask']],
                    dtype=torch.bool, device=policy.device)
                with frozen_base_reference(policy.model):
                    scored = cached_selected_logprobs(policy.model.get_base_model(), ids, selected,
                        prefill_chunk_size=256, projection_chunk_size=16)
                actual = iter(scored.float().cpu().tolist())
                values = [next(actual) if keep else 0.0 for keep in row['mask']]
                del ids, selected, scored
            references.append(values)
            reference_seconds.append(time.monotonic() - reference_started)
        for device in policy.cuda_devices:
            torch.cuda.synchronize(device)
    batch_seconds = time.monotonic() - started
    memory = [dict(device=str(device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device)) for device in policy.cuda_devices]
    results = []
    for row, reference, ref_seconds in zip(requests, references, reference_seconds):
        generated, mask = row['generated'], row['mask']
        batching = dict(batch_index=batch_index, actual_batch_size=len(requests),
            padded_prompt_tokens=width, pad_tokens=width-len(row['prompt']),
            padded_kv_token_budget=_batch_token_slots(requests), session_id=row['session_id'],
            prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
            generation_seconds=generation_seconds, reference_seconds=ref_seconds,
            all_reference_seconds=sum(reference_seconds), batch_seconds=batch_seconds,
            batch_decode_steps=steps, batch_generated_tokens=sum(len(value['generated']) for value in requests),
            timing_unit='Batch wall times overlap across its returned action rows; do not sum rows as GPU wall time')
        results.append(dict(prompt_token_ids=row['prompt'], completion_token_ids=generated,
            behavior_logprobs=row['old'], reference_logprobs=reference, loss_mask=mask,
            raw_text=policy.tokenizer.decode(generated, skip_special_tokens=False,
                clean_up_tokenization_spaces=False),
            finish_reason=row['reason'], seed=row['seed'], thinking_budget=row['thinking_budget'],
            thought_tokens=row['thought_count'], forced_control_tokens=mask.count(0),
            sampling=dict(temperature=1.0, top_p=1.0, top_k=0), prefill_chunk_size=256,
            reference_scoring='actual-independent-unpadded-chunked-backbone-selected-head'
                if row['compute_reference'] else 'not-computed-nonfocal-action',
            probability_execution_sha256=digest(policy.probability_profile),
            reference_forward_performed=row['compute_reference'], seconds=batch_seconds,
            prompt_tokens=len(row['prompt']), completion_tokens=len(generated),
            execution=dict(policy.execution, batching=batching,
                reference_scope='current-sampled-focal-action' if row['compute_reference'] else 'not-computed-nonfocal-action'),
            batching=batching, sampling_device=str(row['generator'].device) if row['generator'] is not None else None,
            prefill_cache=dict(enabled=False, hit=False, reused_tokens=0,
                new_prefill_tokens=len(row['prompt']), reset_reason='fresh-private-batched-action-cache',
                prefill_seconds=prefill_seconds), memory_by_device=memory,
            peak_memory_scope='maximum_of_batch_model_cuda_devices',
            peak_allocated_bytes=max(value['peak_allocated_bytes'] for value in memory),
            peak_reserved_bytes=max(value['peak_reserved_bytes'] for value in memory), **policy.identity))
    print(json.dumps(dict(phase='batched_action_completed', batch_index=batch_index,
        actual_batch_size=len(requests), padded_prompt_tokens=width,
        generated_tokens=sum(len(row['generated']) for row in requests),
        prefill_seconds=prefill_seconds, decode_seconds=decode_seconds,
        reference_seconds=sum(reference_seconds), batch_seconds=batch_seconds)), flush=True)
    return results


@dataclass
class _Queued:
    row: dict
    future: object
    enqueued_at: float


class BatchedPolicyService:
    """Async request queue with exactly one owner of the live CUDA model."""
    def __init__(self, policy, *, max_batch_size=2, wait_ms=50, max_batch_tokens=32768):
        if type(wait_ms) not in (int, float) or not math.isfinite(wait_ms) or wait_ms < 0:
            raise TrainingError('Batch collection delay must be nonnegative')
        self.policy = policy
        self.max_batch_size, self.max_batch_tokens = max_batch_size, max_batch_tokens
        self.wait_seconds = wait_ms / 1000
        configure_batched_policy(policy, max_batch_size=max_batch_size,
                                 max_batch_tokens=max_batch_tokens)
        self._queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='vfarl-batch-owner')
        self._task, self._failure, self._closed = None, None, False
        self._backlog = deque()
        self.batch_histogram = Counter()
        self.batch_count = 0

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def clear_generation_cache(self, *, reason='explicit-reset'):
        # There is no cross-action or cross-session cache in this profile.
        return None

    def sample(self, *args, **kwargs):
        raise TrainingError('Batched policy service requires await sample_async(...)')

    async def sample_async(self, prompt_token_ids, **kwargs):
        if self._closed:
            raise TrainingError('Batched policy service is closed')
        if self._failure is not None:
            raise TrainingError('The batch owner failed; preserve prior behavior') from self._failure
        row = _request(self.policy, prompt_token_ids, **kwargs)
        if _batch_token_slots([row]) > self.max_batch_tokens:
            raise TrainingError('One complete action exceeds the declared batch KV budget')
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._queue.put_nowait(_Queued(row, future, time.monotonic()))
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        return await future

    async def _run(self):
        closing = False
        while not closing:
            first = self._backlog.popleft() if self._backlog else await self._queue.get()
            if first is None:
                self._queue.task_done()
                break
            if first.future.cancelled():
                self._queue.task_done()
                continue
            batch = [first]
            deadline = time.monotonic() + self.wait_seconds
            while len(batch) < self.max_batch_size:
                try:
                    if self._backlog:
                        item = self._backlog.popleft()
                    elif not self._queue.empty():
                        item = self._queue.get_nowait()
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        item = await asyncio.wait_for(self._queue.get(), remaining)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    self._queue.task_done()
                    closing = True
                    break
                if item.future.cancelled():
                    self._queue.task_done()
                    continue
                candidate = [value.row for value in batch] + [item.row]
                if (_batch_token_slots(candidate) > self.max_batch_tokens
                        or _batch_span(candidate) > self.policy.cfg['max_sequence_length']):
                    self._backlog.appendleft(item)
                    break
                batch.append(item)
            dispatched_at = time.monotonic()
            self.batch_count += 1
            self.batch_histogram[len(batch)] += 1
            try:
                loop = asyncio.get_running_loop()
                values = await loop.run_in_executor(self._executor,
                    lambda: sample_batch(self.policy, [value.row for value in batch],
                        max_batch_tokens=self.max_batch_tokens, batch_index=self.batch_count))
                if len(values) != len(batch):
                    raise TrainingError('Batch owner returned a different number of actions')
                for queued, value in zip(batch, values):
                    value['batching']['queue_wait_seconds'] = dispatched_at - queued.enqueued_at
                    value['batching']['request_elapsed_seconds'] = time.monotonic() - queued.enqueued_at
                    if not queued.future.done():
                        queued.future.set_result(value)
            except BaseException as error:
                self._failure = error
                for queued in batch:
                    if not queued.future.done():
                        queued.future.set_exception(error)
                while self._backlog or not self._queue.empty():
                    queued = self._backlog.popleft() if self._backlog else self._queue.get_nowait()
                    if queued is not None and not queued.future.done():
                        queued.future.set_exception(error)
                    self._queue.task_done()
                return
            finally:
                for _ in batch:
                    self._queue.task_done()

    async def drain(self):
        await self._queue.join()
        if self._failure is not None:
            raise TrainingError('The batch owner failed; preserve prior behavior') from self._failure

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        if self._task is not None and not self._task.done():
            self._queue.put_nowait(None)
            await self._task
        self._executor.shutdown(wait=True, cancel_futures=False)
        print(json.dumps(dict(phase='batched_service_closed', batches=self.batch_count,
            batch_histogram=dict(self.batch_histogram))), flush=True)
