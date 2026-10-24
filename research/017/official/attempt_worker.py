"""Run one approved official public development task through official Evaluator.

Only a one-task JSONL is loaded. Official task fields, snapshot reconstruction,
Agent execution, and verification are delegated to the installed official runtime.
Raw inference capture is copied from the exercised P015 worker; evaluation
trajectories remain evaluation-only and must never become training samples.
"""
from __future__ import annotations

import asyncio
import contextvars
import functools
import importlib.metadata
import inspect
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
import uuid


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def result_record(result: object) -> dict:
    return {key: getattr(result, key, None) for key in
            ('exit_code', 'stdout', 'stderr', 'duration_seconds', 'timed_out', 'status')}


def terminal_consumer_close_evidence(attempt: Path, rid: str) -> dict:
    """Narrow, persisted-record proof; does not make a trajectory trainable."""
    evidence = {'model_request_id': rid, 'verified': False}
    try:
        records = {name: [json.loads(line) for line in (attempt/('raw-'+name+'.jsonl')).read_text().splitlines()]
                   for name in ('model', 'wire', 'adk-events')}
        def one(channel, kind):
            matches = [r for r in records[channel] if r.get('model_request_id') == rid and r['kind'] == kind]
            if len(matches) != 1:
                raise ValueError('Expected exactly one '+kind)
            return matches[0]
        mr, response, end = [one('model', kind) for kind in ('model_request', 'model_response', 'model_end')]
        wr, answer, we = [one('wire', kind) for kind in ('wire_request', 'wire_response', 'wire_end')]
        event = one('adk-events', 'adk_event')
        if (mr.get('stream') is not False or end.get('completed') is not False
                or we.get('completed') is not True or len({r['wire_id'] for r in (wr, answer, we)}) != 1):
            raise ValueError('Not one completed non-stream wire call followed by consumer close')
        if any(r['kind'] == 'model_request' and r['timestamp'] > mr['timestamp'] for r in records['model']):
            raise ValueError('Interrupted request is not the terminal request')
        model, adk = response['response'], event['event']
        for value in (model, adk):
            if (str(value.get('finish_reason')).upper() != 'STOP' or value.get('partial') is not False
                    or value.get('interrupted') or value.get('error_code') or value.get('error_message')):
                raise ValueError('Model/ADK response is incomplete or failed')
        if model['content'] != adk['content'] or adk.get('author') in ('user', 'harness', 'system', 'tool'):
            raise ValueError('ADK terminal event differs from the captured model response')
        parts = model['content']['parts']
        if any(p.get('function_call') or p.get('function_response') for p in parts):
            raise ValueError('Terminal response is not a plain final message')
        visible = ''.join(p.get('text') or '' for p in parts if not p.get('thought'))
        thoughts = ''.join(p.get('text') or '' for p in parts if p.get('thought'))
        choices = answer['response']['choices']
        if len(choices) != 1:
            raise ValueError('Ambiguous wire choices')
        choice = choices[0]; message = choice['message']
        if (choice.get('finish_reason') != 'stop' or not visible
                or message.get('content') != visible or (message.get('reasoning_content') or '') != thoughts
                or message.get('tool_calls') or message.get('function_call')):
            raise ValueError('Wire completion does not reproduce the complete final content/thoughts')
        previous, reply = wr['request']['messages'][-2:]
        calls = previous.get('tool_calls') or []
        if len(calls) != 1 or calls[0]['function']['name'] != 'submit_patch' or reply.get('role') != 'tool':
            raise ValueError('Final request does not follow one submit_patch result')
        call = calls[0]; cid = call['id']; result = json.loads(reply['content'])
        if reply.get('tool_call_id') != cid or result.get('status') != 'ok':
            raise ValueError('submit_patch reply is missing or unsuccessful')
        submit_calls, submit_replies = [], []
        for record in records['adk-events']:
            if record['timestamp'] >= wr['timestamp']:
                continue
            for part in (record['event'].get('content') or {}).get('parts') or []:
                for key, matches in (('function_call', submit_calls), ('function_response', submit_replies)):
                    value = part.get(key)
                    if value and value.get('id') == cid and value.get('name') == 'submit_patch':
                        matches.append(value)
        if len(submit_calls) != 1 or len(submit_replies) != 1 or submit_replies[0]['response'] != result:
            raise ValueError('Final-request submit result lacks matching actual ADK call/reply')
        arguments = call['function']['arguments']
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if submit_calls[0]['args'] != arguments:
            raise ValueError('submit_patch argument mismatch')
        times = [mr['timestamp'], wr['timestamp'], answer['timestamp'], we['timestamp'],
                 response['timestamp'], event['timestamp'], end['timestamp']]
        if times != sorted(times):
            raise ValueError('Terminal request/response/event ordering is inconsistent')
        evidence.update(verified=True, wire_id=wr['wire_id'], wire_completed=True,
            wire_finish_reason=choice['finish_reason'], model_finish_reason=model['finish_reason'],
            adk_event_id=adk['id'], submit_call_id=cid, submit_result=result,
            content_sha256=hashlib.sha256(visible.encode()).hexdigest(),
            reasoning_sha256=hashlib.sha256(thoughts.encode()).hexdigest(),
            timestamp_order=times, scope='normal terminal consumer close; encoding eligibility remains separate')
    except Exception as exc:
        evidence['reason'] = str(exc)
    return evidence


class TrajectoryCapture:
    """Append raw inference records without changing the Agent or its messages."""

    def __init__(self, attempt: Path):
        self.attempt = attempt
        self.counts = {}
        self.errors = []
        self.hooks = []
        self.restores = []
        self.request_id = contextvars.ContextVar('p015_model_request_id', default=None)
        self.versions = {}
        for name in ('google-adk', 'google-genai', 'adk-submission', 'adk-eval-core', 'litellm', 'pydantic'):
            try:
                self.versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                self.versions[name] = None

    def clean(self, value):
        if hasattr(value, 'model_dump'):
            value = value.model_dump(mode='json', exclude_none=False)
        if isinstance(value, dict):
            return {str(k): ('[REDACTED]' if str(k).lower().replace('-', '_') in
                    ('api_key', 'authorization', 'proxy_authorization', 'headers',
                     'extra_headers', 'default_headers', 'cookies', 'access_token')
                    else self.clean(v)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.clean(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError('Non-JSON inference value: ' + type(value).__name__)

    def emit(self, channel, kind, **payload):
        try:
            record = self.clean(dict(kind=kind, timestamp=time.time(),
                                     model_request_id=self.request_id.get(), **payload))
            # An open/write/flush per record survives worker interruption and never
            # retains references to mutable ADK request/response objects.
            with (self.attempt / ('raw-' + channel + '.jsonl')).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
            self.counts[kind] = self.counts.get(kind, 0) + 1
        except Exception as exc:
            self.errors.append(dict(channel=channel, kind=kind, error_type=type(exc).__name__, error=str(exc)))

    def patch(self, cls, name, replacement):
        original = getattr(cls, name)
        self.hooks.append(dict(target=cls.__module__ + '.' + cls.__name__ + '.' + name,
                               signature=str(inspect.signature(original)),
                               source_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest()))
        self.restores.append((cls, name, original))
        setattr(cls, name, replacement(original))

    def install(self):
        from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
        from adk_eval_core.tracing import SessionTrace
        capture = self

        def model_wrapper(original):
            @functools.wraps(original)
            async def generate(model, llm_request, stream=False):
                rid = uuid.uuid4().hex
                token = capture.request_id.set(rid)
                completed = False
                termination = None
                capture.emit('model', 'model_request', request=llm_request, stream=stream)
                try:
                    async for response in original(model, llm_request, stream=stream):
                        capture.emit('model', 'model_response', response=response)
                        yield response
                    completed = True
                except BaseException as exc:
                    termination = 'consumer_close' if isinstance(exc, GeneratorExit) else type(exc).__name__
                    raise
                finally:
                    capture.emit('model', 'model_end', completed=completed, termination_reason=termination)
                    if not completed:
                        capture.errors.append(dict(kind='model_interrupted', model_request_id=rid,
                                                   termination_reason=termination))
                    capture.request_id.reset(token)
            return generate

        def wire_wrapper(original):
            @functools.wraps(original)
            async def complete(client, model, messages, tools, **kwargs):
                wire_id = uuid.uuid4().hex
                capture.emit('wire', 'wire_request', wire_id=wire_id,
                             request=dict(model=model, messages=messages, tools=tools, **kwargs))
                try:
                    response = await original(client, model, messages, tools, **kwargs)
                    if kwargs.get('stream'):
                        async def chunks():
                            completed = False
                            try:
                                async for chunk in response:
                                    capture.emit('wire', 'wire_chunk', wire_id=wire_id, response=chunk)
                                    yield chunk
                                completed = True
                            finally:
                                capture.emit('wire', 'wire_end', wire_id=wire_id, completed=completed)
                                if not completed:
                                    capture.errors.append(dict(kind='wire_stream_interrupted', wire_id=wire_id))
                        return chunks()
                    capture.emit('wire', 'wire_response', wire_id=wire_id, response=response)
                    capture.emit('wire', 'wire_end', wire_id=wire_id, completed=True)
                    return response
                except BaseException as exc:
                    capture.emit('wire', 'wire_error', wire_id=wire_id, error_type=type(exc).__name__)
                    capture.errors.append(dict(kind='wire_error', wire_id=wire_id, error_type=type(exc).__name__))
                    raise
            return complete

        def event_wrapper(original):
            @functools.wraps(original)
            def record(trace, event):
                capture.emit('adk-events', 'adk_event', event=event)
                return original(trace, event)
            return record

        self.patch(LiteLlm, 'generate_content_async', model_wrapper)
        self.patch(LiteLLMClient, 'acompletion', wire_wrapper)
        self.patch(SessionTrace, 'record_event', event_wrapper)

    def finish(self, trace_path, agent_error=None):
        reasons = list(self.errors)
        classifications = []
        if not agent_error:
            for reason in list(reasons):
                if reason.get('kind') == 'model_interrupted' and reason.get('termination_reason') == 'consumer_close':
                    evidence = terminal_consumer_close_evidence(self.attempt, reason['model_request_id'])
                    classifications.append(evidence)
                    if evidence['verified']:
                        reasons.remove(reason)
        for name in ('model_request', 'model_response', 'wire_request', 'adk_event'):
            if not self.counts.get(name):
                reasons.append(dict(kind='missing_capture', record_kind=name))
        for first, last in (('model_request', 'model_end'), ('wire_request', 'wire_end')):
            if self.counts.get(first, 0) != self.counts.get(last, 0):
                reasons.append(dict(kind='incomplete_capture', request_kind=first))
        if not trace_path or not Path(trace_path).is_file() or not Path(trace_path).stat().st_size:
            reasons.append(dict(kind='official_trace_not_saved'))
        if agent_error:
            reasons.append(dict(kind='agent_error', error=agent_error))
        result = dict(training_trace_usable=not reasons, reasons=reasons,
                      raw_capture_errors=list(self.errors), request_classifications=classifications,
                      counts=self.counts, hooks=self.hooks, versions=self.versions,
                      scope='Raw ADK request/response, post-conversion LiteLLM request/response, and raw ADK events; judge outcome remains independent')
        atomic_json(self.attempt / 'trajectory-capture.json', result)
        return result

    def restore(self):
        for cls, name, original in reversed(self.restores):
            setattr(cls, name, original)
        self.restores.clear()



def execute(spec: dict) -> int:
    entry = spec['entry']
    attempt = Path(entry['results_dir']).resolve()
    attempt.mkdir(parents=True, exist_ok=True)
    scratch = Path(spec.get('scratch', attempt / 'scratch')).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    status = dict(entry, status='initializing', started_at=time.time(), pid=os.getpid())
    started = time.monotonic()
    evaluator = context = result = capture = None
    native_agent = {}
    native_verification = {'called': False, 'returned': False}
    restores = []
    handler = logging.FileHandler(attempt / 'harness.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    logging.getLogger().addHandler(handler)

    def phase(name):
        status.update(phase=name, phase_at=time.time())
        atomic_json(attempt / 'attempt_status.json', status)
        print(json.dumps({'attempt_id': entry.get('attempt_id'), 'phase': name}), flush=True)

    def official_fields(value):
        return {key: getattr(value, key, None) for key in (
            'instance_id', 'repo', 'resolved', 'test_exit_code', 'duration_seconds',
            'tool_calls', 'total_llm_calls', 'error', 'trace_json_path', 'task_index')}

    try:
        phase('load_single_approved_task')
        subset = Path(spec['tasks_subset']).resolve()
        data = Path(spec['data_dir']).resolve()
        if subset == (data / 'tasks.jsonl').resolve():
            raise ValueError('tasks_subset must be a separate approved one-task JSONL')
        # Check the line count without parsing or exposing any unapproved body.
        # The coordinator supplies exactly one selected public development task.
        with subset.open(encoding='utf-8') as stream:
            selected_line = None
            for line in stream:
                if line.strip():
                    if selected_line is not None:
                        raise ValueError('Each official worker requires exactly one approved task')
                    selected_line = line
        if selected_line is None:
            raise ValueError('Empty approved task subset')
        raw_task = json.loads(selected_line)
        if raw_task.get('instance_id') != entry['task_id']:
            raise ValueError('Approved task subset identity differs from this attempt')
        if entry.get('split', 'dev') != 'dev':
            raise ValueError('Only the approved public development split may run')

        phase('load_official_runtime')
        import litellm
        from adk_submission import ModelRegistry, discover_adapters
        from adk_submission.resolvers.generation import install_litellm_reasoning_patch
        from google.adk.agents.context_cache_config import ContextCacheConfig
        from google.adk.apps._configs import EventsCompactionConfig
        from google.adk.models.lite_llm import LiteLlm
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
        from swegemma.context import SwegemmaContext
        import swegemma.evaluate as evaluation_module
        from swegemma.evaluate import Evaluator
        from swegemma.harness.verification import save_trace_artifact
        from swegemma.models import load_tasks
        from swegemma.results import append_task_result

        tasks = load_tasks(subset)
        if len(tasks) != 1 or tasks[0].instance_id != entry['task_id']:
            raise ValueError('Official loader did not retain the approved single task')
        task = tasks[0]
        # Do not silently discard a judge field when a wheel/schema is stale.
        # F2P/P2P may be normalized by the official Task constructor itself.
        for key in ('patch', 'test_patch', 'test_script'):
            if raw_task.get(key) and getattr(task, key, None) != raw_task[key]:
                raise ValueError('Official Task schema failed to preserve ' + key)
        judge_fields = {
            key: {'present': key in raw_task, 'loaded': hasattr(task, key),
                  'sha256': hashlib.sha256(json.dumps(raw_task.get(key), ensure_ascii=False,
                    sort_keys=True).encode()).hexdigest()}
            for key in ('patch', 'test_patch', 'test_script', 'FAIL_TO_PASS', 'PASS_TO_PASS')
        }
        atomic_json(attempt / 'official-task-source.json', {
            'instance_id': task.instance_id, 'repo': task.repo, 'base_commit': task.base_commit,
            'tasks_subset': str(subset), 'subset_sha256': hashlib.sha256(subset.read_bytes()).hexdigest(),
            'loaded_task_count': 1, 'judge_fields': judge_fields,
            'data_dir': str(data), 'evaluation_only': True,
            'secret_hydration': 'disabled; selected public fields must be self-contained',
        })
        install_litellm_reasoning_patch()
        capture = TrajectoryCapture(attempt)
        try:
            capture.install()
        except Exception as exc:
            capture.errors.append(dict(kind='hook_installation_error',
                error_type=type(exc).__name__, error=str(exc)))
            capture.restore()
        atomic_json(attempt / 'trajectory-capture-start.json',
                    dict(hooks=capture.hooks, versions=capture.versions, errors=capture.errors))
        litellm.drop_params = True
        models = ModelRegistry()
        models.register(spec['model_alias'], LiteLlm(
            model='openai/' + spec['served_model'], api_base=spec['base_url'], api_key='EMPTY'))
        adapters = discover_adapters(spec['agent_dir'], adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
        for adapter_name in adapters.adapters:
            for alias in (adapter_name, 'adapter:' + adapter_name):
                models.register(alias, LiteLlm(model='openai/' + adapter_name,
                    api_base=spec['base_url'], api_key='EMPTY'))
        limits, generation_constraints = build_submission_limits()
        budgets = spec['budgets']
        if (float(budgets['max_time_minutes']) != 5 or int(budgets['max_tool_calls']) != 28
                or int(budgets['max_turns']) != 40):
            raise ValueError('Official weight comparison must retain P015 runtime-v2 Agent budgets')
        verification_timeout = int(spec.get('verification_timeout_seconds', 300))
        if verification_timeout != 300:
            raise ValueError('Official verification timeout must remain 300 seconds')
        config = EvalConfig(
            tasks_path=subset, snapshots_dir=data / 'snapshots', results_dir=attempt,
            submission_dir=Path(spec['agent_dir']).resolve(), models=models, sandbox='subprocess',
            timeout_seconds=int(budgets.get('timeout_seconds', 240)),
            verification_timeout_seconds=verification_timeout,
            max_tool_calls=int(budgets['max_tool_calls']),
            max_time_minutes=float(budgets['max_time_minutes']), max_turns=int(budgets['max_turns']),
            limits=limits, generation_constraints=generation_constraints, adapter_manifest=adapters,
            context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
            events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                           token_threshold=14336, event_retention_size=5),
            graph_dir=str(data / 'graphs'), embeddings_dir=str(data / 'embeddings'),
            wheels_dir=data / 'wheels', verbose=False, display_mode='quiet', enable_sandbox_testing=True)
        evaluator = Evaluator(config)
        # The official public task is self-contained. Do not discover/read sibling
        # secret bundles if a public task field or snapshot happens to be missing.
        if hasattr(evaluator, '_secret_hydration_cache'):
            evaluator._secret_hydration_cache = ({}, {}, None)
        if evaluator.docker.sandboxes:
            raise RuntimeError('New official Evaluator unexpectedly owns existing sandboxes')
        evaluator.docker.base_dir = scratch / 'sandboxes'
        evaluator.docker.base_dir.mkdir(parents=True, exist_ok=False)
        context = SwegemmaContext(docker_manager=evaluator.docker, task=task,
            graph_dir=config.graph_dir, embeddings_dir=config.embeddings_dir,
            budget=config.budget, harness=config.harness)

        # Passive wrappers preserve both the runner's error (which the official
        # successful judge may clear) and the native verification return value.
        original_agent = evaluator._run_agent_sandbox
        @functools.wraps(original_agent)
        async def observed_agent(*args, **kwargs):
            phase('official_agent')
            outcome = await original_agent(*args, **kwargs)
            patch, error, trace = outcome
            native_agent.update(returned=True, error=error,
                                agent_patch_size=len((patch or '').encode()))
            atomic_json(attempt / 'official-agent-result.json', native_agent)
            atomic_text(attempt / 'agent.patch', patch or '')
            return outcome
        evaluator._run_agent_sandbox = observed_agent
        restores.append((evaluator, '_run_agent_sandbox', original_agent))
        original_verify = evaluation_module.verify_task
        @functools.wraps(original_verify)
        async def observed_verification(*args, **kwargs):
            phase('official_verification')
            native_verification.update(called=True, started_at=time.time())
            atomic_json(attempt / 'official-verification.json', native_verification)
            try:
                value = await original_verify(*args, **kwargs)
                native_verification.update(returned=True, finished_at=time.time(), **official_fields(value))
                atomic_json(attempt / 'official-verification.json', native_verification)
                return value
            except BaseException as exc:
                native_verification.update(finished_at=time.time(), error_type=type(exc).__name__, error=str(exc))
                atomic_json(attempt / 'official-verification.json', native_verification)
                raise
        evaluation_module.verify_task = observed_verification
        restores.append((evaluation_module, 'verify_task', original_verify))

        status['status'] = 'running'
        phase('official_evaluate_task')
        result = asyncio.run(evaluator.evaluate_task(task=task,
            task_index=int(entry.get('ordinal', 1)), total_tasks=int(spec.get('total_tasks', 1)), context=context))
        phase('archive')
        original = official_fields(result)
        atomic_json(attempt / 'official-result.json', original)
        safe_id = task.instance_id.replace('/', '__')
        trace_path = getattr(result, 'trace_json_path', None)
        if not trace_path or not Path(trace_path).is_file():
            trace_path = save_trace_artifact(getattr(result, 'trace', None), attempt, task.instance_id)
        atomic_text(attempt / 'agent.patch', result.agent_patch or '')
        atomic_text(attempt / 'patches' / (safe_id + '.patch'), result.agent_patch or '')
        atomic_text(attempt / 'test_outputs' / (safe_id + '.log'), result.test_output or '')
        append_task_result(result, attempt, [result])
        captured = capture.finish(trace_path, native_agent.get('error') or original['error'])
        capture_complete = captured['training_trace_usable']
        captured.update(training_trace_usable=False, capture_complete=capture_complete,
                        evaluation_only=True, scope=captured['scope'] + '; evaluation data must not enter training')
        atomic_json(attempt / 'trajectory-capture.json', captured)
        # A timeout/turn limit/no submitted patch is an observed Agent outcome,
        # including when the native Evaluator returns early without judge tests.
        # Reserve top-level error for evidence/asset/infrastructure failures.
        native_error = str(original['error'] or '')
        infrastructure = next((prefix for prefix in (
            'Snapshot file not found:', 'Missing test specification',
            'Failed to apply test_patch:', 'Evaluation error:') if native_error.startswith(prefix)), None)
        judge_valid = type(result.resolved) is bool and infrastructure is None
        payload = dict(entry, **{key: value for key, value in original.items() if key not in ('error', 'trace_json_path')},
            agent_error=original['error'], runner_agent_error=native_agent.get('error'),
            official_result_returned=True, official_verification_called=native_verification['called'],
            official_verification_returned=native_verification['returned'], judge_valid=judge_valid,
            error=(native_error or 'Official result is not Boolean') if not judge_valid else None,
            agent_patch_size=len((result.agent_patch or '').encode()), trace_json_path=trace_path,
            trace_present=bool(trace_path and Path(trace_path).is_file()),
            p2p_pass=None, p2p_reporting='Native official aggregate verdict; no invented separate P2P result',
            training_trace_usable=False, evaluation_only=True, capture_complete=capture_complete,
            training_trace_reasons=captured['reasons'], trajectory_capture_path=str(attempt / 'trajectory-capture.json'))
        atomic_json(attempt / 'result.json', payload)
        status.update(status='completed' if judge_valid else 'official_environment_error',
                      trace_present=payload['trace_present'], resolved=result.resolved)
        return 0 if judge_valid else 1
    except BaseException as exc:
        status.update(status='error', error_type=type(exc).__name__, error=str(exc))
        atomic_text(attempt / 'exception.txt', traceback.format_exc())
        if not (attempt / 'result.json').exists():
            atomic_json(attempt / 'result.json', dict(entry, instance_id=entry['task_id'],
                resolved=None, judge_valid=False, official_result_returned=result is not None,
                agent_error=getattr(result, 'error', None), runner_agent_error=native_agent.get('error'),
                error_type=type(exc).__name__, error=str(exc), training_trace_usable=False, evaluation_only=True))
        if capture is not None and not (attempt / 'trajectory-capture.json').exists():
            captured = capture.finish(None, type(exc).__name__)
            captured.update(training_trace_usable=False, evaluation_only=True)
            atomic_json(attempt / 'trajectory-capture.json', captured)
        return 1
    finally:
        if context is not None:
            fields = ('task_start_time', 'agent_start_time', 'agent_end_time', 'agent_elapsed_seconds',
                      'task_elapsed_seconds', 'patch_submitted', 'tool_calls_used', 'llm_calls_used')
            atomic_json(attempt / 'context_metrics.json', {key: getattr(context, key, None) for key in fields})
            if result is None and getattr(context, 'submitted_patch', None) is not None:
                atomic_text(attempt / 'submitted_patch_before_exception.patch', context.submitted_patch)
        for owner, name, original in reversed(restores):
            setattr(owner, name, original)
        if capture is not None:
            capture.restore()
        if evaluator is not None:
            try:
                evaluator.docker.cleanup_all()
            except Exception:
                atomic_text(attempt / 'sandbox_cleanup_error.txt', traceback.format_exc())
        status.update(finished_at=time.time(), outer_seconds=time.monotonic() - started)
        atomic_json(attempt / 'attempt_status.json', status)
        logging.getLogger().removeHandler(handler)
        handler.close()


if __name__ == '__main__':
    raise SystemExit(execute(json.loads(Path(sys.argv[1]).read_text())))
