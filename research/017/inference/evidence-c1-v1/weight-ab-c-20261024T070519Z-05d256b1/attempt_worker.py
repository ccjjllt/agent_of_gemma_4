"""Run one external development task with the official Agent runner and a fresh judge.

The Agent receives only the original issue and buggy snapshot. Dataset-provided
validation patches and existing F2P/P2P nodes stay in the host judge workspace.
"""
from __future__ import annotations

import asyncio
import copy
import contextvars
import functools
import importlib.metadata
import inspect
import tarfile
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
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


def workspace_snapshot(source: Path, scratch: Path) -> Path:
    """Flatten the CPU material's documented repo/ or workspace/ envelope.

    The official extraction routine expects .git and source at archive root.
    Preserve every original file and link; never add criterion/reference files.
    """
    with tarfile.open(source, 'r:*') as archive:
        members = archive.getmembers()
        names = [m.name.removeprefix('./') for m in members]
        roots = {name.split('/')[0] for name in names if name}
        if roots not in ({'repo'}, {'workspace'}):
            return source
        prefix = next(iter(roots)) + '/'
        target = scratch / 'workspace-snapshot.tar.gz'
        with tarfile.open(target, 'w:gz') as output:
            for member, name in zip(members, names):
                if name.rstrip('/') == prefix.rstrip('/'):
                    continue
                if not name.startswith(prefix):
                    raise RuntimeError('Snapshot archive has inconsistent envelope')
                item = copy.copy(member)
                item.name = name[len(prefix):]
                if item.islnk() and item.linkname.startswith(prefix):
                    item.linkname = item.linkname[len(prefix):]
                output.addfile(item, archive.extractfile(member) if member.isfile() else None)
        return target


def execute(spec: dict) -> int:
    entry = spec['entry']
    attempt = Path(entry['results_dir']).resolve()
    attempt.mkdir(parents=True, exist_ok=True)
    safe_attempt_id = ''.join(ch if ch.isalnum() or ch in '-_.' else '_'
                              for ch in str(entry.get('attempt_id', attempt.name)))
    scratch = Path(spec.get('scratch', Path('/tmp') / 'p015-attempts' / safe_attempt_id)).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    status = dict(entry, status='running', pid=os.getpid(), started_at=time.time())
    start = time.monotonic()
    manager = judge_manager = context = None
    capture = None
    capture_result = dict(training_trace_usable=False,
                          training_trace_reasons=[dict(kind='agent_not_completed')])
    handler = logging.FileHandler(attempt / 'harness.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    logging.getLogger().addHandler(handler)

    def phase(name: str) -> None:
        status.update(phase=name, phase_at=time.time())
        atomic_json(attempt / 'attempt_status.json', status)
        print(json.dumps({'attempt_id': entry.get('attempt_id'), 'phase': name}), flush=True)

    try:
        phase('load_official_runtime')
        import litellm
        from adk_submission import ModelRegistry, discover_adapters
        from adk_submission.resolvers.generation import install_litellm_reasoning_patch
        from google.adk.agents.context_cache_config import ContextCacheConfig
        from google.adk.apps._configs import EventsCompactionConfig
        from google.adk.models.lite_llm import LiteLlm
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
        from swegemma.context import SwegemmaContext
        from swegemma.harness import agent_runner
        from swegemma.harness.container_setup import extract_snapshot, apply_patch_in_container
        from swegemma.harness.verification import save_trace_artifact, strip_protected_files_from_patch
        from swegemma.models import Task
        from swegemma.sandbox import SubprocessManager

        environment = spec['judge'].get('environment', {})
        pytest_args = environment.get('pytest_args', [])
        if not isinstance(pytest_args, list) or not all(isinstance(arg, str) for arg in pytest_args):
            raise RuntimeError('Frozen pytest_args must be a list of strings')
        pytest_addopts_export = ('export PYTEST_ADDOPTS=' + shlex.quote(shlex.join(pytest_args)) + '; '
                                 if pytest_args else '')

        class SourceManager(SubprocessManager):
            def exec(self, sandbox_id, command, *, timeout=None):
                prefix = ('export PYTHONPATH=/workspace/src:/workspace; '
                          'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; ')
                prefix += ('export PYTEST_PLUGINS=' + shlex.quote(
                    ','.join(environment.get('pytest_plugins', []))) + '; ')
                return super().exec(sandbox_id, prefix + pytest_addopts_export + command, timeout=timeout)

            def stop(self, sandbox_id):
                if sandbox_id in self.sandboxes:
                    try:
                        state = self.exec(sandbox_id, 'git status --short', timeout=10)
                        atomic_text(attempt / 'agent-final-status.txt',
                                    (state.stdout or '') + (state.stderr or ''))
                    except Exception:
                        atomic_text(attempt / 'agent-final-status-error.txt', traceback.format_exc())
                super().stop(sandbox_id)

        def prepared_dependencies(docker, sandbox_id, *_args, **_kwargs):
            if not environment.get('requirements'):
                return
            panel = Path(spec['panel_root']).resolve()
            wheelhouse = (panel / environment['wheelhouse']).resolve()
            requirements = (panel / environment['requirements']).resolve()
            if not wheelhouse.is_relative_to(panel) or not requirements.is_relative_to(panel):
                raise RuntimeError('Dependency path escapes the frozen material root')
            if not requirements.is_file():
                raise RuntimeError('CPU-prepared requirements missing')
            if not any(line.strip() and not line.lstrip().startswith('#') for line in requirements.read_text().splitlines()):
                return
            if not wheelhouse.is_dir():
                raise RuntimeError('CPU-prepared per-repository wheelhouse missing')
            for wheel in sorted(wheelhouse.glob('*.whl')):
                docker.copy_to(sandbox_id, wheel, '/wheels/' + wheel.name)
            docker.copy_to(sandbox_id, requirements, '/tmp/requirements.txt')
            result = docker.exec(sandbox_id,
                'python3 -m pip install --no-index --no-deps --find-links=/wheels '
                '-r /tmp/requirements.txt', timeout=120)
            if result.exit_code:
                raise RuntimeError('Offline dependency install failed: ' +
                    (result.stdout or '')[-3000:] + (result.stderr or '')[-3000:])

        def prepared_editable(docker, sandbox_id):
            if not environment.get('install_editable', False):
                return
            result = docker.exec(sandbox_id,
                'python3 -m pip install --no-index --no-build-isolation --no-deps '
                '--find-links=/wheels -e /workspace', timeout=90)
            if result.exit_code:
                raise RuntimeError('Editable source installation failed: ' +
                    (result.stdout or '')[-3000:] + (result.stderr or '')[-3000:])

        agent_runner.setup_container_wheels = prepared_dependencies
        agent_runner.install_editable_package = prepared_editable
        agent_runner.install_test_dependencies = lambda *_args, **_kwargs: None
        agent_runner.setup_workspace_test_config = lambda *_args, **_kwargs: None
        original_baseline = agent_runner.setup_baseline_commit

        def baseline_with_local_identity(docker, sandbox_id, commit_msg='baseline'):
            configured = docker.exec(sandbox_id,
                "git config user.email 'research@localhost' && "
                "git config user.name 'P015 research'", timeout=10)
            if configured.exit_code:
                raise RuntimeError('cannot configure snapshot Git identity')
            original_baseline(docker, sandbox_id, commit_msg)
            tagged = docker.exec(sandbox_id, 'git rev-parse _swegemma_baseline', timeout=10)
            if tagged.exit_code:
                raise RuntimeError('official baseline commit was not created')
        agent_runner.setup_baseline_commit = baseline_with_local_identity

        task_input = spec['task']
        task = Task(instance_id=task_input.get('instance_id', task_input.get('iid')),
                    repo=task_input['repo'],
                    base_commit=task_input.get('base_commit', task_input.get('basecommit')),
                    problem_statement=task_input.get('problem_statement', task_input.get('issue')))
        assert not task.patch and not task.test_patch and not task.FAIL_TO_PASS and not task.PASS_TO_PASS
        snapshot = Path(spec['snapshot']).resolve()
        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
        original_snapshot = snapshot
        snapshot = workspace_snapshot(snapshot, scratch)
        atomic_json(attempt / 'snapshot-envelope.json', {'source': str(original_snapshot), 'source_sha256': hashlib.sha256(original_snapshot.read_bytes()).hexdigest(), 'runtime_snapshot': str(snapshot), 'runtime_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest()})
        install_litellm_reasoning_patch()
        capture = TrajectoryCapture(attempt)
        try:
            capture.install()
        except Exception as exc:
            capture.errors.append(dict(kind='hook_installation_error',
                                       error_type=type(exc).__name__, error=str(exc)))
            capture.restore()
        # Record the actual runtime hook/version contract before the first call.
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
        config = EvalConfig(
            tasks_path=attempt / 'agent-task.json', snapshots_dir=snapshot.parent,
            results_dir=attempt, submission_dir=Path(spec['agent_dir']).resolve(),
            models=models, sandbox='subprocess',
            timeout_seconds=int(budgets.get('timeout_seconds', 240)),
            verification_timeout_seconds=int(spec['judge'].get('timeout_seconds', spec['judge'].get('timeout', 120))),
            max_tool_calls=int(budgets['max_tool_calls']),
            max_time_minutes=float(budgets['max_time_minutes']),
            max_turns=int(budgets['max_turns']),
            limits=limits, generation_constraints=generation_constraints, adapter_manifest=adapters,
            context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
            events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                           token_threshold=14336, event_retention_size=5),
            graph_dir=str(scratch / 'absent-graphs'), embeddings_dir=str(scratch / 'absent-embeddings'),
            verbose=False, display_mode='quiet', enable_sandbox_testing=True)
        # This metadata contains only the issue given to the Agent, with no criterion.
        atomic_json(attempt / 'agent-task.json', {
            'instance_id': task.instance_id, 'repo': task.repo,
            'base_commit': task.base_commit, 'problem_statement': task.problem_statement})
        manager = SourceManager(timeout_seconds=config.harness.command_timeout_seconds,
                                base_dir=scratch / 'agent-sandboxes', system_site_packages=True)
        manager.base_dir.mkdir(parents=True, exist_ok=True)
        context = SwegemmaContext(docker_manager=manager, task=task,
            graph_dir=config.graph_dir, embeddings_dir=config.embeddings_dir,
            budget=config.budget, harness=config.harness)
        phase('agent')
        patch, agent_error, trace = asyncio.run(agent_runner.run_agent_sandbox(
            docker=manager, config=config, task=task, snapshot_path=snapshot,
            task_index=int(entry.get('ordinal', 1)), total_tasks=int(spec.get('total_tasks', 1)),
            context=context))
        patch = patch or ''
        atomic_text(attempt / 'agent.patch', patch)
        scored_patch = strip_protected_files_from_patch(patch)
        atomic_text(attempt / 'scored.patch', scored_patch)
        trace_path = save_trace_artifact(trace, attempt, task.instance_id)
        captured = capture.finish(trace_path, agent_error)
        capture_result = dict(training_trace_usable=captured['training_trace_usable'],
                              training_trace_reasons=captured['reasons'],
                              trajectory_capture_path=str(attempt / 'trajectory-capture.json'))
        context_metrics = {key: getattr(context, key, None) for key in (
            'task_start_time', 'agent_start_time', 'agent_end_time', 'agent_elapsed_seconds',
            'task_elapsed_seconds', 'patch_submitted', 'tool_calls_used', 'llm_calls_used')}
        atomic_json(attempt / 'context_metrics.json', context_metrics)

        phase('fresh_judge')
        # SourceManager's final-status hook is Agent-specific, so use a separate
        # minimal source-prefix subclass for the independent judge workspace.
        class JudgeManager(SubprocessManager):
            def exec(self, sandbox_id, command, *, timeout=None):
                return super().exec(sandbox_id,
                    'export PYTHONPATH=/workspace/src:/workspace; '
                    'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; ' + command, timeout=timeout)
        judge_manager = JudgeManager(timeout_seconds=120,
            base_dir=scratch / 'judge-sandboxes', system_site_packages=True)
        judge_manager.base_dir.mkdir(parents=True, exist_ok=True)
        judge_id = judge_manager.start()
        extract_snapshot(judge_manager, judge_id, snapshot)
        head = judge_manager.exec(judge_id, 'git rev-parse HEAD', timeout=10)
        atomic_json(attempt / 'judge-source.json', {
            'expected_base_commit': task.base_commit, 'actual_base_commit': (head.stdout or '').strip(),
            'snapshot': str(snapshot), 'snapshot_sha256': hashlib.sha256(snapshot.read_bytes()).hexdigest()})
        expected = judge_manager.exec(judge_id, 'git rev-parse --verify ' +
            shlex.quote(task.base_commit + '^{commit}'), timeout=10)
        if (head.exit_code or expected.exit_code or
                (head.stdout or '').strip() != (expected.stdout or '').strip()):
            raise RuntimeError('judge snapshot base commit does not match frozen task')
        if scored_patch.strip():
            judge_manager.copy_to(judge_id, attempt / 'scored.patch', '/tmp/agent.patch')
            code, out, err = apply_patch_in_container(judge_manager, judge_id, '/tmp/agent.patch')
            atomic_json(attempt / 'patch-application.json', {'exit_code': code, 'stdout': out, 'stderr': err})
            if code:
                payload = dict(entry, **capture_result, instance_id=task.instance_id, repo=task.repo, resolved=False,
                    agent_patch_size=len(patch.encode()), agent_error=agent_error,
                    judge_valid=True, judge_error='agent patch could not apply to clean base',
                    tool_calls=context_metrics.get('tool_calls_used'), trace_json_path=trace_path)
                atomic_json(attempt / 'result.json', payload)
                status.update(status='completed', resolved=False)
                return 0

        criterion = spec['judge']
        panel_root = Path(spec['panel_root']).resolve()
        criterion_patch = criterion.get('criterion_patch_path')
        if criterion_patch:
            patch_path = (panel_root / criterion_patch).resolve()
            if not patch_path.is_relative_to(panel_root):
                raise RuntimeError('Criterion path escapes material root')
            if not patch_path.stat().st_size:
                raise RuntimeError('Frozen criterion patch is empty; omit its path instead')
            judge_manager.copy_to(judge_id, patch_path, '/tmp/criterion.patch')
            code, stdout, stderr = apply_patch_in_container(judge_manager, judge_id,
                                                           '/tmp/criterion.patch')
            atomic_json(attempt / 'criterion-patch-application.json',
                        {'exit_code': code, 'stdout': stdout, 'stderr': stderr})
            if code:
                raise RuntimeError('Frozen criterion patch could not apply')
        prepared_dependencies(judge_manager, judge_id)
        prepared_editable(judge_manager, judge_id)
        module = criterion['module']
        origin_code = ('import importlib.util,json,pathlib; '
            's=importlib.util.find_spec(' + repr(module) + '); '
            'p=pathlib.Path(s.origin).resolve() if s and s.origin else None; '
            'print(json.dumps({"module":' + repr(module) + ',"origin":str(p),'
            '"source_valid":bool(p and p.is_relative_to(pathlib.Path.cwd().resolve()))}))')
        source = judge_manager.exec(judge_id, 'python3 -c ' + shlex.quote(origin_code), timeout=15)
        report = None
        if source.exit_code == 0:
            report = json.loads((source.stdout or '').strip().splitlines()[-1])
        atomic_json(attempt / 'module-source.json', {'execution': result_record(source), 'report': report})
        if report and not report.get('source_valid'):
            # The clean base passed CPU source validation before model use.
            # A removed/displaced candidate module must not be scored via a
            # globally installed package or omitted as an environment failure.
            payload = dict(entry, instance_id=task.instance_id, repo=task.repo,
                resolved=False, agent_patch_size=len(patch.encode()), agent_error=agent_error,
                judge_valid=True, judge_error='agent patch removed or displaced the task module',
                criterion_pass=False, p2p_pass=None, module_source=report,
                duration_seconds=time.monotonic() - start,
                tool_calls=context_metrics.get('tool_calls_used'),
                total_llm_calls=context_metrics.get('llm_calls_used'), trace_json_path=trace_path)
            atomic_json(attempt / 'result.json', payload)
            status.update(status='completed', resolved=False)
            return 0
        timeout = int(criterion.get('timeout_seconds', criterion.get('timeout', 120)))
        python = shlex.quote(spec.get('runtime_python', 'python3'))
        f2p_nodes = criterion.get('f2p_nodes', [])
        p2p_nodes = criterion.get('p2p_nodes', [])
        if not f2p_nodes and not criterion.get('f2p_command'):
            raise RuntimeError('No frozen fail-to-pass nodes or command')
        pytest_prefix = python + ' -m pytest -q -p no:cacheprovider -o addopts= '
        if pytest_args:
            pytest_prefix += shlex.join(pytest_args) + ' '
        plugin_flags = ''.join(' -p ' + shlex.quote(name) for name in environment.get('pytest_plugins', []))
        f2p_command = (pytest_addopts_export + criterion['f2p_command'] if criterion.get('f2p_command')
                       else pytest_prefix + plugin_flags + ' ' + ' '.join(map(shlex.quote, f2p_nodes)))
        f2p = judge_manager.exec(judge_id, f2p_command, timeout=timeout)
        atomic_json(attempt / 'f2p-execution.json', {'command': f2p_command, **result_record(f2p)})
        p2p_pass = None
        if p2p_nodes or criterion.get('p2p_command'):
            p2p_command = (pytest_addopts_export + criterion['p2p_command'] if criterion.get('p2p_command')
                           else pytest_prefix + plugin_flags + ' ' + ' '.join(map(shlex.quote, p2p_nodes)))
            p2p = judge_manager.exec(judge_id, p2p_command, timeout=timeout)
            atomic_json(attempt / 'p2p-execution.json', {'command': p2p_command, **result_record(p2p)})
            p2p_pass = p2p.exit_code == 0
        # The original environment and criterion were validated before model use.
        # Import/collection failures caused by the Agent's patch are real failures.
        # Missing/foreign baseline assets fail earlier and are recorded separately.
        judge_valid = True
        resolved = bool(f2p.exit_code == 0 and p2p_pass is not False)
        payload = dict(entry, **capture_result, instance_id=task.instance_id, repo=task.repo,
            resolved=resolved, agent_patch_size=len(patch.encode()), agent_error=agent_error,
            test_exit_code=f2p.exit_code, judge_valid=judge_valid,
            criterion_pass=f2p.exit_code == 0,
            p2p_pass=p2p_pass, p2p_nodes=p2p_nodes, module_source=report,
            duration_seconds=time.monotonic() - start,
            tool_calls=context_metrics.get('tool_calls_used'),
            total_llm_calls=context_metrics.get('llm_calls_used'), trace_json_path=trace_path)
        atomic_json(attempt / 'result.json', payload)
        status.update(status='completed' if judge_valid else 'judge_environment_error', resolved=resolved)
        return 0 if judge_valid else 1
    except BaseException as exc:
        status.update(status='error', error_type=type(exc).__name__, error=str(exc))
        atomic_text(attempt / 'exception.txt', traceback.format_exc())
        if capture is not None and capture_result.get('training_trace_reasons') == [dict(kind='agent_not_completed')]:
            captured = capture.finish(None, type(exc).__name__)
            capture_result = dict(training_trace_usable=False,
                                  training_trace_reasons=captured['reasons'],
                                  trajectory_capture_path=str(attempt / 'trajectory-capture.json'))
        if not (attempt / 'result.json').exists():
            atomic_json(attempt / 'result.json', dict(entry, **capture_result, resolved=None,
                error_type=type(exc).__name__, error=str(exc)))
        return 1
    finally:
        if capture is not None:
            capture.restore()
        for instance in (manager, judge_manager):
            if instance is not None:
                try:
                    instance.cleanup_all()
                except Exception:
                    atomic_text(attempt / 'cleanup-error.txt', traceback.format_exc())
        status.update(finished_at=time.time(), outer_seconds=time.monotonic() - start)
        atomic_json(attempt / 'attempt_status.json', status)
        logging.getLogger().removeHandler(handler)
        handler.close()


if __name__ == '__main__':
    raise SystemExit(execute(json.loads(Path(sys.argv[1]).read_text())))
