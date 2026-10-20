"""Run one 12B base/behavior-adapter task with the official Agent runner and a fresh judge.

The Agent receives only the original issue and repository snapshot. Frozen criterion
text remains in the host specification and executes inline only after Agent cleanup.
No criterion/test file is created in either workspace.
"""
from __future__ import annotations

import asyncio
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


# This executes the already-frozen criterion text without creating a test file.
# The synthetic __file__ preserves its original relative src-path convention.
CRITERION_PROGRAM = r'''
import importlib, importlib.util, json, pathlib, sys, traceback
payload = json.loads(sys.argv[1])
workspace = pathlib.Path.cwd().resolve()
sys.path[:0] = [str(workspace / 'src'), str(workspace)]
report = {'criterion_pass': False, 'source_valid': False, 'workspace': str(workspace)}
try:
    module_name = payload['module']
    module_spec = importlib.util.find_spec(module_name)
    if module_spec is None or not module_spec.origin:
        raise RuntimeError('target module has no source origin: ' + module_name)
    origin = pathlib.Path(module_spec.origin).resolve()
    report['module_origin'] = str(origin)
    if not origin.is_relative_to(workspace):
        raise RuntimeError('judge imported external source: ' + str(origin))
    report['source_valid'] = True
    namespace = {'__name__': '_p014_frozen_criterion',
                 '__file__': str(workspace / 'tests' / '__criterion_inline__.py')}
    exec(compile(payload['source'], '<frozen-development-criterion>', 'exec'), namespace)
    node = payload['node'].split('::')[-1]
    namespace[node]()
    module = sys.modules.get(module_name)
    report['loaded_module_file'] = getattr(module, '__file__', None)
    report['criterion_pass'] = True
except BaseException as exc:
    report['exception_type'] = type(exc).__name__
    report['exception'] = str(exc)
    if isinstance(exc, ModuleNotFoundError):
        report['missing_module'] = exc.name
        target = payload['module']
        report['external_dependency_missing'] = bool(exc.name and
            exc.name != target and not exc.name.startswith(target + '.'))
    traceback.print_exc()
print('P014_CRITERION_RESULT=' + json.dumps(report, ensure_ascii=False), flush=True)
sys.exit(0 if report['criterion_pass'] else 1)
'''


def execute(spec: dict) -> int:
    entry = spec['entry']
    attempt = Path(entry['results_dir']).resolve()
    attempt.mkdir(parents=True, exist_ok=True)
    safe_attempt_id = ''.join(ch if ch.isalnum() or ch in '-_.' else '_'
                              for ch in str(entry.get('attempt_id', attempt.name)))
    scratch = Path(spec.get('scratch', Path('/tmp') / 'p014-attempts' / safe_attempt_id)).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    status = dict(entry, status='running', pid=os.getpid(), started_at=time.time())
    start = time.monotonic()
    manager = judge_manager = context = None
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

        class SourceManager(SubprocessManager):
            def exec(self, sandbox_id, command, *, timeout=None):
                prefix = ('export PYTHONPATH=/workspace/src:/workspace; '
                          'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; ')
                return super().exec(sandbox_id, prefix + command, timeout=timeout)

            def stop(self, sandbox_id):
                if sandbox_id in self.sandboxes:
                    try:
                        state = self.exec(sandbox_id, 'git status --short', timeout=10)
                        atomic_text(attempt / 'agent-final-status.txt',
                                    (state.stdout or '') + (state.stderr or ''))
                    except Exception:
                        atomic_text(attempt / 'agent-final-status-error.txt', traceback.format_exc())
                super().stop(sandbox_id)

        # The host already supplies the CPU-prepared dependencies. Keep the official
        # extraction, source tree, baseline, tools, compiler, budget and Agent loop.
        def prepared_environment(*_args, **_kwargs):
            return None
        for hook in ('setup_container_wheels', 'install_editable_package',
                     'install_test_dependencies', 'setup_workspace_test_config'):
            setattr(agent_runner, hook, prepared_environment)
        original_baseline = agent_runner.setup_baseline_commit

        def baseline_with_local_identity(docker, sandbox_id, commit_msg='baseline'):
            configured = docker.exec(sandbox_id,
                "git config user.email 'research@localhost' && "
                "git config user.name 'P014 research'", timeout=10)
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
                    problem_statement=task_input['problem_statement'])
        assert not task.patch and not task.test_patch and not task.FAIL_TO_PASS and not task.PASS_TO_PASS
        snapshot = Path(spec['snapshot']).resolve()
        if not snapshot.is_file():
            raise FileNotFoundError(snapshot)
        install_litellm_reasoning_patch()
        litellm.drop_params = True
        if spec['model_alias'] != 'gemma-4-12b-it-qat-w4a16-ct':
            raise RuntimeError('Expected the pinned 12B model alias')
        original_generate = LiteLlm.generate_content_async
        async def observed_generate(model, *args, **kwargs):
            with (attempt / 'model-requests.jsonl').open('a') as observed:
                observed.write(json.dumps({'time': time.time(), 'model': model.model}) + '\n')
            async for response in original_generate(model, *args, **kwargs):
                yield response
        LiteLlm.generate_content_async = observed_generate
        models = ModelRegistry()
        models.register(spec['model_alias'], LiteLlm(
            model='openai/' + spec['served_model'], api_base=spec['base_url'], api_key='EMPTY'))
        adapters = discover_adapters(spec['agent_dir'], adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
        expected_adapter_names = ['p013_behavior_lora'] if entry['variant'] == 'adapter' else []
        if sorted(adapters.adapters) != expected_adapter_names:
            raise RuntimeError('Wrong adapter discovered for 12B arm: ' + entry['variant'])
        atomic_json(attempt / 'registered-routing.json', {
            'arm': entry['variant'], 'model_alias': spec['model_alias'],
            'adapter_names': sorted(adapters.adapters),
            'expected_repair_model': 'p013_behavior_lora' if expected_adapter_names else spec['served_model'],
            'expected_review_model': spec['served_model']})
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
            verification_timeout_seconds=int(spec['judge'].get('timeout_seconds', 120)),
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
                payload = dict(entry, instance_id=task.instance_id, repo=task.repo, resolved=False,
                    agent_patch_size=len(patch.encode()), agent_error=agent_error,
                    judge_valid=True, judge_error='agent patch could not apply to clean base',
                    tool_calls=context_metrics.get('tool_calls_used'), trace_json_path=trace_path)
                atomic_json(attempt / 'result.json', payload)
                status.update(status='completed', resolved=False)
                return 0

        criterion = spec['judge']
        module_name = criterion.get('module', task.repo.rsplit('/', 1)[-1])
        source = criterion['criterion_source']
        node = criterion['f2p_node']
        criterion_payload = json.dumps({'source': source, 'node': node, 'module': module_name})
        python = shlex.quote(spec.get('runtime_python', 'python3'))
        command = python + ' -c ' + shlex.quote(CRITERION_PROGRAM) + ' ' + shlex.quote(criterion_payload)
        f2p = judge_manager.exec(judge_id, command, timeout=int(criterion.get('timeout_seconds', 120)))
        atomic_json(attempt / 'criterion-execution.json', result_record(f2p))
        report = None
        for line in (f2p.stdout or '').splitlines():
            if line.startswith('P014_CRITERION_RESULT='):
                report = json.loads(line.partition('=')[2])
        atomic_json(attempt / 'criterion-result.json', report)
        judge_valid = bool(report and report.get('source_valid') and
                           not report.get('external_dependency_missing'))
        p2p_nodes = criterion.get('p2p_nodes', [])
        p2p_pass = None
        if p2p_nodes:
            p2p_command = python + ' -m pytest -q -x -p no:cacheprovider ' + ' '.join(map(shlex.quote, p2p_nodes))
            p2p = judge_manager.exec(judge_id, p2p_command, timeout=int(criterion.get('timeout_seconds', 120)))
            atomic_json(attempt / 'p2p-execution.json', result_record(p2p))
            p2p_pass = p2p.exit_code == 0
            if p2p.exit_code in (2, 3, 4, 5) or getattr(p2p, 'timed_out', False):
                judge_valid = False
        resolved = (bool(f2p.exit_code == 0 and report and report.get('criterion_pass') and
                         p2p_pass is not False) if judge_valid else None)
        payload = dict(entry, instance_id=task.instance_id, repo=task.repo,
            resolved=resolved, agent_patch_size=len(patch.encode()), agent_error=agent_error,
            test_exit_code=f2p.exit_code, judge_valid=judge_valid,
            criterion_pass=bool(report and report.get('criterion_pass')),
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
        if not (attempt / 'result.json').exists():
            atomic_json(attempt / 'result.json', dict(entry, resolved=None,
                error_type=type(exc).__name__, error=str(exc)))
        return 1
    finally:
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
