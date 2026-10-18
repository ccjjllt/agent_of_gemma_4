
from pathlib import Path
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import traceback
import uuid

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + uuid.uuid4().hex)
    with tmp.open('w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + '\n')



SPEC = json.loads(Path(sys.argv[1]).read_text())
ENTRY = SPEC['entry']
ATTEMPT = Path(ENTRY['results_dir'])
STATUS = dict(ENTRY, status='initializing', started_at=time.time(), pid=os.getpid())


def phase(name):
    STATUS['phase'] = name
    STATUS['phase_at'] = time.time()
    atomic_json(ATTEMPT / 'attempt_status.json', STATUS)
    print(json.dumps({'attempt_id': ENTRY['attempt_id'], 'phase': name, 'time': time.time()}), flush=True)


def execute():
    evaluator = ctx = result = None
    start = time.monotonic()
    handler = logging.FileHandler(ATTEMPT / 'harness.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
    logging.getLogger().addHandler(handler)
    try:
        phase('import')
        import litellm
        from adk_submission import ModelRegistry, discover_adapters
        from adk_submission.resolvers.generation import install_litellm_reasoning_patch
        from google.adk.models.lite_llm import LiteLlm
        from google.adk.agents.context_cache_config import ContextCacheConfig
        from google.adk.apps._configs import EventsCompactionConfig
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
        from swegemma.context import SwegemmaContext
        from swegemma.evaluate import Evaluator
        from swegemma.harness.verification import save_trace_artifact
        from swegemma.models import load_tasks
        from swegemma.results import append_task_result
        # Same aliases as official VllmServer.create_model_registry(),
        # without constructing or starting another inference server.
        install_litellm_reasoning_patch()
        litellm.drop_params = True
        models = ModelRegistry()
        models.register(SPEC['model_alias'], LiteLlm(
            model='openai/' + SPEC['served_model'], api_base=SPEC['base_url'], api_key='EMPTY'))
        data = Path(SPEC['data_dir'])
        task = next(task for task in load_tasks(Path(SPEC['tasks_subset'])) if task.instance_id == ENTRY['task_id'])
        budgets = SPEC['budgets']
        adapters = discover_adapters(SPEC['agent_dir'], adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
        for adapter_name in adapters.adapters:
            for alias in (adapter_name, 'adapter:' + adapter_name):
                models.register(alias, LiteLlm(model='openai/' + adapter_name,
                    api_base=SPEC['base_url'], api_key='EMPTY'))
        limits, generation_constraints = build_submission_limits()
        config = EvalConfig(
            tasks_path=Path(SPEC['tasks_subset']), snapshots_dir=data / 'snapshots', results_dir=ATTEMPT,
            submission_dir=Path(SPEC['agent_dir']), models=models, sandbox='subprocess',
            timeout_seconds=int(budgets['timeout_seconds']), verification_timeout_seconds=int(SPEC['verification_timeout_seconds']),
            max_tool_calls=int(budgets['max_tool_calls']),
            max_time_minutes=float(budgets['max_time_minutes']), max_turns=int(budgets['max_turns']),
            limits=limits, generation_constraints=generation_constraints, adapter_manifest=adapters,
            context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
            events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                           token_threshold=14336, event_retention_size=5),
            graph_dir=str(data / 'graphs'), embeddings_dir=str(data / 'embeddings'),
            wheels_dir=data / 'wheels', verbose=False)
        evaluator = Evaluator(config)
        assert not evaluator.docker.sandboxes
        # Official SubprocessManager accepts a base_dir. Fresh attempt-specific
        # directories isolate repository, venv, skill staging, and judge state.
        evaluator.docker.base_dir = Path(SPEC['scratch']) / 'sandboxes'
        evaluator.docker.base_dir.mkdir(parents=True, exist_ok=False)
        ctx = SwegemmaContext(docker_manager=evaluator.docker, task=task,
                              graph_dir=config.graph_dir, embeddings_dir=config.embeddings_dir,
                              budget=config.budget, harness=config.harness)
        
        assert config.harness.verification_timeout_seconds == 300
        assert config.harness.command_timeout_seconds == int(budgets['timeout_seconds'])
        if SPEC.get('prepare_only'):
            phase('compile_without_model_or_judge')
            from adk_submission import compile_submission
            from swegemma.sandbox import AdkSandboxCodeExecutor
            ctx.container_id = evaluator.docker.start()
            executor = AdkSandboxCodeExecutor(
                sandbox=ctx.sandbox, timeout_seconds=config.harness.command_timeout_seconds,
                start_time=ctx.task_start_time, max_time_minutes=config.budget.time_minutes,
                budget_check_fn=ctx.check_budget)
            agent = compile_submission(
                submission_dir=config.submission_dir, tool_registry=ctx.create_tools(),
                model_registry=models, code_executor=executor,
                script_timeout=config.harness.command_timeout_seconds,
                limits=limits, generation_constraints=generation_constraints,
                adapter_manifest=adapters)
            STATUS.update(status='prepared', agent_type=type(agent).__name__, agent_name=agent.name,
                          tool_types=[type(tool).__name__ for tool in agent.tools],
                          command_timeout_seconds=config.harness.command_timeout_seconds,
                          verification_timeout_seconds=config.harness.verification_timeout_seconds,
                          model_requests=0, judge_runs=0)
            atomic_json(ATTEMPT / 'prepared.json', STATUS)
            print(json.dumps(STATUS), flush=True)
            return 0

        STATUS['status'] = 'running'
        phase('evaluate_task')
        result = asyncio.run(evaluator.evaluate_task(
            task=task, task_index=ENTRY['ordinal'], total_tasks=SPEC['total_tasks'], context=ctx))
        phase('archive')
        payload = {'instance_id': result.instance_id, 'repo': result.repo, 'resolved': result.resolved,
                   'agent_patch_size': len(result.agent_patch or ''), 'test_exit_code': result.test_exit_code,
                   'duration_seconds': result.duration_seconds, 'tool_calls': result.tool_calls,
                   'total_llm_calls': result.total_llm_calls, 'error': result.error,
                   'trace_json_path': getattr(result, 'trace_json_path', None)}
        atomic_json(ATTEMPT / 'result.json', payload)
        safe_id = task.instance_id.replace('/', '__')
        trace = ATTEMPT / 'traces' / f'trace_{safe_id}.json'
        if not trace.is_file() and getattr(result, 'trace', None) is not None:
            save_trace_artifact(result.trace, ATTEMPT, task.instance_id)
        atomic_text(ATTEMPT / 'patches' / f'{safe_id}.patch', result.agent_patch or '')
        atomic_text(ATTEMPT / 'test_outputs' / f'{safe_id}.log', result.test_output or '')
        append_task_result(result, ATTEMPT, [result])
        STATUS.update(status='completed' if trace.is_file() else 'completed_evidence_gap',
                      trace_present=trace.is_file(), resolved=result.resolved)
    except BaseException as exc:
        STATUS.update(status='error', error_type=type(exc).__name__, error=str(exc))
        atomic_text(ATTEMPT / 'exception.txt', traceback.format_exc())
        if not (ATTEMPT / 'result.json').exists():
            atomic_json(ATTEMPT / 'result.json', {'instance_id': ENTRY['task_id'], 'resolved': None,
                        'error_type': type(exc).__name__, 'error': str(exc)})
    finally:
        STATUS.update(finished_at=time.time(), outer_seconds=time.monotonic() - start)
        if ctx is not None:
            fields = ('task_start_time', 'agent_start_time', 'agent_end_time', 'agent_elapsed_seconds',
                      'task_elapsed_seconds', 'patch_submitted', 'tool_calls_used', 'llm_calls_used')
            atomic_json(ATTEMPT / 'context_metrics.json', {field: getattr(ctx, field, None) for field in fields})
            if result is None and getattr(ctx, 'submitted_patch', None) is not None:
                atomic_text(ATTEMPT / 'submitted_patch_before_exception.patch', ctx.submitted_patch)
        if evaluator is not None:
            try:
                evaluator.docker.cleanup_all()
            except Exception:
                atomic_text(ATTEMPT / 'sandbox_cleanup_error.txt', traceback.format_exc())
        logging.getLogger().removeHandler(handler)
        handler.close()
        atomic_json(ATTEMPT / 'attempt_status.json', STATUS)
    return 0 if STATUS['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(execute())
