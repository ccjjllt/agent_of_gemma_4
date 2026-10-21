# Frozen P014 runtime repair inference; no competition submission.

from pathlib import Path
import importlib
import json
import os
import subprocess
import sys
import time

os.environ.update({'LITELLM_LOCAL_MODEL_COST_MAP':'True', 'TRANSFORMERS_NO_TF':'1',
                   'VLLM_WORKER_MULTIPROC_METHOD':'spawn', 'VLLM_NO_USAGE_STATS':'1',
                   'VLLM_ENGINE_READY_TIMEOUT_S':'1200', 'OTEL_SDK_DISABLED':'true',
                   'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
inputs = Path('/kaggle/input')
markers = list(inputs.rglob('swegemma-0.2.11-py3-none-any.whl'))
if len(markers) != 1:
    raise RuntimeError('Official wheelhouse mount missing/ambiguous: '+str(markers))
wheelhouse = markers[0].parent
staging = Path('/tmp/p014-official-wheels')
staging.mkdir(exist_ok=True)
for path in wheelhouse.glob('*.whl'):
    if 'cutlass' in path.name.lower():
        continue
    name = path.name.replace('cu128', '+cu128') if 'cu128' in path.name and '+' not in path.name else path.name
    dest = staging/name
    if not dest.exists():
        dest.symlink_to(path)
print('PHASE install_official_runtime', flush=True)
subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps','--force-reinstall',*[str(p) for p in sorted(staging.glob('*.whl'))]],check=True)
panel_markers = list(inputs.rglob('agent-tasks.json'))
if len(panel_markers) != 1:
    raise RuntimeError('CPU panel mount missing/ambiguous: '+str(panel_markers))
panel = panel_markers[0].parent
extra_wheels = list((panel/'wheelhouse').glob('*.whl')) + list((panel/'wheels').glob('*.whl'))
if extra_wheels:
    subprocess.run([sys.executable,'-m','pip','install','-q','--no-deps',*[str(p) for p in extra_wheels]],check=True)
importlib.invalidate_caches()
print('PHASE inputs_ready', flush=True)

SETTINGS = {'study': '31b-runtime-v2', 'model_alias': 'gemma-4-31b-it-qat-w4a16-ct', 'model_path_pattern': 'gemma-4-31b-it-qat-w4a16-ct/2', 'task_ids': ['pallets__click.fde47b4b.pr_2604', 'pallets__markupsafe.620c06c9.pr_402', 'pallets__markupsafe.620c06c9.pr_418', 'pallets__markupsafe.620c06c9.pr_379', 'pallets__markupsafe.620c06c9.pr_400', 'pydicom__pydicom.7d361b3d.pr_1938', 'pydicom__pydicom.7d361b3d.pr_1867', 'pydicom__pydicom.7d361b3d.pr_2163'], 'task_count': 8, 'budgets': {'timeout_seconds': 60, 'max_tool_calls': 28, 'max_time_minutes': 5, 'max_turns': 40}, 'concurrency': 4, 'wall_seconds': 2700, 'attempt_seconds': 480, 'code_commit': 'c995639cf8824c2f69a929c0fa1c1275672290dd', 'quota_estimate_upper_hours': 1.6666666666666667, 'purpose': 'runtime repair on the same frozen eight developer tasks; single arm; no new independent repair-gain claim'}
BUNDLE = {'p014_single_v2_runtime': {'agent.yaml': 'name: p014_single_v2_runtime\nmodel: gemma-4-31b-it-qat-w4a16-ct\ninstruction: !include prompts/system.md\ntools:\n  - run_command\n  - read_file\n  - edit_file\n  - write_file\n  - get_status\n  - submit_patch\ngenerate_content_config: !include configs/sampling.yaml\n', 'configs/sampling.yaml': 'temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: 8192\nthinking_config:\n  thinking_budget: 4096\n  include_thoughts: true\n', 'eval_config.yaml': 'evaluation:\n  timeout_seconds: 60\n  max_tool_calls: 28\n  max_time_minutes: 5\n  max_turns: 40\n', 'prompts/system.md': "You are an autonomous software engineer. Make the smallest complete production-source change in /workspace that satisfies the original issue. The issue specifies the expected names, types, messages, return values and edge cases.\n\n# LIMITS AND PRIORITIES\n- You have 28 tool calls, at most 40 model turns and 5 minutes in total. Wall clock includes tool round trips. The working tree is graded when time expires; an existing edit can count without submit_patch. Finish the production change while there is still time to verify it.\n- Never create, modify or delete test files (test_*.py, *_test.py, anything under tests/). Never run a full test suite. Never look outside /workspace.\n- Spend the initial calls locating and reading the relevant implementation and contract. Aim to complete the production edit by call 10; reserve the last part of the budget for a bounded check, correction and final diff. Do not keep exploring or writing repro variants after the cause is supported by the source.\n\n# UNDERSTAND AND LOCATE\n- Extract the issue's exact API/CLI names, required behavior and every stated edge case. For a feature, find the closest existing sibling and its registration/export/call sites. For a bug, find where the wrong behavior is produced.\n- Use paths and identifiers in the issue directly. Batch independent git grep searches with short output; reuse paths already found. Read the relevant source plus one caller or existing test contract. Avoid repository dumps.\n- Before changing several related locations, read each required region. Cover sibling branches, imports/exports, dispatch entries and argument forwarding together when the issue requires them. Do not edit an unseen branch from memory.\n- Run a pre-edit reproduction only when it distinguishes a concrete hypothesis. Use one inline command, never a new repro file; then proceed to the production change.\n\n# EDIT COMPLETELY\n- Use edit_file with all three required arguments: filepath, old_string and new_string. Copy a short unique old_string exactly from the source; even an insertion requires old_string. Use write_file only for a genuinely new production file required by the issue.\n- If a tool rejects missing or invalid arguments, read its required parameter names and correct the request once. Never repeat the same rejected request. If it fails again, use one equivalent run_command operation instead.\n- After a failed anchor match, read the actual region before retrying. After two matching failures, use one inline Python replacement with assert text.count(old) == 1; preserve unrelated content and inspect the resulting diff.\n- Fix the cause rather than special-casing the example. Preserve existing signatures, defaults and compatibility unless the issue requests a change. Follow neighboring conventions and handle all stated cases.\n- A production edit is the main deliverable. Do not spend the remaining budget repeatedly adjusting a reproduction while leaving the implementation unchanged.\n\n# VERIFY THE MODIFIED SOURCE\n- Prefer an existing focused test or a short inline behavior check. Run Python from /workspace with PYTHONPATH=/workspace/src:/workspace. In the same command, print the target module's __file__ once and confirm it resolves under /workspace before trusting the behavior result. A system-installed copy cannot validate your edit.\n- Bound a short inline reproduction with timeout 10. For a named existing test file use timeout 45, for example: PYTHONPATH=/workspace/src:/workspace timeout 45 python3 -m pytest tests/path/test_x.py -x -q -k keyword. Never run bare pytest or an entire repository suite.\n- Exit 124 means the command did not finish. Treat it as evidence of a hang or an incomplete observation; do not revert a supported edit solely because of that exit code. Avoid another long blocked command.\n- Distinguish an implementation failure from a wrong witness command or unavailable external dependency/fixture. Correct your command when its arguments or import path are wrong. A missing symbol in the target project can be the issue itself: do not dismiss every ImportError as an environment problem.\n- If a focused check exposes a specific defect in the edit, correct the source and rerun that check once. Do not repair tests or dependencies and do not begin a second broad exploration. If no behavior check can run, inspect the complete diff and state the verification limit honestly.\n\n# FINISH\n- Run git status --short and git diff. Confirm the diff covers the original requirement and associated branches, only necessary production files changed, and no debug code or temporary products from this attempt remain.\n- Remove only scratch files that you created during this attempt and have identified as temporary. Never delete pre-existing files or tests using broad globs.\n- Call submit_patch after inspecting the real diff. If it reports patch_size == 0 or files_changed == 0, inspect the tree and make the missing production edit before another submit. Never repeat a zero-patch submission without an intervening edit and new diff evidence.\n- When the tool or time budget is exhausted, do not make further tool requests. After a successful non-empty submit_patch, stop immediately; do not restart verification.\n- End with a short accurate summary of the fix and the checks actually run. A clean submission action alone does not establish correctness.\n"}}
WORKER_SOURCE = '"""Run one external development task with the official Agent runner and a fresh judge.\n\nThe Agent receives only the original issue and repository snapshot. Frozen criterion\ntext remains in the host specification and executes inline only after Agent cleanup.\nNo criterion/test file is created in either workspace.\n"""\nfrom __future__ import annotations\n\nimport asyncio\nimport hashlib\nimport json\nimport logging\nimport os\nfrom pathlib import Path\nimport shlex\nimport sys\nimport time\nimport traceback\nimport uuid\n\n\ndef atomic_text(path: Path, value: str) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    temporary = path.with_name(path.name + \'.tmp.\' + uuid.uuid4().hex)\n    temporary.write_text(value, encoding=\'utf-8\')\n    temporary.replace(path)\n\n\ndef atomic_json(path: Path, value: object) -> None:\n    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + \'\\n\')\n\n\ndef result_record(result: object) -> dict:\n    return {key: getattr(result, key, None) for key in\n            (\'exit_code\', \'stdout\', \'stderr\', \'duration_seconds\', \'timed_out\', \'status\')}\n\n\n# This executes the already-frozen criterion text without creating a test file.\n# The synthetic __file__ preserves its original relative src-path convention.\nCRITERION_PROGRAM = r\'\'\'\nimport importlib, importlib.util, json, pathlib, sys, traceback\npayload = json.loads(sys.argv[1])\nworkspace = pathlib.Path.cwd().resolve()\nsys.path[:0] = [str(workspace / \'src\'), str(workspace)]\nreport = {\'criterion_pass\': False, \'source_valid\': False, \'workspace\': str(workspace)}\ntry:\n    module_name = payload[\'module\']\n    module_spec = importlib.util.find_spec(module_name)\n    if module_spec is None or not module_spec.origin:\n        raise RuntimeError(\'target module has no source origin: \' + module_name)\n    origin = pathlib.Path(module_spec.origin).resolve()\n    report[\'module_origin\'] = str(origin)\n    if not origin.is_relative_to(workspace):\n        raise RuntimeError(\'judge imported external source: \' + str(origin))\n    report[\'source_valid\'] = True\n    namespace = {\'__name__\': \'_p014_frozen_criterion\',\n                 \'__file__\': str(workspace / \'tests\' / \'__criterion_inline__.py\')}\n    exec(compile(payload[\'source\'], \'<frozen-development-criterion>\', \'exec\'), namespace)\n    node = payload[\'node\'].split(\'::\')[-1]\n    namespace[node]()\n    module = sys.modules.get(module_name)\n    report[\'loaded_module_file\'] = getattr(module, \'__file__\', None)\n    report[\'criterion_pass\'] = True\nexcept BaseException as exc:\n    report[\'exception_type\'] = type(exc).__name__\n    report[\'exception\'] = str(exc)\n    if isinstance(exc, ModuleNotFoundError):\n        report[\'missing_module\'] = exc.name\n        target = payload[\'module\']\n        report[\'external_dependency_missing\'] = bool(exc.name and\n            exc.name != target and not exc.name.startswith(target + \'.\'))\n    traceback.print_exc()\nprint(\'P014_CRITERION_RESULT=\' + json.dumps(report, ensure_ascii=False), flush=True)\nsys.exit(0 if report[\'criterion_pass\'] else 1)\n\'\'\'\n\n\ndef execute(spec: dict) -> int:\n    entry = spec[\'entry\']\n    attempt = Path(entry[\'results_dir\']).resolve()\n    attempt.mkdir(parents=True, exist_ok=True)\n    safe_attempt_id = \'\'.join(ch if ch.isalnum() or ch in \'-_.\' else \'_\'\n                              for ch in str(entry.get(\'attempt_id\', attempt.name)))\n    scratch = Path(spec.get(\'scratch\', Path(\'/tmp\') / \'p014-attempts\' / safe_attempt_id)).resolve()\n    scratch.mkdir(parents=True, exist_ok=True)\n    status = dict(entry, status=\'running\', pid=os.getpid(), started_at=time.time())\n    start = time.monotonic()\n    manager = judge_manager = context = None\n    handler = logging.FileHandler(attempt / \'harness.log\', encoding=\'utf-8\')\n    handler.setFormatter(logging.Formatter(\'%(asctime)s %(levelname)s %(name)s %(message)s\'))\n    logging.getLogger().addHandler(handler)\n\n    def phase(name: str) -> None:\n        status.update(phase=name, phase_at=time.time())\n        atomic_json(attempt / \'attempt_status.json\', status)\n        print(json.dumps({\'attempt_id\': entry.get(\'attempt_id\'), \'phase\': name}), flush=True)\n\n    try:\n        phase(\'load_official_runtime\')\n        import litellm\n        from adk_submission import ModelRegistry, discover_adapters\n        from adk_submission.resolvers.generation import install_litellm_reasoning_patch\n        from google.adk.agents.context_cache_config import ContextCacheConfig\n        from google.adk.apps._configs import EventsCompactionConfig\n        from google.adk.models.lite_llm import LiteLlm\n        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits\n        from swegemma.context import SwegemmaContext\n        from swegemma.harness import agent_runner\n        from swegemma.harness.container_setup import extract_snapshot, apply_patch_in_container\n        from swegemma.harness.verification import save_trace_artifact, strip_protected_files_from_patch\n        from swegemma.models import Task\n        from swegemma.sandbox import SubprocessManager\n\n        class SourceManager(SubprocessManager):\n            def exec(self, sandbox_id, command, *, timeout=None):\n                prefix = (\'export PYTHONPATH=/workspace/src:/workspace; \'\n                          \'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; \')\n                return super().exec(sandbox_id, prefix + command, timeout=timeout)\n\n            def stop(self, sandbox_id):\n                if sandbox_id in self.sandboxes:\n                    try:\n                        state = self.exec(sandbox_id, \'git status --short\', timeout=10)\n                        atomic_text(attempt / \'agent-final-status.txt\',\n                                    (state.stdout or \'\') + (state.stderr or \'\'))\n                    except Exception:\n                        atomic_text(attempt / \'agent-final-status-error.txt\', traceback.format_exc())\n                super().stop(sandbox_id)\n\n        # The host already supplies the CPU-prepared dependencies. Keep the official\n        # extraction, source tree, baseline, tools, compiler, budget and Agent loop.\n        def prepared_environment(*_args, **_kwargs):\n            return None\n        for hook in (\'setup_container_wheels\', \'install_editable_package\',\n                     \'install_test_dependencies\', \'setup_workspace_test_config\'):\n            setattr(agent_runner, hook, prepared_environment)\n        original_baseline = agent_runner.setup_baseline_commit\n\n        def baseline_with_local_identity(docker, sandbox_id, commit_msg=\'baseline\'):\n            configured = docker.exec(sandbox_id,\n                "git config user.email \'research@localhost\' && "\n                "git config user.name \'P014 research\'", timeout=10)\n            if configured.exit_code:\n                raise RuntimeError(\'cannot configure snapshot Git identity\')\n            original_baseline(docker, sandbox_id, commit_msg)\n            tagged = docker.exec(sandbox_id, \'git rev-parse _swegemma_baseline\', timeout=10)\n            if tagged.exit_code:\n                raise RuntimeError(\'official baseline commit was not created\')\n        agent_runner.setup_baseline_commit = baseline_with_local_identity\n\n        task_input = spec[\'task\']\n        task = Task(instance_id=task_input.get(\'instance_id\', task_input.get(\'iid\')),\n                    repo=task_input[\'repo\'],\n                    base_commit=task_input.get(\'base_commit\', task_input.get(\'basecommit\')),\n                    problem_statement=task_input[\'problem_statement\'])\n        assert not task.patch and not task.test_patch and not task.FAIL_TO_PASS and not task.PASS_TO_PASS\n        snapshot = Path(spec[\'snapshot\']).resolve()\n        if not snapshot.is_file():\n            raise FileNotFoundError(snapshot)\n        install_litellm_reasoning_patch()\n        litellm.drop_params = True\n        models = ModelRegistry()\n        models.register(spec[\'model_alias\'], LiteLlm(\n            model=\'openai/\' + spec[\'served_model\'], api_base=spec[\'base_url\'], api_key=\'EMPTY\'))\n        adapters = discover_adapters(spec[\'agent_dir\'], adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)\n        for adapter_name in adapters.adapters:\n            for alias in (adapter_name, \'adapter:\' + adapter_name):\n                models.register(alias, LiteLlm(model=\'openai/\' + adapter_name,\n                    api_base=spec[\'base_url\'], api_key=\'EMPTY\'))\n        limits, generation_constraints = build_submission_limits()\n        budgets = spec[\'budgets\']\n        config = EvalConfig(\n            tasks_path=attempt / \'agent-task.json\', snapshots_dir=snapshot.parent,\n            results_dir=attempt, submission_dir=Path(spec[\'agent_dir\']).resolve(),\n            models=models, sandbox=\'subprocess\',\n            timeout_seconds=int(budgets.get(\'timeout_seconds\', 240)),\n            verification_timeout_seconds=int(spec[\'judge\'].get(\'timeout_seconds\', 120)),\n            max_tool_calls=int(budgets[\'max_tool_calls\']),\n            max_time_minutes=float(budgets[\'max_time_minutes\']),\n            max_turns=int(budgets[\'max_turns\']),\n            limits=limits, generation_constraints=generation_constraints, adapter_manifest=adapters,\n            context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),\n            events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,\n                                                           token_threshold=14336, event_retention_size=5),\n            graph_dir=str(scratch / \'absent-graphs\'), embeddings_dir=str(scratch / \'absent-embeddings\'),\n            verbose=False, display_mode=\'quiet\', enable_sandbox_testing=True)\n        # This metadata contains only the issue given to the Agent, with no criterion.\n        atomic_json(attempt / \'agent-task.json\', {\n            \'instance_id\': task.instance_id, \'repo\': task.repo,\n            \'base_commit\': task.base_commit, \'problem_statement\': task.problem_statement})\n        manager = SourceManager(timeout_seconds=config.harness.command_timeout_seconds,\n                                base_dir=scratch / \'agent-sandboxes\', system_site_packages=True)\n        manager.base_dir.mkdir(parents=True, exist_ok=True)\n        context = SwegemmaContext(docker_manager=manager, task=task,\n            graph_dir=config.graph_dir, embeddings_dir=config.embeddings_dir,\n            budget=config.budget, harness=config.harness)\n        phase(\'agent\')\n        patch, agent_error, trace = asyncio.run(agent_runner.run_agent_sandbox(\n            docker=manager, config=config, task=task, snapshot_path=snapshot,\n            task_index=int(entry.get(\'ordinal\', 1)), total_tasks=int(spec.get(\'total_tasks\', 1)),\n            context=context))\n        patch = patch or \'\'\n        atomic_text(attempt / \'agent.patch\', patch)\n        scored_patch = strip_protected_files_from_patch(patch)\n        atomic_text(attempt / \'scored.patch\', scored_patch)\n        trace_path = save_trace_artifact(trace, attempt, task.instance_id)\n        context_metrics = {key: getattr(context, key, None) for key in (\n            \'task_start_time\', \'agent_start_time\', \'agent_end_time\', \'agent_elapsed_seconds\',\n            \'task_elapsed_seconds\', \'patch_submitted\', \'tool_calls_used\', \'llm_calls_used\')}\n        atomic_json(attempt / \'context_metrics.json\', context_metrics)\n\n        phase(\'fresh_judge\')\n        # SourceManager\'s final-status hook is Agent-specific, so use a separate\n        # minimal source-prefix subclass for the independent judge workspace.\n        class JudgeManager(SubprocessManager):\n            def exec(self, sandbox_id, command, *, timeout=None):\n                return super().exec(sandbox_id,\n                    \'export PYTHONPATH=/workspace/src:/workspace; \'\n                    \'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; \' + command, timeout=timeout)\n        judge_manager = JudgeManager(timeout_seconds=120,\n            base_dir=scratch / \'judge-sandboxes\', system_site_packages=True)\n        judge_manager.base_dir.mkdir(parents=True, exist_ok=True)\n        judge_id = judge_manager.start()\n        extract_snapshot(judge_manager, judge_id, snapshot)\n        head = judge_manager.exec(judge_id, \'git rev-parse HEAD\', timeout=10)\n        atomic_json(attempt / \'judge-source.json\', {\n            \'expected_base_commit\': task.base_commit, \'actual_base_commit\': (head.stdout or \'\').strip(),\n            \'snapshot\': str(snapshot), \'snapshot_sha256\': hashlib.sha256(snapshot.read_bytes()).hexdigest()})\n        expected = judge_manager.exec(judge_id, \'git rev-parse --verify \' +\n            shlex.quote(task.base_commit + \'^{commit}\'), timeout=10)\n        if (head.exit_code or expected.exit_code or\n                (head.stdout or \'\').strip() != (expected.stdout or \'\').strip()):\n            raise RuntimeError(\'judge snapshot base commit does not match frozen task\')\n        if scored_patch.strip():\n            judge_manager.copy_to(judge_id, attempt / \'scored.patch\', \'/tmp/agent.patch\')\n            code, out, err = apply_patch_in_container(judge_manager, judge_id, \'/tmp/agent.patch\')\n            atomic_json(attempt / \'patch-application.json\', {\'exit_code\': code, \'stdout\': out, \'stderr\': err})\n            if code:\n                payload = dict(entry, instance_id=task.instance_id, repo=task.repo, resolved=False,\n                    agent_patch_size=len(patch.encode()), agent_error=agent_error,\n                    judge_valid=True, judge_error=\'agent patch could not apply to clean base\',\n                    tool_calls=context_metrics.get(\'tool_calls_used\'), trace_json_path=trace_path)\n                atomic_json(attempt / \'result.json\', payload)\n                status.update(status=\'completed\', resolved=False)\n                return 0\n\n        criterion = spec[\'judge\']\n        module_name = criterion.get(\'module\', task.repo.rsplit(\'/\', 1)[-1])\n        source = criterion[\'criterion_source\']\n        node = criterion[\'f2p_node\']\n        criterion_payload = json.dumps({\'source\': source, \'node\': node, \'module\': module_name})\n        python = shlex.quote(spec.get(\'runtime_python\', \'python3\'))\n        command = python + \' -c \' + shlex.quote(CRITERION_PROGRAM) + \' \' + shlex.quote(criterion_payload)\n        f2p = judge_manager.exec(judge_id, command, timeout=int(criterion.get(\'timeout_seconds\', 120)))\n        atomic_json(attempt / \'criterion-execution.json\', result_record(f2p))\n        report = None\n        for line in (f2p.stdout or \'\').splitlines():\n            if line.startswith(\'P014_CRITERION_RESULT=\'):\n                report = json.loads(line.partition(\'=\')[2])\n        atomic_json(attempt / \'criterion-result.json\', report)\n        judge_valid = bool(report and report.get(\'source_valid\') and\n                           not report.get(\'external_dependency_missing\'))\n        p2p_nodes = criterion.get(\'p2p_nodes\', [])\n        p2p_pass = None\n        if p2p_nodes:\n            p2p_command = python + \' -m pytest -q -x -p no:cacheprovider \' + \' \'.join(map(shlex.quote, p2p_nodes))\n            p2p = judge_manager.exec(judge_id, p2p_command, timeout=int(criterion.get(\'timeout_seconds\', 120)))\n            atomic_json(attempt / \'p2p-execution.json\', result_record(p2p))\n            p2p_pass = p2p.exit_code == 0\n            if p2p.exit_code in (2, 3, 4, 5) or getattr(p2p, \'timed_out\', False):\n                judge_valid = False\n        resolved = (bool(f2p.exit_code == 0 and report and report.get(\'criterion_pass\') and\n                         p2p_pass is not False) if judge_valid else None)\n        payload = dict(entry, instance_id=task.instance_id, repo=task.repo,\n            resolved=resolved, agent_patch_size=len(patch.encode()), agent_error=agent_error,\n            test_exit_code=f2p.exit_code, judge_valid=judge_valid,\n            criterion_pass=bool(report and report.get(\'criterion_pass\')),\n            p2p_pass=p2p_pass, p2p_nodes=p2p_nodes, module_source=report,\n            duration_seconds=time.monotonic() - start,\n            tool_calls=context_metrics.get(\'tool_calls_used\'),\n            total_llm_calls=context_metrics.get(\'llm_calls_used\'), trace_json_path=trace_path)\n        atomic_json(attempt / \'result.json\', payload)\n        status.update(status=\'completed\' if judge_valid else \'judge_environment_error\', resolved=resolved)\n        return 0 if judge_valid else 1\n    except BaseException as exc:\n        status.update(status=\'error\', error_type=type(exc).__name__, error=str(exc))\n        atomic_text(attempt / \'exception.txt\', traceback.format_exc())\n        if not (attempt / \'result.json\').exists():\n            atomic_json(attempt / \'result.json\', dict(entry, resolved=None,\n                error_type=type(exc).__name__, error=str(exc)))\n        return 1\n    finally:\n        for instance in (manager, judge_manager):\n            if instance is not None:\n                try:\n                    instance.cleanup_all()\n                except Exception:\n                    atomic_text(attempt / \'cleanup-error.txt\', traceback.format_exc())\n        status.update(finished_at=time.time(), outer_seconds=time.monotonic() - start)\n        atomic_json(attempt / \'attempt_status.json\', status)\n        logging.getLogger().removeHandler(handler)\n        handler.close()\n\n\nif __name__ == \'__main__\':\n    raise SystemExit(execute(json.loads(Path(sys.argv[1]).read_text())))\n'
"""Run the frozen P014 paired repair experiment on one continuously batched server.

This is an experiment runner, not a test suite. Inputs and answers stay outside
the Agent sandbox; only issue text and the baseline repository enter it.
The launch file supplies SETTINGS, BUNDLE and WORKER_SOURCE as frozen literals.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(data)
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def locate(name):
    found = list(Path('/kaggle/input').rglob(name))
    if len(found) != 1:
        raise RuntimeError(f'Expected one {name}, found {found[:8]}')
    return found[0]


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=4)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


def main():
    started = time.monotonic()
    run_id = 'run-014-' + SETTINGS['study'] + '-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    out = Path('/kaggle/working/results/014') / run_id
    out.mkdir(parents=True)
    deadline = started + SETTINGS.get('wall_seconds', 5100)
    tasks_path = locate('agent-tasks.json')
    panel = tasks_path.parent
    task_document = json.loads(tasks_path.read_text())
    tasks = task_document if isinstance(task_document, list) else task_document['tasks']
    by_id = {t['instance_id']: t for t in tasks}
    selected = SETTINGS.get('task_ids') or [t['instance_id'] for t in tasks][:SETTINGS.get('task_count', 8)]
    tasks = [by_id[iid] for iid in selected]
    criteria_document = json.loads((panel / 'host-criteria.json').read_text())
    criterion_items = criteria_document if isinstance(criteria_document, list) else criteria_document['items']
    criteria = {item['instance_id']: item for item in criterion_items}
    arms = list(BUNDLE)
    agents = out / 'frozen-agents'
    for arm, files in BUNDLE.items():
        for rel, text in files.items():
            save(agents / arm / rel, text)
    manifest = {'run_id': run_id, 'settings': SETTINGS, 'task_ids': selected,
                'task_input_sha256': digest(tasks_path), 'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'config_sha256': {arm: {rel: hashlib.sha256(text.encode()).hexdigest() for rel, text in files.items()} for arm, files in BUNDLE.items()},
                'planned_attempts': len(tasks) * len(arms), 'concurrency': SETTINGS.get('concurrency', 4),
                'source': 'new developer panel; host-only criteria; no reference edit enters agent',
                'versions': {name: importlib.metadata.version(name) for name in ('swegemma', 'adk-submission', 'vllm', 'torch')}}
    save(out / 'experiment-manifest.json', manifest)
    save(out / 'frozen-tasks.json', tasks)
    import torch
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    model = locate('tokenizer.json') if not SETTINGS.get('model_path_pattern') else None
    if model is None:
        model_dirs = [p.parent for p in Path('/kaggle/input').rglob('tokenizer.json') if SETTINGS['model_path_pattern'] in str(p)]
        if len(model_dirs) != 1:
            raise RuntimeError(f'Model mount ambiguous/missing: {model_dirs}')
        model = model_dirs[0]
    else:
        model = model.parent
    manifest['hardware'] = [{'name': torch.cuda.get_device_name(i), 'memory': torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    if len(manifest['hardware']) != 4 or any('L4' not in x['name'] for x in manifest['hardware']):
        raise RuntimeError(f'Requires four L4s, got {manifest["hardware"]}')
    adapter_manifest = discover_adapters(str(agents / arms[-1]), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    config = VllmConfig(model=str(model), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
                        max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90,
                        enable_auto_tool_choice=True, enable_lora=False, tensor_parallel_size=4,
                        startup_timeout=1200, extra_args=['--max-num-seqs', str(SETTINGS.get('concurrency', 4))])
    server = VllmServer(config, adapter_manifest=adapter_manifest)
    worker = out / 'attempt_worker.py'
    save(worker, WORKER_SOURCE)
    schedule = []
    for slot, task in enumerate(tasks):
        for arm in (arms if slot % 2 == 0 else list(reversed(arms))):
            ordinal = len(schedule) + 1
            entry = {'ordinal': ordinal, 'task_slot': slot + 1, 'task_id': task['instance_id'], 'variant': arm,
                     'attempt_id': f'{ordinal:02d}-{task["instance_id"]}-{arm}'}
            entry['results_dir'] = str(out / 'attempts' / entry['attempt_id'])
            schedule.append((entry, task))
    save(out / 'schedule.json', [entry for entry, task in schedule])
    active, rows = [], []
    telemetry_stop = threading.Event()
    def telemetry():
        with (out / 'gpu-utilization.jsonl').open('a') as log:
            while not telemetry_stop.is_set():
                p = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True)
                log.write(json.dumps({'seconds': time.monotonic()-started, 'samples': p.stdout.strip().splitlines()})+'\n'); log.flush()
                telemetry_stop.wait(5)
    thread = threading.Thread(target=telemetry, daemon=True)
    def collect(record, forced=None):
        entry, process, log, launch_time = record
        log.close()
        path = Path(entry['results_dir']) / 'result.json'
        result = json.loads(path.read_text()) if path.exists() else {'resolved': None, 'error': forced or f'worker exit {process.returncode}'}
        row = {**entry, **result, 'worker_exit_code': process.returncode, 'outer_seconds': time.monotonic()-launch_time}
        if forced:
            row['error'] = forced
        rows.append(row)
        save(out / 'analysis-rows.json', rows)
        print('RESULT ' + json.dumps(row, ensure_ascii=False), flush=True)
    try:
        thread.start()
        print('PHASE model_start', flush=True)
        server.start()
        manifest['model_startup_seconds'] = time.monotonic()-started
        manifest['server_command'] = server.build_cmd()
        save(out / 'experiment-manifest.json', manifest)
        print('PHASE rolling_evaluation', flush=True)
        queue = iter(schedule)
        exhausted = False
        while active or not exhausted:
            while not exhausted and len(active) < SETTINGS.get('concurrency', 4) and time.monotonic() < deadline-150:
                try:
                    entry, task = next(queue)
                except StopIteration:
                    exhausted = True
                    break
                attempt = Path(entry['results_dir']); attempt.mkdir(parents=True)
                snapshot = panel / 'snapshots' / (task['instance_id'] + '.tar.gz')
                spec = {'entry': entry, 'agent_dir': str(agents / entry['variant']), 'task': task,
                        'snapshot': str(snapshot), 'base_url': str(server.base_url),
                        'served_model': str(getattr(server, 'effective_model_path', model)), 'model_alias': SETTINGS['model_alias'],
                        'budgets': SETTINGS['budgets'], 'judge': criteria[task['instance_id']],
                        'panel_root': str(panel), 'total_tasks': len(schedule)}
                save(attempt / 'spec.json', spec)
                log = (attempt / 'worker.log').open('w')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
                process = subprocess.Popen([sys.executable, '-u', str(worker), str(attempt/'spec.json')], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active.append((entry, process, log, time.monotonic()))
                print('LAUNCH ' + entry['attempt_id'], flush=True)
            if time.monotonic() >= deadline-150:
                exhausted = True
            for record in active[:]:
                entry, process, log, launch_time = record
                forced = None
                if process.poll() is None and (time.monotonic()-launch_time > SETTINGS.get('attempt_seconds', 1020) or time.monotonic() >= deadline-60):
                    forced = 'experiment_wall_timeout'; stop(process)
                if process.poll() is not None:
                    collect(record, forced); active.remove(record)
            if active:
                time.sleep(1)
    except BaseException:
        save(out / 'failure.txt', traceback.format_exc())
        raise
    finally:
        for record in active:
            stop(record[1]); collect(record, 'interrupted')
        telemetry_stop.set(); thread.join(timeout=10)
        server_log = getattr(server, 'log_path', None)
        if server_log and Path(server_log).exists():
            shutil.copyfile(server_log, out/'vllm-server.log')
        server.stop(timeout=10)
        summary = {'returned_attempts': len(rows), 'completed_attempts': sum(r.get('worker_exit_code') == 0 and r.get('resolved') is not None for r in rows),
                   'planned_attempts': len(schedule), 'wall_seconds': time.monotonic()-started,
                   'arms': {arm: {'resolved': sum(r.get('resolved') is True and r['variant']==arm for r in rows), 'attempts': sum(r['variant']==arm for r in rows)} for arm in arms}}
        save(out / 'summary.json', summary)
        print('FINAL ' + json.dumps(summary), flush=True)
        with zipfile.ZipFile('/kaggle/working/p014-'+SETTINGS['study']+'-output.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in out.rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(out.parent))


if __name__ == '__main__':
    main()
