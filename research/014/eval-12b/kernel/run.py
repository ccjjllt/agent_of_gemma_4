# Frozen P014 12B repair-only LoRA comparison; no training or submission.

from pathlib import Path
import importlib
import json
import os
import subprocess
import sys
import time

P014_SESSION_START = time.monotonic()
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

SETTINGS = {'study': '12b-behavior-pair-v1', 'model_alias': 'gemma-4-12b-it-qat-w4a16-ct', 'model_path_pattern': 'gemma-4-12b-it-qat-w4a16-ct/2', 'task_ids': ['pallets__click.fde47b4b.pr_2604', 'pallets__markupsafe.620c06c9.pr_402', 'pydicom__pydicom.7d361b3d.pr_1938', 'pydicom__pydicom.7d361b3d.pr_1867'], 'subset_selection': {'selected_slots': ['t01', 't02', 't06', 't08'], 'task_ids': ['pallets__click.fde47b4b.pr_2604', 'pallets__markupsafe.620c06c9.pr_402', 'pydicom__pydicom.7d361b3d.pr_1938', 'pydicom__pydicom.7d361b3d.pr_1867'], 'repos': ['pallets/click', 'pallets/markupsafe', 'pydicom/pydicom', 'pydicom/pydicom'], 'selection_rule': 'Frozen before inference outputs: first Click, first MarkupSafe, first and second admitted pydicom slots from the predetermined P014 panel; 4 issues / 3 repositories. Not selected by 31B or 12B outcomes.', 'source_index': 'research/013/eval-31b-pair/criteria-src/criteria-index.json', 'panel_kernel': 'ccjjllt/p014-panel-cpu-v1'}, 'budgets': {'timeout_seconds': 240, 'max_tool_calls': 28, 'max_time_minutes': 8, 'max_turns': 80}, 'concurrency': 4, 'wall_seconds': 6900, 'attempt_seconds': 1020, 'code_commit': 'a2b1298a894f2f7ac8625b8bd0975c55e1b8f830', 'quota_estimate_upper_hours': 4.0, 'purpose': 'Same 12B agent and budget; only repair LoRA differs'}
BUNDLE = {'base': {'agent.yaml': 'name: p014_12b_repair_review\nagent_class: SequentialAgent\ndescription: Same repair-review structure in both arms; only repair adapter differs.\nsub_agents:\n  - config_path: repair.yaml\n  - config_path: review.yaml\n', 'configs/sampling.yaml': 'temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: 8192\nthinking_config:\n  thinking_budget: 4096\n  include_thoughts: true\n', 'eval_config.yaml': 'evaluation:\n  timeout_seconds: 240\n  max_tool_calls: 28\n  max_time_minutes: 8\n  max_turns: 80\n', 'prompts/repair.md': 'You are the repair stage of a software engineer working in /workspace. Read the issue as the specification, locate the cause, and complete the smallest correct production-source change. Then return a concise factual handoff for review.\n\nThe whole task has 8 minutes and 28 tool calls shared with review. Use most of your time to complete the fix; leave time for one focused review. Read the target implementation, each related branch you will modify, and one caller or relevant existing test. Once the cause is supported, edit. Do not keep making reproduction variants while production code remains unchanged.\n\nFix associated branches, exports, registration and argument forwarding together when the issue requires them. Preserve unrelated behaviour and use exact names, exception types and messages from the issue. Use edit_file with a unique source anchor copied from the current file; if an anchor fails, read the actual source and correct the anchor. Never repeat an identical failed edit.\n\nDo not create, modify or delete tests. Never run a full suite. Before the first production edit, use a short reproduction only when its result distinguishes concrete causes. Bound inline reproductions with timeout 10. Confirm Python checks import the working tree with PYTHONPATH=/workspace/src:/workspace and module.__file__; a stale installed module is not valid evidence. A command timeout (124) means verification incomplete. Fix a wrong verification command; do not revert a source fix on a stale-package error or timeout.\n\nReview performs the principal acceptance check. You may run one focused check when needed to guide a correction; avoid repeating equivalent checks in both stages. Return modified file paths, core change, commands actually run with exit codes and key output, and any unverified condition. Do not claim tests passed unless observed. Do not call submit_patch; the review stage submits.\n', 'prompts/review.md': 'You review the actual production fix in /workspace, check the issue contract, and submit.\n\n# Original issue\n{problem_description}\n\n# Repair handoff\n{repair_handoff}\n\nRead git status --short and the actual git diff. Check the exact behaviours, names, exception types, messages and edge cases requested by the issue. The handoff is evidence to examine, not a substitute for the diff. If a concrete requirement is missing, make one targeted correction.\n\nRun one focused acceptance command using existing relevant tests or a short inline check. Use PYTHONPATH=/workspace/src:/workspace and confirm module.__file__ points to the working tree before interpreting Python results. Use timeout 10 for inline reproductions, or timeout 60 (up to 120 if needed) for a named existing test file. A timeout means verification incomplete, not proof the patch is wrong. Do not rerun an equivalent check already sufficient for acceptance unless there is a specific concern.\n\nDo not create, modify or delete test files, and do not run full suites. Remove only temporary products demonstrably created during this attempt; preserve pre-existing tracked and untracked files. Check the final git diff and status. Call submit_patch with a nonempty production patch. If it is empty, inspect actual source state and make a real correction before any retry. End with a factual sentence describing the fix and verification result.\n', 'repair.yaml': 'name: p014_12b_repair\nmodel: gemma-4-12b-it-qat-w4a16-ct\ninstruction: !include prompts/repair.md\noutput_key: repair_handoff\ntools:\n  - run_command\n  - read_file\n  - edit_file\n  - write_file\n  - get_status\ngenerate_content_config: !include configs/sampling.yaml\n', 'review.yaml': 'name: p014_12b_review\nmodel: gemma-4-12b-it-qat-w4a16-ct\ninstruction: !include prompts/review.md\ninclude_contents: none\ntools:\n  - run_command\n  - read_file\n  - edit_file\n  - write_file\n  - get_status\n  - submit_patch\ngenerate_content_config: !include configs/sampling.yaml\n'}, 'adapter': {'agent.yaml': 'name: p014_12b_repair_review\nagent_class: SequentialAgent\ndescription: Same repair-review structure in both arms; only repair adapter differs.\nsub_agents:\n  - config_path: repair.yaml\n  - config_path: review.yaml\n', 'configs/sampling.yaml': 'temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: 8192\nthinking_config:\n  thinking_budget: 4096\n  include_thoughts: true\n', 'eval_config.yaml': 'evaluation:\n  timeout_seconds: 240\n  max_tool_calls: 28\n  max_time_minutes: 8\n  max_turns: 80\n', 'prompts/repair.md': 'You are the repair stage of a software engineer working in /workspace. Read the issue as the specification, locate the cause, and complete the smallest correct production-source change. Then return a concise factual handoff for review.\n\nThe whole task has 8 minutes and 28 tool calls shared with review. Use most of your time to complete the fix; leave time for one focused review. Read the target implementation, each related branch you will modify, and one caller or relevant existing test. Once the cause is supported, edit. Do not keep making reproduction variants while production code remains unchanged.\n\nFix associated branches, exports, registration and argument forwarding together when the issue requires them. Preserve unrelated behaviour and use exact names, exception types and messages from the issue. Use edit_file with a unique source anchor copied from the current file; if an anchor fails, read the actual source and correct the anchor. Never repeat an identical failed edit.\n\nDo not create, modify or delete tests. Never run a full suite. Before the first production edit, use a short reproduction only when its result distinguishes concrete causes. Bound inline reproductions with timeout 10. Confirm Python checks import the working tree with PYTHONPATH=/workspace/src:/workspace and module.__file__; a stale installed module is not valid evidence. A command timeout (124) means verification incomplete. Fix a wrong verification command; do not revert a source fix on a stale-package error or timeout.\n\nReview performs the principal acceptance check. You may run one focused check when needed to guide a correction; avoid repeating equivalent checks in both stages. Return modified file paths, core change, commands actually run with exit codes and key output, and any unverified condition. Do not claim tests passed unless observed. Do not call submit_patch; the review stage submits.\n', 'prompts/review.md': 'You review the actual production fix in /workspace, check the issue contract, and submit.\n\n# Original issue\n{problem_description}\n\n# Repair handoff\n{repair_handoff}\n\nRead git status --short and the actual git diff. Check the exact behaviours, names, exception types, messages and edge cases requested by the issue. The handoff is evidence to examine, not a substitute for the diff. If a concrete requirement is missing, make one targeted correction.\n\nRun one focused acceptance command using existing relevant tests or a short inline check. Use PYTHONPATH=/workspace/src:/workspace and confirm module.__file__ points to the working tree before interpreting Python results. Use timeout 10 for inline reproductions, or timeout 60 (up to 120 if needed) for a named existing test file. A timeout means verification incomplete, not proof the patch is wrong. Do not rerun an equivalent check already sufficient for acceptance unless there is a specific concern.\n\nDo not create, modify or delete test files, and do not run full suites. Remove only temporary products demonstrably created during this attempt; preserve pre-existing tracked and untracked files. Check the final git diff and status. Call submit_patch with a nonempty production patch. If it is empty, inspect actual source state and make a real correction before any retry. End with a factual sentence describing the fix and verification result.\n', 'repair.yaml': 'name: p014_12b_repair\nmodel: gemma-4-12b-it-qat-w4a16-ct\nadapter: p013_behavior_lora\ninstruction: !include prompts/repair.md\noutput_key: repair_handoff\ntools:\n  - run_command\n  - read_file\n  - edit_file\n  - write_file\n  - get_status\ngenerate_content_config: !include configs/sampling.yaml\n', 'review.yaml': 'name: p014_12b_review\nmodel: gemma-4-12b-it-qat-w4a16-ct\ninstruction: !include prompts/review.md\ninclude_contents: none\ntools:\n  - run_command\n  - read_file\n  - edit_file\n  - write_file\n  - get_status\n  - submit_patch\ngenerate_content_config: !include configs/sampling.yaml\n'}}
WORKER_SOURCE = '"""Run one 12B base/behavior-adapter task with the official Agent runner and a fresh judge.\n\nThe Agent receives only the original issue and repository snapshot. Frozen criterion\ntext remains in the host specification and executes inline only after Agent cleanup.\nNo criterion/test file is created in either workspace.\n"""\nfrom __future__ import annotations\n\nimport asyncio\nimport hashlib\nimport json\nimport logging\nimport os\nfrom pathlib import Path\nimport shlex\nimport sys\nimport time\nimport traceback\nimport uuid\n\n\ndef atomic_text(path: Path, value: str) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    temporary = path.with_name(path.name + \'.tmp.\' + uuid.uuid4().hex)\n    temporary.write_text(value, encoding=\'utf-8\')\n    temporary.replace(path)\n\n\ndef atomic_json(path: Path, value: object) -> None:\n    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + \'\\n\')\n\n\ndef result_record(result: object) -> dict:\n    return {key: getattr(result, key, None) for key in\n            (\'exit_code\', \'stdout\', \'stderr\', \'duration_seconds\', \'timed_out\', \'status\')}\n\n\n# This executes the already-frozen criterion text without creating a test file.\n# The synthetic __file__ preserves its original relative src-path convention.\nCRITERION_PROGRAM = r\'\'\'\nimport importlib, importlib.util, json, pathlib, sys, traceback\npayload = json.loads(sys.argv[1])\nworkspace = pathlib.Path.cwd().resolve()\nsys.path[:0] = [str(workspace / \'src\'), str(workspace)]\nreport = {\'criterion_pass\': False, \'source_valid\': False, \'workspace\': str(workspace)}\ntry:\n    module_name = payload[\'module\']\n    module_spec = importlib.util.find_spec(module_name)\n    if module_spec is None or not module_spec.origin:\n        raise RuntimeError(\'target module has no source origin: \' + module_name)\n    origin = pathlib.Path(module_spec.origin).resolve()\n    report[\'module_origin\'] = str(origin)\n    if not origin.is_relative_to(workspace):\n        raise RuntimeError(\'judge imported external source: \' + str(origin))\n    report[\'source_valid\'] = True\n    namespace = {\'__name__\': \'_p014_frozen_criterion\',\n                 \'__file__\': str(workspace / \'tests\' / \'__criterion_inline__.py\')}\n    exec(compile(payload[\'source\'], \'<frozen-development-criterion>\', \'exec\'), namespace)\n    node = payload[\'node\'].split(\'::\')[-1]\n    namespace[node]()\n    module = sys.modules.get(module_name)\n    report[\'loaded_module_file\'] = getattr(module, \'__file__\', None)\n    report[\'criterion_pass\'] = True\nexcept BaseException as exc:\n    report[\'exception_type\'] = type(exc).__name__\n    report[\'exception\'] = str(exc)\n    if isinstance(exc, ModuleNotFoundError):\n        report[\'missing_module\'] = exc.name\n        target = payload[\'module\']\n        report[\'external_dependency_missing\'] = bool(exc.name and\n            exc.name != target and not exc.name.startswith(target + \'.\'))\n    traceback.print_exc()\nprint(\'P014_CRITERION_RESULT=\' + json.dumps(report, ensure_ascii=False), flush=True)\nsys.exit(0 if report[\'criterion_pass\'] else 1)\n\'\'\'\n\n\ndef execute(spec: dict) -> int:\n    entry = spec[\'entry\']\n    attempt = Path(entry[\'results_dir\']).resolve()\n    attempt.mkdir(parents=True, exist_ok=True)\n    safe_attempt_id = \'\'.join(ch if ch.isalnum() or ch in \'-_.\' else \'_\'\n                              for ch in str(entry.get(\'attempt_id\', attempt.name)))\n    scratch = Path(spec.get(\'scratch\', Path(\'/tmp\') / \'p014-attempts\' / safe_attempt_id)).resolve()\n    scratch.mkdir(parents=True, exist_ok=True)\n    status = dict(entry, status=\'running\', pid=os.getpid(), started_at=time.time())\n    start = time.monotonic()\n    manager = judge_manager = context = None\n    handler = logging.FileHandler(attempt / \'harness.log\', encoding=\'utf-8\')\n    handler.setFormatter(logging.Formatter(\'%(asctime)s %(levelname)s %(name)s %(message)s\'))\n    logging.getLogger().addHandler(handler)\n\n    def phase(name: str) -> None:\n        status.update(phase=name, phase_at=time.time())\n        atomic_json(attempt / \'attempt_status.json\', status)\n        print(json.dumps({\'attempt_id\': entry.get(\'attempt_id\'), \'phase\': name}), flush=True)\n\n    try:\n        phase(\'load_official_runtime\')\n        import litellm\n        from adk_submission import ModelRegistry, discover_adapters\n        from adk_submission.resolvers.generation import install_litellm_reasoning_patch\n        from google.adk.agents.context_cache_config import ContextCacheConfig\n        from google.adk.apps._configs import EventsCompactionConfig\n        from google.adk.models.lite_llm import LiteLlm\n        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits\n        from swegemma.context import SwegemmaContext\n        from swegemma.harness import agent_runner\n        from swegemma.harness.container_setup import extract_snapshot, apply_patch_in_container\n        from swegemma.harness.verification import save_trace_artifact, strip_protected_files_from_patch\n        from swegemma.models import Task\n        from swegemma.sandbox import SubprocessManager\n\n        class SourceManager(SubprocessManager):\n            def exec(self, sandbox_id, command, *, timeout=None):\n                prefix = (\'export PYTHONPATH=/workspace/src:/workspace; \'\n                          \'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; \')\n                return super().exec(sandbox_id, prefix + command, timeout=timeout)\n\n            def stop(self, sandbox_id):\n                if sandbox_id in self.sandboxes:\n                    try:\n                        state = self.exec(sandbox_id, \'git status --short\', timeout=10)\n                        atomic_text(attempt / \'agent-final-status.txt\',\n                                    (state.stdout or \'\') + (state.stderr or \'\'))\n                    except Exception:\n                        atomic_text(attempt / \'agent-final-status-error.txt\', traceback.format_exc())\n                super().stop(sandbox_id)\n\n        # The host already supplies the CPU-prepared dependencies. Keep the official\n        # extraction, source tree, baseline, tools, compiler, budget and Agent loop.\n        def prepared_environment(*_args, **_kwargs):\n            return None\n        for hook in (\'setup_container_wheels\', \'install_editable_package\',\n                     \'install_test_dependencies\', \'setup_workspace_test_config\'):\n            setattr(agent_runner, hook, prepared_environment)\n        original_baseline = agent_runner.setup_baseline_commit\n\n        def baseline_with_local_identity(docker, sandbox_id, commit_msg=\'baseline\'):\n            configured = docker.exec(sandbox_id,\n                "git config user.email \'research@localhost\' && "\n                "git config user.name \'P014 research\'", timeout=10)\n            if configured.exit_code:\n                raise RuntimeError(\'cannot configure snapshot Git identity\')\n            original_baseline(docker, sandbox_id, commit_msg)\n            tagged = docker.exec(sandbox_id, \'git rev-parse _swegemma_baseline\', timeout=10)\n            if tagged.exit_code:\n                raise RuntimeError(\'official baseline commit was not created\')\n        agent_runner.setup_baseline_commit = baseline_with_local_identity\n\n        task_input = spec[\'task\']\n        task = Task(instance_id=task_input.get(\'instance_id\', task_input.get(\'iid\')),\n                    repo=task_input[\'repo\'],\n                    base_commit=task_input.get(\'base_commit\', task_input.get(\'basecommit\')),\n                    problem_statement=task_input[\'problem_statement\'])\n        assert not task.patch and not task.test_patch and not task.FAIL_TO_PASS and not task.PASS_TO_PASS\n        snapshot = Path(spec[\'snapshot\']).resolve()\n        if not snapshot.is_file():\n            raise FileNotFoundError(snapshot)\n        install_litellm_reasoning_patch()\n        litellm.drop_params = True\n        if spec[\'model_alias\'] != \'gemma-4-12b-it-qat-w4a16-ct\':\n            raise RuntimeError(\'Expected the pinned 12B model alias\')\n        original_generate = LiteLlm.generate_content_async\n        async def observed_generate(model, *args, **kwargs):\n            with (attempt / \'model-requests.jsonl\').open(\'a\') as observed:\n                observed.write(json.dumps({\'time\': time.time(), \'model\': model.model}) + \'\\n\')\n            async for response in original_generate(model, *args, **kwargs):\n                yield response\n        LiteLlm.generate_content_async = observed_generate\n        models = ModelRegistry()\n        models.register(spec[\'model_alias\'], LiteLlm(\n            model=\'openai/\' + spec[\'served_model\'], api_base=spec[\'base_url\'], api_key=\'EMPTY\'))\n        adapters = discover_adapters(spec[\'agent_dir\'], adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)\n        expected_adapter_names = [\'p013_behavior_lora\'] if entry[\'variant\'] == \'adapter\' else []\n        if sorted(adapters.adapters) != expected_adapter_names:\n            raise RuntimeError(\'Wrong adapter discovered for 12B arm: \' + entry[\'variant\'])\n        atomic_json(attempt / \'registered-routing.json\', {\n            \'arm\': entry[\'variant\'], \'model_alias\': spec[\'model_alias\'],\n            \'adapter_names\': sorted(adapters.adapters),\n            \'expected_repair_model\': \'p013_behavior_lora\' if expected_adapter_names else spec[\'served_model\'],\n            \'expected_review_model\': spec[\'served_model\']})\n        for adapter_name in adapters.adapters:\n            for alias in (adapter_name, \'adapter:\' + adapter_name):\n                models.register(alias, LiteLlm(model=\'openai/\' + adapter_name,\n                    api_base=spec[\'base_url\'], api_key=\'EMPTY\'))\n        limits, generation_constraints = build_submission_limits()\n        budgets = spec[\'budgets\']\n        config = EvalConfig(\n            tasks_path=attempt / \'agent-task.json\', snapshots_dir=snapshot.parent,\n            results_dir=attempt, submission_dir=Path(spec[\'agent_dir\']).resolve(),\n            models=models, sandbox=\'subprocess\',\n            timeout_seconds=int(budgets.get(\'timeout_seconds\', 240)),\n            verification_timeout_seconds=int(spec[\'judge\'].get(\'timeout_seconds\', 120)),\n            max_tool_calls=int(budgets[\'max_tool_calls\']),\n            max_time_minutes=float(budgets[\'max_time_minutes\']),\n            max_turns=int(budgets[\'max_turns\']),\n            limits=limits, generation_constraints=generation_constraints, adapter_manifest=adapters,\n            context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),\n            events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,\n                                                           token_threshold=14336, event_retention_size=5),\n            graph_dir=str(scratch / \'absent-graphs\'), embeddings_dir=str(scratch / \'absent-embeddings\'),\n            verbose=False, display_mode=\'quiet\', enable_sandbox_testing=True)\n        # This metadata contains only the issue given to the Agent, with no criterion.\n        atomic_json(attempt / \'agent-task.json\', {\n            \'instance_id\': task.instance_id, \'repo\': task.repo,\n            \'base_commit\': task.base_commit, \'problem_statement\': task.problem_statement})\n        manager = SourceManager(timeout_seconds=config.harness.command_timeout_seconds,\n                                base_dir=scratch / \'agent-sandboxes\', system_site_packages=True)\n        manager.base_dir.mkdir(parents=True, exist_ok=True)\n        context = SwegemmaContext(docker_manager=manager, task=task,\n            graph_dir=config.graph_dir, embeddings_dir=config.embeddings_dir,\n            budget=config.budget, harness=config.harness)\n        phase(\'agent\')\n        patch, agent_error, trace = asyncio.run(agent_runner.run_agent_sandbox(\n            docker=manager, config=config, task=task, snapshot_path=snapshot,\n            task_index=int(entry.get(\'ordinal\', 1)), total_tasks=int(spec.get(\'total_tasks\', 1)),\n            context=context))\n        patch = patch or \'\'\n        atomic_text(attempt / \'agent.patch\', patch)\n        scored_patch = strip_protected_files_from_patch(patch)\n        atomic_text(attempt / \'scored.patch\', scored_patch)\n        trace_path = save_trace_artifact(trace, attempt, task.instance_id)\n        context_metrics = {key: getattr(context, key, None) for key in (\n            \'task_start_time\', \'agent_start_time\', \'agent_end_time\', \'agent_elapsed_seconds\',\n            \'task_elapsed_seconds\', \'patch_submitted\', \'tool_calls_used\', \'llm_calls_used\')}\n        atomic_json(attempt / \'context_metrics.json\', context_metrics)\n\n        phase(\'fresh_judge\')\n        # SourceManager\'s final-status hook is Agent-specific, so use a separate\n        # minimal source-prefix subclass for the independent judge workspace.\n        class JudgeManager(SubprocessManager):\n            def exec(self, sandbox_id, command, *, timeout=None):\n                return super().exec(sandbox_id,\n                    \'export PYTHONPATH=/workspace/src:/workspace; \'\n                    \'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; \' + command, timeout=timeout)\n        judge_manager = JudgeManager(timeout_seconds=120,\n            base_dir=scratch / \'judge-sandboxes\', system_site_packages=True)\n        judge_manager.base_dir.mkdir(parents=True, exist_ok=True)\n        judge_id = judge_manager.start()\n        extract_snapshot(judge_manager, judge_id, snapshot)\n        head = judge_manager.exec(judge_id, \'git rev-parse HEAD\', timeout=10)\n        atomic_json(attempt / \'judge-source.json\', {\n            \'expected_base_commit\': task.base_commit, \'actual_base_commit\': (head.stdout or \'\').strip(),\n            \'snapshot\': str(snapshot), \'snapshot_sha256\': hashlib.sha256(snapshot.read_bytes()).hexdigest()})\n        expected = judge_manager.exec(judge_id, \'git rev-parse --verify \' +\n            shlex.quote(task.base_commit + \'^{commit}\'), timeout=10)\n        if (head.exit_code or expected.exit_code or\n                (head.stdout or \'\').strip() != (expected.stdout or \'\').strip()):\n            raise RuntimeError(\'judge snapshot base commit does not match frozen task\')\n        if scored_patch.strip():\n            judge_manager.copy_to(judge_id, attempt / \'scored.patch\', \'/tmp/agent.patch\')\n            code, out, err = apply_patch_in_container(judge_manager, judge_id, \'/tmp/agent.patch\')\n            atomic_json(attempt / \'patch-application.json\', {\'exit_code\': code, \'stdout\': out, \'stderr\': err})\n            if code:\n                payload = dict(entry, instance_id=task.instance_id, repo=task.repo, resolved=False,\n                    agent_patch_size=len(patch.encode()), agent_error=agent_error,\n                    judge_valid=True, judge_error=\'agent patch could not apply to clean base\',\n                    tool_calls=context_metrics.get(\'tool_calls_used\'), trace_json_path=trace_path)\n                atomic_json(attempt / \'result.json\', payload)\n                status.update(status=\'completed\', resolved=False)\n                return 0\n\n        criterion = spec[\'judge\']\n        module_name = criterion.get(\'module\', task.repo.rsplit(\'/\', 1)[-1])\n        source = criterion[\'criterion_source\']\n        node = criterion[\'f2p_node\']\n        criterion_payload = json.dumps({\'source\': source, \'node\': node, \'module\': module_name})\n        python = shlex.quote(spec.get(\'runtime_python\', \'python3\'))\n        command = python + \' -c \' + shlex.quote(CRITERION_PROGRAM) + \' \' + shlex.quote(criterion_payload)\n        f2p = judge_manager.exec(judge_id, command, timeout=int(criterion.get(\'timeout_seconds\', 120)))\n        atomic_json(attempt / \'criterion-execution.json\', result_record(f2p))\n        report = None\n        for line in (f2p.stdout or \'\').splitlines():\n            if line.startswith(\'P014_CRITERION_RESULT=\'):\n                report = json.loads(line.partition(\'=\')[2])\n        atomic_json(attempt / \'criterion-result.json\', report)\n        judge_valid = bool(report and report.get(\'source_valid\') and\n                           not report.get(\'external_dependency_missing\'))\n        p2p_nodes = criterion.get(\'p2p_nodes\', [])\n        p2p_pass = None\n        if p2p_nodes:\n            p2p_command = python + \' -m pytest -q -x -p no:cacheprovider \' + \' \'.join(map(shlex.quote, p2p_nodes))\n            p2p = judge_manager.exec(judge_id, p2p_command, timeout=int(criterion.get(\'timeout_seconds\', 120)))\n            atomic_json(attempt / \'p2p-execution.json\', result_record(p2p))\n            p2p_pass = p2p.exit_code == 0\n            if p2p.exit_code in (2, 3, 4, 5) or getattr(p2p, \'timed_out\', False):\n                judge_valid = False\n        resolved = (bool(f2p.exit_code == 0 and report and report.get(\'criterion_pass\') and\n                         p2p_pass is not False) if judge_valid else None)\n        payload = dict(entry, instance_id=task.instance_id, repo=task.repo,\n            resolved=resolved, agent_patch_size=len(patch.encode()), agent_error=agent_error,\n            test_exit_code=f2p.exit_code, judge_valid=judge_valid,\n            criterion_pass=bool(report and report.get(\'criterion_pass\')),\n            p2p_pass=p2p_pass, p2p_nodes=p2p_nodes, module_source=report,\n            duration_seconds=time.monotonic() - start,\n            tool_calls=context_metrics.get(\'tool_calls_used\'),\n            total_llm_calls=context_metrics.get(\'llm_calls_used\'), trace_json_path=trace_path)\n        atomic_json(attempt / \'result.json\', payload)\n        status.update(status=\'completed\' if judge_valid else \'judge_environment_error\', resolved=resolved)\n        return 0 if judge_valid else 1\n    except BaseException as exc:\n        status.update(status=\'error\', error_type=type(exc).__name__, error=str(exc))\n        atomic_text(attempt / \'exception.txt\', traceback.format_exc())\n        if not (attempt / \'result.json\').exists():\n            atomic_json(attempt / \'result.json\', dict(entry, resolved=None,\n                error_type=type(exc).__name__, error=str(exc)))\n        return 1\n    finally:\n        for instance in (manager, judge_manager):\n            if instance is not None:\n                try:\n                    instance.cleanup_all()\n                except Exception:\n                    atomic_text(attempt / \'cleanup-error.txt\', traceback.format_exc())\n        status.update(finished_at=time.time(), outer_seconds=time.monotonic() - start)\n        atomic_json(attempt / \'attempt_status.json\', status)\n        logging.getLogger().removeHandler(handler)\n        handler.close()\n\n\nif __name__ == \'__main__\':\n    raise SystemExit(execute(json.loads(Path(sys.argv[1]).read_text())))\n'
"""Run the frozen P014 paired repair experiment on one continuously batched server.

This is an experiment runner, not a test suite. Inputs and answers stay outside
the Agent sandbox; only issue text and the baseline repository enter it.
The launch file supplies SETTINGS, BUNDLE and WORKER_SOURCE as frozen literals.
This 12B derivative adds repair-only P013 adapter routing and the official 12B override.
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


"""12B routing component for the shared P014 inference carrier.

No model is loaded and no task is run on import. The carrier embeds this source,
the frozen agent text files and both JSON pins in its single uploaded code_file.
"""

from pathlib import Path
import hashlib
import json
import shutil


MODEL_ALIAS = "gemma-4-12b-it-qat-w4a16-ct"
LORA_NAME = "p013_behavior_lora"
ARMS = ("base", "adapter")
MAX_CONCURRENT = 4
VLLM_EXTRA_ARGS = [
    "--max-num-seqs", "4", "--hf-overrides",
    '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}',
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_adapter(input_root, agent_root, adapter_source):
    """Stage only the independently exported final adapter, never checkpoints.

    The SFT kernel output contains several adapters. Restrict to its final
    training/adapter directory, then verify both exact frozen file hashes.
    """
    input_root, agent_root = Path(input_root), Path(agent_root)
    source_slug = adapter_source["source_kernel"].split("/")[1]
    candidates = [
        path for path in input_root.rglob("adapter_model.safetensors")
        if source_slug in path.parts and path.parent.name == "adapter"
        and path.parent.parent.name == "training"
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected one final P013 training/adapter output; found "
            + str([str(path) for path in candidates])
        )
    src = candidates[0].parent
    dest = agent_root / "adapter" / "adapters" / LORA_NAME
    dest.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, pin in adapter_source["files"].items():
        source = src / name
        actual = {"bytes": source.stat().st_size, "sha256": sha256_file(source)}
        if actual != pin:
            raise RuntimeError("P013 adapter differs from frozen source: " + name)
        shutil.copyfile(source, dest / name)
        files[name] = actual
    config = json.loads((dest / "adapter_config.json").read_text())
    if config["r"] != 8 or config["lora_alpha"] != 16:
        raise RuntimeError("Unexpected trained LoRA rank/alpha")
    if (agent_root / "base" / "adapters").exists():
        raise RuntimeError("Base arm must not contain an adapter")
    return {"source": str(src), "staged": str(dest), "files": files,
            "lora_name": LORA_NAME, "repair_only": True}


def prove_role_routing(agent_root):
    """Return the role mapping after checking the actual staged YAML files."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("!include", lambda loader, node: loader.construct_scalar(node))
    root = Path(agent_root)
    repairs = {}
    for arm in ARMS:
        agent = yaml.load((root / arm / "agent.yaml").read_text(), Loader=Loader)
        repair = yaml.load((root / arm / "repair.yaml").read_text(), Loader=Loader)
        review = yaml.load((root / arm / "review.yaml").read_text(), Loader=Loader)
        if agent["agent_class"] != "SequentialAgent" or agent.get("adapter"):
            raise RuntimeError("Root must be a base sequential workflow")
        expected = LORA_NAME if arm == "adapter" else None
        if repair.get("adapter") != expected or review.get("adapter") is not None:
            raise RuntimeError("LoRA must route only to the adapter arm repair role")
        if repair["model"] != MODEL_ALIAS or review["model"] != MODEL_ALIAS:
            raise RuntimeError("12B model identity mismatch")
        if repair["tools"] != [
            "run_command", "read_file", "edit_file", "write_file", "get_status"
        ]:
            raise RuntimeError("Repair must retain the five-tool training protocol")
        if review["include_contents"] != "none":
            raise RuntimeError("Review context separation changed")
        prompt = (root / arm / "prompts/review.md").read_text()
        if "{problem_description}" not in prompt or "{repair_handoff}" not in prompt:
            raise RuntimeError("Review is missing explicit issue or handoff")
        repairs[arm] = repair
    normalized_adapter = dict(repairs["adapter"])
    normalized_adapter.pop("adapter")
    if normalized_adapter != repairs["base"]:
        raise RuntimeError("Repair arms differ beyond the intended LoRA reference")
    for rel in ("agent.yaml", "review.yaml", "prompts/repair.md",
                "prompts/review.md", "configs/sampling.yaml", "eval_config.yaml"):
        if (root / "base" / rel).read_bytes() != (root / "adapter" / rel).read_bytes():
            raise RuntimeError("Unexpected arm difference: " + rel)
    return {"base": {"repair": MODEL_ALIAS, "review": MODEL_ALIAS},
            "adapter": {"repair": LORA_NAME, "review": MODEL_ALIAS}}


def register_worker_aliases(models, agent_dir, served_model, base_url):
    """Use the same adapter aliases as official VllmServer/ModelRegistry."""
    from adk_submission import discover_adapters
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

    models.register(MODEL_ALIAS, LiteLlm(
        model="openai/" + served_model, api_base=base_url, api_key="EMPTY"))
    adapters = discover_adapters(
        agent_dir, adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    for name in adapters.adapters:
        for alias in (name, "adapter:" + name):
            models.register(alias, LiteLlm(
                model="openai/" + name, api_base=base_url, api_key="EMPTY"))
    return adapters

ADAPTER_SOURCE = {'source_kernel': 'ccjjllt/p013-12b-behavior-sft-34/1', 'lora_name': 'p013_behavior_lora', 'files': {'adapter_config.json': {'bytes': 1192, 'sha256': '443b33985cdc21066672bc529b72facdb4a205678944315e7cbee8696b6a4dee'}, 'adapter_model.safetensors': {'bytes': 115020728, 'sha256': '54de0ca38cdcb284509f7678cd59077217b169e3ee84649e78791faa8b5ae2a3'}}, 'training_complete': True, 'runtime_compatibility_verified': False}
MODEL_PIN = {'ref': 'google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2', 'model_alias': 'gemma-4-12b-it-qat-w4a16-ct', 'tokenizer_bytes': 32169626, 'tokenizer_sha256': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f', 'config_sha256': 'ba14713d084391532b285f7c252b954b7c6c0db97f427a9d8fb788edc2949168', 'tokenizer_config_sha256': 'ee66825f1a587ca13bcb90d7e60f59bc99e84f3b8bc0cefdc3ea595cd5abf773', 'vllm_extra_args': ['--max-num-seqs', '4', '--hf-overrides', '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}']}


def main():
    started = globals().get('P014_SESSION_START', time.monotonic())
    run_id = 'run-014-' + SETTINGS['study'] + '-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    out = Path('/kaggle/working/results/014') / run_id
    out.mkdir(parents=True)
    deadline = started + min(6900, SETTINGS.get('wall_seconds', 6900))
    tasks_path = locate('agent-tasks.json')
    panel = tasks_path.parent
    task_document = json.loads(tasks_path.read_text())
    tasks = task_document if isinstance(task_document, list) else task_document['tasks']
    by_id = {t['instance_id']: t for t in tasks}
    selected = SETTINGS['task_ids']
    if len(selected) != 4 or len(set(selected)) != 4:
        raise RuntimeError('12B first batch requires exactly four task IDs fixed before launch')
    if SETTINGS['model_alias'] != MODEL_ALIAS:
        raise RuntimeError('12B model alias mismatch')
    tasks = [by_id[iid] for iid in selected]
    criteria_document = json.loads((panel / 'host-criteria.json').read_text())
    criterion_items = criteria_document if isinstance(criteria_document, list) else criteria_document['items']
    criteria = {item['instance_id']: item for item in criterion_items}
    arms = list(BUNDLE)
    if arms != list(ARMS):
        raise RuntimeError('12B arm order must be base, adapter')
    agents = out / 'frozen-agents'
    for arm, files in BUNDLE.items():
        for rel, text in files.items():
            save(agents / arm / rel, text)
    adapter_identity = stage_adapter('/kaggle/input', agents, ADAPTER_SOURCE)
    role_routing = prove_role_routing(agents)
    manifest = {'run_id': run_id, 'settings': SETTINGS, 'task_ids': selected,
                'task_input_sha256': digest(tasks_path), 'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'config_sha256': {arm: {rel: hashlib.sha256(text.encode()).hexdigest() for rel, text in files.items()} for arm, files in BUNDLE.items()},
                'adapter_source': adapter_identity, 'role_routing': role_routing,
                'model_pin': MODEL_PIN, 'planned_attempts': len(tasks) * len(arms), 'concurrency': SETTINGS.get('concurrency', 4),
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
    for filename, key in [('tokenizer.json', 'tokenizer_sha256'), ('config.json', 'config_sha256'), ('tokenizer_config.json', 'tokenizer_config_sha256')]:
        if digest(model / filename) != MODEL_PIN[key]:
            raise RuntimeError('12B CT v2 source differs: ' + filename)
    if (model / 'tokenizer.json').stat().st_size != MODEL_PIN['tokenizer_bytes']:
        raise RuntimeError('12B tokenizer byte count differs')
    manifest['hardware'] = [{'name': torch.cuda.get_device_name(i), 'memory': torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    if len(manifest['hardware']) != 4 or any('L4' not in x['name'] for x in manifest['hardware']):
        raise RuntimeError(f'Requires four L4s, got {manifest["hardware"]}')
    adapter_manifest = discover_adapters(str(agents / arms[-1]), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    if sorted(adapter_manifest.adapters) != [LORA_NAME]:
        raise RuntimeError('Expected only the P013 behavior LoRA in adapter arm')
    base_manifest = discover_adapters(str(agents / 'base'), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    if base_manifest.adapters:
        raise RuntimeError('Base arm unexpectedly contains adapters')
    config = VllmConfig(model=str(model), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
                        max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90,
                        enable_auto_tool_choice=True, enable_lora=True, tensor_parallel_size=4,
                        startup_timeout=1200, extra_args=['--max-num-seqs', str(SETTINGS.get('concurrency', 4)), *VLLM_EXTRA_ARGS[2:]])
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
        summary = {'completed_attempts': len(rows), 'planned_attempts': len(schedule), 'wall_seconds': time.monotonic()-started,
                   'arms': {arm: {'resolved': sum(r.get('resolved') is True and r['variant']==arm for r in rows), 'attempts': sum(r['variant']==arm for r in rows)} for arm in arms}}
        save(out / 'summary.json', summary)
        print('FINAL ' + json.dumps(summary), flush=True)
        with zipfile.ZipFile('/kaggle/working/p014-'+SETTINGS['study']+'-output.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in out.rglob('*'):
                if path.is_file() and path.suffix not in ('.safetensors', '.pt', '.bin'):
                    archive.write(path, path.relative_to(out.parent))


if __name__ == '__main__':
    main()
