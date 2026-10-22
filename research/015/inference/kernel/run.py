
from pathlib import Path
import importlib
import hashlib
import json
import os
import subprocess
import sys
import tarfile

os.environ.update({'LITELLM_LOCAL_MODEL_COST_MAP':'True', 'TRANSFORMERS_NO_TF':'1',
                   'VLLM_WORKER_MULTIPROC_METHOD':'spawn', 'VLLM_NO_USAGE_STATS':'1',
                   'VLLM_ENGINE_READY_TIMEOUT_S':'1200', 'OTEL_SDK_DISABLED':'true',
                   'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS':'1',
                   'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
inputs = Path('/kaggle/input')
markers = list(inputs.rglob('swegemma-0.2.11-py3-none-any.whl'))
if len(markers) != 1:
    raise RuntimeError('Official wheelhouse mount missing/ambiguous: '+str(markers))
wheelhouse = markers[0].parent
staging = Path('/tmp/p015-official-wheels')
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
archives = list(inputs.rglob('prepared-panel-material.tar.gz'))
if len(archives) != 1:
    raise RuntimeError('CPU-prepared input archive missing/ambiguous: '+str(archives))
if archives[0].stat().st_size != 81303711 or hashlib.sha256(archives[0].read_bytes()).hexdigest() != 'eeebb65c1a03d71d42ed29d6d081b27af59a78a9c1c32ab7593d69b31f6ff8cc':
    raise RuntimeError('Prepared material differs from CPU-validated frozen bytes')
MATERIAL_ROOT = '/tmp/p015-agent-material'
material = Path(MATERIAL_ROOT)
material.mkdir(exist_ok=False)
with tarfile.open(archives[0]) as archive:
    archive.extractall(material, filter='data')
importlib.invalidate_caches()
print('PHASE inputs_ready', flush=True)

SETTINGS = {'study': '31b-base-and-train-sampling-v1', 'model_alias': 'gemma-4-31b-it-qat-w4a16-ct', 'model_path_pattern': 'gemma-4-31b-it-qat-w4a16-ct/2', 'task_ids': ['martinblech__xmltodict-81', 'mahmoud__boltons-338', 'mahmoud__boltons-298', 'tobymao__sqlglot-1001', 'arrow-py__arrow-986', 'asottile__pyupgrade-530', 'joke2k__faker-2190', 'tobymao__sqlglot-2047', 'arrow-py__arrow-1035', 'asottile__pyupgrade-133', 'joke2k__faker-1741', 'tobymao__sqlglot-3401', 'arrow-py__arrow-906', 'asottile__pyupgrade-656', 'joke2k__faker-1869'], 'task_count': 15, 'budgets': {'max_time_minutes': 5, 'max_tool_calls': 28, 'max_turns': 40, 'timeout_seconds': 60}, 'concurrency': 4, 'wall_seconds': 3600, 'attempt_seconds': 720, 'code_commit': 'ce1dc4cbda7328b913ff322749864afdf54f1370', 'quota_estimate_upper_hours': 2.3333333333333335, 'purpose': 'new dev base repairs plus separate train-only rollouts; no LoRA gain claim', 'selection_sha256': 'ae778d1967da550705bba3fec3945184b31dca205a83383b4774f198eae85c4b', 'capacity_adapter_diagnostic': {'source_kernel': 'zhuating/p015-fsdp-capacity', 'name': 'p015_capacity_diag', 'training_manifest_sha256': 'bc7f2a7c3e3db41b27fc49452d8306dddb745ed987244601e2ab3863c1dc222d', 'training_signature': '6a9081272028b26dbea13705e554f2e6168ea9abc7edcbfdc176b22e96ec4073', 'files': {'adapter_config.json': {'bytes': 16897, 'sha256': 'a48cf95e43ea4972c8af7adfab022776b9ec5d49c1faba76d4ec61aa35b07622'}, 'adapter_model.safetensors': {'bytes': 212261592, 'sha256': 'f1d7cfe7479a26b91d2f29b1bba45191368c4575f7b22be42f3b268963173563'}}, 'purpose': 'one hello request only; excluded from repair statistics; not task-SFT evaluation'}}
BUNDLE = {'base': {'agent.yaml': 'name: p014_single_v2_runtime\nmodel: gemma-4-31b-it-qat-w4a16-ct\ninstruction: !include prompts/system.md\ntools:\n  - run_command\n  - read_file\n  - edit_file\n  - write_file\n  - get_status\n  - submit_patch\ngenerate_content_config: !include configs/sampling.yaml\n', 'configs/sampling.yaml': 'temperature: 0.2\ntop_p: 0.95\nmax_output_tokens: 8192\nthinking_config:\n  thinking_budget: 4096\n  include_thoughts: true\n', 'eval_config.yaml': 'evaluation:\n  timeout_seconds: 60\n  max_tool_calls: 28\n  max_time_minutes: 5\n  max_turns: 40\n', 'prompts/system.md': "You are an autonomous software engineer. Make the smallest complete production-source change in /workspace that satisfies the original issue. The issue specifies the expected names, types, messages, return values and edge cases.\n\n# LIMITS AND PRIORITIES\n- You have 28 tool calls, at most 40 model turns and 5 minutes in total. Wall clock includes tool round trips. The working tree is graded when time expires; an existing edit can count without submit_patch. Finish the production change while there is still time to verify it.\n- Never create, modify or delete test files (test_*.py, *_test.py, anything under tests/). Never run a full test suite. Never look outside /workspace.\n- Spend the initial calls locating and reading the relevant implementation and contract. Aim to complete the production edit by call 10; reserve the last part of the budget for a bounded check, correction and final diff. Do not keep exploring or writing repro variants after the cause is supported by the source.\n\n# UNDERSTAND AND LOCATE\n- Extract the issue's exact API/CLI names, required behavior and every stated edge case. For a feature, find the closest existing sibling and its registration/export/call sites. For a bug, find where the wrong behavior is produced.\n- Use paths and identifiers in the issue directly. Batch independent git grep searches with short output; reuse paths already found. Read the relevant source plus one caller or existing test contract. Avoid repository dumps.\n- Before changing several related locations, read each required region. Cover sibling branches, imports/exports, dispatch entries and argument forwarding together when the issue requires them. Do not edit an unseen branch from memory.\n- Run a pre-edit reproduction only when it distinguishes a concrete hypothesis. Use one inline command, never a new repro file; then proceed to the production change.\n\n# EDIT COMPLETELY\n- Use edit_file with all three required arguments: filepath, old_string and new_string. Copy a short unique old_string exactly from the source; even an insertion requires old_string. Use write_file only for a genuinely new production file required by the issue.\n- If a tool rejects missing or invalid arguments, read its required parameter names and correct the request once. Never repeat the same rejected request. If it fails again, use one equivalent run_command operation instead.\n- After a failed anchor match, read the actual region before retrying. After two matching failures, use one inline Python replacement with assert text.count(old) == 1; preserve unrelated content and inspect the resulting diff.\n- Fix the cause rather than special-casing the example. Preserve existing signatures, defaults and compatibility unless the issue requests a change. Follow neighboring conventions and handle all stated cases.\n- A production edit is the main deliverable. Do not spend the remaining budget repeatedly adjusting a reproduction while leaving the implementation unchanged.\n\n# VERIFY THE MODIFIED SOURCE\n- Prefer an existing focused test or a short inline behavior check. Run Python from /workspace with PYTHONPATH=/workspace/src:/workspace. In the same command, print the target module's __file__ once and confirm it resolves under /workspace before trusting the behavior result. A system-installed copy cannot validate your edit.\n- Bound a short inline reproduction with timeout 10. For a named existing test file use timeout 45, for example: PYTHONPATH=/workspace/src:/workspace timeout 45 python3 -m pytest tests/path/test_x.py -x -q -k keyword. Never run bare pytest or an entire repository suite.\n- Exit 124 means the command did not finish. Treat it as evidence of a hang or an incomplete observation; do not revert a supported edit solely because of that exit code. Avoid another long blocked command.\n- Distinguish an implementation failure from a wrong witness command or unavailable external dependency/fixture. Correct your command when its arguments or import path are wrong. A missing symbol in the target project can be the issue itself: do not dismiss every ImportError as an environment problem.\n- If a focused check exposes a specific defect in the edit, correct the source and rerun that check once. Do not repair tests or dependencies and do not begin a second broad exploration. If no behavior check can run, inspect the complete diff and state the verification limit honestly.\n\n# FINISH\n- Run git status --short and git diff. Confirm the diff covers the original requirement and associated branches, only necessary production files changed, and no debug code or temporary products from this attempt remain.\n- Remove only scratch files that you created during this attempt and have identified as temporary. Never delete pre-existing files or tests using broad globs.\n- Call submit_patch after inspecting the real diff. If it reports patch_size == 0 or files_changed == 0, inspect the tree and make the missing production edit before another submit. Never repeat a zero-patch submission without an intervening edit and new diff evidence.\n- When the tool or time budget is exhausted, do not make further tool requests. After a successful non-empty submit_patch, stop immediately; do not restart verification.\n- End with a short accurate summary of the fix and the checks actually run. A clean submission action alone does not establish correctness.\n"}}
WORKER_SOURCE = '"""Run one external development task with the official Agent runner and a fresh judge.\n\nThe Agent receives only the original issue and buggy snapshot. Dataset-provided\nvalidation patches and existing F2P/P2P nodes stay in the host judge workspace.\n"""\nfrom __future__ import annotations\n\nimport asyncio\nimport copy\nimport contextvars\nimport functools\nimport importlib.metadata\nimport inspect\nimport tarfile\nimport hashlib\nimport json\nimport logging\nimport os\nfrom pathlib import Path\nimport shlex\nimport sys\nimport time\nimport traceback\nimport uuid\n\n\ndef atomic_text(path: Path, value: str) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    temporary = path.with_name(path.name + \'.tmp.\' + uuid.uuid4().hex)\n    temporary.write_text(value, encoding=\'utf-8\')\n    temporary.replace(path)\n\n\ndef atomic_json(path: Path, value: object) -> None:\n    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + \'\\n\')\n\n\ndef result_record(result: object) -> dict:\n    return {key: getattr(result, key, None) for key in\n            (\'exit_code\', \'stdout\', \'stderr\', \'duration_seconds\', \'timed_out\', \'status\')}\n\n\nclass TrajectoryCapture:\n    """Append raw inference records without changing the Agent or its messages."""\n\n    def __init__(self, attempt: Path):\n        self.attempt = attempt\n        self.counts = {}\n        self.errors = []\n        self.hooks = []\n        self.restores = []\n        self.request_id = contextvars.ContextVar(\'p015_model_request_id\', default=None)\n        self.versions = {}\n        for name in (\'google-adk\', \'google-genai\', \'adk-submission\', \'adk-eval-core\', \'litellm\', \'pydantic\'):\n            try:\n                self.versions[name] = importlib.metadata.version(name)\n            except importlib.metadata.PackageNotFoundError:\n                self.versions[name] = None\n\n    def clean(self, value):\n        if hasattr(value, \'model_dump\'):\n            value = value.model_dump(mode=\'json\', exclude_none=False)\n        if isinstance(value, dict):\n            return {str(k): (\'[REDACTED]\' if str(k).lower().replace(\'-\', \'_\') in\n                    (\'api_key\', \'authorization\', \'proxy_authorization\', \'headers\',\n                     \'extra_headers\', \'default_headers\', \'cookies\', \'access_token\')\n                    else self.clean(v)) for k, v in value.items()}\n        if isinstance(value, (list, tuple)):\n            return [self.clean(v) for v in value]\n        if value is None or isinstance(value, (str, int, float, bool)):\n            return value\n        raise TypeError(\'Non-JSON inference value: \' + type(value).__name__)\n\n    def emit(self, channel, kind, **payload):\n        try:\n            record = self.clean(dict(kind=kind, timestamp=time.time(),\n                                     model_request_id=self.request_id.get(), **payload))\n            # An open/write/flush per record survives worker interruption and never\n            # retains references to mutable ADK request/response objects.\n            with (self.attempt / (\'raw-\' + channel + \'.jsonl\')).open(\'a\', encoding=\'utf-8\') as stream:\n                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + \'\\n\')\n                stream.flush()\n            self.counts[kind] = self.counts.get(kind, 0) + 1\n        except Exception as exc:\n            self.errors.append(dict(channel=channel, kind=kind, error_type=type(exc).__name__, error=str(exc)))\n\n    def patch(self, cls, name, replacement):\n        original = getattr(cls, name)\n        self.hooks.append(dict(target=cls.__module__ + \'.\' + cls.__name__ + \'.\' + name,\n                               signature=str(inspect.signature(original)),\n                               source_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest()))\n        self.restores.append((cls, name, original))\n        setattr(cls, name, replacement(original))\n\n    def install(self):\n        from google.adk.models.lite_llm import LiteLlm, LiteLLMClient\n        from adk_eval_core.tracing import SessionTrace\n        capture = self\n\n        def model_wrapper(original):\n            @functools.wraps(original)\n            async def generate(model, llm_request, stream=False):\n                rid = uuid.uuid4().hex\n                token = capture.request_id.set(rid)\n                completed = False\n                capture.emit(\'model\', \'model_request\', request=llm_request, stream=stream)\n                try:\n                    async for response in original(model, llm_request, stream=stream):\n                        capture.emit(\'model\', \'model_response\', response=response)\n                        yield response\n                    completed = True\n                finally:\n                    capture.emit(\'model\', \'model_end\', completed=completed)\n                    if not completed:\n                        capture.errors.append(dict(kind=\'model_interrupted\', model_request_id=rid))\n                    capture.request_id.reset(token)\n            return generate\n\n        def wire_wrapper(original):\n            @functools.wraps(original)\n            async def complete(client, model, messages, tools, **kwargs):\n                wire_id = uuid.uuid4().hex\n                capture.emit(\'wire\', \'wire_request\', wire_id=wire_id,\n                             request=dict(model=model, messages=messages, tools=tools, **kwargs))\n                try:\n                    response = await original(client, model, messages, tools, **kwargs)\n                    if kwargs.get(\'stream\'):\n                        async def chunks():\n                            completed = False\n                            try:\n                                async for chunk in response:\n                                    capture.emit(\'wire\', \'wire_chunk\', wire_id=wire_id, response=chunk)\n                                    yield chunk\n                                completed = True\n                            finally:\n                                capture.emit(\'wire\', \'wire_end\', wire_id=wire_id, completed=completed)\n                                if not completed:\n                                    capture.errors.append(dict(kind=\'wire_stream_interrupted\', wire_id=wire_id))\n                        return chunks()\n                    capture.emit(\'wire\', \'wire_response\', wire_id=wire_id, response=response)\n                    capture.emit(\'wire\', \'wire_end\', wire_id=wire_id, completed=True)\n                    return response\n                except BaseException as exc:\n                    capture.emit(\'wire\', \'wire_error\', wire_id=wire_id, error_type=type(exc).__name__)\n                    capture.errors.append(dict(kind=\'wire_error\', wire_id=wire_id, error_type=type(exc).__name__))\n                    raise\n            return complete\n\n        def event_wrapper(original):\n            @functools.wraps(original)\n            def record(trace, event):\n                capture.emit(\'adk-events\', \'adk_event\', event=event)\n                return original(trace, event)\n            return record\n\n        self.patch(LiteLlm, \'generate_content_async\', model_wrapper)\n        self.patch(LiteLLMClient, \'acompletion\', wire_wrapper)\n        self.patch(SessionTrace, \'record_event\', event_wrapper)\n\n    def finish(self, trace_path, agent_error=None):\n        reasons = list(self.errors)\n        for name in (\'model_request\', \'model_response\', \'wire_request\', \'adk_event\'):\n            if not self.counts.get(name):\n                reasons.append(dict(kind=\'missing_capture\', record_kind=name))\n        for first, last in ((\'model_request\', \'model_end\'), (\'wire_request\', \'wire_end\')):\n            if self.counts.get(first, 0) != self.counts.get(last, 0):\n                reasons.append(dict(kind=\'incomplete_capture\', request_kind=first))\n        if not trace_path or not Path(trace_path).is_file() or not Path(trace_path).stat().st_size:\n            reasons.append(dict(kind=\'official_trace_not_saved\'))\n        if agent_error:\n            reasons.append(dict(kind=\'agent_error\', error=agent_error))\n        result = dict(training_trace_usable=not reasons, reasons=reasons,\n                      counts=self.counts, hooks=self.hooks, versions=self.versions,\n                      scope=\'Raw ADK request/response, post-conversion LiteLLM request/response, and raw ADK events; judge outcome remains independent\')\n        atomic_json(self.attempt / \'trajectory-capture.json\', result)\n        return result\n\n    def restore(self):\n        for cls, name, original in reversed(self.restores):\n            setattr(cls, name, original)\n        self.restores.clear()\n\n\ndef workspace_snapshot(source: Path, scratch: Path) -> Path:\n    """Flatten the CPU material\'s documented repo/ or workspace/ envelope.\n\n    The official extraction routine expects .git and source at archive root.\n    Preserve every original file and link; never add criterion/reference files.\n    """\n    with tarfile.open(source, \'r:*\') as archive:\n        members = archive.getmembers()\n        names = [m.name.removeprefix(\'./\') for m in members]\n        roots = {name.split(\'/\')[0] for name in names if name}\n        if roots not in ({\'repo\'}, {\'workspace\'}):\n            return source\n        prefix = next(iter(roots)) + \'/\'\n        target = scratch / \'workspace-snapshot.tar.gz\'\n        with tarfile.open(target, \'w:gz\') as output:\n            for member, name in zip(members, names):\n                if name.rstrip(\'/\') == prefix.rstrip(\'/\'):\n                    continue\n                if not name.startswith(prefix):\n                    raise RuntimeError(\'Snapshot archive has inconsistent envelope\')\n                item = copy.copy(member)\n                item.name = name[len(prefix):]\n                if item.islnk() and item.linkname.startswith(prefix):\n                    item.linkname = item.linkname[len(prefix):]\n                output.addfile(item, archive.extractfile(member) if member.isfile() else None)\n        return target\n\n\ndef execute(spec: dict) -> int:\n    entry = spec[\'entry\']\n    attempt = Path(entry[\'results_dir\']).resolve()\n    attempt.mkdir(parents=True, exist_ok=True)\n    safe_attempt_id = \'\'.join(ch if ch.isalnum() or ch in \'-_.\' else \'_\'\n                              for ch in str(entry.get(\'attempt_id\', attempt.name)))\n    scratch = Path(spec.get(\'scratch\', Path(\'/tmp\') / \'p015-attempts\' / safe_attempt_id)).resolve()\n    scratch.mkdir(parents=True, exist_ok=True)\n    status = dict(entry, status=\'running\', pid=os.getpid(), started_at=time.time())\n    start = time.monotonic()\n    manager = judge_manager = context = None\n    capture = None\n    capture_result = dict(training_trace_usable=False,\n                          training_trace_reasons=[dict(kind=\'agent_not_completed\')])\n    handler = logging.FileHandler(attempt / \'harness.log\', encoding=\'utf-8\')\n    handler.setFormatter(logging.Formatter(\'%(asctime)s %(levelname)s %(name)s %(message)s\'))\n    logging.getLogger().addHandler(handler)\n\n    def phase(name: str) -> None:\n        status.update(phase=name, phase_at=time.time())\n        atomic_json(attempt / \'attempt_status.json\', status)\n        print(json.dumps({\'attempt_id\': entry.get(\'attempt_id\'), \'phase\': name}), flush=True)\n\n    try:\n        phase(\'load_official_runtime\')\n        import litellm\n        from adk_submission import ModelRegistry, discover_adapters\n        from adk_submission.resolvers.generation import install_litellm_reasoning_patch\n        from google.adk.agents.context_cache_config import ContextCacheConfig\n        from google.adk.apps._configs import EventsCompactionConfig\n        from google.adk.models.lite_llm import LiteLlm\n        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits\n        from swegemma.context import SwegemmaContext\n        from swegemma.harness import agent_runner\n        from swegemma.harness.container_setup import extract_snapshot, apply_patch_in_container\n        from swegemma.harness.verification import save_trace_artifact, strip_protected_files_from_patch\n        from swegemma.models import Task\n        from swegemma.sandbox import SubprocessManager\n\n        class SourceManager(SubprocessManager):\n            def exec(self, sandbox_id, command, *, timeout=None):\n                prefix = (\'export PYTHONPATH=/workspace/src:/workspace; \'\n                          \'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; \')\n                prefix += (\'export PYTEST_PLUGINS=\' + shlex.quote(\n                    \',\'.join(environment.get(\'pytest_plugins\', []))) + \'; \')\n                return super().exec(sandbox_id, prefix + command, timeout=timeout)\n\n            def stop(self, sandbox_id):\n                if sandbox_id in self.sandboxes:\n                    try:\n                        state = self.exec(sandbox_id, \'git status --short\', timeout=10)\n                        atomic_text(attempt / \'agent-final-status.txt\',\n                                    (state.stdout or \'\') + (state.stderr or \'\'))\n                    except Exception:\n                        atomic_text(attempt / \'agent-final-status-error.txt\', traceback.format_exc())\n                super().stop(sandbox_id)\n\n        environment = spec[\'judge\'].get(\'environment\', {})\n\n        def prepared_dependencies(docker, sandbox_id, *_args, **_kwargs):\n            if not environment.get(\'requirements\'):\n                return\n            panel = Path(spec[\'panel_root\']).resolve()\n            wheelhouse = (panel / environment[\'wheelhouse\']).resolve()\n            requirements = (panel / environment[\'requirements\']).resolve()\n            if not wheelhouse.is_relative_to(panel) or not requirements.is_relative_to(panel):\n                raise RuntimeError(\'Dependency path escapes the frozen material root\')\n            if not requirements.is_file():\n                raise RuntimeError(\'CPU-prepared requirements missing\')\n            if not any(line.strip() and not line.lstrip().startswith(\'#\') for line in requirements.read_text().splitlines()):\n                return\n            if not wheelhouse.is_dir():\n                raise RuntimeError(\'CPU-prepared per-repository wheelhouse missing\')\n            for wheel in sorted(wheelhouse.glob(\'*.whl\')):\n                docker.copy_to(sandbox_id, wheel, \'/wheels/\' + wheel.name)\n            docker.copy_to(sandbox_id, requirements, \'/tmp/requirements.txt\')\n            result = docker.exec(sandbox_id,\n                \'python3 -m pip install --no-index --no-deps --find-links=/wheels \'\n                \'-r /tmp/requirements.txt\', timeout=120)\n            if result.exit_code:\n                raise RuntimeError(\'Offline dependency install failed: \' +\n                    (result.stdout or \'\')[-3000:] + (result.stderr or \'\')[-3000:])\n\n        def prepared_editable(docker, sandbox_id):\n            if not environment.get(\'install_editable\', False):\n                return\n            result = docker.exec(sandbox_id,\n                \'python3 -m pip install --no-index --no-build-isolation --no-deps \'\n                \'--find-links=/wheels -e /workspace\', timeout=90)\n            if result.exit_code:\n                raise RuntimeError(\'Editable source installation failed: \' +\n                    (result.stdout or \'\')[-3000:] + (result.stderr or \'\')[-3000:])\n\n        agent_runner.setup_container_wheels = prepared_dependencies\n        agent_runner.install_editable_package = prepared_editable\n        agent_runner.install_test_dependencies = lambda *_args, **_kwargs: None\n        agent_runner.setup_workspace_test_config = lambda *_args, **_kwargs: None\n        original_baseline = agent_runner.setup_baseline_commit\n\n        def baseline_with_local_identity(docker, sandbox_id, commit_msg=\'baseline\'):\n            configured = docker.exec(sandbox_id,\n                "git config user.email \'research@localhost\' && "\n                "git config user.name \'P015 research\'", timeout=10)\n            if configured.exit_code:\n                raise RuntimeError(\'cannot configure snapshot Git identity\')\n            original_baseline(docker, sandbox_id, commit_msg)\n            tagged = docker.exec(sandbox_id, \'git rev-parse _swegemma_baseline\', timeout=10)\n            if tagged.exit_code:\n                raise RuntimeError(\'official baseline commit was not created\')\n        agent_runner.setup_baseline_commit = baseline_with_local_identity\n\n        task_input = spec[\'task\']\n        task = Task(instance_id=task_input.get(\'instance_id\', task_input.get(\'iid\')),\n                    repo=task_input[\'repo\'],\n                    base_commit=task_input.get(\'base_commit\', task_input.get(\'basecommit\')),\n                    problem_statement=task_input.get(\'problem_statement\', task_input.get(\'issue\')))\n        assert not task.patch and not task.test_patch and not task.FAIL_TO_PASS and not task.PASS_TO_PASS\n        snapshot = Path(spec[\'snapshot\']).resolve()\n        if not snapshot.is_file():\n            raise FileNotFoundError(snapshot)\n        original_snapshot = snapshot\n        snapshot = workspace_snapshot(snapshot, scratch)\n        atomic_json(attempt / \'snapshot-envelope.json\', {\'source\': str(original_snapshot), \'source_sha256\': hashlib.sha256(original_snapshot.read_bytes()).hexdigest(), \'runtime_snapshot\': str(snapshot), \'runtime_sha256\': hashlib.sha256(snapshot.read_bytes()).hexdigest()})\n        install_litellm_reasoning_patch()\n        capture = TrajectoryCapture(attempt)\n        try:\n            capture.install()\n        except Exception as exc:\n            capture.errors.append(dict(kind=\'hook_installation_error\',\n                                       error_type=type(exc).__name__, error=str(exc)))\n            capture.restore()\n        # Record the actual runtime hook/version contract before the first call.\n        atomic_json(attempt / \'trajectory-capture-start.json\',\n                    dict(hooks=capture.hooks, versions=capture.versions, errors=capture.errors))\n        litellm.drop_params = True\n        models = ModelRegistry()\n        models.register(spec[\'model_alias\'], LiteLlm(\n            model=\'openai/\' + spec[\'served_model\'], api_base=spec[\'base_url\'], api_key=\'EMPTY\'))\n        adapters = discover_adapters(spec[\'agent_dir\'], adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)\n        for adapter_name in adapters.adapters:\n            for alias in (adapter_name, \'adapter:\' + adapter_name):\n                models.register(alias, LiteLlm(model=\'openai/\' + adapter_name,\n                    api_base=spec[\'base_url\'], api_key=\'EMPTY\'))\n        limits, generation_constraints = build_submission_limits()\n        budgets = spec[\'budgets\']\n        config = EvalConfig(\n            tasks_path=attempt / \'agent-task.json\', snapshots_dir=snapshot.parent,\n            results_dir=attempt, submission_dir=Path(spec[\'agent_dir\']).resolve(),\n            models=models, sandbox=\'subprocess\',\n            timeout_seconds=int(budgets.get(\'timeout_seconds\', 240)),\n            verification_timeout_seconds=int(spec[\'judge\'].get(\'timeout_seconds\', spec[\'judge\'].get(\'timeout\', 120))),\n            max_tool_calls=int(budgets[\'max_tool_calls\']),\n            max_time_minutes=float(budgets[\'max_time_minutes\']),\n            max_turns=int(budgets[\'max_turns\']),\n            limits=limits, generation_constraints=generation_constraints, adapter_manifest=adapters,\n            context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),\n            events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,\n                                                           token_threshold=14336, event_retention_size=5),\n            graph_dir=str(scratch / \'absent-graphs\'), embeddings_dir=str(scratch / \'absent-embeddings\'),\n            verbose=False, display_mode=\'quiet\', enable_sandbox_testing=True)\n        # This metadata contains only the issue given to the Agent, with no criterion.\n        atomic_json(attempt / \'agent-task.json\', {\n            \'instance_id\': task.instance_id, \'repo\': task.repo,\n            \'base_commit\': task.base_commit, \'problem_statement\': task.problem_statement})\n        manager = SourceManager(timeout_seconds=config.harness.command_timeout_seconds,\n                                base_dir=scratch / \'agent-sandboxes\', system_site_packages=True)\n        manager.base_dir.mkdir(parents=True, exist_ok=True)\n        context = SwegemmaContext(docker_manager=manager, task=task,\n            graph_dir=config.graph_dir, embeddings_dir=config.embeddings_dir,\n            budget=config.budget, harness=config.harness)\n        phase(\'agent\')\n        patch, agent_error, trace = asyncio.run(agent_runner.run_agent_sandbox(\n            docker=manager, config=config, task=task, snapshot_path=snapshot,\n            task_index=int(entry.get(\'ordinal\', 1)), total_tasks=int(spec.get(\'total_tasks\', 1)),\n            context=context))\n        patch = patch or \'\'\n        atomic_text(attempt / \'agent.patch\', patch)\n        scored_patch = strip_protected_files_from_patch(patch)\n        atomic_text(attempt / \'scored.patch\', scored_patch)\n        trace_path = save_trace_artifact(trace, attempt, task.instance_id)\n        captured = capture.finish(trace_path, agent_error)\n        capture_result = dict(training_trace_usable=captured[\'training_trace_usable\'],\n                              training_trace_reasons=captured[\'reasons\'],\n                              trajectory_capture_path=str(attempt / \'trajectory-capture.json\'))\n        context_metrics = {key: getattr(context, key, None) for key in (\n            \'task_start_time\', \'agent_start_time\', \'agent_end_time\', \'agent_elapsed_seconds\',\n            \'task_elapsed_seconds\', \'patch_submitted\', \'tool_calls_used\', \'llm_calls_used\')}\n        atomic_json(attempt / \'context_metrics.json\', context_metrics)\n\n        phase(\'fresh_judge\')\n        # SourceManager\'s final-status hook is Agent-specific, so use a separate\n        # minimal source-prefix subclass for the independent judge workspace.\n        class JudgeManager(SubprocessManager):\n            def exec(self, sandbox_id, command, *, timeout=None):\n                return super().exec(sandbox_id,\n                    \'export PYTHONPATH=/workspace/src:/workspace; \'\n                    \'export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; \' + command, timeout=timeout)\n        judge_manager = JudgeManager(timeout_seconds=120,\n            base_dir=scratch / \'judge-sandboxes\', system_site_packages=True)\n        judge_manager.base_dir.mkdir(parents=True, exist_ok=True)\n        judge_id = judge_manager.start()\n        extract_snapshot(judge_manager, judge_id, snapshot)\n        head = judge_manager.exec(judge_id, \'git rev-parse HEAD\', timeout=10)\n        atomic_json(attempt / \'judge-source.json\', {\n            \'expected_base_commit\': task.base_commit, \'actual_base_commit\': (head.stdout or \'\').strip(),\n            \'snapshot\': str(snapshot), \'snapshot_sha256\': hashlib.sha256(snapshot.read_bytes()).hexdigest()})\n        expected = judge_manager.exec(judge_id, \'git rev-parse --verify \' +\n            shlex.quote(task.base_commit + \'^{commit}\'), timeout=10)\n        if (head.exit_code or expected.exit_code or\n                (head.stdout or \'\').strip() != (expected.stdout or \'\').strip()):\n            raise RuntimeError(\'judge snapshot base commit does not match frozen task\')\n        if scored_patch.strip():\n            judge_manager.copy_to(judge_id, attempt / \'scored.patch\', \'/tmp/agent.patch\')\n            code, out, err = apply_patch_in_container(judge_manager, judge_id, \'/tmp/agent.patch\')\n            atomic_json(attempt / \'patch-application.json\', {\'exit_code\': code, \'stdout\': out, \'stderr\': err})\n            if code:\n                payload = dict(entry, **capture_result, instance_id=task.instance_id, repo=task.repo, resolved=False,\n                    agent_patch_size=len(patch.encode()), agent_error=agent_error,\n                    judge_valid=True, judge_error=\'agent patch could not apply to clean base\',\n                    tool_calls=context_metrics.get(\'tool_calls_used\'), trace_json_path=trace_path)\n                atomic_json(attempt / \'result.json\', payload)\n                status.update(status=\'completed\', resolved=False)\n                return 0\n\n        criterion = spec[\'judge\']\n        panel_root = Path(spec[\'panel_root\']).resolve()\n        criterion_patch = criterion.get(\'criterion_patch_path\')\n        if criterion_patch:\n            patch_path = (panel_root / criterion_patch).resolve()\n            if not patch_path.is_relative_to(panel_root):\n                raise RuntimeError(\'Criterion path escapes material root\')\n            if not patch_path.stat().st_size:\n                raise RuntimeError(\'Frozen criterion patch is empty; omit its path instead\')\n            judge_manager.copy_to(judge_id, patch_path, \'/tmp/criterion.patch\')\n            code, stdout, stderr = apply_patch_in_container(judge_manager, judge_id,\n                                                           \'/tmp/criterion.patch\')\n            atomic_json(attempt / \'criterion-patch-application.json\',\n                        {\'exit_code\': code, \'stdout\': stdout, \'stderr\': stderr})\n            if code:\n                raise RuntimeError(\'Frozen criterion patch could not apply\')\n        prepared_dependencies(judge_manager, judge_id)\n        prepared_editable(judge_manager, judge_id)\n        module = criterion[\'module\']\n        origin_code = (\'import importlib.util,json,pathlib; \'\n            \'s=importlib.util.find_spec(\' + repr(module) + \'); \'\n            \'p=pathlib.Path(s.origin).resolve() if s and s.origin else None; \'\n            \'print(json.dumps({"module":\' + repr(module) + \',"origin":str(p),\'\n            \'"source_valid":bool(p and p.is_relative_to(pathlib.Path.cwd().resolve()))}))\')\n        source = judge_manager.exec(judge_id, \'python3 -c \' + shlex.quote(origin_code), timeout=15)\n        report = None\n        if source.exit_code == 0:\n            report = json.loads((source.stdout or \'\').strip().splitlines()[-1])\n        atomic_json(attempt / \'module-source.json\', {\'execution\': result_record(source), \'report\': report})\n        if report and not report.get(\'source_valid\'):\n            # The clean base passed CPU source validation before model use.\n            # A removed/displaced candidate module must not be scored via a\n            # globally installed package or omitted as an environment failure.\n            payload = dict(entry, instance_id=task.instance_id, repo=task.repo,\n                resolved=False, agent_patch_size=len(patch.encode()), agent_error=agent_error,\n                judge_valid=True, judge_error=\'agent patch removed or displaced the task module\',\n                criterion_pass=False, p2p_pass=None, module_source=report,\n                duration_seconds=time.monotonic() - start,\n                tool_calls=context_metrics.get(\'tool_calls_used\'),\n                total_llm_calls=context_metrics.get(\'llm_calls_used\'), trace_json_path=trace_path)\n            atomic_json(attempt / \'result.json\', payload)\n            status.update(status=\'completed\', resolved=False)\n            return 0\n        timeout = int(criterion.get(\'timeout_seconds\', criterion.get(\'timeout\', 120)))\n        python = shlex.quote(spec.get(\'runtime_python\', \'python3\'))\n        f2p_nodes = criterion.get(\'f2p_nodes\', [])\n        p2p_nodes = criterion.get(\'p2p_nodes\', [])\n        if not f2p_nodes and not criterion.get(\'f2p_command\'):\n            raise RuntimeError(\'No frozen fail-to-pass nodes or command\')\n        pytest_prefix = python + \' -m pytest -q -p no:cacheprovider -o addopts= \'\n        plugin_flags = \'\'.join(\' -p \' + shlex.quote(name) for name in environment.get(\'pytest_plugins\', []))\n        f2p_command = criterion.get(\'f2p_command\') or (pytest_prefix + plugin_flags + \' \' + \' \'.join(map(shlex.quote, f2p_nodes)))\n        f2p = judge_manager.exec(judge_id, f2p_command, timeout=timeout)\n        atomic_json(attempt / \'f2p-execution.json\', {\'command\': f2p_command, **result_record(f2p)})\n        p2p_pass = None\n        if p2p_nodes or criterion.get(\'p2p_command\'):\n            p2p_command = criterion.get(\'p2p_command\') or (pytest_prefix + plugin_flags + \' \' + \' \'.join(map(shlex.quote, p2p_nodes)))\n            p2p = judge_manager.exec(judge_id, p2p_command, timeout=timeout)\n            atomic_json(attempt / \'p2p-execution.json\', {\'command\': p2p_command, **result_record(p2p)})\n            p2p_pass = p2p.exit_code == 0\n        # The original environment and criterion were validated before model use.\n        # Import/collection failures caused by the Agent\'s patch are real failures.\n        # Missing/foreign baseline assets fail earlier and are recorded separately.\n        judge_valid = True\n        resolved = bool(f2p.exit_code == 0 and p2p_pass is not False)\n        payload = dict(entry, **capture_result, instance_id=task.instance_id, repo=task.repo,\n            resolved=resolved, agent_patch_size=len(patch.encode()), agent_error=agent_error,\n            test_exit_code=f2p.exit_code, judge_valid=judge_valid,\n            criterion_pass=f2p.exit_code == 0,\n            p2p_pass=p2p_pass, p2p_nodes=p2p_nodes, module_source=report,\n            duration_seconds=time.monotonic() - start,\n            tool_calls=context_metrics.get(\'tool_calls_used\'),\n            total_llm_calls=context_metrics.get(\'llm_calls_used\'), trace_json_path=trace_path)\n        atomic_json(attempt / \'result.json\', payload)\n        status.update(status=\'completed\' if judge_valid else \'judge_environment_error\', resolved=resolved)\n        return 0 if judge_valid else 1\n    except BaseException as exc:\n        status.update(status=\'error\', error_type=type(exc).__name__, error=str(exc))\n        atomic_text(attempt / \'exception.txt\', traceback.format_exc())\n        if capture is not None and capture_result.get(\'training_trace_reasons\') == [dict(kind=\'agent_not_completed\')]:\n            captured = capture.finish(None, type(exc).__name__)\n            capture_result = dict(training_trace_usable=False,\n                                  training_trace_reasons=captured[\'reasons\'],\n                                  trajectory_capture_path=str(attempt / \'trajectory-capture.json\'))\n        if not (attempt / \'result.json\').exists():\n            atomic_json(attempt / \'result.json\', dict(entry, **capture_result, resolved=None,\n                error_type=type(exc).__name__, error=str(exc)))\n        return 1\n    finally:\n        if capture is not None:\n            capture.restore()\n        for instance in (manager, judge_manager):\n            if instance is not None:\n                try:\n                    instance.cleanup_all()\n                except Exception:\n                    atomic_text(attempt / \'cleanup-error.txt\', traceback.format_exc())\n        status.update(finished_at=time.time(), outer_seconds=time.monotonic() - start)\n        atomic_json(attempt / \'attempt_status.json\', status)\n        logging.getLogger().removeHandler(handler)\n        handler.close()\n\n\nif __name__ == \'__main__\':\n    raise SystemExit(execute(json.loads(Path(sys.argv[1]).read_text())))\n'
"""Run the frozen P015 paired repair experiment on one continuously batched server.

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
import urllib.error
import urllib.request
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


def prepare_capacity_diagnostic(settings, out, discover_adapters, extensions):
    record = {'purpose': settings['purpose'], 'name': settings['name'],
              'request_attempted': False, 'repair_statistics_included': False,
              'runtime_compatibility_verified': False}
    try:
        candidates = [path for path in Path('/kaggle/input').rglob('training_manifest.json')
                      if digest(path) == settings['training_manifest_sha256']]
        if len(candidates) != 1:
            raise RuntimeError('Completed capacity training manifest missing/ambiguous')
        source_manifest = candidates[0]
        actual = json.loads(source_manifest.read_text())
        if (actual['signature'] != settings['training_signature']
                or actual['adapter']['files'] != settings['files']
                or actual['status'] != 'training_completed'):
            raise RuntimeError('Mounted capacity manifest differs from frozen source')
        source = source_manifest.parent/'adapter'  # final, never a checkpoint
        root = Path('/tmp')/('p015-capacity-diagnostic-'+out.name)
        destination = root/'adapters'/settings['name']
        destination.mkdir(parents=True, exist_ok=False)
        identities = {}
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            expected = settings['files'][name]
            path = source/name
            if path.is_symlink() or path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
                raise RuntimeError('Capacity adapter file identity mismatch: '+name)
            shutil.copyfile(path, destination/name)
            if digest(destination/name) != expected['sha256']:
                raise RuntimeError('Capacity adapter copy changed: '+name)
            identities[name] = expected
        adapters = discover_adapters(str(root), adapter_extensions=extensions)
        if set(adapters.adapters) != {settings['name']} or adapters.adapters[settings['name']].rank != 8:
            raise RuntimeError('Capacity adapter discovery differs from one rank8 adapter')
        record.update(status='prepared', source_manifest=str(source_manifest),
                      source_adapter=str(source), actual_served_path=str(destination),
                      files=identities, discovered_names=list(adapters.adapters))
        return adapters, record
    except Exception:
        record.update(status='preparation_failed', error=traceback.format_exc())
        return None, record


def request_capacity_diagnostic(server, record, out):
    endpoint = str(server.base_url).rstrip('/')+'/chat/completions'
    payload = {'model': record['name'], 'messages': [{'role': 'user', 'content': 'hello'}],
               'max_tokens': 8, 'temperature': 0}
    record.update(request_attempted=True, request_url=endpoint, request=payload,
                  actual_base_model_path=str(server.config.model))
    started = time.monotonic()
    try:
        request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=60) as response:
            record['http_status'] = response.status
            record['response'] = json.loads(response.read())
        result = record['response']
        record['actual_response_model'] = result.get('model')
        record['runtime_compatibility_verified'] = bool(
            result.get('model') == record['name'] and result.get('choices'))
        record['status'] = 'request_completed' if record['runtime_compatibility_verified'] else 'response_model_unconfirmed'
    except Exception as exc:
        record.update(status='request_failed', error=traceback.format_exc())
        if isinstance(exc, urllib.error.HTTPError):
            record.update(http_status=exc.code, response_error_body=exc.read().decode(errors='replace'))
    record['request_seconds'] = time.monotonic()-started
    save(out/'capacity-adapter-diagnostic.json', record)


def main():
    started = time.monotonic()
    run_id = 'run-015-' + SETTINGS['study'] + '-' + time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    out = Path('/kaggle/working/results/015') / run_id
    out.mkdir(parents=True)
    deadline = started + SETTINGS.get('wall_seconds', 5100)
    panel = Path(MATERIAL_ROOT)
    tasks_path = panel / 'agent-tasks.json'
    task_document = json.loads(tasks_path.read_text())
    tasks = task_document if isinstance(task_document, list) else task_document['tasks']
    tasks = [{**t, 'instance_id': t.get('instance_id', t.get('iid')), 'problem_statement': t.get('problem_statement', t.get('issue'))} for t in tasks]
    by_id = {t['instance_id']: t for t in tasks}
    selected = SETTINGS.get('task_ids') or [t['instance_id'] for t in tasks][:SETTINGS.get('task_count', 8)]
    tasks = [by_id[iid] for iid in selected]
    criteria_document = json.loads((panel / 'host-criteria.json').read_text())
    criterion_items = criteria_document if isinstance(criteria_document, list) else criteria_document['tasks']
    criteria = {item.get('instance_id', item.get('iid')): item for item in criterion_items}
    arms = list(BUNDLE)
    agents = out / 'frozen-agents'
    for arm, files in BUNDLE.items():
        for rel, text in files.items():
            save(agents / arm / rel, text)
    manifest = {'run_id': run_id, 'settings': SETTINGS, 'task_ids': selected,
                'task_input_sha256': digest(tasks_path), 'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'config_sha256': {arm: {rel: hashlib.sha256(text.encode()).hexdigest() for rel, text in files.items()} for arm, files in BUNDLE.items()},
                'planned_attempts': len(tasks) * len(arms), 'concurrency': SETTINGS.get('concurrency', 4),
                'source': 'frozen natural-PR dev panel and separate train sampling; source criteria stay host-side',
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
    base_adapters = discover_adapters(str(agents / arms[-1]), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    if base_adapters.adapters:
        raise RuntimeError('Base sampling Agent must not contain an adapter')
    diagnostic_manifest, diagnostic = prepare_capacity_diagnostic(
        SETTINGS['capacity_adapter_diagnostic'], out, discover_adapters, ALLOWED_ADAPTER_EXTENSIONS)
    save(out/'capacity-adapter-diagnostic.json', diagnostic)
    def make_server(with_diagnostic):
        config = VllmConfig(model=str(model), port=8000, host='127.0.0.1', tool_call_parser='gemma4', reasoning_parser='gemma4',
                            max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90,
                            enable_auto_tool_choice=True, enable_lora=with_diagnostic,
                            max_loras=1, max_lora_rank=8, tensor_parallel_size=4,
                            startup_timeout=1200, extra_args=['--max-num-seqs', str(SETTINGS.get('concurrency', 4))])
        return VllmServer(config, adapter_manifest=diagnostic_manifest if with_diagnostic else base_adapters)
    serving_diagnostic = diagnostic_manifest is not None
    server = make_server(serving_diagnostic)
    manifest['server_launch_attempts'] = []
    worker = out / 'attempt_worker.py'
    save(worker, WORKER_SOURCE)
    schedule = []
    for slot, task in enumerate(tasks):
        for arm in (arms if slot % 2 == 0 else list(reversed(arms))):
            ordinal = len(schedule) + 1
            entry = {'ordinal': ordinal, 'task_slot': slot + 1, 'task_id': task['instance_id'], 'variant': arm,
                     'attempt_id': f'{ordinal:02d}-{task["instance_id"]}-{arm}',
                     'split': criteria[task['instance_id']]['split']}
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
        launch = {'enable_lora': serving_diagnostic, 'command': server.build_cmd()}
        manifest['server_launch_attempts'].append(launch)
        save(out/'experiment-manifest.json', manifest)
        try:
            server.start()
            launch['status'] = 'started'
        except Exception:
            launch.update(status='startup_failed', error=traceback.format_exc())
            if not serving_diagnostic:
                raise
            diagnostic.update(status='adapter_server_startup_failed', startup_error=launch['error'],
                              fallback_to_base=True,
                              comparison_requirement='Future weight A/B must rerun both arms with this same fallback service configuration')
            # The official start() also invokes stop() on startup failure. Call
            # it explicitly before replacing the server; preserve any remaining
            # log first because official stop() may delete its temporary file.
            failed_log = getattr(server, 'log_path', None)
            if failed_log and Path(failed_log).exists():
                shutil.copyfile(failed_log, out/'vllm-capacity-startup-failed.log')
            server.stop(timeout=10)
            serving_diagnostic = False
            save(out/'capacity-adapter-diagnostic.json', diagnostic)
            server = make_server(False)
            fallback = {'enable_lora': False, 'command': server.build_cmd(), 'reason': 'capacity adapter startup failed'}
            manifest['server_launch_attempts'].append(fallback)
            save(out/'experiment-manifest.json', manifest)
            server.start()
            fallback['status'] = 'started'
        manifest['model_startup_seconds'] = time.monotonic()-started
        manifest['server_command'] = server.build_cmd()
        manifest['serving_conditions'] = {
            'actual_base_model_path': str(server.config.model), 'enable_lora': serving_diagnostic,
            'mounted_adapters': [diagnostic['name']] if serving_diagnostic else [],
            'capacity_diagnostic_excluded_from_repair_statistics': True,
            'base_requests_use_adapter': False,
            'future_weight_ab_requirement': 'Both arms must use identical enable_lora, mounted adapters, rank, concurrency and service settings; fallback configuration requires a matching rerun'}
        save(out / 'experiment-manifest.json', manifest)
        if serving_diagnostic:
            diagnostic['server_command'] = server.build_cmd()
            request_capacity_diagnostic(server, diagnostic, out)
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
                        'served_model': str(server.config.model), 'model_alias': SETTINGS['model_alias'],
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
                   'by_split': {split: {
                       'planned_attempts': sum(entry['split'] == split for entry, _task in schedule),
                       'arms': {arm: {
                           'resolved': sum(r.get('resolved') is True and r['variant'] == arm and r['split'] == split for r in rows),
                           'attempts': sum(r['variant'] == arm and r['split'] == split for r in rows),
                           'completed_attempts': sum(r.get('worker_exit_code') == 0 and r.get('resolved') is not None and r['variant'] == arm and r['split'] == split for r in rows),
                       } for arm in arms},
                   } for split in sorted({entry['split'] for entry, _task in schedule})}}
        save(out / 'summary.json', summary)
        print('FINAL ' + json.dumps(summary), flush=True)
        with zipfile.ZipFile('/kaggle/working/p015-'+SETTINGS['study']+'-output.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in out.rglob('*'):
                if path.is_file():
                    archive.write(path, path.relative_to(out.parent))


if __name__ == '__main__':
    main()
