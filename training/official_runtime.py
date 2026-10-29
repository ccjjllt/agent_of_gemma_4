"""Run one public competition task through the official CPU harness.

This module belongs in the separate Python 3.13 official runtime environment.
It does not load a model or start a GPU service. The caller supplies an existing
OpenAI-compatible endpoint and a unique rollout ID used as its API routing key.

The subprocess backend adds a Linux UID boundary around model-executed commands.
It is not a replacement for a network-isolated Docker sandbox. Official task
assets and judge outputs must remain inaccessible to the agent UID.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.parse import urlsplit


COMPETITION = "gemma-4-developer-agent"
MODEL_ALIAS = "gemma-4-31b-it-qat-w4a16-ct"
OFFICIAL_STARTER_SHA256 = "8e7d84a326acae509fb596163ba085831961b588b86d51af31e037c5319b917c"
CONTEXT_SETTINGS = {
    "compaction_interval": 15,
    "overlap_size": 2,
    "token_threshold": 14336,
    "event_retention_size": 5,
}
FORBIDDEN_COMPONENTS = {"secret", "heldout", "held-out", "held_out", "reserved"}


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _public_path(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if any(part.lower() in FORBIDDEN_COMPONENTS for part in resolved.parts):
        raise ValueError("Only the official public training mount may be used.")
    return resolved


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def _select_task(path: Path, task_id: str | None, index: int | None) -> tuple[Any, int]:
    """Read the requested original public row; never create a task-data copy."""
    from swegemma.models import Task

    fields = {field.name for field in dataclasses.fields(Task)}
    seen: set[str] = set()
    position = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = row.get("instance_id")
            if not instance_id or instance_id in seen:
                continue
            seen.add(instance_id)
            if instance_id == task_id or (index is not None and position == index):
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", instance_id):
                    raise ValueError("Unexpected instance_id in public task data.")
                if ".." in instance_id.split("/"):
                    raise ValueError("Unsafe instance_id in public task data.")
                task = Task(**{name: value for name, value in row.items() if name in fields})
                if not (task.test_patch or task.FAIL_TO_PASS or task.PASS_TO_PASS):
                    raise ValueError("The selected public task has no verification specification; secret hydration is disabled.")
                return task, position
            position += 1
    raise ValueError("Requested task was not found in the explicit public training file.")


def _private_root(path: Path, *, create: bool = False) -> None:
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.chmod(0o700)
    info = path.stat()
    if info.st_uid != 0 or stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError(f"Judge-owned directory must be root-owned and mode 0700: {path}")


def _drop_prefix(uid: int) -> list[str]:
    binary = shutil.which("setpriv")
    if not binary:
        raise RuntimeError("The subprocess backend requires Linux util-linux setpriv.")
    return [
        binary, f"--reuid={uid}", f"--regid={uid}", "--clear-groups",
        "--no-new-privs", "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all",
    ]


def _chown_tree(root: Path, uid: int) -> None:
    """Change only this sandbox's ownership; never follow symlink targets."""
    os.chown(root, uid, uid, follow_symlinks=False)
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            os.chown(Path(directory) / name, uid, uid, follow_symlinks=False)


def _uid_processes(uid: int) -> list[int]:
    found = []
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            row = next(line for line in path.read_text().splitlines() if line.startswith("Uid:"))
            if int(row.split()[1]) == uid:
                found.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
            continue
    return found


def _make_subprocess_manager(uid: int, timeout: int, sandbox_root: Path) -> Any:
    """Reuse official path translation and timeouts, lowering only agent commands."""
    import swegemma.sandbox.subprocess as backend

    original_execute = backend.execute_subprocess_command

    # Keep the official class name: verification.py uses it when enforcing the
    # presence of a real JUnit report. Renaming it would weaken that check.
    class SubprocessManager(backend.SubprocessManager):
        def __init__(self) -> None:
            super().__init__(timeout_seconds=timeout, base_dir=sandbox_root, system_site_packages=True)
            self.restricted: set[str] = set()

        def start(self) -> str:
            # The judge parent stays private throughout setup/verification.
            # Only the agent phase temporarily receives traverse permission.
            sandbox_root.chmod(0o700)
            sandbox_id = super().start()
            self._sandboxes[sandbox_id]["root"].chmod(0o700)
            return sandbox_id

        def restrict_agent(self, sandbox_id: str) -> None:
            with self._lock:
                root = self._sandboxes[sandbox_id]["root"]
                _chown_tree(root, uid)
                root.chmod(0o700)
                self.restricted.add(sandbox_id)
                sandbox_root.chmod(0o711)

        def _agent_path(self, sandbox_id: str, raw: str) -> Path:
            with self._lock:
                paths = dict(self._sandboxes[sandbox_id])
            root = paths["root"].resolve()
            target: Path | None = None
            for prefix, directory in (
                ("/workspace", paths["workspace"]), ("/tmp", paths["tmp"]),
                ("/wheels", paths["wheels"]), ("/usr/local/bin", paths["venv"] / "bin"),
            ):
                if raw.rstrip("/") == prefix:
                    target = directory
                    break
                if raw.startswith(prefix + "/"):
                    target = directory / raw[len(prefix) + 1:]
                    break
            if target is None:
                target = root / raw.lstrip("/") if raw.startswith("/") else paths["workspace"] / raw
            if not target.resolve().is_relative_to(root):
                raise PermissionError("Copied path escapes the agent sandbox.")
            return target

        def _copy_to_sandbox(self, sandbox_id: str, src: Path, dst: str) -> None:
            if sandbox_id not in self.restricted:
                return super()._copy_to_sandbox(sandbox_id, src, dst)
            if not src.is_file():
                raise ValueError("Agent-phase file transfer expects a single file.")
            target = self._agent_path(sandbox_id, dst)
            if dst.endswith("/") or target.is_dir():
                target = target / src.name
            # Root never writes through agent-controlled symlinks. The final
            # filesystem operation runs as the agent, also removing repeated
            # whole-repository ownership walks on every write_file invocation.
            with tempfile.TemporaryDirectory(prefix="transfer_", dir=sandbox_root) as folder:
                transfer = Path(folder)
                payload = transfer / "payload"
                shutil.copy2(src, payload)
                _chown_tree(transfer, uid)
                with self._lock:
                    workspace = self._sandboxes[sandbox_id]["workspace"]
                    relative_payload = os.path.relpath(payload, workspace)
                    relative_target = os.path.relpath(target, workspace)
                    relative_parent = os.path.relpath(target.parent, workspace)
                command = (
                    f"mkdir -p -- {shlex.quote(relative_parent)} && "
                    f"cp --preserve=mode,timestamps -- {shlex.quote(relative_payload)} {shlex.quote(relative_target)}"
                )
                result = self.exec(sandbox_id, command)
                if result.exit_code:
                    raise OSError(f"Agent file write failed: {result.stderr or result.stdout}")

        def _copy_from_sandbox(self, sandbox_id: str, src: str, dst: Path) -> None:
            if sandbox_id not in self.restricted:
                return super()._copy_from_sandbox(sandbox_id, src, dst)
            source = self._agent_path(sandbox_id, src)
            with tempfile.TemporaryDirectory(prefix="transfer_", dir=sandbox_root) as folder:
                transfer = Path(folder)
                os.chown(transfer, uid, uid)
                payload = transfer / "payload"
                with self._lock:
                    workspace = self._sandboxes[sandbox_id]["workspace"]
                    relative_payload = os.path.relpath(payload, workspace)
                    relative_source = os.path.relpath(source, workspace)
                result = self.exec(sandbox_id, f"cp -- {shlex.quote(relative_source)} {shlex.quote(relative_payload)}")
                if result.exit_code:
                    raise OSError(f"Agent file read failed: {result.stderr or result.stdout}")
                # Open the copied file without following a last-moment symlink
                # swap by an agent background process. Its parent is created
                # beneath a root-owned, non-writable directory.
                descriptor = os.open(payload, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb") as incoming:
                    info = os.fstat(incoming.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_uid != uid:
                        raise PermissionError("Unexpected agent transfer file.")
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with dst.open("wb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)

        def stop(self, sandbox_id: str) -> None:
            if sandbox_id in self.restricted:
                sandbox_root.chmod(0o700)
                # Commands may spawn a detached process beyond the official
                # per-command process group. This UID was unused on entry.
                for pid in _uid_processes(uid):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            self.restricted.discard(sandbox_id)
            super().stop(sandbox_id)

    manager = SubprocessManager()

    def execute_with_uid(*args: Any, **kwargs: Any) -> Any:
        cwd = Path(kwargs["cwd"])
        with manager._lock:
            restricted = any(
                sid in manager.restricted and cwd == paths["workspace"]
                for sid, paths in manager._sandboxes.items()
            )
        if restricted:
            command = kwargs.get("command")
            if not isinstance(command, list) or kwargs.get("shell", False):
                raise RuntimeError("Official subprocess execution API changed; expected an argv list.")
            kwargs["command"] = _drop_prefix(uid) + command
            # The official helper already sanitizes its environment. Apply an
            # explicit second allowlist at our privilege boundary so a future
            # harness change cannot leak Kaggle/HF/API credentials to tools.
            allowed = {
                "PATH", "LANG", "LC_ALL", "TERM", "LD_LIBRARY_PATH",
                "HOME", "TMPDIR", "TEST_TMPDIR", "PYTHONNOUSERSITE",
                "PYTHONPATH", "VIRTUAL_ENV", "PYTHONSAFEPATH",
            }
            kwargs["env"] = {key: value for key, value in kwargs["env"].items() if key in allowed}
            kwargs["env"].update({"CUDA_VISIBLE_DEVICES": "", "USER": str(uid), "LOGNAME": str(uid)})
        return original_execute(*args, **kwargs)

    # Worker processes are single-rollout and own this module instance. No
    # process-wide credential changes, preexec_fn, or official-file edits occur.
    backend.execute_subprocess_command = execute_with_uid
    return manager


def _check_uid_access(uid: int, data_root: Path, workspace: Path, sandbox_root: Path) -> None:
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("UID-isolated subprocess execution requires a Linux root judge process.")
    if uid < 10000 or uid >= 2**31:
        raise ValueError("Use a dedicated unprivileged agent UID in [10000, 2^31).")
    import pwd
    try:
        pwd.getpwuid(uid)
    except KeyError:
        pass
    else:
        raise ValueError("The agent UID is already assigned to a host account; choose an unused UID.")
    if _uid_processes(uid):
        raise ValueError("The agent UID already has running processes; choose an unused UID.")
    _private_root(data_root)
    _private_root(workspace, create=True)
    sandbox_root.mkdir(parents=True, exist_ok=True, mode=0o711)
    info = sandbox_root.stat()
    if info.st_uid != 0 or info.st_mode & 0o022:
        raise PermissionError("Agent sandbox parent must be root-owned and not group/world-writable.")
    sandbox_root.chmod(0o711)
    # Check actual permissions under the destination UID without reading any
    # task content; /opt runtime installations are normally traversable.
    probe = subprocess.run(
        _drop_prefix(uid) + [sys.executable, "-c",
            "import os,sys;sys.exit(0 if not os.access(sys.argv[1],os.R_OK) "
            "and not os.access(sys.argv[2],os.R_OK) and os.access(sys.argv[3],os.X_OK) else 1)",
            str(data_root / "tasks.jsonl"), str(workspace), str(sandbox_root)],
        check=False, capture_output=True, text=True, timeout=15,
    )
    if probe.returncode:
        raise PermissionError(
            "Agent UID isolation could not be established. Keep official data and judge outputs "
            "root-owned mode 0700; install the official Python runtime at a traversable path such as /opt/gemma-runtime."
        )


def _evaluation_settings(submission: Path) -> dict[str, Any]:
    import yaml
    path = next((submission / name for name in ("eval_config.yaml", "eval_config.yml") if (submission / name).is_file()), None)
    document = yaml.safe_load(path.read_text(encoding="utf-8")) if path else {}
    document = document or {}
    section = document.get("evaluation", document)
    turns = section.get("max_turns", section.get("max_llm_calls"))
    # Defaults match the current official Getting Started notebook.
    return {
        "timeout_seconds": int(section.get("timeout_seconds", 300)),
        "verification_timeout_seconds": int(section.get("verification_timeout_seconds", 300)),
        "max_tool_calls": int(section.get("max_tool_calls", 100)),
        "max_time_minutes": float(section.get("max_time_minutes", 60.0)),
        "max_turns": int(turns) if turns is not None else None,
    }


def _capture_final_verification(task: Any) -> tuple[dict[str, Any], Any]:
    """Read the final judge JUnit before its sandbox is destroyed.

    The official verifier itself retains only a binary verdict and reads XML
    only on exit code zero. We observe its completed pytest invocation, then
    read the same final report for dense reward statistics on successes and
    failures. This never reruns tests or exposes test identifiers to the model.
    """
    import swegemma.harness.verification as verification
    try:
        from .rewards import junit_reward_features
    except ImportError:
        from rewards import junit_reward_features

    original = verification.sandbox_exec
    captured: dict[str, Any] = {}

    async def observed(docker: Any, sandbox_id: str, command: str, *args: Any, **kwargs: Any) -> Any:
        started = time.monotonic()
        output = await original(docker, sandbox_id, command, *args, **kwargs)
        parts = shlex.split(command) if isinstance(command, str) else []
        is_pytest = any(parts[index:index + 2] == ["-m", "pytest"] for index in range(len(parts) - 1))
        reports = [part.split("=", 1)[1] for part in parts if part.startswith("--junitxml=")]
        if is_pytest and len(reports) == 1 and re.fullmatch(r"/tmp/_swegemma_junit_[a-f0-9]+\.xml", reports[0]):
            captured.update({"pytest_exit_code": output.exit_code,
                             "verification_duration_seconds": time.monotonic() - started})
            try:
                report = await original(docker, sandbox_id, f"cat {shlex.quote(reports[0])}", timeout=30)
                xml_text = report.stdout if report.exit_code == 0 else ""
                fallback = ()
                if not task.FAIL_TO_PASS and not task.PASS_TO_PASS:
                    fallback = tuple(verification._extract_test_functions_from_patch(task.test_patch))
                captured["reward_features"] = junit_reward_features(
                    xml_text, fail_to_pass=task.FAIL_TO_PASS, pass_to_pass=task.PASS_TO_PASS,
                    fallback_requirements=fallback, node_matcher=verification._node_matches_requirement,
                )
            except Exception as error:
                captured["reward_features"] = {
                    "junit_available": False, "dense_coverage_complete": False,
                    "repair_fraction": None, "regression_fraction": None,
                    "unavailable_reason": f"junit_capture_{type(error).__name__}",
                }
        return output

    verification.sandbox_exec = observed

    def restore() -> None:
        verification.sandbox_exec = original

    return captured, restore


def _training_submission(source: Path, destination: Path) -> Path:
    """Copy agent configuration, never task data, and make sampling explicit."""
    import yaml

    if destination.exists():
        raise FileExistsError(f"A rollout workspace must be fresh: {destination}")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("adapters", "*.safetensors"))

    def adjust(node: Any) -> None:
        if isinstance(node, list):
            for value in node:
                adjust(value)
        elif isinstance(node, dict):
            node.pop("adapter", None)
            if isinstance(node.get("model"), str):
                node["model"] = MODEL_ALIAS
                cfg = node.setdefault("generate_content_config", {})
                if not isinstance(cfg, dict):
                    raise ValueError("Expected mapping generate_content_config in agent configuration.")
                cfg.update({"temperature": 1.0, "top_p": 1.0, "top_k": 0})
                for name in ("presence_penalty", "frequency_penalty", "stop_sequences", "seed"):
                    cfg.pop(name, None)
                cfg["thinking_config"] = {"include_thoughts": True}
            for value in list(node.values()):
                adjust(value)

    for path in sorted(destination.rglob("*")):
        if path.suffix not in {".yaml", ".yml"} or not path.is_file():
            continue
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        adjust(document)
        path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return destination


async def run_task(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    data_root = _public_path(args.data_root)
    tasks_path = _public_path(data_root / "tasks.jsonl")
    source_submission = args.submission_dir.resolve(strict=True)
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.sandbox == "subprocess":
        _check_uid_access(args.agent_uid, data_root, workspace, args.sandbox_root)
    submission = _training_submission(source_submission, workspace / "runtime_submission")
    endpoint = urlsplit(args.endpoint)
    if endpoint.scheme not in {"http", "https"} or not endpoint.netloc or endpoint.username or endpoint.password:
        raise ValueError("endpoint must be an HTTP(S) API base without embedded credentials.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.rollout_id):
        raise ValueError("rollout_id must be a safe unique identifier.")

    os.environ.update({
        "OPENAI_BASE_URL": args.endpoint.rstrip("/"),
        "LOCAL_INFERENCE_URL": args.endpoint.rstrip("/"),
        "LITELLM_API_BASE": args.endpoint.rstrip("/"),
        "OPENAI_API_KEY": args.rollout_id,
        "LOCAL_API_KEY": args.rollout_id,
        "LITELLM_API_KEY": args.rollout_id,
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_TELEMETRY": "False",
        "OTEL_SDK_DISABLED": "true",
        "CUDA_VISIBLE_DEVICES": "",
        "KAGGLE_SANDBOX_DIR": str(data_root / "sandbox"),
    })
    # The official CLI gives MODEL_PROXY_* precedence over the local endpoint.
    os.environ.pop("MODEL_PROXY_URL", None)
    os.environ.pop("MODEL_PROXY_API_KEY", None)

    import litellm
    from adk_submission import discover_adapters
    from adk_submission.discovery import discover_declared_models
    from adk_submission.resolvers.generation import install_litellm_reasoning_patch
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.cli import _build_cli_model_registry
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.context import SwegemmaContext
    from swegemma.evaluate import Evaluator
    from swegemma.results import append_task_result

    litellm.drop_params = True
    install_litellm_reasoning_patch()
    declared = set(discover_declared_models(submission))
    if declared != {MODEL_ALIAS}:
        raise ValueError(f"Training submission must declare only the official 31B alias: {MODEL_ALIAS}")
    models = _build_cli_model_registry(submission_dir=submission)
    task, position = _select_task(tasks_path, args.task_id, args.task_index)
    settings = _evaluation_settings(submission)
    limits, generation_constraints = build_submission_limits()
    generation_constraints = dataclasses.replace(
        generation_constraints,
        thinking_budget=None,
        defaults={"max_output_tokens": 16384, "thinking_config": {"include_thoughts": True}},
    )
    config = EvalConfig(
        tasks_path=tasks_path,
        snapshots_dir=_public_path(data_root / "snapshots"),
        results_dir=workspace / "official_results",
        submission_dir=submission,
        models=models,
        sandbox=args.sandbox,
        wheels_dir=_public_path(data_root / "wheels"),
        graph_dir=str(_public_path(data_root / "graphs")),
        embeddings_dir=str(_public_path(data_root / "embeddings")),
        limits=limits,
        generation_constraints=generation_constraints,
        adapter_manifest=discover_adapters(submission),
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(**CONTEXT_SETTINGS),
        display_mode="quiet",
        **settings,
    )

    class PublicOnlyEvaluator(Evaluator):
        def _get_secret_bundle_data(self) -> tuple[dict[str, str], dict[str, str], None]:
            return {}, {}, None

        def _hydrate_task_from_secret(self, original_task: Any) -> Any:
            return original_task

    evaluator = PublicOnlyEvaluator(config)
    context_class = SwegemmaContext
    if args.sandbox == "subprocess":
        private_sandboxes = args.sandbox_root / args.rollout_id
        private_sandboxes.mkdir(mode=0o700)
        manager = _make_subprocess_manager(args.agent_uid, settings["timeout_seconds"], private_sandboxes)
        evaluator.sandbox = evaluator.docker = manager

        class AgentContext(SwegemmaContext):
            def create_tools(self) -> dict[str, Any]:
                # Official setup is complete here; every subsequent shell/skill
                # command in this sandbox runs under the agent UID. The judge
                # starts a separate, unrestricted sandbox after this one stops.
                manager.restrict_agent(self.container_id)
                return super().create_tools()

        context_class = AgentContext

    context = context_class(
        docker_manager=evaluator.docker,
        container_id="",
        problem_statement=task.problem_statement,
        hints_text=task.hints_text,
        task=task,
        repo=task.repo,
        graph_dir=config.graph_dir,
        embeddings_dir=config.embeddings_dir,
        budget=config.budget,
        harness=config.harness,
    )
    captured, restore_verification = _capture_final_verification(task)
    try:
        result = await evaluator.evaluate_task(task, task_index=position + 1, total_tasks=1, context=context)
        append_task_result(result, config.results_dir, [result])
    finally:
        restore_verification()
        cleanup = getattr(evaluator.docker, "cleanup_all", None)
        if cleanup:
            cleanup()

    error = result.error or None
    # A completed pytest verdict is meaningful even if the agent exhausted its
    # own official budget. Infrastructure failures must never become reward 0.
    agent_error = error if error and error.startswith(("Agent ", "Failed to apply agent patch:")) else None
    verification_ran = result.test_output.startswith("STDOUT:\n")
    verdict_valid = verification_ran and result.test_exit_code in (0, 1) and (not error or agent_error is not None)
    infrastructure_error = None if verdict_valid else (error or f"No complete official verification verdict (exit={result.test_exit_code}).")
    safe_id = task.instance_id.replace("/", "__")
    patch_path = config.results_dir / "patches" / f"{safe_id}.patch"
    reward_features = dict(captured.get("reward_features") or {
        "junit_available": False, "dense_coverage_complete": False,
        "repair_fraction": None, "regression_fraction": None,
        "unavailable_reason": "no_completed_pytest_capture",
    })
    reward_features.update({
        "patch_nonempty": bool(result.agent_patch.strip()),
        "patch_sha256": hashlib.sha256(result.agent_patch.encode()).hexdigest(),
        "verification_duration_seconds": captured.get("verification_duration_seconds"),
    })
    return {
        "task_id": task.instance_id,
        "rollout_id": args.rollout_id,
        "success": bool(result.resolved) if verdict_valid else None,
        "reward": float(bool(result.resolved)) if verdict_valid else None,
        "verification_complete": verification_ran,
        "verdict_valid": verdict_valid,
        "verdict_kind": "tests_completed" if verdict_valid else "unclassified_verification_failure",
        "failure_stage": None if verdict_valid else (
            "patch_application" if error and error.startswith("Failed to apply agent patch:")
            else "pytest" if captured.get("pytest_exit_code") is not None
            else "setup_or_agent"
        ),
        "infrastructure_error": infrastructure_error,
        "agent_error": agent_error,
        "reward_features": reward_features,
        "test_exit_code": result.test_exit_code,
        "duration_seconds": time.monotonic() - started,
        "llm_calls": getattr(result, "total_llm_calls", 0),
        "tool_calls": getattr(result, "tool_calls", 0),
        "patch_path": str(patch_path) if patch_path.is_file() else None,
        "trace_path": result.trace_path,
        "official_results_dir": str(config.results_dir),
        "provenance": {
            "competition": COMPETITION,
            "public_tasks_sha256": _sha256(tasks_path),
            "official_starter_source_sha256": OFFICIAL_STARTER_SHA256,
            "packages": {name: importlib.metadata.version(name) for name in ("swegemma", "adk-eval-core", "adk-submission", "google-adk", "litellm")},
            "evaluation": settings,
            "context_management": CONTEXT_SETTINGS,
            "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "enable_thinking": True},
            "submission_source": str(source_submission),
            "submission_runtime": str(submission),
            "sampling_departures": [
                "Untruncated categorical sampling for matching behavior and learner probabilities.",
                "No forced thinking_token_budget processor; thinking remains enabled within the original max_output_tokens.",
                "All agents use the current shared learner LoRA; sample dummy adapters are excluded.",
            ],
            "sandbox": args.sandbox,
            "agent_uid": args.agent_uid if args.sandbox == "subprocess" else None,
            "model_alias": MODEL_ALIAS,
            "backend": "caller-supplied endpoint; training model identity is recorded by the learner",
            "secret_hydration": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--task-id")
    selector.add_argument("--task-index", type=int, help="Zero-based index in the official public training file.")
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="Existing OpenAI-compatible API base, including /v1.")
    parser.add_argument("--rollout-id", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--sandbox", choices=("subprocess", "docker"), default="subprocess")
    parser.add_argument("--agent-uid", type=int, default=61000)
    parser.add_argument("--sandbox-root", type=Path, default=Path("/tmp/gemma-agent-sandboxes"))
    args = parser.parse_args(argv)
    if args.task_index is not None and args.task_index < 0:
        parser.error("task-index must be nonnegative")
    def terminate(_signum: int, _frame: Any) -> None:
        # Let asyncio unwind official sandbox finally blocks before the parent
        # escalates to SIGKILL; agent commands have their own process groups.
        raise KeyboardInterrupt("Official rollout termination requested")
    signal.signal(signal.SIGTERM, terminate)
    try:
        result = asyncio.run(run_task(args))
    except (Exception, KeyboardInterrupt) as error:
        result = {
            "task_id": args.task_id,
            "rollout_id": args.rollout_id,
            "success": None,
            "reward": None,
            "verification_complete": False,
            "verdict_valid": False,
            "infrastructure_error": f"{type(error).__name__}: {error}",
            "agent_error": None,
        }
    _write_json(args.result_file.resolve(), result)
    print(json.dumps({key: result.get(key) for key in ("task_id", "rollout_id", "reward", "verdict_valid", "infrastructure_error")}, ensure_ascii=False), flush=True)
    return 0 if result.get("verdict_valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
