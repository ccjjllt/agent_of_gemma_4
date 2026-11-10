"""Launch one official worker subprocess and preserve its verdict evidence."""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from ..recovery import atomic_json, value_digest
from .sandbox import SandboxConfig, SandboxGuard, UidAllocator
from .workspace_cache import AttemptPaths, WorkspaceCache


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL_RUNTIME = PROJECT_ROOT / "training" / "official_runtime.py"


@dataclass(frozen=True)
class OfficialWorkerConfig:
    data_root: Path
    submission_dir: Path
    endpoint_url: str
    sandbox_root: Path
    baseline_python: Path
    agent_uid_base: int
    # Size of the UID pair pool, not a retry limit. Attempt numbers come from a
    # persisted global sequence and grow without bound; keep the pool far above
    # any realistic run length (range check: base + 2*pool must fit int32).
    max_attempts: int = 1000000
    runtime_python: Path = Path("/opt/gemma-runtime/bin/python")
    session_clock: str = "queue_excluded"
    event_summarizer: str = "official"
    queue_timing_poll_seconds: float = 0.2
    evaluation_overrides: Mapping[str, Any] = field(default_factory=dict)
    parent_pid: int | None = None

    def validated(self) -> "OfficialWorkerConfig":
        for name in ("data_root", "submission_dir", "sandbox_root"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError(f"{name} must be an absolute path")
        for name in ("baseline_python", "runtime_python"):
            path = getattr(self, name)
            if not isinstance(path, Path) or not path.is_absolute() or not path.is_file():
                raise ValueError(f"{name} must be an absolute executable file")
        if not _OFFICIAL_RUNTIME.is_file():
            raise ValueError("official worker implementation is missing")
        endpoint = urlsplit(self.endpoint_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.netloc or endpoint.username or endpoint.password:
            raise ValueError("endpoint must be an HTTP(S) API base without embedded credentials")
        if self.session_clock not in {"queue_excluded", "wall"}:
            raise ValueError("session_clock must be queue_excluded or wall")
        if self.event_summarizer not in {"official", "tool_aware"}:
            raise ValueError("event_summarizer must be official or tool_aware")
        if not math.isfinite(self.queue_timing_poll_seconds) or not 0.02 <= self.queue_timing_poll_seconds <= 1:
            raise ValueError("queue timing poll must be within [0.02, 1]")
        overrides = dict(self.evaluation_overrides)
        if any(not isinstance(key, str) for key in overrides):
            raise ValueError("evaluation overrides must use string keys")
        return self


@dataclass(frozen=True)
class AttemptContext:
    attempt_id: str
    attempt_index: int
    agent_uid: int
    judge_uid: int
    paths: AttemptPaths
    endpoint_rollout_id: str


class OfficialWorker:
    def __init__(self, config: OfficialWorkerConfig, cache: WorkspaceCache,
                 *, uid_allocator: Any | None = None):
        self.config = config.validated()
        self.cache = cache
        self.sandbox = SandboxConfig(
            config.sandbox_root, config.baseline_python, config.agent_uid_base, config.max_attempts
        )
        # Runs share one allocator across workers and baseline probes so UID
        # pairs stay unique; ad-hoc use falls back to a private allocator.
        self.uids = uid_allocator if uid_allocator is not None else UidAllocator(
            config.agent_uid_base, config.max_attempts)

    def prepare_attempt(self, assignment: Any, attempt_index: int,
                          *, endpoint_rollout_id: str | None = None) -> AttemptContext:
        trajectory_id = getattr(assignment, "trajectory_id", None)
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("assignment must carry a trajectory_id")
        task_id = getattr(getattr(assignment, "task_key", None), "task_id", None)
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("assignment must carry an official task_key.task_id")
        if endpoint_rollout_id is not None and (
                not isinstance(endpoint_rollout_id, str) or not endpoint_rollout_id):
            raise ValueError("endpoint rollout namespace must be a nonempty string")
        namespace = endpoint_rollout_id or trajectory_id
        attempt_id = f"{trajectory_id}-a{attempt_index:06d}-{uuid.uuid4().hex[:8]}"
        agent_uid, judge_uid = self.uids.allocate(attempt_index)
        paths = self.cache.prepare_attempt(attempt_id, task_id=task_id)
        SandboxGuard.check_execution(
            self.sandbox,
            data_root=self.config.data_root,
            workspace=paths.workspace_dir,
            agent_uid=agent_uid,
            judge_uid=judge_uid,
        )
        return AttemptContext(
            attempt_id=attempt_id,
            attempt_index=attempt_index,
            agent_uid=agent_uid,
            judge_uid=judge_uid,
            paths=paths,
            endpoint_rollout_id=namespace,
        )

    def build_command(self, assignment: Any, attempt: AttemptContext, *, endpoint_token_path: Path) -> list[str]:
        task_id = assignment.task_key.task_id
        return [
            str(self.config.runtime_python),
            str(_OFFICIAL_RUNTIME),
            "--data-root", str(self.config.data_root),
            "--submission-dir", str(self.config.submission_dir),
            "--endpoint", self.config.endpoint_url.rstrip("/"),
            "--rollout-id", attempt.endpoint_rollout_id,
            "--workspace", str(attempt.paths.workspace_dir),
            "--result-file", str(attempt.paths.result_path),
            "--sandbox", "subprocess",
            "--agent-uid", str(attempt.agent_uid),
            "--judge-uid", str(attempt.judge_uid),
            "--auth-token-file", str(endpoint_token_path),
            "--parent-pid", str(self.config.parent_pid or os.getpid()),
            "--sandbox-root", str(self.config.sandbox_root),
            "--sandbox-baseline-python", str(self.config.baseline_python),
            "--session-clock", self.config.session_clock,
            "--event-summarizer", self.config.event_summarizer,
            "--queue-timing-poll-seconds", str(self.config.queue_timing_poll_seconds),
            "--evaluation-overrides-json", json.dumps(dict(self.config.evaluation_overrides),
                                                      sort_keys=True, separators=(",", ":"), allow_nan=False),
            "--task-id", task_id,
        ]

    @staticmethod
    def _launcher_env() -> dict[str, str]:
        allowed = {"PATH", "LANG", "LC_ALL", "TERM", "LD_LIBRARY_PATH", "HOME", "TMPDIR",
                   "PYTHONNOUSERSITE", "PYTHONSAFEPATH", "PYTHONPATH", "VIRTUAL_ENV"}
        environment = {key: os.environ[key] for key in allowed if key in os.environ}
        environment.update({
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
        })
        return environment

    @staticmethod
    def _terminate_tree(process: subprocess.Popen[bytes]) -> None:
        try:
            targets = {process.pid}
            if Path("/proc").is_dir():
                snapshot: dict[int, int] = {}
                for entry in Path("/proc").glob("[0-9]*/stat"):
                    try:
                        fields = entry.read_text().rpartition(")")[2].split()
                        snapshot[int(entry.parent.name)] = int(fields[1])
                    except (OSError, ValueError, IndexError):
                        continue
                changed = True
                while changed:
                    changed = False
                    for pid, ppid in snapshot.items():
                        if ppid in targets and pid not in targets:
                            targets.add(pid)
                            changed = True
                for pid in sorted(targets, reverse=True):
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            process.wait(timeout=10)

    def run(
        self,
        assignment: Any,
        attempt: AttemptContext,
        *,
        endpoint_token_path: Path,
        endpoint_token: str,
        cancel_event: threading.Event | None = None,
        deadline_unix: float | None = None,
    ) -> dict[str, Any]:
        if not isinstance(endpoint_token, str) or not endpoint_token:
            raise ValueError("an endpoint credential is required")
        command = self.build_command(assignment, attempt, endpoint_token_path=endpoint_token_path)
        descriptor = os.open(endpoint_token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="ascii") as credential:
                credential.write(endpoint_token)
        except BaseException:
            os.close(descriptor)
            raise
        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        with attempt.paths.log_path.open("ab") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT), start_new_session=True, env=self._launcher_env(),
            )
        try:
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    raise TimeoutError("official worker was cancelled")
                if deadline_unix is not None and time.time() >= deadline_unix:
                    raise TimeoutError("official worker exceeded its deadline")
                try:
                    process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    continue
            outcome: dict[str, Any] = {
                "attempt_id": attempt.attempt_id,
                "task_id": assignment.task_key.task_id,
                "rollout_id": attempt.endpoint_rollout_id,
                "trajectory_id": assignment.trajectory_id,
                "returncode": process.returncode,
                "worker_log": str(attempt.paths.log_path),
                "result_path": str(attempt.paths.result_path),
                "elapsed_seconds": time.monotonic() - started,
            }
            if not attempt.paths.result_path.is_file():
                outcome["worker_error"] = f"official worker exited {process.returncode} without result.json"
                return outcome
            try:
                outcome["result"] = json.loads(attempt.paths.result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                outcome["worker_error"] = f"official result is unreadable: {type(error).__name__}: {error}"
                return outcome
            if not isinstance(outcome["result"], dict):
                outcome["worker_error"] = "official worker result is not a verdict object"
                outcome.pop("result")
            return outcome
        finally:
            if process is not None and process.poll() is None:
                self._terminate_tree(process)
            try:
                endpoint_token_path.unlink()
            except FileNotFoundError:
                pass
            if process is not None:
                atomic_json(
                    attempt.paths.attempt_dir / "worker_process.json",
                    {"pid": process.pid, "returncode": process.returncode,
                     "attempt_id": attempt.attempt_id, "command_sha256": value_digest(command)},
                )
