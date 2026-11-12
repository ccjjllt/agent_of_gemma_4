"""Judge-side empty-patch baseline probes for uncached tasks.

Runs :mod:`training.official_runtime` in ``--empty-patch-probe`` mode as a
subprocess under the configured runtime interpreter: the official harness sets
up the judge sandbox and verifies the pristine snapshot with an empty patch
(``skip_agent_patch``), so no model is called and no delivery or timing
assertions apply. The captured node report is returned;
:class:`EmptyPatchBaselineStore` normalizes and caches it keyed by TaskKey.

Each probe consumes one number from the run-wide attempt sequence, so its
UID pair and rollout namespace can never collide with a training attempt.
Failed probes keep their workspace for forensics and raise; they never
fabricate a baseline.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL_RUNTIME = PROJECT_ROOT / "training" / "official_runtime.py"


class EmptyPatchProbeRunner:
    """Execute the official empty-patch probe for one TaskKey."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        submission_dir: str | Path,
        runtime_python: str | Path,
        sandbox_root: str | Path,
        baseline_python: str | Path,
        uid_allocator: Any,
        attempt_seq: Callable[[], int],
        workspace_root: str | Path,
        evaluation_overrides: Mapping[str, Any] | None = None,
        timeout_seconds: float = 1800.0,
        emit: Callable[..., None] | None = None,
    ):
        self.data_root = Path(data_root)
        self.submission_dir = Path(submission_dir)
        self.runtime_python = Path(runtime_python)
        self.sandbox_root = Path(sandbox_root)
        self.baseline_python = Path(baseline_python)
        self.uid_allocator = uid_allocator
        self.attempt_seq = attempt_seq
        self.workspace_root = Path(workspace_root)
        self.evaluation_overrides = dict(evaluation_overrides or {})
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("probe timeout must be positive")
        self.timeout_seconds = float(timeout_seconds)
        self.emit = emit
        for name in ("data_root", "submission_dir", "workspace_root"):
            path = getattr(self, name)
            if not path.is_absolute():
                raise ValueError(f"{name} must be absolute")
        for name in ("runtime_python", "baseline_python"):
            path = getattr(self, name)
            if not path.is_file():
                raise ValueError(f"{name} must be an existing interpreter")
        if not _OFFICIAL_RUNTIME.is_file():
            raise ValueError("official runtime implementation is missing")

    def _event(self, name: str, **fields: Any) -> None:
        if self.emit is not None:
            self.emit(name, **fields)

    def __call__(self, task_key: Mapping[str, Any] | object) -> Mapping[str, Any]:
        from .baseline import TaskKey

        key = task_key if isinstance(task_key, TaskKey) else TaskKey.from_dict(task_key)
        seq = int(self.attempt_seq())
        agent_uid, judge_uid = self.uid_allocator.allocate(seq)
        folder = self.workspace_root / f"probe-{key.cache_key}"
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        rollout_id = f"probe-{key.cache_key[:56]}"
        result_path = folder / "result.json"
        credential_path = folder / "probe_token"
        descriptor = os.open(credential_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="ascii") as credential:
                credential.write(secrets.token_hex(32))
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        command = [
            str(self.runtime_python), str(_OFFICIAL_RUNTIME),
            "--data-root", str(self.data_root),
            "--submission-dir", str(self.submission_dir),
            "--task-id", key.task_id,
            "--workspace", str(folder / "workspace"),
            "--result-file", str(result_path),
            "--sandbox", "subprocess",
            "--agent-uid", str(agent_uid),
            "--judge-uid", str(judge_uid),
            "--sandbox-root", str(self.sandbox_root),
            "--sandbox-baseline-python", str(self.baseline_python),
            "--evaluation-overrides-json", json.dumps(
                self.evaluation_overrides, sort_keys=True, separators=(",", ":"), allow_nan=False),
            "--rollout-id", rollout_id,
            "--auth-token-file", str(credential_path),
            # The probe never calls a model; the runtime still requires a
            # syntactically valid, credential-free API base, so pass a
            # deliberately unreachable placeholder instead of omitting it.
            "--endpoint", "http://127.0.0.1:9/v1",
            "--empty-patch-probe",
        ]
        self._event("baseline_probe_started", task_id=key.task_id, attempt=seq)
        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        try:
            with (folder / "probe.log").open("ab") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT), start_new_session=True,
                    env=_launcher_env())
            try:
                process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(
                    f"empty-patch probe exceeded {self.timeout_seconds}s") from error
            if process.returncode != 0:
                raise OSError(f"empty-patch probe exited {process.returncode}")
            try:
                document = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise OSError(f"empty-patch probe result unreadable: {error}") from error
            if not isinstance(document, dict):
                raise ValueError("empty-patch probe result is not a verdict object")
            report = document.get("baseline_report")
            if not isinstance(report, Mapping) or not report.get("expected_universes"):
                raise ValueError(
                    "empty-patch probe produced no usable baseline report: "
                    f"{document.get('infrastructure_error')}")
            self._event("baseline_probe_completed", task_id=key.task_id,
                        elapsed_seconds=time.monotonic() - started)
            return dict(report)
        finally:
            if process is not None and process.poll() is None:
                _terminate_tree(process)
            try:
                credential_path.unlink()
            except OSError:
                pass


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

