"""Asynchronous, atomic full-checkpoint writer for SPO training state.

The trainer clones LoRA/Adam/scheduler/RNG tensors to CPU before calling this
writer, so only filesystem I/O stays on the background thread. A background
write may overlap the next forward/backward; a true recovery boundary calls
``submit(..., sync=True)`` and waits for the fsynced atomic rename.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid
from typing import Any, Mapping

from .recovery import atomic_json


def _sync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _adapter_identity(adapter: Any) -> dict[str, Any]:
    sha = getattr(adapter, "adapter_sha256", None)
    if not isinstance(sha, str) or len(sha) != 64 \
            or any(character not in "0123456789abcdef" for character in sha.lower()):
        raise ValueError("adapter identity requires a 64-hex SHA256")
    version = getattr(adapter, "version", None)
    parent = getattr(adapter, "parent_version", None)
    step = getattr(adapter, "optimizer_step", None)
    if type(version) is not int or version < 0:
        raise ValueError("adapter identity requires a non-negative version")
    if parent is not None and (type(parent) is not int or parent < 0):
        raise ValueError("adapter identity has an invalid parent version")
    if step is not None and (type(step) is not int or step < 0):
        raise ValueError("adapter identity has an invalid optimizer step")
    return {
        "version": version,
        "parent_version": parent,
        "optimizer_step": step,
        "adapter_sha256": sha.lower(),
    }


def write_transaction(path: str | os.PathLike[str], transaction: Mapping[str, Any], *,
                      checkpoint_path: str | os.PathLike[str],
                      adapter: Any = None) -> dict[str, Any]:
    """Atomically write the lightweight phase/KL/link record.

    A LEARNING record may reference no full checkpoint (first update). Every
    post-optimizer phase must reference the full optimizer_step file. The
    exported adapter SHA256 is attached at EXPORTED/REGISTERED.
    """
    record = dict(transaction)
    phase = record.get("phase")
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_file():
        record["checkpoint"] = {
            "file": checkpoint.name,
            "phase": "optimizer_step",
            "completed_optimizer_steps": record.get("completed_optimizer_steps"),
            "bytes": checkpoint.stat().st_size,
        }
    elif phase == "LEARNING":
        record["checkpoint"] = None
    else:
        raise RuntimeError(
            f"transaction phase {phase} requires the full optimizer_step checkpoint")
    if adapter is not None:
        record["adapter"] = _adapter_identity(adapter)
    atomic_json(path, record)
    return record


def read_transaction(path: str | os.PathLike[str], *, checkpoint_path: str | os.PathLike[str]
                     ) -> dict[str, Any] | None:
    """Read and validate one transaction record; missing file returns None."""
    path = Path(path)
    checkpoint = Path(checkpoint_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"trainer transaction record is unreadable: {exc}") from exc
    try:
        record = json.loads(raw)
    except ValueError as exc:
        raise RuntimeError("trainer transaction record is corrupt") from exc
    if not isinstance(record, dict) or record.get("schema_version") != 1 \
            or record.get("algorithm") != "single_stream_spo":
        raise RuntimeError("trainer transaction record has an unsupported schema")
    reference = record.get("checkpoint")
    if reference is None:
        if record.get("phase") != "LEARNING":
            raise RuntimeError("post-optimizer transaction has no full checkpoint reference")
    elif not isinstance(reference, dict) or reference.get("file") != checkpoint.name:
        raise RuntimeError("trainer transaction record points at another checkpoint")
    elif not checkpoint.is_file():
        raise RuntimeError("trainer transaction record exists without its full checkpoint")
    adapter = record.get("adapter")
    if adapter is not None:
        if not isinstance(adapter, dict):
            raise ValueError("transaction adapter identity is not a mapping")
        sha = adapter.get("adapter_sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("transaction adapter identity has no SHA256")
    return record


class CheckpointWriter:
    """Single-writer queue preserving snapshot order and surfacing I/O errors."""

    def __init__(self, path: str | os.PathLike[str], torch_module: object):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.torch = torch_module
        # A SIGKILL/OOM can leave a multi-GB temporary file behind. Only reap
        # a temp whose embedded PID is no longer alive; a second live process
        # on the same run_dir must not lose its in-flight write.
        for stale in self.path.parent.glob(f"{self.path.name}.tmp-*"):
            try:
                pid_text = stale.name[len(self.path.name) + len(".tmp-"):].split("-", 1)[0]
                pid = int(pid_text)
            except (ValueError, IndexError):
                continue
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
            except OSError:
                alive = True
            if not alive:
                try:
                    stale.unlink()
                except OSError:
                    pass
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._closed = False
        self._writes = 0
        self._last: dict[str, object] | None = None
        self._history: list[dict[str, object]] = []
        self._thread = threading.Thread(
            target=self._run, name="spo-checkpoint-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                state, event = item
                try:
                    started = time.perf_counter()
                    size = self._write(state)
                    elapsed = time.perf_counter() - started
                    with self._lock:
                        self._writes += 1
                        self._last = {
                            "bytes": size,
                            "seconds": elapsed,
                            "completed_unix": time.time(),
                        }
                        self._history.append(dict(self._last))
                        del self._history[:-8]
                except BaseException as error:  # preserve for the next sync caller
                    with self._lock:
                        self._error = error
                finally:
                    event.set()
            finally:
                self._queue.task_done()

    def _write(self, state: object) -> int:
        temporary = self.path.with_name(
            f"{self.path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        try:
            self.torch.save(dict(state), temporary)  # type: ignore[attr-defined]
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            _sync_directory(self.path.parent)
            return self.path.stat().st_size
        finally:
            temporary.unlink(missing_ok=True)

    def _raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"asynchronous checkpoint write failed: {error}") from error

    def submit(self, state: object, *, sync: bool) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("checkpoint writer is closed")
        self._raise_if_failed()
        event = threading.Event()
        self._queue.put((state, event))
        if sync:
            event.wait()
            self._raise_if_failed()

    def flush(self) -> None:
        """Wait for every queued write; used only at an explicit boundary."""
        self._queue.join()
        self._raise_if_failed()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(None)
        self._thread.join()
        self._raise_if_failed()

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {"writes": self._writes, "last": dict(self._last or {}),
                    "history": [dict(entry) for entry in self._history],
                    "failed": self._error is not None}
