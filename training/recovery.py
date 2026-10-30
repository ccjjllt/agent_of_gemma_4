"""Durable, bounded checkpoints and explicit run-budget ownership.

Checkpoint payloads contain generated training state, never a copied base model
or task dataset. A checksum-verified atomic pointer is the commit boundary.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import signal
import tempfile
import time
import uuid


class WorkerAttemptError(RuntimeError):
    """A recoverable infrastructure/delivery failure, never a zero task reward."""


class CollectionInterrupted(RuntimeError):
    """Preserve this collection for a later explicitly budgeted continuation."""


def bind_parent_lifetime(parent_pid):
    """Linux child lifetime follows its trusted supervisor, including SIGKILL.

    Install after exec and recheck ppid to cover a supervisor exiting before
    prctl was armed. No model, data or accelerator imports are required.
    """
    import ctypes

    if type(parent_pid) is not int or parent_pid <= 1:
        raise ValueError("A worker requires its supervisor PID")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != parent_pid:
        raise RuntimeError("Supervisor exited before worker startup")


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".commit-", delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output, sort_keys=True, ensure_ascii=False, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def run_lock(run_dir):
    """A second live learner cannot load another GPU copy into the same run."""
    import fcntl

    path = Path(run_dir) / ".run.lock"
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This run is still owned by a live process") from exc
        # A supervised learner inherits this same open-file description.
        # Close only this descriptor on exit: LOCK_UN would also release a
        # still-live child's shared lock after an interrupted cleanup.
        yield lock.fileno()


class CheckpointStore:
    """Keep the two latest committed generations of each state stream.

    Large optimizer/model snapshots and small collection/RNG snapshots are
    separate. A collection snapshot is valid only for its exact model checkpoint.
    Incomplete staging directories never become candidates for recovery.
    """

    def __init__(self, run_dir, torch, binding):
        self.root = Path(run_dir) / "recovery"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.torch = torch
        self.binding = value_digest(binding)

    def _pointer(self, kind):
        if kind not in ("checkpoint", "collection"):
            raise ValueError("Unknown recovery stream")
        return self.root / (kind + "_latest.json")

    def discard_uncommitted(self):
        """Reclaim interrupted writes while the caller exclusively owns this run.

        Rank-128 snapshots are large enough for abandoned staging files to
        prevent the next save. Retain both committed generations; never promote
        an uncommitted directory or touch a different source binding.
        """
        keep = set()
        for kind in ("checkpoint", "collection"):
            pointer = self._pointer(kind)
            if not pointer.exists():
                continue
            manifest = json.loads(pointer.read_text())
            identifier = manifest.get("id", "")
            if (not isinstance(identifier, str)
                    or not re.fullmatch(kind + r"-\d+-[0-9a-f]{12}", identifier)
                    or manifest.get("binding") != self.binding
                    or (self.root / identifier).is_symlink()
                    or json.loads((self.root / identifier / "manifest.json").read_text()) != manifest):
                raise ValueError("Cannot clean checkpoints with an invalid committed pointer")
            keep.update((identifier, manifest.get("previous_id")))
        discarded = []
        for path in self.root.iterdir():
            if not path.is_dir() or path.is_symlink() or path.name in keep:
                continue
            if path.name.startswith(".staging-"):
                # SIGKILL can also leave atomic_json's temporary manifest.
                if any((item.name not in {"state.pt", "manifest.json"}
                        and not item.name.startswith(".commit-"))
                       or item.is_symlink() or not item.is_file()
                       for item in path.iterdir()):
                    continue
                metadata = path / "manifest.json"
                if metadata.exists() and json.loads(metadata.read_text()).get("binding") != self.binding:
                    continue
            elif re.fullmatch(r"(?:checkpoint|collection)-\d+-[0-9a-f]{12}", path.name):
                metadata = path / "manifest.json"
                if not metadata.is_file():
                    continue
                manifest = json.loads(metadata.read_text())
                if manifest.get("id") != path.name or manifest.get("binding") != self.binding:
                    continue
            else:
                continue
            shutil.rmtree(path)
            discarded.append(path.name)
        if discarded:
            _sync_directory(self.root)
        return discarded

    def save(self, kind, payload):
        pointer = self._pointer(kind)
        previous = json.loads(pointer.read_text())["id"] if pointer.exists() else None
        identifier = kind + "-" + str(time.time_ns()) + "-" + uuid.uuid4().hex[:12]
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))
        destination = self.root / identifier
        try:
            data_path = stage / "state.pt"
            with data_path.open("wb") as output:
                self.torch.save({"schema_version": 1, "binding": self.binding, "payload": payload}, output)
                output.flush()
                os.fsync(output.fileno())
            manifest = {"schema_version": 1, "id": identifier, "kind": kind,
                        "previous_id": previous,
                        "binding": self.binding, "sha256": file_digest(data_path),
                        "bytes": data_path.stat().st_size}
            atomic_json(stage / "manifest.json", manifest)
            os.replace(stage, destination)
            _sync_directory(self.root)
            atomic_json(pointer, manifest)
            self._prune(kind, identifier, previous)
            return identifier
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def _prune(self, kind, current, previous):
        # Only self-created complete generations; no data/weights elsewhere.
        candidates = []
        for path in self.root.glob(kind + "-*"):
            if path.is_dir() and not path.is_symlink() and (path / "manifest.json").is_file():
                candidates.append(path)
        keep = {current, previous}
        for path in candidates:
            if path.name not in keep:
                shutil.rmtree(path)

    def inspect(self, kind):
        """Verify a committed file before model loading without materializing it."""
        pointer = self._pointer(kind)
        if not pointer.is_file():
            return None
        manifest = json.loads(pointer.read_text())
        identifier = manifest.get("id", "")
        if (not isinstance(identifier, str) or not identifier.startswith(kind + "-")
                or Path(identifier).name != identifier or manifest.get("binding") != self.binding
                or manifest.get("schema_version") != 1):
            raise ValueError("Checkpoint identity/source binding does not match this run")
        directory = self.root / identifier
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("Checkpoint generation is missing or is a symlink")
        if json.loads((directory / "manifest.json").read_text()) != manifest:
            raise ValueError("Checkpoint pointer and generation manifest disagree")
        path = directory / "state.pt"
        if path.is_symlink() or path.stat().st_size != manifest["bytes"] or file_digest(path) != manifest["sha256"]:
            raise ValueError("Checkpoint payload checksum failed")
        return manifest

    def load(self, kind):
        manifest = self.inspect(kind)
        if manifest is None:
            return None
        identifier = manifest["id"]
        path = self.root / identifier / "state.pt"
        # Large adapter/Adam states need not be copied wholesale into host RAM
        # before restoration. Private mappings cannot mutate the saved file.
        with self.torch.serialization.set_default_mmap_options(mmap.MAP_PRIVATE):
            state = self.torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if state.get("schema_version") != 1 or state.get("binding") != self.binding:
            raise ValueError("Checkpoint payload binding is invalid")
        return identifier, state["payload"]


class RunBudget:
    """Resuming does not silently grant another full run's GPU-time allowance.

    Clean exits account active wall time. After an unclean exit, unobserved time
    is conservatively charged up to the old remaining allowance; explicit
    --additional-walltime-seconds is the only way to extend that allowance.
    """

    @staticmethod
    def remaining(run_dir, initial, additional=0.0, now=None):
        now = time.time() if now is None else now
        path = Path(run_dir) / "recovery" / "budget.json"
        if not path.exists():
            return float(initial) + float(additional)
        state = json.loads(path.read_text())
        spent = float(state["spent_seconds"])
        if state.get("active_started_unix") is not None:
            spent = min(float(state["allowed_seconds"]), spent + max(0.0, now - state["active_started_unix"]))
        return max(0.0, float(state["allowed_seconds"]) - spent) + float(additional)

    def __init__(self, run_dir, initial, additional=0.0):
        if any(not math.isfinite(value) or value < 0 for value in (float(initial), float(additional))):
            raise ValueError("Run budget must be finite and nonnegative")
        self.path = Path(run_dir) / "recovery" / "budget.json"
        now = time.time()
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get("active_started_unix") is not None:
                self.state["spent_seconds"] = min(self.state["allowed_seconds"], self.state["spent_seconds"] +
                                                   max(0.0, now - self.state["active_started_unix"]))
        else:
            self.state = {"allowed_seconds": float(initial), "spent_seconds": 0.0, "extensions": []}
        if additional:
            self.state["allowed_seconds"] += float(additional)
            self.state["extensions"].append({"time_unix": now, "seconds": float(additional)})
        self.remaining_seconds = max(0.0, self.state["allowed_seconds"] - self.state["spent_seconds"])
        if not self.remaining_seconds:
            raise RuntimeError("No run budget remains; an explicit additional allowance is required")
        self.started = time.monotonic()
        self.state["active_started_unix"] = now
        atomic_json(self.path, self.state)

    def close(self):
        if self.state.get("active_started_unix") is None:
            return
        self.state["spent_seconds"] = min(self.state["allowed_seconds"], self.state["spent_seconds"] +
                                           max(0.0, time.monotonic() - self.started))
        self.state["active_started_unix"] = None
        atomic_json(self.path, self.state)
