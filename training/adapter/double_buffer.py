"""N+1 learner / N rollout ownership with trajectory-pinned adapters."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import threading
from typing import Any

from training.rollout.backend import GpuLease, InferenceBackend
from training.rollout.policy_version import AdapterManifest, PolicyVersion

from .manager import AdapterManager


class DoubleBuffer:
    """Own one writable candidate and one frozen rollout version.

    Trajectories retain their original manifest independently of the current
    rollout pointer. In the default non-resident mode, publication returns
    False while any old trajectory is active; it never rebinds those requests.
    """

    def __init__(self, manager: AdapterManager, backend: InferenceBackend | None = None,
                 *, gpu_lease: GpuLease | None = None, allow_resident_pair: bool = False):
        if type(allow_resident_pair) is not bool:
            raise TypeError("allow_resident_pair must be a bool")
        if gpu_lease is not None and backend is not None and gpu_lease.backend is not backend:
            raise ValueError("gpu_lease and backend disagree")
        self.manager = manager
        self.backend = backend if backend is not None else (gpu_lease.backend if gpu_lease else None)
        self.gpu_lease = gpu_lease or (GpuLease(self.backend) if self.backend is not None else None)
        self.allow_resident_pair = allow_resident_pair
        self._lock = threading.RLock()
        self._training: AdapterManifest | None = None
        self._rollout: AdapterManifest | None = manager.active
        self._trajectories: dict[str, AdapterManifest] = {}
        self._trajectory_refs: Counter[int] = Counter()
        if self._rollout is not None:
            self.manager.acquire(self._rollout)

    @property
    def training(self) -> AdapterManifest | None:
        with self._lock:
            return self._training

    @property
    def rollout(self) -> AdapterManifest | None:
        with self._lock:
            return self._rollout

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._training is not None

    @staticmethod
    def cache_namespace(adapter: AdapterManifest | PolicyVersion | str) -> str:
        if isinstance(adapter, (AdapterManifest, PolicyVersion)):
            return adapter.adapter_sha256
        if isinstance(adapter, str) and len(adapter) == 64 \
                and all(char in "0123456789abcdef" for char in adapter):
            return adapter
        raise ValueError("cache namespace requires an exact adapter SHA256")

    def _inference(self):
        if self.backend is None:
            return _NullBackendContext()
        return self.gpu_lease.inference()

    def _registered(self, manifest: AdapterManifest, source: str | Path | None) -> AdapterManifest:
        try:
            return self.manager.get(manifest)
        except KeyError:
            if source is None:
                raise ValueError("an unregistered training candidate requires its source directory")
            return self.manager.publish(manifest, source, activate=False)

    def begin_training(self, manifest: AdapterManifest, source: str | Path | None = None) -> AdapterManifest:
        """Register if necessary and acquire the unique writable N+1 slot."""
        if not isinstance(manifest, AdapterManifest):
            raise TypeError("training candidate must be an AdapterManifest")
        with self._lock:
            if self._training is not None:
                raise ValueError("the training N+1 slot is already owned")
            base = self._rollout
            if base is None:
                if manifest.version != 0:
                    raise ValueError("the first buffer version must be genesis version 0")
            elif manifest.parent_version != base.version or manifest.version != base.version + 1:
                raise ValueError("training N+1 must be the direct child of rollout N")
            registered = self._registered(manifest, source)
            self.manager.acquire(registered)
            self._training = registered
            return registered

    def _load_for_publish(self, candidate: AdapterManifest, old: AdapterManifest | None,
                          resident: bool) -> dict[str, Any]:
        """Prepare serving and return compensation information."""
        compensation = {"unloaded_old": False, "loaded_new": False}
        if self.backend is None:
            return compensation
        with self._inference():
            if old is not None and not resident:
                self.backend.unload_adapter(old)
                compensation["unloaded_old"] = True
            try:
                self.backend.load_adapter(candidate, self.manager.path(candidate))
                compensation["loaded_new"] = True
            except BaseException:
                if compensation["unloaded_old"]:
                    self.backend.load_adapter(old, self.manager.path(old))
                    compensation["unloaded_old"] = False
                raise
        return compensation

    def _compensate_publish(self, candidate: AdapterManifest, old: AdapterManifest | None,
                            compensation: dict[str, Any]) -> None:
        if self.backend is None:
            return
        with self._inference():
            if compensation.get("loaded_new"):
                self.backend.unload_adapter(candidate)
            if old is not None and compensation.get("unloaded_old"):
                self.backend.load_adapter(old, self.manager.path(old))

    def publish(self) -> bool:
        """Publish training N+1 if doing so cannot change old trajectories."""
        with self._lock:
            candidate = self._training
            if candidate is None:
                raise ValueError("no training candidate is waiting for publication")
            old = self._rollout
            if old is not None and candidate.parent_version != old.version:
                raise ValueError("candidate is no longer the direct child of rollout")
            old_active = old is not None and self._trajectory_refs[old.version] > 0
            if old_active:
                if not self.allow_resident_pair:
                    return False
                older = [version for version, count in self._trajectory_refs.items()
                         if count and (old is None or version != old.version)]
                if older:
                    # A double buffer never admits a third resident behavior.
                    return False
            resident = self.allow_resident_pair
            compensation = self._load_for_publish(candidate, old, resident)
            try:
                self.manager.acquire(candidate)  # frozen rollout ownership
                try:
                    self.manager.activate(candidate)
                except BaseException:
                    self.manager.release(candidate)
                    raise
            except BaseException:
                self._compensate_publish(candidate, old, compensation)
                raise
            self.manager.release(candidate)  # release writable N+1 ownership
            if old is not None:
                self.manager.release(old)
            self._rollout = candidate
            self._training = None
            # Resident mode permits two versions but does not require retaining
            # an unreferenced old adapter.
            if self.backend is not None and old is not None \
                    and self._trajectory_refs[old.version] == 0 and resident:
                with self._inference():
                    self.backend.unload_adapter(old)
            return True

    publish_candidate = publish

    def abort_training(self) -> None:
        with self._lock:
            if self._training is None:
                return
            self.manager.release(self._training)
            self._training = None

    @property
    def rollout_refcount(self) -> int:
        """Live trajectory references pinning the current rollout version."""
        with self._lock:
            if self._rollout is None:
                return 0
            return int(self._trajectory_refs.get(self._rollout.version, 0))

    def wait_drained(self, timeout: float = 300.0) -> bool:
        """Poll until no trajectory pins the rollout version."""
        import time as _time
        deadline = _time.monotonic() + max(0.0, float(timeout))
        while self.rollout_refcount > 0:
            if _time.monotonic() >= deadline:
                return False
            _time.sleep(0.2)
        return True

    def bind_trajectory(self, trajectory_id: str) -> AdapterManifest:
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("trajectory_id must be a non-empty string")
        with self._lock:
            if trajectory_id in self._trajectories:
                raise ValueError("trajectory_id is already bound")
            if self._rollout is None:
                raise ValueError("no rollout adapter is available")
            manifest = self._rollout
            self.manager.acquire(manifest)
            self._trajectories[trajectory_id] = manifest
            self._trajectory_refs[manifest.version] += 1
            return manifest

    def adapter_for(self, trajectory_id: str) -> AdapterManifest:
        with self._lock:
            try:
                return self._trajectories[trajectory_id]
            except KeyError as exc:
                raise KeyError("trajectory has no pinned adapter") from exc

    def submission(self, trajectory_id: str) -> dict[str, Any]:
        manifest = self.adapter_for(trajectory_id)
        return {"adapter": manifest, "cache_namespace": self.cache_namespace(manifest)}

    def submit(self, trajectory_id: str, input_ids: list[int], **kwargs: Any) -> dict[str, Any]:
        if self.backend is None:
            raise RuntimeError("submit requires an inference backend")
        from ..rollout.gateway import validate_sampling
        kwargs["sampling"] = validate_sampling(kwargs.get("sampling"))
        binding = self.submission(trajectory_id)
        with self._inference():
            return self.backend.submit(
                input_ids,
                adapter=binding["adapter"],
                cache_namespace=binding["cache_namespace"],
                **kwargs,
            )

    def release_trajectory(self, trajectory_id: str, *, auto_publish: bool = True) -> AdapterManifest:
        with self._lock:
            manifest = self.adapter_for(trajectory_id)
            del self._trajectories[trajectory_id]
            self._trajectory_refs[manifest.version] -= 1
            if self._trajectory_refs[manifest.version] <= 0:
                del self._trajectory_refs[manifest.version]
            self.manager.release(manifest)
            retired = (self._rollout is None or manifest.version != self._rollout.version) \
                and (self._training is None or manifest.version != self._training.version) \
                and self._trajectory_refs[manifest.version] == 0
            if retired and self.backend is not None:
                with self._inference():
                    self.backend.unload_adapter(manifest)
            if auto_publish and self._training is not None:
                self.publish()
            return manifest

    def unregister_retired(self, adapter: AdapterManifest | PolicyVersion | int | str,
                           *, delete: bool = False) -> None:
        with self._lock:
            manifest = self.manager.get(adapter)
            if ((self._rollout and manifest.version == self._rollout.version)
                    or (self._training and manifest.version == self._training.version)
                    or self._trajectory_refs[manifest.version]):
                raise ValueError("buffer or trajectory ownership still protects this adapter")
            self.manager.unregister(manifest, delete=delete)

    def ownership(self) -> dict[str, Any]:
        with self._lock:
            return {
                "training": self._training.to_dict() if self._training else None,
                "rollout": self._rollout.to_dict() if self._rollout else None,
                "trajectory_versions": dict(self._trajectory_refs),
                "allow_resident_pair": self.allow_resident_pair,
            }

    def close(self) -> None:
        """Release local registry references without deleting adapter files."""
        with self._lock:
            for manifest in self._trajectories.values():
                self.manager.release(manifest)
            self._trajectories.clear()
            self._trajectory_refs.clear()
            if self._training is not None:
                self.manager.release(self._training)
                self._training = None
            if self._rollout is not None:
                self.manager.release(self._rollout)
                self._rollout = None


class _NullBackendContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):
        return False
