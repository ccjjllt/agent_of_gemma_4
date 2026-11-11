"""Strict HTTP inference backends and colocated GPU ownership.

The HTTP clients do not launch a server and never emulate a missing engine
feature. Control operations require explicit JSON confirmations from the
server; generation requires the original token IDs and sampled logprobs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
import json
import math
from pathlib import Path
import threading
from typing import Any, Iterator, Mapping, Union
import urllib.error
import urllib.request
import uuid

from .policy_version import AdapterManifest, PolicyVersion, sha256_file


class BackendError(RuntimeError):
    """An inference engine request or response violates the backend contract."""


class BackendCapabilityError(BackendError):
    """The selected engine does not implement a required operation."""


class BackendResponseError(BackendError):
    """The engine omitted or invalidated required sampling evidence."""


AdapterIdentity = Union[AdapterManifest, PolicyVersion, str]


def _adapter_hash(adapter: AdapterIdentity) -> str:
    if isinstance(adapter, (AdapterManifest, PolicyVersion)):
        return adapter.adapter_sha256
    if isinstance(adapter, str):
        # Reuse the manifest's strict digest validation without constructing a
        # partial policy identity.
        if len(adapter) != 64 or any(c not in "0123456789abcdef" for c in adapter):
            raise ValueError("adapter must be a PolicyVersion, AdapterManifest, or SHA256")
        return adapter
    raise TypeError("adapter must be a PolicyVersion, AdapterManifest, or SHA256")


def _token_ids(value: Any, name: str, *, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise BackendResponseError(f"{name} must be a {'possibly empty ' if allow_empty else ''}list")
    if any(type(token) is not int or token < 0 for token in value):
        raise BackendResponseError(f"{name} must contain non-negative integer token IDs")
    return list(value)


class InferenceBackend(ABC):
    """Required boundary between rollout and a token-native inference engine."""

    @abstractmethod
    def start(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def health(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def submit(self, input_ids: list[int], *, adapter: AdapterIdentity,
               request_id: str | None = None, sampling: Mapping[str, Any] | None = None,
               cache_namespace: str | None = None, timeout: float | None = None) -> dict[str, Any]: ...

    @abstractmethod
    def cancel(self, request_id: str) -> Mapping[str, Any]: ...

    @abstractmethod
    def drain(self, timeout: float | None = None) -> Mapping[str, Any]: ...

    @abstractmethod
    def pause(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def resume(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def release_memory(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def resume_memory(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def load_adapter(self, manifest: AdapterManifest, path: str | Path) -> Mapping[str, Any]: ...

    @abstractmethod
    def unload_adapter(self, adapter: AdapterIdentity) -> Mapping[str, Any]: ...

    @abstractmethod
    def metrics(self) -> Any: ...


class HTTPInferenceBackend(InferenceBackend):
    """Shared stdlib HTTP transport for strict engine-specific endpoints."""

    name = "http"
    paths = {
        "health": "/health",
        "generate": "/generate",
        "cancel": "/cancel",
        "drain": "/drain",
        "pause": "/pause",
        "resume": "/resume",
        "release_memory": "/release_memory_occupation",
        "resume_memory": "/resume_memory_occupation",
        "load_adapter": "/load_lora_adapter",
        "unload_adapter": "/unload_lora_adapter",
        "metrics": "/metrics",
    }

    def __init__(self, base_url: str, *, timeout: float = 30.0,
                 api_key: str | None = None, endpoints: Mapping[str, str] | None = None):
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an HTTP(S) URL")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) \
                or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.api_key = api_key
        self.endpoints = dict(self.paths)
        for key, path in (endpoints or {}).items():
            if key not in self.endpoints or not isinstance(path, str) or not path.startswith("/"):
                raise ValueError("endpoint overrides require known names and absolute HTTP paths")
            self.endpoints[key] = path
        self._started = False
        self._paused = False
        self._memory_released = False
        self._adapters: dict[str, tuple[AdapterManifest, str]] = {}
        self._active_adapter_hash: str | None = None
        self._state_lock = threading.RLock()

    @property
    def loaded_adapters(self) -> dict[str, str]:
        with self._state_lock:
            return {digest: path for digest, (_, path) in self._adapters.items()}

    @property
    def adapter_manifests(self) -> dict[str, AdapterManifest]:
        with self._state_lock:
            return {digest: manifest for digest, (manifest, _) in self._adapters.items()}

    @property
    def active_adapter(self) -> AdapterManifest | None:
        with self._state_lock:
            item = self._adapters.get(self._active_adapter_hash or "")
            return item[0] if item else None

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None,
                 *, timeout: float | None = None) -> Any:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            try:
                body = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError("request payload is not finite JSON") from exc
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout if timeout is None else timeout) as response:
                raw = response.read()
                content_type = response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            error_type = BackendCapabilityError if exc.code in {404, 405, 501} else BackendError
            raise error_type(f"{self.name} {method} {path} returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BackendError(f"{self.name} {method} {path} failed: {exc}") from exc
        if not raw:
            return None
        if "json" in content_type.lower() or raw.lstrip().startswith((b"{", b"[")):
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise BackendResponseError(f"{self.name} returned invalid JSON for {path}") from exc
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BackendResponseError(f"{self.name} returned a non-text response for {path}") from exc

    @staticmethod
    def _object(value: Any, operation: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise BackendResponseError(f"{operation} requires a JSON object response")
        return value

    @staticmethod
    def _confirmed(value: Any, operation: str, field: str) -> Mapping[str, Any]:
        result = HTTPInferenceBackend._object(value, operation)
        if result.get(field) is not True:
            raise BackendResponseError(f"{operation} requires explicit {field}=true confirmation")
        return result

    def _require_started(self) -> None:
        if not self._started:
            raise BackendError("start() and a successful health check are required first")

    def _require_inference(self) -> None:
        self._require_started()
        if self._paused:
            raise BackendError("inference is paused")
        if self._memory_released:
            raise BackendError("inference memory is released for training")

    def start(self) -> Mapping[str, Any]:
        with self._state_lock:
            result = self.health()
            self._started = True
            return result

    def health(self) -> Mapping[str, Any]:
        result = self._request("GET", self.endpoints["health"])
        # Some engines use an empty HTTP 200 health response. HTTP success is
        # the evidence in that case; no model or adapter support is inferred.
        return {"status": "ok"} if result is None else self._object(result, "health")

    def _normalize_generation(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return payload

    def submit(self, input_ids: list[int], *, adapter: AdapterIdentity,
               request_id: str | None = None, sampling: Mapping[str, Any] | None = None,
               cache_namespace: str | None = None, timeout: float | None = None) -> dict[str, Any]:
        with self._state_lock:
            self._require_inference()
        try:
            prompt = _token_ids(input_ids, "input_ids")
        except BackendResponseError as exc:
            raise ValueError(str(exc)) from exc
        digest = _adapter_hash(adapter)
        if request_id is None:
            request_id = uuid.uuid4().hex
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        if cache_namespace is not None and cache_namespace != digest:
            raise ValueError("cache_namespace must equal the behavior adapter SHA256")
        payload = {
            "request_id": request_id,
            "input_ids": prompt,
            "adapter_sha256": digest,
            "lora_name": digest,
            "cache_namespace": digest,
            "sampling_params": dict(sampling or {}),
        }
        raw = self._object(self._request("POST", self.endpoints["generate"], payload, timeout=timeout), "submit")
        result = self._normalize_generation(raw)
        returned_prompt = _token_ids(result.get("input_ids"), "response input_ids")
        if returned_prompt != prompt:
            raise BackendResponseError("response input_ids do not exactly match the submitted IDs")
        output_ids = _token_ids(result.get("output_ids"), "response output_ids", allow_empty=True)
        logprobs = result.get("logprobs")
        if not isinstance(logprobs, list) or len(logprobs) != len(output_ids):
            raise BackendResponseError("logprobs must be present and aligned one-to-one with output_ids")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
               for item in logprobs):
            raise BackendResponseError("every sampled logprob must be a finite number")
        finish_reason = result.get("finish_reason")
        if isinstance(finish_reason, Mapping):
            finish_reason = finish_reason.get("type")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise BackendResponseError("finish_reason is required")
        if result.get("adapter_sha256") != digest:
            raise BackendResponseError("response adapter_sha256 is missing or differs from the requested adapter")
        returned_request = result.get("request_id", request_id)
        if returned_request != request_id:
            raise BackendResponseError("response request_id differs from the submitted request")
        submitted_sampling = dict(sampling or {})
        echoed = result.get("sampling_params", result.get("sampling"))
        if isinstance(echoed, Mapping):
            for key, value in submitted_sampling.items():
                if key in echoed and echoed[key] != value:
                    raise BackendResponseError(
                        f"response sampling_params[{key}] differs from the submitted value")
            sampling_record = {**submitted_sampling, **{k: v for k, v in echoed.items()
                               if isinstance(k, str)}}
        else:
            sampling_record = dict(submitted_sampling)
        return {
            "request_id": request_id,
            "input_ids": returned_prompt,
            "output_ids": output_ids,
            "logprobs": [float(item) for item in logprobs],
            "finish_reason": finish_reason,
            "adapter_sha256": digest,
            "sampling": sampling_record,
            "sampling_server_echo": isinstance(echoed, Mapping),
        }

    def cancel(self, request_id: str) -> Mapping[str, Any]:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        self._require_started()
        return self._confirmed(
            self._request("POST", self.endpoints["cancel"], {"request_id": request_id}),
            "cancel", "cancelled",
        )

    def drain(self, timeout: float | None = None) -> Mapping[str, Any]:
        self._require_started()
        payload = {} if timeout is None else {"timeout_seconds": timeout}
        return self._confirmed(
            self._request("POST", self.endpoints["drain"], payload, timeout=timeout),
            "drain", "drained",
        )

    def pause(self) -> Mapping[str, Any]:
        self._require_inference()
        result = self._confirmed(self._request("POST", self.endpoints["pause"], {}), "pause", "paused")
        with self._state_lock:
            self._paused = True
        return result

    def resume(self) -> Mapping[str, Any]:
        self._require_started()
        if self._memory_released:
            raise BackendError("resume_memory() is required before resuming generation")
        result = self._confirmed(self._request("POST", self.endpoints["resume"], {}), "resume", "resumed")
        with self._state_lock:
            self._paused = False
        return result

    def release_memory(self) -> Mapping[str, Any]:
        self._require_started()
        result = self._confirmed(
            self._request("POST", self.endpoints["release_memory"], {}),
            "release_memory", "released",
        )
        with self._state_lock:
            self._memory_released = True
        return result

    def resume_memory(self) -> Mapping[str, Any]:
        self._require_started()
        result = self._confirmed(
            self._request("POST", self.endpoints["resume_memory"], {}),
            "resume_memory", "resumed",
        )
        with self._state_lock:
            self._memory_released = False
        return result

    def load_adapter(self, manifest: AdapterManifest, path: str | Path) -> Mapping[str, Any]:
        self._require_started()
        if self._memory_released:
            raise BackendError("adapter memory cannot be loaded while engine memory is released")
        if not isinstance(manifest, AdapterManifest):
            raise TypeError("load_adapter requires an AdapterManifest")
        directory = Path(path).expanduser().resolve(strict=True)
        weights = directory / "adapter_model.safetensors"
        if not weights.is_file() or sha256_file(weights) != manifest.adapter_sha256:
            raise ValueError("adapter weights are missing or do not match the manifest")
        result = self._confirmed(self._request("POST", self.endpoints["load_adapter"], {
            "lora_name": manifest.adapter_sha256,
            "lora_path": str(directory),
            "adapter_sha256": manifest.adapter_sha256,
        }), "load_adapter", "loaded")
        if result.get("adapter_sha256") != manifest.adapter_sha256:
            raise BackendResponseError("load_adapter must confirm the exact adapter_sha256")
        with self._state_lock:
            self._adapters[manifest.adapter_sha256] = (manifest, str(directory))
            self._active_adapter_hash = manifest.adapter_sha256
        return result

    def unload_adapter(self, adapter: AdapterIdentity) -> Mapping[str, Any]:
        self._require_started()
        if self._memory_released:
            raise BackendError("adapter memory cannot be changed while engine memory is released")
        digest = _adapter_hash(adapter)
        result = self._confirmed(self._request("POST", self.endpoints["unload_adapter"], {
            "lora_name": digest,
            "adapter_sha256": digest,
        }), "unload_adapter", "unloaded")
        if result.get("adapter_sha256") != digest:
            raise BackendResponseError("unload_adapter must confirm the exact adapter_sha256")
        with self._state_lock:
            self._adapters.pop(digest, None)
            if self._active_adapter_hash == digest:
                self._active_adapter_hash = None
        return result

    def metrics(self) -> Any:
        self._require_started()
        result = self._request("GET", self.endpoints["metrics"])
        if result is None or result == "":
            raise BackendResponseError("metrics returned an empty response")
        return result

    release_memory_occupation = release_memory
    resume_memory_occupation = resume_memory


class SGLangBackend(HTTPInferenceBackend):
    """Native SGLang protocol translation (verified against sglang 0.4.10).

    Deviations from stock SGLang that this client intentionally does NOT
    assume are remapped here instead of in deployment docs:

    - ``request_id`` -> native ``rid``; the response binds via
      ``meta_info.id``.  Stock /generate does not echo ``input_ids``; the
      client returns the submitted copy labeled ``input_ids_server_echo``.
    - ``top_k=0`` (our "unfiltered" spelling) -> native ``-1``; native
      ``max_new_tokens`` has no safe training default, so the caller must
      supply it explicitly in ``sampling`` (stock default 128 would silently
      truncate rollouts).  Native sampling has no seed channel; ``seed``
      stays a client-side record (``-1`` = engine-owned RNG).
    - ``lora_path`` carries the registered LoRA name (chosen as the adapter
      SHA256 at load time); per-request binding is ``rid``-scoped and the
      name is resolved server-side through the LoRA registry. The weight
      hash is proven at load time, not per response. Submitting with an
      unloaded adapter fails instead of sending an unregistered name.
    - ``drain`` has no native route: pause (idempotent) plus polling
      ``/get_load`` for zero running/waiting requests.
    - Empty-200 native confirmations (abort/release/resume-memory) count as
      success; ``pause``/``resume`` require ``status == "ok"``.
    """

    name = "sglang"
    paths = dict(HTTPInferenceBackend.paths,
                 pause="/pause_generation", resume="/continue_generation",
                 cancel="/abort_request", metrics="/get_load")

    @staticmethod
    def _native_sampling(sampling: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(sampling, Mapping):
            raise BackendResponseError("sampling must be a mapping")
        try:
            temperature = float(sampling["temperature"])
            top_p = float(sampling["top_p"])
            top_k = int(sampling["top_k"])
            max_new_tokens = int(sampling["max_new_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendResponseError(
                "sglang sampling requires temperature/top_p/top_k/max_new_tokens") from exc
        if not math.isfinite(temperature) or temperature < 0:
            raise BackendResponseError("sglang temperature must be finite and non-negative")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise BackendResponseError("sglang top_p must be in (0, 1]")
        native_top_k = -1 if top_k == 0 else top_k
        if native_top_k != -1 and native_top_k < 1:
            raise BackendResponseError("sglang top_k must be 0 (unfiltered), -1 or >= 1")
        if max_new_tokens <= 0:
            raise BackendResponseError("sglang max_new_tokens must be positive")
        return {"temperature": temperature, "top_p": top_p, "top_k": native_top_k,
                "max_new_tokens": max_new_tokens}

    def submit(self, input_ids: list[int], *, adapter: AdapterIdentity,
               request_id: str | None = None, sampling: Mapping[str, Any] | None = None,
               cache_namespace: str | None = None, timeout: float | None = None) -> dict[str, Any]:
        with self._state_lock:
            self._require_inference()
        try:
            prompt = _token_ids(input_ids, "input_ids")
        except BackendResponseError as exc:
            raise ValueError(str(exc)) from exc
        digest = _adapter_hash(adapter)
        rid = request_id if isinstance(request_id, str) and request_id else uuid.uuid4().hex
        if not isinstance(rid, str) or not rid:
            raise ValueError("request_id must be a non-empty string")
        if cache_namespace is not None and cache_namespace != digest:
            raise ValueError("cache_namespace must equal the behavior adapter SHA256")
        native_sampling = self._native_sampling(sampling or {})
        # Per-request binding uses the registered LoRA NAME, not a filesystem
        # path: stock /generate resolves lora_path through the server-side
        # LoRA registry populated by /load_lora_adapter (verified against the
        # sglang 0.4.10 wheel: lora_registry.acquire(obj.lora_path)). The
        # registered name is exactly the adapter SHA256 chosen at load time,
        # so sha256 -> name -> directory stays consistent end to end.
        with self._state_lock:
            registered = self._adapters.get(digest)
        if registered is None:
            raise BackendError(
                "adapter is not loaded on the inference server; "
                "call load_adapter(manifest, path) before submitting")
        lora_name = digest
        payload = {
            "rid": rid,
            "input_ids": prompt,
            "sampling_params": native_sampling,
            "lora_path": lora_name,
            "return_logprob": True,
            "top_logprobs_num": 1,
            "stream": False,
        }
        raw = self._object(self._request("POST", self.endpoints["generate"], payload, timeout=timeout), "submit")
        meta = raw.get("meta_info")
        if not isinstance(meta, Mapping):
            raise BackendResponseError("sglang response is missing meta_info")
        if meta.get("id") != rid:
            raise BackendResponseError("sglang meta_info.id does not bind this rid")
        output_ids = _token_ids(raw.get("output_ids"), "response output_ids", allow_empty=True)
        rows = meta.get("output_token_logprobs")
        if isinstance(rows, list) and rows and all(isinstance(row, (list, tuple)) for row in rows):
            logprobs = [row[0] for row in rows]
        elif isinstance(meta.get("output_token_logprobs_val"), list):
            logprobs = list(meta["output_token_logprobs_val"])
        else:
            raise BackendResponseError("sglang response has no output token logprobs")
        if len(logprobs) != len(output_ids):
            raise BackendResponseError("sglang logprobs are not one-to-one with output_ids")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
               for item in logprobs):
            raise BackendResponseError("every sampled logprob must be a finite number")
        finish_reason = meta.get("finish_reason")
        if isinstance(finish_reason, Mapping):
            finish_reason = finish_reason.get("type")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise BackendResponseError("finish_reason is required")
        submitted_sampling = dict(sampling or {})
        return {
            "request_id": rid,
            "input_ids": list(prompt),
            "input_ids_server_echo": False,
            "output_ids": output_ids,
            "logprobs": [float(item) for item in logprobs],
            "finish_reason": finish_reason,
            "adapter_sha256": digest,
            "sampling": submitted_sampling,
            "sampling_server_echo": False,
        }

    def cancel(self, request_id: str) -> Mapping[str, Any]:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        self._require_started()
        self._request("POST", self.endpoints["cancel"], {"rid": request_id})
        return {"cancelled": True, "request_id": request_id}

    def drain(self, timeout: float | None = None) -> Mapping[str, Any]:
        self._require_started()
        self._request("POST", self.endpoints["pause"], {})
        import time as _time
        deadline = _time.monotonic() + (30.0 if timeout is None else max(0.0, float(timeout)))
        last: Any = None
        while True:
            last = self._request("GET", self.endpoints["metrics"])
            running = waiting = None
            queue: list[Any] = [last]
            while queue:
                node = queue.pop()
                if isinstance(node, Mapping):
                    for key, value in node.items():
                        name = str(key).lower()
                        if name in {"num_running_reqs", "num_running_requests"} and running is None:
                            running = value
                        elif name in {"num_waiting_reqs", "num_waiting_requests", "num_queued_reqs"} \
                                and waiting is None:
                            waiting = value
                        else:
                            queue.append(value)
                elif isinstance(node, list):
                    queue.extend(node)
            if isinstance(running, int) and isinstance(waiting, int):
                if running <= 0 and waiting <= 0:
                    return {"drained": True, "load": last}
            else:
                raise BackendResponseError(
                    f"sglang /get_load has no running/waiting counts: {last!r:.200}")
            if _time.monotonic() >= deadline:
                raise BackendError(f"sglang drain timed out with load: {last!r:.200}")
            _time.sleep(0.5)

    def _ok(self, operation: str, path_key: str, payload: Mapping[str, Any] | None = None) -> Any:
        result = self._request("POST", self.endpoints[path_key], dict(payload or {}))
        if result is None:
            return None
        return self._object(result, operation)

    def pause(self) -> Mapping[str, Any]:
        self._require_inference()
        result = self._ok("pause", "pause")
        if isinstance(result, Mapping) and result.get("status") != "ok":
            raise BackendResponseError("sglang pause requires status ok")
        with self._state_lock:
            self._paused = True
        return {"paused": True}

    def resume(self) -> Mapping[str, Any]:
        self._require_started()
        if self._memory_released:
            raise BackendError("resume_memory() is required before resuming generation")
        result = self._ok("resume", "resume")
        if isinstance(result, Mapping) and result.get("status") != "ok":
            raise BackendResponseError("sglang resume requires status ok")
        with self._state_lock:
            self._paused = False
        return {"resumed": True}

    def release_memory(self) -> Mapping[str, Any]:
        self._require_started()
        self._ok("release_memory", "release_memory")
        with self._state_lock:
            self._memory_released = True
        return {"released": True}

    def resume_memory(self) -> Mapping[str, Any]:
        self._require_started()
        self._ok("resume_memory", "resume_memory")
        with self._state_lock:
            self._memory_released = False
        return {"resumed": True}

    def load_adapter(self, manifest: AdapterManifest, path: str | Path) -> Mapping[str, Any]:
        self._require_started()
        if self._memory_released:
            raise BackendError("adapter memory cannot be loaded while engine memory is released")
        if not isinstance(manifest, AdapterManifest):
            raise TypeError("load_adapter requires an AdapterManifest")
        directory = Path(path).expanduser().resolve(strict=True)
        weights = directory / "adapter_model.safetensors"
        if not weights.is_file() or sha256_file(weights) != manifest.adapter_sha256:
            raise ValueError("adapter weights are missing or do not match the manifest")
        lora_name = manifest.adapter_sha256
        result = self._object(self._request("POST", self.endpoints["load_adapter"], {
            "lora_name": lora_name,
            "lora_path": str(directory),
        }), "load_adapter")
        if result.get("success") is not True:
            raise BackendResponseError(
                f"sglang refused the adapter: {result.get('error_message')}")
        loaded = result.get("loaded_adapters")
        if isinstance(loaded, Mapping) and lora_name not in loaded:
            raise BackendResponseError(
                "sglang load succeeded but did not register the requested lora_name")
        with self._state_lock:
            self._adapters[manifest.adapter_sha256] = (manifest, str(directory))
            self._active_adapter_hash = manifest.adapter_sha256
        return {"loaded": True, "adapter_sha256": manifest.adapter_sha256,
                "lora_name": lora_name}

    def unload_adapter(self, adapter: AdapterIdentity) -> Mapping[str, Any]:
        self._require_started()
        if self._memory_released:
            raise BackendError("adapter memory cannot be changed while engine memory is released")
        digest = _adapter_hash(adapter)
        result = self._object(self._request("POST", self.endpoints["unload_adapter"], {
            "lora_name": digest,
        }), "unload_adapter")
        if result.get("success") is not True:
            raise BackendResponseError(
                f"sglang refused the adapter unload: {result.get('error_message')}")
        with self._state_lock:
            self._adapters.pop(digest, None)
            if self._active_adapter_hash == digest:
                self._active_adapter_hash = None
        return {"unloaded": True, "adapter_sha256": digest}

    def _normalize_generation(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Map SGLang's native meta_info without inventing absent evidence."""
        meta = payload.get("meta_info")
        if meta is None:
            return payload
        if not isinstance(meta, Mapping):
            raise BackendResponseError("SGLang meta_info must be an object")
        result = dict(payload)
        if "logprobs" not in result and "output_token_logprobs" in meta:
            rows = meta["output_token_logprobs"]
            if not isinstance(rows, list) or any(not isinstance(row, (list, tuple)) or not row for row in rows):
                raise BackendResponseError("invalid SGLang output_token_logprobs")
            result["logprobs"] = [row[0] for row in rows]
        if "finish_reason" not in result and "finish_reason" in meta:
            result["finish_reason"] = meta["finish_reason"]
        if "adapter_sha256" not in result and "adapter_sha256" in meta:
            result["adapter_sha256"] = meta["adapter_sha256"]
        if "request_id" not in result and "id" in meta:
            result["request_id"] = meta["id"]
        return result


class VLLMBackend(HTTPInferenceBackend):
    name = "vllm"
    paths = dict(HTTPInferenceBackend.paths,
                 load_adapter="/v1/load_lora_adapter",
                 unload_adapter="/v1/unload_lora_adapter")


# Common spellings retained as aliases, not different implementations.
VllmBackend = VLLMBackend
SGLangHTTPBackend = SGLangBackend
VLLMHTTPBackend = VLLMBackend
VllmHTTPBackend = VLLMBackend


class GpuLease:
    """Reader-writer GPU ownership for colocated inference and training.

    Any number of inference readers may hold the lease concurrently so that
    independent worker requests reach the engine together and benefit from
    continuous batching.  Training takes an exclusive lease: it stops new
    readers, server-side drains, then waits for in-flight readers (drain
    barrier) before releasing engine memory.  Inference and training contexts
    must not nest.
    """

    def __init__(self, backend: InferenceBackend):
        self.backend = backend
        self._cond = threading.Condition(threading.RLock())
        self._readers = 0
        self._writer = False
        self._drain_requested = False
        self._mode = "idle"
        self._resume_required = False
        self._memory_resume_done = False
        self._restore_adapters: dict[str, tuple[AdapterManifest, str]] = {}

    @property
    def mode(self) -> str:
        with self._cond:
            return self._mode

    @property
    def active_readers(self) -> int:
        with self._cond:
            return self._readers

    def _restore_inference(self) -> None:
        if not self._resume_required:
            return
        self._mode = "restoring"
        if not self._memory_resume_done:
            self.backend.resume_memory()
            self._memory_resume_done = True
        for digest, (manifest, path) in list(self._restore_adapters.items()):
            self.backend.load_adapter(manifest, path)
            self._restore_adapters.pop(digest, None)
        self.backend.resume()
        self._resume_required = False
        self._memory_resume_done = False
        self._mode = "inference"

    @contextmanager
    def inference(self) -> Iterator[InferenceBackend]:
        with self._cond:
            while self._writer or self._drain_requested:
                self._cond.wait()
            if self._resume_required and self._readers == 0:
                self._restore_inference()
            self._readers += 1
            self._mode = "inference"
        try:
            self.backend.health()
            yield self.backend
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0 and not self._writer:
                    self._mode = "idle"
                self._cond.notify_all()

    @contextmanager
    def training(self) -> Iterator[InferenceBackend]:
        with self._cond:
            if self._writer:
                raise BackendError("a training lease is already active")
            self._writer = True
            self._drain_requested = True
            if self._resume_required:
                self._writer = False
                self._drain_requested = False
                self._cond.notify_all()
                raise BackendError("inference restoration must succeed before another training lease")
            pause_attempted = False
            release_attempted = False
            primary_error = False
            try:
                self._mode = "draining"
                pause_attempted = True
                self.backend.pause()
                self.backend.drain()
                while self._readers > 0:
                    self._cond.wait()
                manifests = self.backend.adapter_manifests if isinstance(self.backend, HTTPInferenceBackend) else {}
                paths = self.backend.loaded_adapters if isinstance(self.backend, HTTPInferenceBackend) else {}
                self._restore_adapters = {
                    digest: (manifest, paths[digest]) for digest, manifest in manifests.items()
                }
                # A missing or malformed confirmation can mean the server did
                # release memory. Restore conservatively on every attempt, but
                # never enter the learner until release_memory() confirms it.
                self._resume_required = True
                self._memory_resume_done = False
                release_attempted = True
                self.backend.release_memory()
                self._mode = "training"
                yield self.backend
            except BaseException:
                primary_error = True
                raise
            finally:
                try:
                    if release_attempted:
                        self._restore_inference()
                        self._mode = "idle"
                    elif pause_attempted:
                        self.backend.resume()
                        self._mode = "idle"
                    else:
                        self._mode = "idle"
                except BaseException as exc:
                    self._mode = "recovery"
                    if primary_error:
                        # Preserve the learner exception while retaining the
                        # failed restoration as actionable context.
                        import sys
                        original = sys.exc_info()[1]
                        if original is not None:
                            original.add_note(f"GPU inference restoration also failed: {exc}")
                    else:
                        raise
                finally:
                    self._writer = False
                    self._drain_requested = False
                    self._cond.notify_all()
