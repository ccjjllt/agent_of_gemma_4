"""Independent vLLM OpenAI-protocol backend.

This module intentionally does not reuse SGLang's native request/response
schema.  It speaks only to the stock vLLM OpenAI server and uses the vLLM
administrative endpoints that the training lease needs:

- ``/v1/completions`` with raw token-ID prompts, ``return_token_ids`` and
  per-token logprobs;
- ``/v1/load_lora_adapter`` / ``/v1/unload_lora_adapter`` with a stable
  SHA256-to-server-name mapping;
- ``/pause`` / ``/resume`` / ``/abort_requests``;
- Level-2 ``/sleep`` / ``wake_up`` / ``collective_rpc`` (the operator must
  launch the server with ``VLLM_SERVER_DEV_MODE=1 --enable-sleep-mode``).

The shared HTTP transport in :mod:`training.rollout.backend` is reused, but
every protocol operation below is implemented against the vLLM contract.
There is no silent fallback to SGLang and no fabricated token/logprob
evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping
import uuid

from .backend import (
    AdapterIdentity,
    BackendCapabilityError,
    BackendError,
    BackendResponseError,
    HTTPInferenceBackend,
    _adapter_hash,
    _token_ids,
)
from .policy_version import AdapterManifest, sha256_file


_LORA_NAME = re.compile(r"[A-Za-z0-9_-]+\Z")
_PROMETHEUS_RUNNING = re.compile(r"^vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.eE+-]+)")
_PROMETHEUS_WAITING = re.compile(r"^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.eE+-]+)")


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


class VLLMBackend(HTTPInferenceBackend):
    """Stock vLLM OpenAI server with dynamic LoRA and Level-2 sleep support."""

    name = "vllm"
    paths = dict(
        HTTPInferenceBackend.paths,
        health="/health",
        generate="/v1/completions",
        cancel="/abort_requests",
        drain="/pause",
        pause="/pause",
        resume="/resume",
        release_memory="/sleep",
        resume_memory="/wake_up",
        load_adapter="/v1/load_lora_adapter",
        unload_adapter="/v1/unload_lora_adapter",
        metrics="/metrics",
        models="/v1/models",
        server_info="/server_info",
        collective_rpc="/collective_rpc",
        reset_prefix_cache="/reset_prefix_cache",
        is_sleeping="/is_sleeping",
        is_paused="/is_paused",
        version="/version",
    )

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        control_timeout: float = 600.0,
        api_key: str | None = None,
        expected_base_model: str | None = None,
        max_resident_adapters: int = 2,
        lora_name_prefix: str = "spo",
        require_sleep_mode: bool = True,
        require_runtime_lora: bool = True,
        sleep_level: int = 2,
        enable_prefix_cache: bool = False,
        adapter_validator: Callable[[AdapterManifest, Path], Mapping[str, Any]] | None = None,
    ):
        super().__init__(base_url, timeout=timeout, api_key=api_key)
        self.control_timeout = _positive_number(control_timeout, "control_timeout")
        if expected_base_model is not None and not isinstance(expected_base_model, str):
            raise TypeError("expected_base_model must be a string or None")
        if type(max_resident_adapters) is not int or max_resident_adapters < 1:
            raise ValueError("max_resident_adapters must be a positive integer")
        if not isinstance(lora_name_prefix, str) or not _LORA_NAME.fullmatch(lora_name_prefix):
            raise ValueError("lora_name_prefix must contain only letters, digits, _ or -")
        if sleep_level not in (1, 2):
            raise ValueError("sleep_level must be 1 or 2")
        if adapter_validator is not None and not callable(adapter_validator):
            raise TypeError("adapter_validator must be callable or None")
        self.expected_base_model = expected_base_model
        self.max_resident_adapters = max_resident_adapters
        self.lora_name_prefix = lora_name_prefix
        self.require_sleep_mode = bool(require_sleep_mode)
        self.require_runtime_lora = bool(require_runtime_lora)
        self.sleep_level = sleep_level
        self.enable_prefix_cache = bool(enable_prefix_cache)
        self.adapter_validator = adapter_validator
        self.base_model: str | None = None
        self.server_info: Mapping[str, Any] | None = None
        self.deployment: dict[str, Any] = {}
        self._paused_confirmed = False
        self._sleeping_confirmed = False

    # ------------------------------------------------------------------
    # Deployment contract
    # ------------------------------------------------------------------
    def _optional_get(self, key: str, suffix: str = "") -> Any:
        try:
            return self._request("GET", self.endpoints[key] + suffix,
                                 timeout=self.control_timeout)
        except BackendCapabilityError:
            return None

    @staticmethod
    def _config_section(info: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
        if not isinstance(info, Mapping):
            return {}
        config = info.get("vllm_config")
        if not isinstance(config, Mapping):
            return {}
        section = config.get(name)
        return section if isinstance(section, Mapping) else {}

    def _models(self) -> list[Mapping[str, Any]]:
        raw = self._object(self._request("GET", self.endpoints["models"]), "models")
        data = raw.get("data")
        if not isinstance(data, list):
            raise BackendResponseError("vLLM /v1/models must return a data list")
        return [item for item in data if isinstance(item, Mapping)]

    def _base_model_id(self, models: list[Mapping[str, Any]]) -> str | None:
        ids = [item["id"] for item in models if isinstance(item.get("id"), str)]
        if self.expected_base_model is not None:
            for model_id in ids:
                if model_id in {self.expected_base_model,
                                Path(self.expected_base_model).name,
                                Path(model_id).name}:
                    return model_id
        for item in models:
            if item.get("parent") is None and isinstance(item.get("id"), str):
                return item["id"]
        return ids[0] if ids else None

    def _model_present(self, lora_name: str) -> bool:
        return any(item.get("id") == lora_name for item in self._models())

    def _deployment_summary(self, info: Mapping[str, Any] | None) -> dict[str, Any]:
        lora = self._config_section(info, "lora_config")
        model = self._config_section(info, "model_config")
        cache = self._config_section(info, "cache_config")
        env = info.get("vllm_env") if isinstance(info, Mapping) else None
        env = env if isinstance(env, Mapping) else {}
        runtime_lora = _as_bool(env.get("VLLM_ALLOW_RUNTIME_LORA_UPDATING"))
        max_rank = lora.get("max_lora_rank")
        max_loras = lora.get("max_loras")
        max_cpu_loras = lora.get("max_cpu_loras")
        sleep_enabled = _as_bool(model.get("enable_sleep_mode"))
        prefix_cache = _as_bool(cache.get("enable_prefix_caching"))

        if self.require_runtime_lora and runtime_lora is not True:
            raise BackendCapabilityError(
                "vLLM dynamic LoRA is not enabled; launch with "
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True")
        if type(max_rank) is not int or max_rank < 128:
            raise BackendCapabilityError(
                f"vLLM --max-lora-rank must be at least 128, got {max_rank!r}")
        if type(max_loras) is not int or max_loras < max(2, self.max_resident_adapters):
            raise BackendCapabilityError(
                f"vLLM --max-loras must be at least {max(2, self.max_resident_adapters)}, "
                f"got {max_loras!r}")
        if max_cpu_loras is not None and (type(max_cpu_loras) is not int
                                          or max_cpu_loras < max_loras):
            raise BackendCapabilityError("vLLM --max-cpu-loras must be >= --max-loras")
        if self.require_sleep_mode and sleep_enabled is not True:
            raise BackendCapabilityError(
                "vLLM sleep mode is not enabled; launch with "
                "VLLM_SERVER_DEV_MODE=1 --enable-sleep-mode")
        if self.enable_prefix_cache is False and prefix_cache is True:
            raise BackendCapabilityError(
                "server prefix caching is enabled but inference.enable_prefix_cache=false; "
                "launch with --no-enable-prefix-caching for the first migration")
        sleep_state = self._optional_get("is_sleeping")
        if self.require_sleep_mode and sleep_state is None:
            raise BackendCapabilityError(
                "vLLM sleep endpoints are unavailable; launch with "
                "VLLM_SERVER_DEV_MODE=1 --enable-sleep-mode")
        if isinstance(sleep_state, Mapping) and sleep_state.get("is_sleeping") is True:
            raise BackendError("vLLM server is asleep; wake it before training")
        pause_state = self._optional_get("is_paused")
        if isinstance(pause_state, Mapping) and pause_state.get("is_paused") is True:
            raise BackendError("vLLM server is paused; resume it before training")
        if self.expected_base_model is not None:
            if self.base_model is None:
                raise BackendResponseError("vLLM /v1/models has no base model entry")
            candidates = {self.base_model, Path(self.base_model).name}
            expected = {self.expected_base_model, Path(self.expected_base_model).name}
            if candidates.isdisjoint(expected):
                raise BackendResponseError(
                    f"served base model {self.base_model!r} does not match "
                    f"expected {self.expected_base_model!r}")
        return {
            "lora_enabled": runtime_lora is True or not self.require_runtime_lora,
            "max_lora_rank": max_rank,
            "max_loras": max_loras,
            "max_cpu_loras": max_cpu_loras,
            "sleep_enabled": sleep_enabled is True,
            "sleep_level": self.sleep_level,
            "prefix_cache_server": prefix_cache,
            "prefix_cache_requested": self.enable_prefix_cache,
            "base_model": self.base_model,
        }

    def start(self) -> Mapping[str, Any]:
        with self._state_lock:
            if self._started:
                return {"status": "ok", "engine": self.name, "already_started": True,
                        "deployment": dict(self.deployment)}
            health = self.health()
            self.base_model = self._base_model_id(self._models())
            if self.base_model is None:
                raise BackendResponseError("vLLM /v1/models has no base model entry")
            if self._load_counts() is None:
                raise BackendCapabilityError(
                    "vLLM /metrics has no vllm:num_requests_running/waiting values; "
                    "the training lease cannot prove a drained GPU")
            version = self._optional_get("version")
            info = self._optional_get("server_info", "?config_format=json")
            if info is not None and not isinstance(info, Mapping):
                raise BackendResponseError("vLLM /server_info must return an object")
            self.server_info = info
            deployment = self._deployment_summary(info)
            self.deployment = deployment
            self._started = True
            return {
                "status": "ok",
                "engine": self.name,
                "version": version,
                "health": dict(health),
                "deployment": deployment,
            }

    def health(self) -> Mapping[str, Any]:
        result = self._request("GET", self.endpoints["health"], timeout=self.timeout)
        return {"status": "ok"} if result is None else {"status": "ok", "body": result}

    # ------------------------------------------------------------------
    # Dynamic LoRA
    # ------------------------------------------------------------------
    def lora_name(self, manifest: AdapterManifest) -> str:
        if not isinstance(manifest, AdapterManifest):
            raise TypeError("lora_name requires an AdapterManifest")
        name = f"{self.lora_name_prefix}-v{manifest.version:08d}-{manifest.adapter_sha256[:16]}"
        if not _LORA_NAME.fullmatch(name):
            raise ValueError("derived vLLM LoRA name is invalid")
        return name

    @staticmethod
    def _default_adapter_check(directory: Path) -> dict[str, Any]:
        config_path = directory / "adapter_config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"adapter_config.json is unreadable: {exc}") from exc
        if not isinstance(config, dict) or config.get("peft_type") != "LORA":
            raise ValueError("adapter_config.json is not a PEFT LORA adapter")
        if config.get("r") != 128 or config.get("lora_alpha") != 256:
            raise ValueError("vLLM adapter must be rank 128 / alpha 256")
        try:
            dropout = float(config.get("lora_dropout", -1))
        except (TypeError, ValueError):
            dropout = -1.0
        if dropout != 0.0:
            raise ValueError("vLLM adapter must use dropout 0")
        if config.get("use_dora") or config.get("use_rslora") or config.get("modules_to_save"):
            raise ValueError("vLLM adapter uses an unsupported PEFT variant")
        if config.get("bias") not in (None, "none"):
            raise ValueError("vLLM adapter must not train biases")
        return {"compatible": True, "r": 128, "lora_alpha": 256}

    def _audit_adapter(self, manifest: AdapterManifest, directory: Path) -> dict[str, Any]:
        if self.adapter_validator is not None:
            report = self.adapter_validator(manifest, directory)
            if not isinstance(report, Mapping):
                raise ValueError("adapter_validator must return a mapping")
            if report.get("compatible") is False:
                reasons = report.get("reasons") or ["adapter_target_audit_failed"]
                raise ValueError("vLLM adapter target audit failed: " + "; ".join(map(str, reasons)))
            return dict(report)
        return self._default_adapter_check(directory)

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
        audit = self._audit_adapter(manifest, directory)
        digest = manifest.adapter_sha256
        name = self.lora_name(manifest)
        with self._state_lock:
            existing = self._adapters.get(digest)
            if existing is not None:
                if existing[0] != manifest or existing[1] != str(directory):
                    raise ValueError("adapter SHA256 is already bound to another directory")
            collisions = [other for other, (_, _) in self._adapters.items() if other != digest
                          and self.lora_name(self._adapters[other][0]) == name]
            if collisions:
                raise BackendError("vLLM LoRA name collision")
            if len(self._adapters) >= self.max_resident_adapters:
                raise BackendError(
                    f"refusing a third resident adapter: max_resident_adapters="
                    f"{self.max_resident_adapters}")
        # A process-only crash can leave the same name registered on a live
        # server.  The local SHA/path checks above are the authority; adopt
        # that registration instead of letting vLLM reject a duplicate name.
        if self._model_present(name):
            with self._state_lock:
                self._adapters[digest] = (manifest, str(directory))
                self._active_adapter_hash = digest
            return {"loaded": True, "already_loaded": True, "server_resident": True,
                    "adapter_sha256": digest, "lora_name": name, "audit": audit}
        raw = self._request("POST", self.endpoints["load_adapter"], {
            "lora_name": name,
            "lora_path": str(directory),
        }, timeout=self.control_timeout)
        if isinstance(raw, Mapping) and raw.get("error"):
            raise BackendResponseError(f"vLLM refused the adapter: {raw.get('error')}")
        if not self._model_present(name):
            raise BackendResponseError(
                "vLLM load_lora_adapter returned HTTP 200 but /v1/models does not "
                "list the requested adapter")
        with self._state_lock:
            self._adapters[digest] = (manifest, str(directory))
            self._active_adapter_hash = digest
        return {"loaded": True, "adapter_sha256": digest, "lora_name": name,
                "audit": audit}

    def unload_adapter(self, adapter: AdapterIdentity) -> Mapping[str, Any]:
        self._require_started()
        if self._memory_released:
            raise BackendError("adapter memory cannot be changed while engine memory is released")
        digest = _adapter_hash(adapter)
        with self._state_lock:
            item = self._adapters.get(digest)
            if item is None:
                return {"unloaded": True, "already_absent": True, "adapter_sha256": digest}
            manifest = item[0]
            name = self.lora_name(manifest)
        if not self._model_present(name):
            with self._state_lock:
                self._adapters.pop(digest, None)
                if self._active_adapter_hash == digest:
                    self._active_adapter_hash = None
            return {"unloaded": True, "already_absent": True,
                    "adapter_sha256": digest, "lora_name": name}
        raw = self._request("POST", self.endpoints["unload_adapter"], {
            "lora_name": name,
        }, timeout=self.control_timeout)
        if isinstance(raw, Mapping) and raw.get("error"):
            raise BackendResponseError(f"vLLM refused the adapter unload: {raw.get('error')}")
        if self._model_present(name):
            raise BackendResponseError(
                "vLLM unload_lora_adapter returned HTTP 200 but /v1/models still "
                "lists the adapter")
        with self._state_lock:
            self._adapters.pop(digest, None)
            if self._active_adapter_hash == digest:
                self._active_adapter_hash = None
        return {"unloaded": True, "adapter_sha256": digest, "lora_name": name}

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @staticmethod
    def _native_sampling(sampling: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(sampling, Mapping):
            raise BackendResponseError("sampling must be a mapping")
        try:
            temperature = float(sampling["temperature"])
            top_p = float(sampling["top_p"])
            top_k = int(sampling["top_k"])
            max_new_tokens = int(sampling["max_new_tokens"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendResponseError(
                "vLLM sampling requires temperature/top_p/top_k/max_new_tokens") from exc
        if not math.isfinite(temperature) or temperature < 0:
            raise BackendResponseError("vLLM temperature must be finite and non-negative")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise BackendResponseError("vLLM top_p must be in (0, 1]")
        if top_k == 0:
            native_top_k = -1
        elif top_k == -1 or top_k >= 1:
            native_top_k = top_k
        else:
            raise BackendResponseError("vLLM top_k must be 0 (unfiltered), -1 or >= 1")
        if max_new_tokens <= 0:
            raise BackendResponseError("vLLM max_new_tokens must be positive")
        record = dict(sampling)
        native = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": native_top_k,
            "max_tokens": max_new_tokens,
        }
        seed = sampling.get("seed")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise BackendResponseError("vLLM seed must be an integer")
            if seed != -1:
                native["seed"] = seed
            record["seed"] = seed
        return native, record

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
        if request_id is not None and (not isinstance(request_id, str) or not request_id):
            raise ValueError("request_id must be a non-empty string")
        rid = request_id or uuid.uuid4().hex
        if cache_namespace is not None and cache_namespace != digest:
            raise ValueError("cache_namespace must equal the behavior adapter SHA256")
        native_sampling, sampling_record = self._native_sampling(sampling or {})
        with self._state_lock:
            item = self._adapters.get(digest)
            if item is None:
                raise BackendError(
                    "adapter is not loaded on the vLLM server; call "
                    "load_adapter(manifest, path) before submitting")
            lora_name = self.lora_name(item[0])
        payload = {
            "model": lora_name,
            "prompt": prompt,
            "max_tokens": native_sampling["max_tokens"],
            "temperature": native_sampling["temperature"],
            "top_p": native_sampling["top_p"],
            "top_k": native_sampling["top_k"],
            "logprobs": 1,
            "return_token_ids": True,
            "add_special_tokens": False,
            "echo": False,
            "n": 1,
            "stream": False,
            "request_id": rid,
        }
        if self.enable_prefix_cache:
            # A distinct salt keeps the prefix-cache key scoped to this adapter
            # even when the operator leaves server-side prefix caching enabled.
            payload["cache_salt"] = digest
        if "seed" in native_sampling:
            payload["seed"] = native_sampling["seed"]
        raw = self._object(
            self._request("POST", self.endpoints["generate"], payload, timeout=timeout),
            "submit")
        if raw.get("object") != "text_completion":
            raise BackendResponseError("vLLM response object is not text_completion")
        returned_id = raw.get("id")
        if returned_id not in (rid, f"cmpl-{rid}"):
            raise BackendResponseError("vLLM response id does not bind the submitted request_id")
        if raw.get("model") != lora_name:
            raise BackendResponseError(
                "vLLM response model does not equal the requested LoRA name")
        choices = raw.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
            raise BackendResponseError("vLLM must return exactly one completion choice")
        choice = choices[0]
        returned_prompt = _token_ids(choice.get("prompt_token_ids"), "response prompt_token_ids")
        if returned_prompt != prompt:
            raise BackendResponseError("vLLM prompt_token_ids do not exactly match the submitted IDs")
        output_ids = _token_ids(choice.get("token_ids"), "response token_ids", allow_empty=True)
        logprobs_obj = choice.get("logprobs")
        if not isinstance(logprobs_obj, Mapping):
            raise BackendResponseError("vLLM response has no completion logprobs object")
        logprobs = logprobs_obj.get("token_logprobs")
        if not isinstance(logprobs, list) or len(logprobs) != len(output_ids):
            raise BackendResponseError("vLLM token_logprobs are not one-to-one with token_ids")
        if any(isinstance(item, bool) or not isinstance(item, (int, float))
               or not math.isfinite(float(item)) for item in logprobs):
            raise BackendResponseError("every sampled vLLM logprob must be a finite number")
        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise BackendResponseError("vLLM finish_reason is required")
        return {
            "request_id": rid,
            "input_ids": list(prompt),
            "input_ids_server_echo": True,
            "output_ids": output_ids,
            "logprobs": [float(item) for item in logprobs],
            "finish_reason": finish_reason,
            "adapter_sha256": digest,
            "sampling": sampling_record,
            "sampling_server_echo": False,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def cancel(self, request_id: str) -> Mapping[str, Any]:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        self._require_started()
        # vLLM's completion response id is "cmpl-<request_id>"; the engine
        # tracks that prefixed id.  Send both spellings so a client-side id
        # can never silently cancel nothing.
        ids = [request_id]
        if not request_id.startswith("cmpl-"):
            ids.append(f"cmpl-{request_id}")
        result = self._control("cancel", {"request_ids": ids}, "cancel")
        status = result.get("status")
        if status is not None and status != "aborted":
            raise BackendResponseError("vLLM abort_requests did not confirm status aborted")
        return {"cancelled": True, "request_id": request_id,
                "aborted": result.get("aborted")}

    def _control(self, path_key: str, payload: Mapping[str, Any] | None,
                 operation: str, suffix: str = "") -> dict[str, Any]:
        result = self._request("POST", self.endpoints[path_key] + suffix, payload,
                               timeout=self.control_timeout)
        if result is None:
            return {}
        if isinstance(result, Mapping):
            return dict(result)
        if isinstance(result, str):
            return {"body": result}
        raise BackendResponseError(f"{operation} returned an unsupported response")

    def _course_state(self, key: str) -> bool | None:
        value = self._optional_get(key)
        if not isinstance(value, Mapping):
            return None
        field = "is_paused" if key == "is_paused" else "is_sleeping"
        state = value.get(field)
        return state if type(state) is bool else None

    def _wait_state(self, key: str, expected: bool, limit: float | None = None) -> bool:
        deadline = time.monotonic() + (self.control_timeout if limit is None else limit)
        while True:
            state = self._course_state(key)
            if state is expected:
                return True
            if state is None:
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    def pause(self) -> Mapping[str, Any]:
        self._require_inference()
        result = self._control("pause", None, "pause", "?mode=wait")
        body = result
        status = body.get("status")
        if status is not None and status != "paused":
            raise BackendResponseError("vLLM pause did not confirm status paused")
        if status != "paused" and not self._wait_state("is_paused", True):
            raise BackendResponseError(
                "vLLM pause returned no confirmation and /is_paused did not become true")
        with self._state_lock:
            self._paused = True
            self._paused_confirmed = True
        return {"paused": True}

    def drain(self, timeout: float | None = None) -> Mapping[str, Any]:
        self._require_started()
        if not self._paused:
            self.pause()
        limit = self.control_timeout if timeout is None else _positive_number(timeout, "timeout")
        deadline = time.monotonic() + limit
        while True:
            last = self._load_counts()
            if last is None:
                raise BackendResponseError(
                    "vLLM /metrics has no vllm:num_requests_running/waiting values; "
                    "refusing to declare a drained GPU")
            if last["running"] <= 0 and last["waiting"] <= 0:
                return {"drained": True, "paused": True, "load": last}
            if time.monotonic() >= deadline:
                raise BackendError(f"vLLM drain timed out with load: {last}")
            time.sleep(0.25)

    def resume(self) -> Mapping[str, Any]:
        self._require_started()
        if self._memory_released:
            raise BackendError("resume_memory() is required before resuming generation")
        body = self._control("resume", None, "resume")
        status = body.get("status")
        if status is not None and status != "resumed":
            raise BackendResponseError("vLLM resume did not confirm status resumed")
        if status != "resumed" and not self._wait_state("is_paused", False):
            raise BackendResponseError(
                "vLLM resume returned no confirmation and /is_paused did not become false")
        with self._state_lock:
            self._paused = False
            self._paused_confirmed = False
        return {"resumed": True}

    def release_memory(self) -> Mapping[str, Any]:
        self._require_started()
        if not self._paused:
            self.pause()
        if not self._paused:
            raise BackendError("vLLM must be paused before sleep")
        # LoRA weights do not survive Level-2 sleep reliably.  The GpuLease
        # snapshots manifests before this call and reloads them after wake.
        for digest in list(self._adapters):
            self.unload_adapter(digest)
        body = self._control(
            "release_memory", None, "release_memory",
            f"?level={self.sleep_level}&mode=abort")
        status = body.get("status")
        if status is not None and (status != "sleeping"
                                   or body.get("level") != self.sleep_level):
            raise BackendResponseError("vLLM sleep did not confirm the requested level")
        if status != "sleeping" and not self._wait_state("is_sleeping", True):
            raise BackendResponseError(
                "vLLM sleep returned no confirmation and /is_sleeping did not become true")
        with self._state_lock:
            self._memory_released = True
            self._sleeping_confirmed = True
        return {"released": True, "sleep_level": self.sleep_level}

    def _load_counts(self) -> dict[str, int] | None:
        try:
            text = self._request("GET", self.endpoints["metrics"], timeout=self.timeout)
        except BackendError:
            return None
        if not isinstance(text, str):
            return None
        running = waiting = 0
        seen_running = seen_waiting = False
        for line in text.splitlines():
            match = _PROMETHEUS_RUNNING.match(line.strip())
            if match:
                running += int(float(match.group(1)))
                seen_running = True
                continue
            match = _PROMETHEUS_WAITING.match(line.strip())
            if match:
                waiting += int(float(match.group(1)))
                seen_waiting = True
        if not seen_running or not seen_waiting:
            return None
        return {"running": running, "waiting": waiting}

    def metrics(self) -> Any:
        self._require_started()
        text = self._request("GET", self.endpoints["metrics"], timeout=self.timeout)
        if not isinstance(text, str) or not text.strip():
            raise BackendResponseError("vLLM metrics returned an empty or non-text response")
        counts = self._load_counts()
        return {
            "running": counts["running"] if counts else None,
            "waiting": counts["waiting"] if counts else None,
            "metrics_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "metrics_bytes": len(text.encode()),
        }

    def resume_memory(self) -> Mapping[str, Any]:
        self._require_started()
        if not self._memory_released:
            return {"resumed": True, "already_awake": True}
        if self.sleep_level == 2:
            weights = self._control(
                "resume_memory", None, "wake_weights", "?tags=weights")
            status = weights.get("status")
            if status is not None and status not in {"sleeping", "awake"}:
                raise BackendResponseError("vLLM wake_up?tags=weights did not confirm")
            reload_result = self._control(
                "collective_rpc", {"method": "reload_weights"}, "reload_weights")
            if reload_result.get("error"):
                raise BackendResponseError(
                    f"vLLM reload_weights failed: {reload_result.get('error')}")
            kv = self._control(
                "resume_memory", None, "wake_kv_cache", "?tags=kv_cache")
            status = kv.get("status")
            if status is not None and status != "awake":
                raise BackendResponseError("vLLM wake_up?tags=kv_cache did not confirm awake")
            try:
                self._request("POST", self.endpoints["reset_prefix_cache"], None,
                              timeout=self.control_timeout)
            except BackendCapabilityError:
                if self.enable_prefix_cache:
                    raise BackendResponseError(
                        "vLLM prefix cache is enabled but /reset_prefix_cache is unavailable")
            if not self._wait_state("is_sleeping", False):
                raise BackendResponseError("vLLM did not report is_sleeping=false after wake_up")
        else:
            result = self._control("resume_memory", None, "wake_up")
            status = result.get("status")
            if status is not None and status not in {"awake", "sleeping"}:
                raise BackendResponseError("vLLM wake_up did not confirm")
            if status is None and not self._wait_state("is_sleeping", False):
                raise BackendResponseError("vLLM wake_up did not confirm")
        with self._state_lock:
            self._memory_released = False
            self._sleeping_confirmed = False
        return {"resumed": True, "sleep_level": self.sleep_level}

    release_memory_occupation = release_memory
    resume_memory_occupation = resume_memory


# Aliases preserve the existing import spellings while pointing at the one
# independent implementation.
VllmBackend = VLLMBackend
VLLMHTTPBackend = VLLMBackend
VllmHTTPBackend = VLLMBackend
