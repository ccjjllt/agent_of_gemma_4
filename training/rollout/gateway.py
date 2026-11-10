"""Inference gateway: sampling evidence and deployment contract enforcement.

The engine speaks raw token IDs; training needs complete, auditable evidence
per call: token IDs, behavior logprobs, adapter hash, finish reason and the
full sampling configuration (temperature/top_p/top_k/seed).  The gateway owns
that boundary:

- every submitted generation returns its sampling record (server echo when
  the engine reflects it, otherwise the submitted copy marked as such);
- incomplete sampling is rejected before any GPU work, never fabricated;
- deployment probing is read-only (health/metrics); state-changing custom
  endpoints (drain/cancel/release/resume/load/unload) are contract-required
  and verified at first real use, not by probe side effects.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .backend import BackendError, BackendResponseError, InferenceBackend


REQUIRED_SAMPLING_KEYS = ("temperature", "top_p", "top_k")
ENGINE_OWNED_SEED = -1
CUSTOM_ENDPOINTS = (
    "drain", "cancel", "release_memory", "resume_memory",
    "load_adapter", "unload_adapter",
)
STANDARD_ENDPOINTS = ("health", "generate", "metrics")


def validate_sampling(sampling: Mapping[str, Any] | None) -> dict[str, Any]:
    """Require a complete sampling record; seed defaults to engine-owned."""
    if not isinstance(sampling, Mapping):
        raise ValueError("sampling must be a mapping with temperature/top_p/top_k")
    record = dict(sampling)
    for key in REQUIRED_SAMPLING_KEYS:
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            raise ValueError(f"sampling[{key}] must be a finite number")
    if record.get("top_p") is not None and not 0 < float(record["top_p"]) <= 1:
        raise ValueError("sampling[top_p] must be in (0, 1]")
    seed = record.get("seed", ENGINE_OWNED_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("sampling[seed] must be an integer")
    record["seed"] = seed
    return record


class InferenceGateway:
    """Own the backend sampling contract for one inference deployment."""

    def __init__(self, backend: InferenceBackend):
        if not isinstance(backend, InferenceBackend):
            raise TypeError("backend must be an InferenceBackend")
        self.backend = backend

    def submit(self, input_ids: list[int], *, adapter: Any,
               sampling: Mapping[str, Any] | None,
               request_id: str | None = None,
               cache_namespace: str | None = None,
               timeout: float | None = None) -> dict[str, Any]:
        """Submit one generation and return complete sampling evidence.

        Raises BackendResponseError if the backend drops the sampling record.
        """
        record = validate_sampling(sampling)
        result = self.backend.submit(
            input_ids, adapter=adapter, request_id=request_id,
            sampling=record, cache_namespace=cache_namespace, timeout=timeout)
        returned = result.get("sampling")
        if not isinstance(returned, Mapping):
            raise BackendResponseError("backend dropped the sampling record")
        for key in (*REQUIRED_SAMPLING_KEYS, "seed"):
            if returned.get(key) != record[key]:
                raise BackendResponseError(
                    f"backend sampling[{key}] does not match the submitted configuration")
        return dict(result, sampling=dict(returned))

    def submit_and_ack(self, requests: Any, trajectory_id: str,
                         prompt_ids: list[int], adapter: Any,
                         sampling: Mapping[str, Any] | None, *,
                         request_id: str | None = None,
                         action_mask: list[int] | None = None,
                         raw_tool_call_ids: list[str] | None = None,
                         parser_valid: bool = True,
                         timeout: float | None = None) -> Any:
        """Full chain: submit -> commit -> ACK -> ActionCall.

        Backend token keys (input_ids/output_ids/logprobs) map to the TITO
        record; the sampling configuration travels untouched into the sealed
        call, satisfying RequestManager.to_action_call end to end.
        """
        backend_adapter = adapter.sha256 if hasattr(adapter, "sha256") else adapter
        result = self.submit(prompt_ids, adapter=backend_adapter, sampling=sampling,
                             request_id=request_id, timeout=timeout)
        payload = dict(result)
        payload["action_mask"] = list(action_mask) if action_mask is not None else [1] * len(
            result["output_ids"])
        payload["raw_tool_call_ids"] = list(raw_tool_call_ids or [])
        payload["parser_valid"] = parser_valid
        stored, _ = requests.create_call(
            trajectory_id, adapter, idempotency_key=result["request_id"],
            request={"prompt_ids": list(prompt_ids)},
            metadata={"gateway": "inference_gateway_v1"})
        requests.commit_generated(
            stored["request_id"], payload, trajectory_id=trajectory_id, behavior=adapter)
        return requests.acknowledge(stored["request_id"], trajectory_id, behavior=adapter)

    def probe_deployment(self) -> dict[str, Any]:
        """Read-only deployment report: health plus the endpoint contract.

        Only side-effect-free endpoints are called.  Custom state-changing
        endpoints are listed as contract-required; each is verified with an
        explicit server confirmation at first real use.
        """
        try:
            health = self.backend.health()
        except (BackendError, ValueError, TypeError, OSError) as error:
            return {"reachable": False, "error": f"{type(error).__name__}: {error}",
                    "engine": getattr(self.backend, "name", None)}
        try:
            metrics = self.backend.metrics()
            metrics_ok = True
        except (BackendError, ValueError, TypeError, OSError):
            metrics, metrics_ok = None, False
        return {
            "reachable": True,
            "engine": getattr(self.backend, "name", None),
            "health": health,
            "metrics_available": metrics_ok,
            "metrics": metrics,
            "standard_endpoints": list(STANDARD_ENDPOINTS),
            "custom_endpoints_contract_required": list(CUSTOM_ENDPOINTS),
        }
