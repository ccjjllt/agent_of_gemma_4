"""Complete one trajectory from ACKed calls through official reward and SPO buffer."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Mapping

from ..request_store import RequestConflict, RequestUnavailable
from ..reward.engine import RewardEngine
from ..spo.value_tracker import ValueTracker
from ..trajectory.buffer import TrajectoryBuffer
from ..trajectory.ledger import TrajectoryLedger, canonical_json
from ..trajectory.schema import RewardRecord, Trajectory
from .request_manager import RequestManager


OBJECTIVE_VERSION = "official_differential_spo_v1"


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _validity(raw: Mapping[str, Any], candidate: Mapping[str, Any]) -> Any:
    if "validity" in candidate:
        return candidate["validity"]
    if "validity" in raw:
        return raw["validity"]
    if raw.get("verdict_valid") is True and raw.get("verification_complete") is True:
        # A terminal, observed patch rejection is valid verification but invalid
        # model output.  Other complete official verdicts establish protocol and
        # patch-processing validity without deriving it from reward eligibility.
        return 0.0 if raw.get("verdict_kind") == "agent_patch_rejected" else 1.0
    return None


class RolloutEngine:
    """Single-trajectory postprocessor; no same-task grouping or synchronization.

    ``run`` is safe to invoke again after a crash.  Existing rewards and value
    observations are replayed idempotently, and an ineligible record never calls
    ``ValueTracker.observe`` or ``TrajectoryBuffer.offer``.
    """

    def __init__(
        self,
        request_manager: RequestManager,
        reward_engine: RewardEngine,
        ledger: TrajectoryLedger,
        value_tracker: ValueTracker,
        buffer: TrajectoryBuffer,
        *,
        worker: Any = None,
    ):
        for name, value, expected in (
            ("request_manager", request_manager, RequestManager),
            ("reward_engine", reward_engine, RewardEngine),
            ("ledger", ledger, TrajectoryLedger),
            ("value_tracker", value_tracker, ValueTracker),
            ("buffer", buffer, TrajectoryBuffer),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if ledger is not value_tracker.ledger or ledger is not buffer.ledger:
            raise ValueError("ledger, value tracker and buffer must share one transaction store")
        self.request_manager = request_manager
        self.reward_engine = reward_engine
        self.ledger = ledger
        self.value_tracker = value_tracker
        self.buffer = buffer
        self.worker = worker

    def _existing(self, trajectory_id: str) -> tuple[Trajectory | None, RewardRecord | None]:
        try:
            trajectory = self.ledger.get_trajectory(trajectory_id)
        except KeyError:
            return None, None
        try:
            reward = self.ledger.get_reward(trajectory_id)
        except KeyError:
            reward = None
        return trajectory, reward

    @staticmethod
    def _worker_parts(outcome: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(outcome, Mapping) and isinstance(outcome.get("result"), Mapping):
            raw = dict(outcome["result"])
            verification = _mapping(outcome.get("verification"))
            for key in (
                "candidate_report", "baseline_report", "resource_metrics", "tool_events",
                "edits", "test_events", "candidate_summary_ref", "baseline_summary_ref",
                "artifact_hashes", "result_path", "official_results_dir",
            ):
                if key in outcome and key not in verification:
                    verification[key] = outcome[key]
            return raw, verification
        raw_value = _field(outcome, "result")
        raw = dict(raw_value) if isinstance(raw_value, Mapping) else _mapping(outcome)
        verification = _mapping(_field(outcome, "verification"))
        for key in (
            "candidate_report", "baseline_report", "resource_metrics", "tool_events",
            "edits", "test_events", "candidate_summary_ref", "baseline_summary_ref",
            "artifact_hashes", "result_path", "official_results_dir",
        ):
            value = _field(outcome, key)
            if value is not None and key not in verification:
                verification[key] = value
        return raw, verification

    @staticmethod
    def _verification_ref(trajectory_id: str, raw: Mapping[str, Any], extra: Mapping[str, Any]) -> str:
        for value in (
            extra.get("candidate_summary_ref"), extra.get("result_path"),
            raw.get("result_path"), extra.get("official_results_dir"), raw.get("official_results_dir"),
        ):
            if isinstance(value, (str, Path)) and str(value):
                return str(value)
        return f"official-verdict:{trajectory_id}"

    def _bundle(
        self,
        trajectory: Trajectory,
        raw: Mapping[str, Any],
        extra: Mapping[str, Any],
        verification_ref: str,
        seal_error: str | None,
    ) -> dict[str, Any]:
        candidate = _mapping(extra.get("candidate_report"))
        features = _mapping(raw.get("reward_features"))
        # Raw official lifecycle fields are authoritative even if an adapter also
        # supplies a candidate report.  Per-node data is never inferred from counts.
        for key in (
            "expected_universes", "node_statuses", "nodes", "artifact_hashes",
            "junit_available", "verification_complete", "verdict_valid",
            "infrastructure_error", "validity",
        ):
            if key in extra and key not in candidate:
                candidate[key] = extra[key]
        for key in ("verification_complete", "verdict_valid", "infrastructure_error"):
            if key in raw:
                candidate[key] = raw[key]
        if "junit_available" not in candidate and "junit_available" in features:
            candidate["junit_available"] = features["junit_available"]
        if type(raw.get("success")) is bool:
            candidate["resolved"] = raw["success"]
        validity = _validity(raw, candidate)
        if validity is not None:
            candidate["validity"] = validity
        artifacts = _mapping(candidate.get("artifact_hashes"))
        artifacts.update(_mapping(extra.get("artifact_hashes")))
        artifacts.update(_mapping(raw.get("artifact_hashes")))
        if isinstance(features.get("junit_sha256"), str):
            artifacts.setdefault("candidate_junit", features["junit_sha256"])
        patch_hash = raw.get("patch_sha256", features.get("patch_sha256"))
        if isinstance(patch_hash, str):
            artifacts.setdefault("candidate_patch", patch_hash)
        if artifacts:
            candidate["artifact_hashes"] = artifacts
        if seal_error is not None:
            candidate.update({
                "verification_complete": False,
                "verdict_valid": False,
                "infrastructure_error": seal_error,
            })
        metrics = _mapping(raw.get("resource_metrics"))
        metrics.update(_mapping(extra.get("resource_metrics")))
        if isinstance(raw.get("tool_calls"), int) and not isinstance(raw.get("tool_calls"), bool):
            metrics.setdefault("tool_calls", raw["tool_calls"])
        return {
            "schema_version": "verification_bundle_v1",
            "task_key": trajectory.task_key.to_dict(),
            "candidate_report": candidate,
            "candidate_verdict": raw.get("success") if type(raw.get("success")) is bool else None,
            "candidate_summary_ref": extra.get("candidate_summary_ref", verification_ref),
            "baseline_report": extra.get("baseline_report", extra.get("baseline")),
            "baseline_summary_ref": extra.get("baseline_summary_ref"),
            "resource_metrics": metrics,
            "calls": [call.to_dict() for call in trajectory.calls],
            "tool_events": extra.get("tool_events"),
            "edits": extra.get("edits"),
            "test_events": extra.get("test_events"),
            "patch_sha256": patch_hash,
            "artifact_hashes": artifacts,
        }

    def _ineligible(self, reason: str, resolved: bool | None) -> RewardRecord:
        return RewardRecord(
            resolved=resolved, P=None, G=None, progress=None, regression=None,
            validity=None, base_reward=None, efficiency=None,
            efficiency_weight=float(getattr(self.reward_engine, "efficiency_weight", 0.05)),
            total_reward=None, eligible=False, exclusion_reason=reason,
            objective_version=OBJECTIVE_VERSION,
        )

    def _record_reward(self, trajectory: Trajectory, reward: RewardRecord) -> None:
        self.ledger.record_reward(trajectory.trajectory_id, reward)

    def _ensure_excluded(self, trajectory_id: str, reason: str) -> None:
        with self.ledger.transaction() as conn:
            trajectory = self.ledger._get_trajectory(conn, trajectory_id)
            if trajectory.status != "EXCLUDED":
                self.ledger._put_trajectory(conn, replace(trajectory, status="EXCLUDED"))
            self.ledger._event(conn, "trajectory_excluded", trajectory_id, {"reason": reason})

    def run(self, assignment: Any, worker_result: Any = None) -> Trajectory:
        trajectory_id = assignment.trajectory_id
        existing, stored_reward = self._existing(trajectory_id)
        if existing is not None and stored_reward is not None and existing.status in {
            "BUFFERED", "EXCLUDED", "RESERVED", "CONSUMED", "STALE_EXCLUDED", "NOOP_CONSUMED",
        }:
            return existing
        if worker_result is None:
            if self.worker is None:
                if existing is None:
                    raise ValueError("run requires an official worker result")
            else:
                worker_result = self.worker.run(assignment) if hasattr(self.worker, "run") else self.worker(assignment)
        raw, extra = self._worker_parts(worker_result if worker_result is not None else {})
        verification_ref = self._verification_ref(trajectory_id, raw, extra)
        trajectory = existing
        seal_error: str | None = None
        if trajectory is None:
            try:
                trajectory = self.request_manager.seal_trajectory(
                    trajectory_id,
                    assignment.task_key,
                    assignment.stream_sequence,
                    assignment.behavior_adapter,
                    status="VERIFYING",
                    verification_ref=verification_ref,
                    provenance=dict(assignment.provenance),
                )
            except (RequestUnavailable, RequestConflict, KeyError, ValueError, TypeError) as error:
                seal_error = f"trajectory_seal_{type(error).__name__}: {error}"
                trajectory = self.request_manager.excluded_trajectory(
                    trajectory_id,
                    assignment.task_key,
                    assignment.stream_sequence,
                    assignment.behavior_adapter,
                    reason=seal_error,
                    verification_ref=verification_ref,
                    provenance=dict(assignment.provenance),
                )
        if trajectory.behavior_adapter != assignment.behavior_adapter:
            raise ValueError("trajectory changed behavior adapter during execution")

        reward = stored_reward
        if reward is None:
            bundle = self._bundle(trajectory, raw, extra, verification_ref, seal_error)
            try:
                record = self.reward_engine.spo_record(
                    bundle,
                    task_key=trajectory.task_key.to_dict(),
                    metrics=bundle["resource_metrics"],
                )
                reward = RewardRecord.from_dict(record)
                if seal_error is not None and reward.eligible:
                    reward = replace(
                        reward, total_reward=None, eligible=False, exclusion_reason=seal_error,
                    )
            except (ValueError, TypeError, KeyError, OSError) as error:
                reward = self._ineligible(
                    f"verification_reward_{type(error).__name__}: {error}",
                    raw.get("success") if type(raw.get("success")) is bool else None,
                )
            if existing is None:
                trajectory = replace(
                    trajectory,
                    status="REWARDED" if reward.eligible else "EXCLUDED",
                    verification_ref=verification_ref,
                    reward_ref=f"reward:{trajectory_id}",
                )
                self.ledger.record_trajectory(trajectory)
            self._record_reward(trajectory, reward)
        if not reward.eligible:
            self._ensure_excluded(trajectory_id, reward.exclusion_reason or "ineligible")
            return self.ledger.get_trajectory(trajectory_id)

        # observe persists A=R-V_pre before any new value is visible to the buffer.
        observation = self.value_tracker.observe(trajectory, reward)
        if not math.isfinite(observation.raw_advantage):
            raise RuntimeError("ValueTracker produced a non-finite raw advantage")
        self.buffer.offer(trajectory_id)
        return self.ledger.get_trajectory(trajectory_id)

    def exclude(self, assignment: Any, reason: str) -> Trajectory:
        """Record an infrastructure failure without manufacturing a zero reward."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("exclusion requires an explicit reason")
        return self.run(assignment, {
            "task_id": assignment.task_key.task_id,
            "rollout_id": assignment.trajectory_id,
            "success": None,
            "verification_complete": False,
            "verdict_valid": False,
            "infrastructure_error": reason,
        })
