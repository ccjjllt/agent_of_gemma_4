"""Persistent KL-adaptive value and independent success-probability tracker."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, replace
from typing import Any

from ..trajectory.ledger import TrajectoryLedger, canonical_json, task_key_id
from ..trajectory.schema import RewardRecord, TaskKey, Trajectory


@dataclass(frozen=True)
class ValueRecord:
    task_key: TaskKey
    V: float
    success_p: float
    effective_count: float
    last_observation_sequence: int | None
    last_policy_version: int | None
    kl_cursor: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key.to_dict(), "V": self.V,
            "success_p": self.success_p, "effective_count": self.effective_count,
            "last_observation_sequence": self.last_observation_sequence,
            "last_policy_version": self.last_policy_version, "kl_cursor": self.kl_cursor,
        }


@dataclass(frozen=True)
class ValueObservation:
    trajectory_id: str
    task_key: TaskKey
    V_pre: float
    raw_advantage: float
    V_after: float
    success_p_after: float
    effective_count_after: float
    eta: float
    forgetting_factor: float
    forgotten_kl: float
    kl_cursor: int
    idempotent_replay: bool = False


class ValueTracker:
    """Observe each independent trajectory exactly once.

    ``observe`` is the only method that applies a reward.  ``snapshot`` performs
    lazy KL forgetting only, and is therefore safe for sampler use.  Both advance
    the persisted KL cursor so recovery cannot apply a segment twice.
    """

    def __init__(self, ledger: TrajectoryLedger, *, prior_value: float = 0.5,
                 prior_success: float = 0.5, prior_strength: float = 1.0,
                 kl_half_life: float = 0.05):
        self.ledger = ledger
        self.prior_value = self._unit(prior_value, "prior_value")
        self.prior_success = self._unit(prior_success, "prior_success")
        if isinstance(prior_strength, bool) or not isinstance(prior_strength, (int, float)) \
                or not math.isfinite(float(prior_strength)) or prior_strength < 0:
            raise ValueError("prior_strength must be finite and non-negative")
        if isinstance(kl_half_life, bool) or not isinstance(kl_half_life, (int, float)) \
                or not math.isfinite(float(kl_half_life)) or kl_half_life <= 0:
            raise ValueError("kl_half_life must be finite and positive")
        self.prior_strength = float(prior_strength)
        self.kl_half_life = float(kl_half_life)

    @staticmethod
    def _unit(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0,1]")
        return float(value)

    @staticmethod
    def _record(row: Any) -> ValueRecord:
        return ValueRecord(
            task_key=TaskKey.from_dict(json.loads(row["task_payload"])),
            V=float(row["V"]), success_p=float(row["success_p"]),
            effective_count=float(row["effective_count"]),
            last_observation_sequence=row["last_observation_sequence"],
            last_policy_version=row["last_policy_version"], kl_cursor=int(row["kl_cursor"]),
        )

    def _ensure(self, conn: Any, task_key: TaskKey) -> ValueRecord:
        key = task_key_id(task_key)
        row = conn.execute("SELECT * FROM value_records WHERE task_key=?", (key,)).fetchone()
        if row is not None:
            return self._record(row)
        # History preceding a new task is not evidence decay for that task.
        cursor = conn.execute(
            "SELECT COALESCE(MAX(segment_id),0) FROM kl_segments"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO value_records(task_key,task_payload,V,success_p,effective_count,"
            "last_observation_sequence,last_policy_version,kl_cursor) VALUES(?,?,?,?,?,NULL,NULL,?)",
            (key, canonical_json(task_key), self.prior_value, self.prior_success,
             self.prior_strength, int(cursor)),
        )
        return ValueRecord(task_key, self.prior_value, self.prior_success,
                           self.prior_strength, None, None, int(cursor))

    def _forget(self, conn: Any, record: ValueRecord) -> tuple[ValueRecord, float, float]:
        key = task_key_id(record.task_key)
        rows = conn.execute(
            "SELECT segment_id,kl_value FROM kl_segments WHERE segment_id>? ORDER BY segment_id",
            (record.kl_cursor,),
        ).fetchall()
        if not rows:
            return record, 1.0, 0.0
        divergence = math.fsum(float(row["kl_value"]) for row in rows)
        rho = 2.0 ** (-divergence / self.kl_half_life)
        forgotten = ValueRecord(
            task_key=record.task_key,
            V=self.prior_value + rho * (record.V - self.prior_value),
            success_p=self.prior_success + rho * (record.success_p - self.prior_success),
            effective_count=rho * record.effective_count,
            last_observation_sequence=record.last_observation_sequence,
            last_policy_version=record.last_policy_version,
            kl_cursor=int(rows[-1]["segment_id"]),
        )
        conn.execute(
            "UPDATE value_records SET V=?,success_p=?,effective_count=?,kl_cursor=? "
            "WHERE task_key=?",
            (forgotten.V, forgotten.success_p, forgotten.effective_count,
             forgotten.kl_cursor, key),
        )
        return forgotten, rho, divergence

    def snapshot(self, task_key: TaskKey) -> ValueRecord:
        if not isinstance(task_key, TaskKey):
            raise TypeError("task_key must be TaskKey")
        with self.ledger.transaction() as conn:
            record, _, _ = self._forget(conn, self._ensure(conn, task_key))
            return record

    get = snapshot

    def observe(self, trajectory: Trajectory,
                reward: RewardRecord) -> ValueObservation:
        """Commit lazy forgetting, ``A=R-V_pre``, V/P update and raw A atomically."""
        if not isinstance(trajectory, Trajectory) or not isinstance(reward, RewardRecord):
            raise TypeError("observe requires Trajectory and RewardRecord")
        if not reward.eligible:
            raise ValueError("ineligible rewards cannot update SPO value state")
        with self.ledger.transaction() as conn:
            stored_trajectory = self.ledger._get_trajectory(conn, trajectory.trajectory_id)
            if (stored_trajectory.task_key != trajectory.task_key
                    or stored_trajectory.stream_sequence != trajectory.stream_sequence
                    or stored_trajectory.behavior_adapter != trajectory.behavior_adapter):
                raise ValueError("observed trajectory identity differs from the ledger")
            stored_reward = conn.execute(
                "SELECT payload FROM rewards WHERE trajectory_id=?",
                (trajectory.trajectory_id,),
            ).fetchone()
            if stored_reward is None or stored_reward[0] != canonical_json(reward):
                raise ValueError("record_reward must persist the identical reward before observe")
            existing = conn.execute(
                "SELECT * FROM value_observations WHERE trajectory_id=?",
                (trajectory.trajectory_id,),
            ).fetchone()
            if existing is not None:
                expected = (
                    trajectory.stream_sequence, task_key_id(trajectory.task_key),
                    float(reward.total_reward), int(reward.resolved),
                    trajectory.behavior_adapter.version,
                )
                actual = (
                    existing["stream_sequence"], existing["task_key"], existing["reward"],
                    existing["resolved"], existing["policy_version"],
                )
                if actual != expected:
                    raise ValueError("observation ID replay has different evidence")
                return ValueObservation(
                    trajectory_id=trajectory.trajectory_id, task_key=trajectory.task_key,
                    V_pre=float(existing["V_pre"]),
                    raw_advantage=float(existing["raw_advantage"]),
                    V_after=float(existing["V_after"]),
                    success_p_after=float(existing["success_p_after"]),
                    effective_count_after=float(existing["effective_count_after"]),
                    eta=float(existing["eta"]),
                    forgetting_factor=float(existing["forgetting_factor"]),
                    forgotten_kl=float(existing["forgotten_kl"]),
                    kl_cursor=int(existing["kl_cursor_after"]), idempotent_replay=True,
                )
            sequence_conflict = conn.execute(
                "SELECT trajectory_id FROM value_observations WHERE stream_sequence=?",
                (trajectory.stream_sequence,),
            ).fetchone()
            if sequence_conflict is not None:
                raise ValueError("stream_sequence already belongs to another observation")
            record, rho, divergence = self._forget(
                conn, self._ensure(conn, stored_trajectory.task_key)
            )
            v_pre = record.V
            raw_advantage = float(reward.total_reward) - v_pre
            eta = 1.0 / (record.effective_count + 1.0)
            v_after = v_pre + eta * (float(reward.total_reward) - v_pre)
            success_after = record.success_p + eta * (
                float(reward.resolved) - record.success_p
            )
            effective_after = record.effective_count + 1.0
            now = time.time()
            conn.execute(
                "INSERT INTO value_observations(trajectory_id,stream_sequence,task_key,reward,"
                "resolved,V_pre,raw_advantage,V_after,success_p_after,effective_count_after,"
                "eta,policy_version,kl_cursor_after,forgetting_factor,forgotten_kl,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (trajectory.trajectory_id, trajectory.stream_sequence,
                 task_key_id(trajectory.task_key), float(reward.total_reward),
                 int(reward.resolved), v_pre, raw_advantage, v_after, success_after,
                 effective_after, eta, trajectory.behavior_adapter.version,
                 record.kl_cursor, rho, divergence, now),
            )
            conn.execute(
                "UPDATE value_records SET V=?,success_p=?,effective_count=?,"
                "last_observation_sequence=?,last_policy_version=?,kl_cursor=? WHERE task_key=?",
                (v_after, success_after, effective_after, trajectory.stream_sequence,
                 trajectory.behavior_adapter.version, record.kl_cursor,
                 task_key_id(trajectory.task_key)),
            )
            self.ledger._put_trajectory(conn, replace(
                stored_trajectory, raw_advantage=raw_advantage, status="REWARDED"
            ))
            self.ledger._event(conn, "value_observed", trajectory.trajectory_id, {
                "R": reward.total_reward, "V_pre": v_pre, "A": raw_advantage,
                "V_after": v_after, "eta": eta, "forgotten_kl": divergence,
                "kl_cursor": record.kl_cursor,
            })
            return ValueObservation(
                trajectory_id=trajectory.trajectory_id, task_key=trajectory.task_key,
                V_pre=v_pre, raw_advantage=raw_advantage, V_after=v_after,
                success_p_after=success_after, effective_count_after=effective_after,
                eta=eta, forgetting_factor=rho, forgotten_kl=divergence,
                kl_cursor=record.kl_cursor,
            )
