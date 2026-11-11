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
class FrozenBaseline:
    """Dispatch-time value snapshot pinned to one trajectory.

    ``V``/``success_p``/``effective_count`` already include KL forgetting for
    segments known at dispatch.  Later segments (policy updates during
    execution) must not move this trajectory's advantage baseline; they only
    affect shared value maintenance at observation time.
    """

    task_key: TaskKey
    V: float
    success_p: float
    effective_count: float
    kl_cursor: int
    policy_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key.to_dict(), "V": self.V,
            "success_p": self.success_p, "effective_count": self.effective_count,
            "kl_cursor": self.kl_cursor, "policy_version": self.policy_version,
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
    dispatch_V: float | None = None
    dispatch_kl_cursor: int | None = None
    dispatch_policy_version: int | None = None
    shared_applied: bool = True


class ValueTracker:
    """Observe each independent trajectory exactly once.

    ``observe`` is the only method that applies a reward.  ``snapshot`` performs
    lazy KL forgetting only, and is therefore safe for sampler use.  Both advance
    the persisted KL cursor so recovery cannot apply a segment twice.
    """

    def __init__(self, ledger: TrajectoryLedger, *, prior_value: float = 0.5,
                 prior_success: float = 0.5, prior_strength: float = 1.0,
                 kl_half_life: float = 0.05, stale_dispatch_seconds: float = 3600.0):
        self.ledger = ledger
        self.prior_value = self._unit(prior_value, "prior_value")
        self.prior_success = self._unit(prior_success, "prior_success")
        if isinstance(prior_strength, bool) or not isinstance(prior_strength, (int, float)) \
                or not math.isfinite(float(prior_strength)) or prior_strength < 0:
            raise ValueError("prior_strength must be finite and non-negative")
        if isinstance(kl_half_life, bool) or not isinstance(kl_half_life, (int, float)) \
                or not math.isfinite(float(kl_half_life)) or kl_half_life <= 0:
            raise ValueError("kl_half_life must be finite and positive")
        if isinstance(stale_dispatch_seconds, bool) \
                or not isinstance(stale_dispatch_seconds, (int, float)) \
                or not math.isfinite(float(stale_dispatch_seconds)) \
                or stale_dispatch_seconds <= 0:
            raise ValueError("stale_dispatch_seconds must be finite and positive")
        self.prior_strength = float(prior_strength)
        self.kl_half_life = float(kl_half_life)
        self.stale_dispatch_seconds = float(stale_dispatch_seconds)

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

    def peek(self, task_key: TaskKey) -> ValueRecord:
        """Read-only dispatch view: forgetting without persisting anything.

        Sampling and dispatch freezing must not advance the persisted KL
        cursor; only :meth:`observe` consumes segments.  Repeated peeks with
        no intervening observation return identical values.
        """
        if not isinstance(task_key, TaskKey):
            raise TypeError("task_key must be TaskKey")
        with self.ledger.transaction() as conn:
            record = self._ensure(conn, task_key)
            rows = conn.execute(
                "SELECT segment_id,kl_value FROM kl_segments WHERE segment_id>? ORDER BY segment_id",
                (record.kl_cursor,),
            ).fetchall()
            if not rows:
                return record
            divergence = math.fsum(float(row["kl_value"]) for row in rows)
            rho = 2.0 ** (-divergence / self.kl_half_life)
            return ValueRecord(
                record.task_key,
                self.prior_value + rho * (record.V - self.prior_value),
                self.prior_success + rho * (record.success_p - self.prior_success),
                rho * record.effective_count,
                record.last_observation_sequence, record.last_policy_version,
                int(rows[-1]["segment_id"]),
            )

    def freeze_dispatch(self, task_key: TaskKey, policy_version: int) -> FrozenBaseline:
        """Pin the dispatch-time advantage baseline for one trajectory."""
        if isinstance(policy_version, bool) or not isinstance(policy_version, int) \
                or policy_version < 0:
            raise ValueError("policy_version must be a non-negative integer")
        record = self.peek(task_key)
        return FrozenBaseline(
            task_key=task_key, V=record.V, success_p=record.success_p,
            effective_count=record.effective_count, kl_cursor=record.kl_cursor,
            policy_version=policy_version,
        )

    TERMINAL_UNOBSERVABLE_STATUSES = frozenset({"EXCLUDED", "STALE_EXCLUDED"})

    def note_dispatch(self, task_key: TaskKey, stream_sequence: int,
                      trajectory_id: str) -> None:
        """Record a dispatch marker so later completions wait their turn.

        Markers make earlier-but-slower trajectories visible before they seal.
        They are removed when the observation applies, on requeue, or when a
        stale marker ages out during drain. Idempotent per trajectory.
        """
        if not isinstance(task_key, TaskKey):
            raise TypeError("task_key must be TaskKey")
        if isinstance(stream_sequence, bool) or not isinstance(stream_sequence, int) \
                or stream_sequence <= 0:
            raise ValueError("stream_sequence must be a positive integer")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("trajectory_id must be a non-empty string")
        with self.ledger.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO dispatch_markers(stream_sequence,task_key,"
                "trajectory_id,created_at) VALUES(?,?,?,?)",
                (stream_sequence, task_key_id(task_key), trajectory_id, time.time()),
            )

    def unmark_dispatch(self, trajectory_id: str) -> None:
        """Drop a dispatch marker for an assignment that will never run."""
        with self.ledger.transaction() as conn:
            conn.execute("DELETE FROM dispatch_markers WHERE trajectory_id=?",
                         (trajectory_id,))

    def drain_pending(self, task_key: TaskKey) -> int:
        """Apply ready pending observations for one task in stream order."""
        if not isinstance(task_key, TaskKey):
            raise TypeError("task_key must be TaskKey")
        with self.ledger.transaction() as conn:
            before = conn.execute(
                "SELECT COUNT(*) FROM value_pending WHERE task_key=?",
                (task_key_id(task_key),)).fetchone()[0]
            self._drain_task(conn, task_key)
            after = conn.execute(
                "SELECT COUNT(*) FROM value_pending WHERE task_key=?",
                (task_key_id(task_key),)).fetchone()[0]
            return int(before - after)

    def _turn_ready(self, conn: Any, key: str, seq: int, now: float) -> bool:
        """An earlier sealed-but-unobserved trajectory, or a live earlier
        dispatch marker, blocks this sequence. Terminally unobservable rows
        (excluded, or an ineligible recorded reward) never block."""
        blocking = conn.execute(
            "SELECT t.trajectory_id,t.status,r.payload FROM trajectories t "
            "LEFT JOIN rewards r ON r.trajectory_id=t.trajectory_id "
            "LEFT JOIN value_observations v ON v.trajectory_id=t.trajectory_id "
            "WHERE t.task_key=? AND t.stream_sequence<? AND v.trajectory_id IS NULL",
            (key, seq)).fetchall()
        for row in blocking:
            if row["status"] in self.TERMINAL_UNOBSERVABLE_STATUSES:
                continue
            payload = row["payload"]
            if payload is not None:
                try:
                    staged_reward = RewardRecord.from_dict(json.loads(payload))
                except (ValueError, TypeError, KeyError):
                    return False
                if not staged_reward.eligible:
                    continue
            return False
        markers = conn.execute(
            "SELECT trajectory_id,created_at FROM dispatch_markers "
            "WHERE task_key=? AND stream_sequence<? AND trajectory_id NOT IN "
            "(SELECT trajectory_id FROM trajectories)",
            (key, seq)).fetchall()
        for marker in markers:
            if now - float(marker["created_at"]) > self.stale_dispatch_seconds:
                conn.execute("DELETE FROM dispatch_markers WHERE trajectory_id=?",
                             (marker["trajectory_id"],))
                self.ledger._event(conn, "value_observation_gap_skipped",
                                   marker["trajectory_id"], {"task_key": key})
                continue
            return False
        return True

    def _apply_pending_row(self, conn: Any, row: Any, *, late: bool = False) -> None:
        """Fold one staged observation into shared V/success_p exactly once.

        A late row (dispatched before the current sequence floor but applied
        after it) keeps its frozen reward/snapshot/advantage for training,
        but must not move shared state: no V/success_p/effective update, no
        KL refold, and no last_observation_sequence rollback. It is recorded
        as an explicitly late observation instead.
        """
        trajectory = self.ledger._get_trajectory(conn, row["trajectory_id"])
        reward = RewardRecord.from_dict(json.loads(row["reward_payload"]))
        if late:
            v_pre = float(row["dispatch_v"]) if row["dispatch_v"] is not None else None
            if v_pre is None:
                raise ValueError("late observations require a dispatch-frozen baseline")
            raw_advantage = float(reward.total_reward) - v_pre
            current = self._ensure(conn, trajectory.task_key)
            conn.execute(
                "INSERT INTO value_observations(trajectory_id,stream_sequence,task_key,reward,"
                "resolved,V_pre,raw_advantage,V_after,success_p_after,effective_count_after,"
                "eta,policy_version,kl_cursor_after,forgetting_factor,forgotten_kl,created_at,"
                "dispatch_v,dispatch_kl_cursor) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (trajectory.trajectory_id, trajectory.stream_sequence,
                 task_key_id(trajectory.task_key), float(reward.total_reward),
                 int(reward.resolved), v_pre, raw_advantage, current.V,
                 current.success_p, current.effective_count, 0.0,
                 trajectory.behavior_adapter.version, current.kl_cursor,
                 1.0, 0.0, time.time(),
                 row["dispatch_v"], row["dispatch_kl_cursor"]),
            )
            self.ledger._put_trajectory(conn, replace(
                trajectory, raw_advantage=raw_advantage,
                status="REWARDED" if trajectory.status == "VERIFYING" else trajectory.status,
            ))
            conn.execute("DELETE FROM value_pending WHERE trajectory_id=?",
                         (trajectory.trajectory_id,))
            conn.execute("DELETE FROM dispatch_markers WHERE trajectory_id=?",
                         (trajectory.trajectory_id,))
            self.ledger._event(conn, "value_observed_late", trajectory.trajectory_id, {
                "R": reward.total_reward, "V_pre": v_pre, "A": raw_advantage,
                "reason": "dispatched before the applied sequence floor",
                "dispatch_V": row["dispatch_v"],
                "dispatch_kl_cursor": row["dispatch_kl_cursor"],
            })
            return
        record, rho, divergence = self._forget(
            conn, self._ensure(conn, trajectory.task_key)
        )
        v_pre = float(row["dispatch_v"]) if row["dispatch_v"] is not None else record.V
        raw_advantage = float(reward.total_reward) - v_pre
        eta = 1.0 / (record.effective_count + 1.0)
        v_after = record.V + eta * (float(reward.total_reward) - record.V)
        success_after = record.success_p + eta * (float(reward.resolved) - record.success_p)
        effective_after = record.effective_count + 1.0
        conn.execute(
            "INSERT INTO value_observations(trajectory_id,stream_sequence,task_key,reward,"
            "resolved,V_pre,raw_advantage,V_after,success_p_after,effective_count_after,"
            "eta,policy_version,kl_cursor_after,forgetting_factor,forgotten_kl,created_at,"
            "dispatch_v,dispatch_kl_cursor) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trajectory.trajectory_id, trajectory.stream_sequence,
             task_key_id(trajectory.task_key), float(reward.total_reward),
             int(reward.resolved), v_pre, raw_advantage, v_after, success_after,
             effective_after, eta, trajectory.behavior_adapter.version,
             record.kl_cursor, rho, divergence, time.time(),
             row["dispatch_v"], row["dispatch_kl_cursor"]),
        )
        conn.execute(
            "UPDATE value_records SET V=?,success_p=?,effective_count=?,"
            "last_observation_sequence=?,last_policy_version=?,kl_cursor=? WHERE task_key=?",
            (v_after, success_after, effective_after, trajectory.stream_sequence,
             trajectory.behavior_adapter.version, record.kl_cursor,
             task_key_id(trajectory.task_key)),
        )
        self.ledger._put_trajectory(conn, replace(
            trajectory, raw_advantage=raw_advantage,
            status="REWARDED" if trajectory.status == "VERIFYING" else trajectory.status,
        ))
        conn.execute("DELETE FROM value_pending WHERE trajectory_id=?",
                     (trajectory.trajectory_id,))
        conn.execute("DELETE FROM dispatch_markers WHERE trajectory_id=?",
                     (trajectory.trajectory_id,))
        self.ledger._event(conn, "value_observed", trajectory.trajectory_id, {
            "R": reward.total_reward, "V_pre": v_pre, "A": raw_advantage,
            "V_after": v_after, "eta": eta, "forgotten_kl": divergence,
            "kl_cursor": record.kl_cursor, "dispatch_V": row["dispatch_v"],
            "dispatch_kl_cursor": row["dispatch_kl_cursor"], "ordered": True,
            "late": late,
        })

    def _drain_task(self, conn: Any, task_key: TaskKey) -> None:
        """Apply staged observations in dispatch order; stop at the first gap."""
        key = task_key_id(task_key)
        now = time.time()
        while True:
            floor = conn.execute(
                "SELECT last_observation_sequence FROM value_records WHERE task_key=?",
                (key,)).fetchone()
            floor_seq = int(floor[0]) if floor is not None and floor[0] is not None else 0
            nxt = conn.execute(
                "SELECT * FROM value_pending WHERE task_key=? AND stream_sequence>? "
                "ORDER BY stream_sequence LIMIT 1", (key, floor_seq)).fetchone()
            if nxt is None:
                break
            if not self._turn_ready(conn, key, int(nxt["stream_sequence"]), now):
                break
            self._apply_pending_row(conn, nxt)
        # Dispatch markers cannot track assignments that sealed late without
        # one (or predated markers): fold leftovers immediately and log them.
        floor = conn.execute(
            "SELECT last_observation_sequence FROM value_records WHERE task_key=?",
            (key,)).fetchone()
        floor_seq = int(floor[0]) if floor is not None and floor[0] is not None else 0
        leftovers = conn.execute(
            "SELECT * FROM value_pending WHERE task_key=? AND stream_sequence<=? "
            "ORDER BY stream_sequence", (key, floor_seq)).fetchall()
        for row in leftovers:
            self._apply_pending_row(conn, row, late=True)

    def purge_orphan_markers(self) -> int:
        """Delete dispatch markers for assignments that never sealed.

        At startup no worker is alive, so a marker without a trajectory row
        is a dead assignment, not a slow one. Markers attached to sealed
        rows stay under the sealed-row rule and need no cleanup here.
        """
        with self.ledger.transaction() as conn:
            orphans = conn.execute(
                "SELECT trajectory_id,task_key FROM dispatch_markers "
                "WHERE trajectory_id NOT IN (SELECT trajectory_id FROM trajectories)"
            ).fetchall()
            for row in orphans:
                conn.execute("DELETE FROM dispatch_markers WHERE trajectory_id=?",
                             (row["trajectory_id"],))
                self.ledger._event(conn, "dispatch_marker_purged",
                                   row["trajectory_id"], {"task_key": row["task_key"]})
            return len(orphans)

    @staticmethod
    def _baseline_from_provenance(trajectory: Trajectory) -> FrozenBaseline | None:
        """Rebuild a dispatch-frozen baseline from sealed trajectory evidence."""
        try:
            provenance = trajectory.provenance
            if not isinstance(provenance, dict):
                return None
            # All-or-nothing: a partial frozen snapshot is not a baseline.
            # Missing keys (rows sealed before strict dispatch) fall back to
            # the explicitly unordered legacy observation path.
            return FrozenBaseline(
                task_key=trajectory.task_key,
                V=float(provenance["frozen_V"]),
                success_p=float(provenance["frozen_success_p"]),
                effective_count=float(provenance["frozen_effective_count"]),
                kl_cursor=int(provenance["frozen_kl_cursor"]),
                policy_version=trajectory.behavior_adapter.version,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def recover_trajectory(self, trajectory_id: str) -> str:
        """Force one sealed trajectory to a terminal value state (idempotent).

        - No reward: EXCLUDED as ``crash_before_reward`` (never fabricated).
        - Ineligible reward: EXCLUDED, marker cleared.
        - Eligible reward without observation/pending: re-observe from stored
          evidence (provenance-frozen baseline, legacy fallback).
        - Already observed: no-op.
        Returns the action taken. Safe to call from a worker-settle callback;
        it never raises for a missing trajectory (returns ``unknown``).
        """
        with self.ledger.transaction() as conn:
            payload = conn.execute(
                "SELECT payload FROM rewards WHERE trajectory_id=?",
                (trajectory_id,)).fetchone()
            try:
                trajectory = self.ledger._get_trajectory(conn, trajectory_id)
            except (KeyError, ValueError):
                return "unknown"
            observed = conn.execute(
                "SELECT 1 FROM value_observations WHERE trajectory_id=?",
                (trajectory_id,)).fetchone() is not None
            pending = conn.execute(
                "SELECT 1 FROM value_pending WHERE trajectory_id=?",
                (trajectory_id,)).fetchone() is not None
            if payload is None:
                if trajectory.status not in ("VERIFYING", "REWARDED"):
                    return "no_reward_other_state"
                self.ledger._put_trajectory(conn, replace(trajectory, status="EXCLUDED"))
                conn.execute("DELETE FROM dispatch_markers WHERE trajectory_id=?",
                             (trajectory_id,))
                self.ledger._event(conn, "trajectory_excluded", trajectory_id,
                                   {"reason": "crash_before_reward"})
                return "excluded_no_reward"
            reward = RewardRecord.from_dict(json.loads(payload[0]))
            if not reward.eligible:
                if trajectory.status not in self.TERMINAL_UNOBSERVABLE_STATUSES:
                    self.ledger._put_trajectory(conn, replace(trajectory, status="EXCLUDED"))
                conn.execute("DELETE FROM dispatch_markers WHERE trajectory_id=?",
                             (trajectory_id,))
                self.ledger._event(conn, "trajectory_excluded", trajectory_id,
                                   {"reason": "ineligible_reward_recovery"})
                return "excluded_ineligible"
            if observed or pending:
                return "already_staged"
            if trajectory.status not in ("VERIFYING", "REWARDED"):
                raise RuntimeError(
                    f"trajectory {trajectory_id} is {trajectory.status} with no value "
                    "observation; refusing to guess its history")
            baseline = self._baseline_from_provenance(trajectory)
        self.observe(self.ledger.get_trajectory(trajectory_id), reward,
                     baseline=baseline)
        return "reobserved"

    def recover_incomplete(self) -> dict[str, int]:
        """Startup scan for crash-window trajectories (idempotent).

        - Sealed but rewardless (crashed between record_trajectory and
          record_reward): EXCLUDED with an explicit reason; no reward is
          fabricated, and the terminal status unblocks later sequences.
        - Eligible reward but neither observation nor pending: re-observe
          from stored evidence, rebuilding the frozen baseline from sealed
          provenance (legacy immediate path when provenance lacks it).
        - Markers pinning terminally-unobservable sealed rows: cleared.
        Returns stage counts.
        """
        summary = {"excluded_no_reward": 0, "reobserved": 0, "markers_cleared": 0}
        with self.ledger.transaction() as conn:
            rows = conn.execute(
                "SELECT t.trajectory_id FROM trajectories t LEFT JOIN rewards r "
                "ON r.trajectory_id=t.trajectory_id WHERE r.trajectory_id IS NULL "
                "AND t.status IN ('VERIFYING','REWARDED')").fetchall()
            rewardless = [row["trajectory_id"] for row in rows]
            rows = conn.execute(
                "SELECT t.trajectory_id FROM trajectories t "
                "JOIN rewards r ON r.trajectory_id=t.trajectory_id "
                "LEFT JOIN value_observations v ON v.trajectory_id=t.trajectory_id "
                "LEFT JOIN value_pending p ON p.trajectory_id=t.trajectory_id "
                "WHERE v.trajectory_id IS NULL AND p.trajectory_id IS NULL").fetchall()
            unobserved = [row["trajectory_id"] for row in rows]
        for trajectory_id in (*rewardless, *unobserved):
            action = self.recover_trajectory(trajectory_id)
            if action == "excluded_no_reward":
                summary["excluded_no_reward"] += 1
            elif action == "reobserved":
                summary["reobserved"] += 1
            elif action in ("excluded_ineligible",):
                summary["markers_cleared"] += 1
        with self.ledger.transaction() as conn:
            rows = conn.execute(
                "SELECT m.trajectory_id FROM dispatch_markers m "
                "JOIN trajectories t ON t.trajectory_id=m.trajectory_id "
                "LEFT JOIN rewards r ON r.trajectory_id=m.trajectory_id "
                "WHERE t.status IN ('EXCLUDED','STALE_EXCLUDED')").fetchall()
            for row in rows:
                conn.execute("DELETE FROM dispatch_markers WHERE trajectory_id=?",
                             (row["trajectory_id"],))
                summary["markers_cleared"] += 1
        return summary

    def observe(self, trajectory: Trajectory,
                reward: RewardRecord,
                baseline: FrozenBaseline | None = None) -> ValueObservation:
        """Commit one observation with a dispatch-frozen advantage baseline.

        ``A = R - V_dispatch`` where ``V_dispatch`` was pinned at dispatch;
        policy updates during execution cannot move this trajectory's baseline.
        The shared V/success_p update is applied in dispatch (stream_sequence)
        order per task: a completion waits in ``value_pending`` while earlier
        dispatches are unobserved, so completion order cannot reorder history.
        Replaying an observed trajectory returns the stored observation.
        """
        if not isinstance(trajectory, Trajectory) or not isinstance(reward, RewardRecord):
            raise TypeError("observe requires Trajectory and RewardRecord")
        if not reward.eligible:
            raise ValueError("ineligible rewards cannot update SPO value state")
        if baseline is not None:
            if not isinstance(baseline, FrozenBaseline):
                raise TypeError("baseline must be a FrozenBaseline")
            if baseline.task_key != trajectory.task_key:
                raise ValueError("frozen baseline belongs to another task")
            if baseline.policy_version != trajectory.behavior_adapter.version:
                raise ValueError("frozen baseline policy must match the behavior adapter")
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
                columns = {row[1] for row in conn.execute("PRAGMA table_info(value_observations)")}
                dispatch_v = float(existing["dispatch_v"]) \
                    if "dispatch_v" in columns and existing["dispatch_v"] is not None else None
                dispatch_cursor = existing["dispatch_kl_cursor"] \
                    if "dispatch_kl_cursor" in columns else None
                if baseline is not None and (dispatch_v is None
                        or abs(dispatch_v - baseline.V) > 1e-12
                        or dispatch_cursor != baseline.kl_cursor):
                    raise ValueError("replay frozen baseline differs from the stored observation")
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
                    dispatch_V=dispatch_v,
                    dispatch_kl_cursor=int(dispatch_cursor) if dispatch_cursor is not None else None,
                    dispatch_policy_version=trajectory.behavior_adapter.version,
                    shared_applied=True,
                )
            if baseline is None:
                return self._observe_immediate(
                    conn, stored_trajectory, trajectory, reward)
            dispatch_v = baseline.V
            dispatch_cursor = baseline.kl_cursor
            conn.execute(
                "INSERT OR IGNORE INTO value_pending(trajectory_id,stream_sequence,task_key,"
                "reward,resolved,policy_version,dispatch_v,dispatch_kl_cursor,"
                "reward_payload,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (trajectory.trajectory_id, trajectory.stream_sequence,
                 task_key_id(trajectory.task_key), float(reward.total_reward),
                 int(reward.resolved), trajectory.behavior_adapter.version,
                 dispatch_v, dispatch_cursor, canonical_json(reward), time.time()),
            )
            staged = conn.execute(
                "SELECT * FROM value_pending WHERE trajectory_id=?",
                (trajectory.trajectory_id,)).fetchone()
            if (staged is None or staged["stream_sequence"] != trajectory.stream_sequence
                    or staged["task_key"] != task_key_id(trajectory.task_key)
                    or staged["reward_payload"] != canonical_json(reward)
                    or staged["dispatch_v"] is None
                    or abs(float(staged["dispatch_v"]) - dispatch_v) > 1e-12
                    or staged["dispatch_kl_cursor"] != dispatch_cursor):
                raise ValueError("pending observation already staged with different evidence")
            # The frozen advantage is order-independent: publish it on the
            # trajectory now so the buffer can proceed while shared V waits
            # for earlier dispatch sequences.
            raw_advantage = float(reward.total_reward) - dispatch_v
            if stored_trajectory.raw_advantage is None:
                self.ledger._put_trajectory(conn, replace(
                    stored_trajectory, raw_advantage=raw_advantage,
                    status="REWARDED" if stored_trajectory.status == "VERIFYING"
                    else stored_trajectory.status,
                ))
            self._drain_task(conn, trajectory.task_key)
            applied = conn.execute(
                "SELECT * FROM value_observations WHERE trajectory_id=?",
                (trajectory.trajectory_id,)).fetchone()
            if applied is not None:
                return ValueObservation(
                    trajectory_id=trajectory.trajectory_id, task_key=trajectory.task_key,
                    V_pre=float(applied["V_pre"]),
                    raw_advantage=float(applied["raw_advantage"]),
                    V_after=float(applied["V_after"]),
                    success_p_after=float(applied["success_p_after"]),
                    effective_count_after=float(applied["effective_count_after"]),
                    eta=float(applied["eta"]),
                    forgetting_factor=float(applied["forgetting_factor"]),
                    forgotten_kl=float(applied["forgotten_kl"]),
                    kl_cursor=int(applied["kl_cursor_after"]),
                    dispatch_V=dispatch_v, dispatch_kl_cursor=dispatch_cursor,
                    dispatch_policy_version=trajectory.behavior_adapter.version,
                    shared_applied=True,
                )
            current = self._ensure(conn, trajectory.task_key)
            return ValueObservation(
                trajectory_id=trajectory.trajectory_id, task_key=trajectory.task_key,
                V_pre=dispatch_v, raw_advantage=raw_advantage,
                V_after=current.V, success_p_after=current.success_p,
                effective_count_after=current.effective_count, eta=0.0,
                forgetting_factor=1.0, forgotten_kl=0.0,
                kl_cursor=current.kl_cursor,
                dispatch_V=dispatch_v, dispatch_kl_cursor=dispatch_cursor,
                dispatch_policy_version=trajectory.behavior_adapter.version,
                shared_applied=False,
            )

    def _observe_immediate(self, conn: Any, stored_trajectory: Trajectory,
                           trajectory: Trajectory,
                           reward: RewardRecord) -> ValueObservation:
        """Legacy unordered path for manual/recovery callers without a baseline.

        The production engine always passes a dispatch-frozen baseline and
        uses the ordered pending queue above.
        """
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
        v_after = record.V + eta * (float(reward.total_reward) - record.V)
        success_after = record.success_p + eta * (
            float(reward.resolved) - record.success_p
        )
        effective_after = record.effective_count + 1.0
        now = time.time()
        columns = {row[1] for row in conn.execute("PRAGMA table_info(value_observations)")}
        if {"dispatch_v", "dispatch_kl_cursor"} <= columns:
            conn.execute(
                "INSERT INTO value_observations(trajectory_id,stream_sequence,task_key,reward,"
                "resolved,V_pre,raw_advantage,V_after,success_p_after,effective_count_after,"
                "eta,policy_version,kl_cursor_after,forgetting_factor,forgotten_kl,created_at,"
                "dispatch_v,dispatch_kl_cursor) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (trajectory.trajectory_id, trajectory.stream_sequence,
                 task_key_id(trajectory.task_key), float(reward.total_reward),
                 int(reward.resolved), v_pre, raw_advantage, v_after, success_after,
                 effective_after, eta, trajectory.behavior_adapter.version,
                 record.kl_cursor, rho, divergence, now, None, None),
            )
        else:
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
            "kl_cursor": record.kl_cursor, "dispatch_V": None,
            "dispatch_kl_cursor": None, "ordered": False,
        })
        return ValueObservation(
            trajectory_id=trajectory.trajectory_id, task_key=trajectory.task_key,
            V_pre=v_pre, raw_advantage=raw_advantage, V_after=v_after,
            success_p_after=success_after, effective_count_after=effective_after,
            eta=eta, forgetting_factor=rho, forgotten_kl=divergence,
            kl_cursor=record.kl_cursor,
            dispatch_V=None, dispatch_kl_cursor=None,
            dispatch_policy_version=trajectory.behavior_adapter.version,
            shared_applied=True,
        )
