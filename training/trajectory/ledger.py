"""Recoverable SQLite ledger for trajectories, SPO state and update lifecycle.

One supervisor should own a ledger instance.  All state transitions use
``BEGIN IMMEDIATE``; WAL replay plus FULL synchronous commits preserves the last
complete transition without duplicating value observations or KL segments.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .schema import (
    BehaviorAdapter,
    RewardRecord,
    TaskKey,
    Trajectory,
    UpdateBatch,
    validate_update_batch,
)
from .token_alignment import validate_token_alignment


SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def task_key_id(task_key: TaskKey) -> str:
    return task_key.canonical_json()


class StaleTrajectoryError(RuntimeError):
    def __init__(self, update_id: str, trajectory_ids: Sequence[str]):
        self.update_id = update_id
        self.trajectory_ids = tuple(trajectory_ids)
        super().__init__(
            f"update {update_id} exceeds max policy lag: {', '.join(self.trajectory_ids)}"
        )


class TrajectoryLedger:
    """Unified persistent store.  Paths are private (0600), including WAL files."""

    def __init__(self, path: str | os.PathLike[str], *, timeout_seconds: float = 30.0):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_APPEND, 0o600)
        os.close(descriptor)
        os.chmod(self.path, 0o600)
        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(
            str(self.path), timeout=timeout_seconds, isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute(f"PRAGMA busy_timeout={int(timeout_seconds * 1000)}")
        self._create_schema()
        self._private_files()

    def _create_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trajectories (
            trajectory_id TEXT PRIMARY KEY,
            stream_sequence INTEGER NOT NULL UNIQUE,
            task_key TEXT NOT NULL,
            behavior_version INTEGER NOT NULL,
            behavior_sha256 TEXT NOT NULL,
            action_token_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS trajectories_task
            ON trajectories(task_key, stream_sequence);
        CREATE TABLE IF NOT EXISTS rewards (
            trajectory_id TEXT PRIMARY KEY REFERENCES trajectories(trajectory_id),
            total_reward REAL,
            resolved INTEGER,
            eligible INTEGER NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS value_records (
            task_key TEXT PRIMARY KEY,
            task_payload TEXT NOT NULL,
            V REAL NOT NULL,
            success_p REAL NOT NULL,
            effective_count REAL NOT NULL,
            last_observation_sequence INTEGER,
            last_policy_version INTEGER,
            kl_cursor INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS value_observations (
            trajectory_id TEXT PRIMARY KEY REFERENCES trajectories(trajectory_id),
            stream_sequence INTEGER NOT NULL UNIQUE,
            task_key TEXT NOT NULL,
            reward REAL NOT NULL,
            resolved INTEGER NOT NULL,
            V_pre REAL NOT NULL,
            raw_advantage REAL NOT NULL,
            V_after REAL NOT NULL,
            success_p_after REAL NOT NULL,
            effective_count_after REAL NOT NULL,
            eta REAL NOT NULL,
            policy_version INTEGER NOT NULL,
            kl_cursor_after INTEGER NOT NULL,
            forgetting_factor REAL NOT NULL,
            forgotten_kl REAL NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS observations_task
            ON value_observations(task_key, stream_sequence);
        CREATE TABLE IF NOT EXISTS buffer_entries (
            offer_order INTEGER PRIMARY KEY AUTOINCREMENT,
            trajectory_id TEXT NOT NULL UNIQUE REFERENCES trajectories(trajectory_id),
            state TEXT NOT NULL,
            update_id TEXT,
            stale_reason TEXT,
            offered_at REAL NOT NULL,
            transitioned_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS buffer_state
            ON buffer_entries(state, offer_order);
        CREATE TABLE IF NOT EXISTS updates (
            update_id TEXT PRIMARY KEY,
            phase TEXT NOT NULL,
            policy_before_version INTEGER NOT NULL,
            total_action_tokens INTEGER NOT NULL,
            optimizer_cursor INTEGER NOT NULL,
            payload TEXT NOT NULL,
            created_at REAL NOT NULL,
            transitioned_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS update_trajectories (
            update_id TEXT NOT NULL REFERENCES updates(update_id),
            trajectory_id TEXT NOT NULL REFERENCES trajectories(trajectory_id),
            position INTEGER NOT NULL,
            action_token_count INTEGER NOT NULL,
            raw_advantage REAL NOT NULL,
            frozen_advantage REAL NOT NULL,
            behavior_version INTEGER NOT NULL,
            PRIMARY KEY(update_id, trajectory_id),
            UNIQUE(update_id, position)
        );
        CREATE TABLE IF NOT EXISTS kl_segments (
            segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_key TEXT NOT NULL UNIQUE,
            update_id TEXT UNIQUE REFERENCES updates(update_id),
            policy_before TEXT NOT NULL,
            policy_after TEXT NOT NULL,
            kl_value REAL NOT NULL,
            token_count INTEGER NOT NULL,
            estimator TEXT NOT NULL,
            version_chain TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            subject_id TEXT,
            payload TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        """
        with self.transaction() as conn:
            conn.executescript(schema)
            for column, declaration in (
                ("dispatch_v", "REAL"), ("dispatch_kl_cursor", "INTEGER"),
            ):
                try:
                    conn.execute(
                        f"ALTER TABLE value_observations ADD COLUMN {column} {declaration}")
                except sqlite3.OperationalError:
                    pass
            conn.execute(
                "INSERT OR IGNORE INTO metadata(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )
            row = conn.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if row is None or int(row[0]) != SCHEMA_VERSION:
                raise RuntimeError("unsupported trajectory ledger schema")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Public unit-of-work boundary used by ValueTracker and orchestration."""
        with self._lock:
            if self._closed:
                raise RuntimeError("ledger is closed")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()
                self._private_files()

    def _private_files(self) -> None:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            if candidate.exists():
                os.chmod(candidate, 0o600)

    @staticmethod
    def _event(conn: sqlite3.Connection, event_type: str, subject_id: str | None,
               payload: Mapping[str, Any] | None = None) -> None:
        conn.execute(
            "INSERT INTO audit_events(event_type,subject_id,payload,created_at) "
            "VALUES(?,?,?,?)",
            (event_type, subject_id, canonical_json(payload or {}), time.time()),
        )

    @staticmethod
    def _put_trajectory(conn: sqlite3.Connection, trajectory: Trajectory) -> None:
        conn.execute(
            "UPDATE trajectories SET stream_sequence=?,task_key=?,behavior_version=?,"
            "behavior_sha256=?,action_token_count=?,status=?,payload=? WHERE trajectory_id=?",
            (
                trajectory.stream_sequence, task_key_id(trajectory.task_key),
                trajectory.behavior_adapter.version, trajectory.behavior_adapter.sha256,
                trajectory.action_token_count, trajectory.status,
                canonical_json(trajectory), trajectory.trajectory_id,
            ),
        )

    @staticmethod
    def _get_trajectory(conn: sqlite3.Connection, trajectory_id: str) -> Trajectory:
        row = conn.execute(
            "SELECT payload FROM trajectories WHERE trajectory_id=?", (trajectory_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown trajectory: {trajectory_id}")
        return Trajectory.from_dict(json.loads(row[0]))

    @staticmethod
    def _get_update(conn: sqlite3.Connection, update_id: str) -> UpdateBatch:
        row = conn.execute("SELECT payload FROM updates WHERE update_id=?", (update_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown update: {update_id}")
        return UpdateBatch.from_dict(json.loads(row[0]))

    def record_trajectory(self, trajectory: Trajectory | Mapping[str, Any]) -> None:
        if not isinstance(trajectory, Trajectory):
            trajectory = Trajectory.from_dict(trajectory)
        validate_token_alignment(trajectory)
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT payload FROM trajectories WHERE trajectory_id=?",
                (trajectory.trajectory_id,),
            ).fetchone()
            if row is not None:
                if row[0] != canonical_json(trajectory):
                    raise ValueError("trajectory ID already refers to different content")
                return
            conn.execute(
                "INSERT INTO trajectories(trajectory_id,stream_sequence,task_key,"
                "behavior_version,behavior_sha256,action_token_count,status,payload) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    trajectory.trajectory_id, trajectory.stream_sequence,
                    task_key_id(trajectory.task_key), trajectory.behavior_adapter.version,
                    trajectory.behavior_adapter.sha256, trajectory.action_token_count,
                    trajectory.status, canonical_json(trajectory),
                ),
            )
            self._event(conn, "trajectory_recorded", trajectory.trajectory_id)

    save_trajectory = record_trajectory

    def get_trajectory(self, trajectory_id: str) -> Trajectory:
        with self._lock:
            return self._get_trajectory(self._conn, trajectory_id)

    def get_trajectories(self, trajectory_ids: Sequence[str]) -> tuple[Trajectory, ...]:
        return tuple(self.get_trajectory(item) for item in trajectory_ids)

    def record_reward(self, trajectory_id: str,
                      reward: RewardRecord | Mapping[str, Any]) -> None:
        if not isinstance(reward, RewardRecord):
            reward = RewardRecord.from_dict(reward)
        with self.transaction() as conn:
            trajectory = self._get_trajectory(conn, trajectory_id)
            row = conn.execute(
                "SELECT payload FROM rewards WHERE trajectory_id=?", (trajectory_id,)
            ).fetchone()
            if row is not None:
                if row[0] != canonical_json(reward):
                    raise ValueError("reward already exists with different content")
                return
            conn.execute(
                "INSERT INTO rewards(trajectory_id,total_reward,resolved,eligible,payload) "
                "VALUES(?,?,?,?,?)",
                (trajectory_id, reward.total_reward,
                 None if reward.resolved is None else int(reward.resolved),
                 int(reward.eligible), canonical_json(reward)),
            )
            if trajectory.reward_ref is None:
                self._put_trajectory(conn, replace(
                    trajectory, reward_ref=f"reward:{trajectory_id}"
                ))
            self._event(conn, "reward_recorded", trajectory_id, {
                "eligible": reward.eligible, "total_reward": reward.total_reward,
            })

    def get_reward(self, trajectory_id: str) -> RewardRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM rewards WHERE trajectory_id=?", (trajectory_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown reward: {trajectory_id}")
        return RewardRecord.from_dict(json.loads(row[0]))

    def offer_trajectory(self, trajectory_id: str) -> str:
        """Offer one complete rewarded trajectory; never splits or reorders calls."""
        with self.transaction() as conn:
            trajectory = self._get_trajectory(conn, trajectory_id)
            existing = conn.execute(
                "SELECT state FROM buffer_entries WHERE trajectory_id=?", (trajectory_id,)
            ).fetchone()
            if existing is not None:
                return str(existing[0])
            reward_row = conn.execute(
                "SELECT eligible FROM rewards WHERE trajectory_id=?", (trajectory_id,)
            ).fetchone()
            if reward_row is None or not reward_row[0]:
                raise ValueError("only eligible rewarded trajectories can enter the buffer")
            if trajectory.raw_advantage is None:
                raise ValueError("ValueTracker.observe must freeze raw advantage before offer")
            now = time.time()
            conn.execute(
                "INSERT INTO buffer_entries(trajectory_id,state,offered_at,transitioned_at) "
                "VALUES(?,'BUFFERED',?,?)", (trajectory_id, now, now)
            )
            self._put_trajectory(conn, replace(trajectory, status="BUFFERED"))
            self._event(conn, "buffer_offered", trajectory_id, {
                "action_tokens": trajectory.action_token_count,
            })
            return "BUFFERED"

    def _transition_stale(self, conn: sqlite3.Connection, trajectory: Trajectory,
                          reason: str) -> None:
        now = time.time()
        conn.execute(
            "UPDATE buffer_entries SET state='STALE_EXCLUDED',stale_reason=?,"
            "transitioned_at=? WHERE trajectory_id=? AND state='BUFFERED'",
            (reason, now, trajectory.trajectory_id),
        )
        self._put_trajectory(conn, replace(trajectory, status="STALE_EXCLUDED"))
        self._event(conn, "buffer_stale", trajectory.trajectory_id, {"reason": reason})

    def mark_stale(self, trajectory_id: str, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("stale exclusion requires a reason")
        with self.transaction() as conn:
            trajectory = self._get_trajectory(conn, trajectory_id)
            row = conn.execute(
                "SELECT state FROM buffer_entries WHERE trajectory_id=?", (trajectory_id,)
            ).fetchone()
            if row is None or row[0] != "BUFFERED":
                raise RuntimeError("only a buffered trajectory can be marked stale directly")
            self._transition_stale(conn, trajectory, reason)

    def reserve_update(self, update_id: str, *, policy_before: BehaviorAdapter,
                       target_action_tokens: int, max_policy_lag: int,
                       advantage_epsilon: float = 1.0e-6,
                       force: bool = False) -> UpdateBatch | None:
        """Atomically reserve a FIFO prefix and freeze SPO++ advantages.

        The trajectory crossing the token target is included whole.  A small
        prefix can be forced or selected when its oldest call approaches max lag.
        Existing update IDs return their persisted frozen batch on recovery.
        """
        if isinstance(target_action_tokens, bool) or not isinstance(target_action_tokens, int) \
                or target_action_tokens <= 0:
            raise ValueError("target_action_tokens must be positive")
        if isinstance(max_policy_lag, bool) or not isinstance(max_policy_lag, int) \
                or max_policy_lag < 0:
            raise ValueError("max_policy_lag must be non-negative")
        if not isinstance(policy_before, BehaviorAdapter):
            raise TypeError("policy_before must be BehaviorAdapter")
        from ..spo.advantage import freeze_advantages

        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT payload FROM updates WHERE update_id=?", (update_id,)
            ).fetchone()
            if existing is not None:
                return UpdateBatch.from_dict(json.loads(existing[0]))
            rows = conn.execute(
                "SELECT trajectory_id FROM buffer_entries "
                "WHERE state='BUFFERED' ORDER BY offer_order"
            ).fetchall()
            candidates: list[Trajectory] = []
            for row in rows:
                trajectory = self._get_trajectory(conn, row[0])
                if trajectory.action_token_count == 0:
                    conn.execute(
                        "UPDATE buffer_entries SET state='NOOP_CONSUMED',transitioned_at=? "
                        "WHERE trajectory_id=?", (time.time(), trajectory.trajectory_id)
                    )
                    self._put_trajectory(conn, replace(trajectory, status="NOOP_CONSUMED"))
                    self._event(conn, "buffer_noop", trajectory.trajectory_id)
                    continue
                lags = [policy_before.version - call.behavior_adapter.version
                        for call in trajectory.calls]
                if not lags:
                    self._transition_stale(conn, trajectory, "trajectory has no action calls")
                    continue
                if min(lags) < 0:
                    self._transition_stale(
                        conn, trajectory,
                        "behavior adapter newer than policy_before",
                    )
                    continue
                if max(lags) > max_policy_lag:
                    self._transition_stale(
                        conn, trajectory,
                        f"policy lag {max(lags)} exceeds {max_policy_lag}",
                    )
                    continue
                candidates.append(trajectory)
            if not candidates:
                return None
            available = sum(item.action_token_count for item in candidates)
            oldest_lag = max(
                policy_before.version - call.behavior_adapter.version
                for call in candidates[0].calls
            )
            near_lag_limit = oldest_lag >= max(0, max_policy_lag - 1)
            if not force and available < target_action_tokens and not near_lag_limit:
                return None
            selected: list[Trajectory] = []
            tokens = 0
            for trajectory in candidates:
                selected.append(trajectory)
                tokens += trajectory.action_token_count
                if tokens >= target_action_tokens:
                    break
            batch = freeze_advantages(
                selected, update_id=update_id, policy_before=policy_before,
                max_policy_lag=max_policy_lag, epsilon=advantage_epsilon,
            )
            # A batch whose centered advantages are all exactly zero carries no
            # policy signal (a lone trajectory is its own mean). Never spend an
            # optimizer step, an LR-schedule advance and a version bump on it:
            # first extend the FIFO prefix with further buffered trajectories
            # from other tasks, else consume the selection as an explicit NOOP
            # terminal state so the pipeline keeps moving.
            while all(value == 0.0 for value in batch.frozen_advantages.values()) \
                    and len(selected) < len(candidates):
                selected.append(candidates[len(selected)])
                batch = freeze_advantages(
                    selected, update_id=update_id, policy_before=policy_before,
                    max_policy_lag=max_policy_lag, epsilon=advantage_epsilon,
                )
            if all(value == 0.0 for value in batch.frozen_advantages.values()):
                for trajectory in selected:
                    conn.execute(
                        "UPDATE buffer_entries SET state='NOOP_CONSUMED',transitioned_at=? "
                        "WHERE trajectory_id=?", (time.time(), trajectory.trajectory_id)
                    )
                    self._put_trajectory(conn, replace(trajectory, status="NOOP_CONSUMED"))
                    self._event(conn, "buffer_noop", trajectory.trajectory_id,
                                {"reason": "zero_frozen_advantage"})
                return None
            now = time.time()
            conn.execute(
                "INSERT INTO updates(update_id,phase,policy_before_version,"
                "total_action_tokens,optimizer_cursor,payload,created_at,transitioned_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (update_id, batch.phase, policy_before.version, batch.total_action_tokens,
                 batch.optimizer_cursor, canonical_json(batch), now, now),
            )
            for position, trajectory in enumerate(selected):
                trajectory_id = trajectory.trajectory_id
                conn.execute(
                    "INSERT INTO update_trajectories(update_id,trajectory_id,position,"
                    "action_token_count,raw_advantage,frozen_advantage,behavior_version) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (update_id, trajectory_id, position, trajectory.action_token_count,
                     batch.raw_advantages[trajectory_id], batch.frozen_advantages[trajectory_id],
                     trajectory.behavior_adapter.version),
                )
                conn.execute(
                    "UPDATE buffer_entries SET state='RESERVED',update_id=?,transitioned_at=? "
                    "WHERE trajectory_id=? AND state='BUFFERED'",
                    (update_id, now, trajectory_id),
                )
                self._put_trajectory(conn, replace(trajectory, status="RESERVED"))
            self._event(conn, "update_reserved", update_id, {
                "trajectories": list(batch.trajectory_ids),
                "action_tokens": batch.total_action_tokens,
                "token_mean": batch.token_mean, "token_std": batch.token_std,
            })
            return batch

    def get_update(self, update_id: str) -> UpdateBatch:
        with self._lock:
            return self._get_update(self._conn, update_id)

    def set_update_phase(self, update_id: str, phase: str,
                         optimizer_cursor: int | None = None) -> UpdateBatch:
        order = {"RESERVED": 0, "LEARNING": 1, "OPTIMIZED": 2, "EXPORTED": 3,
                 "REGISTERED": 4, "ACTIVATED": 5, "CONSUMED": 6}
        if phase not in order:
            raise ValueError("invalid update phase")
        with self.transaction() as conn:
            batch = self._get_update(conn, update_id)
            cursor = batch.optimizer_cursor if optimizer_cursor is None else optimizer_cursor
            if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
                raise ValueError("optimizer_cursor must be non-negative")
            if order[phase] < order[batch.phase] or cursor < batch.optimizer_cursor:
                raise ValueError("update phase and optimizer cursor are monotonic")
            if phase == "CONSUMED":
                raise RuntimeError("use commit_update/mark_consumed for atomic buffer consumption")
            updated = replace(batch, phase=phase, optimizer_cursor=cursor)
            conn.execute(
                "UPDATE updates SET phase=?,optimizer_cursor=?,payload=?,transitioned_at=? "
                "WHERE update_id=?",
                (phase, cursor, canonical_json(updated), time.time(), update_id),
            )
            self._event(conn, "update_phase", update_id, {
                "phase": phase, "optimizer_cursor": cursor,
            })
            return updated

    def assert_update_lag(self, update_id: str, current_policy_version: int,
                          max_policy_lag: int) -> None:
        """Trainer calls this at every optimizer boundary before using a call."""
        batch = self.get_update(update_id)
        stale: list[str] = []
        for trajectory in self.get_trajectories(batch.trajectory_ids):
            for call in trajectory.calls:
                lag = current_policy_version - call.behavior_adapter.version
                if lag < 0:
                    raise ValueError("current policy predates a behavior adapter")
                if lag > max_policy_lag:
                    stale.append(trajectory.trajectory_id)
                    break
        if stale:
            raise StaleTrajectoryError(update_id, stale)

    @staticmethod
    def _kl_payload(segment_key: str, policy_before: BehaviorAdapter,
                    policy_after: BehaviorAdapter, kl_value: float, token_count: int,
                    estimator: str, version_chain: Sequence[BehaviorAdapter]) -> dict[str, Any]:
        if not isinstance(segment_key, str) or not segment_key.strip():
            raise ValueError("segment_key is required")
        if not isinstance(policy_before, BehaviorAdapter) or not isinstance(
                policy_after, BehaviorAdapter):
            raise TypeError("KL policies must be BehaviorAdapter records")
        if policy_after.version <= policy_before.version:
            raise ValueError("KL policy versions must increase")
        if policy_after.sha256 == policy_before.sha256:
            raise ValueError("a publishable KL segment requires an adapter hash change")
        if isinstance(kl_value, bool) or not isinstance(kl_value, (int, float)) \
                or not math.isfinite(float(kl_value)) or kl_value < 0:
            raise ValueError("kl_value must be finite and non-negative")
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
            raise ValueError("KL token_count must be positive")
        if not isinstance(estimator, str) or not estimator.strip():
            raise ValueError("KL estimator is required")
        chain = tuple(version_chain)
        if len(chain) < 2 or any(not isinstance(item, BehaviorAdapter) for item in chain):
            raise ValueError("version_chain must contain at least two BehaviorAdapters")
        if chain[0] != policy_before or chain[-1] != policy_after:
            raise ValueError("version_chain endpoints do not match the KL policies")
        if any(chain[index].version >= chain[index + 1].version
               for index in range(len(chain) - 1)):
            raise ValueError("version_chain must be strictly increasing")
        return {
            "segment_key": segment_key,
            "policy_before": policy_before.to_dict(),
            "policy_after": policy_after.to_dict(),
            "kl_value": float(kl_value),
            "token_count": token_count,
            "estimator": estimator,
            "version_chain": [item.to_dict() for item in chain],
        }

    def _append_kl(self, conn: sqlite3.Connection, *, update_id: str | None,
                   **payload: Any) -> int:
        data = self._kl_payload(**payload)
        existing = conn.execute(
            "SELECT segment_id,policy_before,policy_after,kl_value,token_count,estimator,"
            "version_chain FROM kl_segments WHERE segment_key=?", (data["segment_key"],)
        ).fetchone()
        if existing is not None:
            stored = {
                "segment_key": data["segment_key"],
                "policy_before": json.loads(existing[1]),
                "policy_after": json.loads(existing[2]),
                "kl_value": existing[3], "token_count": existing[4],
                "estimator": existing[5], "version_chain": json.loads(existing[6]),
            }
            if canonical_json(stored) != canonical_json(data):
                raise ValueError("KL segment key already has different content")
            return int(existing[0])
        cursor = conn.execute(
            "INSERT INTO kl_segments(segment_key,update_id,policy_before,policy_after,"
            "kl_value,token_count,estimator,version_chain,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (data["segment_key"], update_id, canonical_json(data["policy_before"]),
             canonical_json(data["policy_after"]), data["kl_value"], data["token_count"],
             data["estimator"], canonical_json(data["version_chain"]), time.time()),
        )
        self._event(conn, "kl_appended", data["segment_key"], {
            "kl_value": data["kl_value"], "token_count": data["token_count"],
        })
        return int(cursor.lastrowid)

    def append_kl_segment(self, segment_key: str, *, policy_before: BehaviorAdapter,
                          policy_after: BehaviorAdapter, kl_value: float,
                          token_count: int, estimator: str,
                          version_chain: Sequence[BehaviorAdapter]) -> int:
        """Persist an SFT/distillation KL segment not associated with an RL update."""
        with self.transaction() as conn:
            return self._append_kl(
                conn, update_id=None, segment_key=segment_key,
                policy_before=policy_before, policy_after=policy_after,
                kl_value=kl_value, token_count=token_count, estimator=estimator,
                version_chain=version_chain,
            )

    def _consume(self, conn: sqlite3.Connection, update_id: str) -> UpdateBatch:
        batch = self._get_update(conn, update_id)
        if batch.phase == "CONSUMED":
            return batch
        if batch.phase not in ("OPTIMIZED", "EXPORTED", "REGISTERED", "ACTIVATED"):
            raise RuntimeError("only a post-optimizer update can be consumed")
        if conn.execute(
                "SELECT 1 FROM kl_segments WHERE update_id=?", (update_id,)).fetchone() is None:
            raise RuntimeError("update consumption requires its committed KL segment")
        now = time.time()
        for trajectory_id in batch.trajectory_ids:
            trajectory = self._get_trajectory(conn, trajectory_id)
            conn.execute(
                "UPDATE buffer_entries SET state='CONSUMED',transitioned_at=? "
                "WHERE trajectory_id=? AND update_id=? AND state='RESERVED'",
                (now, trajectory_id, update_id),
            )
            self._put_trajectory(conn, replace(trajectory, status="CONSUMED"))
        consumed = replace(batch, phase="CONSUMED")
        conn.execute(
            "UPDATE updates SET phase='CONSUMED',payload=?,transitioned_at=? WHERE update_id=?",
            (canonical_json(consumed), now, update_id),
        )
        self._event(conn, "update_consumed", update_id)
        return consumed

    def commit_update(self, update_id: str, *, policy_after: BehaviorAdapter,
                      kl_value: float, token_count: int, estimator: str,
                      version_chain: Sequence[BehaviorAdapter] | None = None) -> UpdateBatch:
        """Atomically append update KL and consume its complete trajectory batch."""
        with self.transaction() as conn:
            batch = self._get_update(conn, update_id)
            chain = tuple(version_chain or (batch.policy_before, policy_after))
            self._append_kl(
                conn, update_id=update_id, segment_key=f"update:{update_id}",
                policy_before=batch.policy_before, policy_after=policy_after,
                kl_value=kl_value, token_count=token_count, estimator=estimator,
                version_chain=chain,
            )
            return self._consume(conn, update_id)

    def mark_consumed(self, update_id: str) -> UpdateBatch:
        with self.transaction() as conn:
            return self._consume(conn, update_id)

    def list_updates(self, phases: Sequence[str] | None = None) -> tuple[UpdateBatch, ...]:
        with self._lock:
            if phases:
                placeholders = ",".join("?" for _ in phases)
                rows = self._conn.execute(
                    f"SELECT payload FROM updates WHERE phase IN ({placeholders}) "
                    "ORDER BY created_at,update_id", tuple(phases)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT payload FROM updates ORDER BY created_at,update_id"
                ).fetchall()
        return tuple(UpdateBatch.from_dict(json.loads(row[0])) for row in rows)

    def max_stream_sequence(self) -> int:
        """Highest stream_sequence ever issued (trajectories and observations)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT COALESCE(MAX(stream_sequence),0) FROM trajectories"
            ).fetchone()
            peak = int(rows[0])
            try:
                obs = self._conn.execute(
                    "SELECT COALESCE(MAX(stream_sequence),0) FROM value_observations"
                ).fetchone()
                peak = max(peak, int(obs[0]))
            except Exception:
                pass
            return peak

    def buffer_state(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT state,COUNT(*) AS n FROM buffer_entries GROUP BY state"
            ).fetchall()
        return {str(row["state"]): int(row["n"]) for row in rows}

    def recover(self) -> dict[str, Any]:
        """Validate replayed state and return exact resume points without resetting them."""
        with self._lock:
            quick = self._conn.execute("PRAGMA quick_check").fetchone()[0]
            if quick != "ok":
                raise RuntimeError(f"SQLite recovery check failed: {quick}")
            foreign = self._conn.execute("PRAGMA foreign_key_check").fetchall()
            if foreign:
                raise RuntimeError("SQLite foreign-key recovery check failed")
            for batch in self.list_updates(("RESERVED", "LEARNING", "OPTIMIZED",
                                               "EXPORTED", "REGISTERED", "ACTIVATED")):
                trajectories = self.get_trajectories(batch.trajectory_ids)
                validate_update_batch(batch, trajectories)
            counts = self.buffer_state()
            latest_kl = self._conn.execute(
                "SELECT COALESCE(MAX(segment_id),0) FROM kl_segments"
            ).fetchone()[0]
            values = self._conn.execute("SELECT COUNT(*) FROM value_records").fetchone()[0]
        return {
            "schema_version": SCHEMA_VERSION,
            "buffer": counts,
            "resume_updates": [batch.to_dict() for batch in self.list_updates(
                ("RESERVED", "LEARNING", "OPTIMIZED", "EXPORTED", "REGISTERED", "ACTIVATED")
            )],
            "value_records": int(values),
            "kl_cursor": int(latest_kl),
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.close()
            self._closed = True
            self._private_files()

    def __enter__(self) -> "TrajectoryLedger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
