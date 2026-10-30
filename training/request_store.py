"""Durable generation/request acknowledgements for one private learner run.

The database is authoritative; model_calls.jsonl is a materialized view of only
consumed results. A generated or delivered response is not evidence that the
agent accepted it. Pending GPU work cannot be resumed from this database.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid


class RequestConflict(ValueError):
    pass


class RequestUnavailable(RuntimeError):
    pass


class RequestStore:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.run_dir / "requests.sqlite"
        self._lock = threading.RLock()
        self._owner = (self.run_dir / "requests.lock").open("a+b")
        os.chmod(self.run_dir / "requests.lock", 0o600)
        try:
            fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._owner.close()
            raise RuntimeError("Another request endpoint owns this run directory") from None
        try:
            self.db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            os.chmod(self.path, 0o600)
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    rollout_id TEXT NOT NULL,
                    policy_version INTEGER NOT NULL,
                    idempotency_key TEXT,
                    request_hash TEXT NOT NULL,
                    call_index INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    generated_at REAL,
                    delivered_at REAL,
                    consumed_at REAL,
                    finished_at REAL,
                    delivery_attempts INTEGER NOT NULL DEFAULT 0,
                    delivery_failures INTEGER NOT NULL DEFAULT 0,
                    ack_count INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    UNIQUE(rollout_id, policy_version, idempotency_key),
                    UNIQUE(rollout_id, policy_version, call_index)
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    time_unix REAL NOT NULL,
                    state TEXT NOT NULL,
                    detail TEXT
                );
                CREATE TABLE IF NOT EXISTS cancellation_keys (
                    rollout_id TEXT NOT NULL,
                    policy_version INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    reason TEXT NOT NULL,
                    PRIMARY KEY(rollout_id, policy_version, idempotency_key)
                );
            """)
            with self.db:
                rows = self.db.execute("SELECT request_id FROM requests WHERE state IN ('queued','generating')").fetchall()
                for row in rows:
                    self._transition(row["request_id"], "failed", "interrupted_before_generation_committed")
        except BaseException:
            if hasattr(self, "db"):
                self.db.close()
            self._owner.close()
            raise

    @staticmethod
    def request_hash(request):
        payload = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        result = dict(row)
        payload = result.pop("result_json")
        result["result"] = json.loads(payload) if payload else None
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def _transition(self, request_id, state, error=None):
        now = time.time()
        self.db.execute("UPDATE requests SET state=?, error=?, finished_at=? WHERE request_id=?",
                        (state, error, now if state in {"cancelled", "failed", "consumed"} else None, request_id))
        self.db.execute("INSERT INTO events(request_id,time_unix,state,detail) VALUES(?,?,?,?)",
                        (request_id, now, state, error))

    def get(self, request_id, rollout_id=None, policy_version=None):
        with self._lock:
            row = self._decode(self.db.execute("SELECT * FROM requests WHERE request_id=?", (request_id,)).fetchone())
        if row is None or (rollout_id is not None and row["rollout_id"] != rollout_id):
            raise KeyError("Unknown request identity")
        if policy_version is not None and row["policy_version"] != policy_version:
            raise RequestConflict("Request belongs to another policy version")
        return row

    def status(self, request_id, rollout_id=None, policy_version=None):
        """Read lifecycle metadata without decoding a complete token response."""
        with self._lock:
            row = self.db.execute(
                "SELECT request_id,rollout_id,policy_version,call_index,state,error "
                "FROM requests WHERE request_id=?", (request_id,)).fetchone()
            result = dict(row) if row is not None else None
        if result is None or (rollout_id is not None and result["rollout_id"] != rollout_id):
            raise KeyError("Unknown request identity")
        if policy_version is not None and result["policy_version"] != policy_version:
            raise RequestConflict("Request belongs to another policy version")
        return result

    def rollout_states(self, identity, policy_version):
        """Cancellation/diagnostics never need to deserialize generated tokens."""
        with self._lock:
            return [dict(row) for row in self.db.execute(
                "SELECT request_id,rollout_id,policy_version,call_index,state,error,"
                "(result_json IS NOT NULL) AS has_result FROM requests "
                "WHERE rollout_id=? AND policy_version=? ORDER BY call_index", (identity, policy_version)).fetchall()]

    def create(self, rollout_id, policy_version, key, request_hash, metadata=None):
        if key is not None and (not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", key)):
            raise ValueError("Idempotency-Key must contain 1..200 safe ASCII characters")
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("Invalid policy version")
        with self._lock, self.db:
            existing = None
            if key is not None:
                existing = self.db.execute(
                    "SELECT * FROM requests WHERE rollout_id=? AND policy_version=? AND idempotency_key=?",
                    (rollout_id, policy_version, key)).fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    raise RequestConflict("Idempotency key was already used with a different request body")
                return self._decode(existing), False
            tombstone = (self.db.execute(
                "SELECT reason FROM cancellation_keys WHERE rollout_id=? AND policy_version=? AND idempotency_key=?",
                (rollout_id, policy_version, key)).fetchone() if key is not None else None)
            call_index = self.db.execute(
                "SELECT COALESCE(MAX(call_index),-1)+1 FROM requests WHERE rollout_id=? AND policy_version=?",
                (rollout_id, policy_version)).fetchone()[0]
            request_id, now = uuid.uuid4().hex, time.time()
            self.db.execute("""INSERT INTO requests
                (request_id,rollout_id,policy_version,idempotency_key,request_hash,call_index,state,created_at,metadata_json)
                VALUES(?,?,?,?,?,?,'queued',?,?)""",
                (request_id, rollout_id, policy_version, key, request_hash, call_index, now,
                 json.dumps(metadata or {}, allow_nan=False)))
            self.db.execute("INSERT INTO events(request_id,time_unix,state) VALUES(?,?,'queued')", (request_id, now))
            if tombstone is not None:
                self._transition(request_id, "cancelled", tombstone["reason"])
            return self.get(request_id), True

    def generating(self, request_id, metadata):
        with self._lock, self.db:
            row = self.get(request_id)
            if row["state"] != "queued":
                return row
            details = {**row["metadata"], **metadata}
            self.db.execute("UPDATE requests SET metadata_json=? WHERE request_id=?",
                            (json.dumps(details, allow_nan=False), request_id))
            self._transition(request_id, "generating")
            return self.get(request_id)

    def generated(self, request_id, result):
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False)
        with self._lock, self.db:
            row = self.get(request_id)
            if row["result"] is not None:
                raise RequestConflict("A request result may only be committed once")
            self.db.execute("UPDATE requests SET result_json=?,generated_at=? WHERE request_id=?",
                            (payload, time.time(), request_id))
            if row["state"] in {"queued", "generating"}:
                self._transition(request_id, "generated")
            # A cancellation can race the completed GPU step. Preserve actual
            # generated work without resurrecting or delivering that request.
            return self.get(request_id)

    def delivered(self, request_id, *, success=True):
        with self._lock, self.db:
            row = self.get(request_id)
            if success:
                self.db.execute("UPDATE requests SET delivery_attempts=delivery_attempts+1, delivered_at=? WHERE request_id=?",
                                (time.time(), request_id))
                if row["state"] == "generated":
                    self._transition(request_id, "delivered")
            else:
                self.db.execute("UPDATE requests SET delivery_failures=delivery_failures+1 WHERE request_id=?", (request_id,))

    def acknowledge(self, request_id, rollout_id, policy_version):
        with self._lock, self.db:
            row = self.get(request_id, rollout_id, policy_version)
            if row["state"] not in {"generated", "delivered", "consumed"} or row["result"] is None:
                raise RequestConflict("Only a completed, uncancelled response can be acknowledged")
            changed = row["state"] != "consumed"
            if changed:
                self._transition(request_id, "consumed")
                self.db.execute("UPDATE requests SET consumed_at=? WHERE request_id=?", (time.time(), request_id))
            self.db.execute("UPDATE requests SET ack_count=ack_count+1 WHERE request_id=?", (request_id,))
            return self.get(request_id), changed

    def fail(self, request_id, reason):
        with self._lock, self.db:
            row = self.get(request_id)
            if row["state"] in {"queued", "generating"}:
                self._transition(request_id, "failed", reason)
            return self.get(request_id)

    def cancel(self, request_id, rollout_id=None, reason="cancelled"):
        with self._lock, self.db:
            row = self.get(request_id, rollout_id)
            if row["state"] not in {"consumed", "cancelled", "failed"}:
                self._transition(request_id, "cancelled", reason)
            return self.get(request_id)

    def cancel_by_key(self, rollout_id, policy_version, key, reason="client_cancelled_by_key"):
        """Persist cancellation even when the completion request has not arrived.

        Tombstones share the request's exact rollout/policy/key namespace. A
        consumed result remains replayable; cancellation never revokes its ACK.
        """
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,200}", key):
            raise ValueError("Idempotency-Key must contain 1..200 safe ASCII characters")
        if type(policy_version) is not int or policy_version < 0:
            raise ValueError("Invalid policy version")
        with self._lock, self.db:
            row = self.db.execute(
                "SELECT request_id,state FROM requests WHERE rollout_id=? AND policy_version=? AND idempotency_key=?",
                (rollout_id, policy_version, key)).fetchone()
            if row is None or row["state"] != "consumed":
                self.db.execute(
                    "INSERT OR IGNORE INTO cancellation_keys(rollout_id,policy_version,idempotency_key,created_at,reason) "
                    "VALUES(?,?,?,?,?)", (rollout_id, policy_version, key, time.time(), reason))
                if row is not None and row["state"] not in {"cancelled", "failed"}:
                    self._transition(row["request_id"], "cancelled", reason)
            state = self.status(row["request_id"])["state"] if row is not None else "cancelled"
            return {"rollout_id": rollout_id, "policy_version": policy_version,
                    "idempotency_key": key, "request_id": row["request_id"] if row is not None else None,
                    "state": state}

    def rollout_rows(self, identity, policy_version):
        with self._lock:
            return [self._decode(row) for row in self.db.execute(
                "SELECT * FROM requests WHERE rollout_id=? AND policy_version=? ORDER BY call_index",
                (identity, policy_version)).fetchall()]

    @staticmethod
    def call_record(row):
        result = row["result"]
        if row["state"] != "consumed" or result is None:
            raise RequestUnavailable("Only explicitly consumed results may enter training calls")
        record = {key: result[key] for key in ("prompt_ids", "completion_ids", "behavior_logprobs", "finish_reason")}
        for key in ("sampling", "requested_max_tokens", "effective_max_tokens", "context_limited",
                    "generation_elapsed_seconds", "first_scores_seconds", "first_token_seconds",
                    "reused_prefix_tokens", "profiled_tokens", "generation_backend", "timing_semantics"):
            if key in result:
                record[key] = result[key]
        metadata = row["metadata"]
        record.update(request_id=row["request_id"], rollout_id=row["rollout_id"], policy_version=row["policy_version"],
                      call_index=row["call_index"], delivery_state="consumed", consumed_at=row["consumed_at"],
                      queue_wait_seconds=result.get("queue_wait_seconds", metadata.get("scheduler_queue_wait_seconds")),
                      scheduler_queue_wait_seconds=metadata.get("scheduler_queue_wait_seconds"),
                      preparation_elapsed_seconds=metadata.get("preparation_elapsed_seconds"))
        return record

    def consumed_calls(self, identity, policy_version):
        return [self.call_record(row) for row in self.rollout_rows(identity, policy_version) if row["state"] == "consumed"]

    def close(self):
        with self._lock:
            if self.db is not None:
                self.db.close()
                self.db = None
                self._owner.close()
