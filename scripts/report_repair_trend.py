#!/usr/bin/env python3
"""Read-only report of official repair outcomes by training step.

The ledger does not store a "step" column on trajectories.  This tool defines a
trajectory's step as the behaviour policy version it was sampled with
(``trajectories.behavior_version``: step 0 is the initial policy, step N is the
policy after N completed optimizer steps).  The optimizer update that later
consumed the trajectory is resolved through ``update_trajectories`` and shown
separately; a trajectory belongs to at most one update.  Comparing outcomes by
sampling step avoids attributing an update's outcome to a trajectory that was
generated before that update ran.

An "official P verdict" is a reward record whose ``P`` is present.  A record
with ``resolved`` but no ``P`` (for example a missing baseline) is reported as a
resolved value with P missing, never as P=0.  Missing evidence is never imputed:
means are computed only over non-None values and a record with P=None is shown
as ``-`` with the missing count.

The database is opened read-only.  A WAL-mode database whose ``-wal`` file is
absent or empty is opened with ``immutable=1`` so SQLite creates no ``-shm`` or
``-wal`` sidecar in the run directory.  A database with a non-empty ``-wal`` is
refused: reading it safely would either create sidecars or miss committed
frames, so make a consistent copy first (``sqlite3 ... ".backup copy.db"`` or
the Python backup API) and report on the copy.  Nothing is written to the run
directory.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.trajectory.schema import RewardRecord


_REQUIRED_TABLES = ("trajectories", "rewards", "updates", "update_trajectories")
_EXIT_DATABASE = 3


@dataclass(frozen=True)
class TrajectoryEvidence:
    trajectory_id: str
    stream_sequence: int
    task_id: str
    step: int
    update_ids: tuple[str, ...]
    status: str
    reward: RewardRecord | None
    reward_error: str | None
    positive_signal: bool | None


@dataclass(frozen=True)
class Selector:
    """A comparison side: a sampling step or a frozen optimizer update."""

    kind: str
    value: int | str

    def label(self) -> str:
        if self.kind == "step":
            return f"step {self.value}"
        return f"update {self.value}"

    def matches(self, record: TrajectoryEvidence) -> bool:
        if self.kind == "step":
            return record.step == self.value
        return self.value in record.update_ids


class ReasonIndex:
    """Assign stable short codes to distinct exclusion-reason strings."""

    def __init__(self) -> None:
        self._codes: dict[str, str] = {}
        self._counts: Counter[str] = Counter()

    def code(self, reason: str) -> str:
        if reason not in self._codes:
            self._codes[reason] = f"R{len(self._codes) + 1}"
        self._counts[reason] += 1
        return self._codes[reason]

    def entries(self) -> list[tuple[str, int, str]]:
        return [
            (self._codes[reason], self._counts[reason], reason)
            for reason in self._codes
        ]


def _select(records: Sequence[TrajectoryEvidence], selector: Selector) \
        -> list[TrajectoryEvidence]:
    return [record for record in records if selector.matches(record)]


def _group_by_task(records: Iterable[TrajectoryEvidence]) \
        -> dict[str, list[TrajectoryEvidence]]:
    grouped: dict[str, list[TrajectoryEvidence]] = defaultdict(list)
    for record in records:
        grouped[record.task_id].append(record)
    return dict(grouped)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _p_values(records: Sequence[TrajectoryEvidence]) -> list[float]:
    return [record.reward.P for record in records
            if record.reward is not None and record.reward.P is not None]


def _g_values(records: Sequence[TrajectoryEvidence]) -> list[float]:
    return [record.reward.G for record in records
            if record.reward is not None and record.reward.G is not None]


def _validity_values(records: Sequence[TrajectoryEvidence]) -> list[float]:
    return [record.reward.validity for record in records
            if record.reward is not None and record.reward.validity is not None]


def _resolved_known(records: Sequence[TrajectoryEvidence]) -> list[bool]:
    return [record.reward.resolved for record in records
            if record.reward is not None and record.reward.resolved is not None]


def _mean_text(values: Sequence[float], total: int) -> str:
    if not values:
        return f"- (missing {total})"
    text = f"{_mean(values):.3f} (n={len(values)}"
    missing = total - len(values)
    if missing:
        text += f", miss={missing}"
    return text + ")"


def _resolved_text(records: Sequence[TrajectoryEvidence]) -> str:
    known = _resolved_known(records)
    if not known:
        return "no verdict"
    text = f"{sum(known)}/{len(known)}"
    unknown = len(records) - len(known)
    if unknown:
        text += f" (+{unknown} n/a)"
    return text


def _eligible_text(records: Sequence[TrajectoryEvidence]) -> str:
    eligible = sum(1 for record in records
                   if record.reward is not None and record.reward.eligible)
    return f"{eligible}/{len(records)}"


def _signal_text(records: Sequence[TrajectoryEvidence]) -> str:
    true_count = sum(1 for record in records if record.positive_signal is True)
    false_count = sum(1 for record in records if record.positive_signal is False)
    absent_count = len(records) - true_count - false_count
    if not true_count and not false_count and not absent_count:
        return "-"
    if len(records) == 1:
        if true_count:
            return "true"
        if false_count:
            return "false"
        return "-"
    return f"{true_count}T/{false_count}F/{absent_count}-"


def _record_reason(record: TrajectoryEvidence) -> str | None:
    if record.reward is not None:
        return record.reward.exclusion_reason
    return record.reward_error


def _cell_reasons(records: Sequence[TrajectoryEvidence]) -> list[str]:
    reasons: list[str] = []
    for record in records:
        reason = _record_reason(record)
        if reason is not None and reason not in reasons:
            reasons.append(reason)
    return reasons


def _updates_text(records: Sequence[TrajectoryEvidence]) -> str:
    update_ids = sorted({update_id for record in records
                         for update_id in record.update_ids})
    return ",".join(update_ids) if update_ids else "-"


def _rate(records: Sequence[TrajectoryEvidence]) -> float | None:
    known = _resolved_known(records)
    return sum(known) / len(known) if known else None


def _rate_text(records: Sequence[TrajectoryEvidence]) -> str:
    known = _resolved_known(records)
    if not known:
        return "no verdict"
    return f"{sum(known)}/{len(known)}"


def _delta_text(value: float | None) -> str:
    return "-" if value is None else f"{value:+.3f}"


def _wrap_list(prefix: str, items: Sequence[str], *, width: int = 100) -> None:
    if not items:
        return
    if len(items) <= 8:
        body = ", ".join(items)
        print(textwrap.fill(prefix + body, width=width,
                            subsequent_indent=" " * 4))
        return
    print(textwrap.fill(prefix + f"{len(items)} tasks: " + ", ".join(items),
                        width=width, subsequent_indent=" " * 4))


def _is_wal_database(db_path: Path) -> bool:
    """Read the SQLite header write/read version bytes (2 means WAL)."""
    try:
        with open(db_path, "rb") as handle:
            header = handle.read(20)
    except OSError:
        return False
    return len(header) >= 20 and header[18] == 2 and header[19] == 2


def _wal_has_frames(db_path: Path) -> bool:
    wal_path = Path(str(db_path) + "-wal")
    try:
        return wal_path.is_file() and wal_path.stat().st_size > 0
    except OSError:
        return True


def _read_only_uri(db_path: Path) -> str | None:
    """Return a URI that cannot create sidecar files, or None if unsafe."""
    uri = f"{db_path.as_uri()}?mode=ro"
    if _is_wal_database(db_path):
        if _wal_has_frames(db_path):
            return None
        uri += "&immutable=1"
    return uri


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")
    return {str(row[0]) for row in rows}


def _task_id_from_value(value: Any) -> str | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, Mapping):
        task_id = value.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            return task_id
    return None


def _resolve_task_id(row: sqlite3.Row, notes: list[str]) -> str:
    task_id = _task_id_from_value(row["task_key"])
    if task_id is not None:
        return task_id
    try:
        payload = json.loads(row["trajectory_payload"])
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, Mapping):
        task_id = _task_id_from_value(payload.get("task_key"))
        if task_id is not None:
            notes.append(
                f"task_id recovered from the trajectory payload for "
                f"{row['trajectory_id']} because the task_key column did not parse")
            return task_id
    notes.append(
        f"task_id unresolved for {row['trajectory_id']} (task_key and payload unreadable)")
    return f"<unresolved:{row['trajectory_id']}>"


def _load_audits(connection: sqlite3.Connection, tables: set[str],
                 notes: list[str]) -> dict[str, Mapping[str, Any]]:
    audits: dict[str, Mapping[str, Any]] = {}
    if "audit_events" not in tables:
        notes.append(
            "audit_events table is absent: positive_repair_signal is reported "
            "as absent for every trajectory")
        return audits
    invalid = 0
    for row in connection.execute(
            "SELECT subject_id,payload FROM audit_events "
            "WHERE event_type='reward_audit' ORDER BY event_id"):
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError):
            invalid += 1
            continue
        if isinstance(payload, Mapping) and isinstance(row["subject_id"], str):
            audits[row["subject_id"]] = payload
    if invalid:
        notes.append(f"{invalid} reward_audit row(s) had an unparseable payload")
    return audits


def _load_records(connection: sqlite3.Connection, tables: set[str],
                  notes: list[str]) -> list[TrajectoryEvidence]:
    audits = _load_audits(connection, tables, notes)
    rows = connection.execute(
        """
        SELECT t.trajectory_id, t.stream_sequence, t.task_key, t.status,
               t.behavior_version, t.payload AS trajectory_payload,
               rw.payload AS reward_payload,
               ut.update_id
        FROM trajectories t
        LEFT JOIN rewards rw ON rw.trajectory_id = t.trajectory_id
        LEFT JOIN update_trajectories ut ON ut.trajectory_id = t.trajectory_id
        ORDER BY t.stream_sequence, ut.update_id
        """)
    ordered: dict[str, dict[str, Any]] = {}
    for row in rows:
        trajectory_id = str(row["trajectory_id"])
        entry = ordered.get(trajectory_id)
        if entry is None:
            entry = {
                "stream_sequence": int(row["stream_sequence"]),
                "task_id": _resolve_task_id(row, notes),
                "step": int(row["behavior_version"]),
                "update_ids": [],
                "status": str(row["status"]),
                "reward_payload": row["reward_payload"],
            }
            ordered[trajectory_id] = entry
        update_id = row["update_id"]
        if isinstance(update_id, str) and update_id not in entry["update_ids"]:
            entry["update_ids"].append(update_id)

    records: list[TrajectoryEvidence] = []
    for trajectory_id, entry in ordered.items():
        reward: RewardRecord | None = None
        reward_error: str | None = None
        if entry["reward_payload"] is None:
            reward_error = "no reward row in ledger"
        else:
            try:
                reward = RewardRecord.from_dict(json.loads(entry["reward_payload"]))
            except (TypeError, ValueError) as error:
                reward_error = (
                    f"invalid reward payload: {type(error).__name__}: {error}")
        audit = audits.get(trajectory_id)
        signal: bool | None = None
        if audit is not None:
            value = audit.get("positive_repair_signal")
            if isinstance(value, bool):
                signal = value
        records.append(TrajectoryEvidence(
            trajectory_id=trajectory_id,
            stream_sequence=entry["stream_sequence"],
            task_id=entry["task_id"],
            step=entry["step"],
            update_ids=tuple(entry["update_ids"]),
            status=entry["status"],
            reward=reward,
            reward_error=reward_error,
            positive_signal=signal,
        ))
    return records


def _print_header(db_path: Path, schema_version: str,
                  records: Sequence[TrajectoryEvidence], update_rows: Sequence[sqlite3.Row],
                  requested_tasks: Sequence[str] | None) -> None:
    reward_rows = sum(1 for record in records if record.reward is not None)
    p_rows = sum(1 for record in records if _p_values([record]))
    resolved_rows = sum(1 for record in records if _resolved_known([record]))
    validity_rows = sum(1 for record in records if _validity_values([record]))
    g_rows = sum(1 for record in records if _g_values([record]))
    eligible_rows = sum(1 for record in records
                        if record.reward is not None and record.reward.eligible)
    audits = sum(1 for record in records if record.positive_signal is not None)
    print("LEDGER REPORT - official repair outcomes by training step")
    print(f"database: {db_path} (opened read-only)")
    print(f"ledger schema_version: {schema_version}")
    print(
        f"trajectories: {len(records)}   rewards: {reward_rows}   "
        f"updates: {len(update_rows)}   update members: "
        f"{sum(1 for record in records if record.update_ids)}   audit signals: {audits}")
    print(
        f"coverage: P {p_rows}/{len(records)}   resolved-known {resolved_rows}/{len(records)}   "
        f"validity {validity_rows}/{len(records)}   G {g_rows}/{len(records)}   "
        f"eligible {eligible_rows}/{len(records)}")
    if requested_tasks is not None:
        print(f"task filter: {', '.join(requested_tasks)}")
    print("step = trajectories.behavior_version (policy version used to sample the "
          "trajectory); update = optimizer update that consumed it")
    print()


def _print_steps(records: Sequence[TrajectoryEvidence]) -> list[int]:
    steps = sorted({record.step for record in records})
    print("STEP SUMMARY")
    header = (f"{'step':>4}  {'trajectories':>12}  {'distinct tasks':>14}  "
              f"{'updates (members)':<28}  {'P verdicts':>10}  {'resolved known':>14}")
    print(header)
    print("-" * len(header))
    for step in steps:
        step_records = [record for record in records if record.step == step]
        tasks = {record.task_id for record in step_records}
        update_counts = Counter(update_id for record in step_records
                                for update_id in record.update_ids)
        updates = ", ".join(f"{update_id} ({count})"
                            for update_id, count in sorted(update_counts.items()))
        updates = updates or "-"
        p_verdicts = sum(1 for record in step_records if _p_values([record]))
        resolved = len(_resolved_known(step_records))
        print(f"{step:>4}  {len(step_records):>12}  {len(tasks):>14}  "
              f"{updates:<28}  {p_verdicts:>10}  {resolved:>14}")
        thin = sorted(task for task, task_records in _group_by_task(step_records).items()
                      if len(task_records) < 2)
        if thin:
            print(
                f"  WARNING: step {step} has {len(thin)}/{len(tasks)} tasks with fewer "
                f"than 2 trajectories ({len(step_records)} trajectories, {len(tasks)} "
                f"tasks); per-task means are single samples, not a trend.")
            if len(thin) <= 8:
                print(f"  tasks with N<2: {', '.join(thin)}")
        if len(step_records) < 2:
            print(f"  WARNING: step {step} has only {len(step_records)} trajectory; "
                  f"no per-step mean is representative.")
    if not steps:
        print("  (no trajectories)")
    print()
    return steps


def _print_table(records: Sequence[TrajectoryEvidence], steps: Sequence[int],
                 index: ReasonIndex) -> None:
    cells: dict[tuple[int, str], list[TrajectoryEvidence]] = defaultdict(list)
    for record in records:
        cells[(record.step, record.task_id)].append(record)
    task_width = max([len("task")] + [len(task) for _, task in cells])
    print("PER-TASK x PER-STEP")
    header = (
        f"{'step':>4}  {'task':<{task_width}}  {'N':>3}  {'resolved':<16}  "
        f"{'mean P':<24}  {'mean G':<24}  {'mean validity':<24}  {'eligible':>8}  "
        f"{'signal':<12}  {'updates':<20}  reasons")
    print(header)
    print("-" * len(header))
    for step in steps:
        tasks = sorted(task for cell_step, task in cells if cell_step == step)
        for task in tasks:
            cell = cells[(step, task)]
            reasons = _cell_reasons(cell)
            codes = ",".join(index.code(reason) for reason in reasons) or "-"
            print(
                f"{step:>4}  {task:<{task_width}}  {len(cell):>3}  "
                f"{_resolved_text(cell):<16}  "
                f"{_mean_text(_p_values(cell), len(cell)):<24}  "
                f"{_mean_text(_g_values(cell), len(cell)):<24}  "
                f"{_mean_text(_validity_values(cell), len(cell)):<24}  "
                f"{_eligible_text(cell):>8}  "
                f"{_signal_text(cell):<12}  {_updates_text(cell):<20}  {codes}")
        if step != steps[-1]:
            print()
    print()
    print("resolved k/n = resolved=True among records whose resolved is not None; "
          "no verdict = no resolved value")
    print("mean P, mean G and mean validity use only non-None values; '-' is "
          "missing evidence, never zero")
    print()


def _print_p_verdicts(records: Sequence[TrajectoryEvidence]) -> None:
    verdicts = [record for record in records if _p_values([record])]
    verdicts.sort(key=lambda record: (record.step, record.task_id, record.stream_sequence))
    print("OFFICIAL P VERDICTS (P present; the only records usable for a P delta)")
    if not verdicts:
        print("  none")
        print()
        return
    for record in verdicts:
        reward = record.reward
        assert reward is not None and reward.P is not None
        resolved = "-" if reward.resolved is None else str(reward.resolved)
        g_value = "-" if reward.G is None else f"{reward.G:.3f}"
        validity = "-" if reward.validity is None else f"{reward.validity:.3f}"
        signal = "-" if record.positive_signal is None else str(record.positive_signal).lower()
        update = ",".join(record.update_ids) or "-"
        print(
            f"  step {record.step}  {record.task_id}  P={reward.P:.3f}  "
            f"G={g_value}  validity={validity}  resolved={resolved}  "
            f"eligible={'yes' if reward.eligible else 'no'}  signal={signal}")
        print(
            f"          trajectory {record.trajectory_id}  status={record.status}  "
            f"update={update}")
    print()
    print(f"  {len(verdicts)} P verdict(s); records with P=None are not counted here")
    print()


def _print_reasons(index: ReasonIndex) -> None:
    entries = index.entries()
    print("EXCLUSION REASONS")
    if not entries:
        print("  none")
        print()
        return
    for code, count, reason in entries:
        display = reason.replace(";", "; ")
        wrapped = textwrap.wrap(display, width=100)
        print(f"  {code} [{count} record(s)] {wrapped[0]}")
        for line in wrapped[1:]:
            print(f"      {line}")
    print()


def _print_notes(notes: Sequence[str]) -> None:
    print("DATA NOTES")
    if not notes:
        print("  none")
    else:
        for note in notes:
            lines = textwrap.wrap(note, width=98)
            print(f"  - {lines[0]}")
            for line in lines[1:]:
                print(f"    {line}")
    print()


def _resolve_selector(token: str, known_updates: set[str]) -> Selector | None:
    if token in known_updates:
        return Selector("update", token)
    try:
        value = int(token)
    except ValueError:
        return None
    if value < 0:
        return None
    return Selector("step", value)


def _print_side(label: str, selector: Selector,
                records: Sequence[TrajectoryEvidence]) -> None:
    if not records:
        if selector.kind == "step":
            reason = f"no trajectory has behavior_version {selector.value}"
        else:
            reason = f"update {selector.value} has no matching trajectory"
        print(f"  {label:<5} ({selector.label()}): ABSENT - {reason}")
        return
    tasks = _group_by_task(records)
    update_counts = Counter(update_id for record in records
                            for update_id in record.update_ids)
    updates = ", ".join(f"{update_id} ({count})"
                        for update_id, count in sorted(update_counts.items())) or "none"
    print(f"  {label:<5} ({selector.label()}): {len(records)} trajectories, "
          f"{len(tasks)} tasks, updates: {updates}")
    thin = sorted(task for task, task_records in tasks.items()
                  if len(task_records) < 2)
    if thin:
        print(f"    WARNING: {len(thin)}/{len(tasks)} tasks have fewer than 2 "
              f"trajectories; their per-task values are single samples.")


def _print_comparison(records: Sequence[TrajectoryEvidence],
                      left_selector: Selector, right_selector: Selector,
                      index: ReasonIndex, notes: list[str]) -> None:
    left = _select(records, left_selector)
    right = _select(records, right_selector)
    print(f"COMPARISON {left_selector.label()} vs {right_selector.label()}")
    _print_side("left", left_selector, left)
    _print_side("right", right_selector, right)
    print()

    left_by_task = _group_by_task(left)
    right_by_task = _group_by_task(right)
    left_tasks = set(left_by_task)
    right_tasks = set(right_by_task)
    common = sorted(left_tasks & right_tasks)
    comparable_p = [task for task in common
                    if _p_values(left_by_task[task])
                    and _p_values(right_by_task[task])]
    comparable_resolved = [task for task in common
                           if _resolved_known(left_by_task[task])
                           and _resolved_known(right_by_task[task])]
    comparable = sorted(set(comparable_p) | set(comparable_resolved))
    non_comparable = [task for task in common if task not in comparable]

    overlap = {record.trajectory_id for record in left} & \
              {record.trajectory_id for record in right}
    if overlap:
        notes.append(
            f"comparison selections overlap on {len(overlap)} trajectory(ies); "
            f"the two sides are not independent samples")

    for metric, tasks in (("resolved", comparable_resolved), ("P", comparable_p)):
        print(f"  tasks with an official {metric} verdict at BOTH selectors: {len(tasks)}")
        if tasks:
            _wrap_list("    ", tasks)
        else:
            print("    none")
    print()

    if comparable:
        task_width = max([len("task")] + [len(task) for task in comparable])
        header = (
            f"  {'task':<{task_width}}  {'N_L':>3}  {'N_R':>3}  "
            f"{'resolved_L':<11}  {'resolved_R':<11}  {'d_res':>7}  "
            f"{'P_L':>6}  {'P_R':>6}  {'d_P':>7}  sample")
        print(header)
        print("  " + "-" * (len(header) - 2))
        thin: list[str] = []
        for task in comparable:
            left_task = left_by_task[task]
            right_task = right_by_task[task]
            left_rate = _rate(left_task)
            right_rate = _rate(right_task)
            left_p = _mean(_p_values(left_task))
            right_p = _mean(_p_values(right_task))
            left_resolved_text = "-" if left_rate is None else _rate_text(left_task)
            right_resolved_text = "-" if right_rate is None else _rate_text(right_task)
            left_p_text = "-" if left_p is None else f"{left_p:.3f}"
            right_p_text = "-" if right_p is None else f"{right_p:.3f}"
            single = min(len(left_task), len(right_task)) < 2
            if single:
                thin.append(task)
            print(
                f"  {task:<{task_width}}  {len(left_task):>3}  {len(right_task):>3}  "
                f"{left_resolved_text:<11}  {right_resolved_text:<11}  "
                f"{_delta_text(None if left_rate is None or right_rate is None else right_rate - left_rate):>7}  "
                f"{left_p_text:>6}  {right_p_text:>6}  "
                f"{_delta_text(None if left_p is None or right_p is None else right_p - left_p):>7}  "
                f"{'single' if single else 'ok'}")
        print()
        print("  d_res = resolved rate (right - left); d_P = mean P (right - left)")
        print("  each delta requires that metric on both sides; '-' is missing evidence")
        if thin:
            print(f"  WARNING: {len(thin)}/{len(comparable)} compared tasks have fewer "
                  f"than 2 trajectories on at least one side; the deltas are single "
                  f"samples, not a trend.")
            _wrap_list("  tasks with N<2 on a side: ", thin)
    else:
        print("  RESULT: no task has an official resolved or P verdict on both sides; no "
              "resolved/P delta is computed. Missing evidence is not a zero delta.")
    print()

    if non_comparable:
        print("  common tasks excluded from the delta table (neither resolved nor P "
              "has evidence on both sides):")
        for task in non_comparable:
            left_task = left_by_task[task]
            right_task = right_by_task[task]
            left_p = _mean(_p_values(left_task))
            right_p = _mean(_p_values(right_task))
            left_text = (f"P={left_p:.3f} (n={len(_p_values(left_task))})"
                         if left_p is not None else "no P verdict")
            right_text = (f"P={right_p:.3f} (n={len(_p_values(right_task))})"
                          if right_p is not None else "no P verdict")
            reasons = []
            for codes in (_cell_reasons(left_task), _cell_reasons(right_task)):
                for reason in codes:
                    code = index.code(reason)
                    if code not in reasons:
                        reasons.append(code)
            reason_text = ", ".join(reasons) or "-"
            print(f"    {task}: left resolved={_rate_text(left_task)}, {left_text}; "
                  f"right resolved={_rate_text(right_task)}, {right_text}; reasons {reason_text}")
        print()

    left_only = sorted(left_tasks - right_tasks)
    right_only = sorted(right_tasks - left_tasks)
    left_only_verdicts = [task for task in left_only
                          if _mean(_p_values(left_by_task[task])) is not None]
    right_only_verdicts = [task for task in right_only
                           if _mean(_p_values(right_by_task[task])) is not None]
    if left_only_verdicts:
        _wrap_list(f"  P verdicts on the left with no right-side trajectory "
                   f"(no verdict there): ", left_only_verdicts)
    if right_only_verdicts:
        _wrap_list(f"  P verdicts on the right with no left-side trajectory "
                   f"(no verdict there): ", right_only_verdicts)
    if not left_only_verdicts and not right_only_verdicts:
        if left_only or right_only:
            print(f"  tasks present on one side only: left {len(left_only)}, "
                  f"right {len(right_only)}; none has a P verdict")
        else:
            print("  every task present on one side is present on the other")
    print()


def _parse_tasks(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    tasks = tuple(dict.fromkeys(
        item.strip() for item in value.split(",") if item.strip()))
    return tasks


def _schema_version(connection: sqlite3.Connection, tables: set[str]) -> str:
    if "metadata" not in tables:
        return "unknown (metadata table absent)"
    row = connection.execute(
        "SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    return "unknown" if row is None else str(row[0])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="report_repair_trend.py",
        description="Read-only report of official repair outcomes per training step.",
        epilog=(
            "A step is trajectories.behavior_version (the policy version used to "
            "sample the trajectory). --compare accepts step numbers or update ids "
            "(for example --compare 0,1 or --compare upd-0000,upd-0001)."))
    parser.add_argument(
        "--run-dir", required=True,
        help="run directory containing spo_stream.db, or the database file itself")
    parser.add_argument(
        "--tasks",
        help="comma-separated task ids to restrict the report to")
    parser.add_argument(
        "--compare", metavar="A,B",
        help="compare two policy steps or update ids, for example 0,1")
    args = parser.parse_args(argv)

    requested_tasks = _parse_tasks(args.tasks)
    if args.tasks is not None and not requested_tasks:
        parser.error("--tasks must name at least one task id")

    compare_tokens: tuple[str, str] | None = None
    if args.compare is not None:
        parts = tuple(part.strip() for part in args.compare.split(","))
        if len(parts) != 2 or not all(parts):
            parser.error("--compare must be exactly two comma-separated values: A,B")
        compare_tokens = parts

    run_path = Path(args.run_dir).expanduser()
    if run_path.is_file():
        db_path = run_path.resolve()
    else:
        db_path = (run_path / "spo_stream.db").resolve()
    if not db_path.is_file():
        print(f"error: ledger database not found: {db_path}", file=sys.stderr)
        if run_path.is_dir():
            print("hint: --run-dir must contain spo_stream.db", file=sys.stderr)
        return _EXIT_DATABASE

    notes: list[str] = []
    uri = _read_only_uri(db_path)
    if uri is None:
        print(
            f"error: database is in WAL mode with a non-empty -wal file: {db_path}",
            file=sys.stderr)
        print(
            "hint: make a consistent copy first (sqlite3 <db> \".backup <copy>\" or "
            "the Python backup API) and point --run-dir at the copy; reading the "
            "live WAL in place would create -shm/-wal sidecars or miss committed "
            "frames", file=sys.stderr)
        return _EXIT_DATABASE
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        tables = _table_names(connection)
        missing = [table for table in _REQUIRED_TABLES if table not in tables]
        if missing:
            print(
                f"error: ledger database is missing required table(s): "
                f"{', '.join(missing)}", file=sys.stderr)
            print(f"database: {db_path}", file=sys.stderr)
            return _EXIT_DATABASE

        records = _load_records(connection, tables, notes)
        if "audit_events" in tables:
            absent_audit = sum(1 for record in records if record.positive_signal is None)
            if absent_audit:
                notes.append(
                    f"{absent_audit} trajectories have no reward_audit "
                    f"positive_repair_signal; reported as absent ('-'), not false")
        update_rows = connection.execute(
            "SELECT update_id,phase,optimizer_cursor,policy_before_version "
            "FROM updates ORDER BY created_at,update_id").fetchall()
        schema_version = _schema_version(connection, tables)
    except sqlite3.Error as error:
        print(f"error: cannot read ledger database {db_path}: {error}",
              file=sys.stderr)
        if "no such table" in str(error).lower() or "no such column" in str(error).lower():
            print("error: ledger schema does not match the expected tables/columns",
                  file=sys.stderr)
        return _EXIT_DATABASE
    finally:
        if connection is not None:
            connection.close()

    known_updates = {str(row["update_id"]) for row in update_rows}
    selectors: tuple[Selector, Selector] | None = None
    if compare_tokens is not None:
        resolved = [_resolve_selector(token, known_updates) for token in compare_tokens]
        if any(selector is None for selector in resolved):
            parser.error(
                f"--compare values must be non-negative step numbers or known update "
                f"ids; unknown: "
                f"{', '.join(token for token, selector in zip(compare_tokens, resolved) if selector is None)}")
        selectors = (resolved[0], resolved[1])  # type: ignore[assignment]

    if requested_tasks is not None:
        present = {record.task_id for record in records}
        absent = [task for task in requested_tasks if task not in present]
        if absent:
            _wrap_list("requested task(s) with no verdict in this ledger: ", absent)
        wanted = set(requested_tasks)
        records = [record for record in records if record.task_id in wanted]

    _print_header(db_path, schema_version, records, update_rows,
                  requested_tasks)
    steps = _print_steps(records)
    index = ReasonIndex()
    _print_table(records, steps, index)
    _print_p_verdicts(records)
    _print_reasons(index)

    if selectors is not None:
        _print_comparison(records, selectors[0], selectors[1], index, notes)

    duplicate_updates = sorted(
        record.trajectory_id for record in records if len(record.update_ids) > 1)
    if duplicate_updates:
        _wrap_list("WARNING: trajectories frozen into more than one update: ",
                   duplicate_updates)
    no_reward = [record.trajectory_id for record in records
                 if record.reward is None]
    if no_reward:
        _wrap_list("trajectories without a usable reward record: ", no_reward)
    _print_notes(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
