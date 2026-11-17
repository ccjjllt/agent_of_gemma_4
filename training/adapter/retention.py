"""Retention for finished SPO runs: keep the published adapter, drop the rest.

An SPO run publishes one immutable adapter version per optimizer step and
rewrites a single trainer checkpoint in place.  Neither is collected
automatically, and :meth:`AdapterManager.unregister` refuses to remove an
ancestor of a registered child, so a linear version chain only disappears as
whole directories.

``prune_finished_run`` is deliberately conservative:

* it refuses a run that still has a non-CONSUMED update;
* it never deletes the active adapter or a version named in ``keep``;
* it verifies every kept adapter's weights against the registry SHA256 before
  deleting anything;
* it records the removal in ``pruned_artifacts.json`` so a later reader can
  tell a pruned run from a corrupted one.

Registry rows are intentionally kept: pruned versions keep their manifest and
lineage, they simply have no weight files.  A finished run only serves and
verifies the active adapter.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

TERMINAL_PHASE = "CONSUMED"
TRANSIENT_DIRS = ("workspaces", "probe_workspaces", "requests", "endpoint")
PRUNED_RECORD = "pruned_artifacts.json"


class RetentionError(RuntimeError):
    """The run cannot be pruned safely (pending work, missing or bad weights)."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pending_updates(database: Path) -> list[tuple[str, str]]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute("select update_id, phase from updates").fetchall()
    finally:
        connection.close()
    return [(update_id, phase) for update_id, phase in rows if phase != TERMINAL_PHASE]


def _bytes_of(path: Path) -> int:
    if not path.is_dir():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def plan_prune(run_dir: Path, *, keep: tuple[int, ...] = (),
               keep_checkpoint: bool = False, keep_transients: bool = False,
               checkpoint_path: Path | None = None) -> list[dict]:
    """Return the removal plan without touching the filesystem."""
    run_dir = Path(run_dir)
    database = run_dir / "spo_stream.db"
    registry_path = run_dir / "adapters" / "registry.json"
    if not database.is_file() or not registry_path.is_file():
        raise RetentionError(f"{run_dir} is not an SPO run directory")
    pending = _pending_updates(database)
    if pending:
        raise RetentionError(
            "run is not finished; pending updates: "
            + ", ".join(f"{update_id}={phase}" for update_id, phase in pending))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    active = int(registry["active_version"])
    rows = registry["versions"]
    wanted = sorted(set(keep) | {active})
    missing = [version for version in wanted if str(version) not in rows]
    if missing:
        raise RetentionError(f"keep names unregistered versions: {missing}")
    for version in wanted:
        row = rows[str(version)]
        weights = run_dir / "adapters" / row["path"] / "adapter_model.safetensors"
        if not weights.is_file():
            raise RetentionError(f"kept v{version} has no weights at {weights}")
        expected = row["manifest"]["adapter_sha256"]
        actual = _sha256(weights)
        if actual != expected:
            raise RetentionError(
                f"kept v{version} SHA256 mismatch ({actual[:16]} != {expected[:16]})")

    targets: list[tuple[str, Path]] = []
    for name, row in rows.items():
        version = int(name)
        if version in wanted:
            continue
        directory = run_dir / "adapters" / row["path"]
        if directory.exists():
            targets.append((f"adapter v{version}", directory))
    for directory in sorted((run_dir / "adapters").glob("staging-*")):
        if directory.is_dir():
            targets.append((f"staging {directory.name}", directory))
    checkpoint = Path(checkpoint_path) if checkpoint_path else run_dir / "spo_trainer_state.pt"
    if not keep_checkpoint and checkpoint.is_file():
        targets.append(("trainer checkpoint", checkpoint))
    if not keep_transients:
        for name in TRANSIENT_DIRS:
            directory = run_dir / name
            if directory.exists():
                targets.append((f"transient {name}", directory))
    return [{"label": label, "path": str(path.relative_to(run_dir)),
             "bytes": _bytes_of(path)} for label, path in targets]


def prune_finished_run(run_dir: Path, *, keep: tuple[int, ...] = (),
                       keep_checkpoint: bool = False, keep_transients: bool = False,
                       checkpoint_path: Path | None = None,
                       dry_run: bool = False) -> dict:
    """Delete planned artifacts; return a JSON-safe record of what was removed."""
    run_dir = Path(run_dir)
    plan = plan_prune(run_dir, keep=keep, keep_checkpoint=keep_checkpoint,
                      keep_transients=keep_transients, checkpoint_path=checkpoint_path)
    total = sum(item["bytes"] for item in plan)
    record = {
        "schema": "gemma.spo_pruned_artifacts.v1",
        "run_dir": str(run_dir),
        "dry_run": bool(dry_run),
        "removed": plan,
        "total_bytes": total,
        "kept_checkpoint": bool(keep_checkpoint),
        "note": ("registry rows are intentionally kept; pruned versions keep their manifest "
                 "and lineage but have no weight files, and a finished run only serves the "
                 "active adapter"),
    }
    if dry_run:
        return record
    for item in plan:
        path = run_dir / item["path"]
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    (run_dir / PRUNED_RECORD).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record
