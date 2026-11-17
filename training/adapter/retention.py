"""Retention for SPO runs: finished-run cleanup and in-run version pruning.

An SPO run publishes one immutable adapter version per optimizer step and
rewrites a single trainer checkpoint in place.  Neither is collected
automatically, and :meth:`AdapterManager.unregister` refuses to remove an
ancestor of a registered child, so a linear version chain only disappears as
whole directories.

Two entry points exist:

``prune_in_run``
    Called after an update is committed and served.  It removes older
    published adapter directories and redundant staging copies while the run
    is still writing new versions.  It keeps the active version, configured
    versions, the last ``keep_recent`` predecessors, every version still
    pinned by a trajectory/buffer reference, and every version whose ledger
    update is not ``CONSUMED``.

``prune_finished_run``
    The conservative end-of-run cleanup: it refuses a run with pending work,
    keeps the active adapter and named versions, verifies their weights, and
    drops intermediate versions, staging, transient dirs and (optionally) the
    checkpoint.

Both write ``adapters/pruned_versions.json`` *before* deleting a version
directory.  Registry rows are intentionally kept: a pruned version retains its
manifest and lineage, it simply has no weight files.  ``AdapterManager`` reads
the marker (and, for runs pruned by older launchers, the legacy
``pruned_artifacts.json`` record) so the registry still opens after pruning.

Removal is deliberately conservative: any unknown phase, live reference,
missing marker, identity mismatch or exception skips the candidate instead of
forcing the deletion.  A run that cannot prune keeps running.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from training.rollout.policy_version import AdapterManifest


TERMINAL_PHASE = "CONSUMED"
PHASES = frozenset({
    "RESERVED", "LEARNING", "OPTIMIZED", "EXPORTED", "REGISTERED",
    "ACTIVATED", "CONSUMED",
})
TRANSIENT_DIRS = ("workspaces", "probe_workspaces", "requests", "endpoint")

# Finished-run record (legacy launchers wrote this and no marker).
PRUNED_RECORD = "pruned_artifacts.json"
PRUNED_ARTIFACTS_SCHEMA = "gemma.spo_pruned_artifacts.v1"
# In-run and finished-run version evidence (authoritative for missing dirs).
PRUNED_VERSIONS = "pruned_versions.json"
PRUNED_VERSIONS_SCHEMA = "gemma.spo_pruned_versions.v1"
IN_RUN_SCHEMA = "gemma.spo_in_run_prune.v1"

_STAGING_VERSION = re.compile(r"staging-v(\d+)\Z")
_ADAPTER_LABEL = re.compile(r"adapter v(\d+)\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RetentionError(RuntimeError):
    """The run cannot be pruned safely (pending work, missing or bad weights)."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}-", delete=False) as output:
            temporary = Path(output.name)
            json.dump(value, output, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _entry(version: int, value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one marker record; a corrupt marker is never tolerated."""
    if not isinstance(value, Mapping):
        raise RetentionError(f"pruned marker entry v{version} is not an object")
    if type(value.get("version")) is not int or value["version"] != version:
        raise RetentionError(f"pruned marker entry v{version} has a mismatched version")
    if not _valid_sha256(value.get("adapter_sha256")):
        raise RetentionError(f"pruned marker entry v{version} has an invalid SHA256")
    size = value.get("bytes")
    if type(size) is not int or size < 0:
        raise RetentionError(f"pruned marker entry v{version} has invalid bytes")
    stamp = value.get("pruned_unix")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) \
            or not math.isfinite(float(stamp)) or float(stamp) < 0:
        raise RetentionError(f"pruned marker entry v{version} has an invalid pruned_unix")
    return {"version": version, "adapter_sha256": value["adapter_sha256"],
            "bytes": size, "pruned_unix": float(stamp)}


def read_pruned_versions(adapters_dir: str | Path) -> dict[int, dict[str, Any]]:
    """Read ``adapters/pruned_versions.json``; absent means no pruned versions.

    A malformed marker raises :class:`RetentionError` (fail closed): silently
    tolerating a corrupt record could turn a missing adapter directory into a
    false "pruned" result.
    """
    path = Path(adapters_dir) / PRUNED_VERSIONS
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise RetentionError(f"pruned versions marker is unreadable: {path}: {exc}") from exc
    try:
        document = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise RetentionError(f"pruned versions marker is not valid JSON: {path}") from exc
    if not isinstance(document, dict) or document.get("schema") != PRUNED_VERSIONS_SCHEMA \
            or not isinstance(document.get("versions"), dict):
        raise RetentionError(f"pruned versions marker has an invalid schema: {path}")
    entries: dict[int, dict[str, Any]] = {}
    for key, value in document["versions"].items():
        if not isinstance(key, str) or not key.isdigit():
            raise RetentionError(f"pruned versions marker has an invalid key: {key!r}")
        version = int(key)
        entries[version] = _entry(version, value)
    return entries


def write_pruned_versions(adapters_dir: str | Path,
                          entries: Mapping[int, Mapping[str, Any]], *,
                          pruned_unix: float | None = None) -> Path:
    """Atomically merge and persist deletion evidence before any rmtree.

    One write covers every planned version, and it happens before the first
    directory is removed.  A crash therefore leaves each directory either
    intact or covered by the marker, never missing without evidence.
    """
    if not entries:
        raise ValueError("at least one pruned version is required")
    directory = Path(adapters_dir)
    path = directory / PRUNED_VERSIONS
    merged = read_pruned_versions(directory)
    stamp = time.time() if pruned_unix is None else float(pruned_unix)
    if not math.isfinite(stamp) or stamp < 0:
        raise ValueError("pruned_unix must be a finite non-negative value")
    for version, value in entries.items():
        if type(version) is not int or version < 0:
            raise ValueError("pruned version must be a non-negative integer")
        record = dict(value)
        record["version"] = version
        record["pruned_unix"] = stamp
        merged[version] = _entry(version, record)
    document = {
        "schema": PRUNED_VERSIONS_SCHEMA,
        "versions": {str(version): merged[version] for version in sorted(merged)},
    }
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, document)
    return path


def _adapters_relative(recorded: str, run_dir: Path) -> Path | None:
    """Resolve a legacy record path relative to the ``adapters`` directory."""
    candidate = Path(recorded)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to((run_dir / "adapters").resolve())
        except (OSError, ValueError):
            return None
    parts = [part for part in candidate.parts if part not in ("", ".")]
    if parts and parts[0] == "adapters":
        parts = parts[1:]
    if not parts or ".." in parts:
        return None
    return Path(*parts)


def legacy_pruned_versions(run_dir: str | Path) -> dict[int, dict[str, Any]]:
    """Translate a legacy ``pruned_artifacts.json`` into version evidence.

    Older launchers kept registry rows and deleted version dirs without a
    marker.  The legacy record is only accepted for entries labelled
    ``adapter vN`` whose recorded path still names that exact immutable
    version directory.  It is not itself rewritten here; reading is
    read-only, and new prunes write the authoritative marker.
    """
    run_dir = Path(run_dir)
    path = run_dir / PRUNED_RECORD
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}
    try:
        document = json.loads(text)
    except (TypeError, ValueError):
        return {}
    if not isinstance(document, dict) \
            or document.get("schema") != PRUNED_ARTIFACTS_SCHEMA \
            or not isinstance(document.get("removed"), list):
        return {}
    entries: dict[int, dict[str, Any]] = {}
    for item in document["removed"]:
        if not isinstance(item, Mapping):
            continue
        label = item.get("label")
        recorded = item.get("path")
        if not isinstance(label, str) or not isinstance(recorded, str):
            continue
        match = _ADAPTER_LABEL.fullmatch(label.strip())
        if match is None:
            continue
        version = int(match.group(1))
        relative = _adapters_relative(recorded, run_dir)
        if relative is None or len(relative.parts) != 2 or relative.parts[0] != "versions":
            continue
        name = relative.parts[1]
        if not name.startswith(f"{version:08d}-"):
            continue
        sha = item.get("adapter_sha256")
        if sha is not None and not _valid_sha256(sha):
            # Unknown evidence must not become a tolerated missing directory.
            continue
        size = item.get("bytes")
        entries[version] = {
            "version": version,
            "adapter_sha256": sha,
            "directory_name": name,
            "bytes": size if type(size) is int and size >= 0 else 0,
            "pruned_unix": None,
            "source": "legacy_pruned_artifacts",
        }
    return entries


def _legacy_marker_entries(run_dir: Path) -> dict[int, dict[str, Any]]:
    """Translate accepted legacy evidence into new-marker records.

    ``prune_finished_run`` rewrites ``pruned_artifacts.json`` from its current
    plan; on a run already pruned by an older launcher that plan contains no
    adapter versions, so preserving the old entries in the new marker first
    prevents the rewrite from destroying the only readable evidence.
    """
    legacy = legacy_pruned_versions(run_dir)
    if not legacy:
        return {}
    registry_path = run_dir / "adapters" / "registry.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    rows = registry.get("versions") if isinstance(registry, dict) else None
    if not isinstance(rows, dict):
        return {}
    entries: dict[int, dict[str, Any]] = {}
    for version, entry in legacy.items():
        row = rows.get(str(version))
        if not isinstance(row, Mapping):
            continue
        manifest = row.get("manifest")
        relative = row.get("path")
        if not isinstance(manifest, Mapping) or not isinstance(relative, str):
            continue
        sha = manifest.get("adapter_sha256")
        if not _valid_sha256(sha):
            continue
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts \
                or path.parts != ("versions", entry["directory_name"]):
            continue
        recorded_sha = entry.get("adapter_sha256")
        if recorded_sha is not None and recorded_sha != sha:
            continue
        if (run_dir / "adapters" / path).is_dir():
            # A directory that still exists is readable without evidence.
            continue
        entries[version] = {"version": version, "adapter_sha256": sha,
                            "bytes": int(entry.get("bytes") or 0)}
    return entries


def _display_path(path: Path, run_dir: Path) -> str:
    """Relative to the run dir when possible, absolute when it lives outside."""
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


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


def _ledger_phases(database: Path) -> dict[int, tuple[str, str]]:
    """Map a produced adapter version to ``(update_id, phase)``.

    Every update produces ``policy_before.version + 1``.  An unreadable or
    unknown-phase ledger row aborts the whole in-run plan: without provenance
    the caller must not delete any version directory.
    """
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "select update_id, phase, policy_before_version from updates").fetchall()
    finally:
        connection.close()
    phases: dict[int, tuple[str, str]] = {}
    for update_id, phase, before in rows:
        if not isinstance(update_id, str) or not update_id:
            raise RetentionError("ledger contains an invalid update id")
        if phase not in PHASES:
            raise RetentionError(f"update {update_id} has unknown phase {phase!r}")
        if type(before) is not int or before < 0:
            raise RetentionError(f"update {update_id} has an invalid policy_before_version")
        version = before + 1
        if version in phases:
            raise RetentionError(f"multiple updates claim adapter version {version}")
        phases[version] = (update_id, phase)
    return phases


def _read_registry(adapters_dir: Path) -> tuple[int, dict[int, dict[str, Any]]]:
    """Read and validate the parts of the registry needed for retention."""
    path = adapters_dir / "registry.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RetentionError(f"adapter registry is missing: {path}") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise RetentionError(f"adapter registry is unreadable: {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1 \
            or not isinstance(document.get("versions"), dict):
        raise RetentionError(f"adapter registry has an invalid schema: {path}")
    raw_rows = document["versions"]
    active = document.get("active_version")
    if type(active) is not int or str(active) not in raw_rows:
        raise RetentionError("adapter registry has no registered active version")
    rows: dict[int, dict[str, Any]] = {}
    for key, row in raw_rows.items():
        if not isinstance(key, str) or not key.isdigit() or not isinstance(row, Mapping):
            raise RetentionError("adapter registry has an invalid version entry")
        version = int(key)
        manifest = row.get("manifest")
        relative = row.get("path")
        references = row.get("references")
        if not isinstance(manifest, Mapping) or manifest.get("version") != version:
            raise RetentionError(f"adapter registry v{version} has a mismatched manifest")
        sha = manifest.get("adapter_sha256")
        if not _valid_sha256(sha):
            raise RetentionError(f"adapter registry v{version} has an invalid SHA256")
        if not isinstance(relative, str):
            raise RetentionError(f"adapter registry v{version} has an invalid path")
        relative_path = Path(relative)
        expected_name = f"{version:08d}-{sha[:16]}"
        if relative_path.is_absolute() or ".." in relative_path.parts \
                or relative_path.parts != ("versions", expected_name):
            raise RetentionError(
                f"adapter registry v{version} path does not match its identity")
        if type(references) is not int or references < 0:
            raise RetentionError(f"adapter registry v{version} has invalid references")
        rows[version] = {"path": relative_path, "references": references,
                         "manifest": dict(manifest), "adapter_sha256": sha}
    return active, rows


def _version_derived_state(rows: Mapping[int, Mapping[str, Any]],
                           phases: Mapping[int, tuple[str, str]]) -> set[int]:
    """Versions with a registered *pending* producing update."""
    protected: set[int] = set()
    for version, (_, phase) in phases.items():
        if phase != TERMINAL_PHASE and version in rows:
            protected.add(version)
    return protected


def plan_in_run_prune(
    run_dir: str | Path, *, keep_versions: Iterable[int] = (),
    keep_recent: int = 1, pinned_versions: Iterable[int] = (),
    expected_active: int | None = None,
) -> dict[str, Any]:
    """Plan an in-run prune without touching the filesystem.

    Only a version directory with a registered, ``CONSUMED`` producing update
    is ever removable.  Genesis v0 has no producing ledger update and is kept
    (conservative provenance), as is any version with an unknown or pending
    phase, a non-zero registry reference count, or an on-disk manifest that
    differs from its registry row.
    """
    run_dir = Path(run_dir)
    adapters_dir = run_dir / "adapters"
    database = run_dir / "spo_stream.db"
    if not database.is_file():
        raise RetentionError(f"{run_dir} has no SPO ledger")
    if type(keep_recent) is not int or isinstance(keep_recent, bool) or keep_recent < 0:
        raise RetentionError("keep_recent must be a non-negative integer")
    keep: list[int] = []
    for value in keep_versions:
        if type(value) is not int or isinstance(value, bool) or value < 0:
            raise RetentionError("keep_versions must contain non-negative integers")
        if value not in keep:
            keep.append(value)
    pinned: set[int] = set()
    for value in pinned_versions:
        if type(value) is not int or isinstance(value, bool) or value < 0:
            raise RetentionError("pinned_versions must contain non-negative integers")
        pinned.add(value)

    active, rows = _read_registry(adapters_dir)
    if expected_active is not None and active != expected_active:
        raise RetentionError(
            f"registry active v{active} is not the expected committed v{expected_active}")
    active_directory = adapters_dir / rows[active]["path"]
    if not active_directory.is_dir():
        raise RetentionError(f"active adapter directory is missing: {active_directory}")
    phases = _ledger_phases(database)

    protected: dict[int, str] = {}
    skipped: list[dict[str, Any]] = []

    def protect(version: int, reason: str) -> None:
        if version in rows:
            protected.setdefault(version, reason)

    protect(active, "active")
    for version in keep:
        if version not in rows:
            skipped.append({"version": version, "reason": "keep_version_unregistered"})
            continue
        protect(version, "keep_adapter_versions")
    for offset in range(1, keep_recent + 1):
        protect(active - offset, "keep_recent_adapters")
    for version in sorted(pinned):
        protect(version, "pinned_trajectory")
    for version in sorted(_version_derived_state(rows, phases)):
        _, phase = phases[version]
        protect(version, f"pending_update:{phase}")

    remove: list[dict[str, Any]] = []
    already_pruned: list[int] = []
    kept_versions: set[int] = {active}
    for version in sorted(rows):
        row = rows[version]
        directory = adapters_dir / row["path"]
        if not directory.is_dir():
            already_pruned.append(version)
            continue
        if version in protected:
            kept_versions.add(version)
            continue
        if version > active:
            # A rollback can leave consumed versions above the active pointer;
            # re-activating them must stay possible.
            skipped.append({"version": version, "reason": "above_active"})
            kept_versions.add(version)
            continue
        phase_info = phases.get(version)
        if phase_info is None:
            # Genesis has no producing update, and an unregistered lineage is
            # not proof of safe removal.
            skipped.append({"version": version, "reason": "no_consumed_update"})
            kept_versions.add(version)
            continue
        update_id, phase = phase_info
        if phase != TERMINAL_PHASE:
            kept_versions.add(version)
            continue
        if row["references"]:
            skipped.append({"version": version, "reason": "registry_references",
                            "references": row["references"], "update_id": update_id})
            kept_versions.add(version)
            continue
        manifest_path = directory / "manifest.json"
        try:
            on_disk = AdapterManifest.read(manifest_path)
        except (OSError, ValueError, TypeError) as exc:
            skipped.append({"version": version, "reason": "unreadable_manifest",
                            "update_id": update_id,
                            "error": f"{type(exc).__name__}: {exc}"})
            kept_versions.add(version)
            continue
        if on_disk.to_dict() != row["manifest"]:
            skipped.append({"version": version, "reason": "manifest_mismatch",
                            "update_id": update_id})
            kept_versions.add(version)
            continue
        remove.append({
            "version": version,
            "update_id": update_id,
            "path": _display_path(directory, run_dir),
            "bytes": _bytes_of(directory),
            "adapter_sha256": row["adapter_sha256"],
        })

    staging: list[dict[str, Any]] = []
    for directory in sorted(adapters_dir.glob("staging-*")):
        if not directory.is_dir():
            continue
        name = directory.name
        reason = None
        if name == "staging-genesis":
            if rows.get(0) is not None:
                reason = "genesis_published"
            else:
                skipped.append({"staging": name, "reason": "genesis_not_published"})
        else:
            match = _STAGING_VERSION.fullmatch(name)
            if match is None:
                skipped.append({"staging": name, "reason": "unrecognized_staging_name"})
                continue
            version = int(match.group(1))
            if version in phases:
                _, phase = phases[version]
                if phase == TERMINAL_PHASE:
                    reason = "active_published" if version == active else "consumed_update"
                else:
                    skipped.append({"staging": name,
                                    "reason": f"pending_update:{phase}"})
            elif version == active:
                # Active v0 has a genesis staging dir with its own branch;
                # any other active version without a ledger row is unknown.
                skipped.append({"staging": name, "reason": "no_ledger_update"})
            elif version > active:
                skipped.append({"staging": name, "reason": "future_version"})
            else:
                skipped.append({"staging": name, "reason": "no_ledger_update"})
        if reason is not None:
            staging.append({
                "name": name,
                "path": _display_path(directory, run_dir),
                "bytes": _bytes_of(directory),
                "reason": reason,
            })

    adapter_bytes = sum(item["bytes"] for item in remove)
    staging_bytes = sum(item["bytes"] for item in staging)
    return {
        "schema": f"{IN_RUN_SCHEMA}.plan",
        "run_dir": str(run_dir),
        "active_version": active,
        "keep_versions": keep,
        "keep_recent": keep_recent,
        "pinned_versions": sorted(pinned),
        "protected": [{"version": version, "reason": protected[version]}
                      for version in sorted(protected)],
        "remove": remove,
        "staging": staging,
        "skipped": skipped,
        "already_pruned": already_pruned,
        "kept_versions": sorted(kept_versions),
        "adapter_bytes": adapter_bytes,
        "staging_bytes": staging_bytes,
        "total_bytes": adapter_bytes + staging_bytes,
    }


def _planned_path(run_dir: Path, recorded: str) -> Path:
    path = Path(recorded)
    return path if path.is_absolute() else run_dir / path


def prune_in_run(
    run_dir: str | Path, *, keep_versions: Iterable[int] = (),
    keep_recent: int = 1, pinned_versions: Iterable[int] = (),
    expected_active: int | None = None, dry_run: bool = False,
) -> dict[str, Any]:
    """Execute an in-run prune; never raises for a single failed removal.

    Deletion is preceded by one atomic marker write covering every planned
    adapter version.  Staging directories are not registry versions, so they
    need no marker.  The returned record is JSON-safe and suitable for the
    ``spo_in_run_pruned`` metrics event.
    """
    run_dir = Path(run_dir)
    plan = plan_in_run_prune(
        run_dir, keep_versions=keep_versions, pinned_versions=pinned_versions,
        keep_recent=keep_recent, expected_active=expected_active)
    record: dict[str, Any] = {
        "schema": IN_RUN_SCHEMA,
        "run_dir": str(run_dir),
        "dry_run": bool(dry_run),
        "active_version": plan["active_version"],
        "keep_versions": plan["keep_versions"],
        "keep_recent": plan["keep_recent"],
        "pinned_versions": plan["pinned_versions"],
        "protected": plan["protected"],
        "kept_versions": plan["kept_versions"],
        "already_pruned": plan["already_pruned"],
        "skipped": plan["skipped"],
        "planned_versions": [item["version"] for item in plan["remove"]],
    }
    if dry_run:
        record.update({
            "removed_versions": [], "removed_staging": [],
            "removed_adapter_bytes": 0, "removed_staging_bytes": 0,
            "removed_bytes": 0, "planned_bytes": plan["total_bytes"],
            "failed": [], "marker": None,
        })
        return record

    stamp = time.time()
    marker = None
    if plan["remove"]:
        entries = {
            item["version"]: {
                "version": item["version"],
                "adapter_sha256": item["adapter_sha256"],
                "bytes": item["bytes"],
            }
            for item in plan["remove"]
        }
        marker = str(write_pruned_versions(
            run_dir / "adapters", entries, pruned_unix=stamp))
    removed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for item in plan["remove"]:
        path = _planned_path(run_dir, item["path"])
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            removed.append(item)
        except OSError as exc:
            failed.append({**item, "error": f"{type(exc).__name__}: {exc}"})

    removed_staging: list[dict[str, Any]] = []
    for item in plan["staging"]:
        path = _planned_path(run_dir, item["path"])
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            removed_staging.append(item)
        except OSError as exc:
            failed.append({**item, "error": f"{type(exc).__name__}: {exc}"})

    adapter_bytes = sum(item["bytes"] for item in removed)
    staging_bytes = sum(item["bytes"] for item in removed_staging)
    record.update({
        "removed_versions": [item["version"] for item in removed],
        "removed_staging": [item["name"] for item in removed_staging],
        "removed_adapter_bytes": adapter_bytes,
        "removed_staging_bytes": staging_bytes,
        "removed_bytes": adapter_bytes + staging_bytes,
        "planned_bytes": plan["total_bytes"],
        "failed": failed,
        "marker": marker,
        "pruned_unix": stamp,
    })
    return record


def plan_prune(run_dir: Path, *, keep: tuple[int, ...] = (),
               keep_checkpoint: bool = False, keep_transients: bool = False,
               checkpoint_path: Path | None = None) -> list[dict]:
    """Return the finished-run removal plan without touching the filesystem."""
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

    targets: list[dict] = []
    for name, row in rows.items():
        version = int(name)
        if version in wanted:
            continue
        directory = run_dir / "adapters" / row["path"]
        if directory.exists():
            targets.append({
                "label": f"adapter v{version}", "path": _display_path(directory, run_dir),
                "bytes": _bytes_of(directory), "kind": "adapter", "version": version,
                "adapter_sha256": row["manifest"]["adapter_sha256"],
            })
    for directory in sorted((run_dir / "adapters").glob("staging-*")):
        if directory.is_dir():
            targets.append({
                "label": f"staging {directory.name}",
                "path": _display_path(directory, run_dir),
                "bytes": _bytes_of(directory), "kind": "staging",
            })
    checkpoint = Path(checkpoint_path) if checkpoint_path else run_dir / "spo_trainer_state.pt"
    if not keep_checkpoint and checkpoint.is_file():
        targets.append({
            "label": "trainer checkpoint",
            "path": _display_path(checkpoint, run_dir),
            "bytes": _bytes_of(checkpoint), "kind": "checkpoint",
        })
    if not keep_transients:
        for name in TRANSIENT_DIRS:
            directory = run_dir / name
            if directory.exists():
                targets.append({
                    "label": f"transient {name}",
                    "path": _display_path(directory, run_dir),
                    "bytes": _bytes_of(directory), "kind": "transient",
                })
    return targets


def _pending_updates(database: Path) -> list[tuple[str, str]]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        rows = connection.execute("select update_id, phase from updates").fetchall()
    finally:
        connection.close()
    return [(update_id, phase) for update_id, phase in rows if phase != TERMINAL_PHASE]


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
        "schema": PRUNED_ARTIFACTS_SCHEMA,
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
    # Marker first, deletion second: a crash can never leave a version
    # directory missing without evidence, and AdapterManager stays readable.
    adapter_entries = {
        item["version"]: {
            "version": item["version"],
            "adapter_sha256": item["adapter_sha256"],
            "bytes": item["bytes"],
        }
        for item in plan if item.get("kind") == "adapter"
    }
    # Preserve evidence left by an older launcher before its record is
    # rewritten below; a second prune has no adapter targets in its plan.
    for version, entry in _legacy_marker_entries(run_dir).items():
        adapter_entries.setdefault(version, entry)
    if adapter_entries:
        write_pruned_versions(run_dir / "adapters", adapter_entries)
    for item in plan:
        path = _planned_path(run_dir, item["path"])
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    (run_dir / PRUNED_RECORD).write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record
