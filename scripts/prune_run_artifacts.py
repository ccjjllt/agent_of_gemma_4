#!/usr/bin/env python3
"""Prune finished-run artifacts while keeping the published adapter and ledger.

A finished SPO run keeps growing on disk: every optimizer step publishes an
immutable adapter version (~1.9 GB at rank 128) and the trainer checkpoint is
rewritten in place but needs a same-size temporary file.  Neither the version
directories nor the checkpoint are collected automatically, and
``AdapterManager.unregister`` cannot remove an ancestor of a registered child,
so a linear version chain can only be deleted as whole directories.

This tool is deliberately offline and conservative:

* it refuses to touch a run that still has non-CONSUMED updates;
* it never deletes the active adapter or a version named by ``--keep``;
* it verifies the kept adapters' weights against their registry SHA256 before
  deleting anything;
* it removes staging copies, the trainer checkpoint and transient work
  directories only when asked;
* it writes ``pruned_artifacts.json`` into the run directory describing exactly
  what was removed, so a later reader can tell a pruned run from a corrupt one.

The registry is left untouched on purpose: rows for pruned versions keep their
manifest and lineage, but their weight files are gone.  A finished run needs
only the active adapter's weights for serving and for the published-version
checks; re-running or resuming a pruned run requires re-exporting the pruned
versions from a checkpoint, which is why the checkpoint can be kept with
``--keep-checkpoint``.

Usage:
    python scripts/prune_run_artifacts.py --run-dir RUN --keep 1 --dry-run
    python scripts/prune_run_artifacts.py --run-dir RUN --keep 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path


TERMINAL = "CONSUMED"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pending_updates(db: Path) -> list[tuple[str, str]]:
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = connection.execute("select update_id, phase from updates").fetchall()
    finally:
        connection.close()
    return [(update_id, phase) for update_id, phase in rows if phase != TERMINAL]


def _directory_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--keep", type=int, action="append", default=[],
                        help="adapter version to retain (repeatable); the active version is always kept")
    parser.add_argument("--keep-checkpoint", action="store_true",
                        help="retain spo_trainer_state.pt (default: delete it in a finished run)")
    parser.add_argument("--keep-transients", action="store_true",
                        help="retain workspaces/, probe_workspaces/, requests/, endpoint/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve(strict=True)
    db = run_dir / "spo_stream.db"
    registry_path = run_dir / "adapters" / "registry.json"
    if not db.is_file() or not registry_path.is_file():
        print(f"error: {run_dir} is not an SPO run directory", file=sys.stderr)
        return 3

    pending = _pending_updates(db)
    if pending:
        print("error: run is not finished; pending updates: "
              + ", ".join(f"{update_id}={phase}" for update_id, phase in pending), file=sys.stderr)
        return 3

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    active = int(registry["active_version"])
    rows = registry["versions"]
    keep = set(args.keep) | {active}

    missing_keep = sorted(version for version in keep if str(version) not in rows)
    if missing_keep:
        print(f"error: --keep names unregistered versions: {missing_keep}", file=sys.stderr)
        return 3

    # Verify every kept adapter's weights before deleting anything.
    for version in sorted(keep):
        row = rows[str(version)]
        weights = run_dir / "adapters" / row["path"] / "adapter_model.safetensors"
        if not weights.is_file():
            print(f"error: kept v{version} has no weights at {weights}", file=sys.stderr)
            return 3
        expected = row["manifest"]["adapter_sha256"]
        actual = _sha256(weights)
        if actual != expected:
            print(f"error: kept v{version} SHA256 mismatch ({actual[:16]} != {expected[:16]})",
                  file=sys.stderr)
            return 3

    targets: list[tuple[str, Path]] = []
    for name, row in rows.items():
        version = int(name)
        if version in keep:
            continue
        directory = run_dir / "adapters" / row["path"]
        if directory.exists():
            targets.append((f"adapter v{version}", directory))
    for pattern in ("staging-*",):
        for directory in sorted((run_dir / "adapters").glob(pattern)):
            if directory.is_dir():
                targets.append((f"staging {directory.name}", directory))
    if not args.keep_checkpoint:
        checkpoint = run_dir / "spo_trainer_state.pt"
        if checkpoint.is_file():
            targets.append(("trainer checkpoint", checkpoint))
    if not args.keep_transients:
        for name in ("workspaces", "probe_workspaces", "requests", "endpoint"):
            directory = run_dir / name
            if directory.exists():
                targets.append((f"transient {name}", directory))

    plan = [{"label": label, "path": str(path.relative_to(run_dir)),
             "bytes": _directory_bytes(path) if path.is_dir() else path.stat().st_size}
            for label, path in targets]
    total = sum(item["bytes"] for item in plan)
    print(f"run: {run_dir}")
    print(f"active v{active}; keeping {sorted(keep)}")
    for item in plan:
        print(f"  remove {item['label']:24s} {item['bytes'] / 1e9:7.3f} GB  {item['path']}")
    print(f"total reclaimable: {total / 1e9:.3f} GB")
    if args.dry_run:
        print("dry run: nothing deleted")
        return 0

    removed = []
    for (label, path), item in zip(targets, plan):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        removed.append(item)

    record = {
        "schema": "gemma.spo_pruned_artifacts.v1",
        "run_dir": str(run_dir),
        "active_version": active,
        "kept_versions": sorted(keep),
        "kept_checkpoint": args.keep_checkpoint,
        "removed": removed,
        "total_bytes": total,
        "note": ("registry rows are intentionally kept; pruned versions have no weights and "
                 "the run is finished, so no serving or resume path loads them"),
    }
    (run_dir / "pruned_artifacts.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"pruned {len(removed)} item(s), {total / 1e9:.3f} GB reclaimed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
