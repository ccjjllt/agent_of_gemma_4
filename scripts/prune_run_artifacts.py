#!/usr/bin/env python3
"""Prune a finished SPO run: keep the published adapter, drop process artifacts.

Thin CLI over :mod:`training.adapter.retention`.  The run must have every
update in ``CONSUMED``; the active adapter and any ``--keep`` version are kept,
their weights are verified against the registry SHA256 before deletion, and the
removal is recorded in ``pruned_artifacts.json``.

Examples:
    python scripts/prune_run_artifacts.py --run-dir RUN --keep 1 --dry-run
    python scripts/prune_run_artifacts.py --run-dir RUN --keep 1
    python scripts/prune_run_artifacts.py --run-dir RUN --keep 5 \
        --checkpoint-path /root/autodl-fs/spo-checkpoints/RUN
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.adapter.retention import RetentionError, prune_finished_run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--keep", type=int, action="append", default=[],
                        help="adapter version to retain (repeatable); the active version is always kept")
    parser.add_argument("--checkpoint-path", default=None,
                        help="checkpoint file or directory when it lives outside the run dir")
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument("--keep-transients", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve(strict=True)
    checkpoint_path = None
    if args.checkpoint_path:
        candidate = Path(args.checkpoint_path).expanduser()
        checkpoint_path = candidate / "spo_trainer_state.pt" if candidate.is_dir() else candidate
    try:
        record = prune_finished_run(
            run_dir, keep=tuple(args.keep), keep_checkpoint=args.keep_checkpoint,
            keep_transients=args.keep_transients, checkpoint_path=checkpoint_path,
            dry_run=args.dry_run)
    except RetentionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    print(json.dumps({key: record[key]
                      for key in ("total_bytes", "file_bytes", "kept_checkpoint", "dry_run")}))
    for item in record["removed"]:
        print(f"  {item['label']:28s} {item['bytes'] / 1e9:8.3f} GB  {item['path']}")
    print(f"total: {record['total_bytes'] / 1e9:.3f} GB unique bytes"
          + (" (dry run)" if args.dry_run else " reclaimed"))
    if record.get("file_bytes", 0) != record["total_bytes"]:
        print("note: per-item sizes count hardlinked paths separately; total counts "
              "each inode once and is the actual reclaim estimate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
