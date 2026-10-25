"""Collect named, reviewed F repair contracts with the existing official CPU workflow.

This selects no candidates or classifications automatically. Each input contract
names its exact train identity, production function and source-bound observation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REPLAY = ROOT / "competition/vfarl/replay"
spec = importlib.util.spec_from_file_location("source_contract_collector",
    HERE.parent / "s600-common-root/collect.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
collector.HERE = HERE
collector.ART = REPLAY / "artifacts/f600-root"
collector.REPORT_PREFIX = "f600-root-"
collector.DEFAULT_COURSE = "F600_specialist"
collector.REVIEW_SCOPE = "Individually reviewed flow/state repair contracts for F600; original full requirements retained."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=HERE / "selected-contracts.json")
    parser.add_argument("--keys", nargs="+")
    parser.add_argument("--stage", choices=("prepare", "replay", "publish", "all"), default="all")
    parser.add_argument("--attempt", default="001")
    parser.add_argument("--review-name", default="reviewed-candidates")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    if not args.attempt.isdecimal() or not args.review_name.replace("-", "").isalnum():
        raise ValueError("Explicit safe attempt and review identifiers required")
    configs = json.loads(args.selection.read_text())["tasks"]
    if args.keys:
        requested = set(args.keys)
        configs = [row for row in configs if row["key"] in requested]
        if {row["key"] for row in configs} != requested:
            raise ValueError("A named contract is absent from the selected source list")
    collector.ART.mkdir(parents=True, exist_ok=True)
    failures = []
    for config in configs:
        try:
            if args.stage in {"prepare", "all"}:
                collector.prepare(config)
                print("prepared", config["key"], flush=True)
            if args.stage in {"replay", "all"}:
                collector.replay(config, args.attempt)
                print("replayed", config["key"], flush=True)
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            if not args.keep_going:
                raise
            failures.append(config["key"])
            print("Retained actual source/replay failure", config["key"], repr(error), flush=True)
    if args.stage in {"publish", "all"}:
        collector.publish(configs, args.attempt, args.review_name)
    if failures:
        raise SystemExit(1)
