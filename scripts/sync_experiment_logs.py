#!/usr/bin/env python3
"""Retrieve analysis logs over an existing SSH connection; never fetch task assets.

Call summaries are computed on the server. Raw prompts, token IDs, model
trajectories, patches, JUnit reports, weights and credentials stay there.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tempfile


REMOTE_EXPORT = r'''
import base64, hashlib, json, math, re, sys
from datetime import datetime, timezone
from pathlib import Path

base = Path(sys.argv[1]).resolve(strict=True)
runs = json.loads(sys.argv[2])
run_pattern = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
for run in runs:
    if not run_pattern.fullmatch(run) or run in (".", ".."):
        raise ValueError("Invalid run directory name")

def read(path):
    relative = path.relative_to(base)
    if any((base.joinpath(*relative.parts[:i])).is_symlink()
           for i in range(1, len(relative.parts) + 1)):
        raise ValueError("Log export does not follow symlinks")
    path.resolve(strict=True).relative_to(base)
    with path.open("rb") as source:
        size = path.stat().st_size
        return source.read(size)

def send(relative, payload, source_payload=None):
    original = payload if source_payload is None else source_payload
    print(json.dumps({
        "path": relative, "payload": base64.b64encode(payload).decode(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_bytes": len(original),
        "source_sha256": hashlib.sha256(original).hexdigest(),
    }), flush=True)

def copy(path):
    if path.is_file():
        send(path.relative_to(base).as_posix(), read(path))

def finite(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None

for run in runs:
    folder = base / "runs" / run
    if not folder.is_dir() or folder.is_symlink():
        raise ValueError("Requested run does not exist as a real directory: " + run)
    for name in ("metrics.jsonl", "run_config.json", "curriculum_state.json"):
        copy(folder / name)
    # Learner stdout contains program diagnostics and numerical events, not
    # the worker's task trace. Do not collect worker.log or official_results/.
    copy(base / "install-records" / (run + ".log"))
    for rollout in sorted(folder.glob("batch-*-group-*-rollout-*")):
        if not rollout.is_dir() or rollout.is_symlink():
            raise ValueError("Unexpected rollout directory")
        for name in ("result.json", "reward.json"):
            copy(rollout / name)
        calls_path = rollout / "model_calls.jsonl"
        if not calls_path.is_file():
            continue
        payload = read(calls_path)
        lines = payload.splitlines(keepends=True)
        rows, incomplete_tail = [], False
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                call = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1 and not line.endswith(b"\n"):
                    incomplete_tail = True
                    break
                raise
            values = call.get("behavior_logprobs", [])
            valid = [value for value in values if finite(value) is not None]
            row = {key: call.get(key) for key in (
                "rollout_id", "call_index", "policy_version", "finish_reason",
                "requested_max_tokens", "effective_max_tokens", "context_limited",
            )}
            row.update(
                prompt_tokens=len(call.get("prompt_ids", [])),
                completion_tokens=len(call.get("completion_ids", [])),
                behavior_logprob_count=len(values),
                behavior_logprob_finite_count=len(valid),
                behavior_logprob_nonfinite_count=len(values) - len(valid),
                behavior_logprob_mean=math.fsum(valid) / len(valid) if valid else None,
                behavior_logprob_min=min(valid) if valid else None,
                behavior_logprob_max=max(valid) if valid else None,
            )
            for key in ("generation_elapsed_seconds", "first_scores_seconds", "queue_wait_seconds"):
                row[key] = finite(call.get(key))
            elapsed = row["generation_elapsed_seconds"]
            row["generation_tokens_per_second"] = row["completion_tokens"] / elapsed if elapsed and elapsed > 0 else None
            rows.append(json.dumps(row, allow_nan=False))
        output = ("\n".join(rows) + ("\n" if rows else "")).encode()
        send((rollout.relative_to(base) / "model_calls.summary.jsonl").as_posix(), output, payload)
        print(json.dumps({"call_summary": {
            "run": run, "rollout": rollout.name, "completed_calls": len(rows),
            "incomplete_tail_omitted": incomplete_tail,
        }}), flush=True)

for path in sorted((base / "install-records").glob("*.freeze.txt")):
    copy(path)
print(json.dumps({"snapshot": {
    "remote_utc": datetime.now(timezone.utc).isoformat(), "remote_base": str(base),
    "runs": runs, "exclusions": ["raw model calls", "worker logs", "task assets", "traces", "patches", "JUnit", "weights", "credentials"],
}}), flush=True)
'''


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".sync-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH destination, e.g. root@example.com")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--control-path", type=Path)
    parser.add_argument("--known-hosts", type=Path)
    parser.add_argument("--remote-python", default="/root/miniconda3/bin/python")
    parser.add_argument("--remote-base", default="/root/autodl-tmp/gemma4")
    parser.add_argument("--run", action="append", required=True, help="Run name; repeat to retrieve several runs")
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "results/autodl")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.@:\[\]-]+", args.host) or args.host.startswith("-"):
        parser.error("Invalid SSH destination")
    if not 1 <= args.port <= 65535:
        parser.error("Invalid SSH port")
    if not args.remote_python.startswith("/") or not args.remote_base.startswith("/"):
        parser.error("Remote interpreter and base must be absolute paths")
    runs = list(dict.fromkeys(args.run))
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run) or run in (".", "..") for run in runs):
        parser.error("--run accepts a directory name, not a path")
    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    command = ["ssh", "-p", str(args.port), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
    if args.control_path:
        command += ["-S", str(args.control_path.expanduser())]
    if args.known_hosts:
        command += ["-o", "UserKnownHostsFile=" + str(args.known_hosts.expanduser())]
    command += [args.host, shlex.join([args.remote_python, "-", args.remote_base, json.dumps(runs)])]
    entries, summaries, snapshot = [], [], None
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        process.stdin.write(REMOTE_EXPORT)
        process.stdin.close()
        for line in process.stdout:
            item = json.loads(line)
            if "snapshot" in item:
                snapshot = item["snapshot"]
                continue
            if "call_summary" in item:
                summaries.append(item["call_summary"])
                continue
            relative = PurePosixPath(item["path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
                raise ValueError("Unsafe path returned by remote export")
            path = destination.joinpath(*relative.parts)
            if not path.resolve().is_relative_to(destination):
                raise ValueError("Local output path escapes destination")
            payload = base64.b64decode(item.pop("payload"), validate=True)
            if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError("Log transfer checksum mismatch")
            atomic_write(path, payload)
            entries.append(item)
        if process.wait() or snapshot is None:
            raise RuntimeError("Remote log export did not finish; already received files remain available")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    snapshot.update(local_utc=datetime.now(timezone.utc).isoformat(), host=args.host, port=args.port,
                    files=entries, call_summaries=summaries)
    atomic_write(destination / "sync_metadata.json", (json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({"destination": str(destination), "runs": runs, "files": len(entries),
                      "completed_model_calls": sum(row["completed_calls"] for row in summaries),
                      "snapshot_utc": snapshot["remote_utc"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
