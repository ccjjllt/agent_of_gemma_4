#!/usr/bin/env python3
"""External wall-clock supervisor for a whole GPU session (stdlib only).

Why this exists
---------------
An in-process run budget is advisory: a SIGTERM delivered while the training
loop is inside a CUDA/optimizer call can be swallowed, so a session outlives
its authorized wall clock and keeps billing.  This script owns the clock from
outside the workload.  It starts counting *before* the first process launches
(vLLM, the trainer, everything), soft-stops the whole process group early
enough for the workload to checkpoint and finish a durable transaction, and
SIGKILLs the group at the authorized deadline.  The grace period is counted
*inside* the budget, not added on top, so the child cannot run for N + G.

CLI
---
    python scripts/session_supervisor.py \
        --budget-seconds N --grace-seconds G \
        --session-file PATH --log PATH \
        -- <command> [args...]

Timeline (measured from the supervisor's clock, start_new_session=True):
    t = 0                  command spawned; clock already running
    t = N - G              SIGTERM to the process group (soft stop)
    t = N                  SIGKILL to the process group if anything survives

G = 0 is allowed: SIGTERM and SIGKILL are then sent back-to-back, so there is
a hard deadline but no checkpoint window.

The child gets its own session and process group, so vLLM workers, DataLoader
workers and shell grandchildren that did not detach are signalled together.
A soft stop is only useful if the workload handles SIGTERM (for example by
finishing the current transaction and exiting); the supervisor cannot make a
SIGTERM-ignoring process save state.

Supervision ends when the direct child exits.  Every group member already got
the group SIGTERM, but a descendant that ignores SIGTERM and outlives its
leader is not waited for: the exit-status rules below preserve the leader's
status instead.  Make the leader wait for its workers (or keep a wrapper
process alive) if that case is possible in the workload.

Files
-----
Session JSON (written atomically via a temp file + os.replace, refreshed at
start, at each signal, and after the child exits).  Keys:

    schema              "gemma.session_supervisor.v1"
    started_unix        wall-clock start, before the child was spawned
    budget_seconds      N
    soft_stop_at_unix   planned SIGTERM time, started_unix + (N - G)
    hard_kill_at_unix   planned SIGKILL time, started_unix + N
    grace_seconds       G
    command             argv list of the supervised command
    pid, pgid           child pid / process-group id (equal here)
    log_path            child stdout+stderr destination
    supervisor_pid      pid of this supervisor
    state               running | soft_stopped | exited |
                        exited_after_soft_stop | hard_killed | spawn_failed |
                        session_write_failed | supervisor_error
    soft_stop_sent_unix actual SIGTERM time (null if never sent)
    hard_kill_sent_unix actual SIGKILL time (null if never sent)
    exit_code           final supervisor exit status (see below)
    elapsed_seconds     monotonic seconds from clock start to finish
    escalated           true once the supervisor attempts SIGKILL
    wait_code           raw Popen return code (negative = killed by signal)
    reaped              whether the direct child was reaped
    finished_unix       wall-clock finish
    interrupted_by      set when SIGINT/SIGTERM/SIGHUP shortened the clock
    forced_kill_by      set when a second stop request forced SIGKILL
    soft_stop_delivery / hard_kill_delivery
                        sent | leader_only | not_found | not_permitted |
                        refused_unsafe_pgid

The log is opened in append mode ("ab"), so reusing a path preserves earlier
output; use a fresh path per session if that is not wanted.  The supervisor
itself writes nothing to the log, only the child's stdout/stderr (merged).
The child's stdin is /dev/null: this is a batch supervisor, not a terminal.

Exit status
-----------
    child status (0..255)  the child ended without SIGKILL, including when it
                           exited voluntarily in response to the soft-stop.
                           128+signum when a signal other than our SIGKILL
                           ended it (e.g. 143 for SIGTERM).
    137                    the supervisor escalated to SIGKILL: at the hard
                           deadline, after a second stop request, or after a
                           second Ctrl-C.  Always 137, matching shell 128+9.
    126 / 127              the command could not be executed (permission /
                           not found).
    2                      usage error (bad/missing option, no "--", G >= N,
                           empty command, same session/log path).
    1                      supervisor setup failure, e.g. the initial session
                           JSON could not be written (the already-spawned
                           child is killed first).

124 is deliberately not used.  Unlike GNU timeout, the grace period is part
of the budget and a cooperative exit keeps its own status; a run stopped at
the deadline is identifiable from state / hard_kill_sent_unix in the JSON.

Signals
-------
SIGTERM, SIGHUP -> first request soft-stops immediately and waits up to G
(never past the hard deadline).  SIGINT raises KeyboardInterrupt and takes the
same path.  A second request while waiting, or a second Ctrl-C, forces SIGKILL
immediately.  SIGKILL to the supervisor itself cannot be caught; the session
JSON then stays in its last state, which is why it is refreshed at every
transition, and its pid/pgid identify the group an external cleanup must kill.

The supervisor is a safety net, not a replacement for a workload that handles
SIGTERM.  It guarantees that the supervised process group is signalled at the
deadline; it cannot guarantee that the workload has flushed its last
checkpoint, and it cannot catch a deliberately detached daemon (setsid or a
double fork).  Anything that must die at the deadline must stay in the group.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Sequence

SCHEMA = "gemma.session_supervisor.v1"
POLL_SECONDS = 0.05
REAP_SECONDS = 5.0

EXIT_SUPERVISOR = 1
EXIT_USAGE = 2
EXIT_NOT_EXECUTABLE = 126
EXIT_NOT_FOUND = 127
EXIT_HARD_KILL = 137

# Stop requests delivered to the supervisor itself.  A list rather than a flag
# so a second request during the grace period can force an immediate SIGKILL.
_STOP_REQUESTS: list[str] = []


class UsageError(ValueError):
    """Invalid command line; reported on stderr with exit status 2."""


def _request_stop(signum: int, _frame: Any) -> None:
    try:
        name = signal.Signals(signum).name.lower()
    except ValueError:  # pragma: no cover - defensive
        name = f"signal_{signum}"
    _STOP_REQUESTS.append(f"supervisor_{name}")


def _install_stop_handlers() -> None:
    for signum in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(signum, _request_stop)
        except (OSError, RuntimeError, ValueError):  # pragma: no cover
            pass


def _split_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split supervisor options from the command at the first literal --."""
    args = list(argv)
    if "--" not in args:
        raise UsageError("missing '--' before the supervised command")
    marker = args.index("--")
    return args[:marker], args[marker + 1 :]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session_supervisor.py",
        description=(
            "External wall-clock supervisor: SIGTERM the process group at "
            "budget-grace, SIGKILL at budget, record a session JSON."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--budget-seconds", type=float, required=True, metavar="N",
                        help="total authorized wall clock, including grace")
    parser.add_argument("--grace-seconds", type=float, required=True, metavar="G",
                        help="seconds between soft stop and hard kill, 0 <= G < N")
    parser.add_argument("--session-file", type=Path, required=True, metavar="PATH",
                        help="session JSON to create/refresh atomically")
    parser.add_argument("--log", type=Path, required=True, metavar="PATH",
                        help="append child stdout+stderr here")
    return parser


def _validate(budget: float, grace: float, command: Sequence[str]) -> None:
    if not math.isfinite(budget) or budget <= 0:
        raise UsageError("--budget-seconds must be a finite number > 0")
    if not math.isfinite(grace) or grace < 0:
        raise UsageError("--grace-seconds must be a finite number >= 0")
    if grace >= budget:
        raise UsageError("--grace-seconds must be smaller than --budget-seconds")
    if not command:
        raise UsageError("supervised command is empty")
    if not command[0]:
        raise UsageError("supervised command has an empty executable")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write the JSON in the same directory, fsync, then rename over path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    data = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    # Best effort: make the rename durable without failing the session over it.
    try:
        directory = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory)
    except OSError:
        pass
    finally:
        os.close(directory)


def _spawn(command: Sequence[str], log_path: Path) -> subprocess.Popen[bytes]:
    with log_path.open("ab", buffering=0) as log:
        return subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )


def _kill_now(proc: subprocess.Popen[bytes]) -> None:
    """Last-resort cleanup when supervision cannot continue (setup failure)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=REAP_SECONDS)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass


def _summary(record: dict[str, Any], session_file: Path) -> str:
    payload = {
        "schema": SCHEMA,
        "state": record.get("state"),
        "exit_code": record.get("exit_code"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "escalated": bool(record.get("escalated")),
        "soft_stop_sent_unix": record.get("soft_stop_sent_unix"),
        "hard_kill_sent_unix": record.get("hard_kill_sent_unix"),
        "pid": record.get("pid"),
        "session_file": str(session_file),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _print_summary(record: dict[str, Any], session_file: Path) -> None:
    print(_summary(record, session_file), flush=True)


def _spawn_failure(
    record: dict[str, Any],
    session_file: Path,
    code: int,
    message: str,
    start_mono: float,
) -> int:
    record["state"] = "spawn_failed"
    record["exit_code"] = code
    record["elapsed_seconds"] = round(time.monotonic() - start_mono, 3)
    record["finished_unix"] = round(time.time(), 3)
    record["error"] = message
    try:
        _write_json_atomic(session_file, record)
    except OSError as exc:
        print(f"session_supervisor: warning: cannot write {session_file}: {exc}",
              file=sys.stderr, flush=True)
    print(f"session_supervisor: error: {message}", file=sys.stderr, flush=True)
    _print_summary(record, session_file)
    return code


def _supervise(
    proc: subprocess.Popen[bytes],
    record: dict[str, Any],
    session_file: Path,
    start_unix: float,
    start_mono: float,
    budget: float,
    grace: float,
) -> int:
    """Run the clock, soft-stop, and hard-kill.  Returns the exit status."""
    pgid = int(record["pgid"])
    soft_deadline = start_mono + (budget - grace)
    hard_deadline = start_mono + budget

    def now_unix() -> float:
        # Derive wall-clock times from the monotonic clock so a system clock
        # adjustment cannot move the recorded timeline relative to the run.
        return start_unix + (time.monotonic() - start_mono)

    def persist() -> None:
        try:
            _write_json_atomic(session_file, record)
        except OSError as exc:
            print(f"session_supervisor: warning: cannot update {session_file}: "
                  f"{exc}", file=sys.stderr, flush=True)

    def deliver(sig: signal.Signals) -> str:
        """Signal the whole group, never this supervisor's own group."""
        if pgid <= 1 or pgid == os.getpgrp():
            return "refused_unsafe_pgid"
        try:
            os.killpg(pgid, sig)
            return "sent"
        except ProcessLookupError:
            pass
        except PermissionError:
            return "not_permitted"
        # The group is gone but the direct child is somehow still alive (it
        # moved itself to another group): fall back to the leader only.
        if proc.poll() is not None:
            return "not_found"
        try:
            proc.send_signal(sig)
            return "leader_only"
        except (ProcessLookupError, PermissionError):
            return "not_found"

    def finish(code: int, state: str) -> int:
        record["state"] = state
        record["exit_code"] = int(code)
        record["elapsed_seconds"] = round(time.monotonic() - start_mono, 3)
        record["finished_unix"] = round(now_unix(), 3)
        persist()
        return int(code)

    def finish_exit(wait_code: int) -> int:
        record["wait_code"] = wait_code
        code = 128 + (-wait_code) if wait_code < 0 else wait_code
        state = (
            "exited_after_soft_stop"
            if record.get("soft_stop_sent_unix") is not None
            else "exited"
        )
        return finish(code, state)

    def hard_kill(forced_by: str | None) -> int:
        record["hard_kill_sent_unix"] = round(now_unix(), 3)
        record["hard_kill_delivery"] = deliver(signal.SIGKILL)
        record["escalated"] = True
        record["state"] = "hard_killed"
        record["exit_code"] = EXIT_HARD_KILL
        record["elapsed_seconds"] = round(time.monotonic() - start_mono, 3)
        if forced_by is not None:
            record["forced_kill_by"] = forced_by
        # Persist before the potentially blocking reap: the deadline fact and
        # the 137 result are durable even if the child hangs in uninterruptible
        # sleep (finish() refreshes elapsed/finished after the reap).
        persist()
        try:
            wait_code: int | None = proc.wait(timeout=REAP_SECONDS)
        except subprocess.TimeoutExpired:
            wait_code = None
        except KeyboardInterrupt:
            wait_code = None
        record["wait_code"] = wait_code
        record["reaped"] = wait_code is not None
        return finish(EXIT_HARD_KILL, "hard_killed")

    def soft_stop(trigger: str | None) -> int:
        record["soft_stop_sent_unix"] = round(now_unix(), 3)
        record["soft_stop_delivery"] = deliver(signal.SIGTERM)
        record["state"] = "soft_stopped"
        if trigger is not None:
            record["interrupted_by"] = trigger
        persist()

        # The grace deadline is capped by the hard deadline, which is what
        # keeps the grace period inside the authorized budget.
        deadline = min(hard_deadline, time.monotonic() + grace)
        while True:
            wait_code = proc.poll()
            if wait_code is not None:
                return finish_exit(wait_code)
            if _STOP_REQUESTS:
                return hard_kill(_STOP_REQUESTS.pop(0))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Short sleeps detect an early exit promptly without busy-spinning
            # (about 20 wakeups/second while waiting).  A second Ctrl-C during
            # the grace cuts it short and forces the hard kill immediately.
            try:
                time.sleep(min(POLL_SECONDS, remaining))
            except KeyboardInterrupt:
                return hard_kill("keyboard_interrupt")
        return hard_kill(None)

    trigger: str | None = None
    try:
        while True:
            wait_code = proc.poll()
            if wait_code is not None:
                return finish_exit(wait_code)
            if _STOP_REQUESTS:
                trigger = _STOP_REQUESTS.pop(0)
                record["interrupted_unix"] = round(now_unix(), 3)
                break
            remaining = soft_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                time.sleep(min(POLL_SECONDS, remaining))
            except KeyboardInterrupt:
                # Handle it here, outside the outer except handler, so a
                # second Ctrl-C during the subsequent grace is caught by
                # soft_stop instead of escaping the supervisor.
                trigger = "keyboard_interrupt"
                record["interrupted_unix"] = round(now_unix(), 3)
                break

        # Re-check: the child may have exited exactly on the soft deadline, in
        # which case it is a normal exit and no signal is sent.
        wait_code = proc.poll()
        if wait_code is not None:
            return finish_exit(wait_code)
        return soft_stop(trigger)
    except KeyboardInterrupt:
        # Fallback for an interrupt raised outside the sleep sites (e.g.
        # during a JSON write).  The sleep sites handle the normal path and
        # soft_stop handles a second Ctrl-C.  If a soft stop was already
        # sent, escalate immediately instead of waiting another grace.
        wait_code = proc.poll()
        if wait_code is not None:
            return finish_exit(wait_code)
        if record.get("soft_stop_sent_unix") is None:
            record["interrupted_unix"] = round(now_unix(), 3)
            return soft_stop("keyboard_interrupt")
        return hard_kill("keyboard_interrupt")
    except BaseException:
        # A supervisor bug must never leave the session running past its
        # budget.  Kill the group, then let the original error propagate.
        try:
            if proc.poll() is None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    try:
                        proc.kill()
                    except OSError:
                        pass
                try:
                    proc.wait(timeout=REAP_SECONDS)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    pass
        finally:
            raise


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    # Help belongs to the supervisor; a --help after the -- separator belongs
    # to the child and must not be intercepted.
    marker = raw.index("--") if "--" in raw else len(raw)
    if any(arg in ("-h", "--help") for arg in raw[:marker]):
        parser.print_help()
        return 0
    try:
        supervisor_argv, command = _split_argv(raw)
    except UsageError as exc:
        parser.print_usage(sys.stderr)
        print(f"session_supervisor: error: {exc}", file=sys.stderr, flush=True)
        return EXIT_USAGE
    try:
        args = parser.parse_args(supervisor_argv)
    except SystemExit as exc:  # argparse already printed its message
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        _validate(args.budget_seconds, args.grace_seconds, command)
    except UsageError as exc:
        parser.print_usage(sys.stderr)
        print(f"session_supervisor: error: {exc}", file=sys.stderr, flush=True)
        return EXIT_USAGE

    budget = float(args.budget_seconds)
    grace = float(args.grace_seconds)
    session_file: Path = args.session_file.expanduser().resolve()
    log_path: Path = args.log.expanduser().resolve()
    if session_file == log_path:
        print("session_supervisor: error: --session-file and --log must differ",
              file=sys.stderr, flush=True)
        return EXIT_USAGE
    try:
        session_file.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"session_supervisor: error: cannot create output directory: {exc}",
              file=sys.stderr, flush=True)
        return EXIT_SUPERVISOR

    # Handlers go in before launch so an orchestrator's SIGTERM during startup
    # is turned into a soft stop instead of killing only the supervisor.
    _install_stop_handlers()

    # The authorized clock starts here, before the child (vLLM, trainer, ...)
    # is spawned.  Wall-clock deadlines are derived from the monotonic clock.
    start_unix = time.time()
    start_mono = time.monotonic()
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "started_unix": round(start_unix, 3),
        "budget_seconds": budget,
        "soft_stop_at_unix": round(start_unix + (budget - grace), 3),
        "hard_kill_at_unix": round(start_unix + budget, 3),
        "grace_seconds": grace,
        "command": list(command),
        "pid": None,
        "pgid": None,
        "log_path": str(log_path),
        "supervisor_pid": os.getpid(),
        "state": "starting",
        "soft_stop_sent_unix": None,
        "hard_kill_sent_unix": None,
        "exit_code": None,
        "elapsed_seconds": None,
        "escalated": False,
    }

    try:
        proc = _spawn(command, log_path)
    except FileNotFoundError as exc:
        return _spawn_failure(record, session_file, EXIT_NOT_FOUND,
                              f"command not found: {exc}", start_mono)
    except PermissionError as exc:
        return _spawn_failure(record, session_file, EXIT_NOT_EXECUTABLE,
                              f"command not executable: {exc}", start_mono)
    except OSError as exc:
        return _spawn_failure(record, session_file, EXIT_SUPERVISOR,
                              f"cannot start command: {exc}", start_mono)

    # start_new_session=True makes the child a session leader, so its pid is
    # also the process-group id.
    record["pid"] = proc.pid
    record["pgid"] = proc.pid
    record["state"] = "running"
    try:
        _write_json_atomic(session_file, record)
    except KeyboardInterrupt:
        # The child is already running.  Turn the interrupt into a stop
        # request; _supervise sees it immediately and soft-stops.
        _STOP_REQUESTS.append("keyboard_interrupt")
    except OSError as exc:
        print(f"session_supervisor: error: cannot write {session_file}: {exc}; "
              "killing the child", file=sys.stderr, flush=True)
        _kill_now(proc)
        record["state"] = "session_write_failed"
        record["exit_code"] = EXIT_SUPERVISOR
        record["elapsed_seconds"] = round(time.monotonic() - start_mono, 3)
        record["finished_unix"] = round(time.time(), 3)
        record["error"] = str(exc)
        _print_summary(record, session_file)
        return EXIT_SUPERVISOR

    try:
        code = _supervise(proc, record, session_file, start_unix, start_mono,
                          budget, grace)
    except KeyboardInterrupt:  # pragma: no cover - _supervise handles this
        _kill_now(proc)
        record["state"] = "supervisor_error"
        record["exit_code"] = EXIT_HARD_KILL
        record["escalated"] = True
        record["elapsed_seconds"] = round(time.monotonic() - start_mono, 3)
        record["finished_unix"] = round(time.time(), 3)
        record["error"] = "KeyboardInterrupt outside supervised waits"
        try:
            _write_json_atomic(session_file, record)
        except OSError:
            pass
        code = EXIT_HARD_KILL

    _print_summary(record, session_file)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
