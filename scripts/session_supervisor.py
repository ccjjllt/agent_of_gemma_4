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

Supervision ends when the direct child has been reaped AND the supervised
tree is empty (process group empty AND cgroup member list empty when a
cgroup is active, plus a best-effort descendant sweep when it is not).
After the child exits the supervisor keeps monitoring until the tree
drains or the hard deadline; survivors at the hard deadline go through
the hard-kill path (SIGKILL the group plus a cgroup/descendant sweep and
a bounded post-kill wait) and the run is reported as hard_killed (137),
never as a success.  On Linux the tree is contained in a dedicated
cgroup (spo-sup-<supervisorpid>) so descendants that call setsid(2) are
still swept; without a writable cgroup the supervisor falls back to a
group kill plus a best-effort ppid-chain sweep and records that the gap
is explicit.  On Linux the supervisor also sets PR_SET_CHILD_SUBREAPER
once on itself, so reparented orphans get ppid == supervisor pid and are
found deterministically even when fork+setsid+leader-exit all happen
inside a single 50 ms poll interval (the snapshot alone would miss that
pattern).  Known residual on non-Linux platforms without subreaper: a
descendant that is forked, re-sessioned AND orphaned inside a single
poll interval is never snapshotted and cannot be linked back afterwards.

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
    cgroup              "v2" | "v1" on success;
                        "unavailable: <reason>" when containment setup failed;
                        "unsupported-platform" when there is no usable cgroup
                        (non-Linux or unwritable) and the supervisor fell back
                        to group-kill plus a best-effort descendant sweep.
    cgroup_path         dedicated cgroup directory or null when disabled.
    cgroup_drained      true when the cgroup member list (or, when disabled,
                        the whole supervised tree) is empty at finish.
    group_drained       true when no process-group member survives at finish.
    remaining_pids      best-effort sorted list of tree survivors at finish
                        (empty when drained).
    tree_pids_at_child_exit / group_pids_at_child_exit
                        best-effort snapshot taken when the direct child was
                        first observed reaped while the tree was still
                        non-empty; proves the supervisor did not finish at
                        leader exit but waited for the drain / hard kill.
    subreaper           true when PR_SET_CHILD_SUBREAPER is active (Linux).
    subreaper_reason    null when active, otherwise why it is off
                        (e.g. "non-linux platform").
    adopted_pids_seen   sorted pids observed with ppid == supervisor pid
                        (reparented orphans caught via subreaper).

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
SIGTERM.  It guarantees that the supervised tree (process group plus, on
Linux, a dedicated cgroup and the subreaper adoption scan) is signalled at
the deadline; it cannot guarantee that the workload has flushed its last
checkpoint.  A deliberately detached daemon that escapes the process group
(setsid) is still swept via the cgroup when writable, and via the
subreaper adoption scan on Linux even when no writable cgroup exists
(recorded explicitly as "unsupported-platform" / "unavailable: ...").
Anything that must die at the deadline must stay in the group, and on
Linux the cgroup keeps even setsid escapees killable.
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


def _group_pids(pgid: int, exclude: Sequence[int] = ()) -> list[int]:
    """Best-effort list of live pids in process group pgid."""
    excluded = set(exclude)
    if os.path.isdir("/proc"):
        found: list[int] = []
        try:
            entries = os.listdir("/proc")
        except OSError:
            entries = []
        for entry in entries:
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid in excluded or pid <= 1:
                continue
            try:
                if os.getpgid(pid) == pgid:
                    found.append(pid)
            except (ProcessLookupError, PermissionError, OSError):
                continue
        return found
    try:
        out = subprocess.run(
            ["ps", "-e", "-o", "pid=,pgid="],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    found = []
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid, pg = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if pg == pgid and pid not in excluded and pid > 1:
            found.append(pid)
    return found


def _ppid_map() -> dict[int, int]:
    """Best-effort pid -> ppid table (Linux /proc, else ps)."""
    table: dict[int, int] = {}
    if os.path.isdir("/proc"):
        try:
            entries = os.listdir("/proc")
        except OSError:
            return table
        for entry in entries:
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{pid}/stat") as handle:
                    data = handle.read()
            except (OSError, ValueError):
                continue
            tail = data.rfind(")")
            if tail == -1:
                continue
            rest = data[tail + 1:].split()
            if len(rest) < 2:
                continue
            try:
                table[pid] = int(rest[1])
            except ValueError:
                continue
        return table
    try:
        out = subprocess.run(
            ["ps", "-e", "-o", "pid=,ppid="],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return table
    if out.returncode != 0:
        return table
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            table[int(parts[0])] = int(parts[1])
        except ValueError:
            continue
    return table


def _descendant_pids(root_pid: int) -> list[int]:
    """Best-effort descendants of root_pid by walking ppid chains."""
    table = _ppid_map()
    if not table:
        return []
    children: dict[int, list[int]] = {}
    for pid, ppid in table.items():
        children.setdefault(ppid, []).append(pid)
    visited = {root_pid}
    queue = list(children.get(root_pid, []))
    result: list[int] = []
    while queue:
        pid = queue.pop()
        if pid in visited or pid <= 1:
            continue
        visited.add(pid)
        result.append(pid)
        queue.extend(children.get(pid, []))
    return result


def _cgroup_pids(cgroup_path: str | None, version: str | None) -> list[int]:
    """Best-effort member list of the dedicated cgroup (empty when disabled)."""
    if not cgroup_path or version not in ("v1", "v2"):
        return []
    names = ["cgroup.procs"] if version == "v2" else ["tasks", "cgroup.procs"]
    for name in names:
        try:
            with open(os.path.join(cgroup_path, name)) as handle:
                data = handle.read()
        except OSError:
            continue
        pids: list[int] = []
        for token in data.split():
            try:
                pid = int(token)
            except ValueError:
                continue
            if pid > 1:
                pids.append(pid)
        return pids
    return []


def _cgroup_signal(cgroup_path: str | None, version: str | None,
                   sig: signal.Signals,
                   skip: frozenset[int] = frozenset()) -> int:
    """Best-effort signal of every cgroup member.  Returns count signalled."""
    if not cgroup_path or version not in ("v1", "v2"):
        return 0
    # v2 fast path for SIGKILL; SIGTERM still needs per-pid signals because
    # cgroup.kill always issues SIGKILL.
    if version == "v2" and sig == signal.SIGKILL:
        kill_file = os.path.join(cgroup_path, "cgroup.kill")
        if os.path.exists(kill_file):
            try:
                with open(kill_file, "w") as handle:
                    handle.write("1")
            except OSError:
                pass
    count = 0
    for pid in _cgroup_pids(cgroup_path, version):
        if pid in skip:
            continue
        try:
            os.kill(pid, sig)
            count += 1
        except (ProcessLookupError, PermissionError, OSError):
            continue
    return count


def _setup_cgroup(child_pid: int) -> dict[str, Any]:
    """Create spo-sup-<supervisorpid> and enrol the child.  Best effort."""
    if sys.platform != "linux":
        return {"cgroup": "unsupported-platform",
                "cgroup_path": None, "version": None}
    if not os.path.isdir("/sys/fs/cgroup"):
        return {"cgroup": "unsupported-platform",
                "cgroup_path": None, "version": None}
    # cgroup v2: single hierarchy, membership via cgroup.procs.
    try:
        if os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
            path = f"/sys/fs/cgroup/spo-sup-{os.getpid()}"
            try:
                os.mkdir(path, 0o755)
            except FileExistsError:
                pass
            except OSError as exc:
                return {"cgroup": f"unavailable: mkdir {path}: "
                                  f"{exc.strerror or exc}",
                        "cgroup_path": None, "version": None}
            try:
                with open(os.path.join(path, "cgroup.procs"), "w") as handle:
                    handle.write(str(child_pid))
            except OSError as exc:
                try:
                    os.rmdir(path)
                except OSError:
                    pass
                return {"cgroup": f"unavailable: write cgroup.procs: "
                                  f"{exc.strerror or exc}",
                        "cgroup_path": None, "version": None}
            return {"cgroup": "v2", "cgroup_path": path, "version": "v2"}
    except OSError as exc:
        return {"cgroup": f"unavailable: {exc.strerror or exc}",
                "cgroup_path": None, "version": None}
    # cgroup v1: per-controller hierarchy; use a writable memory controller.
    candidates: list[str] = []
    try:
        mem = "/sys/fs/cgroup/memory"
        if os.path.isdir(mem):
            candidates.append(mem)
        else:
            try:
                for entry in os.listdir("/sys/fs/cgroup"):
                    if "memory" in entry:
                        full = os.path.join("/sys/fs/cgroup", entry)
                        if os.path.isdir(full):
                            candidates.append(full)
            except OSError:
                pass
        if not candidates:
            return {"cgroup": "unavailable: no cgroup v2 marker and no v1 "
                              "memory controller found",
                    "cgroup_path": None, "version": None}
        last_err: Any = "unknown error"
        for base in candidates:
            path = os.path.join(base, f"spo-sup-{os.getpid()}")
            try:
                os.mkdir(path, 0o755)
            except OSError as exc:
                last_err = exc.strerror or exc
                continue
            written = False
            for name in ("cgroup.procs", "tasks"):
                target = os.path.join(path, name)
                if not os.path.exists(target):
                    continue
                try:
                    with open(target, "w") as handle:
                        handle.write(str(child_pid))
                    written = True
                    break
                except OSError as exc:
                    last_err = exc.strerror or exc
                    continue
            if not written:
                try:
                    os.rmdir(path)
                except OSError:
                    pass
                continue
            return {"cgroup": "v1", "cgroup_path": path, "version": "v1"}
        return {"cgroup": f"unavailable: cannot create cgroup under v1 "
                          f"memory controller ({last_err})",
                "cgroup_path": None, "version": None}
    except OSError as exc:
        return {"cgroup": f"unavailable: {exc.strerror or exc}",
                "cgroup_path": None, "version": None}


def _cleanup_cgroup(cgroup_path: str | None) -> None:
    """Remove the dedicated cgroup dir (best effort, after empty)."""
    if not cgroup_path:
        return
    try:
        os.rmdir(cgroup_path)
    except OSError:
        pass


def _setup_subreaper() -> dict[str, Any]:
    """Mark this supervisor as a child subreaper (Linux only, best effort).

    Nearest-subreaper-ancestor semantics cover the whole tree, so setting
    it once on the supervisor itself is enough; there is no need to set it
    in children.  When active, orphans reparented away from the dead leader
    get ppid == supervisor pid and stay findable deterministically.
    """
    if sys.platform != "linux":
        return {"subreaper": False, "reason": "non-linux platform"}
    try:
        import ctypes
    except Exception as exc:  # pragma: no cover - defensive
        return {"subreaper": False, "reason": f"ctypes unavailable: {exc}"}
    try:
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
        except OSError:
            libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.restype = ctypes.c_int
        try:
            libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong,
                                   ctypes.c_ulong, ctypes.c_ulong,
                                   ctypes.c_ulong]
        except AttributeError:
            pass
        ret = libc.prctl(36, 1, 0, 0, 0)
    except OSError as exc:
        return {"subreaper": False, "reason": f"prctl OSError: {exc}"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"subreaper": False, "reason": f"prctl failed: {exc}"}
    if ret != 0:
        try:
            import ctypes as _ctypes

            err = _ctypes.get_errno()
        except Exception:
            err = -1
        return {"subreaper": False,
                "reason": f"prctl returned {ret} errno {err}"}
    return {"subreaper": True, "reason": None}


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


def _kill_now(proc: subprocess.Popen[bytes],
              cgroup_path: str | None = None,
              cgroup_version: str | None = None) -> None:
    """Last-resort cleanup when supervision cannot continue (setup failure)."""
    try:
        pgid = proc.pid
        if pgid > 1 and pgid != os.getpgrp():
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    else:
        try:
            proc.kill()
        except OSError:
            pass
    _cgroup_signal(cgroup_path, cgroup_version, signal.SIGKILL)
    # Best-effort descendant sweep (covers group members that already left
    # the group when no cgroup is active, plus subreaper-adopted orphans
    # whose ppid is now the supervisor).
    try:
        for pid in _group_pids(proc.pid, exclude=(os.getpid(),)):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                continue
        for pid in _descendant_pids(proc.pid):
            if pid <= 1 or pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                continue
        for pid in _descendant_pids(os.getpid()):
            if pid <= 1 or pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                continue
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
    supervisor_pid = os.getpid()
    known_pids: set[int] = set()
    try:
        known_pids.add(int(proc.pid))
    except (TypeError, ValueError):
        pass
    adopted_seen: set[int] = set()
    try:
        for value in record.get("adopted_pids_seen") or []:
            adopted_seen.add(int(value))
    except (TypeError, ValueError):
        pass
    subreaper_active = bool(record.get("subreaper"))

    def cgroup_path() -> str | None:
        value = record.get("cgroup_path")
        return str(value) if value else None

    def cgroup_version() -> str | None:
        value = record.get("cgroup")
        if value in ("v1", "v2"):
            return str(value)
        return None

    def cgroup_enabled() -> bool:
        return bool(cgroup_path() and cgroup_version() in ("v1", "v2"))

    def group_members() -> list[int]:
        return _group_pids(pgid, exclude=(supervisor_pid,))

    def _reap_adopted() -> None:
        """Reap reparented (adopted) children without touching Popen's own.

        Only call after proc.poll() is not None: Popen has already reaped
        the direct child, so waitpid(-1, WNOHANG) can only return adopted
        orphans.  A zombie answers kill(pid, 0), so without this the tree
        would never drain.
        """
        if proc.poll() is None:
            return
        try:
            while True:
                try:
                    pid, _status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                except OSError:
                    break
                if pid == 0:
                    break
                if pid > 0:
                    known_pids.discard(pid)
                else:
                    break
        except OSError:
            pass

    def tree_pids() -> list[int]:
        """Union of group/cgroup members, snapshots, and adopted orphans."""
        members: set[int] = set()
        try:
            members.update(group_members())
        except OSError:
            pass
        if cgroup_enabled():
            try:
                members.update(_cgroup_pids(cgroup_path(), cgroup_version()))
            except OSError:
                pass
        child_alive = proc.poll() is None
        if not child_alive and subreaper_active and sys.platform == "linux":
            _reap_adopted()
        if child_alive:
            # Snapshot while the child lives: after it is reaped, orphans
            # reparent (ppid 1, or ppid == supervisor with subreaper) so a
            # fresh ppid walk from the dead root finds nothing.  The
            # snapshot preserves setsid escapees observed before the exit
            # for the post-exit sweep.
            try:
                for pid in _descendant_pids(proc.pid):
                    known_pids.add(pid)
            except OSError:
                pass
            try:
                for pid in group_members():
                    known_pids.add(pid)
            except OSError:
                pass
        if subreaper_active and sys.platform == "linux":
            # Deterministic adoption scan: with PR_SET_CHILD_SUBREAPER on
            # the supervisor, reparented orphans get ppid == supervisor pid
            # no matter how fast fork+setsid+leader-exit happened.  No
            # snapshot timing is involved.  Walk the ppid table once from
            # the supervisor to cover adopted orphans plus their children.
            try:
                table = _ppid_map()
            except OSError:
                table = {}
            if table:
                offspring: dict[int, list[int]] = {}
                for child_pid, parent_pid in table.items():
                    offspring.setdefault(parent_pid, []).append(child_pid)
                seen: set[int] = set()
                queue = list(offspring.get(supervisor_pid, []))
                while queue:
                    desc = queue.pop()
                    if desc in seen or desc <= 1 or desc == supervisor_pid:
                        continue
                    seen.add(desc)
                    queue.extend(offspring.get(desc, []))
                for desc in seen:
                    if desc == supervisor_pid or desc <= 1:
                        continue
                    known_pids.add(desc)
                    if table.get(desc) == supervisor_pid and desc != proc.pid:
                        adopted_seen.add(desc)
                if adopted_seen:
                    record["adopted_pids_seen"] = sorted(adopted_seen)
        survivors: set[int] = set(members)
        for pid in list(known_pids):
            if pid == supervisor_pid or pid <= 1:
                known_pids.discard(pid)
                continue
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                known_pids.discard(pid)
                continue
            except PermissionError:
                pass
            except OSError:
                pass
            survivors.add(pid)
        survivors.discard(supervisor_pid)
        return sorted(survivors)

    def tree_empty() -> bool:
        return len(tree_pids()) == 0

    def sweep_tree(sig: signal.Signals,
                   skip: frozenset[int] = frozenset()) -> None:
        """Best-effort signal of the whole tree (group + cgroup + known).

        Pids in ``skip`` already received this round's signal (e.g. via the
        group killpg) and are not signalled again: a soft stop delivers
        exactly one TERM per process, while a hard stop keeps full overlap.
        """
        deliver(sig)
        if cgroup_enabled():
            try:
                _cgroup_signal(cgroup_path(), cgroup_version(), sig, skip)
            except OSError:
                pass
        try:
            for pid in tree_pids():
                if pid == supervisor_pid or pid <= 1 or pid in skip:
                    continue
                try:
                    os.kill(pid, sig)
                except (ProcessLookupError, PermissionError, OSError):
                    continue
        except OSError:
            pass

    def update_drain_records() -> None:
        try:
            group_now = group_members()
        except OSError:
            group_now = []
        if cgroup_enabled():
            try:
                cgroup_now = _cgroup_pids(cgroup_path(), cgroup_version())
            except OSError:
                cgroup_now = []
            record["cgroup_drained"] = len(cgroup_now) == 0
        else:
            try:
                record["cgroup_drained"] = len(tree_pids()) == 0
            except OSError:
                record["cgroup_drained"] = False
        try:
            remaining = tree_pids()
        except OSError:
            remaining = []
        record["group_drained"] = len(group_now) == 0
        record["remaining_pids"] = remaining

    def wait_for_drain(deadline_mono: float) -> str:
        """Wait until the tree is empty or deadline_mono.  Returns a tag."""
        while True:
            if tree_empty():
                return "drained"
            if _STOP_REQUESTS:
                return "stop"
            remaining = deadline_mono - time.monotonic()
            if remaining <= 0:
                return "timeout"
            try:
                time.sleep(min(POLL_SECONDS, remaining))
            except KeyboardInterrupt:
                return "keyboard"

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
        update_drain_records()
        return finish(code, state)

    def finish_after_exit(wait_code: int) -> int:
        """Drain-aware completion once the direct child has been reaped."""
        try:
            initial_tree = tree_pids()
        except OSError:
            initial_tree = []
        try:
            initial_group = group_members()
        except OSError:
            initial_group = []
        if initial_tree:
            # Durable proof the supervisor saw survivors instead of
            # finishing at leader exit (P1a).
            record["tree_pids_at_child_exit"] = list(initial_tree)
            record["group_pids_at_child_exit"] = list(initial_group)
            update_drain_records()
            persist()
        else:
            return finish_exit(wait_code)
        # The leader is gone but the tree is not: keep the clock.  When the
        # soft stop has not been sent yet, an early drain still wins; at the
        # soft deadline the survivors get SIGTERM; at the hard deadline they
        # get SIGKILL and the run is reported as hard_killed, never success.
        if record.get("soft_stop_sent_unix") is None:
            now = time.monotonic()
            if now < soft_deadline:
                outcome = wait_for_drain(soft_deadline)
                if outcome == "drained":
                    return finish_exit(wait_code)
                if outcome == "stop":
                    return hard_kill(_STOP_REQUESTS.pop(0))
                if outcome == "keyboard":
                    return hard_kill("keyboard_interrupt")
            if tree_empty():
                return finish_exit(wait_code)
            if _STOP_REQUESTS:
                return hard_kill(_STOP_REQUESTS.pop(0))
            record["soft_stop_sent_unix"] = round(now_unix(), 3)
            record["soft_stop_delivery"] = deliver(signal.SIGTERM)
            sweep_tree(signal.SIGTERM)
            record["state"] = "soft_stopped"
            persist()
        outcome = wait_for_drain(hard_deadline)
        if outcome == "drained":
            return finish_exit(wait_code)
        if outcome == "stop":
            return hard_kill(_STOP_REQUESTS.pop(0))
        if outcome == "keyboard":
            return hard_kill("keyboard_interrupt")
        return hard_kill(None)

    def hard_kill(forced_by: str | None) -> int:
        record["hard_kill_sent_unix"] = round(now_unix(), 3)
        record["hard_kill_delivery"] = deliver(signal.SIGKILL)
        sweep_tree(signal.SIGKILL)
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
        # Bounded post-kill sweep: the leader may be reaped while group or
        # cgroup orphans (including setsid escapees) survive.  Loop the sweep
        # until the tree is empty or the bound expires, with short sleeps.
        post_deadline = time.monotonic() + REAP_SECONDS
        while True:
            if tree_empty():
                break
            if time.monotonic() >= post_deadline:
                try:
                    sweep_tree(signal.SIGKILL)
                except OSError:
                    pass
                break
            try:
                sweep_tree(signal.SIGKILL)
            except OSError:
                pass
            try:
                time.sleep(POLL_SECONDS)
            except KeyboardInterrupt:
                break
        record["wait_code"] = wait_code
        record["reaped"] = wait_code is not None
        update_drain_records()
        return finish(EXIT_HARD_KILL, "hard_killed")

    def soft_stop(trigger: str | None) -> int:
        record["soft_stop_sent_unix"] = round(now_unix(), 3)
        # One TERM per process: snapshot the group first, killpg it once via
        # deliver(), then sweep only the pids the group kill could not cover.
        # The leader is always in the covered set: deliver() falls back to a
        # direct leader signal when the group is already gone.  (A pid forked
        # into the group between the snapshot and the killpg may still get a
        # second TERM from the fresh sweep; that window is microseconds.)
        try:
            covered: set[int] = set(group_members())
        except OSError:
            covered = set()
        covered.add(proc.pid)
        record["soft_stop_delivery"] = deliver(signal.SIGTERM)
        sweep_tree(signal.SIGTERM, skip=frozenset(covered))
        record["soft_stop_term_once"] = True
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
                return finish_after_exit(wait_code)
            if _STOP_REQUESTS:
                return hard_kill(_STOP_REQUESTS.pop(0))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Short sleeps detect an early exit promptly without busy-spinning
            # (about 20 wakeups/second while waiting).  A second Ctrl-C during
            # the grace cuts it short and forces the hard kill immediately.
            # Snapshot descendants so post-exit orphans stay findable after
            # they reparent.
            try:
                for pid in _descendant_pids(proc.pid):
                    known_pids.add(pid)
            except OSError:
                pass
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
                return finish_after_exit(wait_code)
            if _STOP_REQUESTS:
                trigger = _STOP_REQUESTS.pop(0)
                record["interrupted_unix"] = round(now_unix(), 3)
                break
            remaining = soft_deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                for pid in _descendant_pids(proc.pid):
                    known_pids.add(pid)
            except OSError:
                pass
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
        # which case a drained tree is a normal exit; survivors still wait
        # for the drain / hard kill instead of finishing at leader exit.
        wait_code = proc.poll()
        if wait_code is not None:
            return finish_after_exit(wait_code)
        return soft_stop(trigger)
    except KeyboardInterrupt:
        # Fallback for an interrupt raised outside the sleep sites (e.g.
        # during a JSON write).  The sleep sites handle the normal path and
        # soft_stop handles a second Ctrl-C.  If a soft stop was already
        # sent, escalate immediately instead of waiting another grace.
        wait_code = proc.poll()
        if wait_code is not None:
            return finish_after_exit(wait_code)
        if record.get("soft_stop_sent_unix") is None:
            record["interrupted_unix"] = round(now_unix(), 3)
            return soft_stop("keyboard_interrupt")
        return hard_kill("keyboard_interrupt")
    except BaseException:
        # A supervisor bug must never leave the session running past its
        # budget.  Kill the tree, then let the original error propagate.
        try:
            sweep_tree(signal.SIGKILL)
            if proc.poll() is None:
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

    # Subreaper first: nearest-subreaper-ancestor semantics cover the whole
    # tree, so setting PR_SET_CHILD_SUBREAPER on the supervisor itself (no
    # need to set it in children) makes fast fork+setsid+leader-exit orphans
    # reparent to this pid deterministically.  Best effort; any failure is
    # recorded and the supervisor continues with the snapshot sweep.
    subreaper_info = _setup_subreaper()

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
        "subreaper": bool(subreaper_info.get("subreaper")),
        "subreaper_reason": subreaper_info.get("reason"),
        "adopted_pids_seen": [],
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
    # Contain the tree in a dedicated cgroup (Linux only, best effort) so
    # setsid escapees are still swept.  Enrol immediately: grandchildren
    # forked after this inherit membership, including across setsid.
    cgroup_info = _setup_cgroup(proc.pid)
    record["cgroup"] = cgroup_info.get("cgroup")
    record["cgroup_path"] = cgroup_info.get("cgroup_path")
    record["cgroup_drained"] = False
    record["group_drained"] = False
    record["remaining_pids"] = []
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
        _kill_now(proc, record.get("cgroup_path"),
                  (record.get("cgroup")
                   if record.get("cgroup") in ("v1", "v2") else None))
        _cleanup_cgroup(record.get("cgroup_path"))
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
        _kill_now(proc, record.get("cgroup_path"),
                  (record.get("cgroup")
                   if record.get("cgroup") in ("v1", "v2") else None))
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
    finally:
        # Remove the dedicated cgroup dir on clean exit (best effort, after
        # empty — rmdir fails while members remain, which is fine).
        _cleanup_cgroup(record.get("cgroup_path"))

    _print_summary(record, session_file)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
