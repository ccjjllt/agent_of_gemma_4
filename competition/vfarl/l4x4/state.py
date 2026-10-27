"""Real task-root snapshots and bounded shell isolation for VFARL collection.

Call configure_isolation before creating/using a manager. The collector must
serialize each task's tool requests and capture between its completed calls. A bwrap probe
failure is an observable-only fallback, never evidence of complete restoration.
Host clock/entropy and filesystem inode/ctime are explicitly outside the scope.
"""
from __future__ import annotations

from contextlib import ExitStack
import ast
import dataclasses
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time


_LOCK = threading.RLock()
_TASK_LOCKS = {}
_ORIGINAL = None
_ORIGINAL_STOP = None
_MODULE = None
_CONFIG = {"requested_mode": "observable", "effective_mode": "observable",
           "reason": "Isolation has not been configured", "probe_passed": False}
_ROOT_CALLS = {}
_OBSERVABLE_ROOTS = set()
_CONTEXT_FIELDS = (
    "tool_calls_used", "llm_calls_used", "submitted_patch", "patch_submitted",
    "current_phase", "current_activity", "_handled_event_ids", "_total_cost_usd",
    "_total_tokens", "max_time_minutes", "max_tool_calls", "max_exec_seconds",
    "max_stdout_chars",
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _encode(value):
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(item) for item in value]}
    if isinstance(value, set):
        return {"__set__": sorted((_encode(item) for item in value), key=_canonical)}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unsupported live context value: " + type(value).__name__)


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return tuple(_decode(item) for item in value["__tuple__"])
        if set(value) == {"__set__"}:
            return set(_decode(item) for item in value["__set__"])
        return {key: _decode(item) for key, item in value.items()}
    return [_decode(item) for item in value] if isinstance(value, list) else value


def _policy(value):
    # Branch RNG belongs to the intervention, not to environment state identity.
    if isinstance(value, dict):
        return {key: _policy(item) for key, item in value.items()
                if key not in {"branch_seed", "sampling_seed", "seed"}}
    return [_policy(item) for item in value] if isinstance(value, list) else value


def _bwrap_command(binary, root, command, hidden_paths=()):
    root = Path(root).resolve()
    hidden = sorted({Path(value).resolve() for value in hidden_paths}, key=lambda path: (len(path.parts), str(path)))
    # A parent mask already hides all descendants; applying another mask under
    # it would require recreating private host layout for no useful reason.
    masks = []
    for path in hidden:
        if path == root or path.is_relative_to(root):
            raise ValueError("Privileged hidden data cannot be inside the writable task root")
        if not any(path.is_relative_to(parent) for parent in masks):
            masks.append(path)
    result = [binary, "--die-with-parent", "--new-session", "--unshare-user",
              "--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts",
              "--ro-bind", "/", "/"]
    for path in masks:
        result += ["--tmpfs", str(path)]
        if root.is_relative_to(path):
            directory = path
            for part in root.relative_to(path).parts[:-1]:
                directory /= part
                result += ["--dir", str(directory)]
    # Re-expose exactly this task root, even when its host parent is hidden.
    result += ["--bind", str(root), str(root)]
    for path in masks:
        result += ["--remount-ro", str(path)]
    return result + ["--proc", "/proc", "--dev", "/dev", "--", *command]


def configure_isolation(mode="bwrap", hidden_paths=None):
    """Install a wrapper at the official command executor, returning real status.

    Tool schemas, path translation, budgets, output clipping and ExecutionResult
    handling remain official. bwrap changes the execution environment and is
    recorded as such; this does not establish competition Docker equivalence.
    """
    global _ORIGINAL, _ORIGINAL_STOP, _MODULE, _CONFIG
    if mode not in {"bwrap", "chroot", "observable"}:
        raise ValueError("mode must be bwrap, chroot or observable")
    hidden = sorted({str(Path(value).resolve()) for value in (hidden_paths or [])})
    if any(path == "/" or not Path(path).is_dir() for path in hidden):
        raise ValueError("Hidden paths must be existing directories other than filesystem root")
    with _LOCK:
        config = {"requested_mode": mode, "effective_mode": "observable",
                  "probe_passed": False, "reason": "Observable mode explicitly selected",
                  "hidden_paths": hidden}
        if mode == "chroot":
            from competition.vfarl.l4x4 import jail
            config.update(jail.configure(hidden))
        if mode == "bwrap":
            binary = shutil.which("bwrap")
            config["binary"] = binary
            if sys.platform != "linux" or not binary:
                config["reason"] = "Linux bubblewrap is unavailable"
            else:
                with tempfile.TemporaryDirectory(prefix="vfarl-bwrap-") as directory:
                    private = Path(directory).resolve() / "private"
                    root = private / "sandboxes" / "task"
                    root.mkdir(parents=True)
                    (private / "hidden-marker").write_text("privileged host data")
                    # Exercise the real nested-output layout, confidentiality,
                    # writable task root and read-only hidden parent together.
                    command = ["/bin/sh", "-c", 'printf ok > "$1/probe" && test "$(cat "$1/probe")" = ok && test ! -e "$2/hidden-marker" && ! touch "$2/outside-write" 2>/dev/null', "probe", str(root), str(private)]
                    try:
                        result = subprocess.run(_bwrap_command(binary, root, command, [*hidden, private]),
                                                capture_output=True, timeout=15, check=False)
                        config.update({"probe_exit_code": result.returncode,
                                       "probe_stderr": result.stderr.decode(errors="replace")[-3000:]})
                        if result.returncode == 0 and (root / "probe").read_text() == "ok":
                            config.update({"effective_mode": "bwrap", "probe_passed": True,
                                           "binary_sha256": _sha(binary), "hidden_path_probe_passed": True,
                                           "reason": "Actual namespace, hidden-host-data and writable-root probe succeeded"})
                        else:
                            config["reason"] = "Actual bubblewrap namespace probe failed"
                    except (OSError, subprocess.TimeoutExpired) as error:
                        config["reason"] = type(error).__name__ + ": " + str(error)
        try:
            module = importlib.import_module("swegemma.sandbox.subprocess")
        except ImportError as error:
            config.update({"effective_mode": "observable", "reason": "Official manager import unavailable: " + str(error)})
            _CONFIG = config
            return dict(config)
        if _ORIGINAL is None:
            _ORIGINAL = module.execute_subprocess_command
            _MODULE = module
        elif module is not _MODULE:
            raise RuntimeError("Official manager module changed after isolation installation")
        _CONFIG = config
        module.execute_subprocess_command = _execute
        if _ORIGINAL_STOP is None:
            _ORIGINAL_STOP = module.SubprocessManager.stop
            module.SubprocessManager.stop = _stop
        return dict(config)


def _stop(manager, sandbox_id):
    paths = manager.sandboxes.get(sandbox_id)
    if paths:
        from competition.vfarl.l4x4 import jail
        # Cleanup must happen while the original root symlink is still known.
        with _task_lock(paths["root"]):
            jail.cleanup(paths["root"])
            return _ORIGINAL_STOP(manager, sandbox_id)
    return _ORIGINAL_STOP(manager, sandbox_id)


def _task_lock(root):
    # Manager paths retain the logical root when chroot materialization moves
    # its contents behind a symlink. Do not resolve that symlink into a second
    # lock identity. Independent task roots can run official tools in parallel.
    key = os.path.abspath(root)
    with _LOCK:
        return _TASK_LOCKS.setdefault(key, threading.RLock())


def _execute(*args, **kwargs):
    bound = inspect.signature(_ORIGINAL).bind(*args, **kwargs)
    bound.apply_defaults()
    cwd = Path(bound.arguments["cwd"]).resolve()
    root = cwd.parent
    if cwd.name != "workspace" or not (root / "venv").is_dir() or not (root / "tmp").is_dir():
        raise ValueError("Cannot establish the official task root for shell isolation")
    with _LOCK:
        config = dict(_CONFIG)
    logical_root = Path(bound.arguments["cwd"]).absolute().parent
    with _task_lock(logical_root):
        if config["effective_mode"] == "chroot":
            from competition.vfarl.l4x4 import jail
            if bound.arguments.get("shell") or not isinstance(bound.arguments["command"], (list, tuple)):
                raise ValueError("Expected the official shell=False argument-vector executor")
            result, info = jail.execute(logical_root, bound.arguments, _ORIGINAL)
            actual_root = info["physical_root"]
            _ROOT_CALLS[actual_root] = _ROOT_CALLS.get(actual_root, 0) + 1
            return result
        _ROOT_CALLS[str(root)] = _ROOT_CALLS.get(str(root), 0) + 1
        if config["effective_mode"] == "bwrap":
            command = bound.arguments["command"]
            if bound.arguments.get("shell") or not isinstance(command, (list, tuple)):
                raise ValueError("Expected the official shell=False argument-vector executor")
            bound.arguments["command"] = _bwrap_command(config["binary"], root, command,
                                                        config.get("hidden_paths", []))
        else:
            _OBSERVABLE_ROOTS.add(str(root))
        return _ORIGINAL(*bound.args, **bound.kwargs)


def _inventory(root):
    entries, hardlinks = [], {}

    def visit(path):
        info = path.lstat()
        relative = str(path.relative_to(root))
        row = {"path": relative, "mode": stat.S_IMODE(info.st_mode),
               "uid": info.st_uid, "gid": info.st_gid, "mtime_ns": info.st_mtime_ns}
        if stat.S_ISLNK(info.st_mode):
            row.update({"kind": "symlink", "target": os.readlink(path)})
        elif stat.S_ISDIR(info.st_mode):
            row["kind"] = "directory"
        elif stat.S_ISREG(info.st_mode):
            identity = (info.st_dev, info.st_ino)
            row.update({"kind": "file", "size": info.st_size, "sha256": _sha(path)})
            if identity in hardlinks:
                row["hardlink_to"] = hardlinks[identity]
            else:
                hardlinks[identity] = relative
        else:
            raise ValueError("Live socket/device/FIFO cannot be captured as a complete task root: " + relative)
        entries.append(row)
        if row["kind"] == "directory":
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                visit(child)

    visit(root)
    return entries


def _copy_tree(source, destination):
    hardlinks = {}

    def copy_file(src, dst):
        info = os.stat(src, follow_symlinks=False)
        key = (info.st_dev, info.st_ino)
        if key in hardlinks:
            os.link(hardlinks[key], dst)
        else:
            shutil.copy2(src, dst, follow_symlinks=False)
            os.chown(dst, info.st_uid, info.st_gid, follow_symlinks=False)
            hardlinks[key] = dst
        return dst

    shutil.copytree(source, destination, symlinks=True, copy_function=copy_file)
    for directory, dirs, files in os.walk(source, followlinks=False):
        for path in [Path(directory), *(Path(directory) / name for name in dirs + files)]:
            info = path.lstat()
            os.lchown(Path(destination) / path.relative_to(source), info.st_uid, info.st_gid)


def _remove_directory(path):
    def make_writable_and_retry(function, failed, exception):
        target = Path(failed)
        if target.is_symlink():
            raise exception[1]
        target.chmod(stat.S_IMODE(target.stat().st_mode) | stat.S_IRWXU)
        function(failed)

    # A branch can legitimately chmod a source/cache directory read-only.
    # Only the already-validated task tree is affected by this recovery step.
    for directory, subdirs, _ in os.walk(path, followlinks=False):
        current = Path(directory)
        current.chmod(stat.S_IMODE(current.stat().st_mode) | stat.S_IRWXU)
    shutil.rmtree(path, onerror=make_writable_and_retry)


def _task_runtime_bindings(paths):
    """Bind task-specific external profiles referenced by the snapshotted venv."""
    root = Path(paths["root"]).resolve()
    binaries, profiles, queued, seen = {}, {}, [], set()
    for binary in sorted(Path(paths["venv"]).glob("bin/python*")):
        target = binary.resolve()
        if target.is_file() and not target.is_relative_to(root):
            binaries[str(target)] = _sha(target)

    def inspect_pth(path):
        text = path.read_text(errors="replace")
        candidates = []
        for line in text.splitlines():
            if line.strip().startswith("import "):
                try:
                    candidates.extend(node.value for node in ast.walk(ast.parse(line))
                                      if isinstance(node, ast.Constant) and isinstance(node.value, str)
                                      and os.path.isabs(node.value))
                except SyntaxError:
                    continue
            elif line.strip() and not line.startswith("#"):
                candidates.append(line.strip())
        for value in candidates:
            if any(character in value for character in ("\n", "\r", "\0")) or len(value) > 4096:
                continue
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            candidate = candidate.resolve()
            try:
                if candidate.is_dir() and not candidate.is_relative_to(root):
                    queued.append(candidate)
            except OSError:
                # Arbitrary strings in executable .pth code are not paths.
                continue

    for pth in sorted(Path(paths["venv"]).glob("lib/python*/site-packages/*.pth")):
        inspect_pth(pth)
    while queued:
        profile = queued.pop()
        if profile in seen:
            continue
        seen.add(profile)
        if len(seen) > 128:
            raise ValueError("Task runtime has unexpectedly many external dependency roots")
        identities = {}
        for pattern in ("*.dist-info/RECORD", "*.dist-info/METADATA", "*.dist-info/direct_url.json", "*.pth"):
            for path in sorted(profile.glob(pattern)):
                if path.is_file():
                    identities[str(path.relative_to(profile))] = _sha(path)
                    if path.suffix == ".pth":
                        inspect_pth(path)
        profiles[str(profile)] = identities
    return {"external_python_binaries": binaries, "external_profile_metadata": profiles}


def _runtime(manager, paths):
    from competition.vfarl.l4x4 import jail
    sources = {}
    for name in ("swegemma.sandbox.subprocess", "swegemma.context", "swegemma.budget",
                 "swegemma.tools", "swegemma.tools.base", "swegemma.tools.execution",
                 "swegemma.tools.workspace", "swegemma.harness.verification",
                 "swegemma.harness.container_setup"):
        try:
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            sources[name] = {"path": str(path), "sha256": _sha(path)}
        except ImportError:
            sources[name] = None
    if _ORIGINAL is not None:
        path = inspect.getsourcefile(_ORIGINAL)
        sources["original_command_executor"] = {"path": path, "sha256": _sha(path)} if path else None
    for method in ("start", "stop", "exec"):
        path = inspect.getsourcefile(getattr(manager, method))
        sources["bound_manager_" + method] = {"path": path, "sha256": _sha(path)} if path else None
    packages = []
    for dist in importlib.metadata.distributions():
        record = dist.read_text("RECORD")
        packages.append({"name": dist.metadata.get("Name", ""), "version": dist.version,
                         "installed_record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None})
    packages.sort(key=lambda item: (item["name"].lower(), item["version"]))
    # Values are hashed so credentials cannot enter the snapshot manifest.
    environment = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in sorted(os.environ.items())}
    return {"python": sys.version, "python_executable": str(Path(sys.executable).resolve()),
            "python_executable_sha256": _sha(Path(sys.executable).resolve()),
            "platform": platform.platform(), "official_sources": sources,
            "package_identity": packages, "host_environment_value_hashes": environment,
            "task_runtime_bindings": _task_runtime_bindings(paths),
            "chroot_runtime": jail.identity(paths["root"]),
            "sys_path": list(sys.path), "manager_timeout_seconds": manager.timeout_seconds,
            "manager_system_site_packages": manager.system_site_packages,
            "manager_paths": {key: str(Path(value).resolve()) for key, value in paths.items()},
            "isolation": dict(_CONFIG), "snapshot_implementation_sha256": _sha(__file__)}


def _context(ctx):
    now = time.perf_counter()
    result = {"fields": {key: _encode(getattr(ctx, key)) for key in _CONTEXT_FIELDS if hasattr(ctx, key)},
              "budget": dataclasses.asdict(ctx.budget), "harness": dataclasses.asdict(ctx.harness),
              "task_elapsed_seconds": now - ctx.task_start_time,
              "start_elapsed_seconds": now - ctx.start_time,
              "agent_started": ctx.agent_start_time is not None,
              "agent_stopped": ctx.agent_end_time is not None,
              "agent_elapsed_seconds": ((ctx.agent_end_time if ctx.agent_end_time is not None else now) - ctx.agent_start_time)
              if ctx.agent_start_time is not None else 0.0,
              "task_identity": {"task_id": getattr(ctx, "task_id", None), "repo": getattr(ctx, "repo", None)},
              "token_budget_attached": getattr(ctx, "_token_budget", None) is not None}
    return result


def _restore_context(ctx, saved):
    for key, value in saved["fields"].items():
        setattr(ctx, key, _decode(value))
    ctx.budget = type(ctx.budget)(**saved["budget"])
    ctx.harness = type(ctx.harness)(**saved["harness"])
    now = time.perf_counter()
    ctx.task_start_time = now - saved["task_elapsed_seconds"]
    ctx.start_time = now - saved["start_elapsed_seconds"]
    ctx.agent_start_time = now - saved["agent_elapsed_seconds"] if saved["agent_started"] else None
    ctx.agent_end_time = now if saved["agent_stopped"] else None


def _locks(stack, manager, ctx, sandbox_id):
    # Official tools hold tool_lock before invoking the manager executor.
    for lock in (getattr(ctx, "tool_lock", None), getattr(manager, "_lock", None),
                 _task_lock(manager.sandboxes[sandbox_id]["root"]),
                 getattr(ctx, "_lock", None)):
        if lock is not None:
            stack.enter_context(lock)


class Snapshot:
    @staticmethod
    def capture(manager, sandbox_id, ctx, prompt_token_ids, ledger, policy_identity, destination):
        """Capture at a completed-tool boundary; return the persisted manifest."""
        destination = Path(destination).resolve()
        with ExitStack() as stack:
            _locks(stack, manager, ctx, sandbox_id)
            paths = manager.sandboxes[sandbox_id]
            root = Path(paths["root"]).resolve()
            if destination == root or destination.is_relative_to(root) or root.is_relative_to(destination):
                raise ValueError("Snapshot storage must be separate from the task root")
            if ctx.container_id != sandbox_id:
                raise ValueError("Context and sandbox IDs differ")
            ids = list(prompt_token_ids)
            if not ids or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("A real complete model-visible token prefix is required")
            if not policy_identity or not isinstance(ledger, dict):
                raise ValueError("Policy identity and original/consumed/remaining ledger are required")
            inventory = _inventory(root)
            runtime = _runtime(manager, paths)
            context = _context(ctx)
            reasons = []
            from competition.vfarl.l4x4 import jail
            actual_chroot = _CONFIG["effective_mode"] == "chroot" and jail.complete(root)
            if _CONFIG["effective_mode"] != "bwrap" and not actual_chroot:
                reasons.append("Shell PID/network/filesystem isolation is not active")
            if not _CONFIG.get("hidden_paths") and not actual_chroot:
                reasons.append("No privileged host asset/output/code directories have been hidden from task commands")
            if not _ROOT_CALLS.get(str(root)):
                reasons.append("No task-root execution through the installed isolation wrapper was observed")
            if str(root) in _OBSERVABLE_ROOTS:
                reasons.append("This task root previously executed a nonisolated command")
            if context["token_budget_attached"]:
                reasons.append("An external mutable TokenBudget object is attached and is not restorable")
            if any(value is None for value in runtime["official_sources"].values()):
                reasons.append("Some official runtime implementation identities are unavailable")
            scope = "observable_replay_only" if reasons else "complete_declared_scope"
            state = {"schema_version": "vfarl-task-state-v1", "task_root": str(root),
                     "filesystem": inventory, "context": context, "prompt_token_ids": ids,
                     "prompt_token_ids_sha256": _digest(ids), "ledger": _encode(ledger),
                     "policy_identity": _policy(_encode(policy_identity)), "runtime": runtime,
                     "restoration_scope": scope,
                     "declared_components": ["Full writable task root including workspace/.git/index, ignored/untracked files, tmp, wheels and venv; chroot immutable runtime is separately identified and not copied per snapshot", "File bytes, modes, owners, mtime, symlinks and internal hardlink relationships", "Exact model-visible token prefix, context tool/LLM counters, submitted patch and logical budgets", "Same-path task bindings, official tool/executor source hashes, Python and installed package identity including external task profiles referenced by venv .pth files", "No surviving child task processes: bwrap PID namespace termination or chroot unique-UID kill and quiescence verification", "Host asset/output/code visibility excluded by actual bwrap masks or by an allowlisted chroot runtime plus capability/UID/seccomp restrictions"],
                     "unrestorable_components": ["Host absolute clock, kernel entropy, scheduling and external read-only files not covered by runtime bindings", "Filesystem inode numbers, ctime and atime; copied file bytes/modes/mtime are restored", *reasons],
                     "external_runtime_boundary": "The host runtime is mounted read-only to task commands. Package versions/RECORD hashes identify dependencies; they are not a byte-for-byte snapshot of the entire host filesystem. The collector must not mutate host dependencies during this experiment.",
                     "caller_contract": "Collector serializes requests within each independent task; capture occurs with no in-flight model or tool request for this task. Other task roots may execute concurrently. Branch seeds are intervention metadata. Clock/entropy-sensitive tasks are outside complete declared scope.",
                     "competition_Docker_environment_equivalent": False}
            destination.mkdir(parents=True, exist_ok=False)
            destination.chmod(0o700)
            _copy_tree(root, destination / "root")
            if _inventory(destination / "root") != inventory or _inventory(root) != inventory:
                raise RuntimeError("Filesystem changed during capture or the copy did not preserve declared state")
            result = {"state_id": _digest(state), "state": state,
                      "restoration_scope": scope, "snapshot_directory": str(destination),
                      "captured_at_unix": time.time(), "root_command_count": _ROOT_CALLS.get(str(root), 0)}
            (destination / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            return result

    @staticmethod
    def restore(snapshot, manager, sandbox_id, ctx, *, prompt_token_ids=None, ledger=None, policy_identity=None):
        """Restore into the original root, removing every branch-created entry."""
        if isinstance(snapshot, (str, Path)):
            path = Path(snapshot)
            snapshot = json.loads((path / "manifest.json" if path.is_dir() else path).read_text())
        state = snapshot["state"]
        if _digest(state) != snapshot["state_id"]:
            raise ValueError("Snapshot manifest/state ID mismatch")
        stored = Path(snapshot["snapshot_directory"]) / "root"
        with ExitStack() as stack:
            _locks(stack, manager, ctx, sandbox_id)
            paths = manager.sandboxes[sandbox_id]
            root = Path(paths["root"]).resolve()
            if str(root) != state["task_root"] or ctx.container_id != sandbox_id:
                raise ValueError("Restoration requires the original task root path and sandbox context")
            if _runtime(manager, paths) != state["runtime"]:
                raise ValueError("Runtime, dependencies, environment or isolation changed since capture")
            if policy_identity is not None and _policy(_encode(policy_identity)) != state["policy_identity"]:
                raise ValueError("Continuation policy differs from the captured state")
            if prompt_token_ids is not None and list(prompt_token_ids) != state["prompt_token_ids"]:
                raise ValueError("Model-visible prefix differs from the captured state")
            if _inventory(stored) != state["filesystem"]:
                raise ValueError("Stored snapshot files differ from their manifest")
            if state["context"]["token_budget_attached"]:
                raise ValueError("Cannot restore an external mutable TokenBudget object")
            original_identity = state["context"]["task_identity"]
            if original_identity != {"task_id": getattr(ctx, "task_id", None), "repo": getattr(ctx, "repo", None)}:
                raise ValueError("Task identity changed since capture")
            # Preserve the root directory itself and all absolute venv/script paths.
            root.chmod(stat.S_IMODE(root.stat().st_mode) | stat.S_IRWXU)
            for entry in root.iterdir():
                if entry.is_dir() and not entry.is_symlink():
                    _remove_directory(entry)
                else:
                    entry.unlink()
            with tempfile.TemporaryDirectory(prefix="vfarl-restore-", dir=root.parent) as temporary:
                copied = Path(temporary) / "root"
                _copy_tree(stored, copied)
                for entry in copied.iterdir():
                    shutil.move(str(entry), str(root / entry.name))
                shutil.copystat(stored, root, follow_symlinks=False)
            if _inventory(root) != state["filesystem"]:
                raise RuntimeError("Restored root differs from the captured bytes or metadata")
            _restore_context(ctx, state["context"])
            restored_ledger = _decode(state["ledger"])
            if ledger is not None:
                ledger.clear()
                ledger.update(restored_ledger)
            result = {"state_id": snapshot["state_id"], "restoration_scope": state["restoration_scope"],
                      "filesystem_verified": True, "runtime_verified": True,
                      "logical_elapsed_restored_seconds": state["context"]["agent_elapsed_seconds"],
                      "prompt_token_ids": state["prompt_token_ids"], "ledger": restored_ledger,
                      "restored_at_unix": time.time(), "task_root": str(root),
                      "policy_identity": state["policy_identity"]}
            with (Path(snapshot["snapshot_directory"]) / "restores.jsonl").open("a") as stream:
                stream.write(json.dumps({key: value for key, value in result.items() if key != "prompt_token_ids"}, ensure_ascii=False) + "\n")
            return result


restore = Snapshot.restore
