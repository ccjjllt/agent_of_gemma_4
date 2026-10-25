"""Small Linux chroot executor for hosts that prohibit mount/user namespaces.

Only the task tree is writable. Runtime files are copied to a root-owned cache
and hardlinked from that cache, never hardlinked to mutable host originals.
The launcher is a fresh ``python -I -S`` process, not a preexec_fn in CUDA Python.
"""
from __future__ import annotations

import ast
import ctypes
import ctypes.util
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid


_JAILS = {}
_BUILDING = {}
_RUNTIME_CACHE = {}
_FILE_CACHE = {}
_HIDDEN = []
_PROBE = False
_COMMANDS = "bash sh git cat head tail sed awk grep find xargs ls stat wc sort uniq cut tr diff patch tar gzip bzip2 xz unzip cp mv rm mkdir rmdir touch chmod ln pwd env printenv id date sleep timeout kill tee which realpath readlink basename dirname".split()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)


def _processes(uid):
    found = []
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            values = dict(line.split(":", 1) for line in path.read_text().splitlines() if ":" in line)
            if uid in [int(value) for value in values["Uid"].split()]:
                found.append((int(path.parent.name), values["State"].strip()[0]))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found


def quiesce(info):
    """Kill every live process with this never-shared task UID, including daemons."""
    deadline = time.monotonic() + 5
    stable = 0
    while time.monotonic() < deadline:
        live = [(pid, state) for pid, state in _processes(info["uid"]) if state not in {"Z", "X"}]
        if not live:
            stable += 1
            if stable == 2:
                info["quiescent"] = True
                return
        else:
            stable = 0
            for pid, _ in live:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        time.sleep(0.05)
    info["quiescent"] = False
    raise RuntimeError("Task UID still has live processes; no complete state can be captured")


def _allocate_uid():
    leases = Path(tempfile.gettempdir()) / "vfarl-private-uid-leases"
    leases.mkdir(mode=0o700, exist_ok=True)
    if leases.stat().st_uid != 0 or stat.S_IMODE(leases.stat().st_mode) != 0o700:
        raise RuntimeError("UID lease directory must be private and root-owned")
    for _ in range(100):
        uid = 200000 + int(uuid.uuid4().hex[:6], 16)
        lease = leases / str(uid)
        try:
            lease.mkdir(mode=0o700)
        except FileExistsError:
            continue
        if not _processes(uid):
            return uid
    raise RuntimeError("Could not lease a unique task UID")


def _external_profiles(root):
    profiles = set()
    for pth in root.glob("venv/lib/python*/site-packages/*.pth"):
        for line in pth.read_text(errors="replace").splitlines():
            if line.strip().startswith("import "):
                try:
                    values = [node.value for node in ast.walk(ast.parse(line))
                              if isinstance(node, ast.Constant) and isinstance(node.value, str)
                              and os.path.isabs(node.value)]
                except SyntaxError:
                    raise ValueError("Unsupported executable task .pth file")
            else:
                values = [line.strip()] if line.strip() and not line.lstrip().startswith("#") else []
            for value in values:
                path = Path(value)
                path = (path if path.is_absolute() else pth.parent / path).resolve()
                if path.is_relative_to(root) or not path.exists():
                    continue
                if path.name not in {"site-packages", "dist-packages"}:
                    raise ValueError("External .pth path is not an explicit dependency profile: " + str(path))
                profiles.add(path)
    return sorted(profiles)


def _runtime(root):
    python = root / "venv/bin/python"
    if not python.exists():
        python = root / "venv/bin/python3"
    metadata = json.loads(subprocess.check_output([str(python), "-I", "-S", "-c",
        "import json,sys,sysconfig; print(json.dumps({'executable':sys.executable,'stdlib':sysconfig.get_path('stdlib')}))"], text=True))
    profiles = _external_profiles(root)
    stdlib = Path(metadata["stdlib"]).resolve()
    executable = python.resolve(strict=True)
    # Creating a venv from another venv can leave an absolute intermediate
    # interpreter symlink outside the copied runtime. Bash then skips that
    # broken executable and silently falls through to /usr/bin/python3, losing
    # the task .pth dependency profile. Require the caller's resolved-base venv.
    for candidate in (root / "venv/bin").glob("python*"):
        if not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", candidate.name):
            continue
        current, visited = candidate, set()
        while current.is_symlink():
            if str(current) in visited:
                raise ValueError("Cyclic task Python symlink: " + str(candidate))
            visited.add(str(current))
            target = Path(os.readlink(current))
            target = Path(os.path.abspath(target if target.is_absolute() else current.parent / target))
            if not target.is_relative_to(root) and target != executable:
                raise ValueError("Task Python has an uncopied intermediate interpreter link; create the venv from the resolved base executable: " + str(target))
            current = target
        if current.resolve(strict=True) != executable:
            raise ValueError("Task Python aliases do not share the declared base interpreter")
    key = (str(executable), str(stdlib), tuple(map(str, profiles)), tuple(map(str, _HIDDEN)))
    if key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key]
    sources = {}

    def add(path, destination=None):
        path = Path(path)
        actual = path.resolve()
        if any(actual == hidden or actual.is_relative_to(hidden) for hidden in _HIDDEN):
            raise ValueError("A runtime source overlaps privileged task/output/code material: " + str(actual))
        if actual.is_file():
            # ldd can spell $ORIGIN dependencies as package/../package.libs/x.
            # Normalize only lexical components; resolving destination symlinks
            # would erase valid loader alias names that must exist in the jail.
            target = os.path.abspath(os.fspath(destination if destination is not None else path))
            prior = sources.get(target)
            if prior is not None and not os.path.samefile(prior, actual):
                if sha(prior) != sha(actual) or bool(prior.stat().st_mode & 0o111) != bool(actual.stat().st_mode & 0o111):
                    raise ValueError("Conflicting runtime sources map to the same normalized destination: " + target)
            sources[target] = actual

    def tree(path):
        for directory, dirs, files in os.walk(path, followlinks=False):
            dirs[:] = [name for name in dirs if name not in {"__pycache__", ".git"}
                       and not (Path(directory) == stdlib and name in {"site-packages", "dist-packages"})]
            for name in files:
                if not name.endswith((".pyc", ".pyo")):
                    add(Path(directory) / name)
            for name in list(dirs):
                child = Path(directory) / name
                if child.is_symlink():
                    raise ValueError("External runtime directory symlink needs explicit packaging: " + str(child))

    tree(stdlib)
    for profile in profiles:
        tree(profile)
    add(executable)
    for name in _COMMANDS:
        path = shutil.which(name)
        if path:
            add(path)
            if name in {"bash", "sh"}:
                add(path, Path("/bin") / name)
    git = shutil.which("git")
    if git:
        git_core = Path(subprocess.check_output([git, "--exec-path"], text=True).strip())
        if git_core.is_dir():
            tree(git_core)
    for path in ("/etc/ld.so.cache", "/etc/nsswitch.conf", "/etc/localtime"):
        if Path(path).is_file():
            add(Path(path))
    # Collect transitive ELF dependencies, including dynamically imported Python
    # extension modules. No host library directory is granted wholesale.
    inspected = set()
    while True:
        pending = set(sources.values()) - inspected
        if not pending:
            break
        for path in pending:
            inspected.add(path)
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    continue
            result = subprocess.run(["ldd", str(path)], capture_output=True, text=True, timeout=20)
            for match in re.finditer(r"(?:=>\s*)?(/[^\s()]+)", result.stdout):
                dependency = Path(match.group(1))
                if dependency.is_file():
                    add(dependency)
    cache = Path(tempfile.mkdtemp(prefix="vfarl-readonly-runtime-"))
    records = []
    for destination, source in sorted(sources.items()):
        path = cache / destination.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = 0o555 if source.stat().st_mode & 0o111 else 0o444
        details = source.stat()
        file_key = (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns, mode)
        if file_key in _FILE_CACHE:
            cached, fingerprint = _FILE_CACHE[file_key]
            _link_or_copy(cached, path)
        else:
            shutil.copyfile(source, path)
            path.chmod(mode)
            fingerprint = sha(path)
            _FILE_CACHE[file_key] = (path, fingerprint)
        records.append({"path": destination, "sha256": fingerprint, "bytes": path.stat().st_size, "mode": mode})
    for alias in ("/usr/bin/python3", "/usr/bin/python"):
        if alias != str(executable):
            records.append({"path": alias, "kind": "symlink", "target": str(executable)})
    result = {"cache": str(cache), "files": records,
              "manifest_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
              "python_executable": str(executable), "dependency_profiles": list(map(str, profiles))}
    _RUNTIME_CACHE[key] = result
    return result


def ensure(root):
    root = Path(root).absolute()
    resolved = root.resolve()
    if str(resolved) in _JAILS:
        return _JAILS[str(resolved)]
    if os.geteuid() != 0 or sys.platform != "linux":
        raise RuntimeError("The chroot backend needs the CPU-proven Linux root capabilities")
    runtime = _runtime(resolved)
    uid = _allocate_uid()
    jail = Path(tempfile.mkdtemp(prefix="vfarl-task-jail-", dir=root.parent))
    _BUILDING[str(root)] = jail
    jail.chmod(0o755)
    for row in runtime["files"]:
        destination = jail / row["path"].lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if row.get("kind") == "symlink":
            destination.symlink_to(row["target"])
        else:
            _link_or_copy(Path(runtime["cache"]) / row["path"].lstrip("/"), destination)
    # The original absolute task path exists inside the chroot. Host-side tools
    # retain their original paths through this private symlink; no command-text
    # or Python .pth rewriting is needed for the ordinary official path mapping.
    physical = jail / str(root).lstrip("/")
    physical.parent.mkdir(parents=True, exist_ok=True)
    for name, major, minor, mode in (("null", 1, 3, 0o666), ("zero", 1, 5, 0o666),
                                    ("random", 1, 8, 0o444), ("urandom", 1, 9, 0o444)):
        device = jail / "dev" / name
        device.parent.mkdir(exist_ok=True)
        os.mknod(device, stat.S_IFCHR | mode, os.makedev(major, minor))
        device.chmod(mode)
    (jail / "tmp").mkdir(exist_ok=True)
    (jail / "dev/shm").symlink_to(str(root / "tmp/shm"), target_is_directory=True)
    (jail / "etc").mkdir(exist_ok=True)
    (jail / "etc/passwd").write_text(f"root:x:0:0:root:/root:/bin/bash\ntask:x:{uid}:{uid}:task:{root / 'workspace'}:/bin/bash\n")
    (jail / "etc/group").write_text(f"root:x:0:\ntask:x:{uid}:\n")
    (jail / "etc/hosts").write_text("127.0.0.1 localhost\n::1 localhost\n")
    for name in ("passwd", "group", "hosts"):
        (jail / "etc" / name).chmod(0o444)
    for directory, dirs, _ in os.walk(jail, followlinks=False):
        path = Path(directory)
        if path == physical:
            dirs[:] = []
            continue
        path.chmod(0o755)
    ownership = {}
    try:
        root.rename(physical)
        root.symlink_to(physical, target_is_directory=True)
        (physical / "tmp" / "shm").mkdir(exist_ok=True)
        for directory, dirs, files in os.walk(physical, followlinks=False):
            for path in [Path(directory), *(Path(directory) / name for name in dirs + files)]:
                details = path.lstat()
                ownership[str(path.relative_to(physical))] = (details.st_uid, details.st_gid)
                os.lchown(path, uid, uid)
    except Exception:
        if physical.exists():
            for relative, owner in ownership.items():
                path = physical / relative
                if path.exists() or path.is_symlink():
                    os.lchown(path, *owner)
            if root.is_symlink():
                root.unlink()
            physical.rename(root)
        shutil.rmtree(jail, ignore_errors=True)
        raise
    info = {"logical_root": str(root), "physical_root": str(physical), "jail": str(jail), "uid": uid,
            "runtime_manifest_sha256": runtime["manifest_sha256"], "runtime_files": len(runtime["files"]),
            "runtime_cache": runtime["cache"], "python_executable": runtime["python_executable"],
            "dependency_profiles": runtime["dependency_profiles"], "successful_isolated_commands": 0,
            "quiescent": True, "launcher_sha256": sha(__file__)}
    _JAILS[str(physical)] = info
    _JAILS[str(root)] = info
    _BUILDING.pop(str(root), None)
    return info


def execute(root, arguments, executor):
    info = ensure(root)
    quiesce(info)
    # copy_to/write_file may introduce a new root-owned task file. Normalize
    # only task-tree ownership before the isolated command can access it.
    for directory, dirs, files in os.walk(info["physical_root"], followlinks=False):
        for path in [Path(directory), *(Path(directory) / name for name in dirs + files)]:
            os.lchown(path, info["uid"], info["uid"])
    receipt_fd, receipt_name = tempfile.mkstemp(prefix="vfarl-jail-receipt-")
    os.close(receipt_fd)
    receipt = Path(receipt_name)
    call = dict(arguments)
    env = dict(call["env"])
    env.pop("LD_PRELOAD", None)
    env.pop("LD_LIBRARY_PATH", None)
    env["TMPDIR"] = str(Path(info["logical_root"]) / "tmp")
    env["TMP"] = env["TMPDIR"]
    env["TEMP"] = env["TMPDIR"]
    command = list(call["command"])
    # A manager may pass a resolved physical path after a host-side file copy.
    # Replace only exact root prefixes followed by a path boundary.
    pattern = re.escape(info["physical_root"]) + r"(?=/|$)"
    command = [re.sub(pattern, info["logical_root"], value) for value in command]
    env = {key: re.sub(pattern, info["logical_root"], value) for key, value in env.items()}
    cwd = re.sub(pattern, info["logical_root"], str(call["cwd"]))
    call["command"] = [sys.executable, "-I", "-S", str(Path(__file__).resolve()), "--launch",
                       info["jail"], str(info["uid"]), cwd, str(receipt), *command]
    call["env"] = env
    info["quiescent"] = False
    try:
        result = executor(**call)
    finally:
        quiesce(info)
    try:
        observed = json.loads(receipt.read_text())
    finally:
        receipt.unlink(missing_ok=True)
    if observed != {"uid": info["uid"], "gid": info["uid"], "capabilities_zero": True,
                    "bounding_and_ambient_zero": True,
                    "no_new_privs": True, "seccomp_loaded": True, "host_fds_closed": True}:
        raise RuntimeError("The isolated launcher did not provide a valid privilege/filter receipt")
    info["successful_isolated_commands"] += 1
    return result, info


def identity(root):
    info = _JAILS.get(str(Path(root).resolve()))
    if not info:
        return None
    quiesce(info)
    return {key: info[key] for key in ("logical_root", "physical_root", "uid", "runtime_manifest_sha256",
            "runtime_files", "dependency_profiles", "launcher_sha256")}


def complete(root):
    info = _JAILS.get(str(Path(root).resolve()))
    return bool(_PROBE and info and info["quiescent"] and info["successful_isolated_commands"])


def cleanup(root):
    info = _JAILS.get(str(Path(root).resolve())) or _JAILS.get(str(root))
    if not info:
        pending = _BUILDING.pop(str(Path(root).absolute()), None)
        if pending:
            original = Path(root)
            if original.is_symlink() and original.resolve().is_relative_to(pending):
                physical = original.resolve()
                original.unlink()
                physical.rename(original)
            shutil.rmtree(pending, ignore_errors=True)
            return True
        return False
    quiesce(info)
    Path(info["logical_root"]).unlink(missing_ok=True)
    shutil.rmtree(info["jail"])
    _JAILS.pop(info["physical_root"], None)
    _JAILS.pop(info["logical_root"], None)
    return True


def configure(hidden_paths):
    """Run a real CPU probe before allowing complete-declared-scope claims."""
    global _HIDDEN, _PROBE
    _HIDDEN = [Path(path).resolve() for path in hidden_paths]
    _PROBE = False
    if sys.platform != "linux" or os.geteuid() != 0:
        return {"effective_mode": "observable", "probe_passed": False, "reason": "chroot requires Linux root"}
    from adk_eval_core.sandbox import execute_subprocess_command
    with tempfile.TemporaryDirectory(prefix="vfarl-jail-probe-") as temporary:
        parent = Path(temporary)
        root = parent / "task"
        for name in ("workspace", "tmp", "venv/bin"):
            (root / name).mkdir(parents=True)
        (root / "venv/bin/python").symlink_to(Path(sys.executable).resolve(strict=True))
        (parent / "gold-marker").write_text("must remain host-only")
        probe = ("import os,socket,time,pathlib; "
                 "assert os.getuid()!=0; "
                 f"assert not pathlib.Path({str(parent / 'gold-marker')!r}).exists(); "
                 "assert not pathlib.Path('/proc').exists(); "
                 "pathlib.Path('writable').write_text('ok'); "
                 "\ntry: open('/bin/bash','wb'); raise AssertionError('runtime writable')\nexcept PermissionError: pass\n"
                 "try: socket.socket(); raise AssertionError('network allowed')\nexcept PermissionError: pass\n"
                 "pid=os.fork()\nif pid==0:\n os.closerange(0,3); time.sleep(2); pathlib.Path('orphan-leak').write_text('bad'); os._exit(0)\n")
        try:
            result, info = execute(root, {"command": ["/usr/bin/python3", "-I", "-S", "-c", probe],
                "cwd": root / "workspace", "env": {"PATH": "/usr/bin:/bin", "HOME": str(root / "workspace")},
                "timeout": 40, "shell": False}, execute_subprocess_command)
            if result.exit_code != 0 or (Path(info["physical_root"]) / "workspace/orphan-leak").exists():
                raise RuntimeError("chroot/filter/runtime/UID-cleanup probe failed: " + result.stderr[-3000:])
            _PROBE = True
            return {"effective_mode": "chroot", "probe_passed": True, "reason": "Actual chroot, privilege-drop, seccomp, gold exclusion, readonly-runtime and UID cleanup probe passed",
                    "probe_uid": info["uid"], "runtime_files": info["runtime_files"], "launcher_sha256": sha(__file__)}
        except Exception as error:
            return {"effective_mode": "observable", "probe_passed": False, "reason": type(error).__name__ + ": " + str(error)}
        finally:
            cleanup(root)


def _launch(jail, uid, cwd, receipt, command):
    libc = ctypes.CDLL(None, use_errno=True)
    # Avoid find_library(): it can invoke host helper commands using a PATH
    # that starts with the task venv while this launcher is still privileged.
    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    context = seccomp.seccomp_init(0x7FFF0000)
    if not context:
        raise RuntimeError("seccomp_init failed")
    denied = "socket connect bind listen accept accept4 ptrace process_vm_readv process_vm_writev pidfd_getfd open_by_handle_at name_to_handle_at mount umount2 pivot_root chroot unshare setns bpf perf_event_open io_uring_setup io_uring_enter io_uring_register keyctl add_key request_key shmget shmat shmdt shmctl semget semop semctl semtimedop msgget msgsnd msgrcv msgctl setuid setreuid setresuid setfsuid setgid setregid setresgid setfsgid setgroups".split()
    for name in denied + ["clone3"]:
        number = seccomp.seccomp_syscall_resolve_name(name.encode())
        if number >= 0 and seccomp.seccomp_rule_add(context, 0x00050000 | (errno.ENOSYS if name == "clone3" else errno.EPERM), number, 0) != 0:
            raise RuntimeError("Could not add seccomp rule for " + name)
    # Prevent namespace creation while retaining ordinary fork/thread clone.
    class Comparison(ctypes.Structure):
        _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_uint), ("a", ctypes.c_uint64), ("b", ctypes.c_uint64)]
    seccomp.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(Comparison)]
    clone = seccomp.seccomp_syscall_resolve_name(b"clone")
    for flag in (0x00020000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000):
        condition = Comparison(0, 7, flag, flag)
        if seccomp.seccomp_rule_add_array(context, 0x00050000 | errno.EPERM, clone, 1, ctypes.byref(condition)) != 0:
            raise RuntimeError("Could not constrain clone namespace flags")
    fd = os.open(receipt, os.O_WRONLY | os.O_TRUNC | os.O_CLOEXEC)
    null = os.open("/dev/null", os.O_RDONLY)
    os.dup2(null, 0)
    if null != fd:
        os.close(null)
    for stream in (1, 2):
        if not stat.S_ISFIFO(os.fstat(stream).st_mode):
            raise RuntimeError("Launcher stdout/stderr must be dedicated pipes")
    limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    os.closerange(3, fd)
    os.closerange(fd + 1, min(limit, 1048576))
    parent_pid = os.getppid()
    os.chroot(jail)
    os.chdir("/")
    for capability in range(64):
        result = libc.prctl(24, capability, 0, 0, 0)  # PR_CAPBSET_DROP
        if result and ctypes.get_errno() != errno.EINVAL:
            raise OSError(ctypes.get_errno(), "Cannot drop capability bounding set")
    os.setgroups([])
    os.setresgid(uid, uid, uid)
    os.setresuid(uid, uid, uid)
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]
    class Caps(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]
    header, caps = Header(0x20080522, 0), (Caps * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(caps)) != 0 or libc.capget(ctypes.byref(header), ctypes.byref(caps)) != 0:
        raise RuntimeError("Cannot clear/read capabilities")
    if any(value.effective or value.permitted or value.inheritable for value in caps):
        raise RuntimeError("Capabilities remain after privilege drop")
    if libc.prctl(47, 4, 0, 0, 0) != 0:  # PR_CAP_AMBIENT_CLEAR_ALL
        raise RuntimeError("Cannot clear ambient capabilities")
    for capability in range(64):
        for operation in ((23, capability), (47, 1, capability)):
            ctypes.set_errno(0)
            result = libc.prctl(*operation, *([0] * (5 - len(operation))))
            if result > 0 or (result < 0 and ctypes.get_errno() != errno.EINVAL):
                raise RuntimeError("Bounding or ambient capabilities remain")
    if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(39, 0, 0, 0, 0) != 1:
        raise RuntimeError("no_new_privs could not be established")
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent_pid:
        raise RuntimeError("Launcher parent disappeared")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
    if seccomp.seccomp_load(context) != 0:
        raise RuntimeError("The host rejected the additional seccomp filter")
    seccomp.seccomp_release(context)
    os.chdir(cwd)
    receipt_data = {"uid": os.getuid(), "gid": os.getgid(), "capabilities_zero": True,
                    "bounding_and_ambient_zero": True,
                    "no_new_privs": True, "seccomp_loaded": True, "host_fds_closed": True}
    os.write(fd, json.dumps(receipt_data).encode())
    os.close(fd)
    os.execve(command[0], command, dict(os.environ))


if __name__ == "__main__":
    if len(sys.argv) < 7 or sys.argv[1] != "--launch":
        raise SystemExit("This module is invoked by the VFARL task executor")
    _launch(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6:])
