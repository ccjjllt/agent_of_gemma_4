"""Subprocess sandbox boundaries and deterministic UID allocation."""
from __future__ import annotations

import os
import pwd
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SandboxConfig:
    sandbox_root: Path
    baseline_python: Path
    agent_uid_base: int
    max_attempts: int = 1024

    def __post_init__(self) -> None:
        if not isinstance(self.sandbox_root, Path) or not self.sandbox_root.is_absolute():
            raise ValueError("sandbox_root must be an absolute path")
        if not isinstance(self.baseline_python, Path) or not self.baseline_python.is_absolute():
            raise ValueError("baseline_python must be an absolute path")
        if type(self.agent_uid_base) is not int or type(self.max_attempts) is not int:
            raise ValueError("UID base and attempt budget must be integers")
        if not 10000 <= self.agent_uid_base <= 2**31 - 2 * self.max_attempts:
            raise ValueError("agent UID range must fit in [10000, 2**31) without collision")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")


class UidAllocator:
    def __init__(self, base_uid: int, max_attempts: int = 1024):
        config = SandboxConfig(Path("/tmp"), Path(sys.executable), base_uid, max_attempts)
        self._base = config.agent_uid_base
        self._limit = config.max_attempts

    def allocate(self, attempt_index: int) -> tuple[int, int]:
        if type(attempt_index) is not int or not 0 <= attempt_index < self._limit:
            raise ValueError("attempt index is outside the allocated UID range")
        agent = self._base + 2 * attempt_index
        return agent, agent + 1


class SandboxGuard:
    @staticmethod
    def _unused_uid(uid: int, role: str) -> None:
        try:
            pwd.getpwuid(uid)
        except KeyError:
            pass
        else:
            raise ValueError(f"The {role} UID is assigned to a host account")
        for path in Path("/proc").glob("[0-9]*/status"):
            try:
                row = next(line for line in path.read_text().splitlines() if line.startswith("Uid:"))
                if int(row.split()[1]) == uid:
                    raise ValueError(f"The {role} UID has running processes")
            except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
                continue

    @staticmethod
    def _probe_access(uid: int, paths: list[Path]) -> None:
        binary = shutil.which("setpriv")
        if not binary:
            raise RuntimeError("UID isolation requires Linux util-linux setpriv")
        command = [
            binary, f"--reuid={uid}", f"--regid={uid}", "--clear-groups",
            "--no-new-privs", "--bounding-set=-all", "--inh-caps=-all",
            "--ambient-caps=-all", sys.executable, "-I", "-c",
            "import os,sys;sys.exit(0 if all(os.access(p, os.X_OK) for p in sys.argv[1:]) else 1)",
            *[str(path) for path in paths],
        ]
        completed = subprocess.run(command, check=False, capture_output=True, timeout=15)
        if completed.returncode:
            raise PermissionError("UID isolation probe failed for sandbox dependencies")

    @classmethod
    def check_execution(
        cls,
        config: SandboxConfig,
        *,
        data_root: Path,
        workspace: Path,
        agent_uid: int,
        judge_uid: int,
    ) -> None:
        if sys.platform != "linux" or os.geteuid() != 0:
            raise RuntimeError("UID-isolated subprocess execution requires a Linux root judge process")
        if agent_uid == judge_uid:
            raise ValueError("Agent and judge must use different unprivileged UIDs")
        if not data_root.is_absolute() or not workspace.is_absolute():
            raise ValueError("data and workspace roots must be absolute")
        if not (data_root / "tasks.jsonl").is_file():
            raise ValueError("data root must contain the official tasks.jsonl inventory")
        config.sandbox_root.mkdir(parents=True, exist_ok=True, mode=0o711)
        info = config.sandbox_root.stat()
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise PermissionError("Agent sandbox parent must be root-owned and not group/world-writable")
        for role, uid in (("agent", agent_uid), ("judge", judge_uid)):
            cls._unused_uid(uid, role)
        cls._probe_access(agent_uid, [config.baseline_python.parent])
        cls._probe_access(judge_uid, [config.baseline_python.parent])
