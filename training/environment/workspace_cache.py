"""Private per-attempt workspaces with a read-only official data boundary."""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _absolute(path: str | os.PathLike[str], name: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return candidate


@dataclass(frozen=True)
class AttemptPaths:
    attempt_id: str
    attempt_dir: Path
    workspace_dir: Path
    result_path: Path
    log_path: Path
    credential_path: Path


class WorkspaceCache:
    def __init__(self, root: str | os.PathLike[str], *, data_root: str | os.PathLike[str] | None = None):
        self.root = _absolute(root, "workspace cache root")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self.data_root = _absolute(data_root, "data root") if data_root is not None else None
        if self.data_root is not None and not (self.data_root / "tasks.jsonl").is_file():
            raise ValueError("data root must contain the official tasks.jsonl inventory")

    @staticmethod
    def check_id(value: str, name: str) -> str:
        if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
            raise ValueError(f"{name} must be a safe public identifier")
        return value

    def prepare_attempt(self, attempt_id: str, *, task_id: str) -> AttemptPaths:
        attempt = self.check_id(attempt_id, "attempt_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        attempt_dir = self.root / attempt
        attempt_dir.mkdir(mode=0o700, exist_ok=False)
        workspace = attempt_dir / "workspace"
        workspace.mkdir(mode=0o700)
        return AttemptPaths(
            attempt_id=attempt,
            attempt_dir=attempt_dir,
            workspace_dir=workspace,
            result_path=attempt_dir / "result.json",
            log_path=attempt_dir / "worker.log",
            credential_path=attempt_dir / "model_auth_token",
        )

    def discard_attempt(self, attempt_id: str) -> None:
        attempt = self.check_id(attempt_id, "attempt_id")
        path = (self.root / attempt).resolve()
        if path.parent != self.root.resolve() or not path.is_dir() or path.is_symlink():
            raise ValueError("refusing to discard a foreign attempt directory")
        shutil.rmtree(path)

    def write_json(self, path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
        target = Path(path)
        if target.resolve().parent != self.root.resolve() and self.root.resolve() not in target.resolve().parents:
            raise ValueError("workspace cache writes must remain beneath its root")
        text = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, allow_nan=False)
        temporary = target.with_name(f".commit-{target.name}")
        with temporary.open("w", encoding="utf-8") as output:
            output.write(text + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
