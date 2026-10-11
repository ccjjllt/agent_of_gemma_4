"""Resolve the official Docker or subprocess task workspace, not the skill cwd.

The official subprocess runner supplies PYTHONPATH=<task>/workspace and
VIRTUAL_ENV=<task>/venv; its skill executor changes cwd before running scripts.
Only that matching pair is accepted automatically. No host task files are read.
"""
import os
import tempfile
from pathlib import Path


def runtime_workspace():
    pythonpath = os.environ.get("PYTHONPATH", "")
    venv = os.environ.get("VIRTUAL_ENV", "")
    if pythonpath and os.pathsep not in pythonpath and venv:
        candidate = Path(pythonpath)
        environment = Path(venv)
        if (candidate.name == "workspace" and environment.name == "venv"
                and candidate.parent.resolve() == environment.parent.resolve()
                and candidate.is_dir() and environment.is_dir()
                and not candidate.is_symlink()):
            return candidate.resolve()
    workspace = Path("/workspace")
    if workspace.is_dir() and not workspace.is_symlink():
        return workspace.resolve()
    return None


def resolve_task_root(root_arg="/workspace", allow_non_ws=False):
    runtime = runtime_workspace()
    requested = Path(root_arg)
    if requested.is_symlink():
        raise ValueError("symlink root rejected")
    if str(requested) == "/workspace" and runtime is not None:
        root = runtime
    else:
        root = requested.resolve()
    if not root.is_dir():
        raise ValueError("task workspace missing")
    if root != runtime and not allow_non_ws:
        raise ValueError("root is not the official task workspace; local diagnosis "
                         "requires --allow-non-workspace-root")
    return root


def sandbox_temp(root):
    runtime = runtime_workspace()
    if runtime == root and root != Path("/workspace"):
        return root.parent / "tmp"
    return Path(tempfile.gettempdir()).resolve()


def resolve_issue_input(root, spec):
    """Read model-provided issue text only from the current repo or task temp."""
    candidate = Path(spec)
    tmp = sandbox_temp(root)
    if candidate.is_absolute() and (str(candidate) == "/tmp" or str(candidate).startswith("/tmp/")):
        candidate = tmp / candidate.relative_to("/tmp")
    elif not candidate.is_absolute():
        if ".." in candidate.parts:
            raise ValueError("issue path must not contain ..")
        candidate = root / candidate
    if candidate.is_symlink():
        raise ValueError("symlink issue file rejected")
    resolved = candidate.resolve()
    if not any(resolved == base or base in resolved.parents for base in (root, tmp.resolve())):
        raise ValueError("issue file must be in this task workspace or temporary directory")
    if not resolved.is_file():
        raise ValueError("issue file missing")
    return resolved
