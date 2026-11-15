"""Version-aware wheel installation for the official task sandbox.

This module runs inside the isolated task virtual environment.  It reuses the
official ``sandbox/setup.py`` helpers for workspace preparation, but selects
offline wheels by honouring the repository's version specifiers instead of
always choosing the highest version for every distribution.  The latter turns
old FastAPI commits into ``Router.__init__() got an unexpected keyword
argument 'on_startup'`` collection failures because the newest Starlette is
installed.

Only public repository metadata and the local wheel directory are read.  No
network access, task patch, reference patch or held-out content is used.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


_SKIP_PARTS = {".git", ".hg", ".svn", "docker", "node_modules", "build", "dist", "vendor"}
_REQUIREMENT_GLOBS = ("requirements*.txt", "test-requirements*.txt")
_SKIP_FILE_TOKENS = ("docs", "doc-", "translations", "github-actions", "lint", "typing")
_SKIP_GROUP_TOKENS = ("docs", "doc", "translation", "lint", "typing")
_COMPATIBLE_TAGS = set(sys_tags())


def _clean_line(line: str) -> str:
    line = line.split("#", 1)[0].strip()
    if line.endswith("\\"):
        line = line[:-1].strip()
    return line


def _requirements_from_pyproject(workspace: Path) -> list[str]:
    path = workspace / "pyproject.toml"
    if not path.is_file():
        return []
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    requirements: list[str] = []
    project = document.get("project")
    if isinstance(project, Mapping):
        dependencies = project.get("dependencies")
        if isinstance(dependencies, list):
            requirements.extend(item for item in dependencies if isinstance(item, str))
        optional = project.get("optional-dependencies")
        if isinstance(optional, Mapping):
            for group_name, group in optional.items():
                if any(token in str(group_name).lower() for token in _SKIP_GROUP_TOKENS):
                    continue
                if isinstance(group, list):
                    requirements.extend(item for item in group if isinstance(item, str))
    groups = document.get("dependency-groups")
    if isinstance(groups, Mapping):
        for group_name, group in groups.items():
            if any(token in str(group_name).lower() for token in _SKIP_GROUP_TOKENS):
                continue
            if isinstance(group, list):
                requirements.extend(item for item in group if isinstance(item, str))
    return requirements


def _requirements_from_files(workspace: Path) -> list[str]:
    requirements: list[str] = []
    for pattern in _REQUIREMENT_GLOBS:
        for path in sorted(workspace.glob(f"**/{pattern}")):
            if any(part in _SKIP_PARTS for part in path.parts):
                continue
            lower_name = path.name.lower()
            if any(token in lower_name for token in _SKIP_FILE_TOKENS):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for line in lines:
                line = _clean_line(line)
                if not line or line.startswith("-"):
                    continue
                requirements.append(line)
    return requirements


def collect_requirements(workspace: Path) -> list[str]:
    """Return raw PEP 508 strings from the repository's public metadata."""
    return list(dict.fromkeys(
        _requirements_from_pyproject(workspace) + _requirements_from_files(workspace)
    ))


def _wheel_index(wheel_dirs: Iterable[Path]) -> dict[str, list[tuple[Any, Path]]]:
    index: dict[str, list[tuple[Any, Path]]] = {}
    for directory in wheel_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.whl")):
            try:
                name, version, _build, tags = parse_wheel_filename(path.name)
            except (ValueError, TypeError):
                continue
            if not (tags & _COMPATIBLE_TAGS):
                continue
            index.setdefault(canonicalize_name(name), []).append((version, path))
    return index


def select_requirements(
    raw_requirements: Iterable[str], wheel_dirs: Iterable[Path],
) -> tuple[list[str], dict[str, Any]]:
    """Intersect repository specifiers and choose the highest satisfying wheel.

    Requirements whose distribution is absent from the offline wheelhouse are
    reported and skipped, matching the official setup's local-wheel scope.
    An unsatisfiable intersection is also reported instead of silently
    selecting an incompatible highest version.
    """
    index = _wheel_index(wheel_dirs)
    constraints: dict[str, SpecifierSet] = {}
    unavailable: Counter[str] = Counter()
    invalid: list[str] = []
    marker_skipped: Counter[str] = Counter()
    for raw in raw_requirements:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            invalid.append(raw[:200])
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            marker_skipped[requirement.name] += 1
            continue
        name = canonicalize_name(requirement.name)
        if name not in index:
            unavailable[name] += 1
            continue
        current = constraints.get(name, SpecifierSet())
        constraints[name] = current & requirement.specifier

    selected: list[str] = []
    versions: dict[str, str] = {}
    unsatisfied: dict[str, str] = {}
    for name, specifier in sorted(constraints.items()):
        candidates = [version for version, _path in index[name]
                      if specifier.contains(version, prereleases=True)]
        if not candidates:
            unsatisfied[name] = str(specifier) or "<any>"
            continue
        version = max(candidates)
        versions[name] = str(version)
        selected.append(f"{name}=={version}")
    diagnostics = {
        "schema": "gemma.task_dependency_selection.v1",
        "wheel_dirs": [str(path) for path in wheel_dirs],
        "selected_versions": versions,
        "selected_count": len(selected),
        "unavailable_distributions": dict(sorted(unavailable.items())),
        "unsatisfied_constraints": dict(sorted(unsatisfied.items())),
        "invalid_requirements": invalid,
        "marker_skipped": dict(sorted(marker_skipped.items())),
    }
    return selected, diagnostics


def install(
    module: Any,
    workspace: Path,
    wheel_dirs: Iterable[Path],
    repo: str = "",
) -> dict[str, Any]:
    """Prepare the workspace and install the version-correct offline wheels.

    ``module`` is the loaded official ``sandbox/setup.py``.  Its
    ``subprocess.run`` is already wrapped by the caller so pip failures are
    surfaced with their original output.
    """
    workspace = Path(workspace)
    module.setup_git_exclude(workspace)
    module.install_editable_package(workspace)
    raw = collect_requirements(workspace)
    selected, diagnostics = select_requirements(raw, wheel_dirs)
    if selected:
        command = [sys.executable, "-m", "pip", "install", "--no-build-isolation", *selected]
        result = module.subprocess.run(command, capture_output=True, text=True, check=False)
        if getattr(result, "returncode", 1):
            raise RuntimeError(
                "Version-aware task dependency installation failed: "
                + json.dumps({
                    "returncode": result.returncode,
                    "stdout": (result.stdout or "")[-4000:],
                    "stderr": (result.stderr or "")[-4000:],
                }, ensure_ascii=False)
            )
    import importlib.metadata
    installed: dict[str, str | None] = {}
    for name in diagnostics["selected_versions"]:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    diagnostics["installed_versions"] = installed
    diagnostics["version_mismatches"] = {
        name: {"selected": version, "installed": installed.get(name)}
        for name, version in diagnostics["selected_versions"].items()
        if installed.get(name) != version
    }
    module.execute_fast_path(workspace, repo=repo)
    diagnostics["repo"] = repo
    return diagnostics
