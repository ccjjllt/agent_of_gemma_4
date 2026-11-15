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
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


_SKIP_PARTS = {".git", ".hg", ".svn", "docker", "node_modules", "build", "dist", "vendor"}
_REQUIREMENT_GLOBS = ("requirements*.txt", "test-requirements*.txt")
_CONSTRAINT_GLOBS = ("constraints*.txt", "constraint*.txt")
_SKIP_FILE_TOKENS = ("docs", "doc-", "translations", "github-actions", "lint", "typing")
_SKIP_GROUP_TOKENS = ("docs", "doc", "translation", "lint", "typing")
_COMPATIBLE_TAGS = set(sys_tags())
_PYTHON_VERSION = Version(".".join(str(part) for part in sys.version_info[:3]))
_REQUIRES_PYTHON_CACHE: dict[str, SpecifierSet | None] = {}
_WHEEL_REQUIREMENTS_CACHE: dict[str, list[Requirement] | None] = {}


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


def _logical_lines(text: str) -> list[str]:
    """Join backslash continuations before parsing requirement directives."""
    lines: list[str] = []
    pending = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if line.endswith("\\"):
            pending += line[:-1].strip() + " "
            continue
        line = (pending + line).strip()
        pending = ""
        if line:
            lines.append(line)
    if pending.strip():
        lines.append(pending.strip())
    return lines


def _read_requirement_file(
    path: Path,
    workspace: Path,
    *,
    kind: str,
    requirements: list[str],
    constraints: list[str],
    seen: set[Path],
    diagnostics: dict[str, Any],
) -> None:
    """Read one requirements/constraints file and its -r/-c includes.

    Directives other than requirement/constraint includes are recorded, not
    silently dropped: an ignored ``-c`` file changes the selected versions.
    Includes outside the workspace are rejected so repository metadata cannot
    make the offline installer read arbitrary host files.
    """
    try:
        resolved = path.resolve()
        resolved.relative_to(workspace.resolve())
    except (OSError, ValueError):
        diagnostics["outside_workspace"].append(str(path))
        return
    if resolved in seen:
        diagnostics["include_cycles"].append(str(resolved))
        return
    seen.add(resolved)
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        diagnostics["unreadable_files"].append({"path": str(resolved), "error": type(error).__name__})
        return
    destination = requirements if kind == "requirements" else constraints
    for line in _logical_lines(text):
        if not line:
            continue
        if line.startswith("-"):
            match = re.match(r"(?P<flag>--?[A-Za-z][A-Za-z0-9-]*)(?:[=\s]+(?P<value>.*))?$", line)
            flag = match.group("flag") if match else line.split(None, 1)[0]
            value = match.group("value").strip() if match and match.group("value") else ""
            if flag in ("-r", "--requirement"):
                if value:
                    _read_requirement_file(
                        resolved.parent / value, workspace, kind="requirements",
                        requirements=requirements, constraints=constraints,
                        seen=seen, diagnostics=diagnostics,
                    )
                else:
                    diagnostics["unsupported_options"]["-r-missing-value"] += 1
            elif flag in ("-c", "--constraint"):
                if value:
                    _read_requirement_file(
                        resolved.parent / value, workspace, kind="constraints",
                        requirements=requirements, constraints=constraints,
                        seen=seen, diagnostics=diagnostics,
                    )
                else:
                    diagnostics["unsupported_options"]["-c-missing-value"] += 1
            else:
                diagnostics["unsupported_options"][flag] += 1
            continue
        destination.append(line)


def _requirements_from_files(workspace: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    requirements: list[str] = []
    constraints: list[str] = []
    diagnostics: dict[str, Any] = {
        "unsupported_options": Counter(),
        "outside_workspace": [],
        "include_cycles": [],
        "unreadable_files": [],
        "requirement_files": [],
        "constraint_files": [],
    }
    seen: set[Path] = set()
    for patterns, kind, sink in (
        (_REQUIREMENT_GLOBS, "requirements", diagnostics["requirement_files"]),
        (_CONSTRAINT_GLOBS, "constraints", diagnostics["constraint_files"]),
    ):
        for pattern in patterns:
            for path in sorted(workspace.glob(f"**/{pattern}")):
                if any(part in _SKIP_PARTS for part in path.parts):
                    continue
                lower_name = path.name.lower()
                if any(token in lower_name for token in _SKIP_FILE_TOKENS):
                    continue
                sink.append(str(path))
                _read_requirement_file(
                    path, workspace, kind=kind, requirements=requirements,
                    constraints=constraints, seen=seen, diagnostics=diagnostics,
                )
    diagnostics["unsupported_options"] = dict(sorted(diagnostics["unsupported_options"].items()))
    return requirements, constraints, diagnostics


def collect_requirements_and_constraints(
    workspace: Path,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Return repository requirements, constraint-only entries and diagnostics."""
    file_requirements, constraints, diagnostics = _requirements_from_files(workspace)
    requirements = list(dict.fromkeys(
        _requirements_from_pyproject(workspace) + file_requirements
    ))
    diagnostics["constraints"] = list(dict.fromkeys(constraints))
    return requirements, list(dict.fromkeys(constraints)), diagnostics


def collect_requirements(workspace: Path) -> list[str]:
    """Return raw PEP 508 strings from the repository's public metadata."""
    return collect_requirements_and_constraints(workspace)[0]


def _requires_python(path: Path) -> SpecifierSet | None:
    """Read only ``Requires-Python`` from a wheel's dist-info metadata."""
    key = str(path)
    if key in _REQUIRES_PYTHON_CACHE:
        return _REQUIRES_PYTHON_CACHE[key]
    result: SpecifierSet | None = None
    try:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist()
                     if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
            if len(names) == 1:
                metadata = archive.read(names[0]).decode("utf-8", errors="replace")
                result = SpecifierSet()
                for line in metadata.splitlines():
                    if not line:
                        break
                    if line.lower().startswith("requires-python:"):
                        value = line.split(":", 1)[1].strip()
                        result = SpecifierSet(value)
                        break
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, InvalidSpecifier):
        result = None
    _REQUIRES_PYTHON_CACHE[key] = result
    return result


def _wheel_requirements(path: Path) -> list[Requirement] | None:
    """Read ``Requires-Dist`` entries from a wheel, cached by path."""
    key = str(path)
    cached = _WHEEL_REQUIREMENTS_CACHE.get(key)
    if cached is not None or key in _WHEEL_REQUIREMENTS_CACHE:
        return cached
    result: list[Requirement] | None = None
    try:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist()
                     if name.count("/") == 1 and name.endswith(".dist-info/METADATA")]
            if len(names) == 1:
                metadata = archive.read(names[0]).decode("utf-8", errors="replace")
                result = []
                for line in metadata.splitlines():
                    if not line:
                        break
                    if line.lower().startswith("requires-dist:"):
                        value = line.split(":", 1)[1].strip()
                        try:
                            result.append(Requirement(value))
                        except InvalidRequirement:
                            result = None
                            break
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        result = None
    _WHEEL_REQUIREMENTS_CACHE[key] = result
    return result


def _installed_version(name: str) -> Version | None:
    import importlib.metadata
    try:
        return Version(importlib.metadata.version(name))
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return None


def _supported_extras(
    path: Path, extras: Iterable[str], index: Mapping[str, list[tuple[Any, Path]]],
) -> tuple[list[str], dict[str, list[str]]]:
    """Keep extras whose extra-only dependencies exist offline or are installed.

    The official wheel cache is not a complete PyPI mirror.  Blindly preserving
    ``pkg[extra]`` turns an otherwise usable pin into a pip failure when one
    optional dependency is absent (for example FastAPI's ``standard`` extra
    pulling ``fastapi-cloud-cli``).  Unsupported extras are reported and
    dropped; supported extras stay in the exact pin.
    """
    requirements = _wheel_requirements(path)
    if requirements is None:
        return [], {extra: ["wheel_metadata_unreadable"] for extra in extras}
    supported: list[str] = []
    dropped: dict[str, list[str]] = {}
    for extra in extras:
        missing: list[str] = []
        for requirement in requirements:
            marker = requirement.marker
            if marker is None or "extra" not in str(marker):
                continue
            try:
                if not marker.evaluate({"extra": extra}):
                    continue
            except Exception:
                continue
            dependency = canonicalize_name(requirement.name)
            versions = [version for version, _path in index.get(dependency, [])]
            available = any(requirement.specifier.contains(
                version, prereleases=True) for version in versions)
            if not available:
                current = _installed_version(dependency)
                available = current is not None and requirement.specifier.contains(
                    current, prereleases=True)
            if not available:
                missing.append(str(requirement))
        if missing:
            dropped[extra] = missing
        else:
            supported.append(extra)
    return supported, dropped


def _wheel_index(
    wheel_dirs: Iterable[Path], python_skipped: Counter[str],
) -> dict[str, list[tuple[Any, Path]]]:
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
            requires_python = _requires_python(path)
            if requires_python is None:
                python_skipped["unreadable_metadata"] += 1
                continue
            if not requires_python.contains(_PYTHON_VERSION, prereleases=True):
                python_skipped[str(requires_python) or "<any>"] += 1
                continue
            index.setdefault(canonicalize_name(name), []).append((version, path))
    return index


def select_requirements(
    raw_requirements: Iterable[str], wheel_dirs: Iterable[Path],
    raw_constraints: Iterable[str] = (),
) -> tuple[list[str], dict[str, Any]]:
    """Intersect repository specifiers and choose the highest satisfying wheel.

    Requirements whose distribution is absent from the offline wheelhouse are
    reported and skipped, matching the official setup's local-wheel scope.
    An unsatisfiable intersection is also reported instead of silently
    selecting an incompatible highest version.  ``-c`` constraints narrow
    versions but never add a distribution that no requirement requested.
    """
    python_skipped: Counter[str] = Counter()
    index = _wheel_index(wheel_dirs, python_skipped)
    constraints: dict[str, SpecifierSet] = {}
    extras_by_name: dict[str, set[str]] = {}
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
        extras_by_name.setdefault(name, set()).update(requirement.extras)

    constraint_specs: dict[str, SpecifierSet] = {}
    invalid_constraints: list[str] = []
    constraint_marker_skipped: Counter[str] = Counter()
    for raw in raw_constraints:
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            invalid_constraints.append(raw[:200])
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            constraint_marker_skipped[requirement.name] += 1
            continue
        name = canonicalize_name(requirement.name)
        constraint_specs[name] = constraint_specs.get(name, SpecifierSet()) & requirement.specifier
    applied_constraints: dict[str, str] = {}
    for name, specifier in constraint_specs.items():
        if name in constraints:
            constraints[name] = constraints[name] & specifier
            applied_constraints[name] = str(specifier)

    selected: list[str] = []
    versions: dict[str, str] = {}
    selected_extras: dict[str, list[str]] = {}
    dropped_extras: dict[str, dict[str, list[str]]] = {}
    unsatisfied: dict[str, str] = {}
    for name, specifier in sorted(constraints.items()):
        candidates = [version for version, _path in index[name]
                      if specifier.contains(version, prereleases=True)]
        if not candidates:
            unsatisfied[name] = str(specifier) or "<any>"
            continue
        version = max(candidates)
        selected_path = next(path for candidate, path in index[name] if candidate == version)
        versions[name] = str(version)
        extras = sorted(extras_by_name.get(name, ()))
        supported, dropped = _supported_extras(selected_path, extras, index)
        if supported:
            selected_extras[name] = supported
        if dropped:
            dropped_extras[name] = dropped
        suffix = f"[{','.join(supported)}]" if supported else ""
        selected.append(f"{name}{suffix}=={version}")
    diagnostics = {
        "schema": "gemma.task_dependency_selection.v1",
        "wheel_dirs": [str(path) for path in wheel_dirs],
        "selected_versions": versions,
        "selected_count": len(selected),
        "selected_extras": selected_extras,
        "dropped_extras": dropped_extras,
        "unavailable_distributions": dict(sorted(unavailable.items())),
        "unsatisfied_constraints": dict(sorted(unsatisfied.items())),
        "invalid_requirements": invalid,
        "marker_skipped": dict(sorted(marker_skipped.items())),
        "applied_constraints": applied_constraints,
        "invalid_constraints": invalid_constraints,
        "constraint_marker_skipped": dict(sorted(constraint_marker_skipped.items())),
        "requires_python_skipped": dict(sorted(python_skipped.items())),
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
    raw, raw_constraints, source_diagnostics = collect_requirements_and_constraints(workspace)
    selected, diagnostics = select_requirements(raw, wheel_dirs, raw_constraints)
    diagnostics.update(source_diagnostics)
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
