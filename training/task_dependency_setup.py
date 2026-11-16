"""Version-aware wheel installation for the official task sandbox.

This module runs inside the isolated task virtual environment.  It reuses the
official ``sandbox/setup.py`` helpers for workspace preparation, but selects
offline wheels by honouring the repository's version specifiers instead of
always choosing the highest version for every distribution.  The latter turns
old FastAPI commits into ``Router.__init__() got an unexpected keyword
argument 'on_startup'`` collection failures because the newest Starlette is
installed.  A joint pass also checks the chosen wheels' declared
``Requires-Dist`` entries so mutually incompatible selections are not passed
to pip.

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
_CONSTRAINT_GLOBS = ("constraint*.txt",)
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


def _requirement_applies(requirement: Requirement, extra: str | None = None) -> bool:
    """Evaluate a ``Requires-Dist`` marker, with an empty extra by default."""
    marker = requirement.marker
    if marker is None:
        return True
    try:
        if extra is None:
            return bool(marker.evaluate())
        return bool(marker.evaluate({"extra": extra}))
    except Exception:
        return False


def _dependency_edges(
    path: Path, extras: Iterable[str],
) -> list[tuple[Requirement, str | None]] | None:
    """Return applicable ``Requires-Dist`` entries and their enabling extra.

    ``None`` means the wheel metadata could not be read; callers must not treat
    that wheel as verified.  A requirement that applies without an extra is
    recorded once with ``None`` because its specifier always applies.
    """
    requirements = _wheel_requirements(path)
    if requirements is None:
        return None
    edges: list[tuple[Requirement, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    selected_extras = sorted(extras)
    for requirement in requirements:
        if _requirement_applies(requirement):
            edges.append((requirement, None))
            continue
        for extra in selected_extras:
            if not _requirement_applies(requirement, extra):
                continue
            key = (str(requirement), extra)
            if key in seen:
                continue
            seen.add(key)
            edges.append((requirement, extra))
    return edges


class _JointResolution:
    """Resolve cross-distribution ``Requires-Dist`` conflicts on chosen wheels.

    Per-distribution selection cannot see that ``fastapi-cli`` needs a newer
    Typer than the repository pins.  This pass keeps the repository specifiers
    authoritative and changes a second distribution only inside them.  Every
    action is recorded in ``joint_resolution``; an incompatible pair is never
    kept silently.
    """

    def __init__(
        self,
        index: Mapping[str, list[tuple[Any, Path]]],
        constraints: Mapping[str, SpecifierSet],
        versions: dict[str, Version],
        paths: dict[str, Path],
        extras: dict[str, list[str]],
        dropped_extras: dict[str, dict[str, list[str]]],
    ) -> None:
        self.index = index
        self.constraints = constraints
        self.versions = versions
        self.paths = paths
        self.extras = extras
        self.dropped_extras = dropped_extras
        self.conflicts: list[dict[str, Any]] = []
        self.changes: list[dict[str, Any]] = []
        self.missing: dict[str, str] = {}
        self.unreadable: list[str] = []
        self.residual: list[dict[str, Any]] = []
        self._edge_cache: dict[
            tuple[str, str, tuple[str, ...]],
            list[tuple[Requirement, str | None]],
        ] = {}

    def _edges_for(self, name: str) -> list[tuple[Requirement, str | None]]:
        version = str(self.versions[name])
        extra_key = tuple(sorted(self.extras.get(name, ())))
        key = (name, version, extra_key)
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached
        edges = _dependency_edges(self.paths[name], extra_key)
        if edges is None:
            self.unreadable.append(f"{name}=={version}")
            edges = []
        self._edge_cache[key] = edges
        return edges

    def _all_edges(self) -> list[tuple[str, Requirement, str | None]]:
        edges: list[tuple[str, Requirement, str | None]] = []
        for name in sorted(self.versions):
            for requirement, via in self._edges_for(name):
                edges.append((name, requirement, via))
        edges.sort(key=lambda item: (item[0], str(item[1]), item[2] or ""))
        return edges

    def _conflicts(self) -> list[tuple[str, Requirement, str | None, str]]:
        found: list[tuple[str, Requirement, str | None, str]] = []
        for source, requirement, via in self._all_edges():
            target = canonicalize_name(requirement.name)
            if target not in self.versions:
                continue
            if requirement.specifier.contains(
                self.versions[target], prereleases=True
            ):
                continue
            found.append((source, requirement, via, target))
        return found

    def _candidates(self, name: str) -> list[Version]:
        """Versions that satisfy the repository pin and every selected dependent."""
        specifier = self.constraints.get(name, SpecifierSet())
        incoming = [
            requirement
            for _source, requirement, _via in self._all_edges()
            if canonicalize_name(requirement.name) == name
        ]
        candidates: list[Version] = []
        for version, _path in self.index.get(name, []):
            if not specifier.contains(version, prereleases=True):
                continue
            if all(
                requirement.specifier.contains(version, prereleases=True)
                for requirement in incoming
            ):
                candidates.append(version)
        return sorted(set(candidates))

    def _requirement_available(self, requirement: Requirement) -> bool:
        target = canonicalize_name(requirement.name)
        if any(
            requirement.specifier.contains(version, prereleases=True)
            for version, _path in self.index.get(target, [])
        ):
            return True
        installed = _installed_version(target)
        return installed is not None and requirement.specifier.contains(
            installed, prereleases=True
        )

    def _requires_dist_satisfiable(
        self, name: str, version: Version, path: Path, extras: Iterable[str],
    ) -> bool | None:
        edges = _dependency_edges(path, extras)
        if edges is None:
            return None
        for requirement, _via in edges:
            target = canonicalize_name(requirement.name)
            if target == name:
                satisfied = requirement.specifier.contains(version, prereleases=True)
            elif target in self.versions:
                satisfied = requirement.specifier.contains(
                    self.versions[target], prereleases=True
                )
            else:
                satisfied = self._requirement_available(requirement)
            if not satisfied:
                return False
        return True

    def _path_for(self, name: str, version: Version) -> Path:
        return next(path for candidate, path in self.index[name]
                    if candidate == version)

    def _record_change(self, name: str, previous: Version, chosen: Version) -> None:
        self.changes.append({
            "distribution": name,
            "previous": str(previous),
            "chosen": str(chosen),
        })

    def _handle(
        self, source: str, requirement: Requirement, via: str | None, target: str,
    ) -> None:
        source_version = self.versions[source]
        target_version = self.versions[target]
        repository = self.constraints.get(target)
        target_candidates = self._candidates(target)
        satisfiable_targets: list[Version] = []
        for version in target_candidates:
            if self._requires_dist_satisfiable(
                target, version, self._path_for(target, version),
                self.extras.get(target, ()),
            ):
                satisfiable_targets.append(version)
        entry: dict[str, Any] = {
            "dependent": source,
            "dependent_version": str(source_version),
            "requirement": str(requirement),
            "via_extra": via,
            "selected": f"{target}=={target_version}",
            "repository_specifier": str(repository) if repository else "<any>",
            "available_versions": [
                str(version) for version, _path in self.index.get(target, [])
            ],
            "candidates_considered": [str(version) for version in target_candidates],
            "satisfiable_candidates": [str(version) for version in satisfiable_targets],
        }
        if satisfiable_targets and satisfiable_targets[-1] != target_version:
            chosen = satisfiable_targets[-1]
            self.versions[target] = chosen
            self.paths[target] = self._path_for(target, chosen)
            entry.update({
                "action": "changed_version",
                "chosen_version": str(chosen),
                "previous_version": str(target_version),
                "reason": (
                    f"{target} {target_version} does not satisfy {requirement} "
                    f"from {source} {source_version}"
                ),
            })
            self._record_change(target, target_version, chosen)
            self.conflicts.append(entry)
            return

        source_candidates = self._candidates(source)
        satisfiable: list[Version] = []
        for version in reversed(source_candidates):
            if self._requires_dist_satisfiable(
                source, version, self._path_for(source, version),
                self.extras.get(source, ()),
            ):
                satisfiable.append(version)
        if satisfiable:
            chosen = max(satisfiable)
            previous = self.versions[source]
            self.versions[source] = chosen
            self.paths[source] = self._path_for(source, chosen)
            entry.update({
                "action": "downgraded_dependent",
                "chosen_version": str(chosen),
                "previous_version": str(previous),
                "source_candidates_considered": [
                    str(version) for version in source_candidates
                ],
                "source_satisfiable_candidates": [
                    str(version) for version in satisfiable
                ],
                "reason": (
                    f"no {target} version satisfies {requirement} and the "
                    f"repository specifier {repository or '<any>'}; downgraded "
                    f"{source} from {previous} to {chosen}"
                ),
            })
            self._record_change(source, previous, chosen)
            self.conflicts.append(entry)
            return

        if via is not None and via in self.extras.get(source, ()):
            base_conflict = any(
                edge_via is None
                and canonicalize_name(edge_requirement.name) == target
                and not edge_requirement.specifier.contains(
                    target_version, prereleases=True
                )
                for edge_requirement, edge_via in self._edges_for(source)
            )
            if not base_conflict:
                self.extras[source].remove(via)
                if not self.extras[source]:
                    self.extras.pop(source, None)
                self.dropped_extras.setdefault(source, {}).setdefault(via, []).append(
                    f"joint conflict: {source}=={source_version} extra {via} "
                    f"requires {requirement} but selected {target}=={target_version}"
                )
                entry.update({
                    "action": "dropped_extra",
                    "chosen_version": str(source_version),
                    "source_candidates_considered": [
                        str(version) for version in source_candidates
                    ],
                    "source_satisfiable_candidates": [],
                    "reason": (
                        f"no {source} version has a satisfiable Requires-Dist; "
                        f"dropped extra {source}[{via}] that pulled {target}"
                    ),
                })
                self.conflicts.append(entry)
                return

        incoming = [
            (dependent, dependent_requirement, dependent_via)
            for dependent, dependent_requirement, dependent_via in self._all_edges()
            if canonicalize_name(dependent_requirement.name) == source
            and dependent != source
        ]
        base_incoming = [
            item for item in incoming if item[2] is None
        ]
        extra_incoming = [
            item for item in incoming if item[2] is not None
        ]
        dropped_extra_records: list[dict[str, str]] = []
        if not base_incoming:
            for dependent, _dependent_requirement, dependent_via in extra_incoming:
                if dependent_via not in self.extras.get(dependent, ()):
                    continue
                self.extras[dependent].remove(dependent_via)
                if not self.extras[dependent]:
                    self.extras.pop(dependent, None)
                self.dropped_extras.setdefault(
                    dependent, {}
                ).setdefault(dependent_via, []).append(
                    f"joint conflict: {source}=={source_version} dropped; "
                    f"{dependent} extra {dependent_via} depended on it"
                )
                dropped_extra_records.append({
                    "package": dependent, "extra": dependent_via,
                })
        self.versions.pop(source, None)
        self.paths.pop(source, None)
        self.extras.pop(source, None)
        entry.update({
            "action": "dropped_distribution",
            "dropped_version": str(source_version),
            "repository_pin": str(self.constraints.get(source, "")) or "<any>",
            "dropped_extras": dropped_extra_records,
            "source_candidates_considered": [
                str(version) for version in source_candidates
            ],
            "source_satisfiable_candidates": [],
            "transitively_required_by": [
                {
                    "package": dependent,
                    "requirement": str(dependent_requirement),
                    "via_extra": dependent_via,
                }
                for dependent, dependent_requirement, dependent_via in incoming
            ],
            "reason": (
                f"no {source} version has a satisfiable Requires-Dist and "
                f"{target} cannot change: {requirement} conflicts with "
                f"repository specifier {repository or '<any>'}; "
                f"dropped {source}=={source_version}"
            ),
        })
        self.conflicts.append(entry)

    def _handle_unselected(self) -> None:
        """Name transitively required distributions that no wheel can provide.

        The runner can add those wheels to ``extra_wheels``.  If the missing
        dependency was pulled only by an extra, dropping the extra is enough
        and is recorded through ``dropped_extras``.
        """
        for _ in range(200):
            changed = False
            for source, requirement, via in self._all_edges():
                target = canonicalize_name(requirement.name)
                if target in self.versions:
                    continue
                if self._requirement_available(requirement):
                    continue
                if via is not None and via in self.extras.get(source, ()):
                    self.extras[source].remove(via)
                    if not self.extras[source]:
                        self.extras.pop(source, None)
                    self.dropped_extras.setdefault(
                        source, {}
                    ).setdefault(via, []).append(
                        f"joint conflict: no wheel or installed distribution "
                        f"satisfies {requirement} required by {source} extra {via}"
                    )
                    self.conflicts.append({
                        "dependent": source,
                        "dependent_version": str(self.versions[source]),
                        "requirement": str(requirement),
                        "via_extra": via,
                        "selected": None,
                        "action": "dropped_extra",
                        "reason": (
                            f"no wheel or installed distribution satisfies "
                            f"{requirement} required by {source} extra {via}"
                        ),
                    })
                    changed = True
                    break
                if target not in self.missing:
                    self.missing[target] = (
                        f"{source}=={self.versions[source]} requires {requirement}"
                    )
            if not changed:
                return

    def _describe_conflict(
        self, item: tuple[str, Requirement, str | None, str],
    ) -> dict[str, Any]:
        source, requirement, via, target = item
        return {
            "dependent": source,
            "dependent_version": str(self.versions[source]),
            "requirement": str(requirement),
            "via_extra": via,
            "selected": f"{target}=={self.versions[target]}",
            "action": "unresolved",
            "reason": "joint compatibility conflict remains after resolution",
        }

    def run(self) -> None:
        for _ in range(200):
            conflicts = self._conflicts()
            if not conflicts:
                break
            self._handle(*conflicts[0])
        else:
            self.residual = [
                self._describe_conflict(item) for item in self._conflicts()
            ]
        self._handle_unselected()
        if not self.residual:
            self.residual = [
                self._describe_conflict(item) for item in self._conflicts()
            ]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "conflicts": self.conflicts,
            "changes": self.changes,
            "residual_conflicts": self.residual,
            "unreadable_metadata": self.unreadable,
            "resolved": not self.residual and not self.unreadable,
        }


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

    A joint pass then checks the selected wheels' declared ``Requires-Dist``
    entries.  A dependency version may change only inside the repository's own
    specifier, and a dependent may be downgraded or dropped instead; every
    cross-package action is recorded in ``joint_resolution``.  Distributions
    that pip would need transitively but no wheel provides are listed in
    ``missing_transitive_distributions`` rather than being fabricated.
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

    versions: dict[str, Version] = {}
    paths: dict[str, Path] = {}
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
        versions[name] = version
        paths[name] = selected_path
        extras = sorted(extras_by_name.get(name, ()))
        supported, dropped = _supported_extras(selected_path, extras, index)
        if supported:
            selected_extras[name] = supported
        if dropped:
            dropped_extras[name] = dropped

    resolver = _JointResolution(
        index, constraints, versions, paths, selected_extras, dropped_extras,
    )
    resolver.run()

    selected: list[str] = []
    for name in sorted(versions):
        version = str(versions[name])
        extras = sorted(selected_extras.get(name, ()))
        suffix = f"[{','.join(extras)}]" if extras else ""
        selected.append(f"{name}{suffix}=={version}")
    diagnostics = {
        "schema": "gemma.task_dependency_selection.v1",
        "wheel_dirs": [str(path) for path in wheel_dirs],
        "selected_versions": {name: str(version) for name, version in versions.items()},
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
        "joint_resolution": resolver.diagnostics(),
        "missing_transitive_distributions": dict(sorted(resolver.missing.items())),
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
        # Constraints must also reach pip, not only the explicit-name selector:
        # a repository may pin a transitive dependency (for example Starlette)
        # without naming it in its own requirements.
        import tempfile
        constraint_path: Path | None = None
        command = [sys.executable, "-m", "pip", "install", "--no-build-isolation"]
        if raw_constraints:
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", prefix="gemma-constraints-", suffix=".txt",
                delete=False,
            )
            try:
                handle.write("\n".join(raw_constraints) + "\n")
            finally:
                handle.close()
            constraint_path = Path(handle.name)
            command.extend(["-c", str(constraint_path)])
            diagnostics["constraints_passed_to_pip"] = len(raw_constraints)
        command.extend(selected)
        try:
            result = module.subprocess.run(command, capture_output=True, text=True, check=False)
        finally:
            if constraint_path is not None:
                constraint_path.unlink(missing_ok=True)
        if getattr(result, "returncode", 1):
            raise RuntimeError(
                "Version-aware task dependency installation failed: "
                + json.dumps({
                    "returncode": result.returncode,
                    "stdout": (result.stdout or "")[-4000:],
                    "stderr": (result.stderr or "")[-4000:],
                    # The wrapper prints this exception instead of the returned
                    # diagnostics when pip fails, so keep the joint and missing
                    # wheel evidence on the failure path too.
                    "missing_transitive_distributions":
                        diagnostics.get("missing_transitive_distributions", {}),
                    "joint_resolution": diagnostics.get("joint_resolution", {}),
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
