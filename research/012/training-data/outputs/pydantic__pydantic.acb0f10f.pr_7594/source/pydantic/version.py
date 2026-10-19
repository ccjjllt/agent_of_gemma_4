"""The `version` module holds the version information for Pydantic."""

from __future__ import annotations as _annotations

from pydantic_core import __version__ as __pydantic_core_version__

__all__ = 'VERSION', 'version_info'

VERSION = '2.11.0a2'
"""The version of Pydantic."""


def version_short() -> str:
    """Return the major.minor part of Pydantic version.
 
    It return '2.1' if Pydantic version is '2.1.1'.
    """
    return '.'.join(VERSION.split('.')[:2])


def version_info() -> str:
    """Return complete version information for Pydantic and its dependencies."""
    import os
    import platform
    import sys
    from importlib import import_module
    from pathlib import Path

    import pydantic_core._pydantic_core as pdc

    from ._internal import _git as git

    optional_deps = []
    for p in 'devtools', 'email-validator', 'typing-extensions':
        try:
            import_module(p.replace('-', '_'))
        except ImportError:  # pragma: no cover
            continue
        optional_deps.append(p)

    pydantic_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    most_recent_commit = (
        git.git_revision(pydantic_dir) if git.is_git_repo(pydantic_dir) and git.have_git() else 'unknown'
    )

    info = {
        'pydantic version': VERSION,
        'pydantic-core version': pdc.__version__,
        'pydantic-core build': getattr(pdc, 'build_info', None) or pdc.build_profile,
        'install path': Path(__file__).resolve().parent,
        'python version': sys.version,
        'platform': platform.platform(),
        'optional deps. installed': optional_deps,
        'commit': most_recent_commit,
    }
    return '\n'.join('{:>30} {}'.format(k + ':', str(v).replace('\n', ' ')) for k, v in info.items())


def check_pydantic_core_version() -> bool:
    """Check that the installed `pydantic-core` dependency is compatible."""
    # Keep this in sync with the version constraint in the `pyproject.toml` dependencies:
    return __pydantic_core_version__ == '2.30.0'


def parse_mypy_version(version: str) -> tuple[int, int, int]:
    """Parse `mypy` string version to a 3-tuple of ints.

    It parses normal version like `1.11.0` and extra info followed by a `+` sign
    like `1.11.0+dev.d6d9d8cd4f27c52edac1f537e236ec48a01e54cb.dirty`.

    Args:
        version: The mypy version string.

    Returns:
        A triple of ints, e.g. `(1, 11, 0)`.
    """
    return tuple(map(int, version.partition('+')[0].split('.')))  # pyright: ignore[reportReturnType]