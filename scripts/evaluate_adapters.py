#!/usr/bin/env python3
"""Thin CLI for :mod:`training.eval_adapters`.

The implementation and its argument contract live in the module so the same
entry point is importable from the server learner environment.  This wrapper
only fixes ``sys.path`` for execution from a checkout.
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.eval_adapters import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
