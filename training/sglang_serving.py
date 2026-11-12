"""Deprecated import shim for the retired SGLang serving engine.

The active path uses :mod:`training.remote_serving` and the independent vLLM
backend.  This module exists only so older imports fail loudly with a clear
name instead of silently selecting SGLang.
"""
from __future__ import annotations

from .remote_serving import RemoteServingEngine

SGLangServingEngine = RemoteServingEngine

__all__ = ["RemoteServingEngine", "SGLangServingEngine"]
