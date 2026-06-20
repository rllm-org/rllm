"""Trainer tinker package exports.

Provides the RL ``TinkerBackend``. (Tinker SFT now lives in
:mod:`rllm.trainer.sft`.)
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["TinkerBackend"]

if TYPE_CHECKING:
    from rllm.trainer.tinker.tinker_backend import TinkerBackend


def __getattr__(name: str) -> Any:
    if name == "TinkerBackend":
        return import_module("rllm.trainer.tinker.tinker_backend").TinkerBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
