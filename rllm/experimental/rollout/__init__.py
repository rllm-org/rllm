"""Experimental rollout engines.

The base engine stack (``RolloutEngine``, ``TinkerEngine``, token types) lives
in ``rllm.engine.rollout``; this package re-exports it alongside the
experimental ``FireworksEngine``.
"""

from typing import TYPE_CHECKING

from rllm.engine.rollout import ModelOutput, RolloutEngine
from rllm.engine.rollout.types import TinkerTokenInput, TinkerTokenOutput, TokenInput, Tokenizer, TokenOutput, VerlTokenInput, VerlTokenOutput

if TYPE_CHECKING:
    from .fireworks_engine import FireworksEngine

__all__ = [
    "ModelOutput",
    "RolloutEngine",
    "FireworksEngine",
    # Token types
    "TokenInput",
    "TokenOutput",
    "TinkerTokenInput",
    "TinkerTokenOutput",
    "VerlTokenInput",
    "VerlTokenOutput",
    "Tokenizer",
]


def __getattr__(name):
    if name == "FireworksEngine":
        from .fireworks_engine import FireworksEngine as _FireworksEngine

        return _FireworksEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
