"""Backward-compatibility shim — all rollout engines have moved to ``rllm.engine.rollout``.

Importing from ``rllm.experimental.rollout`` still works but will emit a
``DeprecationWarning``.  Please update your imports to use
``rllm.engine.rollout`` instead.
"""

import warnings

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.engine.rollout.types import TinkerTokenInput, TinkerTokenOutput, TokenInput, Tokenizer, TokenOutput, VerlTokenInput, VerlTokenOutput

warnings.warn(
    "Importing from 'rllm.experimental.rollout' is deprecated. Use 'rllm.engine.rollout' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ModelOutput",
    "RolloutEngine",
    "TinkerEngine",
    "VerlEngine",
    "Completer",
    "TITOCompleter",
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
    if name == "TinkerEngine":
        try:
            from rllm.engine.rollout.tinker_engine import TinkerEngine as _TinkerEngine

            return _TinkerEngine
        except Exception:
            raise AttributeError(name) from None
    if name == "VerlEngine":
        try:
            from rllm.engine.rollout.verl_engine import VerlEngine as _VerlEngine

            return _VerlEngine
        except Exception:
            raise AttributeError(name) from None
    if name in ("Completer", "TITOCompleter"):
        from rllm.engine.rollout.completer import Completer, TITOCompleter

        return {"Completer": Completer, "TITOCompleter": TITOCompleter}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
