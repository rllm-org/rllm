from typing import TYPE_CHECKING

from .rollout_engine import ModelOutput, RolloutEngine
from .types import TinkerTokenInput, TinkerTokenOutput, TokenInput, Tokenizer, TokenOutput, VerlTokenInput, VerlTokenOutput

if TYPE_CHECKING:
    from .tinker_engine import TinkerEngine
    from .verl_engine import VerlEngine

__all__ = [
    "ModelOutput",
    # Rollout engines
    "RolloutEngine",
    "TinkerEngine",
    "VerlEngine",
    # Token input/output types
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
        from .tinker_engine import TinkerEngine as _TinkerEngine

        return _TinkerEngine
    if name == "VerlEngine":
        from .verl_engine import VerlEngine as _VerlEngine

        return _VerlEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
