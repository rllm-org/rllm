from typing import TYPE_CHECKING

from .rollout_engine import ModelOutput, RolloutEngine
from .types import TinkerTokenInput, TinkerTokenOutput, TokenInput, Tokenizer, TokenOutput, VerlTokenInput, VerlTokenOutput

if TYPE_CHECKING:
    from .tinker_engine import TinkerEngine
    from .verl_engine import VerlEngine

__all__ = [
    "ModelOutput",
    "RolloutEngine",
    "OpenAIEngine",
    "TinkerEngine",
    "VerlEngine",
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
    if name == "OpenAIEngine":
        from .openai_engine import OpenAIEngine as _OpenAIEngine

        return _OpenAIEngine
    if name == "TinkerEngine":
        from .tinker_engine import TinkerEngine as _TinkerEngine

        return _TinkerEngine
    if name == "VerlEngine":
        try:
            from .verl_engine import VerlEngine as _VerlEngine

            return _VerlEngine
        except Exception:
            raise AttributeError(name) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
