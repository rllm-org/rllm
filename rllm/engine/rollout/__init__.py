# Avoid importing concrete engines at module import time to prevent circular imports
from .rollout_engine import ModelOutput, RolloutEngine

__all__ = [
    "ModelOutput",
    "RolloutEngine",
    "OpenAIEngine",
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
    if name == "OpenAIEngine":
        from .openai_engine import OpenAIEngine as _OpenAIEngine

        return _OpenAIEngine
    if name == "TinkerEngine":
        try:
            from .tinker_engine import TinkerEngine as _TinkerEngine

            return _TinkerEngine
        except Exception:
            raise AttributeError(name) from None
    if name == "VerlEngine":
        try:
            from .verl_engine import VerlEngine as _VerlEngine

            return _VerlEngine
        except Exception:
            raise AttributeError(name) from None
    if name in ("Completer", "TITOCompleter"):
        from .completer import Completer, TITOCompleter

        return {"Completer": Completer, "TITOCompleter": TITOCompleter}[name]
    # Token types
    if name in ("TokenInput", "TokenOutput", "TinkerTokenInput", "TinkerTokenOutput", "VerlTokenInput", "VerlTokenOutput", "Tokenizer"):
        from . import types as _types

        return getattr(_types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
