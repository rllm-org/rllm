from .rollout_engine import ModelOutput, RolloutEngine

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


def __getattr__(name: str):
    """Lazy import for engines and types to avoid importing tinker/verl when not needed."""
    if name == "TinkerEngine":
        from .tinker_engine import TinkerEngine
        return TinkerEngine
    if name == "VerlEngine":
        try:
            from .verl_engine import VerlEngine as _VerlEngine
            return _VerlEngine
        except Exception:
            raise AttributeError(name) from None
    # Lazy import for types
    if name in ("TinkerTokenInput", "TinkerTokenOutput", "TokenInput", "TokenOutput", "VerlTokenInput", "VerlTokenOutput", "Tokenizer"):
        from .types import (
            TinkerTokenInput as _TinkerTokenInput,
            TinkerTokenOutput as _TinkerTokenOutput,
            TokenInput as _TokenInput,
            TokenOutput as _TokenOutput,
            VerlTokenInput as _VerlTokenInput,
            VerlTokenOutput as _VerlTokenOutput,
            Tokenizer as _Tokenizer,
        )
        type_map = {
            "TinkerTokenInput": _TinkerTokenInput,
            "TinkerTokenOutput": _TinkerTokenOutput,
            "TokenInput": _TokenInput,
            "TokenOutput": _TokenOutput,
            "VerlTokenInput": _VerlTokenInput,
            "VerlTokenOutput": _VerlTokenOutput,
            "Tokenizer": _Tokenizer,
        }
        return type_map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
