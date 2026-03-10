# Backward compatibility: re-export from canonical location
from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine  # noqa: F401
from rllm.engine.rollout.types import (  # noqa: F401
    TinkerTokenInput,
    TinkerTokenOutput,
    TokenInput,
    Tokenizer,
    TokenOutput,
    VerlTokenInput,
    VerlTokenOutput,
)

__all__ = [
    "ModelOutput",
    "RolloutEngine",
    "TinkerEngine",
    "VerlEngine",
    "TokenInput",
    "TokenOutput",
    "TinkerTokenInput",
    "TinkerTokenOutput",
    "VerlTokenInput",
    "VerlTokenOutput",
    "Tokenizer",
]


def __getattr__(name):
    # Lazy imports for engines with heavy dependencies
    if name == "TinkerEngine":
        from rllm.engine.rollout.tinker_engine import TinkerEngine as _TinkerEngine

        return _TinkerEngine
    if name == "VerlEngine":
        try:
            from rllm.engine.rollout.verl_engine import VerlEngine as _VerlEngine

            return _VerlEngine
        except Exception:
            raise AttributeError(name) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
