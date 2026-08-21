"""Trainer miles package exports.

Provides the RL ``MilesBackend``: rLLM's UnifiedTrainer drives the loop, Miles
(SGLang rollout + Megatron-LM / FSDP2 training) owns the GPUs. See
``design/miles-training-backend.md``.

Lazy attribute access keeps ``import rllm.trainer.miles`` cheap: the backend and
launcher import miles (and therefore megatron), which only exist inside the Miles
container image. The transform and config bridge stay importable anywhere.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["MilesBackend", "MilesTrainerLauncher"]

if TYPE_CHECKING:
    from rllm.trainer.miles.miles_backend import MilesBackend
    from rllm.trainer.miles.miles_launcher import MilesTrainerLauncher

_LAZY = {
    "MilesBackend": ("rllm.trainer.miles.miles_backend", "MilesBackend"),
    "MilesTrainerLauncher": ("rllm.trainer.miles.miles_launcher", "MilesTrainerLauncher"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        return getattr(import_module(module_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
