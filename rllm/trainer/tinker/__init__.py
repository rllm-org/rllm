"""Trainer tinker package exports.

Provides:
- New backend: ``TinkerBackend``
- Legacy trainer symbols forwarded from ``rllm.trainer.deprecated``
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "TinkerBackend",
    "TinkerAgentTrainer",
    "TinkerSFTTrainer",
    "TinkerWorkflowTrainer",
]

if TYPE_CHECKING:
    from rllm.trainer.deprecated.tinker_agent_trainer import TinkerAgentTrainer
    from rllm.trainer.deprecated.tinker_sft_trainer import TinkerSFTTrainer
    from rllm.trainer.deprecated.tinker_workflow_trainer import TinkerWorkflowTrainer
    from rllm.trainer.tinker.tinker_backend import TinkerBackend


def __getattr__(name: str) -> Any:
    if name == "TinkerBackend":
        return import_module("rllm.trainer.tinker.tinker_backend").TinkerBackend
    if name in {"TinkerAgentTrainer", "TinkerSFTTrainer", "TinkerWorkflowTrainer"}:
        # Keep deprecated trainers available without importing deprecated package eagerly.
        return getattr(import_module("rllm.trainer.deprecated"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
