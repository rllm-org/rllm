"""Engine module for rLLM.

This module contains the core execution infrastructure for agent trajectory rollout.
"""

from .workflow import TerminationEvent, TerminationReason, Workflow

__all__ = [
    "Workflow",
    "TerminationReason",
    "TerminationEvent",
    "SingleTurnWorkflow",
    "MultiTurnWorkflow",
]


def __getattr__(name):
    if name == "SingleTurnWorkflow":
        from .single_turn_workflow import SingleTurnWorkflow as _Single

        return _Single
    if name == "MultiTurnWorkflow":
        from .multi_turn_workflow import MultiTurnWorkflow as _Multi

        return _Multi
    raise AttributeError(name)
