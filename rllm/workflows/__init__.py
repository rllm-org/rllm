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
    "CumulativeWorkflow",
    "SingleTurnWorkflowWithEarlyFinalize",
    "MultiTurnWorkflowWithEarlyFinalize",
    "CumulativeWorkflowWithEarlyFinalize",
    "TimingTrackingMixin",
]


def __getattr__(name):
    if name == "SingleTurnWorkflow":
        from .single_turn_workflow import SingleTurnWorkflow as _Single

        return _Single
    if name == "MultiTurnWorkflow":
        from .multi_turn_workflow import MultiTurnWorkflow as _Multi

        return _Multi
    if name == "CumulativeWorkflow":
        from .cumulative_workflow import CumulativeWorkflow as _Cumulative

        return _Cumulative
    if name == "SingleTurnWorkflowWithEarlyFinalize":
        from .early_finalize_workflows import SingleTurnWorkflowWithEarlyFinalize as _SingleEarlyFinalize

        return _SingleEarlyFinalize
    if name == "MultiTurnWorkflowWithEarlyFinalize":
        from .early_finalize_workflows import MultiTurnWorkflowWithEarlyFinalize as _MultiEarlyFinalize

        return _MultiEarlyFinalize
    if name == "CumulativeWorkflowWithEarlyFinalize":
        from .early_finalize_workflows import CumulativeWorkflowWithEarlyFinalize as _CumulativeEarlyFinalize

        return _CumulativeEarlyFinalize
    if name == "TimingTrackingMixin":
        from .timing_mixin import TimingTrackingMixin as _Mixin

        return _Mixin
    raise AttributeError(name)
