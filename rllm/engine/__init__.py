"""Engine module for rLLM.

This module contains the core execution infrastructure for agent trajectory rollout.
"""

from .agent_execution_engine import AgentExecutionEngine, AsyncAgentExecutionEngine

# Avoid importing rollout submodules eagerly to prevent circular imports with workflows
# Import base class only (no side effects) and lazy-load specific engines via __getattr__
from .rollout.rollout_engine import ModelOutput, RolloutEngine

__all__ = [
    "AgentExecutionEngine",
    "AsyncAgentExecutionEngine",
    "AgentWorkflowEngine",
    "RolloutEngine",
    "ModelOutput",
    "OpenAIEngine",
    "VerlEngine",
]


def __getattr__(name):
    if name == "AgentWorkflowEngine":
        from .agent_workflow_engine import AgentWorkflowEngine as _AgentWorkflowEngine

        return _AgentWorkflowEngine
    if name == "OpenAIEngine":
        from .rollout.openai_engine import OpenAIEngine as _OpenAIEngine

        return _OpenAIEngine
    if name == "VerlEngine":
        try:
            from .rollout.verl_engine import VerlEngine as _VerlEngine

            return _VerlEngine
        except Exception:
            raise AttributeError(name) from None
    raise AttributeError(name)
