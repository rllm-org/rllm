"""Engine module for rLLM.

This module contains the core execution infrastructure for agent trajectory rollout.
"""

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
    "TinkerEngine",
    "VerlEngine",
]


def __getattr__(name):
    if name in ("AgentExecutionEngine", "AsyncAgentExecutionEngine"):
        from .agent_execution_engine import AgentExecutionEngine, AsyncAgentExecutionEngine

        return AgentExecutionEngine if name == "AgentExecutionEngine" else AsyncAgentExecutionEngine
    if name == "AgentWorkflowEngine":
        from .agent_workflow_engine import AgentWorkflowEngine as _AgentWorkflowEngine

        return _AgentWorkflowEngine
    if name == "OpenAIEngine":
        from .rollout.openai_engine import OpenAIEngine

        return OpenAIEngine
    if name == "TinkerEngine":
        from .rollout.tinker_engine import TinkerEngine

        return TinkerEngine
    if name == "VerlEngine":
        from .rollout.verl_engine import VerlEngine

        return VerlEngine
    raise AttributeError(name)
