"""Engine module for rLLM.

This module contains the core execution infrastructure for agent trajectory rollout.
"""

from .agent_execution_engine import AgentExecutionEngine, AsyncAgentExecutionEngine
from .agent_workflow_engine import AgentWorkflowEngine
from .rollout.openai_engine import OpenAIEngine
from .rollout.rollout_engine import RolloutEngine

__all__ = [
    "AgentExecutionEngine",
    "AsyncAgentExecutionEngine",
    "AgentWorkflowEngine",
    "RolloutEngine",
    "OpenAIEngine",
]

# VerlEngine is optional; only export if verl is installed
try:
    from .rollout.verl_engine import VerlEngine

    __all__.append("VerlEngine")
except Exception:
    VerlEngine = None
