"""rLLM: Reinforcement Learning with Language Models

Main package for the rLLM framework.
"""

# Import commonly used classes
from .agents import Action, BaseAgent, Episode, Step, Trajectory
from .engine import AgentWorkflowEngine, OpenAIEngine, RolloutEngine

__all__ = [
    "BaseAgent",
    "Action",
    "Step",
    "Trajectory",
    "Episode",
    "AgentWorkflowEngine",
    "RolloutEngine",
    "OpenAIEngine",
]

# VerlEngine is optional; only export if verl is installed
try:
    from .engine import VerlEngine

    __all__.append("VerlEngine")
except Exception:
    VerlEngine = None
