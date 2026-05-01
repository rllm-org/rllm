"""Backward-compat re-export shim for the trajectory data types + BaseAgent.

The canonical home for these types is :mod:`rllm.types`. ``rllm.agents``
remains as a re-export point for legacy ``from rllm.agents import …``
callers; the only locally-defined symbol is :class:`BaseAgent`, which is
the ABC for the legacy ``Agent + Environment`` execution path. New
agents should implement :class:`rllm.types.AgentFlow` instead.
"""

from rllm.agents.agent import BaseAgent
from rllm.types import Action, Episode, Step, Trajectory

__all__ = [
    "BaseAgent",
    "Action",
    "Step",
    "Trajectory",
    "Episode",
]
