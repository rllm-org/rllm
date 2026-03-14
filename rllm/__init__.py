"""rLLM: Reinforcement Learning with Language Models

Main package for the rLLM framework.
"""

import sys

__all__ = ["BaseAgent", "Action", "Step", "Trajectory", "Episode"]


def __getattr__(name: str):
    if name in __all__:
        from rllm.agents.agent import Action, BaseAgent, Episode, Step, Trajectory

        _exports = {
            "BaseAgent": BaseAgent,
            "Action": Action,
            "Step": Step,
            "Trajectory": Trajectory,
            "Episode": Episode,
        }
        # Cache on the module so __getattr__ isn't called again
        _mod = sys.modules[__name__]
        for k, v in _exports.items():
            setattr(_mod, k, v)
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
