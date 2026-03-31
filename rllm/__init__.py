"""rLLM: Reinforcement Learning with Language Models

Main package for the rLLM framework.
"""

import sys

__all__ = ["BaseAgent", "Action", "Step", "Trajectory", "Episode", "rollout", "evaluator"]


def __getattr__(name: str):
    if name in ("rollout", "evaluator"):
        from rllm.experimental.eval.rollout_decorator import evaluator, rollout

        _mod = sys.modules[__name__]
        _mod.rollout = rollout
        _mod.evaluator = evaluator
        return rollout if name == "rollout" else evaluator

    _agent_exports = {"BaseAgent", "Action", "Step", "Trajectory", "Episode"}
    if name in _agent_exports:
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
