"""Host-side CyberGym evaluator. Reads artifacts from CyberGymRuntime."""

from __future__ import annotations

import logging

from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode

logger = logging.getLogger(__name__)


class CyberGymEvaluator:
    """Score a CyberGym episode from Harbor-trial artifacts.

    Infra failures must be raised by :class:`CyberGymRuntime` before this
    evaluator runs. A missing trial is a misconfigured agent, not reward 0
    from a sidecar outage.
    """

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        artifacts = episode.artifacts or {}
        if artifacts.get("cybergym_trial_ran") or artifacts.get("harbor_trial_ran"):
            reward = float(artifacts.get("cybergym_reward", artifacts.get("harbor_reward", 0.0)) or 0.0)
            is_correct = bool(reward > 0)
            return EvalOutput(
                reward=reward,
                is_correct=is_correct,
                signals=[Signal(name="accuracy", value=1.0 if is_correct else 0.0)],
                metadata={
                    "eval_mode": "cybergym_harbor",
                    "vul_exit_code": artifacts.get("vul_exit_code"),
                    "fix_exit_code": artifacts.get("fix_exit_code"),
                    "n_submissions": artifacts.get("n_submissions"),
                    "cybergym_scoring": artifacts.get("cybergym_scoring"),
                    "scored_poc": artifacts.get("cybergym_scored_poc"),
                },
            )

        for traj in episode.trajectories:
            if traj.reward is not None:
                reward = float(traj.reward)
                return EvalOutput(
                    reward=reward,
                    is_correct=reward > 0,
                    signals=[Signal(name="accuracy", value=1.0 if reward > 0 else 0.0)],
                    metadata={"eval_mode": "cybergym_trajectory_fallback"},
                )

        raise RuntimeError(
            "CyberGymEvaluator: no Harbor/CyberGym trial artifacts on episode "
            f"'{task.get('task_id', getattr(task, 'id', 'unknown'))}'. "
            "Use --agent cybergym:<scaffold> or --agent harbor:<scaffold>. "
            "A missing trial is not reward 0."
        )
