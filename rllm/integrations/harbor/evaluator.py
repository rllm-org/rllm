"""Harbor evaluator: scores Episodes using Harbor's container-based verification.

When used with HarborRuntime, the trial already ran verification (test.sh)
and the reward is stored in episode.artifacts. The evaluator just reads it.

When used with a non-Harbor agent (standalone mode), this evaluator cannot
run container verification -- it returns the pre-computed reward if available,
or 0.0 otherwise.
"""

from __future__ import annotations

import logging

from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode

logger = logging.getLogger(__name__)


class HarborEvaluator:
    """Evaluator that reads Harbor verification results from Episode artifacts.

    Harbor's trial pipeline runs test.sh during the trial itself. The reward
    is already computed by the time the Episode reaches the evaluator. This
    evaluator extracts it from ``episode.artifacts["harbor_reward"]``.
    """

    def evaluate(self, task: dict, episode: Episode) -> EvalOutput:
        # Integrated mode: HarborRuntime already ran the trial + verifier.
        # Check for the sentinel key to distinguish "trial ran" from "no trial".
        if episode.artifacts.get("harbor_trial_ran"):
            reward = float(episode.artifacts.get("harbor_reward", 0.0))
            is_correct = bool(episode.artifacts.get("harbor_is_correct", False))
            return EvalOutput(
                reward=reward,
                is_correct=is_correct,
                signals=[Signal(name="accuracy", value=1.0 if is_correct else 0.0)],
                metadata={"eval_mode": "harbor_integrated"},
            )

        # Fallback: check if there's a reward in trajectory metadata
        for traj in episode.trajectories:
            if traj.reward is not None:
                reward = float(traj.reward)
                is_correct = reward > 0
                return EvalOutput(
                    reward=reward,
                    is_correct=is_correct,
                    signals=[Signal(name="accuracy", value=1.0 if is_correct else 0.0)],
                    metadata={"eval_mode": "harbor_trajectory_fallback"},
                )

        # No Harbor reward available -- agent was not a HarborRuntime
        logger.warning(
            "HarborEvaluator: no harbor_trial_ran in episode artifacts for task '%s'. Harbor tasks require a Harbor agent (e.g., --agent harbor:mini-swe-agent) for container-based verification.",
            task.get("task_id", "unknown"),
        )
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="accuracy", value=0.0)],
            metadata={"eval_mode": "harbor_no_reward", "reason": "no_harbor_agent"},
        )
