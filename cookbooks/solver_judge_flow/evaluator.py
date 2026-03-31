"""Solver-Judge evaluator: scores solver and judge trajectories independently."""

from __future__ import annotations

import rllm
from rllm.experimental.eval.types import EvalOutput, Signal
from rllm.rewards.countdown_reward import compute_score
from rllm.types import Episode


@rllm.evaluator
def solver_judge_countdown_evaluator(task: dict, episode: Episode) -> EvalOutput:
    """Score solver and judge trajectories independently.

    Sets per-trajectory rewards so GRPO can compute advantages separately
    for solver vs judge trajectory groups.
    """
    ground_truth = {"target": task["target"], "numbers": task["nums"]}

    solver_correct = 0
    solver_total = 0
    judge_reward = 0.0
    is_correct = False

    for traj in episode.trajectories:
        answer = traj.steps[-1].action if traj.steps else ""
        score = compute_score(str(answer), ground_truth)
        reward = 1.0 if score >= 1.0 else 0.0
        traj.reward = reward

        if traj.name == "solver":
            solver_total += 1
            solver_correct += int(reward >= 1.0)
        elif traj.name == "judge":
            judge_reward = reward
            is_correct = reward >= 1.0

    solver_acc = solver_correct / solver_total if solver_total > 0 else 0.0
    return EvalOutput(
        reward=judge_reward,
        is_correct=is_correct,
        signals=[
            Signal(name="solver_acc", value=solver_acc),
            Signal(name="judge_acc", value=float(is_correct)),
        ],
    )
