"""ERL utilities: RAFT estimator, prompts, feedback, and prompt extraction."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from rllm.agents.agent import Trajectory
from rllm.experimental.common.advantage import register_rllm_adv_estimator

# ---------------------------------------------------------------------------
# Default advantage estimator map for the four ERL roles
# ---------------------------------------------------------------------------
DEFAULT_ERL_ADV_ESTIMATOR_MAP: dict[str, str] = {
    "erl_first": "grpo",
    "erl_updater": "grpo",
    "erl_second": "grpo",
    "erl_distill": "raft",
}


# ---------------------------------------------------------------------------
# Default updater system prompt
# ---------------------------------------------------------------------------
UPDATER_SYSTEM_PROMPT = (
    "You are an expert prompt updater. Analyze recent task attempts, "
    "their outcomes, and environmental feedback to improve the solver's "
    "system prompt. Return ONLY the revised prompt wrapped in "
    "<prompt>...</prompt> tags."
)


# ---------------------------------------------------------------------------
# RAFT advantage estimator
# ---------------------------------------------------------------------------
@register_rllm_adv_estimator("raft")
def calculate_raft_advantages(
    rewards: list[np.ndarray],
    **kwargs,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """RAFT: REINFORCE on positive-reward trajectories only.

    Implements the indicator gate from the ERL internalization objective.
    Negative-reward trajectories get zero advantage and do not contribute
    to the policy gradient.
    """
    advantages_by_group = []
    for group_rewards in rewards:
        adv = np.where(group_rewards > 0, group_rewards, 0.0)
        advantages_by_group.append(adv)
    return advantages_by_group, advantages_by_group


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------

_PROMPT_PATTERN = re.compile(r"<prompt>(.*?)</prompt>", flags=re.IGNORECASE | re.DOTALL)


def extract_prompt_from_response(text: str) -> str | None:
    """Extract a revised prompt enclosed in ``<prompt>...</prompt>`` tags."""
    match = _PROMPT_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Default feedback builder
# ---------------------------------------------------------------------------


def default_feedback(task: dict[str, Any], trajectory: Trajectory) -> str:
    """Build a generic one-line feedback string from a trajectory."""
    reward = trajectory.reward if trajectory.reward is not None else 0.0
    n_steps = len(trajectory.steps)
    succeeded = reward > 0
    return f"Reward={reward:.2f}, steps={n_steps}, succeeded={succeeded}."
