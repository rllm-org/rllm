"""Algorithm for RL advantage estimation and loss functions"""

import numpy as np


def calculate_grpo_advantages_per_group(rewards: np.ndarray, norm_adv_by_std_in_grpo=True, episilon=1e-6) -> np.ndarray:
    if len(rewards) <= 1:
        group_mean, group_std = 0.0, 1.0
    else:
        group_mean = np.mean(rewards)
        group_std = np.std(rewards)

    if norm_adv_by_std_in_grpo:
        advantages = (rewards - group_mean) / (group_std + episilon)
    else:
        advantages = rewards - group_mean

    return advantages


def calculate_rloo_advantages_per_group(rewards: np.ndarray) -> np.ndarray:
    num_trajs = len(rewards)
    if num_trajs <= 1:
        return rewards

    advantages = num_trajs / (num_trajs - 1) * (rewards - rewards.mean())
    return advantages
