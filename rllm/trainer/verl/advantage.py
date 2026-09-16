"""Trajectory-level GRPO for the legacy workflow trainer's merged rows."""

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def compute_workflow_grpo_advantage(batch: DataProto, *, norm_adv_by_std_in_grpo: bool = True, normalize_by_steps: bool = False) -> DataProto:
    """Compare each rollout once per task/role and broadcast to its action tokens.

    ``step_ids`` identifies a trajectory instance; ``advantage_group_ids``
    identifies the task/role shared by its sibling rollouts. A context reset
    can split one trajectory into several rows. Count its outcome only once,
    retaining reward shaping (e.g. KL penalties) from every segment. Padding
    and invalid rows must not contribute to either reward or group statistics.

    Both non-stepwise and broadcast GRPO use this trajectory-level contract.
    """
    valid = ~batch.non_tensor_batch["is_pad_step"] & batch.non_tensor_batch["is_valid"]
    batch = batch.select_idxs(np.flatnonzero(valid))
    if len(batch) == 0:
        batch.batch["advantages"] = torch.zeros_like(batch.batch["token_level_rewards"])
        batch.batch["returns"] = batch.batch["advantages"].clone()
        return batch

    _, first_rows, inverse = np.unique(batch.non_tensor_batch["step_ids"], return_index=True, return_inverse=True)
    rewards = batch.batch["token_level_rewards"]
    scores = batch.batch["token_level_scores"]
    row_to_trajectory = torch.as_tensor(inverse, device=rewards.device)
    first = torch.as_tensor(first_rows, device=rewards.device)

    # The transform repeats the outcome on every segment. Sum only the
    # shaping terms, then add one copy of the original trajectory outcome.
    outcomes = scores.sum(dim=-1)[first].clone()
    outcomes.scatter_add_(0, row_to_trajectory, (rewards - scores).sum(dim=-1))
    group_ids = batch.non_tensor_batch["advantage_group_ids"][first_rows]
    trajectory_advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=outcomes.unsqueeze(-1),
        response_mask=torch.ones_like(outcomes).unsqueeze(-1),
        index=group_ids,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
    if normalize_by_steps:
        step_counts = torch.as_tensor(batch.non_tensor_batch["step_nums"][first_rows], device=rewards.device, dtype=rewards.dtype)
        trajectory_advantages = trajectory_advantages / step_counts.unsqueeze(-1)

    advantages = trajectory_advantages[row_to_trajectory] * batch.batch["response_mask"]
    batch.non_tensor_batch["uid"] = batch.non_tensor_batch["advantage_group_ids"].copy()
    batch.batch["advantages"] = advantages
    batch.batch["returns"] = advantages.clone()
    return batch
