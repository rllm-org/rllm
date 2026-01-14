"""
Verl-native advantage computation -- serve as opt-in for verl backend to replace
the default rLLM-native advantage computation.
"""

import numpy as np
import torch
from omegaconf import DictConfig
from verl import DataProto
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage


def compute_advantage_verl(batch: DataProto, config: DictConfig) -> tuple[DataProto, dict]:
    """Verl-native advantage computation."""
    metrics = {}
    batch.non_tensor_batch["uid"] = batch.non_tensor_batch["trajectory_ids"]

    if config.rllm.stepwise_advantage.mode == "per_step":
        batch.batch["token_level_scores"] = batch.batch["step_rewards"]
    else:
        batch.batch["token_level_scores"] = batch.batch["traj_rewards"]

    if config.algorithm.use_kl_in_reward:
        batch, kl_metrics = apply_kl_penalty(
            batch,
            kl_ctrl=config.kl_ctrl_in_reward,  # type: ignore[arg-type]
            kl_penalty=config.algorithm.kl_penalty,
        )
        metrics.update(kl_metrics)
    else:
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

    if config.rllm.stepwise_advantage.mode == "broadcast":
        is_last_step = batch.non_tensor_batch["is_last_step"]
        last_step_indices = np.where(is_last_step == True)[0]
        not_last_step_indices = np.where(is_last_step == False)[0]
        non_last_step_batch = batch.select_idxs(not_last_step_indices)
        batch = batch.select_idxs(last_step_indices)
    else:
        batch = _remove_padding(batch)

    batch = compute_advantage(
        batch,
        adv_estimator=config.algorithm.adv_estimator,
        gamma=config.algorithm.gamma,
        lam=config.algorithm.lam,
        num_repeat=config.actor_rollout_ref.rollout.n,
        norm_adv_by_std_in_grpo=config.algorithm.norm_adv_by_std_in_grpo,
        config=config.algorithm,
    )

    if config.rllm.stepwise_advantage.mode == "broadcast":
        _stepwise_advantage_broadcast(batch, non_last_step_batch, config)
        batch = DataProto.concat([batch, non_last_step_batch])

    return batch, metrics


def _stepwise_advantage_broadcast(last_step_batch: DataProto, non_last_step_batch: DataProto, config: DictConfig) -> None:
    """Broadcast advantage from last step to all other steps."""
    src_traj_ids = last_step_batch.non_tensor_batch["trajectory_ids"]
    src_eps_ids = last_step_batch.non_tensor_batch["episode_ids"]
    src_steps = last_step_batch.non_tensor_batch["step_nums"]
    src_mask = last_step_batch.batch["response_mask"]
    src_advantages = last_step_batch.batch["advantages"]

    tgt_traj_ids = non_last_step_batch.non_tensor_batch["trajectory_ids"]
    tgt_eps_ids = non_last_step_batch.non_tensor_batch["episode_ids"]
    tgt_mask = non_last_step_batch.batch["response_mask"]

    traj_ep_to_scalar_adv = {}
    for i, (traj_id, eps_id) in enumerate(zip(src_traj_ids, src_eps_ids, strict=False)):
        mask = src_mask[i].bool()
        scalar = src_advantages[i][mask].mean()

        if config.rllm.stepwise_advantage.get("normalize_by_steps", False):
            scalar = scalar / src_steps[i]
            last_step_batch.batch["advantages"][i][mask] = scalar

        traj_ep_to_scalar_adv[(traj_id, eps_id)] = scalar

    scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=traj_ep_to_scalar_adv[(traj_id, eps_id)], dtype=torch.float32) for i, (traj_id, eps_id) in enumerate(zip(tgt_traj_ids, tgt_eps_ids, strict=False))])

    final_advantage = scalar_rows * tgt_mask
    non_last_step_batch.batch["advantages"] = final_advantage
    non_last_step_batch.batch["returns"] = final_advantage


def _remove_padding(batch: DataProto) -> DataProto:
    """Remove padded steps from batch."""
    is_pad_step = batch.non_tensor_batch["is_pad_step"]
    non_pad_step_indices = np.where(is_pad_step == False)[0]
    return batch.select_idxs(non_pad_step_indices)
