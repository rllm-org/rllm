"""
Verl backend implementation for the UnifiedTrainer.

This backend inherits from both BackendProtocol and RayPPOTrainer to provide
verl-specific implementations while reusing verl's worker group infrastructure.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import DictConfig

from rllm.agents.agent import Episode, TrajectoryGroup
from rllm.data import Dataset
from rllm.engine.rollout import RolloutEngine
from rllm.engine.rollout.verl_engine import VerlEngine
from rllm.experimental.base import BackendProtocol
from rllm.trainer.common.advantage import AlgorithmConfig, compute_advantage_from_trajectory_groups
from rllm.trainer.verl.verl_data_processor.transform import transform_trajectory_groups_to_dataproto, update_dataproto_with_advantages
from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, apply_kl_penalty, compute_advantage
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.metric import reduce_metrics

if TYPE_CHECKING:
    from rllm.engine.unified_workflow_engine import UnifiedWorkflowEngine
    from rllm.experimental.unified_trainer import TrainerState


class VerlBackend(BackendProtocol[Iterable, DataProto], RayPPOTrainer):
    """Verl backend for the unified trainer.

    Inherits from both BackendProtocol and RayPPOTrainer to:
    - Provide the BackendProtocol interface for UnifiedTrainer
    - Reuse RayPPOTrainer's worker group infrastructure and utilities

    This significantly reduces code duplication by leveraging RayPPOTrainer's:
    - Worker group creation and management (actor, critic, ref policy)
    - Async rollout manager
    - Checkpoint loading/saving
    - Batch balancing
    - Profiling utilities
    """

    name: str = "verl"
    requires_loop: bool = True
    requires_preprocess_dataset: bool = False

    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        **kwargs,
    ):
        """Initialize the VerlBackend.

        Args:
            config: The full configuration object.
            tokenizer: The tokenizer for encoding/decoding.
            role_worker_mapping: Mapping from roles to worker types.
            resource_pool_manager: Manager for GPU resource pools.
            ray_worker_group_cls: Class for creating Ray worker groups.
            processor: Optional multimodal processor.
            reward_fn: Optional reward function for training.
            val_reward_fn: Optional reward function for validation.
            **kwargs: Additional arguments.
        """
        # Initialize RayPPOTrainer first - this sets up all worker groups
        RayPPOTrainer.__init__(
            self,
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

        # Initialize BackendProtocol
        BackendProtocol.__init__(self, config, **kwargs)

        # Store full config reference (RayPPOTrainer uses self.config)
        self.full_config = config

        # Rollout engine - will be created in init_rollout_engine
        self.rollout_engine: VerlEngine | None = None

        # Workflow engine reference - to be set by unified trainer
        self.workflow_engine: UnifiedWorkflowEngine | None = None

    # =========================================================================
    # BackendProtocol interface methods
    # =========================================================================

    def init_rollout_engine(self) -> RolloutEngine:
        """Initialize the VerlEngine rollout engine.

        Note: This should be called after init_workers() to ensure
        async_rollout_manager is available.

        Returns:
            VerlEngine: The initialized rollout engine.
        """
        if not hasattr(self, "async_rollout_manager") or self.async_rollout_manager is None:
            raise ValueError("async_rollout_manager is not available. Make sure init_workers() is called before init_rollout_engine().")

        self.rollout_engine = VerlEngine(
            config=self.config,
            rollout_manager=self.async_rollout_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        return self.rollout_engine

    def validate_config(self) -> None:
        """Validate verl-specific configuration settings."""
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported for VerlBackend"
        assert self.use_rm is False, "Reward models are not supported. Rewards should be assigned using a reward function in the workflow or environment."

    def get_dataloader(self, dataset: Dataset) -> Iterable:
        """Get dataloader from dataset."""
        # The dataset should implement __iter__ or provide a dataloader
        # Assuming dataset provides an iterable interface
        return dataset  # type: ignore[return-value]

    def shutdown(self) -> None:
        """Shutdown the backend and cleanup resources."""
        self.rollout_engine = None
        self.workflow_engine = None

    def set_workflow_engine(self, workflow_engine: UnifiedWorkflowEngine) -> None:
        """Set the workflow engine reference.

        Args:
            workflow_engine: The workflow engine to use for episode generation.
        """
        self.workflow_engine = workflow_engine

    def generate_episodes(self, batch: Any, **kwargs) -> list[Episode]:
        """Generate episodes using the workflow engine.

        Args:
            batch: Input batch (dict format from dataloader).
            **kwargs: Additional arguments including 'loop' for async execution.

        Returns:
            List of generated episodes.
        """
        assert "loop" in kwargs and kwargs["loop"] is not None, "async event loop is required"
        loop = kwargs["loop"]

        if self.workflow_engine is None:
            raise ValueError("workflow_engine is not set. Call set_workflow_engine() first.")

        # Convert batch to DataProto if needed
        if isinstance(batch, dict):
            batch = DataProto.from_single_dict(batch)

        # Add task IDs for tracking
        batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        # Repeat for multiple rollouts per task
        n_rollouts = self.config.actor_rollout_ref.rollout.n
        batch = batch.repeat(repeat_times=n_rollouts)

        # Remove fields not needed for environment-based interaction
        batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=["raw_prompt_ids"])

        # Execute tasks using workflow engine
        coro = self.workflow_engine.execute_tasks_verl(batch)

        if loop is not None:
            episodes = asyncio.run_coroutine_threadsafe(coro, loop).result()
        else:
            episodes = asyncio.run(coro)

        return episodes

    def transform_trajectory_groups_to_backend_batch(self, trajectory_groups: list[TrajectoryGroup], **kwargs) -> DataProto:
        """Transform trajectory groups to verl DataProto format."""
        if self.rollout_engine is None:
            raise ValueError("rollout_engine is not initialized.")

        return transform_trajectory_groups_to_dataproto(
            trajectory_groups,
            rollout_engine=self.rollout_engine,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
        )

    def process_backend_batch(self, batch: DataProto, **kwargs) -> DataProto:
        """Compute step-level values: old_log_probs, ref_log_probs, critic values.

        Reuses logic from AgentWorkflowPPOTrainer._compute_step_level_values.
        """
        metrics = {}

        # Compute old_log_probs from actor
        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
        entropys = old_log_prob.batch["entropys"]
        response_masks = batch.batch["response_mask"]
        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
        metrics["actor/entropy"] = entropy_agg.detach().item()
        old_log_prob.batch.pop("entropys")
        batch = batch.union(old_log_prob)

        # Compute rollout log prob diff if available
        if "rollout_log_probs" in batch.batch.keys():
            rollout_old_log_probs = batch.batch["rollout_log_probs"]
            actor_old_log_probs = batch.batch["old_log_probs"]
            attention_mask = batch.batch["attention_mask"]
            responses = batch.batch["responses"]
            response_length = responses.size(1)
            response_mask = attention_mask[:, -response_length:]

            rollout_probs = torch.exp(rollout_old_log_probs)
            actor_probs = torch.exp(actor_old_log_probs)
            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())

            metrics.update(
                {
                    "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
                    "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
                    "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
                }
            )

        # Compute reference log_probs if using reference policy
        if self.use_reference_policy:
            if not self.ref_in_actor:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
            else:
                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

        # Compute critic values if using critic
        if self.use_critic:
            values = self.critic_wg.compute_values(batch)
            batch = batch.union(values)

        # Mask truncated samples if configured
        if self.config.rllm.get("mask_truncated_samples", False):
            mask = batch.batch["attention_mask"][:, -1] == 1
            batch = batch[~mask]  # type: ignore[assignment]

        return batch

    def compute_advantages(self, trainer_state: TrainerState, algorithm_config: AlgorithmConfig, **kwargs) -> None:
        """Compute advantages from trajectory groups."""
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"
        trajectory_groups: list[TrajectoryGroup] = trainer_state.trajectory_groups
        batch: DataProto = trainer_state.backend_batch  # type: ignore[assignment]

        use_rllm_advantage = self.config.rllm.stepwise_advantage.get("use_rllm_advantage", False)

        if use_rllm_advantage:
            compute_advantage_from_trajectory_groups(trajectory_groups, algorithm_config)
            updated_batch = update_dataproto_with_advantages(batch, trajectory_groups, mode=self.config.rllm.stepwise_advantage.mode)
        else:
            updated_batch, adv_metrics = self._compute_advantage_verl(batch)  # type: ignore[return-value]
            trainer_state.metrics.update(adv_metrics)

        trainer_state.backend_batch = updated_batch

    def _compute_advantage_verl(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Verl-native advantage computation."""
        metrics = {}
        batch.non_tensor_batch["uid"] = batch.non_tensor_batch["trajectory_ids"]

        if self.config.rllm.stepwise_advantage.mode == "per_step":
            batch.batch["token_level_scores"] = batch.batch["step_rewards"]
        else:
            batch.batch["token_level_scores"] = batch.batch["traj_rewards"]

        if self.config.algorithm.use_kl_in_reward:
            batch, kl_metrics = apply_kl_penalty(
                batch,
                kl_ctrl=self.kl_ctrl_in_reward,  # type: ignore[arg-type]
                kl_penalty=self.config.algorithm.kl_penalty,
            )
            metrics.update(kl_metrics)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        if self.config.rllm.stepwise_advantage.mode == "broadcast":
            is_last_step = batch.non_tensor_batch["is_last_step"]
            last_step_indices = np.where(is_last_step == True)[0]
            not_last_step_indices = np.where(is_last_step == False)[0]
            non_last_step_batch = batch.select_idxs(not_last_step_indices)
            batch = batch.select_idxs(last_step_indices)
        else:
            batch = self._remove_padding(batch)

        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
            config=self.config.algorithm,
        )

        if self.config.rllm.stepwise_advantage.mode == "broadcast":
            self._stepwise_advantage_broadcast(batch, non_last_step_batch)
            batch = DataProto.concat([batch, non_last_step_batch])

        return batch, metrics

    def _stepwise_advantage_broadcast(self, last_step_batch: DataProto, non_last_step_batch: DataProto) -> None:
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

            if self.config.rllm.stepwise_advantage.get("normalize_by_steps", False):
                scalar = scalar / src_steps[i]
                last_step_batch.batch["advantages"][i][mask] = scalar

            traj_ep_to_scalar_adv[(traj_id, eps_id)] = scalar

        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=traj_ep_to_scalar_adv[(traj_id, eps_id)], dtype=torch.float32) for i, (traj_id, eps_id) in enumerate(zip(tgt_traj_ids, tgt_eps_ids, strict=False))])

        final_advantage = scalar_rows * tgt_mask
        non_last_step_batch.batch["advantages"] = final_advantage
        non_last_step_batch.batch["returns"] = final_advantage

    def _remove_padding(self, batch: DataProto) -> DataProto:
        """Remove padded steps from batch."""
        is_pad_step = batch.non_tensor_batch["is_pad_step"]
        non_pad_step_indices = np.where(is_pad_step == False)[0]
        return batch.select_idxs(non_pad_step_indices)

    def update_policy(self, trainer_state: TrainerState) -> None:
        """Update actor and critic policies."""
        global_steps = trainer_state.global_step
        batch = trainer_state.backend_batch

        # Update critic
        if self.use_critic:
            critic_output = self.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            trainer_state.metrics.update(critic_output_metrics)

        # Update actor (after critic warmup)
        if self.config.trainer.get("critic_warmup", 0) <= global_steps:
            actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            trainer_state.metrics.update(actor_output_metrics)

    # =========================================================================
    # Hook methods - leverage RayPPOTrainer utilities where possible
    # =========================================================================

    def on_train_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of training."""
        self.global_steps = trainer_state.global_step
        self._load_checkpoint()
        # we need to set trainer's global_steps to sync with the loaded checkpoint
        trainer_state.global_step = self.global_steps

    def on_batch_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of each batch."""
        self.global_steps = trainer_state.global_step
        # Start profiling if configured
        do_profile = trainer_state.is_training and trainer_state.global_step in self.config.trainer.profile_steps if self.config.trainer.get("profile_steps") is not None else False
        if do_profile:
            self._start_profiling(do_profile)

    def on_batch_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of each batch."""
        # Stop profiling
        do_profile = trainer_state.is_training and trainer_state.global_step in self.config.trainer.profile_steps if self.config.trainer.get("profile_steps") is not None else False
        if do_profile:
            self._stop_profiling(do_profile)

        # Save checkpoint if configured
        if self.config.trainer.save_freq > 0 and trainer_state.global_step % self.config.trainer.save_freq == 0:
            self._save_checkpoint()

    def on_validation_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of validation."""
        trainer_state.is_training = False
        if self.workflow_engine is not None:
            self.workflow_engine.set_training_step(trainer_state.global_step, mode="val", epoch=trainer_state.epoch)

    def on_validation_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of validation."""
        trainer_state.is_training = True
