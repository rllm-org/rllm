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
from rllm.experimental.verl_advantage import compute_advantage_verl
from rllm.trainer.common.advantage import AlgorithmConfig, compute_advantage_from_trajectory_groups
from rllm.trainer.verl.verl_data_processor.transform import transform_trajectory_groups_to_dataproto, update_dataproto_with_advantages
from rllm.utils import simple_timer
from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.metric import reduce_metrics

if TYPE_CHECKING:
    from rllm.engine.unified_workflow_engine import UnifiedWorkflowEngine
    from rllm.experimental.unified_trainer import TrainerState


class VerlBackend(BackendProtocol[Iterable, DataProto], RayPPOTrainer):
    """
    Verl backend for the unified trainer.

    Inherits from both BackendProtocol and RayPPOTrainer to:
        - Provide the BackendProtocol interface for UnifiedTrainer
        - Reuse RayPPOTrainer's worker group infrastructure and utilities (e.g. work group creation, checkpointing)
    """

    name: str = "verl"

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

    def get_dataloader(self, dataset: Dataset | None, trainer_state: TrainerState) -> Iterable:
        """Get dataloader. Note that for Verl backend, the RayPPOTrainer init already creates the dataloaders."""
        if trainer_state.is_training:
            return self.train_dataloader
        elif self.val_dataloader is not None:
            return self.val_dataloader
        else:
            raise ValueError("No validation dataloader available. Please check the configuration.")

    def generate_episodes(self, batch: Any, agent_workflow_engine: UnifiedWorkflowEngine, **kwargs) -> list[Episode]:
        """Generate episodes using the workflow engine.

        For Verl backend, this function handles the following procedures:

        1. Build an "interleaved" batch, where each task is repeated `rollout.n` times.
        2. Extract the tasks and task IDs from the batch.
        3. Execute the tasks using the agent workflow engine.
        4. Return the episodes.

        Args:
            batch: Input batch (dict format from dataloader).
            **kwargs: Additional arguments including 'loop' for async execution.

        Returns:
            List of generated episodes.
        """
        assert "loop" in kwargs and kwargs["loop"] is not None, "async event loop is required"
        loop = kwargs["loop"]

        # Step 1: build interleaved batch
        if isinstance(batch, dict):
            batch = DataProto.from_single_dict(batch)

        batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n)
        batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=["raw_prompt_ids"])

        # Step 2: execute tasks using the agent workflow engine

        coro = agent_workflow_engine.execute_tasks_verl(batch, **kwargs)

        if loop is not None:
            episodes = asyncio.run_coroutine_threadsafe(coro, loop).result()
        else:
            episodes = asyncio.run(coro)

        return episodes

    async def _execute_tasks_async(self, batch: DataProto, agent_workflow_engine: UnifiedWorkflowEngine, **kwargs) -> list[Episode]:
        """A Verl-specific helper function to execute tasks asynchronously."""
        assert self.rollout_engine is not None, "rollout_engine is not initialized."
        await self.rollout_engine.wake_up()
        tasks = batch.non_tensor_batch["extra_info"].tolist()
        task_ids = batch.non_tensor_batch["task_ids"].tolist()
        episodes = await agent_workflow_engine.execute_tasks(tasks, task_ids, **kwargs)
        await self.rollout_engine.sleep()
        return episodes

    def transform_trajectory_groups_to_backend_batch(self, trajectory_groups: list[TrajectoryGroup], **kwargs) -> DataProto:
        """Transform trajectory groups to verl DataProto format."""
        assert self.rollout_engine is not None, "rollout_engine is not initialized."
        return transform_trajectory_groups_to_dataproto(trajectory_groups, self.rollout_engine, self.config.data.max_prompt_length, self.config.data.max_response_length)

    def process_backend_batch(self, trainer_state: TrainerState, **kwargs) -> None:
        """Compute step-level values: old_log_probs, ref_log_probs, critic values.

        Reuses logic from AgentWorkflowPPOTrainer._compute_step_level_values.
        """
        metrics = trainer_state.metrics
        timing_dict = trainer_state.timing_dict
        batch: DataProto = trainer_state.backend_batch  # type: ignore[assignment]

        with simple_timer("old_log_probs", timing_dict):
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

                rollout_probs_diff_metrics = {
                    "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
                    "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
                    "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
                }
                metrics.update(rollout_probs_diff_metrics)

        # Compute reference log_probs if using reference policy
        if self.use_reference_policy:
            with simple_timer("ref_log_probs", timing_dict):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # Compute critic values if using critic
        if self.use_critic:
            with simple_timer("critic_values", timing_dict):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

        # Mask truncated samples if configured
        if self.config.rllm.get("mask_truncated_samples", False):
            mask = batch.batch["attention_mask"][:, -1] == 1
            batch = batch[~mask]  # type: ignore[assignment]

        trainer_state.backend_batch = batch

    def compute_advantages(self, trainer_state: TrainerState, algorithm_config: AlgorithmConfig, **kwargs) -> None:
        """Compute advantages from trajectory groups."""
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"
        trajectory_groups: list[TrajectoryGroup] = trainer_state.trajectory_groups
        batch: DataProto = trainer_state.backend_batch  # type: ignore[assignment]

        use_rllm = algorithm_config.use_rllm
        if use_rllm:
            compute_advantage_from_trajectory_groups(trajectory_groups, algorithm_config)
            updated_batch = update_dataproto_with_advantages(batch, trajectory_groups, mode=self.config.rllm.stepwise_advantage.mode)
        else:
            updated_batch, adv_metrics = compute_advantage_verl(batch, self.config)  # type: ignore[return-value]
            trainer_state.metrics.update(adv_metrics)

        trainer_state.backend_batch = updated_batch

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

    def on_validation_start(self, trainer_state: TrainerState) -> bool:
        """Called at the start of validation."""
        if self.val_reward_fn is None:
            return False
        else:
            trainer_state.is_training = False
            self.rollout_engine.validate = True  # type: ignore[attr-defined]
            return True

    def on_validation_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of validation."""
        trainer_state.is_training = True
        self.rollout_engine.validate = False  # type: ignore[attr-defined]
