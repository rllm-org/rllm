"""
Verl backend implementation for the UnifiedTrainer.

This backend inherits from both BackendProtocol and RayPPOTrainer to provide
verl-specific implementations while reusing verl's worker group infrastructure.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import DictConfig
from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.metric import reduce_metrics

from rllm.agents.agent import Episode
from rllm.data import Dataset
from rllm.experimental.common import (
    AlgorithmConfig,
    collect_reward_and_advantage_from_trajectory_groups,
    simple_timer,
)
from rllm.experimental.protocol import BackendProtocol
from rllm.experimental.rollout import RolloutEngine, VerlEngine
from rllm.experimental.verl import compute_advantage_verl, transform_episodes_to_dataproto, update_dataproto_with_advantages

if TYPE_CHECKING:
    from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
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
    def init_rollout_engine(self, **kwargs) -> RolloutEngine:
        """Initialize the VerlEngine rollout engine.

        Note: This should be called after init_workers() to ensure
        async_rollout_manager is available.

        Returns:
            VerlEngine: The initialized rollout engine.
        """
        # If SDK is enabled, instrument vLLM replicas before creating workers
        sdk_enabled = self.full_config.rllm.get("sdk", {}).get("enable", False)
        if sdk_enabled:
            self._instrument_vllm_for_sdk()

        # Step 1: call RayPPOTrainer's `init_workers()` function to obtain the async_rollout_manager
        RayPPOTrainer.init_workers(self)

        assert self.async_rollout_manager is not None, "async_rollout_manager is not available. Issues with RayPPOTrainer's `init_workers()` function."

        # Step 2: initialize the rollout engine
        self.rollout_engine = VerlEngine(
            config=self.config,
            rollout_manager=self.async_rollout_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        return self.rollout_engine

    def _instrument_vllm_for_sdk(self) -> None:
        """Monkey-patch vLLM replicas to add logprob/token-id instrumentation for SDK trace collection.

        This replicates the logic from AgentSdkTrainer.init_workers() to ensure
        that when SDK mode is used through the unified trainer, vLLM servers
        return the token IDs and logprobs needed by the trace pipeline.
        """
        import ray
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServerBase, vLLMReplica

        @ray.remote(num_cpus=1)
        class InstrumentedvLLMHttpServer(vLLMHttpServerBase):
            """vLLM HTTP server with automatic vLLM instrumentation in Ray worker."""

            def __init__(self, *args, **kwargs):
                import importlib.util
                import sys
                from pathlib import Path

                instrumentation_path = Path(__file__).parent.parent.parent / "patches" / "vllm_instrumentation.py"

                spec = importlib.util.spec_from_file_location("rllm_vllm_instrumentation_isolated", str(instrumentation_path))
                vllm_instrumentation = importlib.util.module_from_spec(spec)
                sys.modules["rllm_vllm_instrumentation_isolated"] = vllm_instrumentation
                spec.loader.exec_module(vllm_instrumentation)

                vllm_instrumentation.instrument_vllm(add_response_logprobs=True)
                super().__init__(*args, **kwargs)

        _original_vllm_replica_init = vLLMReplica.__init__

        def patched_vllm_replica_init(self, *args, **kwargs):
            _original_vllm_replica_init(self, *args, **kwargs)
            self.server_class = InstrumentedvLLMHttpServer

        vLLMReplica.__init__ = patched_vllm_replica_init

    def validate_config(self) -> None:
        """Validate verl-specific configuration settings."""
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported for VerlBackend"
        assert self.use_rm is False, "Reward models are not supported. Rewards should be assigned using a reward function in the workflow or environment."
        if self.config.rllm.stepwise_advantage.mode != "broadcast":
            # automatically set the stepwise_advantage_mode to "broadcast", the warning is already shown in AlgorithmConfig.from_config
            self.config.rllm.stepwise_advantage.mode = "broadcast"

    def get_dataloader(self, dataset: Dataset | None, trainer_state: TrainerState) -> Iterable:
        """Get dataloader. Note that for Verl backend, the RayPPOTrainer init already creates the dataloaders."""
        if trainer_state.is_training:
            return self.train_dataloader
        elif self.val_dataloader is not None:
            return self.val_dataloader
        else:
            raise ValueError("No validation dataloader available. Please check the configuration.")

    async def generate_episodes(self, batch: Any, agent_workflow_engine: UnifiedWorkflowEngine, is_validation: bool = False, **kwargs) -> list[Episode]:
        """Generate episodes using the workflow engine.

        For Verl backend, this function handles the following procedures:

        1. Build an "interleaved" batch, where each task is repeated `rollout.n` times.
        2. Extract the tasks and task IDs from the batch.
        3. Execute the tasks using the agent workflow engine.
        4. Return the episodes.

        Args:
            batch: Input batch (dict format from dataloader).
            agent_workflow_engine: The workflow engine to use.
            **kwargs: Additional arguments.

        Returns:
            List of generated episodes.
        """
        # Step 1: build interleaved batch
        if isinstance(batch, dict):
            batch = DataProto.from_single_dict(batch)

        batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        if is_validation:
            repeat_times = self.full_config.rllm.rollout.n_val
        else:
            repeat_times = self.full_config.rllm.rollout.n
        batch = batch.repeat(repeat_times=repeat_times)
        batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=["raw_prompt_ids"])

        # Step 2: execute tasks using the agent workflow engine (async)
        episodes = await agent_workflow_engine.execute_tasks_verl(batch, **kwargs)

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

    def transform_to_backend_batch(self, trainer_state: TrainerState, **kwargs) -> DataProto:
        """Transform rllm-native data structures to verl DataProto format."""
        assert trainer_state.episodes is not None, "Episodes are not set"
        episodes: list[Episode] = trainer_state.episodes
        assert self.rollout_engine is not None, "rollout_engine is not initialized."
        return transform_episodes_to_dataproto(episodes, self.rollout_engine, self.config.data.max_prompt_length, self.config.data.max_response_length)

    def _remove_padding(self, batch: DataProto) -> DataProto:
        """Removes padded steps from the batch"""
        is_pad_step = batch.non_tensor_batch["is_pad_step"]
        non_pad_step_indices = np.where(is_pad_step == False)[0]
        batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
        return batch

    def _pad_dataproto_to_world_size(self, batch: DataProto) -> DataProto:
        import math
        from functools import reduce

        from verl.protocol import pad_dataproto_to_divisor

        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if hasattr(self, "actor_wg") and self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if hasattr(self, "rollout_wg") and self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        batch = self._remove_padding(batch)  # Remove any padded steps from the batch (just in case)
        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        pad_start, pad_end = original_batch_size, original_batch_size + pad_size
        batch.non_tensor_batch["is_last_step"][pad_start:pad_end] = False
        batch.non_tensor_batch["is_pad_step"][pad_start:pad_end] = True
        batch.non_tensor_batch["is_valid"][pad_start:pad_end] = False
        return batch

    async def process_backend_batch(self, trainer_state: TrainerState, **kwargs) -> None:
        """Compute step-level values: old_log_probs, ref_log_probs, critic values.

        Reuses logic from AgentWorkflowPPOTrainer._compute_step_level_values.
        Note: This is async for protocol compatibility but operations are sync (blocking)
        """
        metrics = trainer_state.metrics
        timing_dict = trainer_state.timing_dict
        batch: DataProto = trainer_state.backend_batch  # type: ignore[assignment]

        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        if self.config.trainer.balance_batch:
            # pad batch size to world size for batch balancing
            batch = self._pad_dataproto_to_world_size(batch=batch)
            self._balance_batch(batch, metrics=metrics)

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

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
            with simple_timer("ref", timing_dict):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # Compute critic values if using critic
        if self.use_critic:
            with simple_timer("values", timing_dict):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

        # Mask truncated samples if configured
        if self.config.rllm.get("mask_truncated_samples", False):
            mask = batch.batch["attention_mask"][:, -1] == 1
            batch = batch[~mask]

        trainer_state.backend_batch = batch

    async def compute_advantages(self, trainer_state: TrainerState, algorithm_config: AlgorithmConfig, **kwargs) -> None:
        """Compute advantages from trajectory groups.

        Note: This is async for protocol compatibility but operations are sync.
        """
        assert trainer_state.episodes is not None, "Episodes are not set"
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"
        episodes, trajectory_groups = trainer_state.episodes, trainer_state.trajectory_groups
        batch: DataProto = trainer_state.backend_batch  # type: ignore[assignment]

        with simple_timer("adv", trainer_state.timing_dict):
            if algorithm_config.use_rllm:
                adv_metrics = collect_reward_and_advantage_from_trajectory_groups(trajectory_groups, algorithm_config)
                updated_batch = update_dataproto_with_advantages(batch, episodes, mode=algorithm_config.stepwise_advantage_mode)
            else:
                updated_batch, adv_metrics = compute_advantage_verl(batch, self.config)

        trainer_state.metrics.update(adv_metrics)
        trainer_state.backend_batch = updated_batch

    async def update_policy(self, trainer_state: TrainerState, **kwargs) -> None:
        """Update actor and critic policies.

        Note: This is async for protocol compatibility but operations are sync (blocking)
        """
        global_steps = trainer_state.global_step
        batch = trainer_state.backend_batch

        # Update critic
        if self.use_critic:
            with simple_timer("update_critic", trainer_state.timing_dict):
                critic_output = self.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            trainer_state.metrics.update(critic_output_metrics)

        # Update actor (after critic warmup)
        if self.config.trainer.get("critic_warmup", 0) <= global_steps:
            with simple_timer("update_actor", trainer_state.timing_dict):
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            trainer_state.metrics.update(actor_output_metrics)

    def shutdown(self) -> None:
        """Placeholder, just use the BackendProtocol's default shutdown method."""
        pass

    # =========================================================================
    # Async hook methods - leverage RayPPOTrainer utilities where possible
    # =========================================================================

    async def on_train_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of training."""
        self.global_steps = trainer_state.global_step
        self._load_checkpoint()
        # we need to set trainer's global_steps to sync with the loaded checkpoint
        trainer_state.global_step = self.global_steps

    async def on_batch_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of each batch."""
        self.global_steps = trainer_state.global_step
        # Start profiling if configured
        do_profile = trainer_state.is_training and trainer_state.global_step in self.config.trainer.profile_steps if self.config.trainer.get("profile_steps") is not None else False
        if do_profile:
            with simple_timer("start_profile", trainer_state.timing_dict):
                self._start_profiling(do_profile)

    async def on_batch_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of each batch."""
        # Stop profiling
        do_profile = trainer_state.is_training and trainer_state.global_step in self.config.trainer.profile_steps if self.config.trainer.get("profile_steps") is not None else False
        if do_profile:
            with simple_timer("stop_profile", trainer_state.timing_dict):
                self._stop_profiling(do_profile)

        # Save checkpoint if configured
        if self.config.trainer.save_freq > 0 and trainer_state.global_step % self.config.trainer.save_freq == 0:
            with simple_timer("save_checkpoint", trainer_state.timing_dict):
                self._save_checkpoint()

        # Update metrics
        batch: DataProto = trainer_state.backend_batch  # type: ignore[attr-defined]
        metrics = trainer_state.metrics
        metrics.update({"training/global_step": trainer_state.global_step, "training/epoch": trainer_state.epoch})
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=trainer_state.timing_dict))

        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=trainer_state.timing_dict, n_gpus=n_gpus))

    async def on_validation_start(self, trainer_state: TrainerState) -> bool:
        """Called at the start of validation."""
        if self.val_reward_fn is None:
            return False
        else:
            trainer_state.is_training = False
            self.rollout_engine.validate = True  # type: ignore[attr-defined]
            return True

    async def on_validation_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of validation."""
        trainer_state.is_training = True
        self.rollout_engine.validate = False  # type: ignore[attr-defined]
