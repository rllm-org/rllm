"""
Verl backend implementation for the UnifiedTrainer.

This backend inherits from both BackendProtocol and RayPPOTrainer to provide
verl-specific implementations while reusing verl's worker group infrastructure.
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
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
from verl.utils import tensordict_utils as tu
from verl.utils.metric import reduce_metrics
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

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

import logging

logger = logging.getLogger(__name__)

_DEFAULT_VERL_LOSS = "vanilla"
_VERL_KNOWN_LOSSES: set[str] | None = None


class CustomPPOLoss:
    """Wraps Verl's ``ppo_loss`` to support per-call loss mode override.

    When the data TensorDict contains ``policy_loss_mode_override``,
    the loss mode is temporarily overridden for that call.  Instances
    are serialised via cloudpickle and sent to remote workers through
    Verl's ``set_loss_fn`` RPC.
    """

    def __init__(self, config):
        # Convert OmegaConf DictConfig → ActorConfig dataclass
        from verl.utils.config import omega_conf_to_dataclass

        self.config = omega_conf_to_dataclass(config)

    def __call__(self, model_output, data, dp_group=None):
        from verl.utils import tensordict_utils as _tu
        from verl.workers.utils.losses import ppo_loss

        override = _tu.get(data, "policy_loss_mode_override", default=None)
        if override is not None:
            original = self.config.policy_loss.get("loss_mode", "vanilla")
            self.config.policy_loss["loss_mode"] = override
            try:
                return ppo_loss(self.config, model_output, data, dp_group)
            finally:
                self.config.policy_loss["loss_mode"] = original
        return ppo_loss(self.config, model_output, data, dp_group)


def _get_verl_known_losses() -> set[str]:
    """Lazily load the set of registered Verl policy loss function names."""
    global _VERL_KNOWN_LOSSES
    if _VERL_KNOWN_LOSSES is None:
        from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY

        _VERL_KNOWN_LOSSES = set(POLICY_LOSS_REGISTRY.keys())
    return _VERL_KNOWN_LOSSES


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
        )

        # Initialize BackendProtocol
        BackendProtocol.__init__(self, config, **kwargs)

        # RayPPOTrainer no longer accepts them, so we need to store manualll
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # Store full config reference (RayPPOTrainer uses self.config)
        self.full_config = config
        self.algorithm_config: AlgorithmConfig | None = None  # to be set in init_rollout_engine

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
        # Apply Verl patches
        from rllm.experimental.verl.patch import patch_verl_dynamic_batch_sync

        patch_verl_dynamic_batch_sync()

        # If SDK is enabled, instrument vLLM replicas before creating workers
        sdk_enabled = self.full_config.rllm.get("sdk", {}).get("enable", False)
        if sdk_enabled:
            self._instrument_vllm_for_sdk()

        # Step 1: call RayPPOTrainer's `init_workers()` function to obtain the async_rollout_manager
        RayPPOTrainer.init_workers(self)

        assert self.async_rollout_manager is not None, "async_rollout_manager is not available. Issues with RayPPOTrainer's `init_workers()` function."

        # Step 2: replace loss function on remote workers to support per-role loss override
        if hasattr(self.actor_rollout_wg, "set_loss_fn"):
            self.actor_rollout_wg.set_loss_fn(CustomPPOLoss(self.config.actor_rollout_ref.actor))
        else:
            logger.warning("RayWorkerGroup.set_loss_fn not available — skipping custom loss injection")

        # Step 3: initialize the rollout engine
        self.rollout_engine = VerlEngine(
            config=self.config,
            rollout_manager=self.async_rollout_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        # Step 4: store the algorithm config
        self.algorithm_config = kwargs.get("algorithm_config")

        return self.rollout_engine

    def _instrument_vllm_for_sdk(self) -> None:
        """Monkey-patch vLLM replicas to add logprob/token-id instrumentation for SDK trace collection."""
        from rllm.experimental.verl.patch import patch_vllm_for_sdk

        patch_vllm_for_sdk()

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
        # Step 2: execute tasks using the agent workflow engine (async)
        episodes = await self._execute_tasks_async(batch, agent_workflow_engine, is_validation=is_validation, **kwargs)
        # Step 3: sleep the replicas to free kv_cache before weight sync (if free_cache_engine is enabled)
        # Only sleep during training — validation doesn't update weights, so there's no wake_up call after it.
        # Sleeping after validation would leave replicas asleep, causing CUDA illegal memory access on the next generation.
        if not is_validation:
            await self.checkpoint_manager.sleep_replicas()
        return episodes

    async def _execute_tasks_async(self, batch: DataProto, agent_workflow_engine: UnifiedWorkflowEngine, **kwargs) -> list[Episode]:
        """A Verl-specific helper function to execute tasks asynchronously."""
        assert self.rollout_engine is not None, "rollout_engine is not initialized."
        tasks = batch.non_tensor_batch["extra_info"].tolist()
        task_ids = batch.non_tensor_batch["task_ids"].tolist()
        episodes = await agent_workflow_engine.execute_tasks(tasks, task_ids, **kwargs)
        # handle data sources in the input dataproto
        if "data_source" in batch.non_tensor_batch:
            data_sources = batch.non_tensor_batch["data_source"].tolist()
            for episode, data_source in zip(episodes, data_sources, strict=True):
                episode.info["data_source"] = data_source
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

    def _get_dp_world_size(self) -> int | None:
        """Compute the LCM of all worker group world sizes for DP splitting."""

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
            return None
        return reduce(math.lcm, world_sizes)

    def _pad_dataproto_to_world_size(self, batch: DataProto) -> DataProto:
        from verl.protocol import pad_dataproto_to_divisor

        world_size = self._get_dp_world_size()
        if world_size is None:
            return batch

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

        Uses the new EngineWorker path: converts DataProto to TensorDict in
        no-padding format, calls workers, converts results back to padded
        DataProto.  The no-padding TensorDict (batch_td) is created once and
        reused across all inference worker calls.
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

        # Set meta_info needed by workers
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        if "multi_modal_inputs" in batch.non_tensor_batch:
            images_seqlens_all = []
            for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                if "image_grid_thw" not in multi_modal_input:
                    continue
                images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
            batch.meta_info["images_seqlens"] = images_seqlens_all

        # Convert to TensorDict + no-padding ONCE — reused for all inference calls.
        # to_tensordict() does NOT mutate the original DataProto.
        # left_right_2_no_padding mutates batch_td in-place.
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)

        # --- Compute old_log_probs ---
        with simple_timer("old_log_probs", timing_dict):
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            log_probs = no_padding_2_padding(tu.get(output, "log_probs"), batch_td)
            entropy = no_padding_2_padding(tu.get(output, "entropy"), batch_td)

            # Entropy metric (for logging only)
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            metrics["actor/entropy"] = entropy_agg.detach().item()

            # Merge old_log_probs back into the padded DataProto
            old_log_prob = DataProto.from_tensordict(tu.get_tensordict({"old_log_probs": log_probs.float()}))
            batch = batch.union(old_log_prob)

            # Compute rollout log prob diff if available
            if "rollout_log_probs" in batch.batch:
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

        # --- Compute reference log_probs (reuse batch_td) ---
        if self.use_reference_policy:
            with simple_timer("ref", timing_dict):
                tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
                if not self.ref_in_actor:
                    ref_output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
                else:
                    tu.assign_non_tensor(batch_td, no_lora_adapter=True)
                    ref_output = self.actor_rollout_wg.compute_log_prob(batch_td)
                ref_lp = no_padding_2_padding(tu.get(ref_output, "log_probs"), batch_td)
                ref_log_prob = DataProto.from_tensordict(tu.get_tensordict({"ref_log_prob": ref_lp.float()}))
                batch = batch.union(ref_log_prob)

        # --- Compute critic values ---
        if self.use_critic:
            with simple_timer("values", timing_dict):
                tu.assign_non_tensor(batch_td, compute_loss=False)
                values_output = self.critic_wg.infer_batch(batch_td)
                values_output = values_output.get()  # blocking await on future
                values_tensor = no_padding_2_padding(tu.get(values_output, "values"), batch_td)
                values = DataProto.from_tensordict(tu.get_tensordict({"values": values_tensor.float()}))
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

        Uses the new EngineWorker path: converts DataProto to TensorDict in
        no-padding format with training metadata, then calls workers.  The new
        workers handle micro-batching internally, so no manual re-padding is
        needed before the update.
        """
        global_steps = trainer_state.global_step
        batch: DataProto = trainer_state.backend_batch  # type: ignore[assignment]

        # Update critic
        if self.use_critic:
            with simple_timer("update_critic", trainer_state.timing_dict):
                critic_td = batch.to_tensordict()
                critic_td = left_right_2_no_padding(critic_td)
                ppo_mbs_critic = self.config.critic.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
                tu.assign_non_tensor(
                    critic_td,
                    global_batch_size=ppo_mbs_critic,
                    mini_batch_size=ppo_mbs_critic,
                    epochs=self.config.critic.ppo_epochs,
                    seed=self.config.critic.data_loader_seed,
                    dataloader_kwargs={"shuffle": self.config.critic.shuffle},
                )
                critic_output = self.critic_wg.train_mini_batch(critic_td)
                critic_output = critic_output.get()
                critic_output_metrics = tu.get(critic_output, "metrics")
                trainer_state.metrics.update(reduce_metrics(critic_output_metrics))

        # Update actor (after critic warmup)
        if self.config.trainer.get("critic_warmup", 0) <= global_steps:
            with simple_timer("update_actor", trainer_state.timing_dict):
                self._update_actor_with_loss_routing(batch, trainer_state)

    def _update_actor_with_loss_routing(self, batch: DataProto, trainer_state: TrainerState) -> None:
        """Update actor with per-loss-group splitting when ``loss_fn_map`` is set.

        Roles that share the same policy loss function are grouped together
        into a single ``update_actor`` call, minimising the number of
        optimiser steps.  Each (sub-)batch is converted to TensorDict +
        no-padding format with training metadata before being sent to the
        worker.
        """
        loss_fn_map = self.algorithm_config.loss_fn_map if self.algorithm_config is not None else {}
        group_roles = batch.non_tensor_batch.get("group_roles") if hasattr(batch, "non_tensor_batch") and batch.non_tensor_batch is not None else None

        # Common training metadata
        rollout_n = self.config.actor_rollout_ref.rollout.n
        actor_cfg = self.config.actor_rollout_ref.actor
        ppo_mbs = actor_cfg.ppo_mini_batch_size * rollout_n

        def _send_actor_update(sub_batch: DataProto, loss_override: str | None = None) -> None:
            """Convert DataProto to TensorDict, inject metadata, send to worker."""
            batch_td = sub_batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            metadata: dict[str, Any] = dict(
                calculate_entropy=(actor_cfg.entropy_coeff != 0.0),
                global_batch_size=ppo_mbs,
                mini_batch_size=ppo_mbs,
                epochs=actor_cfg.ppo_epochs,
                seed=actor_cfg.data_loader_seed,
                dataloader_kwargs={"shuffle": actor_cfg.shuffle},
            )
            if loss_override is not None:
                metadata["policy_loss_mode_override"] = loss_override
            tu.assign_non_tensor(batch_td, **metadata)
            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_metrics = tu.get(actor_output, "metrics")
            trainer_state.metrics.update(reduce_metrics(actor_metrics))

        # Fast path: no per-role loss overrides or no role annotations.
        if not loss_fn_map or group_roles is None:
            _send_actor_update(batch)
            return

        # Resolve each role to a Verl loss name with validation + fallback.
        known = _get_verl_known_losses()
        role_to_loss: dict[str, str] = {}
        for role in set(group_roles.tolist()):
            loss_name = loss_fn_map.get(role, _DEFAULT_VERL_LOSS)
            if loss_name not in known:
                logger.warning(f"Unknown Verl loss '{loss_name}' for role '{role}', falling back to '{_DEFAULT_VERL_LOSS}'")
                loss_name = _DEFAULT_VERL_LOSS
            role_to_loss[role] = loss_name

        # Regroup: collect roles by their loss function.
        loss_to_roles: dict[str, list[str]] = defaultdict(list)
        for role, loss in role_to_loss.items():
            loss_to_roles[loss].append(role)

        if len(loss_to_roles) <= 1:
            # All roles share the same loss — single update.
            _send_actor_update(batch, next(iter(loss_to_roles)))
            return

        # Multiple distinct losses: split batch by loss group, update each.
        for loss_name, roles in loss_to_roles.items():
            role_set = set(roles)
            mask = np.array([r in role_set for r in group_roles])
            indices = np.where(mask)[0]
            sub_batch = batch[indices]
            _send_actor_update(sub_batch, loss_name)

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
        await self.checkpoint_manager.update_weights(self.global_steps)
        # we need to set trainer's global_steps to sync with the loaded checkpoint
        trainer_state.global_step = self.global_steps
        trainer_state.epoch = self.global_steps // len(self.train_dataloader)

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

        # Weight synchronization
        with simple_timer("update_weights", trainer_state.timing_dict):
            await self.checkpoint_manager.update_weights(trainer_state.global_step)

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
            self.rollout_engine.is_validation = True
            return True

    async def on_validation_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of validation."""
        trainer_state.is_training = True
        self.rollout_engine.is_validation = False
