"""
Verl backend implementation for the UnifiedTrainer.

Uses verl's lower-level worker infrastructure (RayWorkerGroup, AgentLoopManager,
CheckpointEngineManager) directly, without inheriting from verl's RayPPOTrainer.

Automatically selects colocated vs separated mode based on config:
  - async_training.enable=False (default): Colocated hybrid engine mode.
  - async_training.enable=True: Separated resources, fully async training.
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
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.metric import reduce_metrics
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from rllm.agents.agent import Episode
from rllm.data import Dataset
from rllm.experimental.rollout import RolloutEngine, VerlEngine
from rllm.experimental.common import (
    AlgorithmConfig,
    collect_reward_and_advantage_from_trajectory_groups,
    simple_timer,
)
from rllm.experimental.protocol import BackendProtocol
from rllm.experimental.verl import transform_episodes_to_dataproto, update_dataproto_with_advantages
from rllm.experimental.verl.utils import (
    balance_batch,
    build_wg_kwargs,
    create_dataloaders,
    load_checkpoint,
    save_checkpoint,
    start_profiling,
    stop_profiling,
)

if TYPE_CHECKING:
    from rllm.experimental.engine.unified_workflow_engine import UnifiedWorkflowEngine
    from rllm.experimental.unified_trainer import TrainerState

import logging

logger = logging.getLogger(__name__)

_DEFAULT_VERL_LOSS = "vanilla"


def _get_verl_loss_fn(name: str):
    """Look up a loss function from verl's POLICY_LOSS_REGISTRY by name."""
    from verl.trainer.ppo.core_algos import get_policy_loss
    return get_policy_loss(name)


class VerlBackend(BackendProtocol[Iterable, DataProto]):
    """Verl backend for the unified trainer."""

    name: str = "verl"

    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        **kwargs,
    ):
        BackendProtocol.__init__(self, config, **kwargs)

        self.tokenizer = tokenizer
        self.processor = processor
        self.full_config = config

        # Detect mode from config
        self.is_separated = config.rllm.get("async_training", {}).get("enable", False)

        # Derive boolean flags from config
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        self.use_reference_policy = need_reference_policy(config)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        self.use_prefix_grouper = config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.device_name = config.trainer.get("device", "cuda")

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # Store worker setup params
        self._role_worker_mapping = role_worker_mapping
        self._resource_pool_manager = resource_pool_manager
        self._ray_worker_group_cls = ray_worker_group_cls

        # Worker groups (populated in init_rollout_engine)
        self.actor_rollout_wg = None
        self.ref_policy_wg = None

        # Infrastructure (populated in init_rollout_engine)
        self.async_rollout_manager = None
        self.checkpoint_manager: CheckpointEngineManager | None = None
        self.rollout_engine: VerlEngine | None = None
        self.algorithm_config: AlgorithmConfig | None = None

        # Dataloaders
        self.train_dataloader, self.val_dataloader, self.total_training_steps = create_dataloaders(
            config, tokenizer, processor,
        )

    # =========================================================================
    # Worker initialization
    # =========================================================================

    def _init_colocated_workers(self) -> None:
        """Create worker groups for colocated (hybrid engine) mode."""
        config = self.config
        self._resource_pool_manager.create_resource_pool()
        resource_pool_to_cls = {pool: {} for pool in self._resource_pool_manager.resource_pool_dict.values()}

        # Actor/rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self._role_worker_mapping else Role.ActorRollout
        actor_rollout_resource_pool = self._resource_pool_manager.get_resource_pool(actor_role)
        resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = RayClassWithInitArgs(
            cls=self._role_worker_mapping[actor_role], config=config.actor_rollout_ref, role=str(actor_role),
        )

        # Reference policy
        if self.use_reference_policy and Role.RefPolicy in self._role_worker_mapping:
            resource_pool = self._resource_pool_manager.get_resource_pool(Role.RefPolicy)
            resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = RayClassWithInitArgs(
                self._role_worker_mapping[Role.RefPolicy], config=config.actor_rollout_ref, role=str(Role.RefPolicy),
            )

        # Spawn worker groups
        wg_kwargs = build_wg_kwargs(config, self.device_name)
        all_wg = {}
        for resource_pool, class_dict in resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self._ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

        # Initialize models
        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # Actor/rollout initialized last for better KV cache memory estimation
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # AgentLoopManager (async rollout)
        from verl.experimental.agent_loop import AgentLoopManager
        self.async_rollout_manager = AgentLoopManager.create(
            config=config, worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
        )

        # CheckpointEngineManager
        ckpt_cfg = omega_conf_to_dataclass(config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=ckpt_cfg, trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )
        self.checkpoint_manager.sleep_replicas()

    def _init_separated_workers(self) -> None:
        """Create training-side workers and standalone rollout servers for separated (async) mode.

        Training workers are created via RayWorkerGroup on trainer GPUs.
        Rollout servers are launched by AgentLoopManager in standalone mode
        (worker_group=None), using config.actor_rollout_ref.rollout.nnodes/n_gpus_per_node.
        """
        config = self.config
        wg_kwargs = build_wg_kwargs(config, self.device_name)

        # --- Training-side workers ---
        self._resource_pool_manager.create_resource_pool()
        resource_pool_to_cls = {pool: {} for pool in self._resource_pool_manager.resource_pool_dict.values()}

        actor_role = Role.Actor
        if Role.ActorRollout in self._role_worker_mapping:
            actor_role = Role.ActorRollout
        elif Role.Actor not in self._role_worker_mapping:
            raise ValueError(f"Separated mode requires Role.Actor or Role.ActorRollout, got {self._role_worker_mapping.keys()}")

        resource_pool = self._resource_pool_manager.get_resource_pool(actor_role)
        resource_pool_to_cls[resource_pool][str(actor_role)] = RayClassWithInitArgs(
            cls=self._role_worker_mapping[actor_role], config=config.actor_rollout_ref, role=str(actor_role),
        )

        if self.use_reference_policy and Role.RefPolicy in self._role_worker_mapping:
            ref_pool = self._resource_pool_manager.get_resource_pool(Role.RefPolicy)
            resource_pool_to_cls[ref_pool][str(Role.RefPolicy)] = RayClassWithInitArgs(
                self._role_worker_mapping[Role.RefPolicy], config=config.actor_rollout_ref, role=str(Role.RefPolicy),
            )

        all_wg = {}
        for rp, class_dict in resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self._ray_worker_group_cls(resource_pool=rp, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # --- Rollout servers (standalone, launched by AgentLoopManager) ---
        from verl.experimental.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager
        self.async_rollout_manager = FullyAsyncAgentLoopManager.create(
            config=config, worker_group=None,
        )

        # CheckpointEngineManager: trainer=actor_wg, replicas from standalone rollout servers
        ckpt_cfg = omega_conf_to_dataclass(config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=ckpt_cfg, trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

    # =========================================================================
    # BackendProtocol interface methods
    # =========================================================================

    def init_rollout_engine(self, **kwargs) -> RolloutEngine:
        from rllm.experimental.verl.patch import patch_verl_dynamic_batch_sync
        patch_verl_dynamic_batch_sync()

        sdk_enabled = self.full_config.rllm.get("sdk", {}).get("enable", False)
        if sdk_enabled:
            from rllm.experimental.verl.patch import patch_vllm_for_sdk
            patch_vllm_for_sdk()

        if self.is_separated:
            self._init_separated_workers()
        else:
            self._init_colocated_workers()

        assert self.async_rollout_manager is not None

        # Set default loss function from config (or rllm algorithm config)
        default_loss = self.config.actor_rollout_ref.actor.get("policy_loss", {}).get("loss_mode", _DEFAULT_VERL_LOSS)
        self.actor_rollout_wg.set_loss_fn(_get_verl_loss_fn(default_loss))
        self._default_loss_name = default_loss

        # Create server manager
        servers = zip(self.async_rollout_manager.server_addresses, self.async_rollout_manager.server_handles, strict=True)
        if self.is_separated:
            from verl.experimental.fully_async_policy.agent_loop.agent_loop import FullyAsyncLLMServerManager
            server_manager = FullyAsyncLLMServerManager(self.config, servers=servers, load_balancer_handle=self.async_rollout_manager.global_load_balancer)
        else:
            from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
            server_manager = AsyncLLMServerManager(self.config, servers=servers, load_balancer_handle=self.async_rollout_manager.global_load_balancer)

        # Create VerlEngine
        self.rollout_engine = VerlEngine(
            config=self.config, server_manager=server_manager,
            tokenizer=self.tokenizer, processor=self.processor,
        )

        self.algorithm_config = kwargs.get("algorithm_config")
        return self.rollout_engine

    def validate_config(self) -> None:
        if not self.is_separated:
            assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported"
        if self.use_legacy_worker_impl != "disable":
            logger.warning(f"VerlBackend forces use_legacy_worker_impl='disable', got '{self.use_legacy_worker_impl}'.")
            self.config.trainer.use_legacy_worker_impl = "disable"
        if self.config.rllm.stepwise_advantage.mode != "broadcast":
            self.config.rllm.stepwise_advantage.mode = "broadcast"
        # Validate rollout correction config
        rc = self.config.rllm.algorithm.get("rollout_correction", {})
        if rc.get("bypass_mode", False) and rc.get("tis_mode") is not None:
            raise ValueError("bypass_mode=True and tis_mode!=None is invalid: IS correction is meaningless when π_old = π_rollout")

        # Validate router replay
        if self.config.rllm.algorithm.get("router_replay", False):
            strategy = self.config.actor_rollout_ref.actor.strategy
            if strategy != "megatron":
                raise ValueError(f"router_replay (R3) requires megatron strategy, got '{strategy}'")
        if self.config.actor_rollout_ref.actor.get("router_replay", {}).get("mode", "disabled") == "R2":
            raise ValueError("R2 router replay is not supported. Use rllm.algorithm.router_replay=True for R3.")

    def get_dataloader(self, dataset: Dataset | None, trainer_state: TrainerState) -> Iterable:
        if trainer_state.is_training:
            return self.train_dataloader
        elif self.val_dataloader is not None:
            return self.val_dataloader
        else:
            raise ValueError("No validation dataloader available.")

    async def generate_episodes(self, batch: Any, agent_workflow_engine: UnifiedWorkflowEngine, is_validation: bool = False, **kwargs) -> list[Episode]:
        if isinstance(batch, dict):
            batch = DataProto.from_single_dict(batch)

        batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
        repeat_times = self.full_config.rllm.rollout.n_val if is_validation else self.full_config.rllm.rollout.n
        batch = batch.repeat(repeat_times=repeat_times)

        episodes = await self._execute_tasks_async(batch, agent_workflow_engine, **kwargs)

        # In colocated mode, sleep replicas to free KV cache / rollout weights
        # before the training forward pass (process_backend_batch).
        if not is_validation and not self.is_separated and self.checkpoint_manager is not None:
            await self.checkpoint_manager.sleep_replicas()

        return episodes

    async def _execute_tasks_async(self, batch: DataProto, agent_workflow_engine: UnifiedWorkflowEngine, **kwargs) -> list[Episode]:
        assert self.rollout_engine is not None
        tasks = batch.non_tensor_batch["extra_info"].tolist()
        task_ids = batch.non_tensor_batch["task_ids"].tolist()
        episodes = await agent_workflow_engine.execute_tasks(tasks, task_ids, **kwargs)
        if "data_source" in batch.non_tensor_batch:
            data_sources = batch.non_tensor_batch["data_source"].tolist()
            for episode, data_source in zip(episodes, data_sources, strict=True):
                episode.info["data_source"] = data_source
        return episodes

    def transform_to_backend_batch(self, trainer_state: TrainerState, **kwargs) -> DataProto:
        assert trainer_state.episodes is not None
        assert self.rollout_engine is not None
        return transform_episodes_to_dataproto(
            trainer_state.episodes, self.rollout_engine,
            self.config.data.max_prompt_length, self.config.data.max_response_length,
        )

    def _remove_padding(self, batch: DataProto) -> DataProto:
        is_pad_step = batch.non_tensor_batch["is_pad_step"]
        return batch.select_idxs(np.where(is_pad_step == False)[0])  # noqa: E712

    def _get_dp_world_size(self) -> int | None:
        world_sizes = []
        if self.use_reference_policy and self.ref_policy_wg is not None and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.actor_rollout_wg is not None and self.actor_rollout_wg.world_size != 0:
            world_sizes.append(self.actor_rollout_wg.world_size)
        if not world_sizes:
            return None
        return reduce(math.lcm, world_sizes)

    def _pad_dataproto_to_world_size(self, batch: DataProto) -> DataProto:
        from verl.protocol import pad_dataproto_to_divisor

        world_size = self._get_dp_world_size()
        if world_size is None:
            return batch

        batch = self._remove_padding(batch)
        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        pad_start, pad_end = original_batch_size, original_batch_size + pad_size
        batch.non_tensor_batch["is_last_step"][pad_start:pad_end] = False
        batch.non_tensor_batch["is_pad_step"][pad_start:pad_end] = True
        batch.non_tensor_batch["is_valid"][pad_start:pad_end] = False
        return batch

    async def process_backend_batch(self, trainer_state: TrainerState, **kwargs) -> None:
        """Compute old_log_probs and ref_log_probs."""
        metrics = trainer_state.metrics
        timing_dict = trainer_state.timing_dict
        batch: DataProto = trainer_state.backend_batch

        if self.config.trainer.balance_batch:
            batch = self._pad_dataproto_to_world_size(batch=batch)
            balance_batch(batch, self.actor_rollout_wg, self.config, metrics, use_prefix_grouper=self.use_prefix_grouper)

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        if "multi_modal_inputs" in batch.non_tensor_batch:
            images_seqlens_all = []
            for mmi in batch.non_tensor_batch["multi_modal_inputs"]:
                if "image_grid_thw" not in mmi:
                    continue
                images_seqlens_all.extend(mmi["images_seqlens"].tolist())
            batch.meta_info["images_seqlens"] = images_seqlens_all

        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)

        # old_log_probs + rollout correction
        rc = self.algorithm_config.rollout_correction if self.algorithm_config is not None else None
        bypass_mode = rc is not None and rc.bypass_mode

        if bypass_mode:
            # bypass_mode=True: use rollout logprobs as π_old (no forward pass)
            assert "rollout_log_probs" in batch.batch, "bypass_mode requires rollout_log_probs in batch"
            with simple_timer("old_log_probs", timing_dict):
                batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
        else:
            # bypass_mode=False: compute π_old via forward pass
            with simple_timer("old_log_probs", timing_dict):
                tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
                log_probs = no_padding_2_padding(tu.get(output, "log_probs"), batch_td)
                entropy = no_padding_2_padding(tu.get(output, "entropy"), batch_td)

                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_masks, loss_agg_mode=loss_agg_mode, loss_scale_factor=self.config.actor_rollout_ref.actor.loss_scale_factor)
                metrics["actor/entropy"] = entropy_agg.detach().item()

                batch = batch.union(DataProto.from_tensordict(tu.get_tensordict({"old_log_probs": log_probs.float()})))

            # Compute rollout IS weights if tis_mode is set (3-policy correction)
            tis_mode = rc.tis_mode if rc is not None else None
            if tis_mode is not None and "rollout_log_probs" in batch.batch:
                with simple_timer("rollout_correction", timing_dict):
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_weights

                    log_ratio = batch.batch["old_log_probs"] - batch.batch["rollout_log_probs"]
                    response_length = batch.batch["responses"].size(1)
                    response_mask = batch.batch["attention_mask"][:, -response_length:]

                    rollout_is_weights, is_metrics = compute_rollout_correction_weights(
                        log_ratio=log_ratio,
                        response_mask=response_mask,
                        rollout_is=tis_mode,
                        rollout_is_threshold=rc.tis_cap,
                    )
                    batch.batch["rollout_is_weights"] = rollout_is_weights
                    metrics.update({f"rollout_correction/{k}": v for k, v in is_metrics.items()})

            # Off-policy diagnostics (KL, PPL, chi-squared, etc.)
            if "rollout_log_probs" in batch.batch:
                from verl.trainer.ppo.rollout_corr_helper import compute_offpolicy_metrics

                response_length = batch.batch["responses"].size(1)
                response_mask = batch.batch["attention_mask"][:, -response_length:]
                offpolicy_metrics = compute_offpolicy_metrics(
                    old_log_prob=batch.batch["old_log_probs"],
                    rollout_log_prob=batch.batch["rollout_log_probs"],
                    response_mask=response_mask,
                )
                metrics.update({f"offpolicy/{k}": v for k, v in offpolicy_metrics.items()})

        # ref_log_probs
        if self.use_reference_policy:
            with simple_timer("ref", timing_dict):
                tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
                if not self.ref_in_actor:
                    ref_output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
                else:
                    tu.assign_non_tensor(batch_td, no_lora_adapter=True)
                    ref_output = self.actor_rollout_wg.compute_log_prob(batch_td)
                ref_lp = no_padding_2_padding(tu.get(ref_output, "log_probs"), batch_td)
                batch = batch.union(DataProto.from_tensordict(tu.get_tensordict({"ref_log_prob": ref_lp.float()})))

        if self.config.rllm.algorithm.get("mask_truncated_samples", False):
            mask = batch.batch["attention_mask"][:, -1] == 1
            batch = batch[~mask]

        trainer_state.backend_batch = batch

    async def compute_advantages(self, trainer_state: TrainerState, algorithm_config: AlgorithmConfig, **kwargs) -> None:
        assert trainer_state.trajectory_groups is not None
        batch: DataProto = trainer_state.backend_batch

        with simple_timer("adv", trainer_state.timing_dict):
            adv_metrics = collect_reward_and_advantage_from_trajectory_groups(trainer_state.trajectory_groups, algorithm_config)
            updated_batch = update_dataproto_with_advantages(batch, trainer_state.episodes, mode=algorithm_config.stepwise_advantage_mode)

        trainer_state.metrics.update(adv_metrics)
        trainer_state.backend_batch = updated_batch

    async def update_policy(self, trainer_state: TrainerState, **kwargs) -> None:
        batch: DataProto = trainer_state.backend_batch
        with simple_timer("update_actor", trainer_state.timing_dict):
            self._update_actor_with_loss_routing(batch, trainer_state)

    def _update_actor_with_loss_routing(self, batch: DataProto, trainer_state: TrainerState) -> None:
        loss_fn_map = self.algorithm_config.loss_fn_map if self.algorithm_config is not None else {}
        group_roles = batch.non_tensor_batch.get("group_roles") if hasattr(batch, "non_tensor_batch") and batch.non_tensor_batch is not None else None

        rollout_n = self.config.actor_rollout_ref.rollout.n
        actor_cfg = self.config.actor_rollout_ref.actor
        ppo_mbs = actor_cfg.ppo_mini_batch_size * rollout_n

        def _send_actor_update(sub_batch: DataProto) -> None:
            batch_td = sub_batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            tu.assign_non_tensor(batch_td,
                calculate_entropy=(actor_cfg.entropy_coeff != 0.0),
                global_batch_size=ppo_mbs, mini_batch_size=ppo_mbs,
                epochs=actor_cfg.ppo_epochs, seed=actor_cfg.data_loader_seed,
                dataloader_kwargs={"shuffle": actor_cfg.shuffle},
            )
            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            trainer_state.metrics.update(reduce_metrics(tu.get(actor_output, "metrics")))

        # No per-role loss routing — just use the default loss
        if not loss_fn_map or group_roles is None:
            _send_actor_update(batch)
            return

        # Group roles by loss function
        role_to_loss: dict[str, str] = {}
        for role in set(group_roles.tolist()):
            role_to_loss[role] = loss_fn_map.get(role, self._default_loss_name)

        loss_to_roles: dict[str, list[str]] = defaultdict(list)
        for role, loss in role_to_loss.items():
            loss_to_roles[loss].append(role)

        # Single loss for all roles — no need to split
        if len(loss_to_roles) <= 1:
            loss_name = next(iter(loss_to_roles))
            if loss_name != self._default_loss_name:
                self.actor_rollout_wg.set_loss_fn(_get_verl_loss_fn(loss_name))
            _send_actor_update(batch)
            if loss_name != self._default_loss_name:
                self.actor_rollout_wg.set_loss_fn(_get_verl_loss_fn(self._default_loss_name))
            return

        # Multiple losses — split batch by loss, swap loss_fn for each
        for loss_name, roles in loss_to_roles.items():
            self.actor_rollout_wg.set_loss_fn(_get_verl_loss_fn(loss_name))
            role_set = set(roles)
            mask = np.array([r in role_set for r in group_roles])
            _send_actor_update(batch[np.where(mask)[0]])

        # Restore default
        self.actor_rollout_wg.set_loss_fn(_get_verl_loss_fn(self._default_loss_name))

    def shutdown(self) -> None:
        pass

    # =========================================================================
    # Lifecycle hooks
    # =========================================================================

    async def on_train_start(self, trainer_state: TrainerState) -> None:
        resumed_step = load_checkpoint(
            self.config, self.actor_rollout_wg,
            train_dataloader=self.train_dataloader,
        )
        if resumed_step > 0:
            trainer_state.global_step = resumed_step
            trainer_state.epoch = resumed_step // len(self.train_dataloader)

        await self.checkpoint_manager.update_weights(trainer_state.global_step)


    async def on_train_end(self, trainer_state: TrainerState) -> None:
        if self.config.trainer.save_freq <= 0 or trainer_state.global_step % self.config.trainer.save_freq != 0:
            save_checkpoint(self.config, trainer_state.global_step, self.actor_rollout_wg,
                train_dataloader=self.train_dataloader)

    async def on_batch_start(self, trainer_state: TrainerState) -> None:
        do_profile = (
            trainer_state.is_training
            and self.config.trainer.get("profile_steps") is not None
            and trainer_state.global_step in self.config.trainer.profile_steps
        )
        if do_profile:
            with simple_timer("start_profile", trainer_state.timing_dict):
                start_profiling(trainer_state.global_step, self.actor_rollout_wg,
                    ref_policy_wg=self.ref_policy_wg,
                    use_reference_policy=self.use_reference_policy)

    async def on_batch_end(self, trainer_state: TrainerState) -> None:
        do_profile = (
            trainer_state.is_training
            and self.config.trainer.get("profile_steps") is not None
            and trainer_state.global_step in self.config.trainer.profile_steps
        )
        if do_profile:
            with simple_timer("stop_profile", trainer_state.timing_dict):
                stop_profiling(self.actor_rollout_wg,
                    ref_policy_wg=self.ref_policy_wg,
                    use_reference_policy=self.use_reference_policy)

        if self.config.trainer.save_freq > 0 and trainer_state.global_step % self.config.trainer.save_freq == 0:
            with simple_timer("save_checkpoint", trainer_state.timing_dict):
                save_checkpoint(self.config, trainer_state.global_step, self.actor_rollout_wg,
                    train_dataloader=self.train_dataloader)

        # Weight synchronization (colocated only — separated syncs in on_policy_updated)
        if not self.is_separated:
            with simple_timer("update_weights", trainer_state.timing_dict):
                await self.checkpoint_manager.update_weights(trainer_state.global_step)

        batch: DataProto = trainer_state.backend_batch
        metrics = trainer_state.metrics
        metrics.update({"training/global_step": trainer_state.global_step, "training/epoch": trainer_state.epoch})
        metrics.update(compute_data_metrics(batch=batch, use_critic=False))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=trainer_state.timing_dict))
        n_gpus = self._resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=trainer_state.timing_dict, n_gpus=n_gpus))

    async def on_policy_updated(self, trainer_state: TrainerState) -> None:
        """Weight sync hook for async mode (called by _perform_weight_sync)."""
        if self.is_separated and self.checkpoint_manager is not None:
            with simple_timer("weight_sync", trainer_state.timing_dict):
                await self.checkpoint_manager.update_weights(trainer_state.weight_version)

    async def on_validation_start(self, trainer_state: TrainerState) -> bool:
        trainer_state.is_training = False
        if self.rollout_engine is not None:
            self.rollout_engine.is_validation = True
        return True

    async def on_validation_end(self, trainer_state: TrainerState) -> None:
        trainer_state.is_training = True
        if self.rollout_engine is not None:
            self.rollout_engine.is_validation = False
