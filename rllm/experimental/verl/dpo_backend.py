"""Verl-owned DPO backend for the experimental unified trainer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import DictConfig, OmegaConf
from verl import DataProto
from verl.utils import tensordict_utils as tu
from verl.utils.metric import reduce_metrics
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from rllm.experimental.common import AlgorithmConfig, simple_timer
from rllm.experimental.common.preference import DPOConfig, build_preference_pairs
from rllm.experimental.verl.dpo_loss import CustomDPOLoss
from rllm.experimental.verl.dpo_transform import transform_preference_pairs_to_dataproto
from rllm.experimental.verl.verl_backend import VerlBackend

if TYPE_CHECKING:
    from rllm.experimental.rollout import RolloutEngine
    from rllm.experimental.unified_trainer import TrainerState


_EMPTY_DPO_BATCH = {"empty_dpo_batch": True}


def _is_empty_dpo_batch(batch: Any) -> bool:
    return isinstance(batch, dict) and batch.get("empty_dpo_batch") is True


def _read_dpo_config(config: DictConfig) -> DPOConfig:
    dpo_section = config.rllm.get("dpo", {})
    if isinstance(dpo_section, DictConfig):
        dpo_kwargs = OmegaConf.to_container(dpo_section, resolve=True)
    elif dpo_section is None:
        dpo_kwargs = {}
    else:
        dpo_kwargs = dict(dpo_section)
    return DPOConfig(**dpo_kwargs)


def _pad_dpo_dataproto_to_divisor(batch: DataProto, divisor: int) -> DataProto:
    if divisor <= 1 or batch.batch["responses"].shape[0] % divisor == 0:
        return batch

    from verl.protocol import pad_dataproto_to_divisor

    original_batch_size = batch.batch["responses"].shape[0]
    batch, pad_size = pad_dataproto_to_divisor(batch, divisor)
    pad_start, pad_end = original_batch_size, original_batch_size + pad_size

    batch.batch["response_mask"][pad_start:pad_end] = 0
    batch.batch["dpo_pair_weights"][pad_start:pad_end] = 0
    batch.non_tensor_batch["is_last_step"][pad_start:pad_end] = False
    batch.non_tensor_batch["is_pad_step"][pad_start:pad_end] = True
    batch.non_tensor_batch["is_valid"][pad_start:pad_end] = False
    return batch


class VerlDPOBackend(VerlBackend):
    """Verl backend variant that owns strict DPO training mechanics."""

    name: str = "verl_dpo"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpo_config = _read_dpo_config(self.config)

    def validate_config(self) -> None:
        super().validate_config()

        if self.config.rllm.rollout.n < 2:
            raise ValueError("VerlDPOBackend requires rllm.rollout.n >= 2 to build preference pairs")
        if not self.use_reference_policy:
            raise ValueError("VerlDPOBackend requires a reference policy; enable actor_rollout_ref.actor.use_kl_loss so Verl initializes ref logprobs")
        if self.config.rllm.get("async_training", {}).get("enable", False):
            raise ValueError("VerlDPOBackend v1 only supports the synchronous on-policy trainer path")
        if self.config.rllm.algorithm.get("use_precomputed_advantage", False):
            raise ValueError("VerlDPOBackend does not use precomputed advantages")
        if self.config.trainer.get("balance_batch", False):
            raise ValueError("VerlDPOBackend requires trainer.balance_batch=false so chosen/rejected rows stay adjacent")
        if self.config.rllm.get("mask_truncated_samples", False):
            raise ValueError("VerlDPOBackend v1 does not support mask_truncated_samples because it can split DPO pairs")

        rollout_correction = self.config.rllm.algorithm.get("rollout_correction", {})
        if rollout_correction.get("tis_mode") is not None or rollout_correction.get("bypass_mode") is not None:
            raise ValueError("VerlDPOBackend does not use PPO rollout-correction settings")

    def init_rollout_engine(self, **kwargs) -> RolloutEngine:
        rollout_engine = super().init_rollout_engine(**kwargs)
        if hasattr(self.actor_rollout_wg, "set_loss_fn"):
            self.actor_rollout_wg.set_loss_fn(CustomDPOLoss(beta=self.dpo_config.beta))
        return rollout_engine

    def transform_to_backend_batch(self, trainer_state: TrainerState, **kwargs) -> DataProto | dict[str, bool]:
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"
        assert self.rollout_engine is not None, "rollout_engine is not initialized."

        preference_pairs, preference_metrics = build_preference_pairs(trainer_state.trajectory_groups, self.dpo_config)
        trainer_state.metrics.update(preference_metrics)
        if not preference_pairs:
            return dict(_EMPTY_DPO_BATCH)

        return transform_preference_pairs_to_dataproto(
            preference_pairs,
            self.rollout_engine,
            self.config.data.max_prompt_length,
            self.config.data.max_response_length,
        )

    async def process_backend_batch(self, trainer_state: TrainerState, **kwargs) -> None:
        batch = trainer_state.backend_batch
        if _is_empty_dpo_batch(batch):
            return

        assert isinstance(batch, DataProto)
        metrics = trainer_state.metrics
        timing_dict = trainer_state.timing_dict

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        if "multi_modal_inputs" in batch.non_tensor_batch:
            images_seqlens_all = []
            for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                if "image_grid_thw" not in multi_modal_input:
                    continue
                images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
            batch.meta_info["images_seqlens"] = images_seqlens_all

        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)

        with simple_timer("policy_log_probs", timing_dict):
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            log_probs = no_padding_2_padding(tu.get(output, "log_probs"), batch_td)
            entropy = no_padding_2_padding(tu.get(output, "entropy"), batch_td)
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode

            from verl.trainer.ppo.core_algos import agg_loss

            entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            metrics["actor/entropy"] = entropy_agg.detach().item()
            policy_log_prob = DataProto.from_tensordict(tu.get_tensordict({"policy_log_probs": log_probs.float()}))
            batch = batch.union(policy_log_prob)

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

        trainer_state.backend_batch = batch

    async def compute_advantages(self, trainer_state: TrainerState, algorithm_config: AlgorithmConfig, **kwargs) -> None:
        """DPO optimizes pairwise margins directly and has no PPO advantages."""
        return None

    async def update_policy(self, trainer_state: TrainerState, **kwargs) -> None:
        batch = trainer_state.backend_batch
        if _is_empty_dpo_batch(batch):
            trainer_state.metrics["dpo/skipped_empty_batch"] = 1
            return

        assert isinstance(batch, DataProto)
        if self.config.trainer.get("critic_warmup", 0) > trainer_state.global_step:
            return

        dp_size = self._get_aggregate_dp_size()
        if dp_size is not None:
            batch = _pad_dpo_dataproto_to_divisor(batch, math.lcm(dp_size, 2))
            trainer_state.backend_batch = batch

        actor_cfg = self.config.actor_rollout_ref.actor
        row_count = int(batch.batch["responses"].shape[0])

        with simple_timer("update_actor", trainer_state.timing_dict):
            batch_td = batch.to_tensordict()
            batch_td = left_right_2_no_padding(batch_td)
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=(actor_cfg.entropy_coeff != 0.0),
                global_batch_size=row_count,
                mini_batch_size=row_count,
                epochs=actor_cfg.ppo_epochs,
                seed=actor_cfg.data_loader_seed,
                dataloader_kwargs={"shuffle": False},
            )
            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_metrics = tu.get(actor_output, "metrics")
            trainer_state.metrics.update(reduce_metrics(actor_metrics))

    async def on_batch_end(self, trainer_state: TrainerState) -> None:
        do_profile = trainer_state.is_training and trainer_state.global_step in self.config.trainer.profile_steps if self.config.trainer.get("profile_steps") is not None else False
        if do_profile:
            from rllm.experimental.verl.utils import stop_profiling

            with simple_timer("stop_profile", trainer_state.timing_dict):
                stop_profiling(self.actor_rollout_wg, self.ref_policy_wg, self.use_reference_policy)

        if self.config.trainer.save_freq > 0 and trainer_state.global_step % self.config.trainer.save_freq == 0:
            from rllm.experimental.verl.utils import save_checkpoint

            with simple_timer("save_checkpoint", trainer_state.timing_dict):
                save_checkpoint(self.config, self.global_steps, self.actor_rollout_wg, train_dataloader=self.train_dataloader)

        with simple_timer("update_weights", trainer_state.timing_dict):
            await self.checkpoint_manager.update_weights(trainer_state.global_step)

        metrics = trainer_state.metrics
        metrics.update({"training/global_step": trainer_state.global_step, "training/epoch": trainer_state.epoch})

        batch = trainer_state.backend_batch
        if _is_empty_dpo_batch(batch):
            return

        assert isinstance(batch, DataProto)
        from verl.trainer.ppo.metric_utils import compute_throughout_metrics, compute_timing_metrics

        metrics.update(compute_timing_metrics(batch=batch, timing_raw=trainer_state.timing_dict))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=trainer_state.timing_dict, n_gpus=n_gpus))
