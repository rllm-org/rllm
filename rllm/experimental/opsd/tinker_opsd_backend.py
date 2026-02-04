"""
A subclass of TinkerBackend that supports on-policy self-distillation (OPSD).
TODO(listar2000): instead of creating a new backend, we should be able to convert OPSD in an advantage computer form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omegaconf import DictConfig
from typing_extensions import override

from rllm.experimental.common import simple_timer
from rllm.experimental.common.advantage import _collect_metrics_from_trajectory_groups
from rllm.experimental.opsd.advantage import calculate_advantage_opsd
from rllm.experimental.opsd.utils import OPSDConfig
from rllm.experimental.protocol import AlgorithmConfig
from rllm.experimental.tinker.tinker_backend import TinkerBackend
from rllm.experimental.tinker.transform import trajectory_to_datums

if TYPE_CHECKING:
    from rllm.experimental.unified_trainer import TrainerState


class TinkerOPSDBackend(TinkerBackend):
    """
    A subclass of TinkerBackend that supports on-policy self-distillation (OPSD).
    """

    def __init__(self, config: DictConfig, **kwargs):
        TinkerBackend.__init__(self, config, **kwargs)

        self.opsd_config = OPSDConfig(
            kl_penalty_coef=config.opsd.get("kl_penalty_coef", 1.0),
            kl_discount_factor=config.opsd.get("kl_discount_factor", 0.0),
            teacher_messages_key=config.opsd.get("teacher_messages_key", "teacher_messages"),
            teacher_policy_update_freq=config.opsd.get("teacher_policy_update_freq", -1),
        )

    @override
    async def on_train_start(self, trainer_state: TrainerState) -> None:
        """Called at the start of training."""
        await super().on_train_start(trainer_state)
        self.teacher_sampling_client = self.sampling_client

    @override
    async def on_batch_end(self, trainer_state: TrainerState) -> None:
        """Called at the end of each batch."""
        await super().on_batch_end(trainer_state)

        update_freq = self.opsd_config.teacher_policy_update_freq
        if update_freq > 0 and trainer_state.global_step % update_freq == 0:
            self.teacher_sampling_client = self.sampling_client

    @override
    async def compute_advantages(self, trainer_state: TrainerState, algorithm_config: AlgorithmConfig, **kwargs) -> None:
        """Compute advantages from trajectory groups.

        For OPSD, we override the advantage computation logic and separate this step from Tinker datum construction.
        """
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"
        assert self.teacher_sampling_client is not None and self.rollout_engine is not None, "Sampling client or rollout engine are not initialized yet"
        assert self.rollout_engine.chat_parser is not None, "Chat parser is not set"

        self._algorithm_config = algorithm_config
        trajectory_groups = trainer_state.trajectory_groups

        with simple_timer("calculate_advantage_self_distill", trainer_state.timing_dict):
            await calculate_advantage_opsd(trajectory_groups, self.teacher_sampling_client, self.rollout_engine.chat_parser, self.opsd_config)
            # only collect advantage metrics as in OPSD there's no external reward
            adv_metrics = _collect_metrics_from_trajectory_groups(trajectory_groups, algorithm_config.stepwise_advantage_mode, collect_rewards=False, collect_advantage=True)

        trainer_state.metrics.update(adv_metrics)

    @override
    async def process_backend_batch(self, trainer_state: TrainerState, **kwargs) -> None:
        """Perform the following operations in order:

        1. Build up the Tinker datums from the trajectory groups
        2. Run forward-backward pass on the training client
        3. Store logprobs for KL metrics computation
        """
        assert trainer_state.trajectory_groups is not None, "Trajectory groups are not set"
        assert self.policy_trainer is not None and self.policy_trainer.training_client is not None, "Training client is not initialized yet"

        with simple_timer("fused_forward_backward_and_optim_step", trainer_state.timing_dict):
            datums = []
            for group in trainer_state.trajectory_groups:
                for trajectory in group.trajectories:
                    datums.extend(trajectory_to_datums(trajectory))

            loss_fn = self._algorithm_config.loss_fn or "importance_sampling"  # type: ignore[assignment]
            fwd_bwd_future = await self.policy_trainer.training_client.forward_backward_async(
                [self.policy_trainer._remove_mask(datum) for datum in datums],
                loss_fn=loss_fn,
            )

            optim_step_future, scheduled_learning_rate = await self.policy_trainer.optim_step_future(
                step=trainer_state.global_step,
                total_steps=trainer_state.total_steps,
                learning_rate=self.learning_rate,
                beta1=self.beta1,
                beta2=self.beta2,
                eps=self.eps,
            )

            fwd_bwd_result = await fwd_bwd_future.result_async()
            await optim_step_future.result_async()

            training_logprobs = []
            for output in fwd_bwd_result.loss_fn_outputs:
                logprobs = output["logprobs"].to_torch()
                training_logprobs.append(logprobs)

        trainer_state.extra_info["training_logprobs"] = training_logprobs
        trainer_state.extra_info["scheduled_learning_rate"] = scheduled_learning_rate
        trainer_state.backend_batch = datums

    @override
    async def update_policy(self, trainer_state: TrainerState, **kwargs) -> None:
        """For OPSD, this has been fused into the process_backend_batch method."""
        pass
