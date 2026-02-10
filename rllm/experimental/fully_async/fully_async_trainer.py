# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from datetime import datetime
from pprint import pprint
from typing import Any

import ray
from omegaconf import OmegaConf
from tqdm import tqdm

from rllm.experimental.fully_async.message_queue import MessageQueueClient
from rllm.experimental.fully_async.metric_utils import MetricsAggregator, ValidateMetrics
from rllm.experimental.fully_async.utils import (
    assemble_batch_from_trajectory_group_ls,
    compute_grpo_outcome_advantage,
    reduce_metrics_with_flatten,
)
from verl import DataProto
from verl.experimental.fully_async_policy.ray_trainer import FullyAsyncRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, apply_kl_penalty, compute_response_mask
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer


@ray.remote(num_cpus=10)
class FullyAsyncTrainer(FullyAsyncRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_rm = False
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = lora_rank > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        # ==================== fully async config ====================

        self.message_queue_client = None
        self.param_synchronizer = None

        # Statistics
        # we start from step 1
        self.global_steps = 1
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_samples_processed = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0
        self.train_role = Role.Actor  # Trainer only handles Actor role; validation is done by RolloutExecutor

        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.async_training.required_samples
        self.compute_prox_log_prob = self.config.async_training.compute_prox_log_prob
        total_gpus = config.trainer.nnodes * config.trainer.n_gpus_per_node + config.rollout.nnodes * config.rollout.n_gpus_per_node
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    def set_parameter_synchronizer(self, param_synchronizer):
        """Set parameter synchronizer"""
        self.param_synchronizer = param_synchronizer

    def set_total_train_steps(self, total_train_steps):
        self.total_train_steps = total_train_steps
        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    def _create_actor_rollout_classes(self):
        # create actor
        for role in [self.train_role]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.
        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        # self._init_async_objects()
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        # Note: Validation is handled by RolloutExecutor, not trainer

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.param_synchronizer is None:
            raise ValueError("param_synchronizer client not set. Call set_parameter_synchronizer() first.")

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.max_steps_duration = 0

        # get validate data before training
        self._log_validation_data()

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            metrics = {}
            timing_raw = {}

            with marked_timer("step", timing_raw):
                with marked_timer("gen", timing_raw, color="red"):
                    epoch, batch = self._get_samples_from_queue()
                    if batch is None:
                        break
                    self._collect_metrics_from_samples(batch, metrics)
                batch, reward_extra_infos_dict = self._process_batch_common(batch, metrics, timing_raw, self.local_trigger_step if self.compute_prox_log_prob else None)
                self._log_rollout(batch, reward_extra_infos_dict, timing_raw)

            self._collect_metrics(batch, 0, metrics, timing_raw)
            self.metrics_aggregator.add_step_metrics(metrics=metrics, sample_count=self.required_samples, timestamp=time.time())
            # Trigger parameter synchronization after training step
            time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[FullyAsyncTrainer] global_steps: {self.global_steps} local_trigger_step: {self.local_trigger_step} trigger_parameter_sync_step: {self.trigger_parameter_sync_step} {time_str}")
            await self._trigger_parameter_sync_after_step(global_steps=self.global_steps)
            # Note: _log_validation_data() is called inside _trigger_parameter_sync_after_step
            # after wait_last_valid() to ensure we log the completed validation results
            self._check_save_checkpoint(timing_raw)
            self.global_steps += 1

        # final parameter sync and validate
        # 1. waiting remaining validate task
        ray.get(self.param_synchronizer.wait_last_valid.remote())
        self._log_validation_data()
        # 2. perform addtional parameter_sync and validate if trainer already updated
        if self.current_param_version % self.config.rollout.test_freq != 0 or self.local_trigger_step > 1:
            await self._trigger_parameter_sync_after_step(validate=True, global_steps=self.global_steps)
            ray.get(self.param_synchronizer.wait_last_valid.remote())
            self._log_validation_data()
        self.progress_bar.close()

        self._check_save_checkpoint(timing_raw)

    def _check_save_checkpoint(self, timing_raw):
        if self.current_param_version == self.last_ckpt_version:
            return
        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        # Check if the conditions for saving a checkpoint are met.
        # The conditions include a mandatory condition (1) and
        # one of the following optional conditions (2/3/4):
        # 1. The save frequency is set to a positive value.
        # 2. The current step number is a multiple of the save frequency.
        # 3. The ESI(Elastic Server Instance)/training plan is close to expiration.
        if self.config.trainer.save_freq > 0 and (self.current_param_version % self.config.trainer.save_freq == 0 or esi_close_to_expiration):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version

    def _save_checkpoint(self):
        # Warning: Currently, to align the training process and metrics of colocate,
        # we use current_param_version instead of global step.
        # This can be logically aligned with the original self.global_steps of colocate
        # and is used for metrics and ckpt. which means that the parameter synchronization
        # from trainer to rollouter will increase by 1 each time.

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}")

        print(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", str(Role.Critic))
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        ray.get(self.param_synchronizer.rollout_executor_save_checkpoint.remote(local_global_step_folder))
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("[FullyAsyncTrainer] Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        print(f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, current_param_version to {self.current_param_version}")
        print(f"[FullyAsyncTrainer] Resuming from  {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        return self.current_param_version

    async def _trigger_parameter_sync_after_step(self, validate: bool = False, global_steps: int = None):
        """
        Trigger parameter synchronization after training step
        This ensures rollouter always uses the latest trained parameters
        """
        if self.local_trigger_step < self.trigger_parameter_sync_step and not validate:
            self.local_trigger_step += 1
            return

        self.current_param_version += 1
        self.local_trigger_step = 1
        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.current_param_version,
        )
        self.progress_bar.update(1)
        self.metrics_aggregator.reset()
        timing_param_sync = {}
        with marked_timer("timing_s/wait_last_valid", timing_param_sync):
            ray.get(self.param_synchronizer.wait_last_valid.remote())
        # Log validation data from the PREVIOUS sync (now completed)
        self._log_validation_data()
        with marked_timer("timing_s/param_sync", timing_param_sync):
            rollout_executor_timing = ray.get(
                self.param_synchronizer.sync_weights.remote(
                    self.current_param_version,
                    validate=validate,
                    global_steps=global_steps,
                )
            )
        # Validation is now handled by RolloutExecutor, triggered by ParameterSynchronizer
        # Merge rollout executor timing metrics (rollouter/active_time, rollouter/version_time, rollouter/idle_ratio)
        if rollout_executor_timing:
            timing_param_sync.update(rollout_executor_timing)
        self.logger.log(data=timing_param_sync, step=self.current_param_version)

    def _log_validation_data(self):
        """
        Log validation data from RolloutExecutor via MessageQueue.
        Uses current_param_version for step to ensure monotonically increasing steps.
        """
        val_data = self.message_queue_client.get_validate_sync()
        if not val_data:
            return

        val_metrics: ValidateMetrics = ray.cloudpickle.loads(val_data)
        # Use current_param_version for logging step (must be monotonically increasing)
        # Include the actual validated version as a metric for reference
        log_step = self.current_param_version
        if val_metrics.metrics:
            metrics_to_log = val_metrics.metrics.copy()
            metrics_to_log["val/validated_version"] = val_metrics.param_version
            self.logger.log(data=metrics_to_log, step=log_step)
            pprint(f"[FullyAsyncTrainer] Logged val results (validated version {val_metrics.param_version}) at step {log_step}: {val_metrics.metrics}")
        if val_metrics.timing_raw:
            self.logger.log(data=val_metrics.timing_raw, step=log_step)

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples including staleness tracking.

        This tracks:
        - Stale samples: samples generated with a param version older than current
        - Stale trajectories: trajectories that span param version changes
        - All fully_async and timing metrics from batch.meta_info
        """
        if hasattr(batch, "meta_info") and batch.meta_info:
            # Track stale samples (generated with older param version)
            if "rollout_param_versions" in batch.meta_info:
                samples_param_versions = batch.meta_info["rollout_param_versions"]
                stale_count = sum(1 for v in samples_param_versions if self.current_param_version - v >= 1)
                self.stale_samples_processed += stale_count

            # Track stale trajectories
            if "trajectory_param_versions" in batch.meta_info:
                trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
                stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
                self.stale_trajectory_processed += stale_traj_count

            # Add stale tracking metrics
            metrics.update(
                {
                    "fully_async/count/stale_samples_processed": self.stale_samples_processed,
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )

            # Collect all fully_async, timing, custom, and rejection_sample metrics from batch.meta_info
            for key, value in batch.meta_info.items():
                if key.startswith(("fully_async", "timing_s", "custom/", "rejection_sample")):
                    metrics[key] = value

    def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """
        Get samples from message queue and compose gen_batch_output
        Uses a loop to continuously collect samples until enough are gathered

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """
        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
            flush=True,
        )

        # Collect samples using a simple loop calling get_sample
        consumer_start = time.time()
        queue_samples = []
        queue_len = 0
        while len(queue_samples) < self.required_samples:
            # Get a single sample and wait until there is a sample or None is received
            sample, queue_len = self.message_queue_client.get_sample_sync()

            if sample is None:
                print(f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. Collected {len(queue_samples)}/{self.required_samples} samples")
                break

            queue_samples.append(sample)

            if len(queue_samples) % 64 == 0:
                print(f"[FullyAsyncTrainer] Collected {len(queue_samples)}/{self.required_samples} samples. mq_len: {queue_len}")

        consumer_end = time.time()

        if not queue_samples or len(queue_samples) < self.required_samples:
            print("[FullyAsyncTrainer] not enough samples collected after loop")
            return None, None
        total_wait_time = consumer_end - consumer_start

        print(f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} samples, total wait time: {total_wait_time:.2f} seconds.mq_len: {queue_len}")

        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        # Assemble batch - now working directly with TrajectoryGroup objects
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_trajectory_group_ls(queue_samples, self.config, self.tokenizer, self._balance_batch)
        else:
            batch = assemble_batch_from_trajectory_group_ls(queue_samples, self.config, self.tokenizer, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        return 0, batch

    def compute_grpo_advantage(
        self,
        data: DataProto,
        norm_adv_by_std_in_grpo: bool = True,
    ) -> DataProto:
        """Compute advantage estimates for policy optimization.

        This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
        The advantage estimates are used to guide policy optimization in RL algorithms.

        Args:
            data (DataProto): The data containing batched model outputs and inputs.
            norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
                GRPO. Defaults to True.

        Returns:
            DataProto: The updated data with computed advantages and returns.
        """
        # Back-compatible with trainers that do not compute response mask in fit
        if "response_mask" not in data.batch.keys():
            data.batch["response_mask"] = compute_response_mask(data)
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            traj_uuids=data.non_tensor_batch["trajectory_uuids"],
            index=data.non_tensor_batch["uids"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    def _process_batch_common(self, batch, metrics, timing_raw, local_trigger_step=None):
        with marked_timer("old_log_prob", timing_raw, color="blue"):

            def compute_old_log_prob(batch):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                actor_config = self.config.actor_rollout_ref.actor
                entropy_agg = agg_loss(
                    loss_mat=entropys,
                    loss_mask=response_masks,
                    loss_agg_mode=actor_config.loss_agg_mode,
                    loss_scale_factor=actor_config.loss_scale_factor,
                )
                old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)
                if "rollout_log_probs" in batch.batch.keys():
                    # TODO: we may want to add diff of probs too.
                    from verl.utils.debug.metrics import calculate_debug_metrics

                    metrics.update(calculate_debug_metrics(batch))
                return batch

            async_training = self.config.get("async_training", None)
            if async_training and async_training.use_rollout_log_probs:
                # If local_triger_step == 1, load the training engine's parameters to the CPU
                #  and save a copy for subsequent MIS use.
                # If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
                # then restore the parameters of the current version.
                if local_trigger_step == 1:
                    self.actor_rollout_wg.save_model_to_cpu(1)
                    batch = compute_old_log_prob(batch)
                elif local_trigger_step is not None:
                    self.actor_rollout_wg.save_model_to_cpu(local_trigger_step)
                    self.actor_rollout_wg.restore_model_from_cpu(1)
                    batch = compute_old_log_prob(batch)
                    self.actor_rollout_wg.restore_model_from_cpu(local_trigger_step)
                    self.actor_rollout_wg.clear_cpu_model(local_trigger_step)
                else:
                    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
                    batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            else:
                batch = compute_old_log_prob(batch)

            # add this to ensure to eliminate boundary value stem from packing
            batch.batch["old_log_probs"] = batch.batch["old_log_probs"] * batch.batch["response_mask"]

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # compute values
        if self.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

        with marked_timer("adv", timing_raw, color="brown"):
            # compute rewards. apply_kl_penalty if available
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            # Compute rollout correction weights centrally (once per batch)
            # This corrects for off-policy issues (policy mismatch, model staleness, etc.)
            # Also computes off-policy diagnostic metrics (KL, PPL, etc.)
            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

            rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
            if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                # IS and off-policy metrics already have rollout_corr/ prefix
                metrics.update(is_metrics)

            # compute advantages, executed on the driver process
            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

            # TODO: fixed here to calculate advantage correctly because of var len problem.
            # batch = compute_advantage(
            #     batch,
            #     adv_estimator=self.config.algorithm.adv_estimator,
            #     gamma=self.config.algorithm.gamma,
            #     lam=self.config.algorithm.lam,
            #     num_repeat=self.config.actor_rollout_ref.rollout.n,
            #     norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            #     config=self.config.algorithm,
            # )
            batch = self.compute_grpo_advantage(batch, norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo)

        # Some of the rows in batch are padded to be multiple of actor
        # world size, we need to set the attention mask and response mask
        # to zero here so they don't contribute to loss.
        # Note: batch may be shuffled, so we use ignore_in_loss to identify padded samples
        ignore_in_loss = batch.non_tensor_batch["ignore_in_loss"]
        for i, should_ignore in enumerate(ignore_in_loss):
            if should_ignore:
                batch.batch["response_mask"][i] = 0
                batch.batch["attention_mask"][i] = 0

        # update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                critic_output = self.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics_with_flatten(critic_output.meta_info["metrics"])
            metrics.update(critic_output_metrics)

        # implement critic warmup
        if self.config.trainer.critic_warmup <= self.global_steps:
            # update actor
            with marked_timer("update_actor", timing_raw, color="red"):
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(batch)

            actor_output_metrics = reduce_metrics_with_flatten(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)
        return batch, {}