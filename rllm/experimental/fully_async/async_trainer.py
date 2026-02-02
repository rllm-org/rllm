import time
from typing import Any

import ray

from rllm.experimental.fully_async.fully_async_trainer import FullyAsyncTrainer
from rllm.experimental.fully_async.utils import assemble_batch_from_trajectory_group_ls, compute_grpo_outcome_advantage
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_response_mask
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics


@ray.remote(num_cpus=10)
class AsyncTrainer(FullyAsyncTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize stale tracking counters
        self.stale_samples_processed = 0
        self.stale_trajectory_processed = 0

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

            # Collect all fully_async and timing metrics from batch.meta_info
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
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
        # with marked_timer("reward", timing_raw, color="yellow"):
        #     # compute reward model score
        #     if self.use_rm:
        #         reward_tensor = self.rm_wg.compute_rm_score(batch)
        #         batch = batch.union(reward_tensor)

        #     if self.config.reward_model.launch_reward_fn_async:
        #         future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
        #     else:
        #         reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

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
            # we combine with rule-based rm
            # reward_extra_infos_dict: dict[str, list]
            # if self.config.reward_model.launch_reward_fn_async:
            #     reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            # batch.batch["token_level_scores"] = reward_tensor

            # if reward_extra_infos_dict:
            #     batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

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

        # Pad batch to world_size for distributed training
        from verl.protocol import pad_dataproto_to_divisor

        actor_world_size = self.actor_rollout_wg.world_size
        original_batch_size = len(batch)
        batch, pad_size = pad_dataproto_to_divisor(batch, actor_world_size)
        batch.meta_info["pad_size"] = pad_size

        # Zero out masks for padded samples so they don't contribute to loss
        if pad_size > 0:
            batch.batch["response_mask"][original_batch_size:] = 0
            batch.batch["attention_mask"][original_batch_size:] = 0

        # update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                critic_output = self.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_output_metrics)

        # implement critic warmup
        if self.config.trainer.critic_warmup <= self.global_steps:
            # update actor
            with marked_timer("update_actor", timing_raw, color="red"):
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)
        return batch, {}
