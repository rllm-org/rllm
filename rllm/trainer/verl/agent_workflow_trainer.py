import asyncio
import math
import threading
import time
import uuid
from collections import Counter, defaultdict
from functools import reduce
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf

from rllm.agents.agent import Episode
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.verl_engine import VerlEngine
from rllm.trainer.common.advantage import compute_advantage_from_trajectory_groups
from rllm.trainer.common.config import AlgorithmConfig, CompactFilteringConfig, RejectionSamplingConfig, TransformConfig
from rllm.trainer.common.rejection_sampling import RejectionSamplingState, apply_rejection_sampling_and_filtering
from rllm.trainer.common.transform import transform_episodes_to_trajectory_groups
from rllm.trainer.verl.verl_data_processor import transform_episodes_to_dataproto
from rllm.utils import EpisodeLogger, marked_timer, visualize_trajectory_last_steps
from rllm.workflows.workflow import TerminationReason, Workflow
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import (
    agg_loss,
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
)
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.metric import reduce_metrics


class AgentWorkflowPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        workflow_class: type[Workflow],
        workflow_args: dict | None = None,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        super().__init__(config=config, tokenizer=tokenizer, processor=processor, role_worker_mapping=role_worker_mapping, resource_pool_manager=resource_pool_manager, ray_worker_group_cls=ray_worker_group_cls, reward_fn=reward_fn, val_reward_fn=val_reward_fn)

        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self._validate_and_setup_configs()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _validate_and_setup_configs(self):
        assert self.config.actor_rollout_ref.hybrid_engine is True, "Only hybrid engine is supported"
        assert self.config.actor_rollout_ref.rollout.mode == "async", "Only async rollout mode is supported"
        assert self.use_rm is False, "Reward models are not supported. Rewards should be assigned using a reward function in the workflow or environment."
        if self.config.rllm.rejection_sample.multiplier != 1:
            assert self.config.rllm.rejection_sample.enable is True, "rejection sampling is disabled, but rejection_sample.multiplier is not 1"

        # TODO: add these configurations to the hydra config
        # compact filtering config (used for filtering out episodes that are not valid)
        self.cf_config = CompactFilteringConfig.from_config(self.config.rllm.compact_filtering)

        # transform config (used for transforming episodes to trajectory groups)
        self.transform_config = TransformConfig(broadcast=self.config.rllm.stepwise_advantage.mode == "broadcast")

        # rejection sampling config (used for rejection sampling)
        rs_mode = "episode" if self.config.rllm.rejection_sample.enable else "none"

        self.rs_config = RejectionSamplingConfig(
            mode=rs_mode,
            min_partial_solve_tasks=self.config.rllm.rejection_sample.min_partial_solve_tasks,
            min_trajs_per_group=self.config.rllm.rejection_sample.min_trajs_per_group,
        )

        # algorithm config (used for rLLM-native advantage computation)
        self.algorithm_config = AlgorithmConfig(
            estimator=self.config.algorithm.adv_estimator,
            stepwise_advantage_mode=self.config.rllm.stepwise_advantage.mode,
            norm_adv_by_std_in_grpo=self.config.rllm.stepwise_advantage.get("norm_adv_by_std_in_grpo", True),
        )

    def init_workers(self):
        super().init_workers()

        rollout_engine = VerlEngine(
            config=self.config,
            rollout_manager=self.async_rollout_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        # Create episode logger if enabled in config
        episode_logger = None
        if self.config.trainer.get("log_episodes", False):
            # Get episode log directory from config, default to "logs/my_project/my_experiment"
            episode_log_dir = self.config.trainer.get("episode_log_dir", f"logs/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}")
            episode_logger = EpisodeLogger(base_dir=episode_log_dir, subdirectory="episodes")

        self.agent_execution_engine = AgentWorkflowEngine(
            workflow_cls=self.workflow_class,
            workflow_args=self.workflow_args,
            rollout_engine=rollout_engine,
            config=self.config,
            n_parallel_tasks=self.config.rllm.workflow.n_parallel_tasks,
            retry_limit=self.config.rllm.workflow.retry_limit,
            raise_on_error=self.config.rllm.workflow.raise_on_error,
            episode_logger=episode_logger,
        )

        # init workflow workers
        asyncio.run_coroutine_threadsafe(self.agent_execution_engine.initialize_pool(), self._loop).result()

    def _update_episode_metrics_and_termination_counts(self, episodes: list[Episode], workflow_metrics: dict, termination_counter: Counter):
        for episode in episodes:
            for key, value in episode.metrics.items():
                workflow_metrics[key].append(value)
            termination_counter[episode.termination_reason] += 1

    def _compute_step_level_values(self, batch: DataProto, timing_raw: dict, metrics: dict) -> DataProto:
        """
        A DataProto-native function that computes old_log_probs, ref_log_probs, and critic values, etc.
        """
        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

            if "rollout_log_probs" in batch.batch.keys():
                # TODO: we may want to add diff of probs too.
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
                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                metrics.update(
                    {
                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                    }
                )

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

        return batch

    def _update_policy(self, batch: DataProto, timing_raw: dict, metrics: dict):
        """
        A DataProto-native function that updates the policy, including actor and critic
        """
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
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)

    def _compute_advantage_verl(self, batch: DataProto, metrics: dict):
        # step_ids is safe to always use for advantage computation
        # if we're not using computing advantages stepwise (i.e., for cumulative agents or single turn workflows)
        # then step_ids == trajectory_ids
        # NOTE: not very safe actually for a generic workflow. Only work for trajectory with a single step.
        # batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]
        batch.non_tensor_batch["uid"] = batch.non_tensor_batch["trajectory_ids"]

        if self.config.rllm.stepwise_advantage.mode == "per_step":
            batch.batch["token_level_scores"] = batch.batch["step_rewards"]
        else:
            batch.batch["token_level_scores"] = batch.batch["traj_rewards"]

        # compute rewards. apply_kl_penalty if available
        if self.config.algorithm.use_kl_in_reward:
            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
            metrics.update(kl_metrics)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        if self.config.rllm.stepwise_advantage.mode == "broadcast":
            is_last_step = batch.non_tensor_batch["is_last_step"]
            last_step_indices = np.where(is_last_step == True)[0]
            not_last_step_indices = np.where(is_last_step == False)[0]
            non_last_step_batch = batch.select_idxs(not_last_step_indices)
            batch = batch.select_idxs(last_step_indices)  # This batch only has last steps
            # last_step_batch contains no padded steps as it was rounded down (not padded) to a multiple of world size
        else:
            batch = self._remove_padding(batch)  # compute advantages over non-padded steps only

        # compute advantages, executed on the driver process
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
            # Merging the separated out steps using the advantage from last steps
            self._stepwise_advantage_broadcast(batch, non_last_step_batch)
            batch = DataProto.concat([batch, non_last_step_batch])

        return batch

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            self.agent_execution_engine.set_training_step(self.global_steps, mode="val", epoch=0)
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1

        rejection_sampling_state = RejectionSamplingState()
        termination_counts = Counter()
        workflow_metrics = defaultdict(list)
        metrics = {}
        timing_raw = {}

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.get("profile_steps") is not None else False
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n)

                new_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=["raw_prompt_ids"])

                # Update training step in engine for episode logging
                self.agent_execution_engine.set_training_step(self.global_steps, mode="train", epoch=epoch)

                with marked_timer("step", timing_raw):
                    # generate a list of episodes (rollouts)
                    final_gen_episodes = self.generate_episodes(batch=new_batch, timing_raw=timing_raw)

                    final_gen_trajectory_groups, trajectory_group_metrics = transform_episodes_to_trajectory_groups(
                        final_gen_episodes,
                        transform_config=self.transform_config,
                        compact_filtering_config=self.cf_config,
                    )

                    # Log metrics and update termination counts
                    self._update_episode_metrics_and_termination_counts(final_gen_episodes, workflow_metrics, termination_counts)

                    # rejection sampling & filtering
                    # we do rejection sampling at the episode level instead of the traj/step level
                    filtered_groups, filtered_episodes, rejection_sampling_metrics = apply_rejection_sampling_and_filtering(
                        final_gen_episodes,
                        groups=final_gen_trajectory_groups,
                        config=self.rs_config,
                        state=rejection_sampling_state,
                    )

                    if len(filtered_groups) == 0:
                        continue

                    pprint(trajectory_group_metrics)
                    pprint(rejection_sampling_metrics)
                    # metrics.update(trajectory_group_metrics)
                    # metrics.update(rejection_sampling_metrics)

                    use_rllm_advantage = self.config.rllm.stepwise_advantage.get("use_rllm_advantage", True)
                    if use_rllm_advantage:
                        with marked_timer("rllm_adv", timing_raw, color="brown"):
                            compute_advantage_from_trajectory_groups(filtered_groups, self.algorithm_config)

                    batch = transform_episodes_to_dataproto(
                        filtered_episodes,
                        rollout_engine=self.agent_execution_engine.rollout_engine,
                        max_prompt_length=self.config.data.max_prompt_length,
                        max_response_length=self.config.data.max_response_length,
                        stepwise_advantage_mode=self.config.rllm.stepwise_advantage.mode,
                    )

                    if not use_rllm_advantage:
                        with marked_timer("verl_adv", timing_raw, color="brown"):
                            batch = self._compute_advantage_verl(batch=batch, metrics=metrics)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        # pad batch size to world size for batch balancing
                        batch = self._pad_dataproto_to_world_size(batch=batch)
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    batch = self._compute_step_level_values(batch=batch, timing_raw=timing_raw, metrics=metrics)

                    # for backward compatibility
                    if self.config.rllm.mask_truncated_samples:
                        mask = batch.batch["attention_mask"][:, -1] == 1
                        batch = batch[~mask]

                    self._update_policy(batch, timing_raw=timing_raw, metrics=metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with marked_timer("testing", timing_raw, color="green"):
                            self.agent_execution_engine.set_training_step(self.global_steps, mode="val", epoch=epoch)
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    # visualize some trajectory last steps
                    visualize_trajectory_last_steps(
                        filtered_groups,
                        tokenizer=self.tokenizer,
                        max_steps_to_visualize=2,
                        show_workflow_metadata=True,
                    )

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                for key, value in workflow_metrics.items():
                    metrics[f"batch/{key}"] = np.mean(value)

                total_counts = max(sum(termination_counts.values()), 1)
                for r in TerminationReason:
                    metrics[f"batch/{r.value}"] = termination_counts[r.value] / total_counts

                logger.log(data=metrics, step=self.global_steps)

                rejection_sampling_state.reset()
                termination_counts = Counter()
                workflow_metrics = defaultdict(list)
                metrics = {}
                timing_raw = {}

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        self.agent_execution_engine.set_training_step(self.global_steps, mode="val", epoch=epoch)
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def _validate_agent(self):
        is_correct_lst = []
        data_source_lst = []
        uid_lst = []
        workflow_metrics_by_source = defaultdict(lambda: defaultdict(list))

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)

            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)

            test_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=["raw_prompt_ids"])  # these are not needed for environment based interaction

            test_gen_episodes = self.generate_episodes(batch=test_batch)
            test_batch = transform_episodes_to_dataproto(
                test_gen_episodes,
                rollout_engine=self.agent_execution_engine.rollout_engine,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
            )

            test_batch.meta_info = {"validate": True}

            is_correct_lst.extend([episode.is_correct for episode in test_gen_episodes])
            uid_lst.extend([episode.id.split(":")[0] for episode in test_gen_episodes])

            data_sources = [episode.info.get("data_source", "unknown") for episode in test_gen_episodes]
            data_source_lst.extend(data_sources)

            # Collect workflow metrics per episode and data source
            for episode, data_source in zip(test_gen_episodes, data_sources, strict=True):
                for key, value in episode.metrics.items():
                    workflow_metrics_by_source[data_source][key].append(float(value))

        metrics = {}
        is_correct_array = np.array(is_correct_lst)
        uid_array = np.array(uid_lst)
        data_source_array = np.array(data_source_lst)

        for data_source in np.unique(data_source_array):
            pass_rates = defaultdict(list)

            data_source_mask = data_source_array == data_source
            is_correct_data_source = is_correct_array[data_source_mask]
            uids_data_source = uid_array[data_source_mask]

            for is_correct, uid in zip(is_correct_data_source, uids_data_source, strict=False):
                pass_rates[uid].append(is_correct)

            metrics[f"val/{data_source}/pass@1"] = np.mean(is_correct_data_source)
            metrics[f"val/{data_source}/pass@{n_val_samples}"] = np.mean([1 if any(pass_rate) else 0 for pass_rate in pass_rates.values()])

            # Add workflow metrics for this data source
            if data_source in workflow_metrics_by_source:
                for key, values in workflow_metrics_by_source[data_source].items():
                    if values:  # Only add if we have values
                        metrics[f"val/{data_source}/{key}"] = np.mean(values)

        return metrics

    def generate_episodes(self, batch, timing_raw=None, **kwargs) -> list[Episode]:
        """
        Generates trajectories asynchronously using the agent execution engine's excute tasks method.
        Post-processing is done in the engine as well.

        Args:
            batch: The input batch for trajectory generation
            timing_raw: Dictionary to store timing information for profiling
            **kwargs: Additional arguments to pass to trajectory_generator

        Returns:
            list[Episode]: List of collected processed episodes
        """
        if timing_raw is None:
            timing_raw = {}

        with marked_timer("generate_episodes", timing_raw, color="red"):
            coro = self.agent_execution_engine.execute_tasks_verl(batch, **kwargs)
            final_gen_episodes = asyncio.run_coroutine_threadsafe(coro, self._loop).result()

        return final_gen_episodes

    def _stepwise_advantage_broadcast(self, last_step_batch, non_last_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps within the same episode and trajectory.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants

        src_traj_ids = last_step_batch.non_tensor_batch["trajectory_ids"]
        src_eps_ids = last_step_batch.non_tensor_batch["episode_ids"]
        src_steps = last_step_batch.non_tensor_batch["step_nums"]
        src_mask = last_step_batch.batch["response_mask"]
        src_advantages = last_step_batch.batch["advantages"]

        tgt_traj_ids = non_last_step_batch.non_tensor_batch["trajectory_ids"]
        tgt_eps_ids = non_last_step_batch.non_tensor_batch["episode_ids"]
        tgt_mask = non_last_step_batch.batch["response_mask"]

        # Build id -> scalar advantage
        traj_ep_to_scalar_adv = {}
        for i, (traj_id, eps_id) in enumerate(zip(src_traj_ids, src_eps_ids, strict=False)):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.rllm.stepwise_advantage.normalize_by_steps:
                # normalize the advantage against number of steps
                scalar = scalar / src_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            traj_ep_to_scalar_adv[(traj_id, eps_id)] = scalar

        # Create new tensor for non_last_step_batch with per-token assignment
        scalar_rows = torch.stack([torch.full_like(tgt_mask[i], fill_value=traj_ep_to_scalar_adv[(traj_id, eps_id)], dtype=torch.float32) for i, (traj_id, eps_id) in enumerate(zip(tgt_traj_ids, tgt_eps_ids, strict=False))])  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        non_last_step_batch.batch["advantages"] = final_advantage
        non_last_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
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

    def _remove_padding(self, batch):
        """Removes padded steps from the batch"""
        is_pad_step = batch.non_tensor_batch["is_pad_step"]
        non_pad_step_indices = np.where(is_pad_step == False)[0]
        batch = batch.select_idxs(non_pad_step_indices)  # This batch only has non_pad steps
        return batch

    def shutdown(self):
        """A cleanup method to gracefully stop the background event loop."""
        if hasattr(self, "agent_execution_engine") and self.agent_execution_engine is not None:
            self.agent_execution_engine.shutdown()
            self.agent_execution_engine = None
        if hasattr(self, "_loop") and self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_thread") and self._thread is not None:
            self._thread.join()
