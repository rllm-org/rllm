import asyncio
import os
import time
import uuid
from collections import Counter, defaultdict
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.fireworks_engine import FireworksEngine
from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer
from rllm.workflows.workflow import TerminationReason
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import (
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    WorkerType,
    agg_loss,
    apply_kl_penalty,
    compute_advantage,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    marked_timer,
    reduce_metrics,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.tracking import Tracking


class FireworksAgentWorkflowPPOTrainer(AgentWorkflowPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        workflow_class=None,
        workflow_args=None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            workflow_class=workflow_class,
            workflow_args=workflow_args,
        )
        self.hybrid_engine = False

    def init_workers(self):
        """Init resource pool and worker group"""
        assert not self.hybrid_engine, "Pipeline trainer does not support hybrid engine, assumes Rollout and Actor are not in the different worker group"
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        assert Role.Actor in self.role_worker_mapping and Role.Rollout in self.role_worker_mapping, "Actor and Rollout must be in role_worker_mapping"
        actor_resource_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        # actor_gpu_ids = actor_resource_pool.gpu_assignments if isinstance(actor_resource_pool, RayResourcePool) else None

        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Actor],
            config=self.config.actor_rollout_ref,
            role="actor",
            reward_config=self.config.reward_model,
        )
        self.resource_pool_to_cls[actor_resource_pool]["actor"] = actor_cls

        # Get rollout resource pool
        rollout_resource_pool = self.resource_pool_manager.get_resource_pool(Role.Rollout)
        # rollout_gpu_ids = rollout_resource_pool.gpu_assignments if isinstance(rollout_resource_pool, RayResourcePool) else None
        rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.Rollout],
            config=self.config.actor_rollout_ref,
            role="rollout",
            reward_config=self.config.reward_model,
        )
        self.resource_pool_to_cls[rollout_resource_pool]["rollout"] = rollout_cls

        self.actor_wg = RayWorkerGroup(resource_pool=actor_resource_pool, ray_cls_with_init=actor_cls)

        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # for compatibility

        fireworks_engine = FireworksEngine(
            tokenizer=self.tokenizer,
            deployment_id=self.config.fireworks.deployment_id,
            sampling_params={
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": self.config.data.max_prompt_length + self.config.data.max_response_length,
            },
        )
        self.fireworks_engine = fireworks_engine
        self.agent_execution_engine = AgentWorkflowEngine(
            workflow_cls=self.workflow_class,
            workflow_args=self.workflow_args,
            rollout_engine=fireworks_engine,
            config=self.config,
            n_parallel_tasks=self.config.rllm.workflow.n_parallel_tasks,
            retry_limit=self.config.rllm.workflow.retry_limit,
        )

        # init workflow workers
        asyncio.run_coroutine_threadsafe(self.agent_execution_engine.initialize_pool(), self._loop).result()

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """

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
            val_metrics = self._validate_agent()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")
        # we start from step 1
        self.global_steps += 1

        batch = None
        solve_none = 0
        solve_all = 0
        solve_partial = 0
        num_tasks = 0
        termination_counts = Counter()
        workflow_metrics = defaultdict(list)
        metrics = {}
        timing_raw = {}

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_tasks += len(new_batch.batch)

                new_batch.non_tensor_batch["task_ids"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object)
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n)

                new_batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"], non_tensor_batch_keys=["raw_prompt_ids"])

                with marked_timer("step", timing_raw):
                    # generate trajectories
                    final_gen_batch_output = self.generate_trajectories(batch=new_batch, timing_raw=timing_raw)

                    # need to repeat to make shape match
                    repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                    new_batch = new_batch.sample_level_repeat(repeat_counts)
                    final_gen_batch_output.meta_info.pop("repeat_counts", None)  # no longer needed after this
                    new_batch = new_batch.union(final_gen_batch_output)

                    # rejection sampling
                    # we do rejection sampling at the episode level instead of the traj/step level
                    uids = new_batch.non_tensor_batch["task_ids"]
                    unique_uids = np.unique(uids)
                    is_correct = new_batch.non_tensor_batch["is_correct"]
                    drop_uids = set()

                    for uid in unique_uids:
                        candidate_rows = uids == uid
                        candidate_is_correct = is_correct[candidate_rows]

                        # Check if all episodes are correct or incorrect
                        if not candidate_is_correct.any():
                            drop_uids.add(uid)
                            solve_none += 1
                        elif candidate_is_correct.all():
                            drop_uids.add(uid)
                            solve_all += 1
                        else:
                            solve_partial += 1

                    # Build a view with a single item per episode_id for metrics/logging
                    seen_episodes = set()
                    episode_unique_idxs = []
                    for i, episode_id in enumerate(new_batch.non_tensor_batch["episode_ids"]):
                        if episode_id not in seen_episodes:
                            seen_episodes.add(episode_id)
                            episode_unique_idxs.append(i)
                    episode_unique_batch = new_batch.select_idxs(episode_unique_idxs)

                    # log metrics returned by workflows
                    for metric_dict in episode_unique_batch.non_tensor_batch["metrics"]:
                        for key, value in metric_dict.items():
                            workflow_metrics[key].append(value)

                    # collect and log termination reasons
                    termination_reasons = episode_unique_batch.non_tensor_batch["termination_reasons"]
                    termination_counts.update(termination_reasons)

                    # If no valid samples remain, skip this batch and get a new one
                    # if len(drop_uids) == len(unique_uids):
                    #     print("No valid samples remain, skipping batch")
                    #     continue

                    if not self.config.rllm.rejection_sample.enable:
                        batch = new_batch
                    else:
                        rejection_mask = np.isin(uids, list(drop_uids))
                        new_batch = new_batch[~rejection_mask]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        if solve_partial < self.config.data.train_batch_size:
                            continue
                        else:
                            # randomly select bsz task uids from batch, then filter batch to only contain these tasks
                            # TODO: add heuristic for selecting train_batch_size uids
                            uids = batch.non_tensor_batch["task_ids"]
                            unique_uids = np.unique(uids)
                            assert len(unique_uids) >= self.config.data.train_batch_size, "Not enough unique uids to sample from"
                            selected_uids = np.random.choice(unique_uids, size=self.config.data.train_batch_size, replace=False)
                            selected_mask = np.isin(uids, selected_uids)
                            batch = batch[selected_mask]

                    if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                        # need to make sure both number of last steps (number of uids) and number of total steps in the batch
                        # (batch size after processing) are both multiples of world size

                        # first we split the batch in two: one with only the last steps of each trajectory and the other with the remaining steps
                        is_last_step = batch.non_tensor_batch["is_last_step"]
                        valid_last_step_indices = np.where(is_last_step == True)[0]
                        not_last_step_indices = np.where(is_last_step == False)[0]
                        last_step_batch = batch.select_idxs(valid_last_step_indices)  # This batch only has valid last steps
                        non_last_step_batch = batch.select_idxs(not_last_step_indices)

                        # round down last_step_batch to make sure its multiple of world size
                        num_trainer_replicas = self.actor_wg.world_size
                        max_batch_size = (last_step_batch.batch["input_ids"].shape[0] // num_trainer_replicas) * num_trainer_replicas

                        size_mask = torch.zeros(last_step_batch.batch["input_ids"].shape[0], dtype=torch.bool)
                        size_mask[:max_batch_size] = True
                        last_step_batch = last_step_batch[size_mask]  # filtered last steps

                        # now we go through all the non_last_step_batch and keep everything that has same trajectory_id that exists in the filtered last steps
                        valid_last_step_trajectory_ids = last_step_batch.non_tensor_batch["trajectory_ids"]
                        non_last_step_trajectory_ids = non_last_step_batch.non_tensor_batch["trajectory_ids"]
                        non_last_step_mask = np.isin(non_last_step_trajectory_ids, valid_last_step_trajectory_ids)
                        non_last_step_batch = non_last_step_batch[non_last_step_mask]

                        # concatenate then pad
                        batch = DataProto.concat([last_step_batch, non_last_step_batch])
                        batch = self._pad_dataproto_to_world_size(batch)

                    else:
                        # then we just pad the batch size to a multiple of world size
                        batch = self._pad_dataproto_to_world_size(batch=batch)

                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # step_ids is safe to always use for advantage computation
                        # if we're not using computing advantages stepwise (i.e., for cumulative agents or single turn workflows)
                        # then step_ids == trajectory_ids
                        batch.non_tensor_batch["uid"] = batch.non_tensor_batch["step_ids"]

                        if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "per_step":
                            batch.batch["token_level_scores"] = batch.batch["step_rewards"]
                        else:
                            batch.batch["token_level_scores"] = batch.batch["traj_rewards"]

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
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

                        if self.config.rllm.stepwise_advantage.enable and self.config.rllm.stepwise_advantage.mode == "broadcast":
                            # Merging the separated out steps using the advantage from last steps
                            self._stepwise_advantage_broadcast(batch, non_last_step_batch)
                            batch = DataProto.concat([batch, non_last_step_batch])

                    # remove invalid items filtered out due to compact filtering
                    is_valid = batch.non_tensor_batch["is_valid"]
                    valid_idxs = np.where(is_valid == True)[0]
                    batch = batch.select_idxs(valid_idxs)

                    # for backward compatibility
                    if self.config.rllm.mask_truncated_samples:
                        mask = batch.batch["attention_mask"][:, -1] == 1
                        batch = batch[~mask]

                    # re-pad batch size to world size for gradient update
                    batch = self._pad_dataproto_to_world_size(batch=batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

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
                            actor_output = self.actor_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate_agent()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    with marked_timer("rollout_model_update", timing_raw, color="purple"):
                        checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
                        if not os.path.isabs(checkpoint_folder):
                            working_dir = os.getcwd()
                            checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
                        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
                        actor_path = os.path.join(global_step_folder, "actor")
                        lora_adapter_path = os.path.join(actor_path, "lora_adapter")

                        fireworks_model_id_prefix = self.config.fireworks.model_id_prefix
                        self.fireworks_engine.update_model_weights(
                            fireworks_model_id=f"{fireworks_model_id_prefix}-{self.global_steps}",
                            lora_adapter_path=lora_adapter_path,
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

                metrics["batch/solve_none"] = solve_none / num_tasks
                metrics["batch/solve_all"] = solve_all / num_tasks
                metrics["batch/solve_partial"] = solve_partial / num_tasks

                for key, value in workflow_metrics.items():
                    metrics[f"batch/{key}"] = np.mean(value)

                for r in TerminationReason:
                    metrics[f"batch/{r.value}"] = termination_counts[r.value] / len(set(new_batch.non_tensor_batch["episode_ids"]))

                metrics["batch/num_tasks"] = num_tasks

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                batch = None
                solve_none = 0
                solve_all = 0
                solve_partial = 0
                num_tasks = 0
                termination_counts = Counter()
                workflow_metrics = defaultdict(list)
                metrics = {}
                timing_raw = {}

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_agent()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
