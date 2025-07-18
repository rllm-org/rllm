import uuid
from pprint import pprint
from typing import Any, List, Type, Dict
import torch
from copy import deepcopy
import time
import threading
import queue

import ray
import numpy as np
from verl import DataProto

from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    Role,
    compute_timing_metrics,
    compute_data_metrics,
    dataprotoitem_to_dataproto,
    compute_advantage,
    reduce_metrics,
)

class Timer:
    def __init__(self, name, timing_dict, print_func=None):
        self.name = name
        self.timing_dict = timing_dict
        self.start_time = None

        self.print_func = print_func

    def __enter__(self):
        if self.name not in self.timing_dict:
            self.timing_dict[self.name] = 0.0
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        self.timing_dict[self.name] += elapsed

        if self.print_func is not None:
            self.print_func("%s took %.3fs..." % (self.name, elapsed))

def update_metrics(metrics, new_metrics):
    for k, v in new_metrics.items():
        if k not in metrics:
            metrics[k] = []
        metrics[k].append(v)


def split(batch: DataProto, batch_size: int):
    """Split the DataProto into smaller batches of specified size.

    Args:
        batch (DataProto): The DataProto to split
        batch_size (int): Size of each mini batch

    Yields:
        DataProto: Mini batches of the specified size
    """
    total_size = len(batch)

    # Validate batch size
    assert batch_size > 0, "Batch size must be positive"
    assert total_size >= batch_size, f"Total size {total_size} must be >= batch size {batch_size}"

    # Split into chunks
    num_chunks = (total_size + batch_size - 1) // batch_size
    chunks = batch.chunk(num_chunks)

    for chunk in chunks:
        yield chunk


class RayPPOAsyncTrainer(RayPPOTrainer):

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        assert not self.hybrid_engine, "PPO async trainer does not support hybrid engine, assumes Rollout and Actor are not in the same worker group"

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # we start from step 1
        self.global_steps += 1
        replay_queue = queue.Queue()

        print("Broadcasting weights from actor to rollout.")
        # Broadcast weights from actor to rollout.
        with Timer("Broadcasting weights", {}, print_func=print):
            updated_actor_module_fsdp_ref = self.actor_wg.get_state_dict()

            # Assume rank=0 is first
            assert ray.get(updated_actor_module_fsdp_ref[0]) is not None, "Rank=0 broadcast results expected first."

            self.rollout_wg.update_rollout_actor_module(updated_actor_module_fsdp_ref[0])
        print("Done broadcasting weights from actor to rollout.")

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        print('Initializing replay buffer.')
        start_time = time.perf_counter()
        train_dataloader_gen = iter(self.train_dataloader)
        batch_dict = next(train_dataloader_gen)
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        if self.async_rollout_mode:
            self.async_rollout_manager.wake_up()
            batch = self.async_rollout_manager.generate_sequences(batch)
            self.async_rollout_manager.sleep()
        else:
            batch = self.actor_rollout_wg.generate_sequences(batch)
        replay_queue.put(batch)
        end_time = time.perf_counter()
        print(f'Done initializing replay buffer in {end_time - start_time:.2f} seconds')

        for _ in range(self.config.trainer.total_epochs):
            if not train_dataloader_gen:
                train_dataloader_gen = iter(self.train_dataloader)
            for _, batch_dict in enumerate(train_dataloader_gen):
                metrics = {}
                timing_raw = {}
                sample_batch: DataProto = DataProto.from_single_dict(batch_dict)
                sample_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(sample_batch.batch))], dtype=object)

                with Timer('step', timing_raw):
                    if self.async_rollout_mode:
                        def async_sampler(batch):
                            with Timer('gen', timing_raw):
                                self.async_rollout_manager.wake_up()
                                batch = self.async_rollout_manager.generate_sequences(batch)
                                self.async_rollout_manager.sleep()
                                replay_queue.put(batch)
                        thread = threading.Thread(target=async_sampler, args=(sample_batch,))
                        thread.start()
                    else:
                        def sync_sampler(batch):
                            with Timer('gen', timing_raw):
                                batch = self.rollout_wg.generate_sequences(batch)
                                replay_queue.put(batch)
                        thread = threading.Thread(target=sync_sampler, args=(sample_batch))
                        thread.start()


                    ppo_train_batch_size = self.config.data.train_batch_size
                    ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                    assert ppo_train_batch_size % ppo_mini_batch_size == 0, "PPO mini batch size must be a divisor of the total training batch size"
                    assert replay_queue.qsize() >= 1, "Replay queue must have at least one training batch"
                    batch = replay_queue.get()

                    with Timer('adv', timing_raw):
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor
                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        solve_none = 0
                        solve_all = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards == 0).all():
                                valid_mask[uid_mask] = False
                                solve_none += 1
                            elif (uid_rewards == 1).all():
                                valid_mask[uid_mask] = False
                                solve_all += 1# Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_all'] = solve_all


                        if self.config.trainer.rejection_sample:
                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue
                            # Keep track of valid samples and non-valid samples
                            valid_indices = torch.where(valid_mask)[0]
                            non_valid_indices = torch.where(~valid_mask)[0]

                            num_valid_samples = len(valid_indices)
                            # Calculate how many samples we need to add for a full batch
                            total_batch_size = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            padding_needed = total_batch_size - num_valid_samples
                            # If we need padding and have non-valid samples available, use them
                            combined_indices = valid_indices.tolist()
                            if padding_needed > 0 and len(non_valid_indices) > 0:
                                # Select padding_needed non-valid samples (or as many as available)
                                padding_samples = min(padding_needed, len(non_valid_indices))
                                # Randomly select from non-valid indices to use as padding
                                padding_indices = non_valid_indices[torch.randperm(len(non_valid_indices))[:padding_samples]]
                                combined_indices.extend(padding_indices.tolist())

                            # Create a new mask for the combined set of samples
                            final_mask = torch.zeros(len(batch.batch['input_ids']), dtype=torch.bool)
                            final_mask[combined_indices] = True

                            # Apply the mask to keep only selected samples
                            batch = batch[final_mask]
                            batch = dataprotoitem_to_dataproto(batch)


                        # Recompute old_log_probs using Pytorch FSDP.
                        with Timer('old_log_prob', timing_raw):
                            old_log_prob = self.actor_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with Timer('ref', timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  mask_truncated_samples=self.config.algorithm.mask_truncated_samples)
                        # balance the number of valid tokens on each dp rank.
                        # Note that this breaks the order of data inside the batch.
                        # Please take care when you implement group based adv computation such as GRPO and rloo
                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)
                        # compute global_valid tokens
                        batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                        # update actor
                        start_time = time.perf_counter()
                        with Timer('update_actor', timing_raw):
                            actor_output = self.actor_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        end_time = time.perf_counter()
                        print(f"Actor update took {end_time - start_time:.2f} seconds")
                        metrics.update(actor_output_metrics)

                    thread.join()
                    # last_iter_mini_batch_iter = (mini_batch_iter + last_iter_mini_batch_iter - 1) % ppo_step_minibatch_iter
                    with Timer('rollout_model_update', timing_raw):
                        updated_actor_module_fsdp_ref = self.actor_wg.get_state_dict()
                        if isinstance(updated_actor_module_fsdp_ref, list):
                            updated_actor_module_fsdp_ref = updated_actor_module_fsdp_ref[0]
                        self.rollout_wg.update_rollout_actor_module(updated_actor_module_fsdp_ref)

                    # Validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with Timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with Timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                self.global_steps += 1


                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with Timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return
            train_dataloader_gen = None
