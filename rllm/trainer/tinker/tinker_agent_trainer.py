"""Tinker-based trainer for rLLM agents.

This is a simplified wrapper around TinkerTrajectoryGenerator and TinkerPolicyTrainer
that provides backwards compatibility with the original AgentTrainer interface.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING
import uuid

import tinker
from omegaconf import OmegaConf

from rllm.trainer.tinker.tinker_metrics_utils import (
    compute_training_metrics,
    print_metrics_table,
    print_episodes,
)
from rllm.trainer.tinker.tinker_data_processor import Trajectory, Step, Episode
from rllm.trainer.tinker.tinker_policy_trainer import TinkerPolicyTrainer
from rllm.engine.agent_execution_engine import AsyncAgentExecutionEngine
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from queue import Queue
from threading import Thread
from collections import defaultdict
from typing import Dict, List

from rllm.data import DatasetRegistry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


class TinkerAgentTrainer:
    """
    Simplified trainer for agents using Tinker backend.

    This trainer uses the separated architecture with TinkerTrajectoryGenerator
    and TinkerPolicyTrainer for cleaner code organization and maintainability.
    """

    def __init__(
        self,
        config,
        tokenizer=None,  # Deprecated: kept for backwards compatibility, not used
        agent_class=None,
        env_class=None,
        agent_args=None,
        env_args=None,
        train_dataloader=None,
        val_dataloader=None,
    ):
        """
        Initialize the Tinker agent trainer.

        Args:
            config: Training configuration (OmegaConf)
            tokenizer: (Deprecated) Tokenizer - no longer needed, kept for compatibility
            agent_class: Agent class to instantiate
            env_class: Environment class to instantiate
            agent_args: Arguments for agent initialization
            env_args: Arguments for environment initialization
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
        """
        self.config = config
        # self.initialize_dataloaders()
        self.env_class = env_class
        self.agent_class = agent_class
        self.agent_args = agent_args
        self.env_args = env_args
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        service_client = tinker.ServiceClient(base_url=self.config.tinker.base_url)
        self.trainer = TinkerPolicyTrainer(
            config=config,
            service_client=service_client,
        )
        self.agent_execution_engine = AsyncAgentExecutionEngine(
            config=self.config,
            engine_name="tinker",
            tokenizer=tokenizer,
            max_steps=self.config.agent.max_steps,
            max_response_length=self.config.data.max_response_length,
            max_prompt_length=self.config.data.max_prompt_length,
            agent_class=agent_class,
            agent_args=agent_args,
            env_class=env_class,
            env_args=env_args,
            rollout_engine_args={
                "base_url": None,
                "model_name": self.config.tinker.model.name,
                "tokenizer": tokenizer,
                "service_client": service_client,
                "max_prompt_length": self.config.data.max_prompt_length,
                "max_response_length": self.config.data.max_response_length,
                "sampling_params": self.config.tinker.sampling,
            }
        )
        # Track number of batches for progress calculation
        self.num_train_batches = None
    
    def fit_agent(self):
        """Main training loop using Tinker backend - sync wrapper for async training."""
        asyncio.run(self._fit_agent_async())
    
    async def _fit_agent_async(self):
        """Async main training loop using Tinker backend."""
        from verl.utils.tracking import Tracking

        # Ensure checkpoint directory exists
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)

        # Setup logging
        logger_backend = self.config.trainer.logger
        if isinstance(logger_backend, str):
            logger_backend = [logger_backend]

        tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=logger_backend,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # Initialize or resume training client
        start_batch, sampling_client = await self.trainer.initialize_async(resume_from_checkpoint=True)

        # Validation before training
        if self.config.trainer.get("val_before_train", False) and self.val_dataloader:
            self.agent_execution_engine.rollout_engine.set_sampling_client(sampling_client)
            val_metrics = await self.validate_agent(self.val_dataloader, sampling_client)
            if val_metrics:
                tracking_logger.log(data=val_metrics, step=start_batch)

        # Training loop
        batch_idx = 0
        
        # Calculate total batches for progress tracking
        if self.train_dataloader and hasattr(self.train_dataloader, '__len__'):
            self.num_train_batches = len(self.train_dataloader)
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_data in self.train_dataloader:
                if batch_idx < start_batch:
                    batch_idx += 1
                    continue

                t_start = time.time()
                time_metrics = {}

                batch_data = self.build_interleave_batch(batch_data, self.config.tinker.training.group_size)
                self.init_envs_and_agents(batch_data)

                # Pass tokenizer on first call (batch_idx == start_batch)
                logger.info(f"Loading weights for batch {batch_idx}")
                t_load_weights_start = time.time()
                self.agent_execution_engine.rollout_engine.set_sampling_client(sampling_client)
                time_metrics["time/load_weights"] = time.time() - t_load_weights_start

                # Step 2 & 3: Streaming episode generation and training
                logger.info(f"Generating episodes for batch {batch_idx}")
                t_sample_start = time.time()

                # Calculate minibatch size from config
                num_minibatches = self.config.tinker.training.get("num_minibatches", 1)
                total_batch_size = len(batch_data) // self.config.tinker.training.group_size
                minibatch_size = max(1, total_batch_size // num_minibatches)

                # Collect all episodes, logprobs, and datums for metrics
                episodes = []
                training_logprobs = []
                training_datums = []
                minibatch_count = 0
                forward_backward_times = []

                # Stream: train on each minibatch as it arrives
                train_step_start = time.time()
                async for minibatch_episodes in self.generate_agent_episodes(
                    group_size=self.config.tinker.training.group_size,
                    minibatch_size=minibatch_size
                ):
                    episodes.extend(minibatch_episodes)
                    minibatch_count += 1
                    
                    # Track timing on first minibatch
                    if minibatch_count == 1:
                        time_metrics["time/first_minibatch_sample"] = time.time() - t_sample_start
                    
                    if minibatch_count == num_minibatches:
                        time_metrics["time/last_minibatch_sample"] = time.time() - t_sample_start
                    
                    logger.info(f"Training for batch {batch_idx}, minibatch {minibatch_count}/{num_minibatches}")
                    
                    # Train immediately (streaming), only optimize on last minibatch
                    t_train_start = time.time()
                    logprobs, datums = await self.trainer.step(
                        minibatch_episodes,
                        optimizer_step=False
                    )
                    forward_backward_times.append(time.time() - t_train_start)
                    training_logprobs.extend(logprobs)
                    training_datums.extend(datums)
                    logger.info(f"Processed minibatch {minibatch_count}/{num_minibatches} with {len(minibatch_episodes)} episodes")

                optim_step_time = time.time()
                optim_step_future = await self.trainer.optim_step_future(learning_rate=self.config.tinker.training.learning_rate)
                await optim_step_future.result_async()
                time_metrics["time/optim_step"] = time.time() - optim_step_time
                time_metrics["time/forward_backward"] = sum(forward_backward_times)
                time_metrics["time/one_batch_generate_and_train"] = time.time() - train_step_start

                # Print episodes
                print_episodes(episodes, self.tokenizer, num_episodes_to_print=2)

                # Step 4: Compute and log metrics
                time_metrics["time/total"] = time.time() - t_start
                total_batches = self.num_train_batches * self.config.trainer.total_epochs if self.num_train_batches else None

                metrics = compute_training_metrics(
                    episodes=episodes,
                    batch_idx=batch_idx,
                    time_metrics=time_metrics,
                    learning_rate=self.config.tinker.training.learning_rate,
                    total_batches=total_batches,
                    epoch=epoch,
                    training_datums=training_datums,  # Pass datums for KL/perplexity metrics
                    training_logprobs=training_logprobs,
                )
                tracking_logger.log(data=metrics, step=batch_idx)
                print_metrics_table(metrics, batch_idx)
                
                # Validation
                if (
                    self.val_dataloader
                    and self.config.trainer.test_freq > 0
                    and batch_idx % self.config.trainer.test_freq == 0
                    and batch_idx > 0
                ):
                    logger.info(f"Validating at batch {batch_idx}")
                    val_metrics = await self.validate_agent(self.val_dataloader, sampling_client)
                    if val_metrics:
                        tracking_logger.log(data=val_metrics, step=batch_idx)

                batch_idx += 1

                logger.info(f"Saving sampler checkpoint at batch {batch_idx}")
                t_save_sampler_start = time.time()
                path_dict = await self.trainer.save_checkpoint_async(batch_idx, kind="sampler")
                sampling_client = self.trainer.create_sampling_client(path_dict["sampler_path"])
                time_metrics["time/save_sampler"] = time.time() - t_save_sampler_start

                # Checkpoint (full state) - skip if this is the resume batch
                if batch_idx % self.config.tinker.training.save_every == 0:
                    logger.info(f"Saving state checkpoint at batch {batch_idx}")
                    await self.trainer.save_checkpoint_async(batch_idx, kind="state")

        # Save final checkpoint
        if batch_idx % self.config.tinker.training.save_every != 0:
            logger.info(f"Saving final checkpoint at batch {batch_idx}")
            await self.trainer.save_checkpoint_async(batch_idx, kind="state")
        del tracking_logger

    def init_envs_and_agents(self, batch):
        """
        Initialize environment depending on env_class with the necessary extra_info, also set uid of the batch.
        """
        env_args = batch

        full_agent_args = self.agent_args
        base_env_args = self.env_args

        def _create_env(i):
            if isinstance(env_args[i], str):
                env_args[i] = json.loads(env_args[i])
            return i, self.env_class.from_dict({**env_args[i], **base_env_args})

        def _create_agent(i):
            return i, self.agent_class(**full_agent_args)

        # Create environments in parallel while preserving order
        envs = [None] * len(env_args)
        with ThreadPoolExecutor(max_workers=64) as executor:
            env_futures = [executor.submit(_create_env, i) for i in range(len(env_args))]
            for future in as_completed(env_futures):
                idx, env = future.result()
                envs[idx] = env

        # Create agents in parallel while preserving order
        agents = [None] * len(envs)
        with ThreadPoolExecutor(max_workers=64) as executor:
            agent_futures = [executor.submit(_create_agent, i) for i in range(len(envs))]
            for future in as_completed(agent_futures):
                idx, agent = future.result()
                agents[idx] = agent
        self.agent_execution_engine.update_envs_and_agents(envs, agents)
        return envs
    
    async def validate_agent(self, dataloader, sampling_client):
        episodes_ls = []
        self.agent_execution_engine.rollout_engine.set_sampling_client(sampling_client)
        for batch in dataloader:
            batch = self.build_interleave_batch(batch, 1)
            self.init_envs_and_agents(batch)
            # For validation, collect all episodes from generator
            async for episode_batch in self.generate_agent_episodes(
                group_size=1,
                minibatch_size=1
            ):
                episodes_ls.extend(episode_batch)
        
        all_trajectories = []
        for episode in episodes_ls:
            all_trajectories.extend(episode.trajectories)
        
        mean_reward = sum([traj.reward for traj in all_trajectories]) / len(all_trajectories)
        std_reward = sum([(traj.reward - mean_reward) ** 2 for traj in all_trajectories]) / len(all_trajectories)
        min_reward = min([traj.reward for traj in all_trajectories])
        max_reward = max([traj.reward for traj in all_trajectories])
        mean_turns = sum([len(traj.steps) for traj in all_trajectories]) / len(all_trajectories)
        return {
            "val/reward_mean": mean_reward,
            "val/reward_std": std_reward,
            "val/reward_min": min_reward,
            "val/reward_max": max_reward,
            "val/turns_mean": mean_turns,
        }

    async def generate_agent_episodes(self, timing_raw=None, meta_info=None, group_size: int = None, minibatch_size: int = None):
        """
        Generate episodes in minibatches with overlapping generation and training.
        
        This uses a background producer task to continuously generate episodes
        while the main loop yields minibatches for training.
        """
        if timing_raw is None:
            timing_raw = {}

        group_dict = defaultdict(list)
        episode_queue = asyncio.Queue()
        produce_completed = asyncio.Event()

        async def produce_episodes():
            """Background task: continuously generate episodes and fill queue"""
            async for traj in self.agent_execution_engine.trajectory_generator(
                timing_raw=timing_raw, mode="Step", meta_info=meta_info
            ):
                group_dict[traj["idx"] // group_size].append(traj)
                if len(group_dict[traj["idx"] // group_size]) == group_size:
                    episode = self.convert_to_episode(group_dict[traj["idx"] // group_size])
                    await episode_queue.put(episode)  # Use await for async queue
            produce_completed.set()

        # Start producer in background
        producer_task = asyncio.create_task(produce_episodes())

        try:
            # This generator yields to the caller (training loop)
            minibatch = []
            while True:
                # Collect minibatch
                try:
                    episode = await asyncio.wait_for(episode_queue.get(), timeout=0.1)
                    minibatch.append(episode)
                except asyncio.TimeoutError:
                    if produce_completed.is_set():
                        break
                    continue
                
                if len(minibatch) == minibatch_size:
                    yield minibatch
                    minibatch = []
            
            # Yield any remaining episodes
            remaining = episode_queue.qsize()
            if remaining > 0:
                for _ in range(remaining):
                    episode = episode_queue.get_nowait()
                    minibatch.append(episode)
                yield minibatch

        finally:
            await producer_task
    
    def convert_to_episode(self, group: list):
        trajectories = []
        for traj in group:
            steps = []
            for step in traj["steps"]:
                steps.append(Step(
                    prompt_ids=step["prompt_ids"],
                    response_ids=step["response_ids"],
                    logprobs=step["logprobs"],
                ))
            traj = Trajectory(steps=steps, reward=traj["trajectory_reward"])
            trajectories.append(traj)
        return Episode(trajectories=trajectories)

    def build_interleave_batch(self, batch: list, group_size: int):
        interleave_batch = []
        batch_with_uid = []
        for batch_item in batch:
            batch_with_uid.append({**batch_item, "uid": str(uuid.uuid4())})
        
        for batch_item in batch_with_uid:
            interleave_batch.extend([batch_item for _ in range(group_size)])
        return interleave_batch