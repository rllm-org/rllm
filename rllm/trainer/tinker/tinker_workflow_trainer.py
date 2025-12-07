"""Tinker-based trainer for rLLM agents.

This is a simplified wrapper around TinkerTrajectoryGenerator and TinkerPolicyTrainer
that provides backwards compatibility with the original AgentTrainer interface.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import tinker
import torch
from omegaconf import OmegaConf

from rllm.agents.agent import Episode
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.tinker_engine import TinkerEngine
from rllm.trainer.common import (
    AlgorithmConfig,
    CompactFilteringConfig,
    RejectionSamplingConfig,
    TransformConfig,
    build_trajectory_groups,
    reduce_reward_metrics_by_trajectory_name,
)
from rllm.trainer.tinker.tinker_metrics_utils import compute_training_metrics, print_metrics_table
from rllm.trainer.tinker.tinker_policy_trainer import TinkerPolicyTrainer
from rllm.utils import Tracking, marked_timer, visualize_trajectory_last_steps

if TYPE_CHECKING:
    from rllm.data import Dataset
    from rllm.workflows.workflow import Workflow


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


def build_interleave_batch(batch: list, group_size: int):
    interleave_batch = []
    batch_with_uid = []
    for batch_item in batch:
        batch_with_uid.append({**batch_item, "uid": str(uuid.uuid4())})

    for batch_item in batch_with_uid:
        interleave_batch.extend([batch_item for _ in range(group_size)])
    return interleave_batch


class TinkerWorkflowTrainer:
    """
    Simplified trainer for agents using Tinker backend.

    This trainer uses the separated architecture with TinkerTrajectoryGenerator
    and TinkerPolicyTrainer for cleaner code organization and maintainability.
    """

    def __init__(
        self,
        config,
        workflow_class: type[Workflow],
        train_dataset: Dataset,
        workflow_args: dict | None = None,
        val_dataset: Dataset | None = None,
    ):
        """
        Initialize the Tinker workflow trainer.

        Args:
            config: Training configuration (OmegaConf)
            workflow_class: Workflow class to instantiate
            workflow_args: Arguments for workflow initialization
            train_dataset: Training data loader
            val_dataset: Validation data loader
        """
        self.config = config
        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            collate_fn=lambda x: x,  # Return batches as lists
        )

        if val_dataset is not None:
            self.val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.config.data.val_batch_size,
                shuffle=False,
                collate_fn=lambda x: x,  # Return batches as lists
            )
        else:
            self.val_dataloader = None

        self._validate_and_setup_configs()

        service_client = tinker.ServiceClient(base_url=self.config.tinker_base_url)
        self.trainer = TinkerPolicyTrainer(
            config=config,
            service_client=service_client,
        )

        self.rollout_engine = TinkerEngine(
            base_url=self.config.tinker_base_url,
            model_name=self.config.model.name,
            service_client=service_client,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            sampling_params=self.config.sampling,
        )
        self.tokenizer = self.rollout_engine.tokenizer

        self.agent_execution_engine = AgentWorkflowEngine(
            workflow_cls=self.workflow_class,
            workflow_args=self.workflow_args,
            rollout_engine=self.rollout_engine,
            config=self.config,
            n_parallel_tasks=self.config.workflow.n_parallel_tasks,
            retry_limit=self.config.workflow.retry_limit,
        )

        self.n_parallel_tasks = self.config.workflow.n_parallel_tasks
        # Track number of batches for progress calculation
        self.num_train_batches = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        asyncio.run_coroutine_threadsafe(self.agent_execution_engine.initialize_pool(), self._loop).result()

    def _validate_and_setup_configs(self):
        assert self.config.training.num_minibatches == 1, f"Only num_minibatches=1 is supported for workflow trainer, current num_minibatches={self.config.training.num_minibatches}"

        sampling_params = self.config.sampling
        # make an warning when the temperature or top_p is set away from default value
        if sampling_params.get("temperature", 1.0) != 1.0 or sampling_params.get("top_p", 1.0) != 1.0:
            logger.warning("Temperature and top_p are set away from 1.0, this is not recommended by Tinker and can cause mysterious issue with logprobs. See https://github.com/thinking-machines-lab/tinker-cookbook/pull/86 for discussion.")

        self.cf_config = CompactFilteringConfig.from_config(self.config.rllm.compact_filtering)

        # transform config (used for transforming episodes to trajectory groups)
        self.transform_config = TransformConfig(broadcast=self.config.rllm.stepwise_advantage.mode == "broadcast")

        # rejection sampling config (used for rejection sampling)
        rs_mode = "episode" if self.config.rllm.rejection_sample.enable else "none"
        self.rs_config = RejectionSamplingConfig(mode=rs_mode, min_partial_solve_tasks=self.config.data.train_batch_size)

        # algorithm config (used for rLLM-native advantage computation)
        self.algorithm_config = AlgorithmConfig(
            estimator=self.config.algorithm.adv_estimator,
            stepwise_advantage_mode=self.config.rllm.stepwise_advantage.mode,
            norm_adv_by_std_in_grpo=self.config.rllm.stepwise_advantage.get("norm_adv_by_std_in_grpo", True),
        )

    async def validate_agent(self, dataloader, sampling_client):
        all_episodes = []
        self.rollout_engine.set_sampling_client(sampling_client)

        for batch in dataloader:
            current_batch = build_interleave_batch(batch, 1)
            # For validation, collect all episodes from generator
            async for episodes in self.generate_episodes(current_batch):
                all_episodes.extend(episodes)

        all_episode_metrics = [episode.metrics for episode in all_episodes if episode.metrics is not None]

        # Collect workflow metrics per episode (deduplicated by episode.id)
        # all_episode_metrics is: {episode_id: {metric_name: metric_value, ...}, ...}
        workflow_metrics = defaultdict(list)
        for episode_metric_dict in all_episode_metrics:
            for key, value in episode_metric_dict.items():
                workflow_metrics[key].append(float(value))

        # Turn episodes into trajectory groups without filtering or validating
        trajectory_groups = build_trajectory_groups(all_episodes, compact_filtering_config=self.cf_config)

        # Compute trajectory-level reward statistics
        metrics = reduce_reward_metrics_by_trajectory_name(trajectory_groups, prefix="val")

        # Add workflow-provided metrics (e.g., solver_acc, judge_acc)
        for key, values in workflow_metrics.items():
            if values:
                metrics[f"val/{key}"] = sum(values) / len(values)

        return metrics

    async def generate_episodes(self, current_batch: list[dict]) -> AsyncGenerator[list[Episode], None]:
        """
        Generate episodes from workflow execution.

        Args:
            current_batch: The current batch of data to generate episodes from

        Yields:
            list[Episode]
        """
        # TODO: implement mini-batching. Currently we only support one minibatch per batch.
        assert current_batch is not None, "current_batch is None"

        task_ids = [item["uid"] for item in current_batch]

        episodes = await self.agent_execution_engine.execute_tasks(current_batch, task_ids)

        yield episodes

    def fit_agent(self):
        """Main training loop using Tinker backend - sync wrapper for async training."""
        asyncio.run(self._fit_agent_async())

    async def _fit_agent_async(self):
        """Async main training loop using Tinker backend."""

        # Ensure checkpoint directory exists
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)

        # Setup logging
        tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # Initialize or resume training client
        start_batch, sampling_client = await self.trainer.initialize_async(resume_from_checkpoint=True)

        # Validation before training
        if self.config.trainer.get("val_before_train", False) and self.val_dataloader:
            self.rollout_engine.set_sampling_client(sampling_client)
            val_metrics = await self.validate_agent(self.val_dataloader, sampling_client)
            if val_metrics:
                tracking_logger.log(data=val_metrics, step=start_batch)

        # Training loop
        batch_idx = 0
        break_flag = False

        learning_rate = self.config.training.learning_rate
        beta1 = self.config.training.beta1
        beta2 = self.config.training.beta2
        eps = self.config.training.eps

        # Calculate total batches for progress tracking
        if self.train_dataloader and hasattr(self.train_dataloader, "__len__"):
            self.num_train_batches = len(self.train_dataloader)

        for epoch in range(self.config.trainer.total_epochs):
            if break_flag:  # useful when `total_batches` is set
                break
            for batch_data in self.train_dataloader:
                if batch_idx < start_batch:
                    batch_idx += 1
                    continue
                elif self.config.trainer.total_batches >= 0 and batch_idx > self.config.trainer.total_batches:
                    break_flag = True
                    break

                timing_raw = {}

                with marked_timer("total", timing_raw):
                    batch_data = build_interleave_batch(batch_data, self.config.training.group_size)

                    # Pass tokenizer on first call (batch_idx == start_batch)
                    logger.info(f"Loading weights for batch {batch_idx}")
                    with marked_timer("load_weights", timing_raw):
                        self.rollout_engine.set_sampling_client(sampling_client)

                    # Step 2 & 3: Streaming episode generation and training
                    logger.info(f"Generating episodes for batch {batch_idx}")
                    t_sample_start = time.time()

                    # Calculate minibatch size from config
                    num_minibatches = self.config.training.get("num_minibatches", 1)
                    # total_batch_size = len(batch_data) // self.config.training.group_size
                    # minibatch_size = max(1, total_batch_size // num_minibatches)

                    # Collect all episodes, trajectory groups, logprobs, and datums for metrics
                    episodes = []
                    trajectory_groups = []
                    training_logprobs = []
                    training_datums = []
                    minibatch_count = 0

                    # Stream: train on each minibatch as it arrives
                    with marked_timer("one_batch_generate_and_train", timing_raw):
                        all_grouping_metrics = []
                        async for minibatch_episodes in self.generate_episodes(batch_data):
                            episodes.extend(minibatch_episodes)
                            minibatch_count += 1

                            # Track timing on first minibatch
                            if minibatch_count == 1:
                                timing_raw["first_minibatch_sample"] = time.time() - t_sample_start

                            if minibatch_count == num_minibatches:
                                timing_raw["last_minibatch_sample"] = time.time() - t_sample_start

                            logger.info(f"Training for batch {batch_idx}, minibatch {minibatch_count}/{num_minibatches}")

                            # Train immediately (streaming), only optimize on last minibatch
                            with marked_timer("forward_backward", timing_raw):
                                datums, mini_traj_groups, logprobs, grouping_metrics = await self.trainer.step(minibatch_episodes, learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps, optimizer_step=False)

                            trajectory_groups.extend(mini_traj_groups)
                            training_logprobs.extend(logprobs)
                            training_datums.extend(datums)
                            all_grouping_metrics.append(grouping_metrics)
                            logger.info(f"Processed minibatch {minibatch_count}/{num_minibatches} with {len(minibatch_episodes)} episodes")

                        optim_step_future = await self.trainer.optim_step_future(learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps)
                        await optim_step_future.result_async()

                # Visualization some last steps (only for one minibatch)
                visualize_trajectory_last_steps(
                    trajectory_groups,
                    tokenizer=self.tokenizer,
                    max_steps_to_visualize=2,
                    show_workflow_metadata=True,
                )

                # Step 4: Compute and log metrics
                total_batches = self.num_train_batches * self.config.trainer.total_epochs if self.num_train_batches else None

                # Validation
                if self.val_dataloader and self.config.trainer.test_freq > 0 and batch_idx % self.config.trainer.test_freq == 0 and batch_idx > 0:
                    logger.info(f"Validating at batch {batch_idx}")
                    val_metrics = await self.validate_agent(self.val_dataloader, sampling_client)
                    if val_metrics:
                        tracking_logger.log(data=val_metrics, step=batch_idx)

                batch_idx += 1

                logger.info(f"Saving sampler checkpoint at batch {batch_idx}")
                with marked_timer("save_sampler", timing_raw):
                    path_dict = await self.trainer.save_checkpoint_async(batch_idx, kind="sampler")
                    sampling_client = self.trainer.create_sampling_client(path_dict["sampler_path"])

                metrics = compute_training_metrics(
                    trajectory_groups=trajectory_groups,
                    batch_idx=batch_idx,
                    time_metrics=timing_raw,
                    learning_rate=learning_rate,
                    total_batches=total_batches,
                    epoch=epoch,
                    training_datums=training_datums,  # Pass datums for KL/perplexity metrics
                    training_logprobs=training_logprobs,
                )

                # Aggregate grouping metrics from all minibatches
                if all_grouping_metrics:
                    aggregated_grouping_metrics = {}
                    for key in all_grouping_metrics[0].keys():
                        values = [m[key] for m in all_grouping_metrics if key in m]
                        if values:
                            aggregated_grouping_metrics[key] = sum(values) / len(values)
                    metrics.update(aggregated_grouping_metrics)

                tracking_logger.log(data=metrics, step=batch_idx)
                print_metrics_table(metrics, batch_idx)

                # Checkpoint (full state) - skip if this is the resume batch
                if batch_idx % self.config.trainer.save_freq == 0:
                    logger.info(f"Saving state checkpoint at batch {batch_idx}")
                    await self.trainer.save_checkpoint_async(batch_idx, kind="state")

        # Save final checkpoint
        if batch_idx % self.config.trainer.save_freq != 0:
            logger.info(f"Saving final checkpoint at batch {batch_idx}")
            await self.trainer.save_checkpoint_async(batch_idx, kind="state")
        del tracking_logger
