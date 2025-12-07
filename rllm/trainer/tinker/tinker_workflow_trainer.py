"""Tinker-based trainer for rLLM agents.

This is a simplified wrapper around TinkerTrajectoryGenerator and TinkerPolicyTrainer
that provides backwards compatibility with the original AgentTrainer interface.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from typing import TYPE_CHECKING

import tinker
import torch

from rllm.agents.agent import Episode
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.tinker_engine import TinkerEngine
from rllm.trainer.common import AlgorithmConfig, CompactFilteringConfig, RejectionSamplingConfig, TransformConfig
from rllm.trainer.tinker.tinker_agent_trainer import TinkerAgentTrainer
from rllm.trainer.tinker.tinker_policy_trainer import TinkerPolicyTrainer

if TYPE_CHECKING:
    from rllm.data import Dataset
    from rllm.workflows.workflow import Workflow


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


class TinkerWorkflowTrainer(TinkerAgentTrainer):
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
        Initialize the Tinker agent trainer.

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

        if isinstance(val_dataset, Dataset):
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

        # Initialize current_batch to avoid AttributeError
        self.current_batch = None

        asyncio.run_coroutine_threadsafe(self.agent_execution_engine.initialize_pool(), self._loop).result()

    def _validate_and_setup_configs(self):
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
            normalize_by_std=self.config.rllm.stepwise_advantage.get("normalize_by_std", True),
        )

    def init_envs_and_agents(self, batch_data):
        # no need to init envs and agents, thats maintained by the workflow
        # Store batch_data for use in generate_agent_episodes
        self.current_batch = batch_data

    async def validate_agent(self, dataloader, sampling_client):
        all_episodes = []
        all_episode_metrics = {}  # episode_id -> episode.metrics dict
        self.rollout_engine.set_sampling_client(sampling_client)

        for batch in dataloader:
            batch = self.build_interleave_batch(batch, 1)
            self.init_envs_and_agents(batch)
            # For validation, collect all episodes from generator
            async for episodes, episode_metrics in self.generate_agent_episodes(group_size=1, minibatch_size=1, return_metrics=True):
                all_episodes.extend(episodes)
                all_episode_metrics.update(episode_metrics)

        # Collect workflow metrics per episode (deduplicated by episode.id)
        # all_episode_metrics is: {episode_id: {metric_name: metric_value, ...}, ...}
        workflow_metrics = defaultdict(list)
        for episode_id, episode_metric_dict in all_episode_metrics.items():
            if episode_metric_dict:  # Check if metrics dict is not None
                for key, value in episode_metric_dict.items():
                    workflow_metrics[key].append(float(value))

        # Compute trajectory-level statistics from all episodes
        all_trajectories = []
        for episode in all_episodes:
            all_trajectories.extend(episode.trajectories)

        mean_reward = sum([traj.reward for traj in all_trajectories]) / len(all_trajectories)
        std_reward = sum([(traj.reward - mean_reward) ** 2 for traj in all_trajectories]) / len(all_trajectories)
        min_reward = min([traj.reward for traj in all_trajectories])
        max_reward = max([traj.reward for traj in all_trajectories])
        mean_turns = sum([len(traj.steps) for traj in all_trajectories]) / len(all_trajectories)
        metrics = {
            "val/reward_mean": mean_reward,
            "val/reward_std": std_reward,
            "val/reward_min": min_reward,
            "val/reward_max": max_reward,
            "val/turns_mean": mean_turns,
        }

        # Add workflow-provided metrics (e.g., solver_acc, judge_acc)
        for key, values in workflow_metrics.items():
            if values:
                metrics[f"val/{key}"] = sum(values) / len(values)

        return metrics

    async def generate_agent_episodes(self, timing_raw=None, meta_info=None, group_size=None, minibatch_size=None, return_metrics=False):
        """
        Generate episodes from workflow execution.

        Args:
            return_metrics: If True, yields (episodes, metrics) tuple where metrics is
                          {episode_id: {metric_name: value, ...}}. If False, yields only episodes.

        Yields:
            list[Episode] or tuple[list[Episode], dict] depending on return_metrics
        """

        num_minibatches = self.config.training.num_minibatches

        assert num_minibatches == 1, f"Only num_minibatches=1 is supported for workflow trainer, current num_minibatches={num_minibatches}"

        current_batch = self.current_batch
        task_ids = [item["uid"] for item in current_batch]

        episodes = await self.agent_execution_engine.execute_tasks(current_batch, task_ids)
        episodes = self.make_sure_contain_token_and_logprob(episodes)

        # Update trajectory-level rewards from step-level rewards
        for episode in episodes:
            for trajectory in episode.trajectories:
                if trajectory.reward == 0.0 and trajectory.steps:
                    # Compute trajectory reward from step rewards
                    trajectory.reward = sum(step.reward if step.reward is not None else 0.0 for step in trajectory.steps)

        # Extract episode metrics if available
        episode_metrics = {ep.id: ep.metrics for ep in episodes if hasattr(ep, "metrics") and ep.metrics}

        if return_metrics:
            yield episodes, episode_metrics
        else:
            yield episodes

    def make_sure_contain_token_and_logprob(self, episodes: list[Episode]) -> list[Episode]:
        for episode in episodes:
            for trajectory in episode.trajectories:
                for step in trajectory.steps:
                    model_output = step.model_output
                    if not step.prompt_ids:
                        step.prompt_ids = model_output.prompt_ids
                    if not step.response_ids:
                        step.response_ids = model_output.completion_ids
                    if not step.logprobs:
                        step.logprobs = model_output.logprobs

        assert step.prompt_ids, "prompt_ids is None"
        assert step.response_ids, "response_ids is None"
        assert step.logprobs, "logprobs is None"

        return episodes
