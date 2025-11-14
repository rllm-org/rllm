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
from transformers import AutoTokenizer

from rllm.agents.agent import Episode
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.tinker_engine import TinkerEngine
from rllm.trainer.tinker.tinker_agent_trainer import TinkerAgentTrainer
from rllm.trainer.tinker.tinker_policy_trainer import TinkerPolicyTrainer

if TYPE_CHECKING:
    pass

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
        workflow_class=None,
        workflow_args=None,
        train_dataset=None,
        val_dataset=None,
    ):
        """
        Initialize the Tinker agent trainer.

        Args:
            config: Training configuration (OmegaConf)
            agent_class: Agent class to instantiate
            env_class: Environment class to instantiate
            agent_args: Arguments for agent initialization
            env_args: Arguments for environment initialization
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
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.data.val_batch_size,
            shuffle=False,
            collate_fn=lambda x: x,  # Return batches as lists
        )

        service_client = tinker.ServiceClient(base_url=self.config.tinker_base_url)
        self.trainer = TinkerPolicyTrainer(
            config=config,
            service_client=service_client,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        sampling_params = self.config.sampling
        assert sampling_params.get("temperature", 1.0) == 1.0 and sampling_params.get("top_p", 1.0) == 1.0, "temperature and top_p must be 1.0 for tinker workflow trainer"
        self.rollout_engine = TinkerEngine(
            base_url=self.config.tinker_base_url,
            model_name=self.config.model.name,
            tokenizer=self.tokenizer,
            service_client=service_client,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            sampling_params=sampling_params,
        )
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

    def init_envs_and_agents(self, batch_data):
        # no need to init envs and agents, thats maintained by the workflow
        # Store batch_data for use in generate_agent_episodes
        self.current_batch = batch_data

    async def validate_agent(self, dataloader, sampling_client):
        all_episodes = []
        all_episode_metrics = {}  # episode_id -> episode.metrics dict
        self.agent_execution_engine.rollout_engine.set_sampling_client(sampling_client)
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
