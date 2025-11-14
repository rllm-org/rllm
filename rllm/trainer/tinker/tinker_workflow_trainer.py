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

from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.tinker_engine import TinkerEngine
from rllm.trainer.tinker.tinker_agent_trainer import TinkerAgentTrainer
from rllm.trainer.tinker.tinker_data_processor import Episode, Trajectory, TrajectoryGroup
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
        all_trajectory_groups = []
        all_episode_metrics = {}  # episode_id -> episode.metrics dict
        self.agent_execution_engine.rollout_engine.set_sampling_client(sampling_client)
        for batch in dataloader:
            batch = self.build_interleave_batch(batch, 1)
            self.init_envs_and_agents(batch)
            # For validation, collect all trajectory groups from generator
            async for trajectory_groups, episode_metrics in self.generate_agent_episodes(group_size=1, minibatch_size=1, return_metrics=True):
                all_trajectory_groups.extend(trajectory_groups)
                all_episode_metrics.update(episode_metrics)

        # Collect workflow metrics per episode (deduplicated by episode.id)
        # all_episode_metrics is: {episode_id: {metric_name: metric_value, ...}, ...}
        workflow_metrics = defaultdict(list)
        for episode_id, episode_metric_dict in all_episode_metrics.items():
            if episode_metric_dict:  # Check if metrics dict is not None
                for key, value in episode_metric_dict.items():
                    workflow_metrics[key].append(float(value))

        # Compute trajectory-level statistics from all groups
        all_trajectories = []
        for group in all_trajectory_groups:
            all_trajectories.extend(group.trajectories)

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
        Generate trajectory groups in minibatches with overlapping generation and training.

        This uses a background producer task to continuously generate episodes (from rollout)
        and regroups them into TrajectoryGroup objects for advantage computation.

        Args:
            return_metrics: If True, yields (trajectory_groups, metrics) tuple where metrics is
                          {episode_id: {metric_name: value, ...}}. If False, yields only trajectory_groups.

        Yields:
            list[TrajectoryGroup] or tuple[list[TrajectoryGroup], dict] depending on return_metrics
        """

        num_minibatches = self.config.training.num_minibatches

        assert num_minibatches == 1, f"Only num_minibatches=1 is supported for workflow trainer, current num_minibatches={num_minibatches}"

        current_batch = self.current_batch
        task_ids = [item["uid"] for item in current_batch]

        episodes = await self.agent_execution_engine.execute_tasks(current_batch, task_ids)
        episodes = self.make_sure_contain_token_and_logprob(episodes)
        trajectory_groups, episode_metrics = self.regroup(episodes)

        if return_metrics:
            yield trajectory_groups, episode_metrics
        else:
            yield trajectory_groups

    def regroup(self, episodes: list[Episode]) -> tuple[list[TrajectoryGroup], dict]:
        """
        Regroup episodes into TrajectoryGroup objects based on grouping_level configuration.

        The grouping level determines how advantages are computed:

        - trajectory: Group trajectories by (task_id, trajectory_name). Each trajectory
                     keeps all its steps. Trajectories with different names are grouped
                     separately (important for multi-agent scenarios). Advantage is
                     computed per trajectory (e.g., via GRPO across trajectory rewards),
                     then broadcast to all steps in that trajectory during datum creation.

        - step: Group individual steps at the same position (task_id + trajectory_name
                + step_idx) across different rollouts. Each step becomes a single-step
                trajectory in a group. Advantage is computed per step (e.g., via GRPO
                across step rewards).

        The resulting TrajectoryGroup objects are consumed by process_trajectory_groups() which:
        1. Extracts rewards from trajectories in each group
        2. Computes advantages across those rewards
        3. Assigns each trajectory its computed advantage
        4. Broadcasts the advantage to all steps in the trajectory

        Args:
            episodes: List of episodes to regroup

        Returns:
            Tuple of (trajectory_groups, metrics_dict)
        """
        grouping_level = self.config.algorithm.grouping_level
        trajectory_groups = []
        metrics = {}

        def get_task_id(episode: Episode):
            return ":".join(episode.id.split(":")[:-1])

        if grouping_level == "trajectory":
            # Group trajectories by (task_id, trajectory_name)
            # This ensures trajectories with different names are grouped separately
            temp_groups = defaultdict(list)

            for episode in episodes:
                if episode.id not in metrics and episode.metrics:
                    metrics[episode.id] = episode.metrics
                task_id = get_task_id(episode)

                # Add all trajectories to the group for this (task_id, trajectory_name)
                for trajectory in episode.trajectories:
                    # Each trajectory keeps all its steps
                    # Compute trajectory-level reward as the sum/mean of step rewards
                    traj_reward = trajectory.reward if trajectory.reward is not None else sum(step.reward for step in trajectory.steps)
                    # Update trajectory with proper reward
                    updated_trajectory = Trajectory(steps=trajectory.steps, reward=traj_reward, name=trajectory.name)
                    # Group by both task_id and trajectory name
                    group_key = (task_id, trajectory.name)
                    temp_groups[group_key].append(updated_trajectory)

            # Create TrajectoryGroup objects from grouped trajectories
            for group_key, trajectories in temp_groups.items():
                group_id = f"{group_key[0]}:{group_key[1]}"  # "task_id:trajectory_name"
                trajectory_group = TrajectoryGroup(trajectories=trajectories, group_id=group_id)
                trajectory_groups.append(trajectory_group)

            print("Trajectory-level grouping:")
            print(f"  len episodes: {len(episodes)}")
            print(f"  len unique (task_id, traj_name) groups: {len(temp_groups)}")
            print(f"  len trajectory_groups: {len(trajectory_groups)}")

        elif grouping_level == "step":
            # Group individual steps by step position
            unique_step_uids = set()
            unique_task_ids = set()
            step_groupby_step_uid = defaultdict(list)

            for episode in episodes:
                if episode.id not in metrics and episode.metrics:
                    metrics[episode.id] = episode.metrics
                task_id = get_task_id(episode)
                unique_task_ids.add(task_id)

                for trajectory in episode.trajectories:
                    for step_idx, step in enumerate(trajectory.steps):
                        step_uid = f"{task_id}:{trajectory.name}:{step_idx}"
                        if step_uid not in unique_step_uids:
                            unique_step_uids.add(step_uid)

                        step_groupby_step_uid[step_uid].append(step)

            # Create TrajectoryGroup objects where each trajectory contains a single step
            for step_uid, steps in step_groupby_step_uid.items():
                trajectories = [Trajectory(steps=[step], reward=step.reward) for step in steps]
                trajectory_group = TrajectoryGroup(trajectories=trajectories, group_id=step_uid)
                trajectory_groups.append(trajectory_group)

            print("Step-level grouping:")
            print(f"  len episodes: {len(episodes)}")
            print(f"  len unique_task_ids: {len(unique_task_ids)}")
            print(f"  len unique_step_uids: {len(unique_step_uids)}")
            print(f"  len trajectory_groups: {len(trajectory_groups)}")
        else:
            raise ValueError(f"Invalid grouping_level: {grouping_level}. Must be 'trajectory' or 'step'")

        return trajectory_groups, metrics

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
