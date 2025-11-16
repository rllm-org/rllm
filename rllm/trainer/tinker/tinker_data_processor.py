"""Data processing utilities for converting trajectories to training data.

This module provides the bridge between trajectory generation and training,
handling filtering, advantage computation, and conversion to tinker.Datum format.
"""

import logging
from dataclasses import dataclass

import numpy as np
import tinker
import torch
from tinker.types.tensor_data import TensorData

from rllm.agents.agent import Step, Trajectory

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryGroup:
    """
    A group of trajectories for advantage computation.

    Unlike Episode (which represents raw rollout data), TrajectoryGroup is specifically
    structured for advantage computation. All trajectories in a group will have their
    rewards compared to compute advantages (e.g., via GRPO).

    Attributes:
        trajectories: List of trajectories to compare for advantage computation
        group_id: Optional identifier for the group (e.g., "task1:agent_0")
    """

    trajectories: list[Trajectory]
    group_id: str = None


class TinkerAdvantageComputer:
    """
    Computes advantages using REINFORCE, GRPO, or other algorithms.
    Compatible with rLLM's existing advantage computation.
    """

    def __init__(self, algorithm_config):
        self.adv_estimator = algorithm_config.adv_estimator
        self.gamma = algorithm_config.gamma
        self.norm_by_std = algorithm_config.get("norm_adv_by_std_in_grpo", True)

    def compute_grpo_advantages(self, group_rewards: list[float]) -> list[float]:
        """
        GRPO: advantage = reward - mean(group_rewards)

        Args:
            group_rewards: List of rewards for the group

        Returns:
            List of advantages
        """
        if not group_rewards:
            return []

        if len(group_rewards) == 1:
            return group_rewards

        mean_reward = sum(group_rewards) / len(group_rewards)
        advantages = [r - mean_reward for r in group_rewards]

        # Optional: normalize by std
        if self.norm_by_std and len(advantages) > 1:
            std = np.std(advantages)
            if std > 1e-8:
                advantages = [a / std for a in advantages]

        return advantages

    def compute_reinforce_advantages(self, group_rewards: list[float]) -> list[float]:
        """
        REINFORCE: advantage = reward (no baseline)

        Args:
            group_rewards: List of rewards

        Returns:
            List of advantages (same as rewards)
        """
        return group_rewards

    def compute(self, group_rewards: list[float]) -> list[float]:
        """
        Compute advantages based on algorithm config.

        Args:
            group_rewards: List of rewards for the group

        Returns:
            List of advantages
        """
        if self.adv_estimator == "grpo":
            return self.compute_grpo_advantages(group_rewards)
        elif self.adv_estimator == "reinforce":
            return self.compute_reinforce_advantages(group_rewards)
        else:
            logger.warning(f"Unknown advantage estimator {self.adv_estimator}, using GRPO")
            return self.compute_grpo_advantages(group_rewards)


class TinkerTrajectoryFilter:
    """
    Filters episodes based on configuration (e.g., removing constant-reward episodes).
    Matches tinker-cookbook's remove_constant_reward_groups functionality.
    """

    def __init__(self, algorithm_config):
        """
        Initialize filter with algorithm configuration.

        Args:
            algorithm_config: Configuration with optional remove_constant_reward_groups flag
        """
        self.remove_constant_reward_groups = algorithm_config.get("remove_constant_reward_groups", False)

    @staticmethod
    def _all_same(values: list[float]) -> bool:
        """Check if all values in the list are the same."""
        if not values:
            return True
        first = values[0]
        return all(abs(v - first) < 1e-8 for v in values)

    def filter_groups(self, groups: list[TrajectoryGroup]) -> list[TrajectoryGroup]:
        """
        Filter trajectory groups based on configuration.

        If remove_constant_reward_groups=True, removes groups where all trajectories
        have the same reward. If all groups would be removed, keeps at least one
        group to prevent empty batches.

        Args:
            groups: List of TrajectoryGroup objects

        Returns:
            Filtered list of TrajectoryGroup objects
        """
        if not self.remove_constant_reward_groups:
            # Keep all groups (default behavior)
            return groups

        # Filter out constant-reward groups
        filtered_groups = []
        for group in groups:
            # Get rewards from all trajectories in the group
            group_rewards = [traj.reward for traj in group.trajectories]
            if not self._all_same(group_rewards):
                filtered_groups.append(group)

        # Safety: Never return empty list to prevent batch size issues
        if not filtered_groups:
            logger.warning("All groups have uniform rewards. There will be no gradient. Keeping one group to prevent empty batch.")
            return groups[:1]

        if len(filtered_groups) < len(groups):
            logger.info(f"Filtered {len(groups) - len(filtered_groups)} constant-reward groups (kept {len(filtered_groups)} groups with reward variance)")

        return filtered_groups


class TinkerDatumBuilder:
    """
    Converts trajectory data to Tinker's Datum format.
    """

    @staticmethod
    def _is_prefix(seq1: list[int], seq2: list[int]) -> bool:
        """Check if seq1 is a prefix of seq2."""
        return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1

    @staticmethod
    def build_datum_from_step(step: Step, advantage: float) -> tinker.Datum:
        """
        Create a Tinker Datum from a Step object.

        Args:
            step: Step object with prompt_ids, response_ids, logprobs
            advantage: Computed advantage value

        Returns:
            Tinker Datum object
        """
        prompt_tokens = step.prompt_ids
        response_tokens = step.response_ids
        logprobs = step.logprobs

        # Combine prompt and response
        all_tokens = prompt_tokens + response_tokens
        input_tokens = all_tokens[:-1]
        target_tokens = all_tokens[1:]

        # Create masks (only train on response)
        ob_len = len(prompt_tokens) - 1
        all_logprobs = [0.0] * ob_len + logprobs
        all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
        all_mask = [0.0] * ob_len + [1.0] * (len(input_tokens) - ob_len)

        # Ensure all lists have the same length
        assert len(input_tokens) == len(target_tokens) == len(all_logprobs) == len(all_advantages) == len(all_mask), f"Length mismatch: input={len(input_tokens)}, target={len(target_tokens)}, logprobs={len(all_logprobs)}, advantages={len(all_advantages)}, mask={len(all_mask)}"

        # Create Datum
        datum = tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(tokens=[int(t) for t in input_tokens]),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                "mask": TensorData.from_torch(torch.tensor(all_mask)),
            },
        )

        return datum

    @staticmethod
    def build_datum(trajectory: dict, advantage: float) -> tinker.Datum:
        """
        Create a Tinker Datum from trajectory dict (backwards compatibility).

        Args:
            trajectory: Trajectory dictionary with tokens, logprobs
            advantage: Computed advantage value

        Returns:
            Tinker Datum object
        """
        prompt_tokens = trajectory["prompt_tokens"]
        response_tokens = trajectory["response_tokens"]
        logprobs = trajectory["logprobs"]

        # Combine prompt and response
        all_tokens = prompt_tokens + response_tokens
        input_tokens = all_tokens[:-1]
        target_tokens = all_tokens[1:]

        # Create masks (only train on response)
        ob_len = len(prompt_tokens) - 1
        all_logprobs = [0.0] * ob_len + logprobs
        all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
        all_mask = [0.0] * ob_len + [1.0] * (len(input_tokens) - ob_len)

        # Ensure all lists have the same length
        assert len(input_tokens) == len(target_tokens) == len(all_logprobs) == len(all_advantages) == len(all_mask), f"Length mismatch: input={len(input_tokens)}, target={len(target_tokens)}, logprobs={len(all_logprobs)}, advantages={len(all_advantages)}, mask={len(all_mask)}"

        # Create Datum
        datum = tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(tokens=[int(t) for t in input_tokens]),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                "mask": TensorData.from_torch(torch.tensor(all_mask)),
            },
        )

        return datum

    @staticmethod
    def build_datum_from_trajectory(trajectory: Trajectory, advantage: float) -> list[tinker.Datum]:
        """
        Build one or more Datums from a trajectory, merging steps when possible.

        Steps are merged when the next step's prompt is an extension of the
        previous step's full sequence (prompt + response).

        Args:
            trajectory: Trajectory with steps
            advantage: Advantage value for this trajectory

        Returns:
            List of Datum objects (may contain 1+ datums depending on merging)
        """
        if not trajectory.steps:
            return []

        # DEBUG: Check for data quality issues
        for step_idx, step in enumerate(trajectory.steps):
            # Check for None values in logprobs
            if step.logprobs and None in step.logprobs:
                logger.error(f"Step {step_idx} has None in logprobs: {step.logprobs}")
                raise ValueError(f"Step {step_idx} contains None in logprobs")

            # Check for non-integer values in prompt_ids
            if step.prompt_ids and not all(isinstance(x, int) for x in step.prompt_ids):
                logger.error(f"Step {step_idx} prompt_ids types: {[type(x) for x in step.prompt_ids[:5]]}")
                raise ValueError(f"Step {step_idx} prompt_ids contains non-integer values")

            # Check for non-integer values in response_ids
            if step.response_ids and not all(isinstance(x, int) for x in step.response_ids):
                logger.error(f"Step {step_idx} response_ids types: {[type(x) for x in step.response_ids[:5]]}")
                raise ValueError(f"Step {step_idx} response_ids contains non-integer values")

            # Check for mismatched lengths
            if len(step.response_ids) != len(step.logprobs):
                logger.error(f"Step {step_idx} length mismatch: {len(step.response_ids)} response_ids vs {len(step.logprobs)} logprobs")
                raise ValueError(f"Step {step_idx} has mismatched response_ids and logprobs lengths")

        # Accumulator for building merged sequences
        class SequenceAccumulator:
            def __init__(self):
                self.full_sequence = []
                self.logprobs = []
                self.advantages = []
                self.mask = []

            def is_empty(self):
                return len(self.full_sequence) == 0

            def clear(self):
                self.full_sequence = []
                self.logprobs = []
                self.advantages = []
                self.mask = []

            def add_step(self, step: Step, advantage: float, is_extension: bool = False):
                """Add a step to the accumulated sequence."""
                if is_extension:
                    # Only add the new tokens (delta)
                    prev_len = len(self.full_sequence)
                    delta_prompt = step.prompt_ids[prev_len:]
                    delta_prompt_len = len(delta_prompt)
                else:
                    # Add entire prompt
                    delta_prompt = step.prompt_ids
                    delta_prompt_len = len(delta_prompt)

                # Add prompt tokens (observation)
                self.full_sequence.extend(delta_prompt)
                self.logprobs.extend([0.0] * delta_prompt_len)
                self.advantages.extend([0.0] * delta_prompt_len)
                self.mask.extend([0.0] * delta_prompt_len)

                # Add response tokens (action)
                self.full_sequence.extend(step.response_ids)
                self.logprobs.extend(step.logprobs)
                self.advantages.extend([advantage] * len(step.response_ids))
                self.mask.extend([1.0] * len(step.response_ids))

            def to_datum(self) -> tinker.Datum:
                """Convert accumulated sequence to Datum."""
                if self.is_empty():
                    raise ValueError("Cannot create datum from empty sequence")

                # Create input/target pairs (shift by 1)
                input_tokens = self.full_sequence[:-1]
                target_tokens = self.full_sequence[1:]

                # Shift logprobs, advantages, mask to align with targets
                shifted_logprobs = self.logprobs[1:]
                shifted_advantages = self.advantages[1:]
                shifted_mask = self.mask[1:]

                assert len(input_tokens) == len(target_tokens) == len(shifted_logprobs) == len(shifted_advantages) == len(shifted_mask)

                return tinker.types.Datum(
                    model_input=tinker.types.ModelInput.from_ints(tokens=[int(t) for t in input_tokens]),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(shifted_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(shifted_advantages)),
                        "mask": TensorData.from_torch(torch.tensor(shifted_mask)),
                    },
                )

        # Build datums by iterating through steps
        datums = []
        accumulator = SequenceAccumulator()

        for step_idx, step in enumerate(trajectory.steps):
            if accumulator.is_empty():
                # First step - start accumulating
                accumulator.add_step(step, advantage, is_extension=False)
            else:
                # Check if current step extends previous sequence
                prev_full_sequence = accumulator.full_sequence
                current_prompt = step.prompt_ids

                if TinkerDatumBuilder._is_prefix(prev_full_sequence, current_prompt):
                    # Step extends previous - merge
                    accumulator.add_step(step, advantage, is_extension=True)
                else:
                    # Step doesn't extend - create datum and start fresh
                    datums.append(accumulator.to_datum())
                    accumulator.clear()
                    accumulator.add_step(step, advantage, is_extension=False)

        # Create final datum from accumulated sequence
        if not accumulator.is_empty():
            datums.append(accumulator.to_datum())

        return datums


def process_episodes(
    episodes: list,
    advantage_computer: TinkerAdvantageComputer,
    trajectory_filter: TinkerTrajectoryFilter,
    algorithm_config,
) -> tuple[list[tinker.Datum], dict]:
    """
    Main pipeline to convert Episode objects to training datums.

    This function:
    1. Groups trajectories based on grouping_level configuration
    2. Computes advantages for each group
    3. Builds Tinker Datums for training

    Grouping levels:
    - trajectory: Group trajectories by (task_id, trajectory_name) for multi-agent workflows.
                 Advantage computed across trajectory rewards.
    - step: Group individual steps at same position for step-level advantage computation.
    - episode: Each episode's trajectories form one group (simple single-agent case).

    Args:
        episodes: List of Episode objects
        advantage_computer: Computer for calculating advantages
        trajectory_filter: Filter for removing constant-reward groups
        algorithm_config: Configuration with grouping_level setting

    Returns:
        Tuple of (datums, metrics_dict):
        - datums: List of Tinker Datum objects ready for training
        - metrics_dict: Dictionary with grouping and advantage statistics
    """
    from collections import defaultdict

    import numpy as np

    grouping_level = algorithm_config.get("grouping_level", "episode")

    # Group trajectories based on grouping_level
    trajectory_groups_dict = defaultdict(list)

    def get_task_id(episode):
        """Extract task_id from episode.id (format: task_id:rollout_idx)"""
        return ":".join(episode.id.split(":")[:-1]) if ":" in episode.id else episode.id

    if grouping_level == "trajectory":
        # Group by (task_id, trajectory_name) - for multi-agent workflows like solver-judge
        for episode in episodes:
            task_id = get_task_id(episode)
            for trajectory in episode.trajectories:
                group_key = (task_id, trajectory.name)
                trajectory_groups_dict[group_key].append(trajectory)

    elif grouping_level == "step":
        # Group by (task_id, trajectory_name, step_idx) - for step-level advantages
        for episode in episodes:
            task_id = get_task_id(episode)
            for trajectory in episode.trajectories:
                for step_idx, step in enumerate(trajectory.steps):
                    group_key = (task_id, trajectory.name, step_idx)
                    # Create single-step trajectory
                    from rllm.agents.agent import Trajectory

                    single_step_traj = Trajectory(steps=[step], reward=step.reward, name=trajectory.name)
                    trajectory_groups_dict[group_key].append(single_step_traj)

    else:  # "episode" or default
        # Simple grouping: all trajectories in an episode form one group
        for episode in episodes:
            group_key = episode.id
            trajectory_groups_dict[group_key].extend(episode.trajectories)

    # Convert dict to list of TrajectoryGroup objects for filtering
    trajectory_groups = [TrajectoryGroup(trajectories=trajs, group_id=str(key)) for key, trajs in trajectory_groups_dict.items()]

    # Apply filtering based on configuration
    filtered_groups = trajectory_filter.filter_groups(trajectory_groups)

    # Track metrics
    all_advantages = []
    group_sizes = []

    training_datums = []
    for group in filtered_groups:
        # Extract rewards for the group (from all trajectories)
        group_rewards = [traj.reward for traj in group.trajectories]

        # Compute advantages
        advantages = advantage_computer.compute(group_rewards)

        # Track for metrics
        all_advantages.extend(advantages)
        group_sizes.append(len(group.trajectories))

        # Create datums for all trajectories in the group
        for trajectory, advantage in zip(group.trajectories, advantages, strict=False):
            # Use trajectory-level building (merges steps when possible)
            new_datums = TinkerDatumBuilder.build_datum_from_trajectory(trajectory, advantage)
            training_datums.extend(new_datums)

    # Compute grouping and advantage metrics
    metrics = {}
    if filtered_groups:
        metrics["grouping/num_groups"] = len(filtered_groups)
        metrics["grouping/num_groups_before_filter"] = len(trajectory_groups)
        metrics["grouping/avg_group_size"] = np.mean(group_sizes)
        metrics["grouping/max_group_size"] = np.max(group_sizes)
        metrics["grouping/min_group_size"] = np.min(group_sizes)

    if all_advantages:
        metrics["advantage/mean"] = np.mean(all_advantages)
        metrics["advantage/std"] = np.std(all_advantages)
        metrics["advantage/max"] = np.max(all_advantages)
        metrics["advantage/min"] = np.min(all_advantages)
        metrics["advantage/fraction_zero"] = np.sum(np.abs(all_advantages) < 1e-8) / len(all_advantages)

    return training_datums, metrics


def process_trajectory_groups(
    groups: list[TrajectoryGroup],
    advantage_computer: TinkerAdvantageComputer,
    trajectory_filter: TinkerTrajectoryFilter,
) -> list[tinker.Datum]:
    """
    Main pipeline to convert TrajectoryGroup objects to training datums.

    This function:
    1. Filters groups (if configured)
    2. Computes advantages for each group
    3. Builds Tinker Datums for training

    TrajectoryGroup structure depends on grouping_level (set in regroup()):
    - trajectory-level: Each group contains multiple complete trajectories (with all steps).
                       Advantages are computed across trajectory rewards, then broadcast
                       to all steps in each trajectory.
    - step-level: Each group contains multiple single-step trajectories.
                 Advantages are computed across step rewards.

    Args:
        groups: List of TrajectoryGroup objects (created by regroup())
        advantage_computer: Computer for calculating advantages
        trajectory_filter: Filter for removing constant-reward groups

    Returns:
        List of Tinker Datum objects ready for training
    """
    # Apply filtering based on configuration
    filtered_groups = trajectory_filter.filter_groups(groups)

    training_datums = []
    for group in filtered_groups:
        # Extract rewards for the group (from all trajectories)
        group_rewards = [traj.reward for traj in group.trajectories]

        # Compute advantages
        advantages = advantage_computer.compute(group_rewards)

        # Create datums for all trajectories in the group
        for trajectory, advantage in zip(group.trajectories, advantages, strict=False):
            # Use trajectory-level building (merges steps when possible)
            new_datums = TinkerDatumBuilder.build_datum_from_trajectory(trajectory, advantage)
            training_datums.extend(new_datums)

    return training_datums
