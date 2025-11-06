"""Data processing utilities for converting trajectories to training data.

This module provides the bridge between trajectory generation and training,
handling filtering, advantage computation, and conversion to tinker.Datum format.
"""

import logging

import numpy as np
import tinker
import torch
from tinker.types.tensor_data import TensorData

from rllm.agents.agent import Episode, Step, Trajectory

logger = logging.getLogger(__name__)


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

    def filter_episodes(self, episodes: list[Episode]) -> list[Episode]:
        """
        Filter episodes based on configuration.

        If remove_constant_reward_groups=True, removes episodes where all trajectories
        have the same reward. If all episodes would be removed, keeps at least one
        episode to prevent empty batches.

        Args:
            episodes: List of Episode objects

        Returns:
            Filtered list of Episode objects
        """
        if not self.remove_constant_reward_groups:
            # Keep all episodes (default behavior)
            return episodes

        # Filter out constant-reward episodes
        filtered_episodes = []
        for episode in episodes:
            # Get rewards from all trajectories in the episode
            episode_rewards = [traj.reward for traj in episode.trajectories]
            if not self._all_same(episode_rewards):
                filtered_episodes.append(episode)

        # Safety: Never return empty list to prevent batch size issues
        if not filtered_episodes:
            logger.warning("All episodes have uniform rewards. There will be no gradient. Keeping one episode to prevent empty batch.")
            return episodes[:1]

        if len(filtered_episodes) < len(episodes):
            logger.info(f"Filtered {len(episodes) - len(filtered_episodes)} constant-reward episodes (kept {len(filtered_episodes)} episodes with reward variance)")

        return filtered_episodes


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
    episodes: list[Episode],
    advantage_computer: TinkerAdvantageComputer,
    trajectory_filter: TinkerTrajectoryFilter,
) -> list[tinker.Datum]:
    """
    Main pipeline to convert Episode objects to training datums.

    This function:
    1. Filters episodes (if configured)
    2. Computes advantages for each episode
    3. Builds Tinker Datums for training

    Args:
        episodes: List of Episode objects
        advantage_computer: Computer for calculating advantages
        trajectory_filter: Filter for removing constant-reward episodes

    Returns:
        List of Tinker Datum objects ready for training
    """
    # Apply filtering based on configuration
    filtered_episodes = trajectory_filter.filter_episodes(episodes)

    training_datums = []
    for episode in filtered_episodes:
        # Extract rewards for the episode (from all trajectories)
        episode_rewards = [traj.reward for traj in episode.trajectories]

        # Compute advantages
        advantages = advantage_computer.compute(episode_rewards)

        # Create datums for all trajectories in the episode
        for trajectory, advantage in zip(episode.trajectories, advantages, strict=False):
            # Use trajectory-level building (merges steps when possible)
            new_datums = TinkerDatumBuilder.build_datum_from_trajectory(trajectory, advantage)
            training_datums.extend(new_datums)

    return training_datums
