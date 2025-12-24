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
    Computes advantages using REINFORCE, GRPO, or distillation.
    Compatible with rLLM's existing advantage computation.
    """

    def __init__(self, algorithm_config, teacher_engine=None, student_tokenizer=None, teacher_tokenizer=None):
        self.adv_estimator = algorithm_config.adv_estimator
        self.gamma = algorithm_config.gamma
        self.norm_by_std = algorithm_config.get("norm_adv_by_std_in_grpo", True)
        self.distill_enabled = self.adv_estimator == "distill"
        self.teacher_engine = teacher_engine
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.shared_tokenizer = algorithm_config.get("shared_tokenizer", False)

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

    def compute_distill_advantages_per_token(
        self,
        student_logprobs: list[float],
        teacher_logprobs: list[float],
    ) -> list[float]:
        """
        Compute per-token advantages for distillation: advantage = teacher_logprob - student_logprob.

        Args:
            student_logprobs: Student logprobs for each completion token
            teacher_logprobs: Aligned teacher logprobs for each completion token (same length as student)

        Returns:
            List of per-token advantages
        """
        if len(student_logprobs) != len(teacher_logprobs):
            raise ValueError(f"Length mismatch: student_logprobs={len(student_logprobs)}, teacher_logprobs={len(teacher_logprobs)}")

        advantages = [t_lp - s_lp for t_lp, s_lp in zip(teacher_logprobs, student_logprobs, strict=False)]
        return advantages


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
    def build_distillation_datum(
        prompt_ids: list[int],
        response_ids: list[int],
        logprobs: list[float],
        advantages: list[float],
    ) -> tinker.Datum:
        """
        Build a Datum for distillation from raw data (no trajectory needed).

        Args:
            prompt_ids: Prompt token IDs
            response_ids: Response token IDs (completion)
            logprobs: Student logprobs for response tokens
            advantages: Per-token advantages (teacher_logprob - student_logprob)

        Returns:
            A single tinker.Datum
        """
        if len(response_ids) != len(logprobs):
            raise ValueError(f"Length mismatch: {len(response_ids)} response_ids vs {len(logprobs)} logprobs")
        if len(response_ids) != len(advantages):
            raise ValueError(f"Length mismatch: {len(response_ids)} response_ids vs {len(advantages)} advantages")

        # Build full sequence
        full_sequence = list(prompt_ids) + list(response_ids)

        # Logprobs/advantages: 0 for prompt tokens, actual values for response
        full_logprobs = [0.0] * len(prompt_ids) + list(logprobs)
        full_advantages = [0.0] * len(prompt_ids) + list(advantages)
        full_mask = [0.0] * len(prompt_ids) + [1.0] * len(response_ids)

        # Create input/target pairs (shift by 1)
        input_tokens = full_sequence[:-1]
        target_tokens = full_sequence[1:]

        # Shift to align with targets
        shifted_logprobs = full_logprobs[1:]
        shifted_advantages = full_advantages[1:]
        shifted_mask = full_mask[1:]

        return tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(tokens=[int(t) for t in input_tokens]),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                "logprobs": TensorData.from_torch(torch.tensor(shifted_logprobs)),
                "advantages": TensorData.from_torch(torch.tensor(shifted_advantages)),
                "mask": TensorData.from_torch(torch.tensor(shifted_mask)),
            },
        )

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
            if not step.prompt_ids and step.model_output and step.model_output.prompt_ids:
                step.prompt_ids = step.model_output.prompt_ids
            if not step.response_ids and step.model_output and step.model_output.completion_ids:
                step.response_ids = step.model_output.completion_ids
            if not step.logprobs and step.model_output and step.model_output.logprobs:
                step.logprobs = step.model_output.logprobs

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


async def process_episodes(
    episodes: list,
    advantage_computer: TinkerAdvantageComputer,
    trajectory_filter: TinkerTrajectoryFilter,
    algorithm_config,
) -> tuple[list[tinker.Datum], dict]:
    """
    Main pipeline to convert Episode objects to training datums.

    This function:
    1. For distillation: Flattens episodes → list of trajectories (skips grouping/filtering)
    2. For RL (GRPO/REINFORCE): Groups trajectories based on grouping_level configuration
    3. Computes advantages for each trajectory/group
    4. Builds Tinker Datums for training

    Distillation mode:
    - Skips grouping and filtering (advantages computed per-token from teacher/student logprobs)
    - Processes all trajectories independently

    RL mode grouping levels:
    - trajectory: Group trajectories by (task_id, trajectory_name) for multi-agent workflows.
                 Advantage computed across trajectory rewards.
    - step: Group individual steps at same position for step-level advantage computation.
    - episode: Each episode's trajectories form one group (simple single-agent case).

    Args:
        episodes: List of Episode objects
        advantage_computer: Computer for calculating advantages
        trajectory_filter: Filter for removing constant-reward groups (only used in RL mode)
        algorithm_config: Configuration with grouping_level setting (only used in RL mode)

    Returns:
        Tuple of (datums, metrics_dict):
        - datums: List of Tinker Datum objects ready for training
        - metrics_dict: Dictionary with grouping and advantage statistics
    """
    from collections import defaultdict

    import numpy as np

    # Track metrics
    all_advantages = []
    group_sizes = []

    training_datums = []

    if advantage_computer.distill_enabled:
        import asyncio

        teacher_chat_parser = None
        if not advantage_computer.shared_tokenizer:
            if not hasattr(advantage_computer.teacher_engine, "chat_parser") or advantage_computer.teacher_engine.chat_parser is None:
                raise ValueError("Teacher engine does not have a chat_parser.")
            teacher_chat_parser = advantage_computer.teacher_engine.chat_parser

        async def process_step(global_idx: int, step: Step) -> tuple[tinker.Datum, float]:
            """
            Process a single step: query teacher, align, compute advantages, build datum.

            Args:
                global_idx: Global step index (for visualization on idx=0)
                step: The step to process

            Returns:
                Tuple of (datum, avg_advantage)
            """
            if not step.model_output or not step.model_output.completion_ids or not step.model_output.logprobs:
                raise ValueError("Missing model_output with completion_ids or logprobs for distillation.")

            prompt_ids = step.prompt_ids or step.model_output.prompt_ids
            if not prompt_ids:
                raise ValueError("Missing prompt_ids in step for distillation.")

            student_completion_ids = step.model_output.completion_ids
            student_logprobs = step.model_output.logprobs

            if advantage_computer.shared_tokenizer:
                # Fast path: student and teacher use the same tokenizer
                # Directly use student token IDs for teacher query
                student_prompt_ids = prompt_ids
                teacher_ids = student_prompt_ids + student_completion_ids
                teacher_prompt_length = len(student_prompt_ids)

                teacher_resp = await advantage_computer.teacher_engine.completion(
                    teacher_ids,
                    max_tokens=1,
                    extra_body={"prompt_logprobs": 1},
                )
                if not teacher_resp.prompt_logprobs:
                    raise ValueError("Teacher missing prompt_logprobs.")

                aligned_teacher_logprobs = teacher_resp.prompt_logprobs[teacher_prompt_length:]

            else:
                # Slow path: different tokenizers, need byte-level alignment
                from rllm.trainer.distill import align_teacher_logprobs

                if not step.chat_completions:
                    raise ValueError("Missing chat_completions in step for distillation.")

                teacher_prompt_messages = step.chat_completions[:-1]
                teacher_completion_messages = step.chat_completions[-1:]

                reasoning_str = teacher_completion_messages[0].get("reasoning", "")
                content_str = teacher_completion_messages[0].get("content", "")
                if not reasoning_str and not content_str:
                    raise ValueError("Missing both reasoning and content in teacher completion message.")

                # Build teacher prompt and completion
                teacher_prompt = teacher_chat_parser.parse(
                    teacher_prompt_messages,
                    is_first_msg=True,
                    add_generation_prompt=True,
                    tools=[],
                    accumulate_reasoning=False,
                )
                teacher_prompt_ids = advantage_computer.teacher_tokenizer.encode(teacher_prompt, add_special_tokens=False)

                teacher_completion = teacher_chat_parser.parse(
                    teacher_completion_messages,
                    is_first_msg=False,
                    add_generation_prompt=False,
                    tools=[],
                    accumulate_reasoning=True,
                )
                if teacher_completion.startswith(teacher_chat_parser.generation_prompt):
                    teacher_completion = teacher_completion[len(teacher_chat_parser.generation_prompt) :]

                # Query teacher for logprobs (async)
                teacher_full_prompt = teacher_prompt + teacher_completion
                teacher_resp = await advantage_computer.teacher_engine.completion(
                    teacher_full_prompt,
                    max_tokens=1,
                    extra_body={"prompt_logprobs": 1},
                )
                if not teacher_resp.prompt_logprobs:
                    raise ValueError("Teacher missing prompt_logprobs.")

                teacher_prompt_len = len(teacher_prompt_ids)
                teacher_completion_ids = teacher_resp.prompt_ids[teacher_prompt_len:]
                teacher_logprobs = teacher_resp.prompt_logprobs[teacher_prompt_len:]

                # Align teacher logprobs to student tokens
                aligned_teacher_logprobs = align_teacher_logprobs(
                    student_ids=student_completion_ids,
                    student_tokenizer=advantage_computer.student_tokenizer,
                    teacher_ids=teacher_completion_ids,
                    teacher_tokenizer=advantage_computer.teacher_tokenizer,
                    teacher_logprobs=teacher_logprobs,
                    student_logprobs=student_logprobs,
                    reasoning_str=reasoning_str,
                    content_str=content_str,
                )

                # Visualize first step
                if global_idx == 0:
                    from rllm.trainer.distill import visualize_alignment

                    visualize_alignment(
                        student_ids=student_completion_ids,
                        student_tokenizer=advantage_computer.student_tokenizer,
                        teacher_ids=teacher_completion_ids,
                        teacher_tokenizer=advantage_computer.teacher_tokenizer,
                        teacher_logprobs=teacher_logprobs,
                        student_logprobs=student_logprobs,
                        reasoning_str=reasoning_str,
                        content_str=content_str,
                        max_tokens=150,
                    )

            # Compute per-token advantages
            advantages = advantage_computer.compute_distill_advantages_per_token(
                student_logprobs,
                aligned_teacher_logprobs,
            )

            datum = TinkerDatumBuilder.build_distillation_datum(
                prompt_ids=prompt_ids,
                response_ids=student_completion_ids,
                logprobs=student_logprobs,
                advantages=advantages,
            )

            avg_advantage = float(np.mean(advantages)) if advantages else 0.0
            return datum, avg_advantage

        # Collect all steps
        all_steps: list[Step] = []
        for episode in episodes:
            for trajectory in episode.trajectories:
                for step in trajectory.steps:
                    all_steps.append(step)

        import time

        start_time = time.time()
        print(f"[Distillation] Processing {len(all_steps)} steps in parallel...")

        # Process all steps in parallel
        tasks = [process_step(idx, step) for idx, step in enumerate(all_steps)]
        results = await asyncio.gather(*tasks)

        print(f"[Distillation] Processing complete in {time.time() - start_time:.2f} seconds.")

        # Collect results
        for datum, avg_advantage in results:
            training_datums.append(datum)
            all_advantages.append(avg_advantage)
    else:
        # Standard RL mode: group trajectories and compute advantages from rewards
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

        for group in filtered_groups:
            # Standard RL mode: compute scalar advantages from rewards
            group_rewards = [traj.reward for traj in group.trajectories]
            advantages = advantage_computer.compute(group_rewards)

            # Track for metrics
            all_advantages.extend(advantages)
            group_sizes.append(len(group.trajectories))

            # Create datums for all trajectories in the group
            for trajectory, advantage in zip(group.trajectories, advantages, strict=False):
                new_datums = TinkerDatumBuilder.build_datum_from_trajectory(trajectory, advantage)
                training_datums.extend(new_datums)

    # Compute grouping and advantage metrics
    metrics = {}
    if not advantage_computer.distill_enabled and filtered_groups:
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


# TODO: is this dead code?
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
