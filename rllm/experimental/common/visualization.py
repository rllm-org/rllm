import warnings
from dataclasses import dataclass, field
from typing import Any

import click

from rllm.agents.agent import Trajectory, TrajectoryGroup


@dataclass
class VisualizationConfig:
    """Configuration for trajectory visualization colors and styles."""

    # Colors for different elements
    masked_token_style: dict[str, Any] = field(default_factory=lambda: {"dim": True})
    unmasked_token_style: dict[str, Any] = field(default_factory=lambda: {"fg": "blue"})
    reward_pos_style: dict[str, Any] = field(default_factory=lambda: {"bg": "green"})
    reward_neg_style: dict[str, Any] = field(default_factory=lambda: {"bg": "red"})

    header_style: dict[str, Any] = field(default_factory=lambda: {"fg": "cyan", "bold": True})
    label_style: dict[str, Any] = field(default_factory=lambda: {"fg": "yellow"})

    success_style: dict[str, Any] = field(default_factory=lambda: {"fg": "green", "bold": True})
    failure_style: dict[str, Any] = field(default_factory=lambda: {"fg": "red", "bold": True})


def colorful_print(string: str, *args, **kwargs) -> None:
    end = kwargs.pop("end", "\n")
    print(click.style(string, *args, **kwargs), end=end, flush=True)


def colorful_warning(string: str, *args, **kwargs) -> None:
    warnings.warn(click.style(string, *args, **kwargs), stacklevel=2)


def abbreviate_string(full_string: str, max_length: int = 512):
    """
    Abbreviate a string to a maximum length, and mention how many characters are skipped in between.
    Note that the "...(skipped X characters)" is only a display hint and the exact skipped length is not guaranteed.
    """
    full_string = full_string.strip()
    if len(full_string) <= max_length:
        return full_string
    # we first use space to split the string into words
    words = full_string.split(" ")

    truncate_goal = len(full_string) - max_length

    if len(words) <= 2:  # no way for our downstream algorithm to handle. Simply return the last max_length characters
        return f"(skipped first {truncate_goal} characters)..." + full_string[-max_length:]

    # we try to retain the both ends and truncate the middle
    left_idx, right_idx = 0, len(words) - 1
    truncate_length = len(full_string)
    while left_idx <= right_idx:
        truncate_length = truncate_length - len(words[left_idx]) - len(words[right_idx]) - 2
        if truncate_length <= truncate_goal:
            break
        left_idx += 1
        right_idx -= 1

    if left_idx > right_idx:
        return f"(skipped first {truncate_goal} characters)..." + full_string[-max_length:]
    else:
        left_str = " ".join(words[:left_idx])
        right_str = " ".join(words[right_idx + 1 :])
        skip_str = f"(skipped middle {truncate_length} characters)"
        return left_str + "..." + skip_str + "..." + right_str


def _format_token(token: str, style: dict[str, Any]) -> str:
    """Formats a token string with the given click style."""
    # Escape special characters for display
    display_token = token.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return click.style(display_token, **style)


def _visualize_metadata(trajectory: Trajectory, metadata: dict, config: VisualizationConfig):
    """
    Visualizes workflow metadata for a given trajectory.
    """
    header_parts = []
    if "task_id" in metadata and "rollout_idx" in metadata:
        header_parts.append(f"Task ID: {metadata['task_id']}")
        header_parts.append(f"Rollout: #{metadata['rollout_idx']}")
    header_parts.append(f"Trajectory: {trajectory.name}")
    colorful_print(" | ".join(header_parts), **config.header_style)

    if "is_correct" in metadata:
        is_correct = metadata["is_correct"]
        style = config.success_style if is_correct else config.failure_style
        colorful_print(f"Episode Outcome: {'✓ Correct' if is_correct else '✗ Incorrect'}", **style)
    if "termination_reason" in metadata:
        termination_reason = metadata["termination_reason"]
        colorful_print(f"Termination Status: {termination_reason}", **config.label_style)


def visualize_trajectory_last_steps(
    groups: list[TrajectoryGroup],
    tokenizer,
    max_steps_to_visualize: int = 1,
    show_workflow_metadata: bool = True,
    config: VisualizationConfig | None = None,
):
    """
    Visualizes some last steps of trajectories in different trajectory groups.
    Let `g` be the number of trajectory groups. If `g` is greater than `max_steps_to_visualize`,
    we will first sample `max_steps_to_visualize` trajectory groups.

    For each of the trajectory group, we will randomly sample a trajectory and visualize its last step.

    Args:
        groups: List of trajectory groups.
        tokenizer: Tokenizer for decoding.
        max_steps_to_visualize: Maximum number of steps to visualize.
        show_workflow_metadata: Whether to show workflow metadata (episode_ids, trajectory_ids, etc.).
        config: VisualizationConfig for colors.
    """
    if config is None:
        config = VisualizationConfig()

    import random

    sampled_groups = random.sample(groups, k=min(max_steps_to_visualize, len(groups)))

    legends = []
    legends.append(_format_token("masked", config.masked_token_style))
    legends.append(_format_token("unmasked", config.unmasked_token_style))
    legends.append(_format_token("reward > 0", config.reward_pos_style))
    legends.append(_format_token("reward <= 0", config.reward_neg_style))

    legend_str = " ".join(legends)

    # visualize the last step of one trajectory in each sampled group
    for group in sampled_groups:
        trajectory_idx = random.randint(0, len(group.trajectories) - 1)
        trajectory = group.trajectories[trajectory_idx]
        if len(trajectory.steps) == 0:
            continue

        # 1. [Optional] Print Workflow Metadata
        if show_workflow_metadata:
            _visualize_metadata(trajectory, group.metadata[trajectory_idx], config)

        # 2. Print Legend (simplified)
        print(f"[{legend_str}]")

        # 3. Render Prompt and Response
        last_step = trajectory.steps[-1]
        # extract the prompt and response
        prompt_ids = last_step.prompt_ids
        response_ids = last_step.response_ids

        # special handling for prompt ids, we will skip any non-int elements
        clean_prompt_ids = [elem for elem in prompt_ids if isinstance(elem, int)]
        if len(clean_prompt_ids) != len(prompt_ids):
            colorful_warning(f"During visualization, skipped {len(prompt_ids) - len(clean_prompt_ids)} non-int elements in prompt_ids.")

        prompt_str = abbreviate_string(tokenizer.decode(clean_prompt_ids))

        print(_format_token(prompt_str, config.masked_token_style))
        print("----------------")

        # for response string, we simply highlight the last token
        response_str_prev = abbreviate_string(tokenizer.decode(response_ids[:-1]))
        response_str_last = tokenizer.decode([response_ids[-1]])

        response_style = config.unmasked_token_style
        response_str_prev = _format_token(response_str_prev, response_style)

        reward_style = config.reward_pos_style if last_step.reward > 0 else config.reward_neg_style
        last_token_style = response_style | reward_style
        last_token_str = _format_token(response_str_last, last_token_style)
        print(response_str_prev + last_token_str)
