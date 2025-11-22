from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from verl import DataProto


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


def _format_token(token: str, style: dict[str, Any]) -> str:
    """Formats a token string with the given click style."""
    # Escape special characters for display
    display_token = token.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return click.style(display_token, **style)


def _visualize_metadata(batch: DataProto, sample_idx: int, config: VisualizationConfig):
    """
    Visualizes workflow metadata for a given sample index.
    """
    header_parts = []
    if "episode_ids" in batch.non_tensor_batch:
        eid = batch.non_tensor_batch["episode_ids"][sample_idx]
        header_parts.append(f"Episode: {eid}")
    if "trajectory_ids" in batch.non_tensor_batch:
        tid = batch.non_tensor_batch["trajectory_ids"][sample_idx]
        header_parts.append(f"Trajectory: {tid}")

    colorful_print(" | ".join(header_parts), **config.header_style)

    if "is_correct" in batch.non_tensor_batch:
        is_correct = batch.non_tensor_batch["is_correct"][sample_idx]
        style = config.success_style if is_correct else config.failure_style
        colorful_print(f"Outcome: {'✓ Correct' if is_correct else '✗ Incorrect'}", **style)
    if "termination_reasons" in batch.non_tensor_batch:
        termination_reason = batch.non_tensor_batch["termination_reasons"][sample_idx]
        colorful_print(f"Termination: {termination_reason}", **config.label_style)


def colorful_print(string: str, *args, **kwargs) -> None:
    end = kwargs.pop("end", "\n")
    print(click.style(string, *args, **kwargs), end=end, flush=True)


def colorful_warning(string: str, *args, **kwargs) -> None:
    warnings.warn(click.style(string, *args, **kwargs), stacklevel=2)


def visualize_trajectories(
    batch: DataProto,
    tokenizer,
    sample_indices: list[int],
    mask_key: str = "response_mask",
    reward_key: str = "token_level_scores",
    show_workflow_metadata: bool = True,
    config: VisualizationConfig | None = None,
):
    """
    Visualizes trajectories from a DataProto batch.

    Args:
        batch: The DataProto batch containing trajectory data.
        tokenizer: Tokenizer for decoding.
        sample_indices: Specific indices to visualize. Overrides num_samples.
        mask_key: Key for the response mask (loss mask) in the batch.
        reward_key: Key for token-level scores/rewards.
        show_workflow_metadata: Whether to show workflow metadata (episode_ids, trajectory_ids, etc.).
        config: VisualizationConfig for colors.
    """
    if config is None:
        config = VisualizationConfig()

    # Extract Tensors
    prompts = batch.batch["prompts"]
    responses = batch.batch["responses"]
    full_attn_mask = batch.batch["attention_mask"]

    # Optional Tensors
    masks = batch.batch.get(mask_key)
    rewards_tensor = batch.batch.get(reward_key)

    # Dimensions
    prompt_len = prompts.shape[1]
    resp_len = responses.shape[1]

    prompt_attn_mask = full_attn_mask[:, :prompt_len]
    response_attn_mask = full_attn_mask[:, -resp_len:]

    batch_size = prompts.shape[0]
    for idx in sample_indices:
        if idx >= batch_size:
            continue

        colorful_print("\n" + "=" * 60 + f"\nSample {idx}", **config.header_style)

        # 1. [Optional] Print Workflow Metadata
        if show_workflow_metadata:
            _visualize_metadata(batch, idx, config)

        # 2. Print Legend (simplified)
        legend = " ".join([_format_token("masked", config.masked_token_style), _format_token("unmasked", config.unmasked_token_style), _format_token("reward > 0", config.reward_pos_style), _format_token("reward <= 0", config.reward_neg_style)])
        print(f"[{legend}]")

        # 3. Render Prompt
        prompt_tokens = prompts[idx]
        prompt_valid = prompt_attn_mask[idx].bool()

        prompt_str_parts = []
        for t_id, is_valid in zip(prompt_tokens.tolist(), prompt_valid.tolist(), strict=False):
            if not is_valid:
                continue
            token_str = tokenizer.decode([t_id])
            prompt_str_parts.append(_format_token(token_str, config.masked_token_style))

        print("".join(prompt_str_parts))
        print("----------------")

        # 4. Render Response
        resp_tokens = responses[idx]
        resp_valid = response_attn_mask[idx].bool()
        loss_mask = masks[idx] if masks is not None else resp_valid
        token_rewards = rewards_tensor[idx] if rewards_tensor is not None else None

        # Determine reward highlight position
        reward_idx = None
        reward_val = 0.0

        if token_rewards is not None:
            # Find last valid token with non-zero reward or just last valid token
            valid_indices = [i for i, v in enumerate(resp_valid.tolist()) if v]

            if valid_indices:
                # Try to find non-zero reward
                for i in valid_indices:
                    val = float(token_rewards[i].item()) if hasattr(token_rewards[i], "item") else float(token_rewards[i])
                    if abs(val) > 1e-9:
                        reward_idx = i
                        reward_val = val

                # Fallback to last valid token if no reward found
                if reward_idx is None:
                    reward_idx = valid_indices[-1]
                    val = float(token_rewards[reward_idx].item()) if hasattr(token_rewards[reward_idx], "item") else float(token_rewards[reward_idx])
                    reward_val = val

        response_str_parts = []
        for i, t_id in enumerate(resp_tokens.tolist()):
            if not bool(resp_valid[i].item() if hasattr(resp_valid[i], "item") else resp_valid[i]):
                continue

            token_str = tokenizer.decode([t_id])

            # Determine Style
            is_in_loss = bool(loss_mask[i].item()) if hasattr(loss_mask[i], "item") else bool(loss_mask[i])

            style = config.unmasked_token_style if is_in_loss else config.masked_token_style

            # Update background color for reward token
            if reward_idx is not None and i == reward_idx:
                style.update(config.reward_pos_style if reward_val > 0 else config.reward_neg_style)

            response_str_parts.append(_format_token(token_str, style))

        print("".join(response_str_parts))
