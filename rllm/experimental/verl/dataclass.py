from dataclasses import dataclass, field

import torch


@dataclass
class ProcessedStepData:
    """Represents one tokenized step/trajectory unit ready for batching.

    This is the atomic unit that will become one row in the final batch.
    In non-stepwise mode, this represents an entire trajectory.
    In stepwise mode, this represents a single step within a trajectory.
    """

    prompt: torch.Tensor
    response: torch.Tensor
    mask: torch.Tensor
    step_reward: float
    step_id: str
    multi_modal_inputs: dict = field(default_factory=dict)  # Optional multimodal inputs (e.g., image_grid_thw for Qwen-VL)
    advantage: float | None = None


@dataclass
class AccumulatedData:
    """Container for all accumulated lists during episode processing.

    Each list has one entry per ProcessedStepData (i.e., per batch row).
    """

    # Tensor data (to be batched)
    prompts: list[torch.Tensor] = field(default_factory=list)
    responses: list[torch.Tensor] = field(default_factory=list)
    traj_mask: list[torch.Tensor] = field(default_factory=list)

    # Reward data (parallel to tensor lists)
    traj_rewards: list[float] = field(default_factory=list)
    step_rewards: list[float] = field(default_factory=list)

    # ID tracking (parallel to tensor lists)
    trajectory_ids: list[str] = field(default_factory=list)
    step_ids: list[str] = field(default_factory=list)

    # Metadata (parallel to tensor lists)
    step_nums: list[int] = field(default_factory=list)  # number of steps in the trajectory
    is_last_step: list[bool] = field(default_factory=list)

    # Multimodal data (parallel to tensor lists)
    multi_modal_inputs: list[dict] = field(default_factory=list)  # Optional multimodal inputs per step

    # Episode-level tracking
    repeat_counts: list[int] = field(default_factory=list)  # number of batch rows per episode

    # Advantage data (not None if stepwise advantages are already computed)
    advantages: list[float] = field(default_factory=list)

    def add_step(self, step_data: ProcessedStepData, trajectory_id: str, traj_reward: float, step_num: int, is_last: bool):
        """Add a single processed step to all accumulator lists.

        This helper ensures all lists stay in sync and reduces boilerplate.
        """
        self.prompts.append(step_data.prompt)
        self.responses.append(step_data.response)
        self.traj_mask.append(step_data.mask)
        self.step_rewards.append(step_data.step_reward)
        self.traj_rewards.append(traj_reward)
        self.step_ids.append(step_data.step_id)

        if step_data.advantage is not None:  # make sure to not add None to the list
            self.advantages.append(step_data.advantage)

        self.trajectory_ids.append(trajectory_id)
        self.step_nums.append(step_num)
        self.is_last_step.append(is_last)
        self.multi_modal_inputs.append(step_data.multi_modal_inputs)

    def __len__(self) -> int:
        """Return the total number of batch rows accumulated."""
        return len(self.prompts)
