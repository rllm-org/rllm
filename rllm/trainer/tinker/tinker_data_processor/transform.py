import numpy as np
import tinker
from tinker.types.tensor_data import TensorData

from rllm.agents.agent import Step, TrajectoryGroup
from rllm.trainer.common import AlgorithmConfig, compute_advantage_from_trajectory_groups


def _validate_and_build_datum(all_tokens: list[int], prompt_tokens: list[int], logprobs: list[float], advantage: float) -> tinker.Datum:
    """
    Helper function to validate and build a Tinker Datum using numpy.
    """
    input_tokens = all_tokens[:-1]  # no need to convert to numpy array
    target_tokens = np.array(all_tokens[1:], dtype=np.int32)
    # Create `all_logprobs` by padding `logprobs` with `prompt_tokens` - 1 length zeros to the left
    ob_len = len(prompt_tokens) - 1
    all_logprobs = np.pad(logprobs, (ob_len, 0), mode="constant", constant_values=0.0)
    # Create `all_advantages` by concatenating `ob_len` zeros with `advantage` for the response length
    all_advantages = np.concatenate([np.zeros(ob_len, dtype=np.float32), np.full(len(input_tokens) - ob_len, advantage, dtype=np.float32)])
    # Create `all_mask` by concatenating `ob_len` zeros and `len(input_tokens) - ob_len` ones
    all_mask = np.concatenate([np.zeros(ob_len, dtype=np.float32), np.full(len(input_tokens) - ob_len, 1.0, dtype=np.float32)])

    # Validate that all arrays have the same length
    assert len(input_tokens) == len(target_tokens) == len(all_logprobs) == len(all_advantages) == len(all_mask), f"Length mismatch: input={len(input_tokens)}, target={len(target_tokens)}, logprobs={len(all_logprobs)}, advantages={len(all_advantages)}, mask={len(all_mask)}"

    # Create Datum
    return tinker.types.Datum(
        model_input=tinker.types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_numpy(target_tokens),
            "logprobs": TensorData.from_numpy(all_logprobs),
            "advantages": TensorData.from_numpy(all_advantages),
            "mask": TensorData.from_numpy(all_mask),
        },
    )


def build_datum_from_step(step: Step) -> tinker.Datum:
    """Create a Tinker Datum from a Step object.

    Args:
        step: Step object with prompt_ids, response_ids, logprobs, and advantage filled in

    Returns:
        Tinker Datum object
    """
    assert step.advantage is not None, "step.advantage is None. This indicates that advantage computation has not been performed yet."

    prompt_tokens = step.prompt_ids
    response_tokens = step.response_ids

    # Combine prompt and response
    all_tokens = prompt_tokens + response_tokens
    return _validate_and_build_datum(all_tokens, prompt_tokens, step.logprobs, step.advantage)


def transform_trajectory_groups_to_datums(
    trajectory_groups: list[TrajectoryGroup],
    algorithm_config: AlgorithmConfig,
) -> list[tinker.Datum]:
    """
    Transform a list of TrajectoryGroup objects to a list of Tinker Datum objects. Two things are done here:
    1. Compute the advantages for each group
    2. Build the Tinker Datum objects for each group
    """
    # step 1: compute the advantages for each group using the common functionality
    # this fills the `advantage` attribute of all the steps in the trajectory groups
    compute_advantage_from_trajectory_groups(trajectory_groups, algorithm_config)

    # step 2: iterate over all steps and build the Tinker Datum objects
    datums = []
    for group in trajectory_groups:
        for trajectory in group.trajectories:
            for step in trajectory.steps:
                datums.append(build_datum_from_step(step))
    return datums
