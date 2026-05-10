"""Tinker Datum builders. Adapted from
https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/data_processing.py
"""

from collections import defaultdict
from dataclasses import dataclass

import tinker
from tinker.types.tensor_data import TensorData
from tinker_cookbook.supervised.common import create_rightshifted_model_input_and_leftshifted_targets

from rllm.experimental.common import AlgorithmConfig, collect_reward_and_advantage_from_trajectory_groups
from rllm.experimental.common.step_merge import (
    MergedSegment,
    PerTokenExtras,
    merge_trajectory_steps,
)
from rllm.experimental.rollout.tinker_engine import _flat_token_input_to_model_input
from rllm.experimental.rollout.types import TinkerTokenInput
from rllm.types import Trajectory, TrajectoryGroup

_ROUTING_KEY = "routing_matrices"


@dataclass(frozen=True)
class _TinkerTokenOps:
    """Chunk-aware ``TokenOps`` for ``TinkerTokenInput`` (mixed int + ModelInputChunk)."""

    def flatten_prompt(self, prompt: TinkerTokenInput) -> TinkerTokenInput:
        out: list = []
        for elem in prompt:
            if isinstance(elem, tinker.EncodedTextChunk):
                out.extend(elem.tokens)
            else:
                # Int or other ModelInputChunk subclasses (e.g. ImageChunk) are
                # preserved by reference so prefix-detection can match them
                # by equality on the next step.
                out.append(elem)
        return out

    def flat_token_length(self, token_input: TinkerTokenInput) -> int:
        length = 0
        for elem in token_input:
            if isinstance(elem, int):
                length += 1
            else:
                length += elem.length
        return length


_TINKER_OPS = _TinkerTokenOps()


def _segment_to_datum(seg: MergedSegment, *, router_replay: bool) -> tinker.Datum:
    full_seq = list(seg.prompt_ids) + list(seg.response_ids)
    model_input = _flat_token_input_to_model_input(full_seq)
    input_T, target_T = create_rightshifted_model_input_and_leftshifted_targets(list(model_input.chunks))

    prompt_n = _TINKER_OPS.flat_token_length(seg.prompt_ids)
    pad = [0.0] * prompt_n
    logprobs_T = (pad + list(seg.response_logprobs))[1:]
    advantages_T = (pad + list(seg.response_advantages))[1:]
    mask_T = (pad + [float(m) for m in seg.response_mask])[1:]

    assert input_T.length == len(target_T) == len(logprobs_T) == len(advantages_T) == len(mask_T)

    if router_replay and _ROUTING_KEY in seg.extras:
        rm = ([""] * prompt_n + list(seg.extras[_ROUTING_KEY]))[1:]
        input_T = input_T.model_copy(update={_ROUTING_KEY: rm})

    return tinker.Datum(
        model_input=input_T,
        loss_fn_inputs={
            "target_tokens": TensorData(data=target_T, dtype="int64"),
            "logprobs": TensorData(data=logprobs_T, dtype="float32"),
            "advantages": TensorData(data=advantages_T, dtype="float32"),
            "mask": TensorData(data=mask_T, dtype="float32"),
        },
    )


def trajectory_to_datums(traj: Trajectory, router_replay: bool = False) -> list[tinker.Datum]:
    """Merge ``traj`` into one Datum per cumulative-prefix run.

    Example: prompts/responses ``(O1, A1)``, ``(O1+A1+O2, A2)``, ``(O3, A3)``
    merge the first two into one Datum and emit a separate Datum for the
    third (its prompt isn't an extension of step 2's full sequence).
    """
    per_token_extras = None
    if router_replay:
        per_token_extras = {
            _ROUTING_KEY: PerTokenExtras(
                extractor=lambda s: s.routing_matrices,
                pad_value="",
            )
        }
    segments = merge_trajectory_steps(
        traj,
        token_ops=_TINKER_OPS,
        require_logprobs=True,
        require_advantage=True,
        per_token_extras=per_token_extras,
    )
    return [_segment_to_datum(s, router_replay=router_replay) for s in segments]


def transform_trajectory_groups_to_datums(
    trajectory_groups: list[TrajectoryGroup],
    algorithm_config: AlgorithmConfig,
) -> tuple[list[tinker.Datum] | dict[str, list[tinker.Datum]], dict]:
    """Build Datums for each TrajectoryGroup, plus advantage + merge metrics.

    If ``algorithm_config.estimator_map`` is set, returns a dict keyed by
    trajectory-group role; otherwise a flat list. Metric names mirror
    verl's ``transform_episodes_to_dataproto``.
    """
    has_advantages = any(step.advantage is not None for group in trajectory_groups for traj in group.trajectories for step in traj.steps)
    adv_metrics = {} if has_advantages else collect_reward_and_advantage_from_trajectory_groups(trajectory_groups, algorithm_config)

    if algorithm_config.estimator_map:
        datums_dict = defaultdict(list)
    else:
        datums = []

    steps_per_traj = []
    step_response_lengths = []
    action_token_ratios = []
    total_agent_steps = 0
    for group in trajectory_groups:
        for trajectory in group.trajectories:
            total_agent_steps += len(trajectory.steps)
            traj_datums = trajectory_to_datums(trajectory, router_replay=algorithm_config.router_replay)
            steps_per_traj.append(len(traj_datums))
            for d in traj_datums:
                mask_data = d.loss_fn_inputs["mask"].data
                # Mask is 0 over the prompt prefix and 0/1-interleaved after,
                # so the first 1 marks the prompt/response boundary.
                first_action = next((i for i, m in enumerate(mask_data) if m > 0.5), len(mask_data))
                resp_len = len(mask_data) - first_action
                action_count = sum(1 for m in mask_data[first_action:] if m > 0.5)
                step_response_lengths.append(resp_len)
                action_token_ratios.append(action_count / resp_len if resp_len > 0 else 0.0)
            if algorithm_config.estimator_map:
                datums_dict[group.group_role].extend(traj_datums)
            else:
                datums.extend(traj_datums)

    if steps_per_traj:
        import numpy as _np

        total_emitted_rows = sum(steps_per_traj)
        adv_metrics["batch/steps_per_traj/mean"] = _np.mean(steps_per_traj)
        adv_metrics["batch/steps_per_traj/min"] = _np.min(steps_per_traj)
        adv_metrics["batch/steps_per_traj/max"] = _np.max(steps_per_traj)
        adv_metrics["batch/step_response_length/mean"] = _np.mean(step_response_lengths)
        adv_metrics["batch/step_response_length/min"] = _np.min(step_response_lengths)
        adv_metrics["batch/step_response_length/max"] = _np.max(step_response_lengths)
        adv_metrics["batch/action_token_ratio/mean"] = _np.mean(action_token_ratios)
        adv_metrics["batch/action_token_ratio/min"] = _np.min(action_token_ratios)
        adv_metrics["batch/action_token_ratio/max"] = _np.max(action_token_ratios)
        adv_metrics["batch/merge_compression_ratio"] = total_agent_steps / total_emitted_rows if total_emitted_rows > 0 else 0.0

    return (datums if not algorithm_config.estimator_map else datums_dict), adv_metrics
