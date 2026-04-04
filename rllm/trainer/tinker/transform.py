"""
Transformation utilities for converting token input (TinkerTokenInput) to Tinker Datum.
Code is adapted from https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/data_processing.py
"""

from collections import defaultdict
from typing import cast

import tinker
from tinker.types.tensor_data import TensorData
from tinker_cookbook.supervised.common import create_rightshifted_model_input_and_leftshifted_targets

from rllm.agents.agent import Trajectory, TrajectoryGroup
from rllm.experimental.rollout.tinker_engine import _flat_token_input_length, _flat_token_input_to_model_input
from rllm.experimental.rollout.types import TinkerTokenInput
from rllm.experimental.common import AlgorithmConfig, collect_reward_and_advantage_from_trajectory_groups


def _is_prefix(seq1: TinkerTokenInput, seq2: TinkerTokenInput) -> bool:
    """
    Check if seq1 is a prefix of seq2.
    """
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _flatten_token_input(token_input: TinkerTokenInput) -> TinkerTokenInput:
    """
    Flatten a tinker token input so it becomes a list with no `EncodedTextChunk`.
    """
    flattened: TinkerTokenInput = []
    for elem in token_input:
        if isinstance(elem, tinker.EncodedTextChunk):
            flattened.extend(elem.tokens)
        else:
            flattened.append(elem)
    return flattened


def trajectory_to_datums(traj: Trajectory, router_replay: bool = False) -> list[tinker.Datum]:
    """
    Return one or more Datum objects corresponding to the trajectory.
    If the sequence grows by appending, i.e., each successive observation contains
    the previous observation+action as a prefix, then we can return a single Datum.
    However, if we get a sequence that's not an extension of the previous sequence,
    then that results in a new Datum.

    For example, let O1 denote a chunk of observation tokens, and let A1 denote an action.

    Then let's say ob_ac_pairs is as follows.

    (O1, A1)
    (O1+A1+O2, A2)
    (O3, A3)

    Then we will merge the first two observation-action pairs into a single Datum,
    and the last observation-action pair into a separate Datum.
    """

    class SequenceAccumulator:
        full_sequence: TinkerTokenInput = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []
        routing_matrices: list[str] = []

        @classmethod
        def clear(cls):
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []
            cls.routing_matrices = []

    def make_datum_from_state():
        all_tokens_T = _flat_token_input_to_model_input(SequenceAccumulator.full_sequence)
        # this should help handle image chunk as well
        input_tokens_T, target_tokens_T = create_rightshifted_model_input_and_leftshifted_targets(list(all_tokens_T.chunks))
        sampled_logprobs_T = SequenceAccumulator.sampled_logprobs[1:]
        advantages_T = SequenceAccumulator.advantages[1:]
        mask_T = SequenceAccumulator.mask[1:]
        assert input_tokens_T.length == len(target_tokens_T) == len(sampled_logprobs_T) == len(advantages_T) == len(mask_T)
        if router_replay and SequenceAccumulator.routing_matrices:
            rm_shifted = SequenceAccumulator.routing_matrices[1:]  # match rightshift
            input_tokens_T = input_tokens_T.model_copy(update={"routing_matrices": rm_shifted})
        return tinker.Datum(
            model_input=input_tokens_T,
            loss_fn_inputs={
                "target_tokens": TensorData(data=target_tokens_T, dtype="int64"),
                "logprobs": TensorData(data=sampled_logprobs_T, dtype="float32"),
                "advantages": TensorData(data=advantages_T, dtype="float32"),
                "mask": TensorData(data=mask_T, dtype="float32"),
            },
        )

    data: list[tinker.Datum] = []
    for step in traj.steps:
        token_input = cast(TinkerTokenInput, step.prompt_ids)
        token_input_flat = _flatten_token_input(token_input)

        output_token_ids, output_logprobs = step.response_ids, step.logprobs
        assert len(output_logprobs) > 0, "output_logprobs is empty. Cannot build Tinker Datum for training."
        assert step.advantage is not None, "step.advantage is None. This indicates that advantage computation has not been performed yet."

        # build advantage list -- match length of token_output.tokens
        if isinstance(step.advantage, list):
            assert len(step.advantage) == len(output_token_ids), "length mismatch between step.advantage and token_output.tokens"
            advantages = step.advantage
        else:  # float
            advantages = [step.advantage] * len(output_token_ids)

        if len(SequenceAccumulator.full_sequence) == 0:
            delta_token_input_flat = token_input_flat
        elif _is_prefix(SequenceAccumulator.full_sequence, token_input_flat):
            delta_token_input_flat = token_input_flat[len(SequenceAccumulator.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            SequenceAccumulator.clear()
            delta_token_input_flat = token_input_flat

        delta_token_input_length = _flat_token_input_length(delta_token_input_flat)
        SequenceAccumulator.full_sequence.extend(delta_token_input_flat)
        SequenceAccumulator.full_sequence.extend(output_token_ids)
        SequenceAccumulator.sampled_logprobs.extend([0.0] * delta_token_input_length + output_logprobs)
        SequenceAccumulator.advantages.extend([0] * delta_token_input_length + advantages)
        SequenceAccumulator.mask.extend([0.0] * delta_token_input_length + [1.0] * len(output_token_ids))
        if router_replay:
            step_rm = step.routing_matrices or []
            SequenceAccumulator.routing_matrices.extend(
                [""] * delta_token_input_length
                + (list(step_rm) if step_rm else [""] * len(output_token_ids))
            )

    if SequenceAccumulator.full_sequence:
        data.append(make_datum_from_state())

    return data


def transform_trajectory_groups_to_datums(
    trajectory_groups: list[TrajectoryGroup],
    algorithm_config: AlgorithmConfig,
) -> tuple[list[tinker.Datum] | dict[str, list[tinker.Datum]], dict]:
    """
    Transform a list of TrajectoryGroup objects to a list of Tinker Datum objects. Two things are done here:
    1. Compute the advantages for each group
    2. Build the Tinker Datum objects for each group

    If the `estimator_map` is used in the algorithm config, we return a dictionary of datums, keyed by the trajectory group role.
    Otherwise, we return a list of datums.
    """
    # step 1: compute advantages (skip if already pre-computed by buffer)
    has_advantages = any(
        step.advantage is not None
        for group in trajectory_groups
        for traj in group.trajectories
        for step in traj.steps
    )
    if has_advantages:
        adv_metrics = {}
    else:
        adv_metrics = collect_reward_and_advantage_from_trajectory_groups(trajectory_groups, algorithm_config)

    if algorithm_config.estimator_map:
        datums_dict = defaultdict(list)
    else:
        datums = []

    # step 2: iterate over all steps and build the Tinker Datum objects
    seqs_per_traj = []
    seq_lengths = []
    for group in trajectory_groups:
        for trajectory in group.trajectories:
            traj_datums = trajectory_to_datums(trajectory, router_replay=algorithm_config.router_replay)
            seqs_per_traj.append(len(traj_datums))
            for d in traj_datums:
                seq_lengths.append(d.model_input.length)
            if algorithm_config.estimator_map:
                datums_dict[group.group_role].extend(traj_datums)
            else:
                datums.extend(traj_datums)

    if seqs_per_traj:
        import numpy as _np
        adv_metrics["batch/seqs_per_traj/mean"] = _np.mean(seqs_per_traj)
        adv_metrics["batch/seqs_per_traj/min"] = _np.min(seqs_per_traj)
        adv_metrics["batch/seqs_per_traj/max"] = _np.max(seqs_per_traj)
        adv_metrics["batch/seq_length/mean"] = _np.mean(seq_lengths)
        adv_metrics["batch/seq_length/min"] = _np.min(seq_lengths)
        adv_metrics["batch/seq_length/max"] = _np.max(seq_lengths)

    return (datums if not algorithm_config.estimator_map else datums_dict), adv_metrics
