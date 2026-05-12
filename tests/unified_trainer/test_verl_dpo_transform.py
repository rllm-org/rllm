"""Shape tests for the Verl DPO preference-pair transform."""

from __future__ import annotations

import pytest

pytest.importorskip("verl")

from rllm.experimental.common.preference import DPOConfig, build_preference_pairs
from rllm.experimental.verl.dpo_transform import transform_preference_pairs_to_dataproto
from rllm.types import Step, Trajectory, TrajectoryGroup


class _Tokenizer:
    pad_token_id = 0


class _RolloutEngine:
    tokenizer = _Tokenizer()
    processor = None


def _trajectory(reward: float, response_ids: list[int], logprobs: list[float] | None = None) -> Trajectory:
    return Trajectory(
        steps=[
            Step(
                prompt_ids=[1, 2, 3],
                response_ids=response_ids,
                logprobs=logprobs or [],
                reward=reward,
            )
        ],
        reward=reward,
    )


def test_transform_emits_adjacent_chosen_rejected_rows():
    group = TrajectoryGroup(
        group_id="task:solver",
        trajectories=[
            _trajectory(1.0, [10, 11], [-0.1, -0.2]),
            _trajectory(0.0, [12], [-0.3]),
        ],
    )
    pairs, _ = build_preference_pairs([group], DPOConfig())

    batch = transform_preference_pairs_to_dataproto(pairs, _RolloutEngine(), max_prompt_length=5, max_response_length=4)

    assert batch.batch["responses"].shape == (2, 4)
    assert batch.batch["response_mask"].tolist() == [[1, 1, 0, 0], [1, 0, 0, 0]]
    assert batch.batch["dpo_is_chosen"].tolist() == [True, False]
    assert batch.batch["dpo_pair_indices"].tolist() == [0, 0]
    assert batch.non_tensor_batch["pair_ids"].tolist() == ["task:solver:0", "task:solver:0"]
    assert batch.non_tensor_batch["is_chosen"].tolist() == [True, False]
    assert batch.non_tensor_batch["group_roles"].tolist() == ["solver", "solver"]
