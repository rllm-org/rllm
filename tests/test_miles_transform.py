"""Miles transform: trajectory steps -> merged Miles sample payloads."""

import pytest

from rllm.trainer.miles.transform import (
    SamplePayload,
    trajectory_groups_to_payloads,
    trajectory_to_payloads,
)
from rllm.types import Step, Trajectory, TrajectoryGroup


def step(prompt_ids, response_ids, advantage=1.0, logprobs=None, lineage_id=None):
    return Step(
        prompt_ids=list(prompt_ids),
        response_ids=list(response_ids),
        logprobs=logprobs if logprobs is not None else [-0.5] * len(response_ids),
        advantage=advantage,
        metadata={"lineage_id": lineage_id} if lineage_id is not None else None,
    )


def traj(*steps, reward=1.0):
    return Trajectory(steps=list(steps), reward=reward)


class TestSingleStep:
    def test_prompt_is_masked_response_is_not(self):
        (p,) = trajectory_to_payloads(traj(step([1, 2, 3], [4, 5])))
        assert p.tokens == [1, 2, 3, 4, 5]
        assert p.prompt_length == 3
        assert p.response_length == 2
        assert p.loss_mask == [1, 1]
        assert p.rollout_log_probs == [-0.5, -0.5]
        assert p.advantages == [1.0, 1.0]

    def test_scalar_advantage_is_broadcast_over_response_tokens(self):
        (p,) = trajectory_to_payloads(traj(step([1], [2, 3, 4], advantage=0.25)))
        assert p.advantages == [0.25, 0.25, 0.25]

    def test_per_token_advantage_is_used_verbatim(self):
        (p,) = trajectory_to_payloads(traj(step([1], [2, 3], advantage=[0.1, 0.9])))
        assert p.advantages == [0.1, 0.9]

    def test_reward_comes_from_the_trajectory(self):
        (p,) = trajectory_to_payloads(traj(step([1], [2]), reward=0.75))
        assert p.reward == 0.75


class TestMerging:
    def test_prefix_extension_merges_into_one_sequence(self):
        # step 2's prompt is step 1's prompt+response plus a new observation.
        p1 = step([1, 2], [3, 4])
        p2 = step([1, 2, 3, 4, 9], [5])
        (p,) = trajectory_to_payloads(traj(p1, p2))
        assert p.tokens == [1, 2, 3, 4, 9, 5]
        assert p.prompt_length == 2
        # response region: [3,4] action, [9] observation, [5] action
        assert p.loss_mask == [1, 1, 0, 1]
        assert p.advantages == [1.0, 1.0, 0.0, 1.0]
        assert p.rollout_log_probs == [-0.5, -0.5, 0.0, -0.5]

    def test_non_extension_starts_a_new_payload(self):
        payloads = trajectory_to_payloads(traj(step([1, 2], [3]), step([7, 8], [9])))
        assert len(payloads) == 2
        assert payloads[0].tokens == [1, 2, 3]
        assert payloads[1].tokens == [7, 8, 9]

    def test_three_way_chain_stays_one_sequence(self):
        payloads = trajectory_to_payloads(
            traj(
                step([1], [2]),
                step([1, 2], [3]),
                step([1, 2, 3], [4]),
            )
        )
        assert len(payloads) == 1
        assert payloads[0].tokens == [1, 2, 3, 4]
        assert payloads[0].loss_mask == [1, 1, 1]

    def test_mixed_chain_then_break(self):
        payloads = trajectory_to_payloads(
            traj(
                step([1], [2]),
                step([1, 2], [3]),
                step([50], [51]),
            )
        )
        assert [p.tokens for p in payloads] == [[1, 2, 3], [50, 51]]


class TestLineagePartitioning:
    def test_interleaved_lineages_merge_independently(self):
        payloads = trajectory_to_payloads(
            traj(
                step([1], [2], lineage_id="parent"),
                step([90], [91], lineage_id="sub"),
                step([1, 2], [3], lineage_id="parent"),
                step([90, 91], [92], lineage_id="sub"),
            )
        )
        # Without partitioning the interleaving would break both chains into 4.
        assert len(payloads) == 2
        assert sorted(p.tokens for p in payloads) == [[1, 2, 3], [90, 91, 92]]

    def test_untagged_steps_form_a_single_partition(self):
        payloads = trajectory_to_payloads(traj(step([1], [2]), step([1, 2], [3])))
        assert len(payloads) == 1


class TestValidation:
    def test_logprob_count_must_match_response_tokens(self):
        with pytest.raises(ValueError, match="logprobs"):
            trajectory_to_payloads(traj(step([1], [2, 3], logprobs=[-0.1])))

    def test_missing_advantage_is_rejected(self):
        with pytest.raises(ValueError, match="advantage is None"):
            trajectory_to_payloads(traj(step([1], [2], advantage=None)))

    def test_per_token_advantage_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="per-token advantage"):
            trajectory_to_payloads(traj(step([1], [2, 3], advantage=[0.5])))

    def test_non_integer_tokens_are_rejected(self):
        with pytest.raises(TypeError, match="flat integer"):
            trajectory_to_payloads(traj(step([1, {"image": "x"}], [2])))

    def test_step_with_no_response_contributes_only_context(self):
        # An empty-response step is not trainable on its own...
        assert trajectory_to_payloads(traj(step([1, 2], []))) == []
        # ...but its tokens still become context for a step that extends it.
        (p,) = trajectory_to_payloads(traj(step([1, 2], []), step([1, 2, 7], [8])))
        assert p.tokens == [1, 2, 7, 8]
        assert p.loss_mask == [0, 1]

    def test_payload_validate_catches_array_length_drift(self):
        bad = SamplePayload(tokens=[1, 2, 3], prompt_length=1, loss_mask=[1], rollout_log_probs=[0.0, 0.0], advantages=[0.0, 0.0])
        with pytest.raises(ValueError, match="loss_mask has length 1"):
            bad.validate()

    def test_payload_validate_rejects_prompt_longer_than_tokens(self):
        bad = SamplePayload(tokens=[1], prompt_length=5, loss_mask=[], rollout_log_probs=[], advantages=[])
        with pytest.raises(ValueError, match="exceeds token count"):
            bad.validate()


class TestGroups:
    def test_groups_are_preserved_as_nested_lists(self):
        g1 = TrajectoryGroup(trajectories=[traj(step([1], [2])), traj(step([3], [4]))], group_id="a:agent")
        g2 = TrajectoryGroup(trajectories=[traj(step([5], [6]))], group_id="b:agent")
        grouped = trajectory_groups_to_payloads([g1, g2])
        assert [len(g) for g in grouped] == [2, 1]

    def test_group_with_nothing_trainable_is_dropped(self):
        empty = TrajectoryGroup(trajectories=[traj(step([1, 2], []))], group_id="empty:agent")
        good = TrajectoryGroup(trajectories=[traj(step([1], [2]))], group_id="good:agent")
        grouped = trajectory_groups_to_payloads([empty, good])
        assert len(grouped) == 1
        assert grouped[0][0].tokens == [1, 2]
