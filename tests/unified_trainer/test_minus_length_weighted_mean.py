import pytest

from rllm.trainer.algorithms.advantage import collect_reward_and_advantage_from_trajectory_groups
from rllm.trainer.algorithms.config import AlgorithmConfig, rLLMAdvantageEstimator
from rllm.trainer.buffer import _advantage_is_nonzero
from rllm.trainer.tinker.transform import trajectory_to_datums
from rllm.types import Step, Trajectory, TrajectoryGroup


def _trajectory(reward: float, length: int, token_rewards: list[float] | None = None) -> Trajectory:
    metadata = {} if token_rewards is None else {"per_token_rewards": token_rewards}
    return Trajectory(
        reward=reward,
        steps=[Step(prompt_ids=[99], response_ids=list(range(length)), logprobs=[-0.1] * length, metadata=metadata)],
    )


def _config() -> AlgorithmConfig:
    return AlgorithmConfig(estimator=rLLMAdvantageEstimator.MINUS_LENGTH_WEIGHTED_MEAN)


def test_deserialized_step_without_advantage_remains_uncomputed():
    serialized = Step(response_ids=[1]).to_dict()
    serialized.pop("advantage")

    assert Step.from_dict(serialized).advantage is None


def test_collector_materializes_trajectory_advantage_per_response_token():
    trajectory = Trajectory(
        reward=0.75,
        steps=[
            Step(response_ids=[1, 2]),
            Step(response_ids=[3]),
        ],
    )
    group = TrajectoryGroup(group_id="task:solver", trajectories=[trajectory])

    collect_reward_and_advantage_from_trajectory_groups(
        [group],
        AlgorithmConfig(estimator=rLLMAdvantageEstimator.REINFORCE),
    )

    assert [step.advantage for step in trajectory.steps] == [[0.75, 0.75], [0.75]]


def test_outcome_rewards_use_one_length_weighted_role_baseline():
    groups = [
        TrajectoryGroup(group_id="task-a:solver", trajectories=[_trajectory(0.9, 100)]),
        TrajectoryGroup(group_id="task-b:solver", trajectories=[_trajectory(0.0, 50)]),
    ]

    metrics = collect_reward_and_advantage_from_trajectory_groups(groups, _config())

    assert groups[0].trajectories[0].steps[0].advantage == pytest.approx([0.3] * 100)
    assert groups[1].trajectories[0].steps[0].advantage == pytest.approx([-0.6] * 50)
    assert metrics["advantage/solver/mean"] == pytest.approx(-0.15)


def test_token_rewards_produce_one_target_per_response_token():
    groups = [
        TrajectoryGroup(group_id="task-a:solver", trajectories=[_trajectory(1.0, 4, [1.0, 1.0, 0.8, 0.6])]),
        TrajectoryGroup(group_id="task-b:solver", trajectories=[_trajectory(0.0, 2, [0.0, 0.0])]),
    ]

    metrics = collect_reward_and_advantage_from_trajectory_groups(groups, _config())

    assert groups[0].trajectories[0].steps[0].advantage == pytest.approx([0.433333, 0.433333, 0.233333, 0.033333], abs=1e-6)
    assert groups[1].trajectories[0].steps[0].advantage == pytest.approx([-0.566667, -0.566667], abs=1e-6)
    assert metrics["advantage/solver/mean"] == pytest.approx(-0.066667, abs=1e-6)

    datums = [trajectory_to_datums(group.trajectories[0])[0] for group in groups]
    active_targets = [[value for value, mask in zip(datum.loss_fn_inputs["advantages"].data, datum.loss_fn_inputs["mask"].data, strict=True) if mask] for datum in datums]
    assert active_targets[0] == pytest.approx([0.433333, 0.433333, 0.233333, 0.033333], abs=1e-6)
    assert active_targets[1] == pytest.approx([-0.566667, -0.566667], abs=1e-6)


def test_token_mode_rejects_partial_or_misaligned_rewards():
    partial = TrajectoryGroup(
        group_id="task:solver",
        trajectories=[_trajectory(1.0, 2, [1.0, 0.5]), _trajectory(0.0, 1)],
    )
    with pytest.raises(ValueError, match="every non-empty step"):
        collect_reward_and_advantage_from_trajectory_groups([partial], _config())

    misaligned = TrajectoryGroup(group_id="task:solver", trajectories=[_trajectory(1.0, 2, [1.0])])
    with pytest.raises(ValueError, match="does not match response_ids"):
        collect_reward_and_advantage_from_trajectory_groups([misaligned], _config())


@pytest.mark.parametrize(
    ("advantage", "expected"),
    [(None, False), ([0.0, 0.0], False), ([0.0, 0.1], True)],
)
def test_uniform_filter_handles_token_targets(advantage, expected):
    assert _advantage_is_nonzero(advantage) is expected
