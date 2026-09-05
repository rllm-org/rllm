"""CPU regressions for colon-containing task IDs and GRPO grouping (#829)."""

import pytest

from rllm.trainer.algorithms.advantage import collect_reward_and_advantage_from_trajectory_groups
from rllm.trainer.algorithms.config import AlgorithmConfig, TransformConfig, rLLMAdvantageEstimator
from rllm.trainer.algorithms.transform import transform_episodes_to_trajectory_groups
from rllm.types import Episode, Step, Trajectory, TrajectoryGroup
from rllm.workflows.workflow import TerminationReason


@pytest.mark.parametrize(
    "episode_id, task_id, rollout_idx",
    [
        ("aeread:integration-v1:case01:s1200:0", "aeread:integration-v1:case01:s1200", "0"),
        ("task:0", "task", "0"),
        ("task:12", "task", "12"),
        ("123e4567-e89b-12d3-a456-426614174000:1", "123e4567-e89b-12d3-a456-426614174000", "1"),
        ("task::0", "task:", "0"),
        ("task:", "task", ""),
        ("task:custom", "task", "custom"),
    ],
)
def test_episode_id_parsing(episode_id, task_id, rollout_idx):
    episode = Episode(id=episode_id)

    assert episode.task_id == task_id
    assert episode.rollout_idx == rollout_idx


def test_default_episode_keeps_bare_uuid_behavior():
    episode = Episode()

    assert ":" not in episode.id
    assert episode.task_id == episode.id
    with pytest.raises(IndexError):
        _ = episode.rollout_idx


@pytest.mark.parametrize(
    "group_id, task_id, group_role",
    [
        ("aeread:integration-v1:case01:s1200:solver", "aeread:integration-v1:case01:s1200", "solver"),
        ("task:solver", "task", "solver"),
        ("task::solver", "task:", "solver"),
        ("bare", "bare", "all_groups"),
        ("task:", "task", "all_groups"),
        ("", "", "all_groups"),
    ],
)
def test_trajectory_group_id_parsing(group_id, task_id, group_role):
    group = TrajectoryGroup(group_id=group_id, trajectories=[])

    assert group.task_id == task_id
    assert group.group_role == group_role


@pytest.mark.parametrize("names", [("solver",), ("solver", "judge")])
def test_transform_groups_by_full_task_id_and_trajectory_name(names):
    task_ids = ["aeread:integration-v1:case01:s1200", "aeread:integration-v1:case02:s1200", "aeread", "other:case01"]
    episodes = [
        Episode(
            id=f"{task_id}:{rollout_idx}",
            trajectories=[Trajectory(name=name, steps=[Step()], reward=rollout_idx) for name in names],
            termination_reason=TerminationReason.ENV_DONE,
            is_correct=bool(rollout_idx),
        )
        for task_id in task_ids
        for rollout_idx in range(2)
    ]

    groups, metrics = transform_episodes_to_trajectory_groups(episodes, TransformConfig())

    assert len(groups) == len(task_ids) * len(names)
    assert metrics["groups/num_groups"] == len(groups)
    assert metrics["groups/avg_group_size"] == 2
    assert metrics["groups/num_trajs_after_filter"] == len(episodes) * len(names)
    by_id = {group.group_id: group for group in groups}
    for task_idx, task_id in enumerate(task_ids):
        for name_idx, name in enumerate(names):
            group = by_id[f"{task_id}:{name}"]
            assert group.task_id == task_id
            assert group.group_role == name
            assert len(group.trajectories) == 2
            assert group.metadata == [{"task_id": task_id, "rollout_idx": str(i), "termination_reason": TerminationReason.ENV_DONE, "is_correct": bool(i)} for i in range(2)]
            for rollout_idx in range(2):
                assert group.trajectories[rollout_idx] is episodes[2 * task_idx + rollout_idx].trajectories[name_idx]


@pytest.mark.parametrize("name", ["b:c", ":solver", "solver:"])
def test_default_grouping_rejects_colon_in_trajectory_name(name):
    # (task='a:b', name='c') and (task='a', name='b:c') serialize identically.
    episodes = [
        Episode(id="a:b:0", trajectories=[Trajectory(name="c", steps=[Step()], reward=0.0)]),
        Episode(id="a:0", trajectories=[Trajectory(name=name, steps=[Step()], reward=0.0)]),
    ]

    with pytest.raises(ValueError, match="Trajectory name .* must not contain ':'") as exc_info:
        transform_episodes_to_trajectory_groups(episodes, TransformConfig())
    assert repr(name) in str(exc_info.value)


@pytest.mark.parametrize("use_role_override", [False, True])
def test_grpo_advantages_are_centered_per_hierarchical_task(use_role_override):
    episodes = [
        Episode(
            id=f"aeread:integration-v1:{case}:s1200:{rollout_idx}",
            trajectories=[Trajectory(name="solver", steps=[Step()], reward=reward)],
        )
        for case, rewards in [("case01", [0.0, 2.0]), ("case02", [100.0, 102.0])]
        for rollout_idx, reward in enumerate(rewards)
    ]
    groups, _ = transform_episodes_to_trajectory_groups(episodes, TransformConfig())
    config = AlgorithmConfig(norm_adv_by_std_in_grpo=False)
    if use_role_override:
        config = AlgorithmConfig(
            norm_adv_by_std_in_grpo=False,
            estimator=rLLMAdvantageEstimator.REINFORCE,
            estimator_map={"solver": rLLMAdvantageEstimator.GRPO},
        )

    metrics = collect_reward_and_advantage_from_trajectory_groups(groups, config)

    assert [episode.trajectories[0].steps[0].advantage for episode in episodes] == pytest.approx([-1.0, 1.0, -1.0, 1.0])
    assert metrics["advantage/solver/mean"] == pytest.approx(0.0)
