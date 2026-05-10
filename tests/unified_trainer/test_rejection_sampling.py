from rllm.experimental.common.config import RejectionSamplingConfig
from rllm.experimental.common.rejection_sampling import (
    RejectionSamplingState,
    apply_rejection_sampling_and_filtering,
)
from rllm.types import Episode, Step, Trajectory, TrajectoryGroup


def _single_traj_episode() -> tuple[Episode, TrajectoryGroup]:
    trajectory = Trajectory(steps=[Step(reward=1.0, done=True)], reward=1.0)
    episode = Episode(id="task-1:0", is_correct=True, trajectories=[trajectory])
    group = TrajectoryGroup(trajectories=[trajectory], group_id="task-1:default")
    return episode, group


def test_rejection_sampling_none_keeps_single_trajectory_groups():
    episode, group = _single_traj_episode()

    groups, episodes, metrics = apply_rejection_sampling_and_filtering(
        episodes=[episode],
        groups=[group],
        config=RejectionSamplingConfig(mode="none", min_trajs_per_group=2),
        state=RejectionSamplingState(),
    )

    assert groups == [group]
    assert episodes == [episode]
    assert metrics["batch/groups_before_filter"] == 1
    assert metrics["batch/groups_after_filter"] == 1
    assert metrics["batch/groups_dropped_insufficient_trajs"] == 0


def test_rejection_sampling_episode_still_drops_insufficient_groups():
    episode, group = _single_traj_episode()

    groups, episodes, metrics = apply_rejection_sampling_and_filtering(
        episodes=[episode],
        groups=[group],
        config=RejectionSamplingConfig(mode="episode", min_trajs_per_group=2),
        state=RejectionSamplingState(),
    )

    assert groups == []
    assert episodes == []
    assert metrics["batch/groups_before_filter"] == 1
    assert metrics["batch/groups_after_filter"] == 0
    assert metrics["batch/groups_dropped_insufficient_trajs"] == 1
