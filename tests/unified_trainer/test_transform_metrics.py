from rllm.experimental.common.config import CompactFilteringConfig, TransformConfig
from rllm.experimental.common.transform import transform_episodes_to_trajectory_groups
from rllm.types import Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationReason


def test_transform_metrics_handles_empty_groups_after_filtering():
    episode = Episode(
        id="task-1:0",
        termination_reason=TerminationReason.ERROR,
        trajectories=[Trajectory(steps=[Step(reward=0.0, done=True)], reward=0.0)],
    )
    filtering_config = CompactFilteringConfig(enable=True, mask_error=True)

    groups, metrics = transform_episodes_to_trajectory_groups(
        [episode],
        TransformConfig(),
        filtering_config,
    )

    assert groups == []
    assert metrics["groups/num_trajs_before_filter"] == 1
    assert metrics["groups/num_trajs_after_filter"] == 0
    assert metrics["groups/num_groups"] == 0
    assert metrics["groups/avg_group_size"] == 0.0
    assert metrics["groups/max_group_size"] == 0
    assert metrics["groups/min_group_size"] == 0
