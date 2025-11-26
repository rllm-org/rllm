import logging
from collections import defaultdict

from rllm.agents.agent import Episode, TrajectoryGroup

logger = logging.getLogger(__name__)


def _transform_episodes_to_trajectory_groups(episodes: list[Episode], do_augment: bool = False) -> list[TrajectoryGroup]:
    """
    Transform the given episodes into trajectory groups via grouping by trajectory name.

    Args:
        episodes: List of episodes to transform.
        do_augment: If True, we will also augment the trajectory names based on position, if applicable.

    Returns:
        List of trajectory groups.
    """
    # n_rollouts = len(episodes)
    traj_name_mapping = defaultdict(list)

    for episode in episodes:
        for trajectory in episode.trajectories:
            traj_name_mapping[trajectory.name].append(trajectory)

    return [TrajectoryGroup(trajectories=[trajectory], group_id=traj_name_mapping[trajectory.name]) for trajectory in traj_name_mapping.values()]
