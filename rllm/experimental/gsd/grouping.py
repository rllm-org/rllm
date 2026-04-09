"""Custom trajectory grouping hook for GSD hint GRPO.

rLLM's default grouping hook (``_default_traj_grouping_hook`` in
``rllm/experimental/common/transform.py``) groups trajectories by
``f"{task_id}:{trajectory.name}"``.  This means each hint trajectory lands
in a group of size 1 — useless for group-centered advantage estimation
(``REINFORCE`` / ``GRPO``) because the baseline has nothing to average
against.

GSD needs hint trajectories from *different* tasks to form a single group,
so the advantage of each hint is ``reward - mean(all_hint_rewards_in_batch)``.
This module provides a grouping hook that:

1. Runs the default grouping to build per-task-per-role groups as usual.
2. Finds all groups whose ``group_role == hint_role`` (default ``"gsd_hint"``)
   and merges them into a single cross-task group with ``group_id`` of the
   form ``f"cross_task:{hint_role}"``.

Non-hint roles (``gsd_ce``, ``gsd_is``, ``gsd_grpo``) are left untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from rllm.agents.agent import TrajectoryGroup
from rllm.experimental.common.transform import _default_traj_grouping_hook
from rllm.experimental.gsd.losses import HINT_ROLE

if TYPE_CHECKING:
    from rllm.agents.agent import Episode
    from rllm.experimental.common.config import CompactFilteringConfig, TransformConfig

logger = logging.getLogger(__name__)


def make_gsd_grouping_hook(
    hint_role: str = HINT_ROLE,
) -> Callable:
    """Return a ``traj_grouping_hook`` that groups ``hint_role`` across tasks.

    Args:
        hint_role: The trajectory name (and therefore role) whose instances
            should be pulled together into a single cross-task group.
            Defaults to ``"gsd_hint"``.

    Returns:
        A callable matching ``UnifiedTrainer``'s ``traj_grouping_hook`` slot:
        ``(episodes, transform_config, compact_filtering_config) ->
        list[TrajectoryGroup]``.
    """

    def hook(
        episodes: list[Episode],
        transform_config: TransformConfig,
        compact_filtering_config: CompactFilteringConfig | None = None,
    ) -> list[TrajectoryGroup]:
        # Start with the default per-task-per-role grouping.  This also
        # handles reward validation / propagation which we don't want to
        # reimplement.
        groups = _default_traj_grouping_hook(episodes, transform_config, compact_filtering_config)

        # Partition into hint groups (to be merged) and everything else.
        hint_groups: list[TrajectoryGroup] = []
        other_groups: list[TrajectoryGroup] = []
        for g in groups:
            if g.group_role == hint_role:
                hint_groups.append(g)
            else:
                other_groups.append(g)

        if not hint_groups:
            return groups

        # Merge all hint groups into one cross-task group.
        merged_trajectories = [t for g in hint_groups for t in g.trajectories]
        merged_metadata = [m for g in hint_groups for m in g.metadata]
        merged = TrajectoryGroup(
            trajectories=merged_trajectories,
            group_id=f"cross_task:{hint_role}",
            metadata=merged_metadata,
        )
        logger.info(f"[GSD grouping] merged {len(hint_groups)} per-task '{hint_role}' groups into one cross-task group of size {len(merged_trajectories)}")
        return other_groups + [merged]

    return hook


__all__ = ["make_gsd_grouping_hook"]
