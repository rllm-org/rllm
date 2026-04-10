"""GSD-specific trajectory-to-datum transform.

Routes trajectories to datums based on their role:

* :data:`CE_ROLE` (``"gsd_ce"``) → :func:`build_sft_style_ce_datum`,
  consumed by Tinker's built-in ``cross_entropy`` loss.
* :data:`IS_ROLE` (``"gsd_is"``) → :func:`build_is_datum`, consumed by
  Tinker's built-in ``importance_sampling`` loss.  Advantages are already
  populated on ``step.advantage`` by the workflow (using the frozen
  reference teacher's logprobs).
* :data:`GRPO_ROLE` (``"gsd_grpo"``) → default ``trajectory_to_datums``
  path with group-centered GRPO advantages computed from the trajectory
  rewards.
* :data:`HINT_ROLE` (``"gsd_hint"``) → default path; the cross-task
  grouping hook (see :mod:`rllm.experimental.gsd.grouping`) collapses
  hints from all tasks into one group so REINFORCE's baseline has
  something to average against.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from rllm.experimental.common import (
    AlgorithmConfig,
    collect_reward_and_advantage_from_trajectory_groups,
)
from rllm.experimental.gsd.losses import (
    CE_ROLE,
    GRPO_ROLE,
    HINT_ROLE,
    IS_ROLE,
    build_is_datum,
    build_sft_style_ce_datum,
)
from rllm.trainer.tinker.transform import trajectory_to_datums

if TYPE_CHECKING:
    import tinker

    from rllm.agents.agent import TrajectoryGroup

logger = logging.getLogger(__name__)

# Roles whose datums are constructed by GSD's own helpers, bypassing the
# default ``trajectory_to_datums`` pipeline.
_GSD_CUSTOM_DATUM_ROLES = frozenset({CE_ROLE, IS_ROLE})


def gsd_transform_trajectory_groups_to_datums(
    trajectory_groups: list[TrajectoryGroup],
    algorithm_config: AlgorithmConfig,
) -> tuple[dict[str, list[tinker.Datum]], dict]:
    """Transform trajectory groups into per-role datum dicts for GSD.

    Returns a ``(datums_dict, metrics)`` tuple where ``datums_dict`` is keyed
    by role.  The Tinker policy trainer then fires one
    ``forward_backward_async`` call per role, each with its own built-in
    loss function (see :func:`build_gsd_estimator_map`).
    """
    adv_metrics: dict = {}

    # 1. Compute advantages for the GRPO / hint roles that need group-centered
    #    rewards.  Skip CE/IS groups — their steps either have no advantage
    #    (CE) or already have per-token advantages pre-populated by the
    #    workflow (IS).
    groups_needing_adv = [g for g in trajectory_groups if g.group_role not in _GSD_CUSTOM_DATUM_ROLES and not any(step.advantage is not None for traj in g.trajectories for step in traj.steps)]
    if groups_needing_adv:
        adv_metrics = collect_reward_and_advantage_from_trajectory_groups(
            groups_needing_adv,
            algorithm_config,
        )

    # 2. Lightweight reward summaries for CE / IS groups so the metrics
    #    dashboard still has visibility into distillation reward.
    distill_rewards = [traj.reward for g in trajectory_groups if g.group_role in _GSD_CUSTOM_DATUM_ROLES for traj in g.trajectories if traj.reward is not None]
    if distill_rewards:
        import numpy as _np

        adv_metrics["reward/gsd_distill/mean"] = float(_np.mean(distill_rewards))
        adv_metrics["reward/gsd_distill/min"] = float(_np.min(distill_rewards))
        adv_metrics["reward/gsd_distill/max"] = float(_np.max(distill_rewards))
        adv_metrics["reward/gsd_distill/std"] = float(_np.std(distill_rewards))
        adv_metrics["reward/gsd_distill/count"] = float(len(distill_rewards))

    # 3. Build datums per role.
    datums_dict: dict[str, list] = defaultdict(list)

    for group in trajectory_groups:
        role = group.group_role

        if role == CE_ROLE:
            for traj in group.trajectories:
                for step in traj.steps:
                    if not step.response_ids:
                        continue
                    datums_dict[CE_ROLE].append(
                        build_sft_style_ce_datum(
                            prompt_ids=step.prompt_ids,
                            response_ids=step.response_ids,
                        )
                    )

        elif role == IS_ROLE:
            for traj in group.trajectories:
                for step in traj.steps:
                    if not step.response_ids:
                        continue
                    # Advantage was populated by the workflow from the frozen
                    # teacher's logprobs.  Default to 0.0 if somehow missing
                    # (e.g. sequence was entirely truncated during scoring).
                    adv = step.advantage
                    if adv is None:
                        adv = [0.0] * len(step.response_ids)
                    datums_dict[IS_ROLE].append(
                        build_is_datum(
                            prompt_ids=step.prompt_ids,
                            response_ids=step.response_ids,
                            logprobs=step.logprobs,
                            advantages=adv,
                        )
                    )

        else:
            # Default path for gsd_grpo, gsd_hint, and any other role.
            # collect_reward_and_advantage_from_trajectory_groups above has
            # already populated step.advantage with group-centered values.
            for traj in group.trajectories:
                datums_dict[role].extend(
                    trajectory_to_datums(
                        traj,
                        router_replay=algorithm_config.router_replay,
                    )
                )

    # Bookkeeping metrics
    for role in (CE_ROLE, IS_ROLE, GRPO_ROLE, HINT_ROLE):
        adv_metrics[f"train/gsd_role_count/{role}"] = float(len(datums_dict.get(role, [])))

    return dict(datums_dict), adv_metrics


__all__ = ["gsd_transform_trajectory_groups_to_datums"]
