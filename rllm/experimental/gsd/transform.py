"""GSD-specific trajectory-to-datum transform.

Dispatches to different datum builders per trajectory role:

* ``gsd_distill_supervised`` → :func:`build_topk_fkl_datum` (forward KL via
  Tinker ``cross_entropy`` with Top-K teacher soft targets).
* Everything else (``gsd_distill_onpolicy``, ``gsd_student``, ``gsd_hint``)
  → standard :func:`trajectory_to_datums` (``importance_sampling`` with
  pre-computed scalar advantages).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from rllm.experimental.common import AlgorithmConfig, collect_reward_and_advantage_from_trajectory_groups
from rllm.experimental.gsd.losses import build_topk_fkl_datum
from rllm.trainer.tinker.transform import trajectory_to_datums

if TYPE_CHECKING:
    import tinker

    from rllm.agents.agent import TrajectoryGroup

logger = logging.getLogger(__name__)

# Roles whose datums are built via the Top-K forward-KL cross_entropy path
CROSS_ENTROPY_ROLES = frozenset({"gsd_distill_supervised"})


def gsd_transform_trajectory_groups_to_datums(
    trajectory_groups: list[TrajectoryGroup],
    algorithm_config: AlgorithmConfig,
) -> tuple[dict[str, list[tinker.Datum]], dict]:
    """GSD-specific transform: dispatch CE vs IS datums per role.

    For ``CROSS_ENTROPY_ROLES``, each trajectory's step must have
    ``step.info["teacher_topk"]`` populated by the workflow.  A
    :func:`build_topk_fkl_datum` call converts this into a (T, K)-shaped
    cross-entropy datum.

    For all other roles, the standard advantage pipeline runs and
    :func:`trajectory_to_datums` produces importance-sampling datums.

    Returns:
        A ``(datums_dict, adv_metrics)`` tuple.  ``datums_dict`` is keyed
        by ``group_role`` (e.g. ``"gsd_distill_onpolicy"``).
    """
    # --- Step 1: Compute advantages for non-CE groups that need them ---
    # Groups with pre-computed advantages (e.g. gsd_distill_onpolicy) are
    # skipped; groups without (e.g. gsd_student / GRPO fallback) get
    # advantages computed by the standard pipeline.
    non_ce_groups = [g for g in trajectory_groups if g.group_role not in CROSS_ENTROPY_ROLES]
    needs_adv = [g for g in non_ce_groups if not any(step.advantage is not None for traj in g.trajectories for step in traj.steps)]
    adv_metrics: dict = {}
    if needs_adv:
        adv_metrics = collect_reward_and_advantage_from_trajectory_groups(
            needs_adv,
            algorithm_config,
        )

    # --- Step 2: Build datums per role ---
    datums_dict: dict[str, list] = defaultdict(list)

    for group in trajectory_groups:
        role = group.group_role
        if role in CROSS_ENTROPY_ROLES:
            for traj in group.trajectories:
                for step in traj.steps:
                    teacher_topk = (step.info or {}).get("teacher_topk")
                    if teacher_topk is None:
                        logger.warning(f"[{role}] Step missing teacher_topk in info, skipping.")
                        continue
                    loss_clamp = (step.info or {}).get("distill_loss_clamp")
                    datum = build_topk_fkl_datum(
                        prompt_ids=step.prompt_ids,
                        response_ids=step.response_ids,
                        teacher_topk=teacher_topk,
                        loss_clamp=loss_clamp,
                    )
                    datums_dict[role].append(datum)
        else:
            for traj in group.trajectories:
                traj_datums = trajectory_to_datums(
                    traj,
                    router_replay=algorithm_config.router_replay,
                )
                datums_dict[role].extend(traj_datums)

    return datums_dict, adv_metrics
