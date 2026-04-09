"""GSD-specific trajectory-to-datum transform.

Dispatches to different datum builders per trajectory role:

* ``gsd_distill`` (and legacy ``gsd_distill_onpolicy`` / ``gsd_distill_supervised``)
  → :func:`build_combined_gsd_datum` producing ``(N, K+1)`` datums for the
  combined CE + IS custom loss via ``forward_backward_custom_async``.
* Everything else (``gsd_student``, ``gsd_hint``) → standard
  :func:`trajectory_to_datums` for ``importance_sampling`` / ``ppo``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from rllm.experimental.common import AlgorithmConfig, collect_reward_and_advantage_from_trajectory_groups
from rllm.experimental.gsd.legacy.losses_topk import build_combined_gsd_datum
from rllm.trainer.tinker.transform import trajectory_to_datums

if TYPE_CHECKING:
    import tinker

    from rllm.agents.agent import TrajectoryGroup

logger = logging.getLogger(__name__)

# Roles whose datums are built via the combined (K+1) custom-loss path.
COMBINED_DISTILL_ROLES = frozenset({"gsd_distill", "gsd_distill_onpolicy", "gsd_distill_supervised"})


def gsd_transform_trajectory_groups_to_datums(
    trajectory_groups: list[TrajectoryGroup],
    algorithm_config: AlgorithmConfig,
) -> tuple[dict[str, list[tinker.Datum]], dict]:
    """GSD-specific transform: combined CE+IS datums for distillation roles.

    For ``COMBINED_DISTILL_ROLES``, each trajectory's step is converted into
    a ``(N, K+1)`` shaped combined datum via :func:`build_combined_gsd_datum`.
    The transform infers what to populate from the step's fields:

    * ``step.advantage is not None`` → IS fields populated (on-policy).
    * ``step.info.get("teacher_topk") is not None`` → CE fields populated
      (supervised).
    * Both can be True simultaneously.

    All combined datums land under the ``"gsd_distill"`` key, regardless of
    their original trajectory name.

    For all other roles, the standard advantage pipeline runs and
    :func:`trajectory_to_datums` produces importance-sampling datums.

    Returns:
        A ``(datums_dict, adv_metrics)`` tuple.
    """
    # --- Step 1: Compute advantages for non-distill groups that need them ---
    non_distill_groups = [g for g in trajectory_groups if g.group_role not in COMBINED_DISTILL_ROLES]
    needs_adv = [g for g in non_distill_groups if not any(step.advantage is not None for traj in g.trajectories for step in traj.steps)]
    adv_metrics: dict = {}
    if needs_adv:
        adv_metrics = collect_reward_and_advantage_from_trajectory_groups(
            needs_adv,
            algorithm_config,
        )

    # Collect reward metrics for distill groups (advantages are pre-computed
    # or handled by the custom loss, but reward summaries should still be logged).
    distill_groups = [g for g in trajectory_groups if g.group_role in COMBINED_DISTILL_ROLES]
    if distill_groups:
        import numpy as _np

        all_rewards = [traj.reward for g in distill_groups for traj in g.trajectories if traj.reward is not None]
        if all_rewards:
            adv_metrics["train/gsd_distill_reward/mean"] = _np.mean(all_rewards)
            adv_metrics["train/gsd_distill_reward/min"] = _np.min(all_rewards)
            adv_metrics["train/gsd_distill_reward/max"] = _np.max(all_rewards)
            adv_metrics["train/gsd_distill_reward/std"] = _np.std(all_rewards)
            adv_metrics["train/gsd_distill_count"] = len(all_rewards)

    # --- Step 2: Build datums per role ---
    datums_dict: dict[str, list] = defaultdict(list)

    for group in trajectory_groups:
        role = group.group_role
        if role in COMBINED_DISTILL_ROLES:
            for traj in group.trajectories:
                for step in traj.steps:
                    teacher_topk = (step.info or {}).get("teacher_topk")

                    # Infer K from teacher_topk or fall back to 20
                    K = len(teacher_topk["topk_ids"][0]) if teacher_topk else 20

                    datum = build_combined_gsd_datum(
                        prompt_ids=step.prompt_ids,
                        response_ids=step.response_ids,
                        teacher_topk=teacher_topk,
                        is_advantages=step.advantage if isinstance(step.advantage, list) else None,
                        is_old_logprobs=step.logprobs if step.advantage is not None else None,
                        K=K,
                    )
                    datums_dict["gsd_distill"].append(datum)
        else:
            for traj in group.trajectories:
                traj_datums = trajectory_to_datums(
                    traj,
                    router_replay=algorithm_config.router_replay,
                )
                datums_dict[role].extend(traj_datums)

    return datums_dict, adv_metrics
