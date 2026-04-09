"""Generalized Self-Distillation (GSD) module.

The active pipeline uses:

* **SFT-style CE loss** — student is trained on the frozen teacher's correct
  responses via Tinker's built-in ``cross_entropy`` loss (no Top-K).
* **IS loss** — student is trained on its own responses with per-token
  reverse-KL advantages computed against a *frozen reference* teacher
  (``teacher_lp - student_lp``) via ``importance_sampling``.
* **GRPO hint optimization** — hint-generation trajectories are grouped
  across tasks via a custom :mod:`rllm.experimental.gsd.grouping` hook and
  trained with REINFORCE / GRPO using ``R_T_avg - R_S_avg`` as the reward.
* **Case 2 fallback** — when the teacher doesn't help, student rollouts go
  through standard GRPO (``ppo`` loss with REINFORCE++ baseline).

See :mod:`rllm.experimental.gsd.legacy` for the original Top-K CE + custom
combined loss implementation.
"""

from rllm.experimental.gsd.grouping import make_gsd_grouping_hook
from rllm.experimental.gsd.losses import (
    CE_ROLE,
    GRPO_ROLE,
    GSD_ROLES,
    HINT_ROLE,
    IS_ROLE,
    build_gsd_estimator_map,
    build_is_datum,
    build_sft_style_ce_datum,
    compute_teacher_logprobs_for_response,
    kl_advantages_from_logprobs,
)
from rllm.experimental.gsd.teacher_ref import FrozenTeacherRef
from rllm.experimental.gsd.transform import gsd_transform_trajectory_groups_to_datums
from rllm.experimental.gsd.utils import (
    EmbeddingExperienceStore,
    HintPool,
    ScoringAccumulator,
)
from rllm.experimental.gsd.workflow import GsdConfig, GsdWorkflow

__all__ = [
    # Roles
    "CE_ROLE",
    "IS_ROLE",
    "GRPO_ROLE",
    "HINT_ROLE",
    "GSD_ROLES",
    # Workflow + config
    "GsdWorkflow",
    "GsdConfig",
    # Infrastructure
    "FrozenTeacherRef",
    "make_gsd_grouping_hook",
    "gsd_transform_trajectory_groups_to_datums",
    "build_gsd_estimator_map",
    # Datum builders / helpers
    "build_sft_style_ce_datum",
    "build_is_datum",
    "compute_teacher_logprobs_for_response",
    "kl_advantages_from_logprobs",
    # Optional utilities (re-exported from gsd.utils)
    "EmbeddingExperienceStore",
    "HintPool",
    "ScoringAccumulator",
]
