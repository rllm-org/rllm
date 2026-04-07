"""Generalized Self-Distillation (GSD) module.

Provides building blocks for the GSD training paradigm, which combines
on-policy reverse-KL distillation with supervised forward-KL distillation
from a hint-conditioned pseudo-teacher.

Quick start::

    from rllm.experimental.gsd import (
        DEFAULT_GSD_ADV_ESTIMATOR_MAP,
        GsdConfig,
        GsdWorkflow,
    )

    trainer = AgentTrainer(
        ...,
        workflow_class=GsdWorkflow,
        traj_group_adv_estimator_map=DEFAULT_GSD_ADV_ESTIMATOR_MAP,
    )
"""

from rllm.experimental.gsd.experience_store import EmbeddingExperienceStore
from rllm.experimental.gsd.losses import (
    DEFAULT_GSD_ADV_ESTIMATOR_MAP,
    build_gsd_estimator_map,
    build_topk_fkl_datum,
    compute_sampled_rkl_advantages,
    compute_student_logprobs_for_teacher_topk,
    compute_topk_rkl_at_position,
    score_teacher_for_response,
)
from rllm.experimental.gsd.workflow import GsdConfig, GsdWorkflow

__all__ = [
    "DEFAULT_GSD_ADV_ESTIMATOR_MAP",
    "EmbeddingExperienceStore",
    "GsdConfig",
    "GsdWorkflow",
    "build_gsd_estimator_map",
    "build_topk_fkl_datum",
    "compute_sampled_rkl_advantages",
    "compute_student_logprobs_for_teacher_topk",
    "compute_topk_rkl_at_position",
    "score_teacher_for_response",
]
