"""Experiential Reinforcement Learning (ERL) module.

Provides reusable building blocks for implementing the ERL training
paradigm (Shi et al., 2026) on top of rLLM's unified trainer.

Reference official implementation of ERL: https://github.com/microsoft/experiential_rl

Quick start::

    from rllm.experimental.erl import (
        DEFAULT_ERL_ADV_ESTIMATOR_MAP,
        ErlWorkflow,
    )

    trainer = AgentTrainer(
        ...,
        workflow_class=ErlWorkflow,
        traj_group_adv_estimator_map=DEFAULT_ERL_ADV_ESTIMATOR_MAP,
    )
"""

from rllm.experimental.erl.updater import ErlPromptUpdater
from rllm.experimental.erl.utils import (
    DEFAULT_ERL_ADV_ESTIMATOR_MAP,
    UPDATER_SYSTEM_PROMPT,
    calculate_raft_advantages,
    default_feedback,
    extract_prompt_from_response,
)
from rllm.experimental.erl.workflow import ErlConfig, ErlWorkflow

__all__ = [
    "DEFAULT_ERL_ADV_ESTIMATOR_MAP",
    "ErlConfig",
    "ErlPromptUpdater",
    "ErlWorkflow",
    "UPDATER_SYSTEM_PROMPT",
    "calculate_raft_advantages",
    "default_feedback",
    "extract_prompt_from_response",
]
