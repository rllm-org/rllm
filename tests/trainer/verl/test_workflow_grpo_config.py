"""Validate supported workflow advantage modes without starting Ray workers."""

import pytest
from omegaconf import OmegaConf

from rllm.trainer.verl.agent_workflow_trainer import AgentWorkflowPPOTrainer


@pytest.mark.parametrize("enable, mode", [(False, "broadcast"), (False, "per_step"), (True, "broadcast"), (True, "per_step")])
def test_workflow_trainer_rejects_enabled_per_step(enable, mode):
    trainer = AgentWorkflowPPOTrainer.__new__(AgentWorkflowPPOTrainer)
    trainer.workflow_class = object
    trainer.use_rm = False
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {"hybrid_engine": True, "rollout": {"mode": "async"}},
            "rllm": {"rejection_sample": {"multiplier": 1}, "stepwise_advantage": {"enable": enable, "mode": mode}},
            "algorithm": {"adv_estimator": "grpo"},
        }
    )
    if enable and mode == "per_step":
        with pytest.raises(ValueError, match="per_step is unsupported"):
            trainer._validate_config()
    else:
        trainer._validate_config()
