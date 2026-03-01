"""Compatibility exports for legacy tinker trainer imports."""

from rllm.trainer.deprecated import (
    TinkerAgentTrainer,
    TinkerSFTTrainer,
    TinkerWorkflowTrainer,
)

__all__ = ["TinkerAgentTrainer", "TinkerSFTTrainer", "TinkerWorkflowTrainer"]
