"""Deprecated trainer backends retained for backward compatibility."""

from __future__ import annotations

import warnings

warnings.warn(
    (
        "`rllm.trainer.deprecated` contains deprecated Tinker trainer backends and may be removed in a future release.\n"
        "If you are using the TinkerWorkflowTrainer, we recommend you migrate to the experimental unified trainer with the Tinker backend.\n"
        "The change to config will be minimal, and will become the standard way to train Tinker workflows in the future.\n"
        "See https://rllm-project.readthedocs.io/en/latest/experimental/unified-trainer.html for more details."
    ),
    FutureWarning,
    stacklevel=2
)

from rllm.trainer.deprecated.tinker_agent_trainer import TinkerAgentTrainer
from rllm.trainer.deprecated.tinker_sft_trainer import TinkerSFTTrainer
from rllm.trainer.deprecated.tinker_workflow_trainer import TinkerWorkflowTrainer

__all__ = ["TinkerAgentTrainer", "TinkerSFTTrainer", "TinkerWorkflowTrainer"]
