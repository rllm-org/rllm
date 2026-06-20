"""Deprecated trainer backends retained for backward compatibility.

SFT now lives in :mod:`rllm.trainer.sft` (via
:class:`rllm.trainer.agent_sft_trainer.AgentSFTTrainer`); the old Tinker SFT
trainer/dataset have been removed. For RL/workflow training use
:class:`rllm.trainer.unified_trainer.AgentTrainer`.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "`rllm.trainer.deprecated` contains deprecated Tinker trainer backends and "
    "may be removed in a future release. For SFT use "
    "`rllm.trainer.agent_sft_trainer.AgentSFTTrainer`; for RL use "
    "`rllm.trainer.unified_trainer.AgentTrainer`.",
    FutureWarning,
    stacklevel=2,
)

__all__: list[str] = []
