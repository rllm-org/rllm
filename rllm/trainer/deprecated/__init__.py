"""Deprecated trainer backends retained for backward compatibility."""

from __future__ import annotations

import warnings

from rllm.trainer.deprecated.tinker_sft_trainer import TinkerSFTTrainer

warnings.warn(
    (
        "`rllm.trainer.deprecated` contains deprecated Tinker trainer backends "
        "and may be removed in a future release.\n"
        "If you are doing SFT, this path still works; for RL/workflow training "
        "use `rllm.experimental.unified_trainer.AgentTrainer` with "
        '`backend="tinker"`.\n'
        "See https://rllm-project.readthedocs.io/en/latest/experimental/"
        "unified-trainer.html for more details."
    ),
    FutureWarning,
    stacklevel=2,
)

__all__ = ["TinkerSFTTrainer"]
