"""Unified SFT trainer: a backend-agnostic spec + per-backend implementations.

- :class:`~rllm.trainer.sft.spec.SFTSpec` — what to train (backend-agnostic).
- :class:`~rllm.trainer.sft.backend.SFTBackend` — the per-backend contract.
- :class:`rllm.trainer.agent_sft_trainer.AgentSFTTrainer` — the dispatcher that
  picks a backend and runs it.

Concrete backends (``TinkerSFTBackend``, …) are imported lazily by the
dispatcher so this package stays importable without torch/tinker/verl.
"""

from __future__ import annotations

from rllm.trainer.sft.backend import SFTBackend, SFTConfigError, validate_messages_dataset
from rllm.trainer.sft.spec import SFTSpec

__all__ = ["SFTSpec", "SFTBackend", "SFTConfigError", "validate_messages_dataset"]
