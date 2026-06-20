"""The SFT backend contract.

Mirrors the RL stack's dispatcher/launcher seam (``AgentTrainer`` /
``TrainerLauncher``) rather than its shared training loop: SFT backends do not
expose shareable per-step primitives (verl SFT is a monolithic FSDP loop under
``torchrun``; tinker SFT is an async future-pipelined loop), so each backend
**owns its own ``fit()``**. The shared layer is the spec, config translation,
data prep, and launch topology — handled by the dispatcher
(:class:`rllm.trainer.agent_sft_trainer.AgentSFTTrainer`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from rllm.trainer.sft.spec import SFTSpec


class SFTConfigError(Exception):
    """Raised for invalid SFT specs/config or unsupported backends."""


def validate_messages_dataset(dataset, label: str = "train") -> None:
    """Validate that *dataset* has the SFT ``messages`` schema.

    Mirrors the check the old experimental SFT CLI did, raising
    :class:`SFTConfigError` with an actionable message.
    """
    if dataset is None or len(dataset) == 0:
        raise SFTConfigError(f"{label} dataset is empty.")
    row = dataset[0]
    if "messages" not in row:
        raise SFTConfigError(f"{label} dataset is missing a 'messages' column. SFT datasets need a list of {{'role','content'}} turns per row.")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        raise SFTConfigError(f"{label} dataset: 'messages' must be a non-empty list of conversation turns.")
    first = messages[0]
    if not isinstance(first, dict) or "role" not in first or "content" not in first:
        raise SFTConfigError(f"{label} dataset: each message must have 'role' and 'content' keys.")


class SFTBackend(ABC):
    """Owns one complete SFT run for a given backend.

    Lifecycle (driven by the dispatcher): ``validate_spec`` → ``build_config`` →
    ``prepare_data`` → ``fit``.
    """

    name: str = "base"
    # True for backends that must run under a distributed launcher (e.g. verl
    # under torchrun); the dispatcher spawns the launcher for these.
    requires_distributed: bool = False

    def __init__(self, spec: SFTSpec):
        self.spec = spec

    @abstractmethod
    def validate_spec(self) -> None:
        """Validate the spec for this backend; raise SFTConfigError on problems."""

    @abstractmethod
    def build_config(self) -> DictConfig:
        """Translate the SFTSpec into this backend's native config."""

    @abstractmethod
    def prepare_data(self) -> None:
        """Materialize/resolve datasets into the form the backend's loop wants."""

    @abstractmethod
    def fit(self) -> None:
        """Run the full SFT training loop."""

    @property
    @abstractmethod
    def checkpoint_dir(self) -> str:
        """Where checkpoints are written."""
