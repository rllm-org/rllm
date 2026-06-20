"""Unified SFT trainer (dispatcher).

``AgentSFTTrainer`` is the SFT analogue of the RL stack's ``AgentTrainer``: it
takes a backend-agnostic :class:`~rllm.trainer.sft.spec.SFTSpec`, picks a
backend, and runs it. Backends own their own training loop
(see :class:`rllm.trainer.sft.backend.SFTBackend`); this dispatcher owns the
backend-agnostic concerns: backend selection and launch topology (in-process for
hosted backends, a distributed launcher for verl).

    from rllm.trainer.agent_sft_trainer import AgentSFTTrainer
    from rllm.trainer.sft import SFTSpec

    spec = SFTSpec(model="Qwen/Qwen2.5-7B-Instruct", train_dataset=ds)
    AgentSFTTrainer(spec, backend="tinker").train()
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from rllm.trainer.sft.backend import SFTConfigError

if TYPE_CHECKING:
    from rllm.trainer.sft.backend import SFTBackend
    from rllm.trainer.sft.spec import SFTSpec

# Backends available today. verl/fireworks land in milestone 4.
_IMPLEMENTED = {"tinker"}
_PLANNED = {"verl", "fireworks"}


def _inside_torchrun() -> bool:
    """True when already running inside a torchrun/distributed process group."""
    return os.environ.get("RLLM_SFT_IN_TORCHRUN") == "1" or "RANK" in os.environ


class AgentSFTTrainer:
    """Dispatches an :class:`SFTSpec` to the selected SFT backend."""

    def __init__(self, spec: SFTSpec, backend: str = "tinker"):
        self.spec = spec
        self.backend_name = backend

    def train(self) -> None:
        backend = self._make_backend()
        backend.validate_spec()
        backend.build_config()
        backend.prepare_data()
        if backend.requires_distributed and not _inside_torchrun():
            self._launch_distributed(backend)
        else:
            backend.fit()

    def _make_backend(self) -> SFTBackend:
        name = self.backend_name
        if name == "tinker":
            from rllm.trainer.sft.tinker_backend import TinkerSFTBackend

            return TinkerSFTBackend(self.spec)
        if name in _PLANNED:
            raise SFTConfigError(f"SFT backend {name!r} is not wired yet (lands in milestone 4). Use backend='tinker' for now.")
        raise SFTConfigError(f"Unknown SFT backend {name!r}. Available: {', '.join(sorted(_IMPLEMENTED))}.")

    def _launch_distributed(self, backend: SFTBackend) -> None:
        # The torchrun launcher for distributed backends (verl) lands in
        # milestone 4. Until then, distributed backends must be run inside an
        # existing torchrun process group.
        raise SFTConfigError(f"Backend {backend.name!r} requires a distributed launcher (milestone 4). Run inside torchrun, or use backend='tinker'.")
