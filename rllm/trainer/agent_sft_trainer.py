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

# Backends available today.
_IMPLEMENTED = {"tinker", "fireworks", "verl"}
_PLANNED: set[str] = set()


def _inside_torchrun() -> bool:
    """True when already running inside a torchrun/distributed process group."""
    return os.environ.get("RLLM_SFT_IN_TORCHRUN") == "1" or "RANK" in os.environ


class AgentSFTTrainer:
    """Dispatches an :class:`SFTSpec` to the selected SFT backend."""

    def __init__(self, spec: SFTSpec, backend: str = "tinker"):
        self.spec = spec
        self.backend_name = backend
        self._backend: SFTBackend | None = None

    def prepare(self) -> SFTBackend:
        """Instantiate + configure the backend without provisioning/training.

        Runs ``validate_spec`` → ``build_config`` → ``prepare_data`` (all local,
        no network) and caches the backend so callers can read the resolved
        ``backend.config`` (e.g. for a CLI summary) before ``train()``.
        """
        if self._backend is None:
            backend = self._make_backend()
            backend.validate_spec()
            backend.build_config()
            backend.prepare_data()
            self._backend = backend
        return self._backend

    def train(self) -> None:
        backend = self.prepare()
        if backend.requires_distributed and not _inside_torchrun():
            self._launch_distributed(backend)
        else:
            backend.fit()

    def _make_backend(self) -> SFTBackend:
        name = self.backend_name
        if name == "tinker":
            from rllm.trainer.sft.tinker_backend import TinkerSFTBackend

            return TinkerSFTBackend(self.spec)
        if name == "fireworks":
            from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend

            return FireworksSFTBackend(self.spec)
        if name == "verl":
            from rllm.trainer.sft.verl_backend import VerlSFTBackend

            return VerlSFTBackend(self.spec)
        if name in _PLANNED:
            raise SFTConfigError(f"SFT backend {name!r} is not wired yet. Use backend='tinker' or 'fireworks' for now.")
        raise SFTConfigError(f"Unknown SFT backend {name!r}. Available: {', '.join(sorted(_IMPLEMENTED))}.")

    def _launch_distributed(self, backend: SFTBackend) -> None:
        """Spawn ``torchrun`` for a distributed backend (verl) and run its loop.

        ``prepare()`` has already materialized the data (parquet) and built the
        config; we serialize the config and re-enter under a torchrun process
        group via :mod:`rllm.trainer.sft.verl_entry`, which calls the backend's
        ``run_sft``. ``RLLM_SFT_IN_TORCHRUN`` marks the child so a nested
        dispatch would call ``fit()`` directly instead of relaunching.
        """
        import subprocess
        import sys

        cfg = backend.config
        nproc = int(cfg.trainer.get("n_gpus_per_node", 1) or 1)
        nnodes = int(cfg.trainer.get("nnodes", 1) or 1)
        config_path = backend.serialize_config()

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nnodes={nnodes}",
            f"--nproc_per_node={nproc}",
            "-m",
            "rllm.trainer.sft.verl_entry",
            "--config",
            config_path,
        ]
        env = {**os.environ, "RLLM_SFT_IN_TORCHRUN": "1"}
        # verl's worker refuses to start when ROCR_VISIBLE_DEVICES (an AMD/ROCm
        # var) and CUDA_VISIBLE_DEVICES are both set. On NVIDIA hosts ROCR is a
        # stray cluster default; drop it so CUDA_VISIBLE_DEVICES wins.
        if env.get("ROCR_VISIBLE_DEVICES") and env.get("CUDA_VISIBLE_DEVICES"):
            env.pop("ROCR_VISIBLE_DEVICES", None)
        print(f"[rllm sft] launching torchrun: nnodes={nnodes} nproc_per_node={nproc}\n  config: {config_path}")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise SFTConfigError(f"verl SFT torchrun exited with code {result.returncode}. See the torchrun output above.")
