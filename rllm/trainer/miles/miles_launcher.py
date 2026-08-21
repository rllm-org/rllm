"""Launcher for the Miles backend.

Miles ships as a Ray driver (``ray job submit train.py``), not a library, so rLLM
has to own Ray init before handing placement groups to Miles' ``create_*`` helpers.
Miles' own launcher additionally runs a ``pkill -9 sglang; ray stop --force``
preamble that we bypass -- stale SGLang processes from a crashed run are therefore
this process's problem, not Miles'.
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig, OmegaConf

from rllm.data import Dataset
from rllm.trainer.miles.miles_backend import MilesBackend
from rllm.trainer.miles.utils import sync_config
from rllm.trainer.unified_trainer import TrainerLauncher, UnifiedTrainer
from rllm.workflows.workflow import Workflow

logger = logging.getLogger(__name__)


class MilesTrainerLauncher(TrainerLauncher):
    """Scaffolds Ray, then hands off to the UnifiedTrainer."""

    def __init__(
        self,
        config: DictConfig,
        workflow_class: type[Workflow] | None = None,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        workflow_args: dict | None = None,
        **kwargs,
    ):
        super().__init__(config, workflow_class, train_dataset, val_dataset, workflow_args, **kwargs)

    def _hydra_overrides(self) -> list[str]:
        try:
            from hydra.core.hydra_config import HydraConfig

            return list(HydraConfig.get().overrides.task)
        except (ValueError, AttributeError, ImportError):
            return []

    def _init_ray(self) -> None:
        import ray

        if ray.is_initialized():
            logger.info("Ray already initialized; reusing the existing cluster.")
            return

        from rllm.trainer.ray_init_utils import get_forwarded_env_vars, get_ray_init_settings

        # Not verl's get_ppo_ray_runtime_env(): that sets VLLM_* defaults, which are
        # meaningless for Miles (SGLang). The env forwarding is the part worth reusing --
        # MEGATRON_ / SGLANG_ / NCCL_ / CUDA_ prefixes are exactly what Miles' actors need.
        env_vars = {"TOKENIZERS_PARALLELISM": "true", "NCCL_CUMEM_ENABLE": "0", **get_forwarded_env_vars()}
        ray.init(
            runtime_env={
                "env_vars": env_vars,
                # Driver-side monkey-patches do not reach worker processes, and the
                # advantages CP-slice patch has to run inside Miles' train workers.
                "worker_process_setup_hook": "rllm.trainer.miles.patch.apply_all_miles_patches",
            },
            **get_ray_init_settings(self.config),
        )

    def train(self):
        sync_config(self.config, self._hydra_overrides())
        OmegaConf.resolve(self.config)
        self._init_ray()

        trainer = None
        try:
            trainer = UnifiedTrainer(
                backend_cls=MilesBackend,
                config=self.config,
                workflow_class=self.workflow_class,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                workflow_args=self.workflow_args,
                store=self.store,
                **self.kwargs,
            )
            trainer.fit()
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user.")
        finally:
            if trainer is not None:
                trainer.shutdown()
