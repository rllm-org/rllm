"""verl SFT backend.

Wraps verl 0.8.0's FSDP SFT trainer (``verl.trainer.sft_trainer``). Unlike the
hosted backends (tinker/fireworks), verl SFT is a monolithic FSDP loop that must
run inside a ``torchrun`` process group, so ``requires_distributed = True`` and
the dispatcher spawns the launcher (see
:meth:`rllm.trainer.agent_sft_trainer.AgentSFTTrainer._launch_distributed`).

The data seam is verl's ``data.custom_cls``: we point it at
:class:`rllm.trainer.verl.sft_dataset.RLLMSFTDataset`, which reads the curated
``{"messages": [...]}`` parquet rows and applies rLLM's tokenize/mask method.
``verl``/``torch`` are imported lazily inside :meth:`fit` (and the launcher entry)
so the module — and the dispatcher that imports it — stay importable without the
verl stack present.
"""

from __future__ import annotations

import logging
import os

from omegaconf import DictConfig, OmegaConf

from rllm.trainer.sft.backend import SFTBackend, SFTConfigError, validate_messages_dataset

logger = logging.getLogger(__name__)

# verl's hydra config package (composed for the full sft_trainer_engine schema).
_VERL_CONFIG_MODULE = "verl.trainer.config"
_VERL_CONFIG_NAME = "sft_trainer_engine"

# rLLM dataset injected via verl's data.custom_cls (pkg:// import path).
_CUSTOM_CLS_PATH = "pkg://rllm.trainer.verl.sft_dataset"
_CUSTOM_CLS_NAME = "RLLMSFTDataset"

# SFTSpec.lr_schedule -> verl optim.lr_scheduler_type. verl 0.8.0 ships
# constant/cosine/wsd; "linear" has no direct analogue, so fall back to cosine.
_LR_SCHEDULE_MAP = {"constant": "constant", "cosine": "cosine", "linear": "cosine"}


class VerlSFTBackend(SFTBackend):
    """Supervised fine-tuning on verl's FSDP trainer (multi-GPU, torchrun)."""

    name = "verl"
    requires_distributed = True

    # -- contract -----------------------------------------------------------

    def validate_spec(self) -> None:
        validate_messages_dataset(self.spec.train_dataset, "train")
        if self.spec.val_dataset is not None:
            validate_messages_dataset(self.spec.val_dataset, "val")
        if self.spec.lr_schedule not in _LR_SCHEDULE_MAP:
            raise SFTConfigError(f"Unsupported lr_schedule {self.spec.lr_schedule!r} for verl. Use one of {sorted(_LR_SCHEDULE_MAP)}.")
        if self.spec.lr_schedule == "linear":
            logger.warning("verl has no 'linear' LR schedule; using 'cosine' instead.")

    def _compose_base(self) -> DictConfig:
        """Compose verl's full ``sft_trainer_engine`` config (all sub-groups)."""
        from hydra import compose, initialize_config_module
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        with initialize_config_module(config_module=_VERL_CONFIG_MODULE, version_base=None):
            cfg = compose(config_name=_VERL_CONFIG_NAME)
        return cfg

    def build_config(self) -> DictConfig:
        """SFTSpec -> verl ``sft_trainer_engine`` DictConfig.

        Only keys already present in verl's schema are overridden, plus the one
        rLLM-specific addition ``data.rllm.tokenize_and_mask_method`` (read by
        ``RLLMSFTDataset``). LoRA is opt-in: ``lora_rank == 0`` => full FT.
        """
        spec = self.spec
        base = self._compose_base()
        # data.rllm is a new sub-tree verl doesn't declare; open struct to add it.
        OmegaConf.set_struct(base, False)

        lora_rank = int(spec.lora_rank or 0)
        max_token_len = max(int(spec.max_length), 8192)
        overrides = OmegaConf.create(
            {
                "model": {
                    "path": spec.model,
                    "lora_rank": lora_rank,
                    "lora_alpha": (2 * lora_rank if lora_rank else 16),
                    "use_remove_padding": True,
                    "enable_gradient_checkpointing": True,
                },
                "data": {
                    "train_batch_size": int(spec.batch_size),
                    "micro_batch_size_per_gpu": 1,
                    "max_length": int(spec.max_length),
                    "max_token_len_per_gpu": max_token_len,
                    "use_dynamic_bsz": True,
                    "messages_key": "messages",
                    "pad_mode": "no_padding",
                    "truncation": "right",
                    "custom_cls": {"path": _CUSTOM_CLS_PATH, "name": _CUSTOM_CLS_NAME},
                    "rllm": {"tokenize_and_mask_method": spec.tokenize_method},
                },
                "optim": {
                    "lr": float(spec.lr),
                    "lr_scheduler_type": _LR_SCHEDULE_MAP[spec.lr_schedule],
                },
                "trainer": {
                    "total_epochs": int(spec.epochs),
                    "save_freq": int(spec.save_freq),
                    "test_freq": int(spec.val_freq),
                    "project_name": spec.project,
                    "experiment_name": spec.experiment or "default",
                    "logger": ["console", "wandb"],
                    "default_local_dir": spec.output_dir or self._default_local_dir(),
                },
            }
        )
        cfg = OmegaConf.merge(base, overrides)
        if spec.overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(spec.overrides))
        self._config = cfg
        return cfg

    def _default_local_dir(self) -> str:
        from rllm import paths

        exp = self.spec.experiment or "default"
        return paths.rllm_path("sft_runs", self.spec.project, exp)

    @property
    def workdir(self) -> str:
        """Scratch dir for the materialized parquet + serialized launch config."""
        d = os.path.join(self.config.trainer.default_local_dir, "_verl_inputs")
        os.makedirs(d, exist_ok=True)
        return d

    def prepare_data(self) -> None:
        """Materialize the in-memory ``messages`` datasets to parquet on disk.

        verl's trainer builds its datasets itself from ``data.train_files`` /
        ``data.val_files``, so the curated rows must cross the torchrun boundary
        as parquet. We always re-write rather than trust a registry path so any
        SFTSpec source (registered dataset, ``--train-file``, curation output)
        works identically.
        """
        cfg = self.config
        train_path = os.path.join(self.workdir, "train.parquet")
        self._write_messages_parquet(self.spec.train_dataset, train_path)
        cfg.data.train_files = train_path
        if self.spec.val_dataset is not None:
            val_path = os.path.join(self.workdir, "val.parquet")
            self._write_messages_parquet(self.spec.val_dataset, val_path)
            cfg.data.val_files = val_path
        else:
            cfg.data.val_files = None
            # No val set -> disable verl's periodic validation.
            cfg.trainer.test_freq = -1

    @staticmethod
    def _write_messages_parquet(dataset, path: str) -> None:
        import pandas as pd

        # Normalize messages to plain list[dict] so the parquet round-trips
        # cleanly regardless of source (registry => list; pandas --train-file
        # => np.ndarray of dicts).
        def _norm(messages):
            return [dict(m) for m in messages]

        rows = [{"messages": _norm(row["messages"])} for row in dataset.get_data()]
        pd.DataFrame(rows).to_parquet(path, index=False)
        logger.info("Wrote %d SFT rows to %s", len(rows), path)

    @property
    def checkpoint_dir(self) -> str:
        return self.config.trainer.default_local_dir

    def serialize_config(self) -> str:
        """Persist the resolved config for the torchrun launcher; return its path."""
        path = os.path.join(self.workdir, "verl_sft_config.yaml")
        OmegaConf.save(self.config, path)
        return path

    def fit(self) -> None:
        """Run verl's FSDP SFT loop. Must be called inside a torchrun group."""
        from verl.trainer.sft_trainer import run_sft
        from verl.utils.device import auto_set_device

        cfg = self.config
        auto_set_device(cfg)
        run_sft(cfg)
