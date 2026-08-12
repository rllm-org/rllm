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
from collections.abc import Mapping

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

_HOSTED_OVERRIDE_HINTS = {
    "optim.grad_clip_norm": "use optim.clip_grad",
    "optim.min_lr": "use optim.min_lr_ratio, which is a fraction of optim.lr rather than an absolute learning rate",
    "optim.warmup_ratio": "use optim.lr_warmup_steps_ratio",
    "optim.warmup_steps": "use optim.lr_warmup_steps",
    "optim.warmup_steps_ratio": "use optim.lr_warmup_steps_ratio",
    "trainer.max_steps": "use trainer.total_training_steps",
}
_SUPPORTED_RLLM_DATA_OVERRIDE = "data.rllm.tokenize_and_mask_method"


def _iter_override_paths(node, prefix: tuple[str, ...] = ()):
    if not isinstance(node, Mapping):
        return
    for key, value in node.items():
        path = (*prefix, str(key))
        if isinstance(value, Mapping):
            yield from _iter_override_paths(value, path)
        else:
            yield ".".join(path)


def _hosted_override_hint(path: str) -> str | None:
    if path in _HOSTED_OVERRIDE_HINTS:
        return _HOSTED_OVERRIDE_HINTS[path]
    if path == "data.rllm":
        return f"use a mapping; {_SUPPORTED_RLLM_DATA_OVERRIDE} is the only supported rLLM-specific verl data key"
    if not path.startswith("data.rllm.") or path == _SUPPORTED_RLLM_DATA_OVERRIDE:
        return None

    key = path.removeprefix("data.rllm.")
    if key in {"group_by_length", "length_group_factor", "group_by_length_factor"}:
        return "unsupported on verl; its dynamic token batching (data.use_dynamic_bsz) is a different mechanism"
    if key == "overlength_policy":
        return "use data.truncation with one of 'error', 'left', or 'right'"
    if key in {"loss_reduction", "loss_normalization"}:
        return "unsupported because verl SFT is fixed to a global token mean"
    if key.startswith("strip_"):
        return "unsupported because verl has no renderer/history policy; curate the input messages or use a hosted backend"
    return f"{_SUPPORTED_RLLM_DATA_OVERRIDE} is the only supported rLLM-specific verl data key"


class VerlSFTBackend(SFTBackend):
    """Supervised fine-tuning on verl's FSDP trainer (multi-GPU, torchrun)."""

    name = "verl"
    requires_distributed = True

    # -- contract -----------------------------------------------------------

    def validate_spec(self) -> None:
        self._reject_hosted_overrides()
        validate_messages_dataset(
            self.spec.train_dataset,
            "train",
            allow_structural_inline_thinking_text=True,
        )
        if self.spec.val_dataset is not None:
            validate_messages_dataset(
                self.spec.val_dataset,
                "val",
                allow_structural_inline_thinking_text=True,
            )
        if self.spec.lr_schedule not in _LR_SCHEDULE_MAP:
            raise SFTConfigError(f"Unsupported lr_schedule {self.spec.lr_schedule!r} for verl. Use one of {sorted(_LR_SCHEDULE_MAP)}.")
        if self.spec.lr_schedule == "linear":
            logger.warning("verl has no 'linear' LR schedule; using 'cosine' instead.")
        self._reject_structured_rows(self.spec.train_dataset, "train")
        if self.spec.val_dataset is not None:
            self._reject_structured_rows(self.spec.val_dataset, "val")

    def _reject_hosted_overrides(self) -> None:
        if not self.spec.overrides:
            return
        overrides = OmegaConf.to_container(OmegaConf.create(self.spec.overrides), resolve=False)
        rejected = {path: hint for path in _iter_override_paths(overrides) if (hint := _hosted_override_hint(path)) is not None}
        if not rejected:
            return

        details = "\n".join(f"- {path}: {rejected[path]}" for path in sorted(rejected))
        raise SFTConfigError(
            f"verl cannot use hosted-backend SFT override keys:\n{details}\nUse the listed verl-native keys in --config, or choose --backend tinker/fireworks. These spellings are not translated."
        )

    @staticmethod
    def _reject_structured_rows(dataset, label: str) -> None:
        """Reject the tinker-only structured SFT schema on verl.

        verl's parquet/``messages`` path consumes plain ``{role, content:str}``
        turns; it can't render structured rows — parts-list content (thinking /
        tool-call parts) or per-message ``trainable`` flags. Fail fast and point
        at the hosted backends. Pure dict inspection over the first 64 rows (no
        verl import).
        """
        try:
            rows = dataset.get_data()[:64]
        except Exception:  # noqa: BLE001 - gate is best-effort
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            for msg in row.get("messages") or []:
                if not isinstance(msg, dict):
                    continue
                if isinstance(msg.get("content"), list) or "trainable" in msg:
                    raise SFTConfigError(
                        f"{label} dataset has structured SFT rows (parts-list content / per-message "
                        "'trainable' flags) representing reasoning (<think>) or tool-calls. These are "
                        "not supported on the verl backend yet — use --backend tinker (or fireworks) "
                        "for structured SFT."
                    )

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
        # verl SFT delegates to verl's own Tracking (inside torchrun), which
        # asserts on unknown backends and has no 'ui' logger — drop it with a
        # warning rather than crash the launcher.
        sft_logger = list(spec.logger) if spec.logger else ["console"]
        if "ui" in sft_logger:
            logger.warning("rllm UI logging is not supported on the verl SFT backend; dropping 'ui' from the logger list.")
            sft_logger = [b for b in sft_logger if b != "ui"]
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
                    "logger": sft_logger,
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
