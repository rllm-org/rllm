"""Helpers for the TinkerBackend, mirroring verl's config-sync structure."""

from __future__ import annotations

from omegaconf import DictConfig

from rllm.trainer.algorithms.config import sync_shared_keys

# (tinker_native_path, rllm_path) — kept in parity by sync_config.
_SHARED_KEYS: list[tuple[str, str]] = [
    ("data.train_batch_size", "rllm.data.train_batch_size"),
    ("data.val_batch_size", "rllm.data.val_batch_size"),
    ("data.max_prompt_length", "rllm.data.max_prompt_length"),
    ("data.max_response_length", "rllm.data.max_response_length"),
    ("training.lr_schedule", "rllm.algorithm.lr_schedule"),
    ("training.warmup_steps_ratio", "rllm.algorithm.warmup_steps_ratio"),
    ("rollout_engine.accumulate_reasoning", "rllm.accumulate_reasoning"),
    ("rollout_engine.disable_thinking", "rllm.disable_thinking"),
]


def sync_config(config: DictConfig, hydra_overrides: list[str] | None = None) -> None:
    """Mirror rllm.* into tinker's native config over the shared-keys table."""
    sync_shared_keys(config, _SHARED_KEYS, hydra_overrides=hydra_overrides)
