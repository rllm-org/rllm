"""Helpers for the MilesBackend, mirroring verl's and tinker's config-sync structure."""

from __future__ import annotations

from omegaconf import DictConfig

from rllm.trainer.algorithms.config import sync_shared_keys

# (miles_native_path, rllm_path) -- kept in parity by sync_config.
#
# Note this is the *top-level mirror* table, not the Miles CLI mapping: flags that
# reach Miles' argv live in miles_config.SHARED_KEYS. These are the keys the rest of
# rLLM reads off the top level for the miles backend, the same way tinker does.
_SHARED_KEYS: list[tuple[str, str]] = [
    ("data.train_batch_size", "rllm.data.train_batch_size"),
    ("data.val_batch_size", "rllm.data.val_batch_size"),
    ("data.max_prompt_length", "rllm.data.max_prompt_length"),
    ("data.max_response_length", "rllm.data.max_response_length"),
]


def sync_config(config: DictConfig, hydra_overrides: list[str] | None = None) -> None:
    """Mirror rllm.* into the top-level namespace over the shared-keys table."""
    sync_shared_keys(config, _SHARED_KEYS, hydra_overrides=hydra_overrides)
