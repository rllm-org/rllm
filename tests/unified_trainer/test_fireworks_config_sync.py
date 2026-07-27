"""Tests for ``sync_config`` in ``rllm.trainer.fireworks.utils``.

The Fireworks engine reads the verl-native ``data.*`` knobs while UnifiedTrainer
reads the canonical ``rllm.data.*`` namespace. ``sync_config`` keeps the two in
parity so a run can set either side (and set ``rllm.data.*`` alone).
"""

from omegaconf import OmegaConf

from rllm.trainer.fireworks.utils import _SHARED_KEYS, sync_config


def _make_config():
    """Minimal post-compose fireworks config.

    ``data.*`` carries the fireworks backend yaml defaults; ``rllm.data.*``
    carries the rLLM base defaults — deliberately different so precedence is
    observable.
    """
    return OmegaConf.create(
        {
            "data": {
                "max_prompt_length": 30720,
                "max_response_length": 2048,
                "train_batch_size": 32,
                "val_batch_size": 32,
            },
            "rllm": {
                "data": {
                    "max_prompt_length": 2048,
                    "max_response_length": 30720,
                    "train_batch_size": 64,
                    "val_batch_size": -1,
                },
            },
        }
    )


def test_default_backfills_data_from_rllm():
    """No CLI overrides → verl-native ``data.*`` takes the ``rllm.data.*`` values."""
    cfg = _make_config()
    sync_config(cfg, hydra_overrides=[])
    for native_path, rllm_path in _SHARED_KEYS:
        assert OmegaConf.select(cfg, native_path) == OmegaConf.select(cfg, rllm_path)
    assert cfg.data.max_prompt_length == 2048
    assert cfg.data.max_response_length == 30720
    assert cfg.data.val_batch_size == -1


def test_rllm_cli_propagates_to_data():
    """User set ``rllm.data.*`` → the engine-facing ``data.*`` mirrors it."""
    cfg = _make_config()
    cfg.rllm.data.max_prompt_length = 57344
    cfg.rllm.data.max_response_length = 8192
    sync_config(
        cfg,
        hydra_overrides=[
            "rllm.data.max_prompt_length=57344",
            "rllm.data.max_response_length=8192",
        ],
    )
    assert cfg.data.max_prompt_length == 57344
    assert cfg.data.max_response_length == 8192


def test_native_cli_propagates_to_rllm():
    """User set the verl-native ``data.*`` → ``rllm.data.*`` mirrors it (back-compat)."""
    cfg = _make_config()
    cfg.data.max_prompt_length = 99999
    cfg.data.val_batch_size = 16
    sync_config(
        cfg,
        hydra_overrides=[
            "data.max_prompt_length=99999",
            "data.val_batch_size=16",
        ],
    )
    assert cfg.rllm.data.max_prompt_length == 99999
    assert cfg.rllm.data.val_batch_size == 16


def test_rllm_cli_wins_over_native_conflict():
    """User set both sides → the canonical ``rllm.data.*`` value wins on both."""
    cfg = _make_config()
    cfg.rllm.data.max_prompt_length = 57344
    cfg.data.max_prompt_length = 12345
    sync_config(
        cfg,
        hydra_overrides=[
            "rllm.data.max_prompt_length=57344",
            "data.max_prompt_length=12345",
        ],
    )
    assert cfg.data.max_prompt_length == 57344
    assert cfg.rllm.data.max_prompt_length == 57344
