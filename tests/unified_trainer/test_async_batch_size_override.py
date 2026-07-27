"""Tests for ``_enforce_async_train_batch_size`` in unified_trainer."""

from omegaconf import OmegaConf

from rllm.trainer.unified_trainer import _enforce_async_train_batch_size


def _cfg(train_batch_size: int):
    return OmegaConf.create(
        {
            "data": {"train_batch_size": train_batch_size},
            "rllm": {"data": {"train_batch_size": train_batch_size}},
        }
    )


def test_override_when_async_and_batch_size_not_one():
    cfg = _cfg(8)
    original = _enforce_async_train_batch_size(cfg, async_enabled=True)
    assert original == 8
    # Both the canonical and the mirrored verl-native namespaces are forced to 1.
    assert cfg.rllm.data.train_batch_size == 1
    assert cfg.data.train_batch_size == 1


def test_no_override_when_already_one():
    cfg = _cfg(1)
    assert _enforce_async_train_batch_size(cfg, async_enabled=True) is None
    assert cfg.rllm.data.train_batch_size == 1


def test_no_override_when_async_disabled():
    cfg = _cfg(8)
    assert _enforce_async_train_batch_size(cfg, async_enabled=False) is None
    # Untouched — sync training keeps the user's value.
    assert cfg.rllm.data.train_batch_size == 8
    assert cfg.data.train_batch_size == 8


def test_no_override_when_key_absent():
    cfg = OmegaConf.create({"rllm": {"data": {}}})
    assert _enforce_async_train_batch_size(cfg, async_enabled=True) is None
