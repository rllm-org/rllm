import pytest
from omegaconf import OmegaConf

from rllm.experimental.verl.verl_backend import VerlBackend


def _make_config(*, remote_enabled: bool, partial_rollout: bool):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {"rollout": {"mode": "async"}},
            "rllm": {
                "async_training": {
                    "enable": True,
                    "mini_batch_size": 8,
                    "fwd_bwd_group_size": 8,
                    "partial_rollout": partial_rollout,
                },
                "remote_runtime": {"enabled": remote_enabled},
                "stepwise_advantage": {"mode": "broadcast"},
                "algorithm": {"rollout_correction": {}},
            },
            "algorithm": {},
            "reward": {"reward_model": {"enable": False}},
        }
    )


def _backend(config) -> VerlBackend:
    # Bypass __init__ (heavy: builds dataloaders, worker groups) — validate_config
    # only reads self.config and self.is_separated.
    be = VerlBackend.__new__(VerlBackend)
    be.is_separated = True
    be.config = config
    return be


def test_partial_rollout_with_remote_runtime_raises():
    be = _backend(_make_config(remote_enabled=True, partial_rollout=True))
    with pytest.raises(ValueError, match="partial_rollout"):
        be.validate_config()


def test_partial_rollout_disabled_with_remote_runtime_ok():
    be = _backend(_make_config(remote_enabled=True, partial_rollout=False))
    be.validate_config()


def test_partial_rollout_without_remote_runtime_ok():
    be = _backend(_make_config(remote_enabled=False, partial_rollout=True))
    be.validate_config()
