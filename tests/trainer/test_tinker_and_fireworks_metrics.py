"""Regression tests for Tinker KL metrics and Fireworks off-policy/warmup handling."""

import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from rllm.trainer.algorithms.config import AlgorithmConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_fake_tensor_data(values: list[float]):
    t = torch.tensor(values, dtype=torch.float32)
    return SimpleNamespace(to_torch=lambda: t, data=values)


class _StubModule(ModuleType):
    def __init__(self, name: str):
        super().__init__(name)
        self.__path__ = []  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return object


def _ensure_stub_modules(monkeypatch: pytest.MonkeyPatch, stubs: dict[str, dict[str, object]]) -> None:
    for name, attrs in stubs.items():
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
            continue
        except ModuleNotFoundError:
            pass
        mod = _StubModule(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            if parent_name in sys.modules:
                setattr(sys.modules[parent_name], child_name, mod)


def _import_tinker_metrics_module(monkeypatch: pytest.MonkeyPatch):
    _ensure_stub_modules(monkeypatch, {"tinker": {"Datum": object}})
    if "rllm.trainer.tinker.tinker_metrics_utils" in sys.modules:
        return importlib.reload(sys.modules["rllm.trainer.tinker.tinker_metrics_utils"])
    return importlib.import_module("rllm.trainer.tinker.tinker_metrics_utils")


def _import_fireworks_trainer_module(monkeypatch: pytest.MonkeyPatch):
    class _AdamParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _GradAccNormalization:
        NUM_LOSS_TOKENS = "NUM_LOSS_TOKENS"
        NUM_SEQUENCES = "NUM_SEQUENCES"

    _ensure_stub_modules(
        monkeypatch,
        {
            "tinker": {
                "Datum": object,
                "APIFuture": object,
                "SamplingClient": object,
                "ServiceClient": object,
            },
            "tinker.types": {
                "AdamParams": _AdamParams,
                "ModelInput": object,
                "OptimStepResponse": object,
            },
            "tinker.types.tensor_data": {"TensorData": object},
            "tinker_cookbook": {},
            "tinker_cookbook.model_info": {},
            "tinker_cookbook.renderers": {"Message": object},
            "tinker_cookbook.tokenizer_utils": {"Tokenizer": object},
            "tinker_cookbook.supervised": {},
            "tinker_cookbook.supervised.common": {
                "create_rightshifted_model_input_and_leftshifted_targets": lambda x: (x, x),
            },
            "fireworks": {},
            "fireworks.training": {},
            "fireworks.training.sdk": {
                "DeploymentSampler": object,
                "TrainerJobManager": object,
                "WeightSyncer": object,
            },
            "fireworks.training.sdk.client": {
                "GradAccNormalization": _GradAccNormalization,
            },
            "training": {},
            "training.utils": {},
            "training.utils.client": {"ReconnectableClient": object},
        },
    )

    path = _REPO_ROOT / "rllm" / "trainer" / "fireworks" / "fireworks_policy_trainer.py"
    spec = importlib.util.spec_from_file_location("_test_fireworks_policy_trainer", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tinker_kl_sample_train_v1_uses_signed_logprob_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    """``train/kl_sample_train_v1`` must use signed ``log p - log q`` (unbiased k1 KL)."""
    mod = _import_tinker_metrics_module(monkeypatch)

    # Exact 2-outcome distribution: p = [0.5, 0.5], q = [0.55, 0.45].
    # True KL(p || q) = 0.5 * log(0.5/0.55) + 0.5 * log(0.5/0.45) ~= 0.005025168.
    # Taking .abs() gives ~0.100335 (20x too large).
    p = torch.tensor([0.5, 0.5], dtype=torch.float32)
    q = torch.tensor([0.55, 0.45], dtype=torch.float32)
    sampling_lp = p.log().tolist()
    training_lp = q.log()

    datum = SimpleNamespace(
        loss_fn_inputs={
            "logprobs": _make_fake_tensor_data([0.0] + sampling_lp),
            "mask": _make_fake_tensor_data([0.0, 1.0, 1.0]),
        }
    )
    metrics = mod.compute_kl_and_entropy_metrics([datum], [torch.cat([torch.tensor([0.0]), training_lp])])

    expected_k1 = (p.log() - q.log()).mean().item()
    expected_k2 = (0.5 * (p.log() - q.log()) ** 2).mean().item()
    assert metrics["train/kl_sample_train_v1"] == pytest.approx(expected_k1, abs=1e-6)
    assert metrics["train/kl_sample_train_v2"] == pytest.approx(expected_k2, abs=1e-6)


def test_fireworks_offpolicy_k3_kl_stays_finite_on_extreme_log_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    """``offpolicy/k3_kl`` must clamp extreme log-ratios instead of overflowing to inf."""
    mod = _import_fireworks_trainer_module(monkeypatch)

    # old_lp - rollout_lp = 100.0 overflows float32 exp(100.0) = inf if unclamped.
    metrics = mod.FireworksPolicyTrainer._compute_offpolicy_metrics(
        old_logprobs=[[-0.1, -1.0]],
        rollout_logprobs=[[-100.1, -1.2]],
        masks=[[1, 1]],
    )
    assert torch.isfinite(torch.tensor(metrics["offpolicy/k3_kl"])).item()

    # Normal in-range values must match the exact Schulman k3 formula.
    normal = mod.FireworksPolicyTrainer._compute_offpolicy_metrics(
        old_logprobs=[[-0.5, -1.2]],
        rollout_logprobs=[[-0.6, -1.0]],
        masks=[[1, 1]],
    )
    lr = torch.tensor([-0.5 - (-0.6), -1.2 - (-1.0)], dtype=torch.float32)
    expected_k3 = (torch.exp(lr) - lr - 1.0).mean().item()
    assert normal["offpolicy/k3_kl"] == pytest.approx(expected_k3, abs=1e-6)


def test_fireworks_optim_step_forwards_absolute_warmup_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """``FireworksPolicyTrainer.optim_step`` must pass ``algorithm_config.warmup_steps``."""
    mod = _import_fireworks_trainer_module(monkeypatch)

    trainer = object.__new__(mod.FireworksPolicyTrainer)
    trainer.algorithm_config = AlgorithmConfig(
        lr_schedule="constant",
        warmup_steps=10,
        warmup_steps_ratio=0.5,
        loss_agg_mode="token-mean",
    )
    trainer.training_client = MagicMock()
    trainer.training_client.optim_step.return_value = SimpleNamespace(metrics={})

    # With total_steps=100, warmup_steps=10 should override warmup_steps_ratio=0.5 (=50 steps):
    # at step=5, lr multiplier = 5/10 = 0.5 (not 5/50 = 0.1).
    scheduled_lr, _ = asyncio.run(
        trainer.optim_step(
            step=5,
            total_steps=100,
            learning_rate=1.0,
        )
    )
    assert scheduled_lr == pytest.approx(0.5, abs=1e-6)
