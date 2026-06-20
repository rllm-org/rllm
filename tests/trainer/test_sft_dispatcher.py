"""Tests for the unified SFT dispatcher + TinkerSFTBackend config/validation.

These avoid the tinker stack: only ``fit()`` needs it, and it is patched.
"""

import pytest

from rllm.data import Dataset
from rllm.trainer.agent_sft_trainer import AgentSFTTrainer
from rllm.trainer.sft import SFTSpec
from rllm.trainer.sft.backend import SFTConfigError
from rllm.trainer.sft.tinker_backend import TinkerSFTBackend


def _ds(n: int = 4):
    rows = [{"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]} for i in range(n)]
    return Dataset(data=rows, name="toy", split="train")


def _spec(**kw):
    base = dict(model="Qwen/Qwen2.5-7B-Instruct", train_dataset=_ds())
    base.update(kw)
    return SFTSpec(**base)


def test_build_config_maps_spec():
    spec = _spec(lr=3e-4, epochs=5, batch_size=8, max_length=4096, tokenize_method="stepwise", lr_schedule="cosine", lora_rank=16)
    cfg = TinkerSFTBackend(spec).build_config()
    assert cfg.model.name == spec.model
    assert cfg.model.lora_rank == 16
    assert cfg.optim.lr == 3e-4
    assert cfg.optim.lr_scheduler == "cosine"
    assert cfg.trainer.total_epochs == 5
    assert cfg.data.train_batch_size == 8
    assert cfg.data.max_length == 4096
    assert cfg.data.rllm.tokenize_and_mask_method == "stepwise"


def test_output_dir_and_checkpoint_dir():
    backend = TinkerSFTBackend(_spec(output_dir="/tmp/ckpt-xyz"))
    cfg = backend.build_config()
    assert cfg.trainer.default_local_dir == "/tmp/ckpt-xyz"
    assert backend.checkpoint_dir == "/tmp/ckpt-xyz"


def test_overrides_escape_hatch():
    cfg = TinkerSFTBackend(_spec(overrides={"data": {"renderer_name": "qwen3"}})).build_config()
    assert cfg.data.renderer_name == "qwen3"


def test_validate_spec_accepts_messages():
    TinkerSFTBackend(_spec()).validate_spec()  # no raise


def test_validate_spec_rejects_missing_messages():
    bad = Dataset(data=[{"prompt": "x", "response": "y"}], name="bad", split="train")
    with pytest.raises(SFTConfigError):
        TinkerSFTBackend(_spec(train_dataset=bad)).validate_spec()


def test_validate_spec_rejects_empty():
    empty = Dataset(data=[], name="e", split="train")
    with pytest.raises(SFTConfigError):
        TinkerSFTBackend(_spec(train_dataset=empty)).validate_spec()


def test_dispatch_tinker_runs_lifecycle(monkeypatch):
    calls = []
    monkeypatch.setattr(TinkerSFTBackend, "fit", lambda self: calls.append("fit"))
    AgentSFTTrainer(_spec(), backend="tinker").train()
    assert calls == ["fit"]


@pytest.mark.parametrize("backend", ["verl", "fireworks"])
def test_dispatch_planned_backends_raise(backend):
    with pytest.raises(SFTConfigError, match="milestone 4"):
        AgentSFTTrainer(_spec(), backend=backend).train()


def test_dispatch_unknown_backend_raises():
    with pytest.raises(SFTConfigError, match="Unknown SFT backend"):
        AgentSFTTrainer(_spec(), backend="nope").train()
