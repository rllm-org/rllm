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


def test_dispatch_fireworks_runs_lifecycle(monkeypatch):
    from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend

    calls = []
    monkeypatch.setattr(FireworksSFTBackend, "fit", lambda self: calls.append("fit"))
    AgentSFTTrainer(_spec(), backend="fireworks").train()
    assert calls == ["fit"]


def test_dispatch_planned_backends_raise():
    with pytest.raises(SFTConfigError, match="not wired yet"):
        AgentSFTTrainer(_spec(), backend="verl").train()


def test_dispatch_unknown_backend_raises():
    with pytest.raises(SFTConfigError, match="Unknown SFT backend"):
        AgentSFTTrainer(_spec(), backend="nope").train()


def test_fireworks_build_config_uses_fireworks_template():
    from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend

    cfg = FireworksSFTBackend(_spec(lr=2e-5, lora_rank=8, max_length=4096)).build_config()
    # fireworks template carries fireworks_base_url (tinker's does not); hyperparams apply
    assert "fireworks_base_url" in cfg
    assert "tinker_base_url" not in cfg
    assert cfg.optim.lr == 2e-5
    assert cfg.model.lora_rank == 8
    assert cfg.data.max_length == 4096
    # Fireworks keeps its FW model path + HF tokenizer; a bare HF --model does NOT clobber it.
    assert cfg.model.name == "accounts/fireworks/models/qwen3p5-9b"
    assert cfg.model.tokenizer_model == "Qwen/Qwen3.5-9B"
    assert "fireworks_infra" in cfg


def test_fireworks_model_override_requires_fw_path():
    from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend

    # a FW path replaces the base model; a bare HF name does not
    cfg = FireworksSFTBackend(_spec(model="accounts/fireworks/models/custom")).build_config()
    assert cfg.model.name == "accounts/fireworks/models/custom"


def test_fireworks_provision_doc_parses_sft():
    """The fireworks_infra doc must parse offline into a valid SFT provision
    config on the training-shape path (no network, no superuser)."""
    pytest.importorskip("training.provision")
    import tempfile
    from pathlib import Path

    import yaml
    from omegaconf import OmegaConf
    from training.provision import load_yaml_provision

    from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend

    cfg = FireworksSFTBackend(_spec()).build_config()
    doc = OmegaConf.to_container(cfg.fireworks_infra, resolve=True)
    doc["common"]["learning_rate"] = float(cfg.optim.lr)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(doc, fh)
        p = Path(fh.name)
    try:
        mode, pc = load_yaml_provision(mode="sft", recipe=None, path=p)
    finally:
        p.unlink(missing_ok=True)
    assert mode == "sft"
    assert pc.base_model == "accounts/fireworks/models/qwen3p5-9b"
    assert pc.tokenizer_model == "Qwen/Qwen3.5-9B"
    assert pc.serverless is False
    assert pc.trainer.training_shape_id == "accounts/fireworks/trainingShapes/qwen3p5-9b-256k"


def test_fireworks_inherits_validation():
    from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend

    bad = Dataset(data=[{"text": "x"}], name="bad", split="train")
    with pytest.raises(SFTConfigError):
        FireworksSFTBackend(_spec(train_dataset=bad)).validate_spec()


def test_default_model_is_qwen35_4b():
    # SFTSpec default + both backend templates resolve to the same default model.
    assert SFTSpec(train_dataset=_ds()).model == "Qwen/Qwen3.5-4B"
    assert TinkerSFTBackend(SFTSpec(train_dataset=_ds())).build_config().model.name == "Qwen/Qwen3.5-4B"
