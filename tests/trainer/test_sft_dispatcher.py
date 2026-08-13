"""Tests for the unified SFT dispatcher + TinkerSFTBackend config/validation.

These avoid the tinker stack: only ``fit()`` needs it, and it is patched.
"""

import pytest
from omegaconf import OmegaConf

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


def test_build_config_logger_override():
    """spec.logger overrides the yaml default trainer.logger for tinker."""
    cfg = TinkerSFTBackend(_spec(logger=["console", "wandb"])).build_config()
    assert list(cfg.trainer.logger) == ["console", "wandb"]


def test_build_config_logger_default_when_none():
    """spec.logger=None keeps the tinker.yaml default (['console'])."""
    cfg = TinkerSFTBackend(_spec()).build_config()
    assert list(cfg.trainer.logger) == ["console"]


def test_verl_build_config_filters_ui_logger():
    """verl's Tracking can't do 'ui'; build_config drops it (keeps the rest)."""
    pytest.importorskip("verl")
    from rllm.trainer.sft.verl_backend import VerlSFTBackend

    cfg = VerlSFTBackend(_spec(logger=["console", "wandb", "ui"])).build_config()
    assert list(cfg.trainer.logger) == ["console", "wandb"]


def test_verl_build_config_default_logger():
    """verl falls back to ['console'] when spec.logger is None."""
    pytest.importorskip("verl")
    from rllm.trainer.sft.verl_backend import VerlSFTBackend

    cfg = VerlSFTBackend(_spec()).build_config()
    assert list(cfg.trainer.logger) == ["console"]


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


def test_tinker_family_rejects_hf_template_without_native_serving_parity():
    from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend

    for cls in (TinkerSFTBackend, FireworksSFTBackend):
        with pytest.raises(SFTConfigError, match="train/serve mismatch"):
            cls(_spec(tokenize_method="hf_template")).validate_spec()


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


def test_dispatch_verl_uses_distributed_launcher(monkeypatch):
    """verl is distributed: train() routes to the torchrun launcher, not fit()."""
    from rllm.trainer.sft.verl_backend import VerlSFTBackend

    monkeypatch.delenv("RLLM_SFT_IN_TORCHRUN", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(VerlSFTBackend, "validate_spec", lambda self: None)
    monkeypatch.setattr(VerlSFTBackend, "build_config", lambda self: None)
    monkeypatch.setattr(VerlSFTBackend, "prepare_data", lambda self: None)
    launched = []
    monkeypatch.setattr(AgentSFTTrainer, "_launch_distributed", lambda self, be: launched.append(be.name))
    AgentSFTTrainer(_spec(), backend="verl").train()
    assert launched == ["verl"]


def test_dispatch_verl_runs_fit_inside_torchrun(monkeypatch):
    """Inside an existing process group, verl runs fit() directly (no relaunch)."""
    from rllm.trainer.sft.verl_backend import VerlSFTBackend

    monkeypatch.setenv("RLLM_SFT_IN_TORCHRUN", "1")
    monkeypatch.setattr(VerlSFTBackend, "validate_spec", lambda self: None)
    monkeypatch.setattr(VerlSFTBackend, "build_config", lambda self: None)
    monkeypatch.setattr(VerlSFTBackend, "prepare_data", lambda self: None)
    calls = []
    monkeypatch.setattr(VerlSFTBackend, "fit", lambda self: calls.append("fit"))
    monkeypatch.setattr(AgentSFTTrainer, "_launch_distributed", lambda self, be: calls.append("launch"))
    AgentSFTTrainer(_spec(), backend="verl").train()
    assert calls == ["fit"]


def test_dispatch_unknown_backend_raises():
    with pytest.raises(SFTConfigError, match="Unknown SFT backend"):
        AgentSFTTrainer(_spec(), backend="nope").train()


def test_verl_build_config_maps_spec():
    """VerlSFTBackend translates the SFTSpec into verl's sft_trainer_engine schema."""
    pytest.importorskip("verl")
    from rllm.trainer.sft.verl_backend import VerlSFTBackend

    spec = _spec(lr=2e-5, epochs=2, batch_size=16, max_length=4096, tokenize_method="cumulative", lr_schedule="cosine", lora_rank=0, overrides={"trainer": {"n_gpus_per_node": 4}})
    cfg = VerlSFTBackend(spec).build_config()
    assert cfg.model.path == spec.model
    assert cfg.model.lora_rank == 0  # full FT
    assert cfg.data.train_batch_size == 16
    assert cfg.data.max_length == 4096
    assert cfg.data.pad_mode == "no_padding"
    assert cfg.data.messages_key == "messages"
    assert cfg.data.custom_cls.path == "pkg://rllm.trainer.verl.sft_dataset"
    assert cfg.data.custom_cls.name == "RLLMSFTDataset"
    assert cfg.data.rllm.tokenize_and_mask_method == "cumulative"
    assert cfg.optim.lr == 2e-5
    assert cfg.optim.lr_scheduler_type == "cosine"
    assert cfg.trainer.total_epochs == 2
    assert cfg.trainer.n_gpus_per_node == 4  # routed from --gpus via overrides


def test_verl_linear_schedule_falls_back_to_cosine():
    pytest.importorskip("verl")
    from rllm.trainer.sft.verl_backend import VerlSFTBackend

    cfg = VerlSFTBackend(_spec(lr_schedule="linear")).build_config()
    assert cfg.optim.lr_scheduler_type == "cosine"


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


@pytest.mark.parametrize(
    "spec_kwargs, expect",
    [
        # A Fireworks base-model swap replaces model.name when the tokenizer +
        # shape move with it (a bare HF --model keeps the template — covered by
        # test_fireworks_build_config_uses_fireworks_template above).
        pytest.param(
            dict(
                model="accounts/fireworks/models/custom",
                overrides={
                    "model": {"tokenizer_model": "Qwen/Qwen3.5-9B"},
                    "fireworks_config": {"policy_trainer_shape_id": "accounts/fireworks/trainingShapes/custom-256k-lora"},
                },
            ),
            {"name": "accounts/fireworks/models/custom"},
            id="fw-path-swap-replaces-model",
        ),
        # lora_rank=0 derives the POLICY_TRAINER (non ``-lora``) sibling shape and
        # threads rank 0 + the derived shape into the provision document.
        pytest.param(
            dict(lora_rank=0),
            {
                "name": "accounts/fireworks/models/qwen3p5-9b",
                "lora_rank": 0,
                "shape": "accounts/fireworks/trainingShapes/qwen3p5-9b-256k",
                "infra_shape": "accounts/fireworks/trainingShapes/qwen3p5-9b-256k",
                "infra_lora_rank": 0,
            },
            id="rank0-derives-full-shape",
        ),
        # lora_rank=0 with an explicit non-``-lora`` shape passes through; the
        # swapped model + tokenizer are honored.
        pytest.param(
            dict(
                model="accounts/fireworks/models/qwen3p6-35b-a3b",
                lora_rank=0,
                overrides={
                    "model": {"tokenizer_model": "Qwen/Qwen3.6-35B-A3B"},
                    "fireworks_config": {"policy_trainer_shape_id": "accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-256k"},
                },
            ),
            {
                "name": "accounts/fireworks/models/qwen3p6-35b-a3b",
                "tokenizer": "Qwen/Qwen3.6-35B-A3B",
                "shape": "accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-256k",
            },
            id="rank0-explicit-full-shape-passthrough",
        ),
        # A positive LoRA rank keeps the ``-lora`` (LORA_TRAINER) shape.
        pytest.param(
            dict(lora_rank=8),
            {"shape": "accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora"},
            id="lora-rank-keeps-lora-shape",
        ),
        # Swapping the base model via --model without moving tokenizer + shape
        # is the silent wrong-tokenizer / wrong-shape trap → fail fast.
        pytest.param(
            dict(model="accounts/fireworks/models/qwen3p6-35b-a3b"),
            {"raises": "tokenizer_model"},
            id="swap-via-model-missing-raises",
        ),
        # The same trap through overrides model.name (e.g. --config) is caught.
        pytest.param(
            dict(overrides={"model": {"name": "accounts/fireworks/models/qwen3p6-35b-a3b"}}),
            {"raises": "tokenizer_model"},
            id="swap-via-overrides-name-missing-raises",
        ),
        # Overrides may arrive as a DictConfig (programmatic callers); a complete
        # tokenizer + shape override must not be rejected.
        pytest.param(
            dict(
                model="accounts/fireworks/models/qwen3p6-35b-a3b",
                overrides=OmegaConf.create(
                    {
                        "model": {"tokenizer_model": "Qwen/Qwen3.6-35B-A3B"},
                        "fireworks_config": {"policy_trainer_shape_id": "accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-256k-lora"},
                    }
                ),
            ),
            {"tokenizer": "Qwen/Qwen3.6-35B-A3B"},
            id="swap-via-dictconfig-overrides-builds",
        ),
    ],
)
def test_fireworks_build_config_resolves_shape_and_tokenizer(spec_kwargs, expect):
    """build_config resolves model.name / tokenizer / training shape as a function
    of lora_rank + overrides: rank 0 selects the POLICY_TRAINER sibling shape,
    a positive rank keeps the ``-lora`` shape, and a base-model swap without a
    matching tokenizer + shape fails fast (via ``--model`` or overrides, plain
    dict or DictConfig)."""
    from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend

    if "raises" in expect:
        with pytest.raises(SFTConfigError, match=expect["raises"]):
            FireworksSFTBackend(_spec(**spec_kwargs)).build_config()
        return
    cfg = FireworksSFTBackend(_spec(**spec_kwargs)).build_config()
    if "name" in expect:
        assert cfg.model.name == expect["name"]
    if "tokenizer" in expect:
        assert cfg.model.tokenizer_model == expect["tokenizer"]
    if "lora_rank" in expect:
        assert cfg.model.lora_rank == expect["lora_rank"]
    if "shape" in expect:
        assert cfg.fireworks_config.policy_trainer_shape_id == expect["shape"]
    if "infra_shape" in expect or "infra_lora_rank" in expect:
        doc = OmegaConf.to_container(cfg.fireworks_infra, resolve=True)
        if "infra_shape" in expect:
            assert doc["trainers"]["policy"]["training_shape_id"] == expect["infra_shape"]
        if "infra_lora_rank" in expect:
            assert doc["common"]["lora_rank"] == expect["infra_lora_rank"]


@pytest.mark.parametrize(
    "spec_kwargs, expect",
    [
        pytest.param(
            {},
            {
                "base_model": "accounts/fireworks/models/qwen3p5-9b",
                "tokenizer_model": "Qwen/Qwen3.5-9B",
                "serverless": False,
                "training_shape_id": "accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora",
            },
            id="lora",
        ),
        pytest.param(
            dict(
                model="accounts/fireworks/models/qwen3p6-35b-a3b",
                lora_rank=0,
                overrides={
                    "model": {"tokenizer_model": "Qwen/Qwen3.6-35B-A3B"},
                    "fireworks_config": {"policy_trainer_shape_id": "accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-256k"},
                },
            ),
            {
                "base_model": "accounts/fireworks/models/qwen3p6-35b-a3b",
                "tokenizer_model": "Qwen/Qwen3.6-35B-A3B",
                "lora_rank": 0,
                "training_shape_id": "accounts/fireworks/trainingShapes/qwen3p6-35b-a3b-256k",
            },
            id="full_param",
        ),
    ],
)
def test_fireworks_provision_doc_parses_sft(spec_kwargs, expect):
    """The fireworks_infra doc must parse offline into a valid SFT provision config
    on the training-shape path (no network, no superuser), for both LoRA and
    full-parameter (lora_rank=0) runs."""
    pytest.importorskip("training.provision")
    import tempfile
    from pathlib import Path

    import yaml
    from training.provision import load_yaml_provision

    from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend

    cfg = FireworksSFTBackend(_spec(**spec_kwargs)).build_config()
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
    assert pc.base_model == expect["base_model"]
    assert pc.tokenizer_model == expect["tokenizer_model"]
    assert pc.trainer.training_shape_id == expect["training_shape_id"]
    if "serverless" in expect:
        assert pc.serverless is expect["serverless"]
    if "lora_rank" in expect:
        assert pc.lora_rank == expect["lora_rank"]


def test_fireworks_warmup_default_off_and_overridable():
    """Warmup defaults OFF (the fireworks yaml knob was live-but-0), and both the
    cosine schedule and an overrides warmup ratio reach the fireworks config the
    fit loop reads."""
    from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend

    cfg = FireworksSFTBackend(_spec(lr_schedule="cosine")).build_config()
    assert cfg.optim.lr_scheduler == "cosine"
    assert cfg.optim.warmup_steps_ratio == 0.0
    cfg2 = FireworksSFTBackend(_spec(lr_schedule="cosine", overrides={"optim": {"warmup_steps_ratio": 0.1}})).build_config()
    assert cfg2.optim.warmup_steps_ratio == pytest.approx(0.1)


def test_fireworks_inherits_validation():
    from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend

    bad = Dataset(data=[{"text": "x"}], name="bad", split="train")
    with pytest.raises(SFTConfigError):
        FireworksSFTBackend(_spec(train_dataset=bad)).validate_spec()


def test_default_model_is_qwen35_4b():
    # SFTSpec default + both backend templates resolve to the same default model.
    assert SFTSpec(train_dataset=_ds()).model == "Qwen/Qwen3.5-4B"
    assert TinkerSFTBackend(SFTSpec(train_dataset=_ds())).build_config().model.name == "Qwen/Qwen3.5-4B"


def _structured_ds(n: int = 2):
    """T1-shape structured rows: parts-list content + per-message trainable flags.

    This is the tinker-only schema (rendered via ``rllm.data.sft_schema``); verl's
    parquet/messages path can't consume it, so verl must reject such a spec.
    """
    rows = [
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": f"q{i}"}], "trainable": False},
                {"role": "assistant", "content": [{"type": "text", "text": f"a{i}"}], "trainable": True},
            ]
        }
        for i in range(n)
    ]
    return Dataset(data=rows, name="structured", split="train")


def test_verl_rejects_structured_rows():
    """verl must reject structured (schema) rows and point at the tinker backend."""
    from rllm.trainer.sft.verl_backend import VerlSFTBackend

    spec = _spec(train_dataset=_structured_ds())
    with pytest.raises(SFTConfigError, match="tinker"):
        VerlSFTBackend(spec).validate_spec()


@pytest.mark.parametrize("think_row_index", [0, 1])
def test_verl_keeps_legacy_inline_think_text_regardless_of_row_order(
    think_row_index,
):
    from rllm.trainer.sft.verl_backend import VerlSFTBackend

    plain = {
        "messages": [
            {"role": "user", "content": "plain"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    inline = {
        "messages": [
            {"role": "user", "content": "reason"},
            {
                "role": "assistant",
                "content": "<think>work</think>answer",
            },
        ]
    }
    rows = [plain, plain.copy()]
    rows[think_row_index] = inline
    dataset = Dataset(data=rows, name="legacy-think-text", split="train")

    VerlSFTBackend(_spec(train_dataset=dataset)).validate_spec()


def test_verl_rejects_hosted_override_paths_in_one_actionable_error():
    from rllm.trainer.sft.verl_backend import VerlSFTBackend

    overrides = OmegaConf.create(
        {
            "trainer": {"max_steps": 12},
            "data": {
                "rllm": {
                    "group_by_length": True,
                    "length_group_factor": 4,
                    "group_by_length_factor": 4,
                    "overlength_policy": "error",
                    "loss_reduction": "token_mean",
                    "loss_normalization": "token_mean",
                    "strip_thinking_from_history": True,
                    "strip_tool_history": True,
                }
            },
            "optim": {
                "warmup_steps": 4,
                "warmup_steps_ratio": 0.1,
                "warmup_ratio": 0.1,
                "min_lr": 1e-6,
                "grad_clip_norm": 1.0,
            },
        }
    )

    with pytest.raises(SFTConfigError) as exc_info:
        VerlSFTBackend(_spec(overrides=overrides)).validate_spec()

    message = str(exc_info.value)
    rejected_paths = (
        "trainer.max_steps",
        "data.rllm.group_by_length",
        "data.rllm.length_group_factor",
        "data.rllm.group_by_length_factor",
        "data.rllm.overlength_policy",
        "data.rllm.loss_reduction",
        "data.rllm.loss_normalization",
        "data.rllm.strip_thinking_from_history",
        "data.rllm.strip_tool_history",
        "optim.warmup_steps",
        "optim.warmup_steps_ratio",
        "optim.warmup_ratio",
        "optim.min_lr",
        "optim.grad_clip_norm",
    )
    assert message.startswith("verl cannot use hosted-backend SFT override keys:")
    for path in rejected_paths:
        assert f"- {path}:" in message
    for native_path in (
        "trainer.total_training_steps",
        "data.truncation",
        "optim.lr_warmup_steps",
        "optim.lr_warmup_steps_ratio",
        "optim.min_lr_ratio",
        "optim.clip_grad",
    ):
        assert native_path in message
    assert "global token mean" in message
    assert "dynamic token batching" in message
    assert "renderer/history policy" in message


def test_dispatch_verl_rejects_hosted_overrides_before_build_or_launch(monkeypatch):
    from rllm.trainer.sft.verl_backend import VerlSFTBackend

    calls = []
    monkeypatch.setattr(VerlSFTBackend, "build_config", lambda self: calls.append("build"))
    monkeypatch.setattr(VerlSFTBackend, "prepare_data", lambda self: calls.append("prepare_data"))
    monkeypatch.setattr(AgentSFTTrainer, "_launch_distributed", lambda self, backend: calls.append("launch"))

    with pytest.raises(SFTConfigError, match=r"trainer\.max_steps"):
        AgentSFTTrainer(_spec(overrides={"trainer": {"max_steps": 12}}), backend="verl").train()
    assert calls == []


def test_verl_accepts_native_override_paths():
    pytest.importorskip("verl")
    from rllm.trainer.sft.verl_backend import VerlSFTBackend

    backend = VerlSFTBackend(
        _spec(
            overrides={
                "trainer": {"total_training_steps": 12},
                "data": {"truncation": "error", "rllm": {"tokenize_and_mask_method": "stepwise"}},
                "optim": {
                    "lr_warmup_steps": 4,
                    "lr_warmup_steps_ratio": 0.1,
                    "min_lr_ratio": 0.2,
                    "clip_grad": 1.0,
                    "weight_decay": 0.1,
                },
            }
        )
    )

    backend.validate_spec()
    cfg = backend.build_config()
    assert cfg.trainer.total_training_steps == 12
    assert cfg.data.truncation == "error"
    assert cfg.data.rllm.tokenize_and_mask_method == "stepwise"
    assert cfg.optim.lr_warmup_steps == 4
    assert cfg.optim.lr_warmup_steps_ratio == pytest.approx(0.1)
    assert cfg.optim.min_lr_ratio == pytest.approx(0.2)
    assert cfg.optim.clip_grad == pytest.approx(1.0)
    assert cfg.optim.weight_decay == pytest.approx(0.1)


# -- full-parameter (lora_rank=0) support ------------------------------------
# (build_config shape/tokenizer resolution + the provision doc for rank 0 are
# covered by the parametrized tests above; here: spec + validate_spec gating.)


@pytest.mark.parametrize("rank, valid", [(-1, False), (0, True)])
def test_spec_lora_rank_validation(rank, valid):
    """SFTSpec rejects a negative lora_rank; 0 (full-parameter) is accepted."""
    if valid:
        assert _spec(lora_rank=rank).lora_rank == rank
    else:
        with pytest.raises(ValueError, match="lora_rank"):
            _spec(lora_rank=rank)


@pytest.mark.parametrize(
    "spec_kwargs",
    [
        pytest.param(dict(lora_rank=0), id="direct-field"),
        pytest.param(dict(overrides={"model": {"lora_rank": 0}}), id="via-overrides"),
    ],
)
def test_tinker_rejects_full_finetune(spec_kwargs):
    """Tinker's SDK is LoRA-only (``create_lora_training_client``); rank 0 —
    whether a spec field or set via overrides (e.g. --config) — fails fast with a
    pointer at the full-parameter backends."""
    with pytest.raises(SFTConfigError, match="fireworks"):
        TinkerSFTBackend(_spec(**spec_kwargs)).validate_spec()


def test_fireworks_validate_allows_full_finetune():
    from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend

    FireworksSFTBackend(_spec(lora_rank=0)).validate_spec()  # no raise
