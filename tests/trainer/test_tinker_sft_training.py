"""Tinker SFT dataset, optimizer, scheduling, and fit-loop contracts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

tinker = pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")

from rllm.data import Dataset  # noqa: E402
from rllm.trainer.sft import SFTSpec  # noqa: E402
from rllm.trainer.sft.backend import SFTConfigError  # noqa: E402
from rllm.trainer.sft.tinker_backend import (  # noqa: E402
    SFTResumeContract,
    TinkerSFTBackend,
    build_tinker_resume_contract,
    iter_training_batches,
    iter_training_batches_from_step,
    prepare_tinker_resume_contract,
    resolve_sft_optimizer_settings,
    validate_tinker_checkpoint_identity,
    validate_tinker_resume_cursor,
)
from rllm.trainer.sft.tinker_dataset import TinkerSFTDataset  # noqa: E402


class _TokenRenderer:
    """Small real Renderer boundary without a tokenizer dependency."""

    def build_supervised_example(self, messages, train_on_what):
        import torch

        n = int(messages[-1]["content"][0]["text"])
        return tinker.ModelInput.from_ints(list(range(n + 2))), torch.tensor(
            [0.0, *([1.0] * n), 0.0],
            dtype=torch.float32,
        )


def _length_dataset(lengths):
    return Dataset(
        data=[
            {
                "messages": [
                    {"role": "user", "content": "q", "trainable": False},
                    {"role": "assistant", "content": str(n), "trainable": True},
                ]
            }
            for n in lengths
        ],
        name="lengths",
        split="train",
    )


def test_token_mean_reduction_weights_every_assistant_token_equally():
    ds = TinkerSFTDataset(
        _length_dataset([2, 6]),
        renderer=_TokenRenderer(),
        batch_size=2,
        max_length=100,
        loss_reduction="token_mean",
    )
    batch = ds.get_batch(0)
    weights = [d.loss_fn_inputs["weights"].data for d in batch]

    assert sum(sum(w) for w in weights) == pytest.approx(1.0)
    positive = [x for w in weights for x in w if x > 0]
    assert len(positive) == 8
    assert positive == pytest.approx([1 / 8] * 8)


def test_sequence_mean_reduction_gives_each_trajectory_equal_weight():
    ds = TinkerSFTDataset(
        _length_dataset([2, 6]),
        renderer=_TokenRenderer(),
        batch_size=2,
        max_length=100,
        loss_reduction="sequence_mean",
    )
    weights = [d.loss_fn_inputs["weights"].data for d in ds.get_batch(0)]
    assert [sum(w) for w in weights] == pytest.approx([1.0, 1.0])


def test_default_loss_reduction_preserves_raw_per_token_weights():
    ds = TinkerSFTDataset(
        _length_dataset([2, 6]),
        renderer=_TokenRenderer(),
        batch_size=2,
        max_length=100,
    )
    weights = [d.loss_fn_inputs["weights"].data for d in ds.get_batch(0)]
    assert [sum(w) for w in weights] == pytest.approx([2.0, 6.0])


def test_dataset_rejects_nonpositive_batch_size():
    with pytest.raises(SFTConfigError, match="batch_size must be positive"):
        TinkerSFTDataset(
            _length_dataset([2]),
            renderer=_TokenRenderer(),
            batch_size=0,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_length": 1}, "max_length"),
        ({"overlength_policy": "drop"}, "overlength policy"),
        ({"loss_reduction": "batch_mean"}, "loss reduction"),
    ],
)
def test_dataset_rejects_invalid_render_settings_before_iteration(kwargs, match):
    with pytest.raises(SFTConfigError, match=match):
        TinkerSFTDataset(
            _length_dataset([2]),
            renderer=_TokenRenderer(),
            batch_size=1,
            **kwargs,
        )


def test_dataset_preflight_renders_every_batch_before_training():
    ds = TinkerSFTDataset(
        _length_dataset([2, 20]),
        renderer=_TokenRenderer(),
        batch_size=1,
        max_length=10,
    )

    with pytest.raises(SFTConfigError, match="train preflight.*dataset row 1.*max_length=10"):
        ds.preflight()


def test_fit_preflights_the_shuffled_epoch_order_before_tinker_client(
    monkeypatch,
    tmp_path,
):
    from tinker_cookbook import checkpoint_utils

    import rllm.trainer.sft.tinker_backend as backend_module

    # Identity batches [0, 2] / [0, 2] each have loss tokens. Epoch seed 0
    # produces [0, 0] / [2, 2], so the first submitted batch would be invalid.
    train = TinkerSFTDataset(
        _length_dataset([0, 2, 0, 2]),
        renderer=_TokenRenderer(),
        batch_size=2,
        max_length=100,
    )
    monkeypatch.setattr(
        backend_module,
        "build_sft_data",
        lambda *_: (object(), train, None),
    )
    monkeypatch.setattr(checkpoint_utils, "get_last_checkpoint", lambda *_: None)
    monkeypatch.setattr(
        tinker,
        "ServiceClient",
        lambda **_: pytest.fail("provider client must not be created"),
    )

    backend = TinkerSFTBackend(
        SFTSpec(
            train_dataset=_length_dataset([2]),
            output_dir=str(tmp_path),
            batch_size=2,
            overrides={"trainer": {"max_steps": 2}},
        )
    )
    backend.build_config()

    with pytest.raises(SFTConfigError, match="train preflight.*epoch 0.*batch 0"):
        asyncio.run(backend._fit_async())


def test_dataset_fingerprint_covers_messages_tools_rows_and_batch_size():
    first = _length_dataset([2, 6])
    second = _length_dataset([2, 7])
    a = TinkerSFTDataset(first, renderer=_TokenRenderer(), batch_size=2)
    a_again = TinkerSFTDataset(first, renderer=_TokenRenderer(), batch_size=2)
    changed_row = TinkerSFTDataset(second, renderer=_TokenRenderer(), batch_size=2)
    changed_batch = TinkerSFTDataset(first, renderer=_TokenRenderer(), batch_size=1)

    assert a.content_fingerprint() == a_again.content_fingerprint()
    assert a.content_fingerprint() != changed_row.content_fingerprint()
    assert a.content_fingerprint() != changed_batch.content_fingerprint()


@pytest.mark.parametrize("data_consumed", [-1, 3])
def test_dataset_resume_cursor_requires_an_exact_batch_boundary(data_consumed):
    dataset = TinkerSFTDataset(
        _length_dataset([2, 2, 2, 2, 2]),
        renderer=_TokenRenderer(),
        batch_size=2,
    )

    with pytest.raises(SFTConfigError, match="cursor.*non-negative|cursor.*exact"):
        dataset.step_for_data_cursor(data_consumed)


def test_training_batch_iterator_caps_mid_epoch_at_exact_max_steps():
    batches = list(
        iter_training_batches(
            n_batches=3262,
            total_epochs=1,
            start_epoch=0,
            start_batch=0,
            max_steps=1000,
        )
    )
    assert len(batches) == 1000
    assert batches[0] == (0, 0, 0)
    assert batches[-1] == (999, 0, 999)


def test_training_batch_iterator_resumes_at_next_unseen_batch():
    batches = list(
        iter_training_batches(
            n_batches=20,
            total_epochs=2,
            start_epoch=1,
            start_batch=3,
            max_steps=30,
        )
    )
    assert batches[0] == (23, 1, 3)
    assert batches[-1] == (29, 1, 9)


def test_training_batch_iterator_resumes_from_absolute_step_past_epoch_one():
    batches = list(
        iter_training_batches_from_step(
            n_batches=3,
            total_epochs=3,
            start_step=7,
            max_steps=9,
        )
    )
    assert batches == [(7, 2, 1), (8, 2, 2)]


def test_default_checkpoint_directories_are_isolated_but_explicit_is_exact(tmp_path):
    source = _length_dataset([2])
    first = TinkerSFTBackend(SFTSpec(train_dataset=source, experiment="same"))
    second = TinkerSFTBackend(SFTSpec(train_dataset=source, experiment="same"))

    first_path = first.checkpoint_dir
    assert first.checkpoint_dir == first_path
    assert second.checkpoint_dir != first_path
    assert first_path.startswith("/tmp/rllm-tinker-sft-checkpoints/same/")

    explicit = TinkerSFTBackend(SFTSpec(train_dataset=source, output_dir=str(tmp_path / "resume")))
    assert explicit.checkpoint_dir == str(tmp_path / "resume")


def test_resume_contract_rejects_legacy_and_changed_identity(tmp_path):
    contract = SFTResumeContract(
        payload={"contract_version": 1, "dataset": {"fingerprint": "first"}},
        digest="contract-one",
    )
    prepare_tinker_resume_contract(str(tmp_path), contract, None)
    matching = SimpleNamespace(
        get=lambda key, default=None: {
            "contract_version": 1,
            "contract_hash": "contract-one",
        }.get(key, default)
    )
    prepare_tinker_resume_contract(str(tmp_path), contract, matching)

    changed = SFTResumeContract(
        payload={"contract_version": 1, "dataset": {"fingerprint": "second"}},
        digest="contract-two",
    )
    with pytest.raises(SFTConfigError, match="dataset.fingerprint.*new output"):
        prepare_tinker_resume_contract(str(tmp_path), changed, matching)

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy = SimpleNamespace(get=lambda *_: None)
    with pytest.raises(SFTConfigError, match="legacy checkpoint.*sft-run.json.*missing"):
        prepare_tinker_resume_contract(str(legacy_dir), contract, legacy)


@pytest.mark.parametrize(
    ("cursor", "match"),
    [
        ({"epoch": 0, "batch": 2, "step": 1}, "inconsistent"),
        ({"epoch": 0, "batch": 3, "step": 3}, "inconsistent"),
        ({"epoch": 0, "batch": 1}, "loop_state.step.*integer"),
    ],
)
def test_tinker_resume_cursor_fails_closed(cursor, match):
    resume = SimpleNamespace(get=lambda key, default=None: cursor.get(key, default))

    with pytest.raises(SFTConfigError, match=match):
        validate_tinker_resume_cursor(resume, n_batches=3, total_steps=3)


@pytest.mark.parametrize(
    ("override", "difference"),
    [
        ({"model": {"tokenizer_model": "different/tokenizer"}}, "rendering.tokenizer_model"),
        ({"data": {"rllm": {"strip_thinking_from_history": True}}}, "rendering.strip_thinking_from_history"),
    ],
)
def test_resume_contract_rejects_tokenization_changes_before_tinker_client(
    monkeypatch,
    tmp_path,
    override,
    difference,
):
    from tinker_cookbook import checkpoint_utils

    import rllm.trainer.sft.tinker_backend as backend_module

    train = _FakeDataset([_one_datum()], batches=1)
    backend = TinkerSFTBackend(
        SFTSpec(
            train_dataset=_length_dataset([2]),
            output_dir=str(tmp_path),
            overrides={"trainer": {"max_steps": 1}},
        )
    )
    config = backend.build_config()
    config.data.resolved_renderer_name = "qwen3_5"
    optimizer = resolve_sft_optimizer_settings(config.optim, total_steps=1)
    original = build_tinker_resume_contract(
        config,
        train,
        optimizer,
        n_batches=1,
        total_epochs=1,
        total_steps=1,
    )
    prepare_tinker_resume_contract(str(tmp_path), original, None)

    config = backend_module.OmegaConf.merge(config, backend_module.OmegaConf.create(override))
    backend._config = config
    changed = build_tinker_resume_contract(
        config,
        train,
        optimizer,
        n_batches=1,
        total_epochs=1,
        total_steps=1,
    )
    assert changed.digest != original.digest

    checkpoint = SimpleNamespace(
        state_path="tinker://run/weights/step",
        get=lambda key, default=None: {
            "contract_version": 1,
            "contract_hash": original.digest,
            "epoch": 0,
            "batch": 1,
        }.get(key, default),
    )
    monkeypatch.setattr(
        backend_module,
        "build_sft_data",
        lambda *_: (object(), train, None),
    )
    monkeypatch.setattr(
        checkpoint_utils,
        "get_last_checkpoint",
        lambda *_: checkpoint,
    )
    monkeypatch.setattr(
        tinker,
        "ServiceClient",
        lambda **_: pytest.fail("provider client must not be created"),
    )

    with pytest.raises(SFTConfigError, match=rf"{difference}.*new output"):
        asyncio.run(backend._fit_async())


def test_provider_checkpoint_identity_is_a_hard_error_before_resume_client():
    source = _length_dataset([2])
    backend = TinkerSFTBackend(SFTSpec(train_dataset=source))
    config = backend.build_config()
    config.data.resolved_renderer_name = "qwen3_5"

    class _RestClient:
        async def get_weights_info_by_tinker_path(self, path):
            assert path == "tinker://run/weights/step"
            return SimpleNamespace(
                base_model="different/model",
                lora_rank=8,
                train_unembed=True,
                train_attn=True,
                train_mlp=True,
            )

        async def get_training_run_by_tinker_path_async(self, path):
            assert path == "tinker://run/weights/step"
            return SimpleNamespace(user_metadata={})

    service = SimpleNamespace(create_rest_client=lambda: _RestClient())
    with pytest.raises(
        SFTConfigError,
        match="base_model=.*lora_rank=.*renderer=.*new output directory",
    ):
        asyncio.run(
            validate_tinker_checkpoint_identity(
                service,
                "tinker://run/weights/step",
                config,
            )
        )


def test_fit_rejects_provider_identity_before_resumed_training_client(
    monkeypatch,
    tmp_path,
):
    from tinker_cookbook import checkpoint_utils

    import rllm.trainer.sft.tinker_backend as backend_module
    import rllm.utils.tracking as tracking_module

    contract = SFTResumeContract(
        payload={"contract_version": 1, "dataset": {"fingerprint": "same"}},
        digest="same-contract",
    )
    prepare_tinker_resume_contract(str(tmp_path), contract, None)

    class _Checkpoint:
        state_path = "tinker://run/weights/step"

        def get(self, key, default=None):
            return {
                "contract_version": 1,
                "contract_hash": "same-contract",
                "epoch": 0,
                "batch": 1,
                "step": 1,
            }.get(key, default)

    class _RestClient:
        async def get_weights_info_by_tinker_path(self, path):
            return SimpleNamespace(
                base_model="wrong/model",
                lora_rank=32,
                train_unembed=True,
                train_attn=True,
                train_mlp=True,
            )

        async def get_training_run_by_tinker_path_async(self, path):
            return SimpleNamespace(user_metadata={checkpoint_utils.RENDERER_NAME_METADATA_KEY: "qwen3_5"})

    class _ServiceClient:
        def __init__(self, base_url=None):
            del base_url

        def create_rest_client(self):
            return _RestClient()

        async def create_training_client_from_state_with_optimizer_async(
            self,
            *args,
            **kwargs,
        ):
            pytest.fail("resumed training client must not be created")

    datum = _one_datum()
    train = _FakeDataset([datum], batches=2)
    monkeypatch.setattr(
        backend_module,
        "build_sft_data",
        lambda *_: (object(), train, None),
    )
    monkeypatch.setattr(
        backend_module,
        "build_tinker_resume_contract",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        checkpoint_utils,
        "get_last_checkpoint",
        lambda *_: _Checkpoint(),
    )
    monkeypatch.setattr(tinker, "ServiceClient", _ServiceClient)
    monkeypatch.setattr(tracking_module, "Tracking", _FakeTracking)

    backend = TinkerSFTBackend(
        SFTSpec(
            train_dataset=_length_dataset([2]),
            output_dir=str(tmp_path),
            overrides={"trainer": {"max_steps": 2}},
        )
    )
    config = backend.build_config()
    config.data.resolved_renderer_name = "qwen3_5"

    with pytest.raises(SFTConfigError, match="base_model=.*wrong/model"):
        asyncio.run(backend._fit_async())


def test_fit_matching_resume_starts_at_next_unseen_batch(monkeypatch, tmp_path):
    from tinker_cookbook import checkpoint_utils, display

    import rllm.trainer.sft.tinker_backend as backend_module
    import rllm.utils.tracking as tracking_module

    contract = SFTResumeContract(
        payload={"contract_version": 1, "dataset": {"fingerprint": "same"}},
        digest="same-contract",
    )
    prepare_tinker_resume_contract(str(tmp_path), contract, None)

    class _Checkpoint:
        state_path = "tinker://run/weights/step"

        def get(self, key, default=None):
            return {
                "contract_version": 1,
                "contract_hash": "same-contract",
                "epoch": 0,
                "batch": 2,
                "step": 2,
            }.get(key, default)

    datum = _one_datum()
    train = _FakeDataset([datum], batches=3)
    client = _FakeTrainingClient([datum])
    saves = []

    class _RestClient:
        async def get_weights_info_by_tinker_path(self, path):
            return SimpleNamespace(
                base_model="Qwen/Qwen3.5-4B",
                lora_rank=32,
                train_unembed=True,
                train_attn=True,
                train_mlp=True,
            )

        async def get_training_run_by_tinker_path_async(self, path):
            return SimpleNamespace(user_metadata={checkpoint_utils.RENDERER_NAME_METADATA_KEY: "qwen3_5"})

    class _ServiceClient:
        def __init__(self, base_url=None):
            del base_url

        def create_rest_client(self):
            return _RestClient()

        async def create_training_client_from_state_with_optimizer_async(
            self,
            path,
            **kwargs,
        ):
            assert path == "tinker://run/weights/step"
            return client

    async def save(**kwargs):
        saves.append(kwargs)
        return {}

    monkeypatch.setattr(
        backend_module,
        "build_sft_data",
        lambda *_: (object(), train, None),
    )
    monkeypatch.setattr(
        backend_module,
        "build_tinker_resume_contract",
        lambda *_args, **_kwargs: contract,
    )
    monkeypatch.setattr(
        checkpoint_utils,
        "get_last_checkpoint",
        lambda *_: _Checkpoint(),
    )
    monkeypatch.setattr(checkpoint_utils, "save_checkpoint_async", save)
    monkeypatch.setattr(display, "colorize_example", lambda *_: "example")
    monkeypatch.setattr(tinker, "ServiceClient", _ServiceClient)
    monkeypatch.setattr(tracking_module, "Tracking", _FakeTracking)
    _FakeTracking.instances.clear()

    backend = TinkerSFTBackend(
        SFTSpec(
            train_dataset=_length_dataset([2]),
            output_dir=str(tmp_path),
            overrides={"trainer": {"max_steps": 3}},
        )
    )
    config = backend.build_config()
    config.data.resolved_renderer_name = "qwen3_5"
    asyncio.run(backend._fit_async())

    assert train.batch_calls == [0, 1, 2, 0, 1, 2, 2]
    assert train.epoch_seeds == [0]
    assert client.forward_backward_calls == 1
    assert [save["name"] for save in saves] == ["final"]


class _FakeDataset:
    def __init__(self, datums, batches):
        self.datums = datums
        self.batches = batches
        self.batch_size = 1
        self.dataset = [object()] * batches
        self.batch_calls = []
        self.epoch_seeds = []

    def __len__(self):
        return self.batches

    def get_batch(self, index):
        self.batch_calls.append(index)
        return self.datums

    def set_epoch(self, seed):
        self.epoch_seeds.append(seed)

    def preflight(self, label="train", planned_batches=None):
        del label
        for index in range(len(self)):
            self.get_batch(index)
        if planned_batches is not None:
            for _epoch, index in planned_batches:
                self.get_batch(index)

    def content_fingerprint(self):
        return "fake-dataset"


class _FakeFuture:
    def __init__(self, value, events=None, finish_event=None):
        self.value = value
        self.events = events
        self.finish_event = finish_event

    async def result_async(self):
        if self.events is not None and self.finish_event is not None:
            self.events.append(self.finish_event)
        return self.value


class _FakeTrainingClient:
    def __init__(self, datums):
        self.datums = datums
        self.adam = []
        self.forward_calls = 0
        self.forward_backward_calls = 0
        self.events = []

    def _forward_output(self):
        outputs = []
        for datum in self.datums:
            weights = datum.loss_fn_inputs["weights"]
            outputs.append(
                {
                    "logprobs": tinker.TensorData(
                        data=[-1.0] * len(weights.data),
                        dtype=weights.dtype,
                        shape=list(weights.shape),
                    )
                }
            )
        return SimpleNamespace(loss_fn_outputs=outputs)

    async def forward_async(self, data, loss_fn):
        assert data == self.datums
        assert loss_fn == "cross_entropy"
        self.forward_calls += 1
        return _FakeFuture(self._forward_output())

    async def forward_backward_async(self, data, loss_fn):
        assert data == self.datums
        assert loss_fn == "cross_entropy"
        batch = self.forward_backward_calls
        self.forward_backward_calls += 1
        self.events.append(f"submit-{batch}")
        return _FakeFuture(
            self._forward_output(),
            self.events,
            f"finish-{batch}",
        )

    async def optim_step_async(self, adam):
        self.adam.append(adam)
        return _FakeFuture(SimpleNamespace(metrics={"optim/grad_norm": 0.5}))


class _FakeTracking:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logs = []
        self.finished = False
        self.instances.append(self)

    def log(self, data, step):
        self.logs.append((step, dict(data)))

    def finish(self):
        self.finished = True


def _one_datum():
    import torch
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    return datum_from_model_input_weights(
        tinker.ModelInput.from_ints([1, 2, 3, 4]),
        torch.tensor([0.0, 1.0, 1.0, 0.0]),
        max_length=None,
        reduction="none",
    )


@pytest.mark.parametrize("bad_split", ["train", "validation"])
def test_fit_preflight_fails_before_tinker_service_client(
    monkeypatch,
    tmp_path,
    bad_split,
):
    import rllm.trainer.sft.tinker_backend as backend_module

    datum = _one_datum()
    train = _FakeDataset([datum], batches=2)
    val = _FakeDataset([datum], batches=1)

    def fail_preflight(label, planned_batches=None):
        del planned_batches
        raise SFTConfigError(f"{label} row is invalid")

    (train if bad_split == "train" else val).preflight = fail_preflight
    monkeypatch.setattr(
        backend_module,
        "build_sft_data",
        lambda *_: (object(), train, val),
    )
    monkeypatch.setattr(
        tinker,
        "ServiceClient",
        lambda **_: pytest.fail("provider client must not be created"),
    )
    source = _length_dataset([2])
    backend = TinkerSFTBackend(
        SFTSpec(
            train_dataset=source,
            val_dataset=source,
            output_dir=str(tmp_path),
        )
    )
    backend.build_config()

    with pytest.raises(SFTConfigError, match=f"{bad_split} row is invalid"):
        asyncio.run(backend._fit_async())


def test_fit_loop_uses_completed_step_cadence_and_saves_resume_cursor(
    monkeypatch,
    tmp_path,
):
    """Exercise the real async loop boundary without opening a provider job."""
    from tinker_cookbook import checkpoint_utils, display

    import rllm.trainer.sft.tinker_backend as backend_module
    import rllm.utils.tracking as tracking_module

    datum = _one_datum()
    train = _FakeDataset([datum], batches=3)
    val = _FakeDataset([datum], batches=1)
    client = _FakeTrainingClient([datum])
    saves = []

    class _ServiceClient:
        def __init__(self, base_url=None):
            self.base_url = base_url

        async def create_lora_training_client_async(self, **kwargs):
            return client

    async def _save(**kwargs):
        saves.append(kwargs)
        return {}

    monkeypatch.setattr(backend_module, "build_sft_data", lambda *_: (object(), train, val))
    monkeypatch.setattr(tinker, "ServiceClient", _ServiceClient)
    monkeypatch.setattr(checkpoint_utils, "get_last_checkpoint", lambda *_: None)
    monkeypatch.setattr(checkpoint_utils, "save_checkpoint_async", _save)
    monkeypatch.setattr(display, "colorize_example", lambda *_: "example")
    monkeypatch.setattr(tracking_module, "Tracking", _FakeTracking)
    _FakeTracking.instances.clear()

    spec = SFTSpec(
        train_dataset=_length_dataset([2]),
        val_dataset=_length_dataset([2]),
        output_dir=str(tmp_path),
        overrides={
            "trainer": {
                "total_epochs": 2,
                "max_steps": 3,
                "save_freq": 2,
                "test_freq": 2,
            },
            "optim": {
                "lr": 1e-4,
                "min_lr": 1e-5,
                "lr_scheduler": "cosine",
                "warmup_steps": 1,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 1e-2,
                "grad_clip_norm": 1.0,
            },
        },
    )
    backend = TinkerSFTBackend(spec)
    backend.build_config()
    asyncio.run(backend._fit_async())

    assert train.batch_calls == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert train.epoch_seeds == [0]
    assert client.forward_backward_calls == 3
    assert client.events.index("submit-1") < client.events.index("finish-0")
    assert client.events.index("finish-1") < client.events.index("submit-2")
    assert client.forward_calls == 2  # validation at step 0 and completed step 2
    assert [save["name"] for save in saves] == ["000002", "final"]
    assert saves[0]["loop_state"] == {
        "epoch": 0,
        "batch": 2,
        "step": 2,
        "contract_version": 1,
        "contract_hash": saves[0]["loop_state"]["contract_hash"],
    }
    assert saves[1]["loop_state"] == {
        "epoch": 1,
        "batch": 0,
        "step": 3,
        "final": True,
        "contract_version": 1,
        "contract_hash": saves[0]["loop_state"]["contract_hash"],
    }
    assert len(client.adam) == 3
    assert all(adam.weight_decay == pytest.approx(1e-2) for adam in client.adam)
    assert all(adam.beta1 == pytest.approx(0.9) for adam in client.adam)
    assert all(adam.beta2 == pytest.approx(0.999) for adam in client.adam)
    assert all(adam.grad_clip_norm == pytest.approx(1.0) for adam in client.adam)
    tracking = _FakeTracking.instances[-1]
    assert [step for step, _ in tracking.logs] == [0, 1, 2, 3, 3]
    assert tracking.logs[-1][1] == {"status": "completed"}
    assert tracking.finished is True
