"""Exact hosted-SFT checkpoint and resume contracts, exercised on fakes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

tinker = pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")

from rllm.data import Dataset  # noqa: E402
from rllm.trainer.sft import SFTSpec  # noqa: E402
from rllm.trainer.sft.backend import SFTConfigError  # noqa: E402
from rllm.trainer.sft.fireworks_backend import (  # noqa: E402
    FireworksSFTBackend,
    build_fireworks_resume_contract,
    prepare_fireworks_resume_contract,
    validate_fireworks_resume_contract,
)
from rllm.trainer.sft.tinker_backend import (  # noqa: E402
    SFTResumeContract,
    TinkerSFTBackend,
    build_sft_data,
    build_tinker_resume_contract,
    iter_training_batches_from_step,
    prepare_tinker_resume_contract,
    resolve_sft_optimizer_settings,
    validate_tinker_checkpoint_identity,
    validate_tinker_resume_cursor,
)
from rllm.trainer.sft.tinker_dataset import TinkerSFTDataset  # noqa: E402


class _TokenRenderer:
    def render(self, messages, *, tools=None, add_generation_prompt=False):
        from rllm.renderers.types import RenderedTokens

        del tools, add_generation_prompt
        token = int(messages[-1]["content"])
        return RenderedTokens(
            token_ids=[0, token, 0],
            message_indices=[-1, 1, -1],
        )


def _source(size: int = 7) -> Dataset:
    return Dataset(
        data=[
            {
                "messages": [
                    {"role": "user", "content": "q", "trainable": False},
                    {"role": "assistant", "content": str(index), "trainable": True},
                ]
            }
            for index in range(size)
        ],
        name="resume-order",
        split="train",
    )


def _ordered_batches(dataset, *, total_epochs: int, start_step: int = 0):
    batches = []
    current_epoch = None
    for _, epoch, batch in iter_training_batches_from_step(
        n_batches=len(dataset),
        total_epochs=total_epochs,
        start_step=start_step,
    ):
        if epoch != current_epoch:
            dataset.set_epoch(seed=epoch)
            current_epoch = epoch
        datums = dataset.get_batch(batch)
        batches.append(tuple(datum.model_input.to_ints()[1] for datum in datums))
    return batches


def test_resumed_epoch_order_matches_uninterrupted_next_batches():
    source = _source()
    uninterrupted = TinkerSFTDataset(source, _TokenRenderer(), batch_size=3)
    full_order = _ordered_batches(uninterrupted, total_epochs=3)

    resumed = TinkerSFTDataset(source, _TokenRenderer(), batch_size=3)
    assert _ordered_batches(resumed, total_epochs=3, start_step=4) == full_order[4:]


def test_raw_row_cursor_round_trips_partial_batches_and_epochs():
    dataset = TinkerSFTDataset(_source(5), _TokenRenderer(), batch_size=2)
    cursors = [0, 2, 4, 5, 7, 9, 10]
    assert [dataset.data_cursor_for_step(step) for step in range(7)] == cursors
    assert [dataset.step_for_data_cursor(cursor) for cursor in cursors] == list(range(7))

    for cursor in (-1, 1, 3, 6):
        with pytest.raises(SFTConfigError, match="cursor.*non-negative|cursor.*exact"):
            dataset.step_for_data_cursor(cursor)


@pytest.mark.parametrize(
    "cursor",
    [
        {"epoch": 0, "batch": 2, "step": 1},
        {"epoch": 0, "batch": 3, "step": 3},
        {"epoch": 0, "batch": 1},
    ],
)
def test_tinker_cursor_rejects_off_by_one_or_incomplete_state(cursor):
    with pytest.raises(SFTConfigError, match="inconsistent|loop_state.step"):
        validate_tinker_resume_cursor(cursor, n_batches=3, total_steps=3)


@pytest.mark.parametrize(
    ("backend_cls", "default_root"),
    [
        (TinkerSFTBackend, "/tmp/rllm-tinker-sft-checkpoints"),
        (FireworksSFTBackend, "/tmp/rllm-fireworks-sft-checkpoints"),
    ],
)
def test_default_paths_are_isolated_and_explicit_path_is_resume_identity(tmp_path, backend_cls, default_root):
    first = backend_cls(SFTSpec(train_dataset=_source(), experiment="same"))
    second = backend_cls(SFTSpec(train_dataset=_source(), experiment="same"))
    assert first.checkpoint_dir != second.checkpoint_dir
    assert first.checkpoint_dir.startswith(f"{default_root}/same/")

    explicit_path = str(tmp_path / "resume")
    explicit = backend_cls(SFTSpec(train_dataset=_source(), output_dir=explicit_path))
    assert explicit.checkpoint_dir == explicit_path


def test_fireworks_reattach_requires_explicit_cursor_directory():
    backend = FireworksSFTBackend(
        SFTSpec(
            train_dataset=_source(),
            overrides={"fireworks_infra": {"trainers": {"policy": {"job_id": "job-1"}}}},
        )
    )
    with pytest.raises(SFTConfigError, match=r"job_id requires.*explicit --output"):
        backend.build_config()


@pytest.mark.parametrize(
    ("case", "job_id"),
    [("fresh-run-failure", None), ("reattached-run", "job-1")],
)
def test_fireworks_provision_never_requests_trainer_deletion(monkeypatch, tmp_path, case, job_id):
    provision_module = pytest.importorskip("training.provision")

    overrides = {}
    if job_id is not None:
        overrides = {"fireworks_infra": {"trainers": {"policy": {"job_id": job_id}}}}
    backend = FireworksSFTBackend(
        SFTSpec(
            train_dataset=_source(),
            output_dir=str(tmp_path / case),
            overrides=overrides,
        )
    )
    config = backend.build_config()
    provision_config = object()
    returned_infra = object()
    call = {}

    monkeypatch.setattr(
        provision_module,
        "load_yaml_provision",
        lambda **_kwargs: ("sft", provision_config),
    )

    def init_fireworks_infra(mode, config_arg, **kwargs):
        call.update(mode=mode, config=config_arg, **kwargs)
        return returned_infra

    monkeypatch.setattr(provision_module, "init_fireworks_infra", init_fireworks_infra)

    assert backend._provision(config, "fake-key", "https://example.invalid") is returned_infra
    assert call["mode"] == "sft"
    assert call["config"] is provision_config
    assert call["cleanup_on_close"] is False
    assert call["cleanup_existing"] is False


def test_tinker_manifest_rejects_changed_identity_and_legacy_checkpoint(tmp_path):
    contract = SFTResumeContract(
        payload={"contract_version": 1, "dataset": {"fingerprint": "first"}},
        digest="contract-one",
    )
    prepare_tinker_resume_contract(str(tmp_path), contract, None)
    matching = {
        "contract_hash": "contract-one",
    }
    prepare_tinker_resume_contract(str(tmp_path), contract, matching)

    changed = SFTResumeContract(
        payload={"contract_version": 1, "dataset": {"fingerprint": "second"}},
        digest="contract-two",
    )
    with pytest.raises(SFTConfigError, match="dataset.fingerprint.*new output"):
        prepare_tinker_resume_contract(str(tmp_path), changed, matching)

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    with pytest.raises(SFTConfigError, match="legacy checkpoint.*sft-run.json.*missing"):
        prepare_tinker_resume_contract(str(legacy_dir), contract, {})


def test_fireworks_manifest_covers_identity_and_binds_provider_job(tmp_path):
    dataset = TinkerSFTDataset(_source(5), _TokenRenderer(), batch_size=2)
    backend = FireworksSFTBackend(SFTSpec(train_dataset=_source(), output_dir=str(tmp_path)))
    config = backend.build_config()
    config.data.resolved_renderer_name = "qwen3.5"
    config.data.resolved_renderer_source = "prime"
    config.data.resolved_renderer_identity = {
        "adapter": {"class": "renderers.Qwen35Renderer", "distributions": {"renderers": "1.2.3"}},
        "implementation": {"class": "renderers.Qwen35Renderer", "distributions": {"renderers": "1.2.3"}},
    }
    config.data.resolved_tokenizer_identity = {
        "class": "transformers.QwenTokenizerFast",
        "name_or_path": "Qwen/Qwen3.5-9B",
        "revision": "abc123",
    }
    optimizer = resolve_sft_optimizer_settings(config.optim, total_steps=3)
    contract = build_fireworks_resume_contract(
        config,
        dataset,
        optimizer,
        n_batches=3,
        total_steps=3,
    )

    assert contract.payload["backend"] == {
        "name": "fireworks",
        "provider": {
            "training_shape_id": "accounts/fireworks/trainingShapes/qwen3p5-9b-256k-lora",
        },
    }
    dataset_identity = contract.payload["dataset"]
    assert dataset_identity["fingerprint"] == dataset.content_fingerprint()
    assert dataset_identity["row_count"] == 5
    assert dataset_identity["batch_size"] == 2
    implementation = dataset_identity["implementation"]
    assert implementation["class"] == "rllm.data.dataset.Dataset"
    assert contract.payload["rendering"]["renderer_source"] == "prime"
    assert contract.payload["rendering"]["renderer_name"] == "qwen3.5"
    assert contract.payload["rendering"]["loss_reduction"] == "token_mean"
    assert contract.payload["rendering"]["tokenizer_model"] == "Qwen/Qwen3.5-9B"
    assert contract.payload["model"]["base_model"] == "accounts/fireworks/models/qwen3p5-9b"
    assert contract.payload["model"]["lora_rank"] == 32
    assert contract.payload["optimizer"]["learning_rate"] == pytest.approx(1e-5)
    assert contract.payload["horizon"]["total_steps"] == 3

    prepared = prepare_fireworks_resume_contract(
        str(tmp_path),
        contract,
        configured_job_id=None,
    )
    bound = validate_fireworks_resume_contract(
        prepared,
        configured_job_id=None,
        actual_job_id="job-1",
    )
    assert bound.data["provider_job_id"] == "job-1"
    resumed = prepare_fireworks_resume_contract(
        str(tmp_path),
        contract,
        configured_job_id="job-1",
    )
    validate_fireworks_resume_contract(
        resumed,
        configured_job_id="job-1",
        actual_job_id="job-1",
        resume_info=SimpleNamespace(data_consumed=2),
    )

    config.data.resolved_renderer_source = "tinker"
    changed = build_fireworks_resume_contract(
        config,
        dataset,
        optimizer,
        n_batches=3,
        total_steps=3,
    )
    with pytest.raises(SFTConfigError, match="rendering.renderer_source.*new output"):
        prepare_fireworks_resume_contract(
            str(tmp_path),
            changed,
            configured_job_id="job-1",
        )


def test_tinker_resume_identity_covers_data_model_rendering_optimizer_and_horizon():
    dataset = TinkerSFTDataset(_source(5), _TokenRenderer(), batch_size=2)
    backend = TinkerSFTBackend(SFTSpec(train_dataset=_source()))
    config = backend.build_config()
    config.data.resolved_renderer_name = "qwen3_5"
    config.data.resolved_renderer_source = "prime"
    config.data.resolved_renderer_identity = {
        "adapter": {"class": "renderers.Qwen35Renderer", "distributions": {"renderers": "1.2.3"}},
        "implementation": {"class": "renderers.Qwen35Renderer", "distributions": {"renderers": "1.2.3"}},
    }
    config.data.resolved_tokenizer_identity = {
        "class": "transformers.QwenTokenizerFast",
        "distributions": {"transformers": "4.0.0"},
        "name_or_path": "Qwen/Qwen3.5-4B",
        "revision": "abc123",
        "runtime_versions": {"tokenizers": "0.1.0", "transformers": "4.0.0"},
    }
    optimizer = resolve_sft_optimizer_settings(config.optim, total_steps=3)
    contract = build_tinker_resume_contract(
        config,
        dataset,
        optimizer,
        n_batches=3,
        total_steps=3,
    )

    assert set(contract.payload) == {
        "contract_version",
        "loop_semantics",
        "backend",
        "model",
        "rendering",
        "dataset",
        "optimizer",
        "horizon",
    }
    assert contract.payload["dataset"]["batch_size"] == 2
    assert contract.payload["backend"]["name"] == "tinker"
    assert contract.payload["model"]["base_model"] == "Qwen/Qwen3.5-4B"
    assert contract.payload["rendering"]["renderer_name"] == "qwen3_5"
    assert contract.payload["rendering"]["renderer_source"] == "prime"
    assert contract.payload["rendering"]["renderer_identity"]["implementation"]["distributions"] == {"renderers": "1.2.3"}
    assert contract.payload["rendering"]["tokenizer_identity"]["revision"] == "abc123"
    assert contract.payload["rendering"]["tokenize_and_mask_method"] == "cumulative"
    assert contract.payload["optimizer"]["learning_rate"] == pytest.approx(1e-5)
    assert contract.payload["horizon"] == {
        "batches_per_epoch": 3,
        "total_steps": 3,
    }


def test_build_sft_data_records_resolved_renderer_and_tokenizer_identity(monkeypatch):
    import tinker_cookbook.tokenizer_utils as tokenizer_module

    import rllm.renderers as renderer_module
    import rllm.trainer.sft.tinker_dataset as dataset_module

    class _Tokenizer:
        name_or_path = "example/tokenizer"
        init_kwargs = {"_commit_hash": "revision-1"}

    class _Renderer:
        pass

    tokenizer = _Tokenizer()
    renderer = _Renderer()
    resolution = SimpleNamespace(renderer=renderer, source="prime", name="qwen3.5")
    monkeypatch.setattr(tokenizer_module, "get_tokenizer", lambda _name: tokenizer)
    monkeypatch.setattr(renderer_module, "resolve", lambda *_args, **_kwargs: resolution)
    monkeypatch.setattr(dataset_module, "create_tinker_sft_datasets", lambda **_kwargs: ("train", "val"))

    backend = TinkerSFTBackend(SFTSpec(train_dataset=_source()))
    config = backend.build_config()
    returned_tokenizer, train, val = build_sft_data(config, _source(), None)

    assert returned_tokenizer is tokenizer
    assert (train, val) == ("train", "val")
    assert config.data.resolved_renderer_name == "qwen3.5"
    assert config.data.resolved_renderer_source == "prime"
    assert config.data.resolved_renderer_identity.implementation["class"].endswith("._Renderer")
    assert config.data.resolved_tokenizer_identity.name_or_path == "example/tokenizer"
    assert config.data.resolved_tokenizer_identity.revision == "revision-1"


def test_tinker_provider_identity_mismatch_fails_closed():
    from tinker_cookbook import checkpoint_utils

    backend = TinkerSFTBackend(SFTSpec(train_dataset=_source()))
    config = backend.build_config()
    config.data.resolved_renderer_name = "qwen3_5"

    class _Rest:
        async def get_weights_info_by_tinker_path(self, path):
            del path
            return SimpleNamespace(
                base_model="wrong/model",
                lora_rank=8,
                train_unembed=True,
                train_attn=True,
                train_mlp=True,
            )

        async def get_training_run_by_tinker_path_async(self, path):
            del path
            return SimpleNamespace(user_metadata={checkpoint_utils.RENDERER_NAME_METADATA_KEY: "wrong_renderer"})

    service = SimpleNamespace(create_rest_client=lambda: _Rest())
    with pytest.raises(SFTConfigError, match="base_model=.*lora_rank=.*renderer=.*new output"):
        asyncio.run(
            validate_tinker_checkpoint_identity(
                service,
                "tinker://run/weights/step",
                config,
            )
        )


def _one_datum():
    import torch
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    return datum_from_model_input_weights(
        tinker.ModelInput.from_ints([1, 2, 3]),
        torch.tensor([0.0, 1.0, 0.0]),
        max_length=None,
        reduction="none",
    )


class _HostedDataset:
    def __init__(self, events, batches=3, *, batch_size=1, row_count=None):
        self.events = events
        self.batches = batches
        self.batch_size = batch_size
        self.dataset = [object()] * (row_count if row_count is not None else batches * batch_size)
        self.datum = _one_datum()
        self.training_batches = []
        self.preflighting = False

    def __len__(self):
        return self.batches

    def get_batch(self, index):
        if not self.preflighting:
            self.training_batches.append(index)
            self.events.append(f"batch-{index}")
        return [self.datum]

    def set_epoch(self, seed):
        self.events.append(f"epoch-{seed}")

    def preflight(self, label="train", planned_batches=None):
        self.preflighting = True
        self.events.append(f"preflight-{label}")
        try:
            if planned_batches is not None:
                for _epoch, batch in planned_batches:
                    self.get_batch(batch)
            else:
                for batch in range(self.batches):
                    self.get_batch(batch)
        finally:
            self.preflighting = False

    def content_fingerprint(self):
        return "hosted-dataset"

    def data_cursor_for_step(self, completed_steps):
        completed_epochs, batches_in_epoch = divmod(completed_steps, self.batches)
        rows_in_epoch = min(batches_in_epoch * self.batch_size, len(self.dataset))
        return completed_epochs * len(self.dataset) + rows_in_epoch

    def step_for_data_cursor(self, data_consumed):
        completed_epochs, rows_in_epoch = divmod(data_consumed, len(self.dataset))
        batches_in_epoch = (rows_in_epoch + self.batch_size - 1) // self.batch_size
        step = completed_epochs * self.batches + batches_in_epoch
        if self.data_cursor_for_step(step) != data_consumed:
            raise SFTConfigError("checkpoint cursor is not exact")
        return step


class _AsyncFuture:
    def __init__(self, value, events, event):
        self.value = value
        self.events = events
        self.event = event

    async def result_async(self):
        self.events.append(self.event)
        return self.value


class _TinkerTrainingClient:
    def __init__(self, events, datum):
        self.events = events
        self.datum = datum
        self.submitted = 0

    def _output(self):
        weights = self.datum.loss_fn_inputs["weights"]
        logprobs = tinker.TensorData(
            data=[-1.0] * len(weights.data),
            dtype=weights.dtype,
            shape=list(weights.shape),
        )
        return SimpleNamespace(loss_fn_outputs=[{"logprobs": logprobs}])

    async def forward_backward_async(self, data, loss_fn):
        del data, loss_fn
        step = self.submitted
        self.submitted += 1
        self.events.append(f"submit-{step}")
        return _AsyncFuture(self._output(), self.events, f"finish-fb-{step}")

    async def optim_step_async(self, adam):
        del adam
        step = self.submitted - 1
        return _AsyncFuture(SimpleNamespace(metrics={}), self.events, f"finish-opt-{step}")


class _Tracking:
    def __init__(self, **kwargs):
        del kwargs

    def log(self, data, step):
        del data, step

    def finish(self):
        pass


def test_tinker_checkpoint_drains_pipeline_and_records_next_unseen_cursor(monkeypatch, tmp_path):
    from tinker_cookbook import checkpoint_utils, display

    import rllm.trainer.sft.tinker_backend as module
    import rllm.utils.tracking as tracking_module

    events = []
    train = _HostedDataset(events)
    training_client = _TinkerTrainingClient(events, train.datum)
    saves = []

    class _Service:
        def __init__(self, **kwargs):
            del kwargs

        async def create_lora_training_client_async(self, **kwargs):
            del kwargs
            return training_client

    async def save_checkpoint(**kwargs):
        events.append(f"save-{kwargs['name']}")
        saves.append(kwargs)

    monkeypatch.setattr(module, "build_sft_data", lambda *_: (object(), train, None))
    monkeypatch.setattr(tinker, "ServiceClient", _Service)
    monkeypatch.setattr(checkpoint_utils, "get_last_checkpoint", lambda *_: None)
    monkeypatch.setattr(checkpoint_utils, "save_checkpoint_async", save_checkpoint)
    monkeypatch.setattr(display, "colorize_example", lambda *_: "example")
    monkeypatch.setattr(tracking_module, "Tracking", _Tracking)

    backend = TinkerSFTBackend(
        SFTSpec(
            train_dataset=_source(),
            output_dir=str(tmp_path),
            overrides={"trainer": {"max_steps": 3, "save_freq": 2, "test_freq": -1}},
        )
    )
    config = backend.build_config()
    config.data.resolved_renderer_name = "qwen3_5"
    asyncio.run(backend._fit_async())

    assert events.index("finish-opt-1") < events.index("save-000002") < events.index("submit-2")
    assert events.index("finish-opt-2") < events.index("save-final")
    assert saves[0]["loop_state"]["epoch"] == 0
    assert saves[0]["loop_state"]["batch"] == 2
    assert saves[0]["loop_state"]["step"] == 2
    assert saves[0]["loop_state"]["contract_hash"]
    assert saves[1]["loop_state"]["step"] == 3
    assert saves[1]["loop_state"]["final"] is True
    assert all(save["kind"] == "both" for save in saves)


def test_tinker_resume_restores_optimizer_and_starts_at_next_unseen_batch(monkeypatch, tmp_path):
    from tinker_cookbook import checkpoint_utils, display

    import rllm.trainer.sft.tinker_backend as module
    import rllm.utils.tracking as tracking_module

    events = []
    train = _HostedDataset(events)
    training_client = _TinkerTrainingClient(events, train.datum)
    contract = SFTResumeContract(
        payload={"contract_version": 1, "dataset": {"fingerprint": "same"}},
        digest="same-contract",
    )
    prepare_tinker_resume_contract(str(tmp_path), contract, None)

    class _Checkpoint:
        state_path = "tinker://run/weights/step"

        def get(self, key, default=None):
            return {
                "contract_hash": "same-contract",
                "epoch": 0,
                "batch": 2,
                "step": 2,
            }.get(key, default)

    class _Rest:
        async def get_weights_info_by_tinker_path(self, path):
            assert path == "tinker://run/weights/step"
            return SimpleNamespace(
                base_model="Qwen/Qwen3.5-4B",
                lora_rank=32,
                train_unembed=True,
                train_attn=True,
                train_mlp=True,
            )

        async def get_training_run_by_tinker_path_async(self, path):
            assert path == "tinker://run/weights/step"
            return SimpleNamespace(user_metadata={checkpoint_utils.RENDERER_NAME_METADATA_KEY: "qwen3_5"})

    class _Service:
        def __init__(self, **kwargs):
            del kwargs

        def create_rest_client(self):
            return _Rest()

        async def create_training_client_from_state_with_optimizer_async(self, path, **kwargs):
            del kwargs
            events.append("resume-with-optimizer")
            assert path == "tinker://run/weights/step"
            return training_client

        async def create_training_client_from_state_async(self, *args, **kwargs):
            del args, kwargs
            pytest.fail("weights-only resume must never be used")

    async def save_checkpoint(**kwargs):
        events.append(f"save-{kwargs['name']}")

    monkeypatch.setattr(module, "build_sft_data", lambda *_: (object(), train, None))
    monkeypatch.setattr(module, "build_tinker_resume_contract", lambda *_args, **_kwargs: contract)
    monkeypatch.setattr(tinker, "ServiceClient", _Service)
    monkeypatch.setattr(checkpoint_utils, "get_last_checkpoint", lambda *_: _Checkpoint())
    monkeypatch.setattr(checkpoint_utils, "save_checkpoint_async", save_checkpoint)
    monkeypatch.setattr(display, "colorize_example", lambda *_: "example")
    monkeypatch.setattr(tracking_module, "Tracking", _Tracking)

    backend = TinkerSFTBackend(
        SFTSpec(
            train_dataset=_source(),
            output_dir=str(tmp_path),
            overrides={"trainer": {"max_steps": 3, "save_freq": -1, "test_freq": -1}},
        )
    )
    config = backend.build_config()
    config.data.resolved_renderer_name = "qwen3_5"
    asyncio.run(backend._fit_async())

    assert "resume-with-optimizer" in events
    assert train.training_batches == [2]


class _SyncFuture:
    def __init__(self, value, events, event):
        self.value = value
        self.events = events
        self.event = event

    def result(self, timeout=None):
        del timeout
        self.events.append(self.event)
        return self.value


class _FireworksClient:
    def __init__(self, events):
        self.events = events
        self.submitted = 0

    def submit_forward_backward(self, data, loss_fn):
        del data, loss_fn
        step = self.submitted
        self.submitted += 1
        self.events.append(f"submit-{step}")
        result = SimpleNamespace(metrics={"response_tokens": 1, "loss:sum": 1.0})
        return _SyncFuture(result, self.events, f"finish-fb-{step}")

    def submit_optim_step(self, adam):
        del adam
        step = self.submitted - 1
        return _SyncFuture(None, self.events, f"finish-opt-{step}")


class _FireworksCheckpoints:
    def __init__(self, events, *, data_consumed=None, provider_step=0):
        self.events = events
        self.data_consumed = data_consumed
        self.provider_step = provider_step
        self.saves = []
        self.log_path = None

    def resume(self):
        manifest = json.loads((Path(self.log_path) / "sft-run.json").read_text())
        assert manifest["provider_job_id"] == "fake-job"
        if self.data_consumed is None:
            return None
        return SimpleNamespace(step=self.provider_step, data_consumed=self.data_consumed)

    def save(self, name, **kwargs):
        self.events.append(f"save-{name}")
        self.saves.append((name, kwargs))

    def promote_latest(self, output_model_id, base_model):
        del base_model
        return {"name": output_model_id}


def _run_fireworks(monkeypatch, tmp_path, *, data_consumed=None, provider_step=0):
    checkpoint_module = pytest.importorskip("training.utils.checkpoints")

    import rllm.trainer.sft.fireworks_backend as module
    import rllm.utils.tracking as tracking_module

    events = []
    # Five rows at batch size three yields cursors 0 -> 3 -> 5 -> 8 across
    # the partial final batch and the next epoch.
    train = _HostedDataset(events, batches=2, batch_size=3, row_count=5)
    client = _FireworksClient(events)
    checkpoints = _FireworksCheckpoints(
        events,
        data_consumed=data_consumed,
        provider_step=provider_step,
    )
    infra = SimpleNamespace(
        policy=client,
        service=object(),
        policy_job_id="fake-job",
        close=lambda: events.append("infra-close"),
    )

    overrides = {"trainer": {"max_steps": 3, "save_freq": 2, "test_freq": -1}}
    if data_consumed is not None:
        overrides["fireworks_infra"] = {"trainers": {"policy": {"job_id": "fake-job"}}}
    backend = FireworksSFTBackend(
        SFTSpec(
            train_dataset=_source(),
            output_dir=str(tmp_path),
            epochs=2,
            overrides=overrides,
        )
    )
    config = backend.build_config()
    config.data.resolved_renderer_name = "qwen3_5"
    config.data.resolved_renderer_source = "prime"
    config.data.resolved_renderer_identity = {"implementation": {"class": "renderers.Qwen35Renderer"}}
    config.data.resolved_tokenizer_identity = {"class": "transformers.QwenTokenizerFast"}
    if data_consumed is not None:
        optimizer = resolve_sft_optimizer_settings(config.optim, total_steps=3)
        contract = build_fireworks_resume_contract(
            config,
            train,
            optimizer,
            n_batches=2,
            total_steps=3,
        )
        prepared = prepare_fireworks_resume_contract(
            str(tmp_path),
            contract,
            configured_job_id=None,
        )
        validate_fireworks_resume_contract(
            prepared,
            configured_job_id=None,
            actual_job_id="fake-job",
            resume_info=None,
        )
    monkeypatch.setenv("FIREWORKS_API_KEY", "fake")
    monkeypatch.setattr(module, "build_sft_data", lambda *_: (None, train, None))
    monkeypatch.setattr(backend, "_provision", lambda *_: infra)
    monkeypatch.setattr(tracking_module, "Tracking", _Tracking)

    def checkpoint_factory(*_args, **kwargs):
        checkpoints.log_path = kwargs["log_path"]
        return checkpoints

    monkeypatch.setattr(checkpoint_module, "TrainingCheckpoints", checkpoint_factory)
    backend.fit()
    return events, train, checkpoints


def test_fireworks_checkpoint_drains_pipeline_and_persists_raw_cursor(monkeypatch, tmp_path):
    events, _train, checkpoints = _run_fireworks(monkeypatch, tmp_path)
    assert checkpoints.log_path == str(tmp_path)
    assert (tmp_path / "sft-run.json").exists()
    assert events.index("finish-opt-1") < events.index("save-step-2") < events.index("submit-2")
    assert events.index("finish-opt-2") < events.index("save-step-3")
    assert checkpoints.saves == [
        (
            "step-2",
            {"resumable": True, "promotable": False, "data_consumed": 5},
        ),
        (
            "step-3",
            {"resumable": True, "promotable": True, "data_consumed": 8},
        ),
    ]


def test_fireworks_resume_ignores_provider_renamed_step_and_uses_raw_cursor(monkeypatch, tmp_path):
    events, train, checkpoints = _run_fireworks(
        monkeypatch,
        tmp_path,
        data_consumed=5,
        provider_step=999,
    )
    assert train.training_batches == [0]
    assert "epoch-1" in events
    assert "batch-1" not in events
    assert checkpoints.saves[-1][1]["data_consumed"] == 8


def test_fireworks_resume_rejects_non_boundary_raw_cursor(monkeypatch, tmp_path):
    with pytest.raises(SFTConfigError, match="cursor is not exact"):
        _run_fireworks(
            monkeypatch,
            tmp_path,
            data_consumed=1,
            provider_step=1,
        )
