"""Hosted SFT data is rendered before a provider is started."""

from __future__ import annotations

import pytest

tinker = pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")

from rllm.data import Dataset  # noqa: E402
from rllm.trainer.sft import SFTSpec  # noqa: E402
from rllm.trainer.sft.backend import SFTConfigError  # noqa: E402
from rllm.trainer.sft.fireworks_backend import FireworksSFTBackend  # noqa: E402
from rllm.trainer.sft.tinker_backend import TinkerSFTBackend  # noqa: E402
from rllm.trainer.sft.tinker_dataset import TinkerSFTDataset  # noqa: E402


class _TokenRenderer:
    def __init__(self):
        self.seen: list[int] = []

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        from rllm.renderers.types import RenderedTokens

        del tools, add_generation_prompt
        count = int(messages[-1]["content"])
        self.seen.append(count)
        return RenderedTokens(
            token_ids=list(range(count + 2)),
            message_indices=[-1, *([1] * count), -1],
        )


def _dataset(lengths):
    return Dataset(
        data=[
            {
                "messages": [
                    {"role": "user", "content": "q", "trainable": False},
                    {"role": "assistant", "content": str(length), "trainable": True},
                ]
            }
            for length in lengths
        ],
        name="lengths",
        split="train",
    )


def test_training_batch_skips_prefix_stability_preflight():
    class _PrefixRenderer:
        def __init__(self):
            self.seen: list[int] = []

        def render(self, messages, *, tools=None, add_generation_prompt=False):
            from rllm.renderers.types import RenderedTokens

            del tools, add_generation_prompt
            self.seen.append(len(messages))
            return RenderedTokens(
                token_ids=list(range(len(messages))),
                message_indices=list(range(len(messages))),
            )

    source = Dataset(
        data=[
            {
                "messages": [
                    {"role": "user", "content": "q1", "trainable": False},
                    {"role": "assistant", "content": "a1", "trainable": True},
                    {"role": "user", "content": "q2", "trainable": False},
                    {"role": "assistant", "content": "a2", "trainable": True},
                ]
            }
        ],
        name="prefixes",
        split="train",
    )
    renderer = _PrefixRenderer()
    dataset = TinkerSFTDataset(source, renderer=renderer, batch_size=1)

    dataset.get_batch(0)
    assert renderer.seen == [4]

    renderer.seen.clear()
    dataset.preflight(label="train")
    assert renderer.seen == [4, 2]


def test_validation_preflight_checks_every_batch():
    dataset = TinkerSFTDataset(
        _dataset([2, 20]),
        renderer=_TokenRenderer(),
        batch_size=1,
        max_length=10,
        overlength_policy="error",
    )

    with pytest.raises(SFTConfigError, match="validation preflight.*batch 1.*max_length=10"):
        dataset.preflight(label="validation")


def test_tinker_explicit_preflight_does_not_create_service_client(monkeypatch, tmp_path):
    import rllm.trainer.sft.tinker_backend as backend_module

    train = TinkerSFTDataset(
        _dataset([0, 2, 0, 2]),
        renderer=_TokenRenderer(),
        batch_size=2,
    )
    monkeypatch.setattr(backend_module, "build_sft_data", lambda *_: (object(), train, None))
    monkeypatch.setattr(tinker, "ServiceClient", lambda **_: pytest.fail("provider client must not be created"))

    backend = TinkerSFTBackend(
        SFTSpec(
            train_dataset=_dataset([2]),
            output_dir=str(tmp_path),
            batch_size=2,
            overrides={"trainer": {"max_steps": 2}},
        )
    )
    backend.build_config()

    with pytest.raises(SFTConfigError, match="train preflight.*batch 0"):
        backend.preflight()


def test_fireworks_explicit_preflight_does_not_provision(monkeypatch, tmp_path):
    import rllm.trainer.sft.tinker_backend as backend_module

    train = TinkerSFTDataset(
        _dataset([0, 2, 0, 2]),
        renderer=_TokenRenderer(),
        batch_size=2,
    )
    backend = FireworksSFTBackend(
        SFTSpec(
            train_dataset=_dataset([2]),
            output_dir=str(tmp_path),
            batch_size=2,
            overrides={"trainer": {"max_steps": 2}},
        )
    )
    backend.build_config()

    monkeypatch.setattr(backend_module, "build_sft_data", lambda *_: (None, train, None))
    monkeypatch.setattr(backend, "_provision", lambda *_: pytest.fail("provider trainer must not be provisioned"))

    with pytest.raises(SFTConfigError, match="train preflight.*batch 0"):
        backend.preflight()
