"""Red tests for the tinker SFT render path (PR #739 follow-up).

Regression tests for the canonical SFT rendering path. They exercise the real
production renderer and the conversion from canonical messages to provider
Datums:

  F1  structured reasoning must reach the native model renderer.
  F2  canonical dict-shaped tool calls must render without provider objects.
  F3  a pandas->polars parquet round-trip unifies the per-row struct schema and
      stamps ``None`` into keys only *some* messages carry: ``tool_calls`` (a)
      and ``trainable`` (b) -> ``TypeError`` when the renderer iterates / ints
      the ``None``.

These use the real ``rllm.renderers.resolve`` path and locally cached Qwen
tokenizers (skipped if unavailable / offline).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest  # noqa: E402

pytest.importorskip("tinker_cookbook")
pytest.importorskip("tinker")

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

from rllm.data import Dataset  # noqa: E402
from rllm.renderers import resolve  # noqa: E402
from rllm.trainer.sft import SFTSpec  # noqa: E402
from rllm.trainer.sft.backend import SFTConfigError  # noqa: E402
from rllm.trainer.sft.tinker_backend import TinkerSFTBackend, build_sft_data  # noqa: E402
from rllm.trainer.sft.tinker_dataset import conversation_to_datum  # noqa: E402

QWEN = "Qwen/Qwen3-0.6B"
QWEN35 = "Qwen/Qwen3.5-35B-A3B"


def _load_cached_tokenizer(model: str):
    transformers = pytest.importorskip("transformers")
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(model, local_files_only=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.name_or_path = model
    return tokenizer


@pytest.fixture(scope="module")
def qwen_tokenizer():
    try:
        return _load_cached_tokenizer(QWEN)
    except Exception as e:  # not cached / incompatible local tokenizer
        pytest.skip(f"Qwen3 tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def qwen35_tokenizer():
    try:
        return _load_cached_tokenizer(QWEN35)
    except Exception as e:  # not cached / incompatible local tokenizer
        pytest.skip(f"Qwen3.5 tokenizer unavailable: {e}")


# -- Datum introspection helpers ---------------------------------------------
# A tinker Datum holds right-shifted input tokens in ``model_input`` and aligned
# per-token ``weights`` / ``target_tokens`` TensorData in ``loss_fn_inputs``.


def _input_ids(datum) -> list[int]:
    return list(datum.model_input.to_ints())


def _weights(datum) -> list[float]:
    return list(datum.loss_fn_inputs["weights"].data)


def _targets(datum) -> list[int]:
    return list(datum.loss_fn_inputs["target_tokens"].data)


def _trained_text(datum, tokenizer) -> str:
    """Decode only the tokens that carry loss (weight > 0)."""
    trained = [t for t, w in zip(_targets(datum), _weights(datum), strict=True) if w > 0]
    return tokenizer.decode(trained)


def _full_tokens(datum) -> list[int]:
    """Reconstruct the pre-shift token stream from a Datum."""
    return _input_ids(datum) + _targets(datum)[-1:]


def test_row_tools_reach_renderer_without_tokenizer():
    class RecordingRenderer:
        def render(self, messages, *, tools=None, add_generation_prompt=False):
            self.messages = messages
            self.tools = tools
            return SimpleNamespace(token_ids=[1, 2, 3], message_indices=[-1, 0, 1], is_content=[])

    renderer = RecordingRenderer()
    datum = conversation_to_datum(
        [
            {"role": "user", "content": "inspect", "trainable": False},
            {"role": "assistant", "content": "done", "trainable": True},
        ],
        renderer,
        max_length=None,
        tools=[
            {
                "type": "function",
                "function": {"name": "bash", "parameters": {"type": "object"}},
            }
        ],
    )

    assert datum.model_input.length == 2
    assert renderer.messages[0] == {"role": "user", "content": "inspect", "trainable": False}
    assert renderer.tools[0]["function"]["name"] == "bash"

    with pytest.raises(SFTConfigError, match="canonical rendering"):
        conversation_to_datum(
            [
                {"role": "user", "content": "inspect", "trainable": False},
                {"role": "assistant", "content": "done", "trainable": True},
            ],
            renderer,
            max_length=None,
            tools=[{"type": "custom", "name": "not-supported"}],
        )


def test_content_metadata_excludes_assistant_scaffolding_from_loss():
    class ContentAwareRenderer:
        @staticmethod
        def render(messages, *, tools=None, add_generation_prompt=False):
            # user body, assistant opener, two answer tokens, assistant closer
            return SimpleNamespace(
                token_ids=[10, 11, 12, 13, 14],
                message_indices=[0, 1, 1, 1, 1],
                is_content=[True, False, True, True, False],
            )

    datum = conversation_to_datum(
        [
            {"role": "user", "content": "question", "trainable": False},
            {"role": "assistant", "content": "answer", "trainable": True},
        ],
        ContentAwareRenderer(),
        max_length=None,
    )

    # Datum construction right-shifts the raw token weights by one position.
    assert _weights(datum) == [0.0, 1.0, 1.0, 0.0]


@pytest.mark.parametrize("with_content_metadata", [False, True])
def test_renderer_rejects_rewritten_trainable_target(with_content_metadata):
    class HistoryRewritingRenderer:
        @staticmethod
        def render(messages, *, tools=None, add_generation_prompt=False):
            if len(messages) == 4:
                # The target remains [11, 12], but later history rewrites its
                # causal user context from token 10 to token 99.
                rendered = SimpleNamespace(
                    token_ids=[99, 11, 12, 13, 14],
                    message_indices=[0, 1, 1, 2, 3],
                )
            else:
                rendered = SimpleNamespace(
                    token_ids=list(range(10, 10 + len(messages) + 1)),
                    message_indices=list(range(len(messages))) + [len(messages) - 1],
                )
            if with_content_metadata:
                rendered.is_content = [True] * len(rendered.token_ids)
            return rendered

    with pytest.raises(SFTConfigError, match="Explode this trajectory per target"):
        conversation_to_datum(
            [
                {"role": "user", "content": "first", "trainable": False},
                {"role": "assistant", "content": "old target", "trainable": True},
                {"role": "user", "content": "second", "trainable": False},
                {"role": "assistant", "content": "current target", "trainable": True},
            ],
            HistoryRewritingRenderer(),
            max_length=None,
        )


@pytest.mark.parametrize(
    ("message_indices", "is_content", "error"),
    [
        ([0, 0], [True], "incomplete content-token metadata"),
        ([-2, 0], [True, True], "invalid message index"),
        ([1, 0], [True, True], "invalid message index"),
    ],
)
def test_malformed_attribution_fails_as_dataset_error(message_indices, is_content, error):
    class MalformedRenderer:
        @staticmethod
        def render(messages, *, tools=None, add_generation_prompt=False):
            return SimpleNamespace(
                token_ids=[10, 11],
                message_indices=message_indices,
                is_content=is_content,
            )

    with pytest.raises(SFTConfigError, match=error):
        conversation_to_datum(
            [{"role": "assistant", "content": "answer", "trainable": True}],
            MalformedRenderer(),
            max_length=None,
        )


def test_interleaved_content_parts_fail_instead_of_being_reordered():
    from rllm.renderers.types import RenderedTokens

    class RecordingRenderer:
        @staticmethod
        def render(messages, *, tools=None, add_generation_prompt=False):
            return RenderedTokens(token_ids=[1], message_indices=[0])

    with pytest.raises(SFTConfigError, match="interleaved"):
        conversation_to_datum(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "visible first"},
                        {"type": "thinking", "thinking": "hidden later"},
                    ],
                    "trainable": True,
                }
            ],
            RecordingRenderer(),
            max_length=None,
        )


# -- F1: default renderer must not crash on reasoning rows -------------------


def test_f1_reasoning_row_renders_through_build_sft_data(qwen_tokenizer, monkeypatch):
    """A reasoning row (assistant ``thinking`` + ``text`` parts) must survive the
    full ``build_config`` -> ``build_sft_data`` -> ``get_batch`` path.

    The model resolves to the production Qwen renderer, which preserves the
    reasoning as a ``<think>...</think>`` block.
    """
    rows = [
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}], "trainable": False},
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "Let me add: 2+2=4."}, {"type": "text", "text": "4"}],
                    "trainable": True,
                },
            ]
        }
    ]
    ds = Dataset(data=rows, name="t1_reasoning", split="train")
    spec = SFTSpec(model=QWEN, train_dataset=ds)

    monkeypatch.setattr("tinker_cookbook.tokenizer_utils.get_tokenizer", lambda _: qwen_tokenizer)
    cfg = TinkerSFTBackend(spec).build_config()
    tokenizer, train_ds, val_ds = build_sft_data(cfg, spec.train_dataset, None)

    datums = train_ds.get_batch(0)  # RED: RendererError here on current code
    assert len(datums) == 1
    datum = datums[0]
    assert sum(_weights(datum)) > 0

    text = tokenizer.decode(_input_ids(datum))
    assert "2+2=4" in text  # the thinking text is preserved
    assert "<think>" in text  # rendered as a qwen thinking block


# -- F2: dict tool_calls must render on the production renderer --------------


def test_f2_dict_tool_calls_render(qwen_tokenizer):
    """OpenAI dict-shaped ``tool_calls`` must render.

    The canonical OpenAI-shaped dictionary reaches the native renderer and the
    tool name lands in the rendered assistant turn.
    """
    renderer = resolve(QWEN, qwen_tokenizer).renderer
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "list files"}], "trainable": False},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
            "trainable": True,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}],
        },
    ]

    datum = conversation_to_datum(conversation, renderer, max_length=None)  # RED: AttributeError here
    assert sum(_weights(datum)) > 0
    assert "bash" in qwen_tokenizer.decode(_input_ids(datum))


# -- F3: parquet round-trip None-stamping ------------------------------------


def test_f3_roundtrip_stamps_tool_calls_none(qwen_tokenizer, tmp_path):
    """Registry round-trip (pandas ``to_parquet`` -> polars ``read_parquet``)
    unifies the messages struct schema, so a message WITHOUT ``tool_calls`` gets
    the key stamped to ``None`` (here the user turn).

    A ``None`` ``tool_calls`` is treated as absent and every row renders.
    """
    renderer = resolve(QWEN, qwen_tokenizer).renderer
    rows = [
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "list files"}], "trainable": False},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                    "trainable": True,
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}],
                },
            ]
        }
    ]
    path = tmp_path / "tool_calls.parquet"
    pd.DataFrame(rows).to_parquet(path)
    rt = pl.read_parquet(path).to_dicts()

    # Root cause: the flag-less user turn now carries tool_calls=None.
    assert rt[0]["messages"][0]["tool_calls"] is None

    datums = [conversation_to_datum(row["messages"], renderer, max_length=None) for row in rt]  # RED: TypeError here
    assert all(sum(_weights(d)) > 0 for d in datums)
    # The real tool call still renders once None is handled.
    assert "bash" in qwen_tokenizer.decode(_input_ids(datums[0]))


def test_f3_roundtrip_stamps_trainable_none(qwen_tokenizer, tmp_path):
    """Registry round-trip stamps ``trainable=None`` onto a flag-less row when it
    shares a parquet with a flagged (self-describing) row.

    The flag-less row's first message then *has* a ``trainable`` key (value
    ``None``), so ``_ensure_trainable`` treats it as self-describing and passes
    it straight to the CUSTOMIZED renderer.

    RED today: ``build_supervised_example`` does ``int(None)`` for the flag-less
    row -> ``TypeError: int() argument must be a string ... not 'NoneType'``.

    GREEN: a ``None`` flag is treated as absent; the flag-less row falls back to
    the derived default (train the assistant turn only).

    NOTE: both rows use list-of-parts ``content`` so the parquet ``messages``
    column stays a uniform type. The registry requires this (see
    ``rllm.eval.curation._to_parts``): pandas/pyarrow refuse to write a column
    that mixes list-content and str-content messages, which would fail the write
    before this round-trip bug could surface.
    """
    renderer = resolve(QWEN, qwen_tokenizer).renderer
    rows = [
        # Flagged reasoning row (T1-style): every message carries ``trainable``.
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}], "trainable": False},
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "Let me add: 2+2=4."}, {"type": "text", "text": "4"}],
                    "trainable": True,
                },
            ]
        },
        # Plain flag-less row (e.g. an external --train-file), no ``trainable``.
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            ]
        },
    ]
    path = tmp_path / "trainable.parquet"
    pd.DataFrame(rows).to_parquet(path)
    rt = pl.read_parquet(path).to_dicts()

    # Root cause: the flag-less row's messages were stamped trainable=None.
    assert all(m["trainable"] is None for m in rt[1]["messages"])

    datums = [conversation_to_datum(row["messages"], renderer, max_length=None) for row in rt]  # RED: TypeError here
    assert all(sum(_weights(d)) > 0 for d in datums)
    # Flag-less row falls back to the assistant-only default.
    trained = _trained_text(datums[1], qwen_tokenizer)
    assert "hello" in trained
    assert "user\nhi" not in trained


# -- max_length truncation must be loud (vetting handoff item b) --------------


def test_truncation_past_max_length_warns_loudly(qwen_tokenizer, caplog):
    """Overlength rows warn and fail when truncation removes every target.

    ``SFTSpec.max_length`` defaults to 2048, which silently truncated every
    long trajectory row at datum build. The tail — including the final trainable
    turn — must not disappear silently or enter training with zero loss.
    """
    import logging

    import rllm.trainer.sft.tinker_dataset as td

    renderer = resolve(QWEN, qwen_tokenizer).renderer
    convo = [
        {"role": "user", "content": "count: " + " ".join(str(i) for i in range(300)), "trainable": False},
        {"role": "assistant", "content": "done", "trainable": True},
    ]

    td._truncation_warn_count = 0
    with caplog.at_level(logging.WARNING, logger="rllm.trainer.sft.tinker_dataset"):
        with pytest.raises(SFTConfigError, match="no trainable tokens"):
            conversation_to_datum(convo, renderer, max_length=64)
    warned = [r for r in caplog.records if "max_length" in r.getMessage()]
    assert warned, "expected a loud truncation warning"
    assert "64" in warned[0].getMessage()

    # A row that fits must not warn.
    caplog.clear()
    td._truncation_warn_count = 0
    with caplog.at_level(logging.WARNING, logger="rllm.trainer.sft.tinker_dataset"):
        conversation_to_datum(convo, renderer, max_length=100_000)
    assert not [r for r in caplog.records if "max_length" in r.getMessage()]


def test_tools_reasoning_and_whitespace_match_production_qwen35_renderer(qwen35_tokenizer, monkeypatch):
    from renderers.qwen35 import Qwen35Renderer

    from rllm.data.sft_bridges import bridge_messages

    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    wire_messages = [
        {"role": "system", "content": "Solve the task.\nKeep whitespace."},
        {"role": "user", "content": "Inspect the repository."},
        {
            "role": "assistant",
            "content": "  ",
            "reasoning_content": "I should list the files first.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "a.py"},
        {
            "role": "assistant",
            "content": "Done.",
            "reasoning_content": "The repository has one file.",
        },
    ]
    canonical = bridge_messages([{"messages": wire_messages, "tools": tools}])[0].to_record()
    dataset = Dataset(data=[canonical], name="tool-parity", split="train")
    spec = SFTSpec(
        model=QWEN35,
        train_dataset=dataset,
        max_length=100_000,
        overrides={"data": {"renderer_name": "qwen3_5"}},
    )
    monkeypatch.setattr("tinker_cookbook.tokenizer_utils.get_tokenizer", lambda _: qwen35_tokenizer)
    config = TinkerSFTBackend(spec).build_config()
    _, training_dataset, _ = build_sft_data(config, dataset, None)
    datum = training_dataset.get_batch(0)[0]

    serving_renderer = resolve(QWEN35, qwen35_tokenizer).renderer
    assert isinstance(serving_renderer, Qwen35Renderer)
    serving_ids = serving_renderer.render_ids(
        wire_messages,
        tools=tools,
        add_generation_prompt=False,
    )
    assert _full_tokens(datum) == serving_ids
    trained = _trained_text(datum, qwen35_tokenizer)
    assert "list the files" in trained
    assert "repository has one file" in trained
    assert "Inspect the repository" not in trained
