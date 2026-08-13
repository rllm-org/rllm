"""Red tests for the tinker SFT render path (PR #739 follow-up).

Three empirically-confirmed defects in ``rllm.trainer.sft`` render conversations
into tinker Datums incorrectly. Each test below asserts the *fixed* (green)
behavior and therefore FAILS on current code with the documented exception:

  F1  the default ``role_colon`` renderer crashes on reasoning rows (multimodal
      ``thinking`` content) -> ``tinker_cookbook`` ``RendererError``.
  F2  dict-shaped ``tool_calls`` crash the qwen3 renderer, which expects
      structured ``ToolCall`` objects -> ``AttributeError`` ('dict' has no
      attribute 'function').
  F3  a pandas->polars parquet round-trip unifies the per-row struct schema and
      stamps ``None`` into keys only *some* messages carry: ``tool_calls`` (a)
      and ``trainable`` (b) -> ``TypeError`` when the renderer iterates / ints
      the ``None``.

These use the REAL ``tinker_cookbook`` renderers and a REAL locally-cached
Qwen3-0.6B tokenizer (skipped if unavailable / offline), mirroring the guard
pattern of ``tests/test_renderers.py``.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest  # noqa: E402

pytest.importorskip("tinker_cookbook")
pytest.importorskip("tinker")

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from tinker_cookbook.renderers import get_renderer  # noqa: E402

from rllm.data import Dataset  # noqa: E402
from rllm.trainer.sft import SFTSpec  # noqa: E402
from rllm.trainer.sft.backend import SFTConfigError  # noqa: E402
from rllm.trainer.sft.tinker_backend import TinkerSFTBackend, build_sft_data  # noqa: E402
from rllm.trainer.sft.tinker_dataset import conversation_to_datum  # noqa: E402

QWEN = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(QWEN)
    except OSError as e:  # not cached / offline
        pytest.skip(f"Qwen3-0.6B tokenizer unavailable: {e}")


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
    import tinker
    import torch

    class RecordingRenderer:
        strip_thinking_from_history = False

        def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
            assert tools[0]["name"] == "bash"
            return [{"role": "system", "content": system_prompt or "tools"}]

        def build_supervised_example(self, messages, train_on_what):
            self.messages = messages
            return tinker.ModelInput.from_ints([1, 2, 3]), torch.tensor([0.0, 1.0, 1.0])

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
    assert renderer.messages[0]["role"] == "system"
    assert renderer.messages[0]["trainable"] is False


# -- F1: default renderer must not crash on reasoning rows -------------------


def test_f1_reasoning_row_renders_through_build_sft_data(qwen_tokenizer):
    """A reasoning row (assistant ``thinking`` + ``text`` parts) must survive the
    full ``build_config`` -> ``build_sft_data`` -> ``get_batch`` path.

    RED today: ``build_sft_data`` resolves the ``role_colon`` default renderer
    (tinker.yaml), and ``get_batch(0)`` raises tinker's ``RendererError``
    ("Expected text content, got multimodal content ...").

    GREEN: the model's renderer resolves to a qwen thinking renderer, which
    preserves the reasoning as a ``<think>...</think>`` block. (Qwen3-0.6B is
    absent from ``model_info``'s recommended-renderer map, but every qwen3-family
    thinking renderer emits the ``<think>`` tag, so we assert the literals.)
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

    cfg = TinkerSFTBackend(spec).build_config()
    tokenizer, train_ds, val_ds = build_sft_data(cfg, spec.train_dataset, None)

    datums = train_ds.get_batch(0)  # RED: RendererError here on current code
    assert len(datums) == 1
    datum = datums[0]
    assert sum(_weights(datum)) > 0

    text = tokenizer.decode(_input_ids(datum))
    assert "2+2=4" in text  # the thinking text is preserved
    assert "<think>" in text  # rendered as a qwen thinking block


# -- F2: dict tool_calls must render on the qwen3 renderer -------------------


def test_f2_dict_tool_calls_render(qwen_tokenizer):
    """OpenAI dict-shaped ``tool_calls`` must render.

    RED today: the qwen3 renderer does ``tool_call.function`` on a plain dict ->
    ``AttributeError: 'dict' object has no attribute 'function'``.

    GREEN: the dict is normalized to a structured ``ToolCall`` and the tool name
    lands in the rendered assistant turn.
    """
    renderer = get_renderer("qwen3", qwen_tokenizer)
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

    RED today: the qwen3 renderer iterates ``message["tool_calls"]`` on the user
    turn -> ``TypeError: 'NoneType' object is not iterable``.

    GREEN: a ``None`` ``tool_calls`` is treated as absent and every row renders.
    """
    renderer = get_renderer("qwen3", qwen_tokenizer)
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
    renderer = get_renderer("qwen3", qwen_tokenizer)
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
    assert "hi" not in trained


# -- max_length truncation must be loud (vetting handoff item b) --------------


def test_truncation_past_max_length_warns_loudly(qwen_tokenizer, caplog):
    """A row that renders past ``data.max_length`` must emit a loud warning.

    ``SFTSpec.max_length`` defaults to 2048, which silently truncated every
    long trajectory row at datum build (the tail — including the final trainable
    turn — dropped from training with zero signal to the user).

    RED today: ``datum_from_model_input_weights`` truncates silently.
    """
    import logging

    import rllm.trainer.sft.tinker_dataset as td

    renderer = get_renderer("qwen3", qwen_tokenizer)
    convo = [
        {"role": "user", "content": "count: " + " ".join(str(i) for i in range(300)), "trainable": False},
        {"role": "assistant", "content": "done", "trainable": True},
    ]

    td._truncation_warn_count = 0
    with caplog.at_level(logging.WARNING, logger="rllm.trainer.sft.tinker_dataset"):
        datum = conversation_to_datum(convo, renderer, max_length=64)
    assert datum.model_input.length <= 64
    warned = [r for r in caplog.records if "max_length" in r.getMessage()]
    assert warned, "expected a loud truncation warning"
    assert "64" in warned[0].getMessage()

    # A row that fits must not warn.
    caplog.clear()
    td._truncation_warn_count = 0
    with caplog.at_level(logging.WARNING, logger="rllm.trainer.sft.tinker_dataset"):
        conversation_to_datum(convo, renderer, max_length=100_000)
    assert not [r for r in caplog.records if "max_length" in r.getMessage()]


def test_tools_and_reasoning_match_unified_serving_renderer(qwen_tokenizer):
    from rllm.data.sft_bridges import bridge_messages
    from rllm.renderers.adapters import TinkerRendererAdapter

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
        {"role": "system", "content": "Solve the task."},
        {"role": "user", "content": "Inspect the repository."},
        {
            "role": "assistant",
            "content": "",
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
        model=QWEN,
        train_dataset=dataset,
        max_length=100_000,
        overrides={"data": {"renderer_name": "qwen3"}},
    )
    config = TinkerSFTBackend(spec).build_config()
    _, training_dataset, _ = build_sft_data(config, dataset, None)
    datum = training_dataset.get_batch(0)[0]

    serving_ids = TinkerRendererAdapter(get_renderer("qwen3", qwen_tokenizer)).render_ids(
        wire_messages,
        tools=tools,
        add_generation_prompt=False,
    )
    assert _full_tokens(datum) == serving_ids
    trained = _trained_text(datum, qwen_tokenizer)
    assert "list the files" in trained
    assert "repository has one file" in trained
    assert "Inspect the repository" not in trained


def test_new_user_query_strips_prior_reasoning_in_training_and_serving(qwen_tokenizer):
    from rllm.data.sft_bridges import bridge_messages
    from rllm.renderers.adapters import TinkerRendererAdapter

    wire_messages = [
        {"role": "user", "content": "First question."},
        {
            "role": "assistant",
            "content": "First answer.",
            "reasoning_content": "SECRET-OLD-REASONING",
        },
        {"role": "user", "content": "Second question."},
        {
            "role": "assistant",
            "content": "Second answer.",
            "reasoning_content": "CURRENT-REASONING",
        },
    ]
    canonical = bridge_messages([{"messages": wire_messages}])[0]
    renderer = get_renderer("qwen3", qwen_tokenizer)
    renderer.strip_thinking_from_history = False
    datum = conversation_to_datum(canonical.to_record()["messages"], renderer, max_length=None)
    serving_ids = TinkerRendererAdapter(get_renderer("qwen3", qwen_tokenizer)).render_ids(
        wire_messages,
        add_generation_prompt=False,
    )

    assert _full_tokens(datum) == serving_ids
    rendered = qwen_tokenizer.decode(serving_ids)
    assert "SECRET-OLD-REASONING" not in rendered
    assert "CURRENT-REASONING" in rendered


def test_invalid_row_tool_declaration_fails_with_dataset_error(qwen_tokenizer):
    renderer = get_renderer("qwen3", qwen_tokenizer)
    messages = [
        {"role": "user", "content": "inspect", "trainable": False},
        {"role": "assistant", "content": "done", "trainable": True},
    ]

    with pytest.raises(SFTConfigError, match="invalid tool declarations"):
        conversation_to_datum(
            messages,
            renderer,
            max_length=None,
            tools=[{"type": "custom", "name": "not-supported"}],
        )
