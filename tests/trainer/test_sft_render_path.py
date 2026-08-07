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
NEMOTRON = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(QWEN)
    except OSError as e:  # not cached / offline
        pytest.skip(f"Qwen3-0.6B tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def nemotron_tokenizer():
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    try:
        return get_tokenizer(NEMOTRON)
    except Exception as e:  # noqa: BLE001 - optional offline integration asset
        pytest.skip(f"Nemotron-3 tokenizer unavailable: {e}")


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
    """Reconstruct the pre-shift token stream from a Datum.

    ``datum_from_model_input_weights`` stores right-shifted inputs
    (``tokens[:-1]``) and left-shifted targets (``tokens[1:]``); the original
    render is ``inputs + [last target]``.
    """
    return _input_ids(datum) + _targets(datum)[-1:]


# -- F1: default renderer must not crash on reasoning rows -------------------


def test_hf_template_config_override_is_rejected_before_tokenizer_load(
    monkeypatch,
):
    from tinker_cookbook import tokenizer_utils

    ds = Dataset(
        data=[
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ]
            }
        ],
        name="hf-template-rejection",
        split="train",
    )
    spec = SFTSpec(
        model=QWEN,
        train_dataset=ds,
        overrides={"data": {"rllm": {"tokenize_and_mask_method": "hf_template"}}},
    )
    cfg = TinkerSFTBackend(spec).build_config()
    monkeypatch.setattr(
        tokenizer_utils,
        "get_tokenizer",
        lambda *_: pytest.fail("tokenizer should not load for a rejected mode"),
    )

    with pytest.raises(SFTConfigError, match="train/serve mismatch"):
        build_sft_data(cfg, ds, None)


def test_f1_reasoning_row_renders_through_build_sft_data(qwen_tokenizer):
    """A reasoning row (assistant ``thinking`` + ``text`` parts) must survive the
    full ``build_config`` -> ``build_sft_data`` -> ``get_batch`` path.

    The model's renderer resolves to a qwen thinking renderer, which
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

    datums = train_ds.get_batch(0)
    assert len(datums) == 1
    datum = datums[0]
    assert sum(_weights(datum)) > 0

    text = tokenizer.decode(_input_ids(datum))
    assert "2+2=4" in text  # the thinking text is preserved
    assert "<think>" in text  # rendered as a qwen thinking block


# -- F2: dict tool_calls must render on the qwen3 renderer -------------------


def test_f2_dict_tool_calls_render(qwen_tokenizer):
    """OpenAI dict-shaped ``tool_calls`` must render.

    The dict is normalized to a structured ``ToolCall`` and the tool name
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

    datum = conversation_to_datum(conversation, renderer, max_length=None)
    assert sum(_weights(datum)) > 0
    assert "bash" in qwen_tokenizer.decode(_input_ids(datum))


# -- F3: parquet round-trip None-stamping ------------------------------------


def test_f3_roundtrip_stamps_tool_calls_none(qwen_tokenizer, tmp_path):
    """Registry round-trip (pandas ``to_parquet`` -> polars ``read_parquet``)
    unifies the messages struct schema, so a message WITHOUT ``tool_calls`` gets
    the key stamped to ``None`` (here the user turn).

    A ``None`` ``tool_calls`` is treated as absent and every row renders.
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

    datums = [conversation_to_datum(row["messages"], renderer, max_length=None) for row in rt]
    assert all(sum(_weights(d)) > 0 for d in datums)
    # The real tool call still renders once None is handled.
    assert "bash" in qwen_tokenizer.decode(_input_ids(datums[0]))


def test_f3_roundtrip_stamps_trainable_none(qwen_tokenizer, tmp_path):
    """Registry round-trip stamps ``trainable=None`` onto a flag-less row when it
    shares a parquet with a flagged (self-describing) row.

    The flag-less row's first message then *has* a ``trainable`` key (value
    ``None``), so ``_ensure_trainable`` treats it as self-describing and passes
    it straight to the CUSTOMIZED renderer.

    A ``None`` flag is treated as absent; the flag-less row falls back to
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


# -- max_length truncation must be explicit ----------------------------------


def test_overlength_row_errors_unless_truncation_is_explicit(qwen_tokenizer, caplog):
    """A row past ``data.max_length`` must be rejected by default.

    Right-truncating an agent trajectory preferentially deletes its final patch
    and explanation, so rLLM must never mutate one unless the caller explicitly
    selects the compatibility policy.
    """
    import logging

    import rllm.trainer.sft.tinker_dataset as td

    renderer = get_renderer("qwen3", qwen_tokenizer)
    convo = [
        {"role": "user", "content": "count: " + " ".join(str(i) for i in range(300)), "trainable": False},
        {"role": "assistant", "content": "done", "trainable": True},
    ]

    td._truncation_warn_count = 0
    with pytest.raises(SFTConfigError, match="renders to .*max_length=64"):
        conversation_to_datum(convo, renderer, max_length=64)

    # The lossy compatibility path remains available only by explicit opt-in.
    with caplog.at_level(logging.WARNING, logger="rllm.trainer.sft.tinker_dataset"):
        datum = conversation_to_datum(
            convo,
            renderer,
            max_length=64,
            overlength_policy="truncate",
        )
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
    """Training and Tinker serving must render the same complete trajectory.

    This covers the actual agentic boundary: sibling reasoning, structured tool
    calls, tool-role observations, and row-level tool declarations. The serving
    side is rLLM's unified ``TinkerRendererAdapter``; the training side is the
    SFT ``conversation_to_datum`` path.
    """
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
    canonical = bridge_messages(
        [{"messages": wire_messages, "tools": tools}],
        train_on="all",
    )[0].to_record()

    ds = Dataset(data=[canonical], name="tool-parity", split="train")
    spec = SFTSpec(
        model=QWEN,
        train_dataset=ds,
        max_length=100_000,
        overrides={"data": {"renderer_name": "qwen3"}},
    )
    cfg = TinkerSFTBackend(spec).build_config()
    _, training_ds, _ = build_sft_data(cfg, ds, None)
    datum = training_ds.get_batch(0)[0]
    serving_renderer = TinkerRendererAdapter(get_renderer("qwen3", qwen_tokenizer))
    serving_ids = serving_renderer.render_ids(
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
    """Qwen retains the active trace across tools but clears CoT before a new
    genuine user query; SFT and serving must choose the same boundary."""
    from tinker_cookbook.renderers import get_renderer

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
    canonical = bridge_messages([{"messages": wire_messages}], train_on="all")[0]
    renderer = get_renderer("qwen3", qwen_tokenizer)
    renderer.strip_thinking_from_history = False
    datum = conversation_to_datum(
        canonical.to_record()["messages"],
        renderer,
        max_length=None,
    )
    serving_ids = TinkerRendererAdapter(get_renderer("qwen3", qwen_tokenizer)).render_ids(wire_messages, add_generation_prompt=False)

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


def test_nemotron_multiturn_training_matches_tinker_serving(nemotron_tokenizer):
    """The Nemotron renderer preserves every post-user reasoning turn."""
    from rllm.data.sft_bridges import bridge_messages
    from rllm.renderers.adapters import TinkerRendererAdapter

    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]
    wire_messages = [
        {"role": "system", "content": "Use the shell."},
        {"role": "user", "content": "Inspect the repository."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "First inspect the files.",
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
    canonical = bridge_messages(
        [{"messages": wire_messages, "tools": tools}],
        train_on="all",
    )[0].to_record()
    renderer = get_renderer(
        "nemotron3",
        nemotron_tokenizer,
        model_name=NEMOTRON,
    )
    renderer.strip_thinking_from_history = False
    datum = conversation_to_datum(
        canonical["messages"],
        renderer,
        max_length=None,
        tools=tools,
    )
    serving_ids = TinkerRendererAdapter(get_renderer("nemotron3", nemotron_tokenizer, model_name=NEMOTRON)).render_ids(
        wire_messages,
        tools=tools,
        add_generation_prompt=False,
    )

    assert _full_tokens(datum) == serving_ids
    trained = _trained_text(datum, nemotron_tokenizer)
    assert "First inspect the files" in trained
    assert "repository has one file" in trained
