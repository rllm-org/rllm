"""Tests for the unified renderer layer (rllm.renderers).

Fast unit tests of the generic bridge use a toy renderer (no deps). The tinker
adapter / prime-rl parity tests use the locally-cached Qwen3-0.6B tokenizer and
are skipped if it (or tinker_cookbook) is unavailable. Run offline.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from rllm.renderers import BridgingRendererMixin, get_renderer, resolve  # noqa: E402
from rllm.renderers.adapters import (  # noqa: E402
    ChatTemplateAdapter,
    TinkerRendererAdapter,
    to_tinker_messages,
    to_tinker_tool_specs,
)

QWEN = "Qwen/Qwen3-0.6B"

# Toy template token ids.
BOS = {"user": 1, "assistant": 2, "system": 3, "tool": 4}
CLOSE = 9


class ToyRenderer(BridgingRendererMixin):
    """Deterministic toy renderer: each msg -> [BOS_role, *chars, CLOSE]."""

    close_token_ids = {CLOSE}
    synthesize_close = CLOSE

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        out: list[int] = []
        for m in messages:
            out.append(BOS[m["role"]])
            out.extend(ord(c) for c in (m.get("content") or ""))
            out.append(CLOSE)
        if add_generation_prompt:
            out.append(BOS["assistant"])
        return out


# ── generic bridge unit tests (no deps) ──────────────────────────────────────


def _split(r, messages):
    """prompt_ids, completion_ids for a [.., assistant]-terminated convo."""
    prompt = r.render_ids(messages[:-1], add_generation_prompt=True)
    full = r.render_ids(messages, add_generation_prompt=False)
    return prompt, full[len(prompt) :]


def test_bridge_reconstructs_full_render():
    r = ToyRenderer()
    convo = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    prompt, completion = _split(r, convo)
    new = [{"role": "user", "content": "next"}]

    bridged = r.bridge_to_next_turn(prompt, completion, new)
    expected = r.render_ids(convo + new, add_generation_prompt=True)
    assert bridged is not None
    assert bridged.token_ids == expected


def test_bridge_keeps_sampled_tokens_verbatim():
    r = ToyRenderer()
    prompt = r.render_ids([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    completion = [ord("Z"), ord("Z"), CLOSE]  # arbitrary "sampled" tokens
    prev = prompt + completion

    bridged = r.bridge_to_next_turn(prompt, completion, [{"role": "user", "content": "q"}])
    assert bridged is not None
    assert bridged.token_ids[: len(prev)] == prev  # prefix invariant


def test_bridge_synthesizes_close_when_truncated():
    r = ToyRenderer()
    prompt = r.render_ids([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    completion = [ord("a"), ord("b")]  # no CLOSE -> truncated turn
    bridged = r.bridge_to_next_turn(prompt, completion, [{"role": "user", "content": "q"}])
    assert bridged is not None
    prev = prompt + completion
    assert bridged.token_ids[: len(prev)] == prev


def test_bridge_rejects_assistant_extension():
    r = ToyRenderer()
    prompt = r.render_ids([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    completion = [ord("y"), CLOSE]
    assert r.bridge_to_next_turn(prompt, completion, [{"role": "assistant", "content": "x"}]) is None
    assert r.bridge_to_next_turn(prompt, completion, []) is None


# ── tinker adapter + prime-rl parity (cached tokenizer) ──────────────────────


@pytest.fixture(scope="module")
def qwen_tokenizer():
    pytest.importorskip("tinker_cookbook")
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(QWEN)
    except Exception as e:  # not cached / offline
        pytest.skip(f"Qwen3-0.6B tokenizer unavailable: {e}")


def _tinker_qwen3(tokenizer):
    from tinker_cookbook import renderers as tk

    from rllm.renderers.adapters import TinkerRendererAdapter

    return TinkerRendererAdapter(tk.get_renderer("qwen3", tokenizer))


def _prime_qwen3(tokenizer):
    import renderers

    return renderers.create_renderer(tokenizer, renderers.config_from_name("qwen3"))


def test_tinker_adapter_renders(qwen_tokenizer):
    a = _tinker_qwen3(qwen_tokenizer)
    ids = a.render_ids([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert ids and isinstance(ids[0], int)
    assert a.get_stop_token_ids()  # non-empty


def test_tinker_parse_normalizes_structured_content_and_tool_calls():
    class CookbookRenderer:
        @staticmethod
        def get_stop_sequences():
            return [99]

        @staticmethod
        def parse_response(_token_ids):
            return (
                {
                    "content": [
                        {"type": "thinking", "thinking": "first\n"},
                        {"type": "thinking", "thinking": "second"},
                        {"type": "text", "text": "answer "},
                        {"type": "text", "text": "body"},
                    ],
                    "tool_calls": [
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="bash", arguments='{"command":"ls"}'),
                        )
                    ],
                },
                None,
            )

    parsed = TinkerRendererAdapter(CookbookRenderer()).parse_response([1, 2, 3], tools=[])

    assert parsed.reasoning_content == "first\nsecond"
    assert parsed.content == "answer body"
    import renderers

    parsed_tool_call_type = getattr(renderers, "ParsedToolCall", None)
    if parsed_tool_call_type is None:
        assert parsed.tool_calls == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"ls"}'},
            }
        ]
    else:
        assert isinstance(parsed.tool_calls[0], parsed_tool_call_type)
        assert parsed.tool_calls[0].name == "bash"
        assert parsed.tool_calls[0].arguments == '{"command":"ls"}'
        assert parsed.tool_calls[0].status.value == "ok"


def test_cookbook_input_normalizes_reasoning_tools_and_trainable():
    converted = to_tinker_messages(
        [
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "plan",
                "trainable": True,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"ls"}'},
                    }
                ],
            }
        ]
    )

    assert "reasoning_content" not in converted[0]
    assert "trainable" not in converted[0]
    assert converted[0]["content"] == [
        {"type": "thinking", "thinking": "plan"},
        {"type": "text", "text": "answer"},
    ]
    assert converted[0]["tool_calls"][0].function.name == "bash"


def test_cookbook_adapter_attributes_tokens_by_message():
    class ModelInput:
        def __init__(self, token_ids):
            self._token_ids = token_ids

        def to_ints(self):
            return self._token_ids

    class CookbookRenderer:
        @staticmethod
        def get_stop_sequences():
            return [99]

        @staticmethod
        def build_supervised_example(messages):
            return ModelInput(list(range(1, len(messages) + 1))), None

    rendered = TinkerRendererAdapter(CookbookRenderer()).render(
        [
            {"role": "user", "content": "question", "trainable": False},
            {"role": "assistant", "content": "answer", "trainable": True},
        ]
    )

    assert rendered.token_ids == [1, 2]
    assert rendered.message_indices == [0, 1]

    class RewritingCookbookRenderer(CookbookRenderer):
        @staticmethod
        def build_supervised_example(messages):
            token_ids = [1] if len(messages) == 1 else [9, 2]
            return ModelInput(token_ids), None

    with pytest.raises(ValueError, match="rewrites previously rendered history"):
        TinkerRendererAdapter(RewritingCookbookRenderer()).render(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        )


def test_openai_tool_declarations_become_cookbook_specs():
    specs = to_tinker_tool_specs(
        [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a command",
                    "parameters": {"type": "object"},
                },
            }
        ]
    )

    assert specs[0]["name"] == "bash"
    assert specs[0]["parameters"] == {"type": "object"}


def test_tinker_adapter_bridge_preserves_prefix_and_matches_prime_when_available(qwen_tokenizer):
    adapter = _tinker_qwen3(qwen_tokenizer)
    prime = _prime_qwen3(qwen_tokenizer)

    prompt = prime.render_ids([{"role": "user", "content": "hello there"}], add_generation_prompt=True)
    completion = [
        t
        for t in prime.render_ids(
            [{"role": "user", "content": "hello there"}, {"role": "assistant", "content": "hi back"}],
            add_generation_prompt=False,
        )[len(prompt) :]
    ]

    new = [{"role": "user", "content": "and then?"}]
    a_out = adapter.bridge_to_next_turn(prompt, completion, new)
    p_out = prime.bridge_to_next_turn(prompt, completion, new)
    assert a_out is not None
    assert a_out.token_ids[: len(prompt + completion)] == prompt + completion
    if p_out is not None:  # Newer native renderers may request a full rerender.
        assert a_out.token_ids == p_out.token_ids


# ── registry resolution ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("renderer_name", "prime_name"),
    [
        ("qwen3_5", "qwen3.5"),
        ("nemotron3", "nemotron-3"),
        ("nemotron_3", "nemotron-3"),
        ("nemotron-3", "nemotron-3"),
    ],
)
def test_resolve_legacy_name_through_pinned_typed_factory(monkeypatch, renderer_name, prime_name):
    calls = []
    lookups = []
    config = object()

    class NativeRenderer: ...

    def config_from_name(name):
        lookups.append(name)
        return config

    monkeypatch.setitem(
        sys.modules,
        "renderers",
        SimpleNamespace(
            config_from_name=config_from_name,
            create_renderer=lambda tokenizer, renderer_config, **_kwargs: calls.append((tokenizer, renderer_config)) or NativeRenderer(),
        ),
    )
    tokenizer = object()

    result = resolve("model", tokenizer, renderer_name=renderer_name)

    assert result.source == "prime"
    assert lookups == [prime_name]
    assert calls == [(tokenizer, config)]


def test_resolve_passes_chat_template_kwargs_to_prime_renderer(monkeypatch):
    calls = []
    config = object()

    class NativeRenderer: ...

    monkeypatch.setitem(
        sys.modules,
        "renderers",
        SimpleNamespace(
            config_from_name=lambda _name: config,
            create_renderer=lambda tokenizer, renderer_config, *, chat_template_kwargs=None: calls.append((tokenizer, renderer_config, chat_template_kwargs)) or NativeRenderer(),
        ),
    )
    tokenizer = object()

    result = resolve(
        "model",
        tokenizer,
        renderer_name="nemotron3",
        chat_template_kwargs={"truncate_history_thinking": False},
    )

    assert result.source == "prime"
    assert calls == [(tokenizer, config, {"truncate_history_thinking": False})]


def test_resolve_prime_native_for_qwen(qwen_tokenizer):
    res = resolve(QWEN, qwen_tokenizer)
    assert res.source == "prime"


def test_resolve_chat_template_fallback(qwen_tokenizer, monkeypatch):
    # prime-rl/tinker auto-resolution keys off tokenizer.name_or_path; point it
    # at an unknown id so neither matches and we land on the fallback.
    monkeypatch.setattr(qwen_tokenizer, "name_or_path", "some/Unknown-Finetune-XYZ", raising=False)
    res = resolve("some/Unknown-Finetune-XYZ", qwen_tokenizer, family="auto")
    assert res.source == "chat_template"
    assert isinstance(res.renderer, ChatTemplateAdapter)


def test_unified_renderer_matches_chat_template_for_qwen(qwen_tokenizer):
    """FireworksEngine swaps prompt-building to render_ids; for qwen-family this
    must equal the existing apply_chat_template path (no tokenization regression)."""
    r = get_renderer(QWEN, qwen_tokenizer)
    msgs = [{"role": "user", "content": "hello there"}]
    unified = r.render_ids(msgs, add_generation_prompt=True)
    text = qwen_tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    baseline = qwen_tokenizer.encode(text, add_special_tokens=False)
    assert unified == baseline


# ── TokenAccumulator integration (gateway consumer) ──────────────────────────


def test_token_accumulator_extends_with_adapter():
    """Two cumulative turns through the gateway accumulator using the toy renderer."""
    import sys

    ta_path = os.path.join(os.path.dirname(__file__), "..", "rllm-model-gateway", "src")
    sys.path.insert(0, os.path.abspath(ta_path))
    try:
        from rllm_model_gateway.token_accumulator import TokenAccumulator
    except Exception as e:
        pytest.skip(f"token_accumulator import failed: {e}")

    r = ToyRenderer()
    acc = TokenAccumulator(renderer=r)

    convo = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    prompt, completion = _split(r, convo)
    acc.ingest_turn(prompt, completion)
    acc.update_prefix(convo)

    messages = convo + [{"role": "user", "content": "next"}]
    assert acc.is_cumulative(messages)
    new_messages = [m for m in messages[acc.message_count :] if m.get("role") != "assistant"]
    next_prompt = acc.build_next_prompt(new_messages)
    assert next_prompt is not None
    prev = prompt + completion
    assert next_prompt[: len(prev)] == prev


def test_fireworks_engine_skips_chatparser_when_renderer_pinned(qwen_tokenizer):
    """Pinning renderer_family must use the unified renderer and skip building
    ChatTemplateParser, whose eager verify_equivalence runs apply_chat_template
    and crashes on some served templates (e.g. GLM-5.2). Default keeps chat_parser."""
    pytest.importorskip("tinker")
    try:
        from rllm.engine.rollout.fireworks_engine import FireworksEngine
    except Exception as e:
        pytest.skip(f"FireworksEngine import failed: {e}")

    class _StubSampler: ...

    pinned = FireworksEngine(tokenizer=qwen_tokenizer, sampler=_StubSampler(), renderer_family="qwen3")
    assert pinned.unified_renderer is not None
    assert pinned.chat_parser is None
    assert pinned.bypass_render_with_parser is False  # renderer owns rendering+parsing

    default = FireworksEngine(tokenizer=qwen_tokenizer, sampler=_StubSampler())
    assert default.unified_renderer is None
    assert default.chat_parser is not None
    assert default.bypass_render_with_parser is True


def test_cookbook_renderer_name_prefix_match():
    """Fireworks-cookbook models auto-detect by family prefix (no config), incl.
    point releases prime-rl's exact-match map doesn't list (GLM-5.2)."""
    from rllm.renderers.registry import _cookbook_renderer_name

    assert _cookbook_renderer_name("zai-org/GLM-5.2") == "glm5"
    assert _cookbook_renderer_name("zai-org/GLM-5") == "glm5"
    assert _cookbook_renderer_name("zai-org/GLM-5.1") == "glm5"
    assert _cookbook_renderer_name("deepseek-ai/DeepSeek-V4") == "deepseek_v4"
    assert _cookbook_renderer_name("deepseek-ai/DeepSeek-V4-Flash") == "deepseek_v4"
    # Not cookbook families (prime-rl / fallback handle these):
    assert _cookbook_renderer_name("Qwen/Qwen3-8B") is None
    assert _cookbook_renderer_name("zai-org/GLM-4.5-Air") is None


def test_resolve_cookbook_does_not_crash_when_absent(qwen_tokenizer, monkeypatch):
    """A cookbook-family model still resolves cleanly (to the cookbook renderer if
    installed, else the chat-template fallback) — resolution never hard-fails."""
    monkeypatch.setattr(qwen_tokenizer, "name_or_path", "zai-org/GLM-5.2", raising=False)
    res = resolve("zai-org/GLM-5.2", qwen_tokenizer)
    assert res.source in ("tinker", "chat_template")


def test_assemble_model_output_uses_renderer_when_chat_parser_absent(qwen_tokenizer):
    """With a unified renderer active (chat_parser=None), assemble_model_output must
    parse the completion via the renderer instead of asserting chat_parser is set."""
    pytest.importorskip("tinker")
    try:
        from rllm.engine.rollout.fireworks_engine import FireworksEngine
    except Exception as e:
        pytest.skip(f"FireworksEngine import failed: {e}")

    class _StubSampler: ...

    class _StubOutput:
        def __init__(self, tokens):
            self.tokens = tokens
            self.logprobs = [0.0] * len(tokens)
            self.stop_reason = "stop"

    engine = FireworksEngine(tokenizer=qwen_tokenizer, sampler=_StubSampler(), renderer_family="qwen3")
    assert engine.unified_renderer is not None and engine.chat_parser is None

    completion = qwen_tokenizer.encode("hello", add_special_tokens=False) + engine.unified_renderer.get_stop_token_ids()[:1]
    out = engine.assemble_model_output([1, 2, 3], _StubOutput(completion))  # must not raise
    assert out.completion_ids == completion
    assert out.prompt_ids == [1, 2, 3]
    assert isinstance(out.content, str)
    assert isinstance(out.tool_calls, list)


def test_fireworks_engine_explicit_bypass_overrides_renderer(qwen_tokenizer):
    """bypass_render_with_parser=True is an explicit escape hatch: force
    ChatTemplateParser and skip the unified renderer, even when one is pinned."""
    pytest.importorskip("tinker")
    try:
        from rllm.engine.rollout.fireworks_engine import FireworksEngine
    except Exception as e:
        pytest.skip(f"FireworksEngine import failed: {e}")

    class _StubSampler: ...

    engine = FireworksEngine(
        tokenizer=qwen_tokenizer,
        sampler=_StubSampler(),
        renderer_family="qwen3",  # would normally pin the unified renderer
        bypass_render_with_parser=True,
    )
    assert engine.unified_renderer is None
    assert engine.chat_parser is not None
    assert engine.bypass_render_with_parser is True
