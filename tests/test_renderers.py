"""Tests for the rLLM-native renderer layer (rllm.renderers).

Routing tests are pure (no tokenizer / network). Render/parse/bridge tests are
guarded: they load a cached HF tokenizer and skip if unavailable (offline CI).

NOTE: this file lives at ``tests/`` root (not ``tests/renderers/``) on purpose —
a ``tests/renderers/`` package would land on ``sys.path`` under pytest's prepend
import mode and shadow prime-rl's top-level ``renderers`` package.
"""

from __future__ import annotations

import pytest

import rllm.renderers as R
from rllm.renderers import Backend
from rllm.renderers._common import iter_tool_specs, normalize_tool_calls
from rllm.renderers.types import ParsedResponse, RenderedTokens, ToolCall, ToolSpec

# --- Models with cached tokenizers used by the guarded integration tests. ---
PRIME_MODEL = "Qwen/Qwen3-8B"  # in prime-rl MODEL_RENDERER_MAP -> token bridge
FW_ONLY_MODEL = "deepseek-ai/DeepSeek-V4-Flash"  # prime-rl gap -> tinker/FW adapter


def _load(model: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model)
    except Exception as err:  # noqa: BLE001 - offline / not cached
        pytest.skip(f"tokenizer {model} unavailable: {err}")


def _hf_ids(tok, messages):
    out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(out, "keys"):  # BatchEncoding
        return list(out["input_ids"])
    if hasattr(out, "ids"):  # Encoding
        return list(out.ids)
    return list(out)


# ----------------------------- pure routing -----------------------------


def test_availability_flags_are_bools():
    for flag in (R.PRIME_AVAILABLE, R.TINKER_AVAILABLE, R.FIREWORKS_AVAILABLE):
        assert isinstance(flag, bool)


@pytest.mark.skipif(not R.PRIME_AVAILABLE, reason="prime-rl not installed")
def test_prime_model_routes_to_prime_with_bridge():
    d = R.select_backend("Qwen/Qwen3-8B")
    assert d.backend is Backend.PRIME
    assert d.has_bridge is True


@pytest.mark.skipif(not R.TINKER_AVAILABLE, reason="tinker_cookbook not installed")
def test_fw_only_model_routes_to_tinker():
    from rllm.renderers import _tinker

    d = R.select_backend("deepseek-ai/DeepSeek-V4-Flash")
    assert d.backend is Backend.TINKER
    assert d.renderer_name == "deepseek_v4"
    # The tinker/Fireworks backend now carries a generic cross-turn bridge.
    assert d.has_bridge is _tinker.BRIDGE_AVAILABLE


def test_unknown_model_routes_to_default():
    d = R.select_backend("nonexistent-org/Totally-Unknown-Model-9000")
    assert d.backend is Backend.DEFAULT
    assert d.has_bridge is False


def test_explicit_family_forces_prime():
    d = R.select_backend("nonexistent-org/Unknown", renderer_family="qwen3.5")
    assert d.backend is Backend.PRIME
    assert d.renderer_family == "qwen3.5"
    assert d.has_bridge is True


def test_explicit_name_forces_tinker():
    d = R.select_backend("nonexistent-org/Unknown", renderer_name="deepseek_v4")
    assert d.backend is Backend.TINKER
    assert d.renderer_name == "deepseek_v4"


# ----------------------------- pure helpers -----------------------------


def test_iter_tool_specs_openai_and_bare_and_passthrough():
    openai = {
        "type": "function",
        "function": {"name": "add", "description": "d", "parameters": {"x": 1}},
    }
    bare = {"name": "sub", "parameters": {"y": 2}}
    native = ToolSpec(name="mul")
    specs = iter_tool_specs([openai, bare, native])
    assert [s.name for s in specs] == ["add", "sub", "mul"]
    assert specs[0].parameters == {"x": 1}
    assert specs[0].description == "d"


def test_iter_tool_specs_skips_non_function_and_empty():
    assert iter_tool_specs(None) == []
    assert iter_tool_specs([{"type": "web_search"}]) == []


def test_normalize_tool_calls_variants():
    class _Fn:
        def __init__(self, name, args):
            self.name, self.arguments = name, args

    class _ObjWithFunction:
        def __init__(self, name, args):
            self.function = _Fn(name, args)

    out = normalize_tool_calls(
        [
            {"name": "a", "arguments": {"k": 1}},
            {"name": "b", "arguments": '{"k": 2}'},  # string args -> parsed
            _ObjWithFunction("c", '{"k": 3}'),
            ToolCall(name="d", arguments={"k": 4}),
        ]
    )
    assert [tc.name for tc in out] == ["a", "b", "c", "d"]
    assert out[1].arguments == {"k": 2}
    assert out[2].arguments == {"k": 3}
    assert normalize_tool_calls(None) == []


# --------------------- guarded integration (render) ---------------------


@pytest.mark.skipif(not R.PRIME_AVAILABLE, reason="prime-rl not installed")
def test_prime_render_parity_with_hf_template():
    tok = _load(PRIME_MODEL)
    r = R.resolve(PRIME_MODEL, tok)
    assert r.backend == "prime" and r.has_bridge is True
    msgs = [{"role": "user", "content": "hi"}]
    ids = r.render_ids(msgs, add_generation_prompt=True)
    assert ids == _hf_ids(tok, msgs)


@pytest.mark.skipif(not R.PRIME_AVAILABLE, reason="prime-rl not installed")
def test_prime_bridge_extends_prefix_byte_for_byte():
    tok = _load(PRIME_MODEL)
    r = R.resolve(PRIME_MODEL, tok)
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    prompt_ids = r.render_ids(msgs, add_generation_prompt=True)
    completion_ids = tok.encode("4", add_special_tokens=False)
    bridged = r.bridge_to_next_turn(
        prompt_ids, completion_ids, [{"role": "user", "content": "And 3+3?"}]
    )
    assert isinstance(bridged, RenderedTokens)
    prefix = prompt_ids + completion_ids
    assert bridged.token_ids[: len(prefix)] == prefix


@pytest.mark.skipif(not R.TINKER_AVAILABLE, reason="tinker_cookbook not installed")
def test_tinker_adapter_renders_fw_only_model():
    tok = _load(FW_ONLY_MODEL)
    r = R.resolve(FW_ONLY_MODEL, tok)
    assert r.backend == "tinker"
    msgs = [{"role": "user", "content": "hi"}]
    ids = r.render_ids(msgs, add_generation_prompt=True)
    assert isinstance(ids, list) and len(ids) > 0 and all(isinstance(i, int) for i in ids)
    parsed = r.parse_response(tok.encode("ok", add_special_tokens=False))
    assert isinstance(parsed, ParsedResponse)


@pytest.mark.skipif(not R.TINKER_AVAILABLE, reason="tinker_cookbook not installed")
def test_tinker_adapter_bridge_extends_prefix_for_fw_model():
    """The generic bridge unlocks cumulative-token mode for a model prime-rl
    lacks (DeepSeek-V4): it extends the verbatim prior tokens byte-for-byte."""
    from rllm.renderers import _tinker

    if not _tinker.BRIDGE_AVAILABLE:
        pytest.skip("gateway tinker bridge unavailable")
    tok = _load(FW_ONLY_MODEL)
    r = R.resolve(FW_ONLY_MODEL, tok)
    assert r.has_bridge is True
    prompt_ids = r.render_ids([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    completion_ids = tok.encode("</think>hello", add_special_tokens=False) + [1]  # clean EOS
    bridged = r.bridge_to_next_turn(
        prompt_ids, completion_ids, [{"role": "user", "content": "bye"}]
    )
    assert isinstance(bridged, RenderedTokens)
    prefix = prompt_ids + completion_ids
    assert bridged.token_ids[: len(prefix)] == prefix
    # Assistant content in the new slice is refused (would re-tokenize sampled tokens).
    assert r.bridge_to_next_turn(prompt_ids, completion_ids, [{"role": "assistant", "content": "x"}]) is None


@pytest.mark.skipif(not R.PRIME_AVAILABLE or not R.TINKER_AVAILABLE, reason="need both backends")
def test_tinker_bridge_matches_prime_gold():
    """The generic tinker bridge equals prime-rl's hand-coded bridge byte-for-byte
    for a model both ecosystems support (Qwen3.5)."""
    from rllm.renderers import _tinker

    if not _tinker.BRIDGE_AVAILABLE:
        pytest.skip("gateway tinker bridge unavailable")
    tok = _load("Qwen/Qwen3.5-4B")
    prime = R.resolve("Qwen/Qwen3.5-4B", tok)  # prime backend
    tinker = R.resolve("Qwen/Qwen3.5-4B", tok, renderer_name="qwen3_5")  # tinker backend
    prev_prompt = prime.render_ids([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    prev_completion = tok.encode("hello", add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
    new = [{"role": "user", "content": "bye"}]
    g = prime.bridge_to_next_turn(prev_prompt, prev_completion, new)
    t = tinker.bridge_to_next_turn(prev_prompt, prev_completion, new)
    assert g is not None and t is not None
    assert list(t.token_ids) == list(g.token_ids)


@pytest.mark.skipif(not R.TINKER_AVAILABLE, reason="tinker_cookbook not installed")
def test_tinker_adapter_parity_when_forced():
    """A model in both ecosystems renders identically via the tinker adapter
    (forced by name) and HF apply_chat_template."""
    tok = _load(PRIME_MODEL)
    r = R.resolve(PRIME_MODEL, tok, renderer_name="qwen3")
    assert r.backend == "tinker"
    msgs = [{"role": "user", "content": "hi"}]
    assert r.render_ids(msgs, add_generation_prompt=True) == _hf_ids(tok, msgs)


# --------------------- tool-call bridge parity ---------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def _decode(tok, ids) -> str:
    return "".join(tok.decode([i]) for i in ids)


@pytest.mark.skipif(not R.PRIME_AVAILABLE or not R.TINKER_AVAILABLE, reason="need both backends")
def test_tinker_bridge_matches_prime_gold_with_tool_result():
    """Tool-result bridge: the tinker adapter equals prime-rl's hand-coded
    bridge byte-for-byte for Qwen3.5 (a model both ecosystems support)."""
    from rllm.renderers import _tinker

    if not _tinker.BRIDGE_AVAILABLE:
        pytest.skip("gateway tinker bridge unavailable")
    tok = _load("Qwen/Qwen3.5-4B")
    prime = R.resolve("Qwen/Qwen3.5-4B", tok)
    tinker = R.resolve("Qwen/Qwen3.5-4B", tok, renderer_name="qwen3_5")
    prev_prompt = prime.render_ids(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "weather in SF?"}],
        tools=_TOOLS,
        add_generation_prompt=True,
    )
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    prev_completion = tok.encode(
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "SF"}}\n</tool_call>',
        add_special_tokens=False,
    ) + [im_end]
    new = [{"role": "tool", "content": "sunny 72F", "tool_call_id": "c1", "name": "get_weather"}]
    g = prime.bridge_to_next_turn(prev_prompt, prev_completion, new, tools=_TOOLS)
    t = tinker.bridge_to_next_turn(prev_prompt, prev_completion, new, tools=_TOOLS)
    assert g is not None and t is not None
    assert list(t.token_ids) == list(g.token_ids)


@pytest.mark.skipif(not R.TINKER_AVAILABLE, reason="tinker_cookbook not installed")
def test_deepseek_v4_bridge_merges_tool_results():
    """DeepSeek-V4 (no prime-rl renderer): the bridge keeps prior tokens verbatim
    and merges consecutive tool results into a single user turn — matching the
    renderer's own ``build_generation_prompt`` framing."""
    from rllm.renderers import _tinker

    if not _tinker.BRIDGE_AVAILABLE:
        pytest.skip("gateway tinker bridge unavailable")
    tok = _load(FW_ONLY_MODEL)
    r = R.resolve(FW_ONLY_MODEL, tok)
    assert r.backend == "tinker" and r.has_bridge is True
    prev_prompt = r.render_ids([{"role": "user", "content": "weather?"}], add_generation_prompt=True)
    # Simulated sampled assistant turn (a tool call) ending at EOS (token id 1).
    prev_completion = tok.encode("</think>\n\nchecking", add_special_tokens=False) + [1]
    t1 = {"role": "tool", "content": "SF: sunny", "tool_call_id": "c1", "name": "get_weather"}
    t2 = {"role": "tool", "content": "NY: rainy", "tool_call_id": "c2", "name": "get_weather"}

    out = r.bridge_to_next_turn(prev_prompt, prev_completion, [t1, t2])
    assert out is not None
    prefix = prev_prompt + prev_completion
    assert out.token_ids[: len(prefix)] == prefix  # prior tokens kept verbatim
    delta = _decode(tok, out.token_ids[len(prefix):])
    # Two tool results merge into ONE user turn (not two), then the assistant opener.
    assert delta.count("<｜User｜>") == 1
    assert delta.count("<tool_result>") == 2
    assert delta.rstrip().endswith("<think>")

    # Single tool result + assistant content in the new slice is still refused.
    assert r.bridge_to_next_turn(prev_prompt, prev_completion, [t1]) is not None
    assert r.bridge_to_next_turn(prev_prompt, prev_completion, [{"role": "assistant", "content": "x"}]) is None
