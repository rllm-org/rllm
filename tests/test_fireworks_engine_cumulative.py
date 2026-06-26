"""FireworksEngine cumulative-token-mode (pure token-in/token-out) behavior.

In cumulative mode the gateway renders every turn and parses every completion
via the renderer, so the engine must be a pure sampler: it builds NO
ChatTemplateParser (which would crash in __init__ for models whose Jinja
template the parser's equivalence self-check can't render, e.g. GLM) and
assemble_model_output returns raw-decoded text instead of parsing.
"""

import pytest

pytest.importorskip("fireworks", reason="Fireworks SDK not installed")

from rllm.engine.rollout.fireworks_engine import FireworksEngine
from rllm.engine.rollout.tinker_engine import TinkerEngine


class _BoomTokenizer:
    """Any attribute access raises — proves the engine never consults the
    tokenizer for parser/renderer setup in cumulative mode (the GLM crash path)."""

    name_or_path = "zai-org/GLM-5.1"

    def __getattr__(self, key):
        raise RuntimeError(f"tokenizer.{key} must not be touched in cumulative mode")


class _DecodeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "DECODED:" + ",".join(map(str, ids))


class _DummySampler:
    pass


class _Seq:
    tokens = [10, 11, 12]
    logprobs = [0.0, 0.0, 0.0]
    stop_reason = "stop"


def _make_engine():
    return FireworksEngine(
        tokenizer=_BoomTokenizer(),
        sampler=_DummySampler(),
        max_prompt_length=100,
        max_response_length=50,
        max_model_length=200,
        cumulative_token_mode=True,
    )


def test_cumulative_construct_skips_chat_template_parser():
    eng = _make_engine()
    assert eng.chat_parser is None
    assert eng.renderer is None
    assert eng.raw_token_mode is True
    assert eng.bypass_render_with_parser is False


def test_raw_assemble_decodes_without_parsing():
    eng = _make_engine()
    eng.tokenizer = _DecodeTokenizer()
    out = TinkerEngine.assemble_model_output(eng, [1, 2, 3], _Seq())
    assert out.content == "DECODED:10,11,12"
    assert out.reasoning == ""
    assert out.tool_calls == []
    assert out.completion_ids == [10, 11, 12]
    assert out.prompt_ids == [1, 2, 3]


def test_get_model_response_rejected_in_cumulative_mode():
    """The chat (message-rendering) path must not be reachable in cumulative mode."""
    import asyncio

    eng = _make_engine()
    with pytest.raises(RuntimeError, match="cumulative_token_mode"):
        asyncio.run(eng.get_model_response([{"role": "user", "content": "hi"}]))
