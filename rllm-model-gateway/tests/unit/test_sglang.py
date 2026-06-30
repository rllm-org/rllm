"""Tests for use_sglang mode.

When use_sglang=True the gateway routes every /v1/chat/completions turn to
SGLang's native /generate API (token-in input_ids, token-out via
meta_info.output_token_logprobs), instead of the OpenAI /chat/completions
endpoints. This captures token ids + logprobs natively (no server patch) and
works through sgl-router.

Two prompt-construction modes are exercised:
  * cumulative_token_mode=True  -> turn N>1 bridges (prefix-extends) prior tokens
  * cumulative_token_mode=False -> every turn full-renders the message list

The renderer is mocked so no real tokenizer/model is loaded.
"""

import json

import openai
import pytest
from rllm_model_gateway import GatewayClient, GatewayConfig, create_app

from tests.helpers.gateway_server import GatewayServer
from tests.helpers.mock_sglang import MockSGLangServer

# The mock SGLang /generate returns canned completion ids [10, 11, 12]. The
# gateway decodes them via self.tokenizer, then parses tool calls with SGLang's
# real Qwen FunctionCallParser. So the fake tokenizer decodes those ids to a
# real Qwen tool-call string, which the real parser turns into structured calls.
_QWEN_TOOL_TEXT = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "1+1"}}\n</tool_call>'

# Tool schema the agent advertises; SGLang's FunctionCallParser only fires when
# the request carries tools (matching real agent requests).
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}},
        },
    }
]


class _MockRendered:
    """Stand-in for renderers.RenderedTokens (only .token_ids is consumed)."""

    def __init__(self, token_ids):
        self.token_ids = token_ids


class _FakeTokenizer:
    """Minimal stand-in for an HF tokenizer.

    Mirrors the SGLang render+encode split the gateway now uses:
    - apply_chat_template(tokenize=False): concatenates message content into a
      string whose length reflects the flattened text of all messages (so the
      block-content test can assert a non-empty, content-dependent prompt).
    - encode: returns ids whose COUNT equals the rendered string length, so the
      bridge/length assertions stay deterministic.
    - encode(""): empty -> _tokenizer_auto_adds_specials probes to False.
    - decode: maps the canned completion ids [10,11,12] -> a Qwen tool-call string
      so the real SGLang FunctionCallParser can extract the call.
    """

    def apply_chat_template(self, messages, *, tools=None, tokenize=False, add_generation_prompt=True):
        text = "".join(m.get("content") or "" for m in messages if isinstance(m.get("content"), str))
        rendered = text or "x"  # never empty, mirrors a real template's wrappers
        if tokenize:
            return list(range(1, len(rendered) + 1))
        return rendered

    def encode(self, text, add_special_tokens=True):
        return list(range(1, len(text) + 1))

    def decode(self, ids, skip_special_tokens=False):
        return _QWEN_TOOL_TEXT if list(ids) == [10, 11, 12] else "plain text reply"


class _MockRenderer:
    """Mimics the renderer bridge used only by the cumulative path.

    bridge_to_next_turn: prev_prompt + prev_completion + a fixed extension,
    prefix-extending by contract. (Full-render tokenization is done client-side
    via the tokenizer's apply_chat_template; tool parsing via SGLang's
    FunctionCallParser — neither uses the renderer.)
    """

    def bridge_to_next_turn(self, prev_prompt_ids, prev_completion_ids, new_messages, *, tools=None):
        if not new_messages or any(m.get("role") == "assistant" for m in new_messages):
            return None
        bridge = [200, 201]
        return _MockRendered(list(prev_prompt_ids) + list(prev_completion_ids) + bridge)


def _make_gateway(mock_sglang: MockSGLangServer, *, cumulative: bool) -> GatewayServer:
    """Gateway with use_sglang enabled, a fake tokenizer + real SGLang Qwen tool
    parser injected (create_app is called bare to avoid loading a real model).
    The worker is the mock SGLang base URL (no /v1 — /generate lives at root).
    """
    config = GatewayConfig(
        store_worker="memory",
        workers=[{"url": mock_sglang.url, "worker_id": "w0", "api_path": "/"}],
        health_check_interval=999,
        sync_traces=True,
    )
    app = create_app(config)
    p = app.state.proxy
    p.use_sglang = True
    p.cumulative_token_mode = cumulative
    p.renderer = _MockRenderer()
    p.tokenizer = _FakeTokenizer()
    # Re-probe specials now that a tokenizer is injected (create_app ran with none).
    p._tokenizer_auto_adds_specials = len(p.tokenizer.encode("")) > 0
    p.sglang_tool_call_parser = "qwen"  # use SGLang's real Qwen detector

    server = GatewayServer(app, port=0)
    server.start()
    return server


@pytest.fixture
def sglang_gateway_cumulative(mock_sglang):
    server = _make_gateway(mock_sglang, cumulative=True)
    yield server, mock_sglang
    server.stop()


@pytest.fixture
def sglang_gateway_noncumulative(mock_sglang):
    server = _make_gateway(mock_sglang, cumulative=False)
    yield server, mock_sglang
    server.stop()


def _generate_reqs(mock_sglang) -> list:
    """/generate request bodies (those carrying input_ids)."""
    return [r for r in mock_sglang.request_log if "input_ids" in r]


class TestUseSglang:
    def test_noncumulative_tokenizes_clientside_then_generates(self, sglang_gateway_noncumulative):
        """Non-cumulative: the prompt is tokenized client-side (HF chat template),
        then /generate is called with the resulting input_ids — only /generate hits
        the server (no messages→tokens server endpoint exists behind sgl-router)."""
        server, mock_sglang = sglang_gateway_noncumulative
        oai = openai.OpenAI(base_url=f"{server.url}/sessions/sg-tok/v1", api_key="dummy")
        oai.chat.completions.create(model="mock-model", messages=[{"role": "user", "content": "Hello there"}])

        gen = _generate_reqs(mock_sglang)
        assert len(gen) == 1
        # /generate used client-tokenized ids, return_logprob on, no chat fields
        assert isinstance(gen[0]["input_ids"], list) and len(gen[0]["input_ids"]) > 0
        assert gen[0].get("return_logprob") is True
        assert "messages" not in gen[0] and "prompt" not in gen[0]

    def test_noncumulative_block_content_is_tokenized_nonempty(self, sglang_gateway_noncumulative):
        """Regression: OpenAI block-format content must not produce an empty prompt.

        The fake tokenizer's apply_chat_template derives id count from flattened
        message text length, so block content that was dropped (not flattened)
        would yield a near-empty prompt."""
        server, mock_sglang = sglang_gateway_noncumulative
        oai = openai.OpenAI(base_url=f"{server.url}/sessions/sg-block/v1", api_key="dummy")
        oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": [{"type": "text", "text": "A long question here"}]}],
        )
        gen = _generate_reqs(mock_sglang)
        assert len(gen) == 1
        # text "A long question here" (20 chars) -> 20 ids; certainly not empty/tiny
        assert len(gen[0]["input_ids"]) >= len("A long question here")

    def test_noncumulative_no_accumulator(self, sglang_gateway_noncumulative):
        """Non-cumulative must not create a per-session accumulator."""
        server, mock_sglang = sglang_gateway_noncumulative
        oai = openai.OpenAI(base_url=f"{server.url}/sessions/sg-noacc/v1", api_key="dummy")
        oai.chat.completions.create(model="mock-model", messages=[{"role": "user", "content": "Hi"}])
        assert "sg-noacc" not in server.app.state.proxy._accumulators

    def test_tool_calls_parsed_to_structured(self, sglang_gateway_noncumulative):
        """The renderer-parsed completion is surfaced as structured tool_calls."""
        # Needs SGLang's real FunctionCallParser to turn <tool_call> text into a
        # structured call; sglang is an optional dep, so skip where it's absent.
        pytest.importorskip("sglang")
        server, _ = sglang_gateway_noncumulative
        oai = openai.OpenAI(base_url=f"{server.url}/sessions/sg-tool/v1", api_key="dummy")
        resp = oai.chat.completions.create(model="mock-model", messages=[{"role": "user", "content": "calc"}], tools=_TOOLS)
        choice = resp.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.tool_calls
        tc = choice.message.tool_calls[0]
        assert tc.function.name == "calculator"
        assert tc.id and tc.type == "function"
        # arguments is a JSON string (OpenAI/Strands contract)
        assert json.loads(tc.function.arguments) == {"expression": "1+1"}
        # No-discard contract: content is always present alongside tool_calls
        # (never None), mirroring SGLang native. Here the whole turn was the tool
        # call, so leftover text is the empty string — not a dropped field.
        assert choice.message.content == ""

    def test_trace_captures_native_token_ids_and_logprobs(self, sglang_gateway_noncumulative):
        """The trace has non-empty prompt+completion token ids and aligned logprobs."""
        server, _ = sglang_gateway_noncumulative
        client = GatewayClient(server.url)
        oai = openai.OpenAI(base_url=f"{server.url}/sessions/sg-trace/v1", api_key="dummy")
        oai.chat.completions.create(model="mock-model", messages=[{"role": "user", "content": "Hello"}])

        traces = client.get_session_traces("sg-trace")
        assert len(traces) == 1
        t = traces[0]
        assert len(t.prompt_token_ids) > 0
        assert t.completion_token_ids == [10, 11, 12]
        assert t.logprobs == [-0.5, -0.3, -0.1]
        client.close()

    def test_cumulative_turn0_tokenizes_turn2_bridges(self, sglang_gateway_cumulative):
        """cumulative=True: turn 0 tokenizes client-side (full chat template, NOT
        the renderer — like the vLLM path); only turn 2+ uses the renderer bridge
        to prefix-extend."""
        server, mock_sglang = sglang_gateway_cumulative
        oai = openai.OpenAI(base_url=f"{server.url}/sessions/sg-cum/v1", api_key="dummy")

        # Turn 0 — "Hello" (5 chars) -> fake apply_chat_template returns [1,2,3,4,5].
        oai.chat.completions.create(model="mock-model", messages=[{"role": "user", "content": "Hello"}])
        # Turn 1 — bridge off turn-0 (prompt [1..5] + completion [10,11,12]).
        oai.chat.completions.create(
            model="mock-model",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello from mock!"},
                {"role": "user", "content": "More"},
            ],
        )
        gen = _generate_reqs(mock_sglang)
        assert len(gen) == 2
        # turn-0 generate used the client-tokenized ids; turn-1 used the bridge:
        # turn-0 prompt [1,2,3,4,5] + completion [10,11,12] + bridge [200,201].
        assert gen[1]["input_ids"] == [1, 2, 3, 4, 5, 10, 11, 12, 200, 201]

    def test_streaming_routes_to_generate(self, sglang_gateway_noncumulative):
        """Streaming chat also goes via client tokenize + /generate and emits the
        parsed tool call as chat-format SSE."""
        # Asserts a structured tool_call in the SSE, which needs SGLang's real
        # FunctionCallParser; sglang is optional, so skip where it's absent.
        pytest.importorskip("sglang")
        server, mock_sglang = sglang_gateway_noncumulative
        oai = openai.OpenAI(base_url=f"{server.url}/sessions/sg-stream/v1", api_key="dummy")
        stream = oai.chat.completions.create(model="mock-model", messages=[{"role": "user", "content": "Hello"}], tools=_TOOLS, stream=True)
        tool_call_seen = False
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.tool_calls:
                tool_call_seen = True
        gen = _generate_reqs(mock_sglang)
        assert len(gen) == 1 and gen[0].get("stream") is True
        assert tool_call_seen
