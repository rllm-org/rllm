"""Tests for cumulative token mode.

Verifies that when cumulative_token_mode=True, the gateway:
1. Forwards turn 1 to /v1/chat/completions normally
2. Rewrites turn 2+ to /v1/completions with raw token IDs
3. Translates the response back to chat/completions format
4. Stores traces with correct prompt_token_ids and completion_token_ids
"""

import openai
import pytest
from rllm_model_gateway import GatewayClient, GatewayConfig, create_app

from tests.helpers.gateway_server import GatewayServer
from tests.helpers.mock_vllm import MockVLLMServer


class _MockRendered:
    """Stand-in for renderers.RenderedTokens (only .token_ids is consumed)."""

    def __init__(self, token_ids):
        self.token_ids = token_ids


class _MockRenderer:
    """Mimics renderers.Renderer.bridge_to_next_turn without loading a real model.

    Bridge output = prev_prompt + prev_completion + deterministic extension:
        [100, 10, 101, 1, 10] + content_ids + [100, 10, 101, 2, 10]
    where content_ids = ord() of the first 3 chars of the last user message.

    Returns None when new_messages is empty or contains an assistant turn —
    mirroring renderers' reject_assistant_in_extension contract.
    """

    def bridge_to_next_turn(self, prev_prompt_ids, prev_completion_ids, new_messages, *, tools=None):
        if not new_messages or any(m.get("role") == "assistant" for m in new_messages):
            return None
        content = ""
        for m in reversed(new_messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                break
        content_ids = [ord(c) for c in content[:3]]
        bridge = [100, 10, 101, 1, 10] + content_ids + [100, 10, 101, 2, 10]
        return _MockRendered(list(prev_prompt_ids) + list(prev_completion_ids) + bridge)


@pytest.fixture
def cumulative_gateway(mock_vllm: MockVLLMServer):
    """Gateway with cumulative_token_mode enabled using a mock renderer.

    Creates the app with cumulative_token_mode=False to avoid building a real
    renderer (which would load AutoTokenizer), then injects the mock renderer
    and enables cumulative mode on the proxy.
    """
    config = GatewayConfig(
        store_worker="memory",
        workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
        health_check_interval=999,
        sync_traces=True,
        cumulative_token_mode=False,  # Don't try to build a real renderer
    )
    app = create_app(config)
    # Inject mock renderer and enable cumulative mode
    app.state.proxy.renderer = _MockRenderer()
    app.state.proxy.cumulative_token_mode = True

    server = GatewayServer(app, port=0)
    server.start()
    yield server, mock_vllm
    server.stop()


class TestCumulativeTokenMode:
    def test_turn1_uses_chat_completions(self, cumulative_gateway):
        """First turn goes to /v1/chat/completions normally."""
        server, mock_vllm = cumulative_gateway
        gw_url = server.url

        oai = openai.OpenAI(base_url=f"{gw_url}/sessions/cum-test/v1", api_key="dummy")
        resp = oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert resp.choices[0].message.content == "Hello from mock!"

        # Verify the request went to chat/completions (has messages field)
        assert len(mock_vllm.request_log) == 1
        req = mock_vllm.request_log[0]
        assert "messages" in req

    def test_turn2_uses_completions_with_token_ids(self, cumulative_gateway):
        """Second turn rewrites to /v1/completions with raw token IDs."""
        server, mock_vllm = cumulative_gateway
        gw_url = server.url

        oai = openai.OpenAI(base_url=f"{gw_url}/sessions/cum-test2/v1", api_key="dummy")

        # Turn 1
        oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Turn 2
        resp = oai.chat.completions.create(
            model="mock-model",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello from mock!"},
                {"role": "user", "content": "How are you?"},
            ],
        )
        assert resp.choices[0].message.content == "Hello from mock!"

        # Verify second request used /v1/completions with prompt as token IDs
        assert len(mock_vllm.request_log) == 2
        second_req = mock_vllm.request_log[1]
        assert "prompt" in second_req
        assert isinstance(second_req["prompt"], list)
        assert all(isinstance(t, int) for t in second_req["prompt"])
        assert "messages" not in second_req

    def test_turn2_prompt_extends_turn1(self, cumulative_gateway):
        """Turn 2 prompt token IDs are cumulative (extend turn 1's prompt + completion)."""
        server, mock_vllm = cumulative_gateway
        gw_url = server.url

        oai = openai.OpenAI(base_url=f"{gw_url}/sessions/cum-extend/v1", api_key="dummy")

        # Turn 1
        oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Turn 2
        oai.chat.completions.create(
            model="mock-model",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello from mock!"},
                {"role": "user", "content": "More"},
            ],
        )

        # The turn 2 prompt should START with turn 1's prompt_token_ids + completion_token_ids
        second_req = mock_vllm.request_log[1]
        prompt = second_req["prompt"]
        # Turn 1 returned prompt_token_ids=[1,2,3,4,5] and completion_token_ids=[10,11,12]
        # So turn 2 prompt should start with [1,2,3,4,5,10,11,12] + bridge
        assert prompt[:8] == [1, 2, 3, 4, 5, 10, 11, 12]

    def test_traces_have_correct_token_ids(self, cumulative_gateway):
        """Both turns produce traces with prompt_token_ids and completion_token_ids."""
        server, mock_vllm = cumulative_gateway
        gw_url = server.url

        client = GatewayClient(gw_url)
        oai = openai.OpenAI(base_url=f"{gw_url}/sessions/cum-traces/v1", api_key="dummy")

        # Turn 1
        oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "Hello"}],
        )
        # Turn 2
        oai.chat.completions.create(
            model="mock-model",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello from mock!"},
                {"role": "user", "content": "Tell me more"},
            ],
        )

        traces = client.get_session_traces("cum-traces")
        assert len(traces) == 2

        # Both traces have token IDs
        for trace in traces:
            assert len(trace.prompt_token_ids) > 0
            assert len(trace.completion_token_ids) > 0

        # Turn 2 prompt should be longer (cumulative)
        assert len(traces[1].prompt_token_ids) > len(traces[0].prompt_token_ids)
        client.close()

    def test_sampling_params_forwarded(self, cumulative_gateway):
        """Sampling params from the original request are forwarded to /v1/completions."""
        server, mock_vllm = cumulative_gateway
        gw_url = server.url

        oai = openai.OpenAI(base_url=f"{gw_url}/sessions/cum-params/v1", api_key="dummy")

        # Turn 1
        oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
        )

        # Turn 2 with sampling params
        oai.chat.completions.create(
            model="mock-model",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello from mock!"},
                {"role": "user", "content": "More"},
            ],
            temperature=0.7,
            max_tokens=100,
        )

        second_req = mock_vllm.request_log[1]
        assert second_req.get("temperature") == 0.7
        assert second_req.get("max_tokens") == 100

    def test_reset_on_non_cumulative_messages(self, cumulative_gateway):
        """When message list diverges from accumulated prefix, gateway resets and uses chat path."""
        server, mock_vllm = cumulative_gateway
        gw_url = server.url

        oai = openai.OpenAI(base_url=f"{gw_url}/sessions/cum-reset/v1", api_key="dummy")

        # Turn 1 — normal chat path, seeds accumulator
        oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert len(mock_vllm.request_log) == 1
        assert "messages" in mock_vllm.request_log[0]  # chat/completions

        # Turn 2 — cumulative extension, uses /v1/completions
        oai.chat.completions.create(
            model="mock-model",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello from mock!"},
                {"role": "user", "content": "Follow up"},
            ],
        )
        assert len(mock_vllm.request_log) == 2
        assert "prompt" in mock_vllm.request_log[1]  # completions (token IDs)

        # Turn 3 — NON-CUMULATIVE: different prefix (divergent history)
        # This should trigger a reset and use the chat path
        oai.chat.completions.create(
            model="mock-model",
            messages=[
                {"role": "user", "content": "Completely different start"},
                {"role": "assistant", "content": "Different response"},
                {"role": "user", "content": "New question"},
            ],
        )
        assert len(mock_vllm.request_log) == 3
        # After reset, it goes back to chat/completions (turn-0 behavior)
        assert "messages" in mock_vllm.request_log[2]
        assert "prompt" not in mock_vllm.request_log[2]

    def test_declined_bridge_resets_and_reingests(self, cumulative_gateway):
        """When the renderer declines a cumulative (non-divergent) turn, the
        accumulator must reset so that turn is re-ingested on the chat path.

        Regression: without the reset, the stale prefix drops the declined
        turn's completion tokens from the next cumulative prompt, breaking the
        prefix-extension invariant. A bridge can return None even when the
        message prefix is cumulative (e.g. DefaultRenderer, or a slice the
        renderer can't bridge).
        """
        server, mock_vllm = cumulative_gateway
        gw_url = server.url

        class _DecliningRenderer(_MockRenderer):
            """Declines (returns None) when the new user message is 'DECLINE'."""

            def bridge_to_next_turn(self, prev_prompt_ids, prev_completion_ids, new_messages, *, tools=None):
                for m in reversed(new_messages):
                    if m.get("role") == "user":
                        if m.get("content") == "DECLINE":
                            return None
                        break
                return super().bridge_to_next_turn(prev_prompt_ids, prev_completion_ids, new_messages, tools=tools)

        server.app.state.proxy.renderer = _DecliningRenderer()
        acc_store = server.app.state.proxy._accumulators

        oai = openai.OpenAI(base_url=f"{gw_url}/sessions/cum-decline/v1", api_key="dummy")

        # Turn 1 — seeds accumulator (prompt [1,2,3,4,5] + completion [10,11,12]).
        oai.chat.completions.create(model="mock-model", messages=[{"role": "user", "content": "Hello"}])

        # Turn 2 — cumulative prefix, but the renderer declines the bridge.
        oai.chat.completions.create(
            model="mock-model",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello from mock!"},
                {"role": "user", "content": "DECLINE"},
            ],
        )
        # Declined -> chat path (not /v1/completions).
        assert "messages" in mock_vllm.request_log[1]
        assert "prompt" not in mock_vllm.request_log[1]
        # Reset + re-ingest: prefix snapshot now reflects turn-2's 3 messages,
        # not turn-1's single message (the bug leaves this at 1).
        assert acc_store["cum-decline"].message_count == 3

        # Turn 3 — cumulative extension resumes from the re-ingested turn-2 state.
        oai.chat.completions.create(
            model="mock-model",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello from mock!"},
                {"role": "user", "content": "DECLINE"},
                {"role": "assistant", "content": "Hello from mock!"},
                {"role": "user", "content": "More"},
            ],
        )
        assert "prompt" in mock_vllm.request_log[2]
        # Prompt extends the re-ingested turn-2 full sequence [1,2,3,4,5,10,11,12].
        assert mock_vllm.request_log[2]["prompt"][:8] == [1, 2, 3, 4, 5, 10, 11, 12]

    def test_reset_then_resume_cumulative(self, cumulative_gateway):
        """After a reset, the next cumulative extension works normally again."""
        server, mock_vllm = cumulative_gateway
        gw_url = server.url

        oai = openai.OpenAI(base_url=f"{gw_url}/sessions/cum-resume/v1", api_key="dummy")

        # Turn 1 — seeds accumulator
        oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Turn 2 — divergent (reset)
        oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "Fresh start"}],
        )
        # is_cumulative returns False when len(messages) <= message_count, triggering reset
        assert "messages" in mock_vllm.request_log[1]  # went through chat path

        # Turn 3 — cumulative extension of the new history
        oai.chat.completions.create(
            model="mock-model",
            messages=[
                {"role": "user", "content": "Fresh start"},
                {"role": "assistant", "content": "Hello from mock!"},
                {"role": "user", "content": "Continue from fresh"},
            ],
        )
        # Should now use cumulative path again (turn-1 was re-seeded after reset)
        assert "prompt" in mock_vllm.request_log[2]
        assert isinstance(mock_vllm.request_log[2]["prompt"], list)


class TestCumulativeStreaming:
    """Streaming-specific tests for the cumulative path (turn 2+ rewritten
    to /v1/completions streaming).

    Only covers behavior that is unique to the cumulative-streaming path.
    Generic streaming behavior (vLLM field stripping, content delivery,
    trace capture mechanics) is shared with _handle_streaming and covered
    by TestStreamingProxy in test_server.py.
    """

    def test_turn2_stream_uses_completions_with_token_ids(self, cumulative_gateway):
        """Streaming second turn rewrites to /v1/completions with raw token IDs
        and translates chunks back to chat-format for the client."""
        server, mock_vllm = cumulative_gateway
        gw_url = server.url
        oai = openai.OpenAI(base_url=f"{gw_url}/sessions/cum-stream-1/v1", api_key="dummy")

        # Turn 1 — seed the accumulator.
        oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Turn 2 — streaming.
        stream = oai.chat.completions.create(
            model="mock-model",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello from mock!"},
                {"role": "user", "content": "How are you?"},
            ],
            stream=True,
        )
        content_parts = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content_parts.append(chunk.choices[0].delta.content)

        # Client received chat-format content end-to-end.
        assert "".join(content_parts) == "Hello from mock!"

        # Upstream got a /v1/completions streaming request with token-IDs prompt.
        assert len(mock_vllm.request_log) == 2
        second_req = mock_vllm.request_log[1]
        assert "prompt" in second_req
        assert isinstance(second_req["prompt"], list)
        assert all(isinstance(t, int) for t in second_req["prompt"])
        assert "messages" not in second_req
        assert second_req.get("stream") is True

    def test_streaming_forwards_usage(self, cumulative_gateway):
        """Final usage-only chunk from /v1/completions should reach the client.

        Regression test: cumulative streaming used to drop chunks where
        choices was empty, dropping vLLM's trailing usage-only chunk.
        """
        import httpx as _httpx

        server, _ = cumulative_gateway
        gw_url = server.url
        oai = openai.OpenAI(base_url=f"{gw_url}/sessions/cum-stream-usage/v1", api_key="dummy")

        # Turn 1 — seed.
        oai.chat.completions.create(
            model="mock-model",
            messages=[{"role": "user", "content": "Hello"}],
        )

        # Turn 2 — read SSE directly so we can inspect the usage chunk.
        with _httpx.Client(timeout=10.0) as c:
            resp = c.post(
                f"{gw_url}/sessions/cum-stream-usage/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hello from mock!"},
                        {"role": "user", "content": "More"},
                    ],
                    "stream": True,
                },
            )
            import json as _json

            chunks = [_json.loads(line[6:]) for line in resp.text.strip().split("\n") if line.startswith("data: ") and line.strip() != "data: [DONE]"]

        usage_chunks = [c for c in chunks if c.get("usage")]
        assert len(usage_chunks) >= 1
        usage = usage_chunks[-1]["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage


# ---------------------------------------------------------------------------
# Structured cumulative response (option 3): parse the completion via the
# renderer so cumulative turns match the chat-path shape (clean content +
# reasoning_content + structured tool_calls) instead of dumping raw text.
# ---------------------------------------------------------------------------


class _ParsedResp:
    def __init__(self, content="", reasoning_content="", tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls or []


class _ToolCall:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _MockRendererWithParse(_MockRenderer):
    """Adds parse_response — returns a fixed structured response distinct from
    the mock worker's raw text, so a test can prove parse_response is used."""

    def parse_response(self, token_ids):
        return _ParsedResp(
            content="parsed answer",
            reasoning_content="my thinking",
            tool_calls=[_ToolCall("run_cmd", {"cmd": "ls"})],
        )


class TestStructuredCumulativeResponse:
    def test_pure_message_builder(self):
        from rllm_model_gateway.proxy import _parsed_response_to_message

        msg = _parsed_response_to_message(_ParsedResp("hello", "thinking", [_ToolCall("f", {"a": 1})]))
        assert msg["role"] == "assistant"
        assert msg["content"] == "hello"
        assert msg["reasoning_content"] == "thinking"
        assert msg["tool_calls"][0]["type"] == "function"
        assert msg["tool_calls"][0]["function"]["name"] == "f"
        import json as _json

        assert _json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"a": 1}

    def test_pure_message_builder_omits_empty_fields(self):
        from rllm_model_gateway.proxy import _parsed_response_to_message

        msg = _parsed_response_to_message(_ParsedResp("just text"))
        assert msg == {"role": "assistant", "content": "just text"}  # no reasoning/tool_calls keys

    def test_tool_call_function_shape(self):
        from rllm_model_gateway.proxy import _tool_call_to_openai

        class _FnTc:
            class function:  # noqa: N801
                name = "g"
                arguments = '{"x": 2}'  # already a string

        out = _tool_call_to_openai(_FnTc(), 3)
        assert out["id"] == "call_3"
        assert out["function"] == {"name": "g", "arguments": '{"x": 2}'}

    def test_cumulative_turn_returns_parsed_message(self, mock_vllm):
        """End-to-end: a cumulative (token-in) turn returns the renderer-parsed
        message shape, not the worker's raw completion text."""
        config = GatewayConfig(
            store_worker="memory",
            workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
            health_check_interval=999,
            sync_traces=True,
            cumulative_token_mode=False,
        )
        app = create_app(config)
        app.state.proxy.renderer = _MockRendererWithParse()
        app.state.proxy.cumulative_token_mode = True
        server = GatewayServer(app, port=0)
        server.start()
        try:
            oai = openai.OpenAI(base_url=f"{server.url}/sessions/parse-test/v1", api_key="dummy")
            oai.chat.completions.create(model="m", messages=[{"role": "user", "content": "Hi"}])  # turn 1
            resp = oai.chat.completions.create(  # turn 2 — cumulative
                model="m",
                messages=[
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello from mock!"},
                    {"role": "user", "content": "go"},
                ],
            )
        finally:
            server.stop()
        msg = resp.choices[0].message
        assert msg.content == "parsed answer"  # parsed, not the raw "Hello from mock!"
        assert msg.tool_calls[0].function.name == "run_cmd"
        import json as _json

        assert _json.loads(msg.tool_calls[0].function.arguments) == {"cmd": "ls"}


class _MockRendererFull(_MockRendererWithParse):
    """A complete renderer: bridge + parse_response + render_ids.

    ``render_ids`` is what turn 0 uses, so the WHOLE trajectory is tokenized by
    one renderer (no chat-template-vs-renderer seam). Deterministic encoding:
    ``[200, 10, 201] + ord(last user content[:3]) + [202]``.
    """

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        content = ""
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content", "")
        return [200, 10, 201] + [ord(c) for c in content[:3]] + [202]


@pytest.fixture
def cumulative_gateway_full(mock_vllm: MockVLLMServer):
    """Cumulative gateway whose renderer implements render_ids (turn 0 included)."""
    config = GatewayConfig(
        store_worker="memory",
        workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
        health_check_interval=999,
        sync_traces=True,
        cumulative_token_mode=False,
    )
    app = create_app(config)
    app.state.proxy.renderer = _MockRendererFull()
    app.state.proxy.cumulative_token_mode = True
    server = GatewayServer(app, port=0)
    server.start()
    yield server, mock_vllm
    server.stop()


class TestCumulativeTurn0ViaRenderer:
    """Turn 0 is rendered by the renderer (render_ids) and sent token-in, exactly
    like turns 2+, so one renderer owns the entire trajectory."""

    def test_turn0_uses_completions_with_token_ids(self, cumulative_gateway_full):
        server, mock_vllm = cumulative_gateway_full
        oai = openai.OpenAI(base_url=f"{server.url}/sessions/t0-completions/v1", api_key="dummy")
        oai.chat.completions.create(model="m", messages=[{"role": "user", "content": "Hello"}])

        assert len(mock_vllm.request_log) == 1
        req = mock_vllm.request_log[0]
        # Turn 0 went to /v1/completions (token-in), NOT /chat/completions.
        assert "messages" not in req
        assert isinstance(req["prompt"], list) and all(isinstance(t, int) for t in req["prompt"])

    def test_turn0_prompt_equals_render_ids(self, cumulative_gateway_full):
        server, mock_vllm = cumulative_gateway_full
        oai = openai.OpenAI(base_url=f"{server.url}/sessions/t0-render/v1", api_key="dummy")
        oai.chat.completions.create(model="m", messages=[{"role": "user", "content": "Hello"}])

        expected = _MockRendererFull().render_ids([{"role": "user", "content": "Hello"}])
        assert mock_vllm.request_log[0]["prompt"] == expected

    def test_turn0_returns_parsed_message(self, cumulative_gateway_full):
        """Turn 0's completion is parsed via the renderer too (not raw worker text)."""
        server, _ = cumulative_gateway_full
        oai = openai.OpenAI(base_url=f"{server.url}/sessions/t0-parse/v1", api_key="dummy")
        resp = oai.chat.completions.create(model="m", messages=[{"role": "user", "content": "Hello"}])
        assert resp.choices[0].message.content == "parsed answer"  # parsed, not "Hello from mock!"

    def test_single_renderer_trajectory_no_seam(self, cumulative_gateway_full):
        """The seam fix: turn 1's prompt extends turn 0's render_ids output +
        completion byte-for-byte — both produced by the same renderer, so there is
        no chat-template/renderer boundary mismatch."""
        server, mock_vllm = cumulative_gateway_full
        oai = openai.OpenAI(base_url=f"{server.url}/sessions/t0-seam/v1", api_key="dummy")

        oai.chat.completions.create(model="m", messages=[{"role": "user", "content": "Hello"}])  # turn 0
        oai.chat.completions.create(  # turn 1
            model="m",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hello from mock!"},
                {"role": "user", "content": "More"},
            ],
        )

        turn0_prompt = _MockRendererFull().render_ids([{"role": "user", "content": "Hello"}])
        # mock_vllm echoes the prompt as prompt_token_ids and returns completion [10,11,12].
        turn1_prompt = mock_vllm.request_log[1]["prompt"]
        assert turn1_prompt[: len(turn0_prompt) + 3] == turn0_prompt + [10, 11, 12]
