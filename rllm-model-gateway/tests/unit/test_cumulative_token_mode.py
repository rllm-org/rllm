"""Tests for cumulative token mode.

Verifies that when cumulative_token_mode=True, the gateway:
1. Forwards turn 1 to /v1/chat/completions normally
2. Rewrites turn 2+ to /v1/completions with raw token IDs
3. Translates the response back to chat/completions format
4. Stores traces with correct prompt_token_ids and completion_token_ids
"""

from unittest.mock import MagicMock

import openai
import pytest
from rllm_model_gateway import GatewayClient, GatewayConfig, create_app

from tests.helpers.gateway_server import GatewayServer
from tests.helpers.mock_vllm import MockVLLMServer


def _make_test_tokenizer():
    """Create a mock tokenizer for testing cumulative mode without loading a real model.

    Uses the same scheme as test_token_accumulator.py:
    - Special tokens: <|im_end|>=100, <|im_start|>=101, newline=10
    - apply_chat_template with continue_final_message=True returns [101, 1, 10, 101, 2, 10, 50, 51]
    - apply_chat_template with add_generation_prompt=True returns prefix + bridge + content
    """
    tokenizer = MagicMock()

    def _apply_chat_template(conversation, tokenize=False, **kwargs):
        if kwargs.get("continue_final_message"):
            return [101, 1, 10, 101, 2, 10, 50, 51]

        if kwargs.get("add_generation_prompt"):
            user_content = conversation[-1]["content"]
            prefix = [101, 1, 10, 101, 2, 10, 50, 51]
            content_ids = [ord(c) for c in user_content[:3]]
            bridge = [100, 10, 101, 1, 10] + content_ids + [100, 10, 101, 2, 10]
            return prefix + bridge

        return []

    tokenizer.apply_chat_template = _apply_chat_template
    return tokenizer


@pytest.fixture
def cumulative_gateway(mock_vllm: MockVLLMServer):
    """Gateway with cumulative_token_mode enabled using a mock tokenizer.

    Creates the app with cumulative_token_mode=False to avoid AutoTokenizer.from_pretrained,
    then manually injects the mock tokenizer and enables cumulative mode on the proxy.
    """
    config = GatewayConfig(
        store_worker="memory",
        workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
        health_check_interval=999,
        sync_traces=True,
        cumulative_token_mode=False,  # Don't try to load real tokenizer
    )
    app = create_app(config)
    # Inject mock tokenizer and enable cumulative mode
    mock_tokenizer = _make_test_tokenizer()
    app.state.proxy.tokenizer = mock_tokenizer
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
