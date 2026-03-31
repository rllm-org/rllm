"""Integration test: real TinkerEngine → create_tinker_handler → gateway → trace → trace_record_to_step.

Requires TINKER_API_KEY env var (auto-skipped otherwise).
Uses a small model (Qwen/Qwen3-8B) with short max_tokens to keep costs low.
"""

import json
import threading
import time

import pytest
import uvicorn

from .conftest import TINKER_MODEL_NAME, requires_tinker


def create_tinker_engine():
    """Bootstrap a real TinkerEngine with a sampling client."""
    import tinker
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    from rllm.experimental.rollout.tinker_engine import TinkerEngine

    tokenizer = get_tokenizer(TINKER_MODEL_NAME)
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=TINKER_MODEL_NAME)

    engine = TinkerEngine(
        base_url="",
        model_name=TINKER_MODEL_NAME,
        tokenizer=tokenizer,
        service_client=service_client,
        max_prompt_length=1024,
        max_response_length=128,
        max_model_length=2048,
        sampling_params={"train": {"temperature": 0.0}, "val": {"temperature": 0.0}},
        bypass_render_with_parser=True,
        disable_thinking=True,
    )
    engine.set_sampling_client(sampling_client)
    return engine


class GatewayServer:
    def __init__(self, app, port=0):
        self.host = "127.0.0.1"
        self.port = port
        self.app = app
        self.server = None
        self.thread = None

    @property
    def url(self):
        return f"http://{self.host}:{self.port}"

    def start(self):
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="error")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self.server.started:
                for sock in self.server.servers:
                    self.port = sock.sockets[0].getsockname()[1]
                return
            time.sleep(0.05)
        raise RuntimeError("Server failed to start")

    def stop(self):
        if self.server:
            self.server.should_exit = True
        if self.thread:
            self.thread.join(timeout=5.0)


@pytest.fixture(scope="module")
def tinker_gateway():
    """Start a gateway with a real TinkerEngine behind create_tinker_handler."""
    from rllm_model_gateway import GatewayConfig, create_app

    from rllm.experimental.engine.tinker_adapter import create_tinker_handler

    engine = create_tinker_engine()
    handler = create_tinker_handler(engine)
    config = GatewayConfig(
        store_worker="memory",
        workers=[],
        health_check_interval=999,
        sync_traces=True,
    )
    app = create_app(config, local_handler=handler)
    server = GatewayServer(app)
    server.start()
    yield server
    server.stop()


@requires_tinker
class TestTinkerAdapterE2E:
    def test_basic_completion(self, tinker_gateway):
        """Real tinker inference through gateway, verify clean response."""
        import openai
        from rllm_model_gateway import GatewayClient

        gw = GatewayClient(tinker_gateway.url)
        sid = gw.create_session(session_id="tinker-basic")
        url = gw.get_session_url(sid)

        oai = openai.OpenAI(base_url=url, api_key="dummy")
        resp = oai.chat.completions.create(
            model=TINKER_MODEL_NAME,
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            max_tokens=32,
        )

        assert resp.choices[0].message.content is not None
        assert len(resp.choices[0].message.content) > 0
        assert resp.choices[0].finish_reason in ("stop", "length")
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0
        gw.close()

    def test_vllm_fields_stripped(self, tinker_gateway):
        """Raw response must not leak token_ids or prompt_token_ids."""
        import httpx
        from rllm_model_gateway import GatewayClient

        gw = GatewayClient(tinker_gateway.url)
        sid = gw.create_session(session_id="tinker-strip")
        url = gw.get_session_url(sid)

        raw = httpx.post(
            f"{url}/chat/completions",
            json={
                "model": TINKER_MODEL_NAME,
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 16,
            },
        )
        data = raw.json()
        assert "prompt_token_ids" not in data
        for choice in data.get("choices", []):
            assert "token_ids" not in choice
        gw.close()

    def test_trace_has_token_data(self, tinker_gateway):
        """Trace should contain real prompt_token_ids, completion_token_ids, logprobs."""
        import openai
        from rllm_model_gateway import GatewayClient

        gw = GatewayClient(tinker_gateway.url)
        sid = gw.create_session(session_id="tinker-trace")
        url = gw.get_session_url(sid)

        oai = openai.OpenAI(base_url=url, api_key="dummy")
        oai.chat.completions.create(
            model=TINKER_MODEL_NAME,
            messages=[{"role": "user", "content": "What is 1+1?"}],
            max_tokens=32,
        )

        traces = gw.get_session_traces(sid)
        assert len(traces) == 1
        t = traces[0]

        assert len(t.prompt_token_ids) > 0, "trace should have real prompt token IDs"
        assert len(t.completion_token_ids) > 0, "trace should have real completion token IDs"
        assert t.logprobs is not None and len(t.logprobs) > 0, "trace should have logprobs"
        assert len(t.logprobs) == len(t.completion_token_ids), "logprobs count should match completion tokens"
        assert t.response_message["role"] == "assistant"
        assert len(t.response_message.get("content", "")) > 0
        assert t.finish_reason in ("stop", "length")
        assert t.latency_ms > 0
        gw.close()

    def test_trace_round_trips_to_step(self, tinker_gateway):
        """trace_record_to_step should reconstruct a valid ModelOutput from real trace."""
        import openai
        from rllm_model_gateway import GatewayClient

        from rllm.experimental.engine.trace_converter import trace_record_to_step

        gw = GatewayClient(tinker_gateway.url)
        sid = gw.create_session(session_id="tinker-step")
        url = gw.get_session_url(sid)

        oai = openai.OpenAI(base_url=url, api_key="dummy")
        oai.chat.completions.create(
            model=TINKER_MODEL_NAME,
            messages=[{"role": "user", "content": "What is 3+4?"}],
            max_tokens=32,
        )

        traces = gw.get_session_traces(sid)
        step = trace_record_to_step(traces[0])

        assert step.model_output.prompt_ids is not None
        assert len(step.model_output.prompt_ids) > 0
        assert step.model_output.completion_ids is not None
        assert len(step.model_output.completion_ids) > 0
        assert step.model_output.logprobs is not None
        assert len(step.model_output.logprobs) > 0
        assert step.model_output.content is not None
        assert len(step.model_output.content) > 0
        assert step.model_output.finish_reason in ("stop", "length")
        assert step.model_output.prompt_ids == traces[0].prompt_token_ids
        assert step.model_output.completion_ids == traces[0].completion_token_ids
        gw.close()

    def test_streaming_delivers_content_and_trace(self, tinker_gateway):
        """Streaming through the gateway should deliver real content and capture trace."""
        import openai
        from rllm_model_gateway import GatewayClient

        gw = GatewayClient(tinker_gateway.url)
        sid = gw.create_session(session_id="tinker-stream")
        url = gw.get_session_url(sid)

        oai = openai.OpenAI(base_url=url, api_key="dummy")
        stream = oai.chat.completions.create(
            model=TINKER_MODEL_NAME,
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=16,
            stream=True,
        )

        parts = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
        full = "".join(parts)
        assert len(full) > 0

        traces = gw.get_session_traces(sid)
        assert len(traces) == 1
        assert len(traces[0].prompt_token_ids) > 0
        assert len(traces[0].completion_token_ids) > 0
        gw.close()

    def test_multi_turn(self, tinker_gateway):
        """Multiple calls under one session should produce separate traces."""
        import openai
        from rllm_model_gateway import GatewayClient

        gw = GatewayClient(tinker_gateway.url)
        sid = gw.create_session(session_id="tinker-multi")
        url = gw.get_session_url(sid)

        oai = openai.OpenAI(base_url=url, api_key="dummy")
        for q in ["What is 1+1?", "What is 2+2?"]:
            oai.chat.completions.create(
                model=TINKER_MODEL_NAME,
                messages=[{"role": "user", "content": q}],
                max_tokens=16,
            )

        traces = gw.get_session_traces(sid)
        assert len(traces) == 2
        for t in traces:
            assert len(t.prompt_token_ids) > 0
            assert len(t.completion_token_ids) > 0
        gw.close()

    def test_tool_calling_multi_turn(self, tinker_gateway):
        """Send tools, get tool_calls back, send tool result, get final answer.

        Simulates what an agent with tool does:
        1. Request with tools defined → model responds with tool_calls
        2. Send tool result back → model responds with final answer
        Verifies tools flow through the prompt and tool_calls appear in traces.
        """
        import openai
        from rllm_model_gateway import GatewayClient

        from rllm.experimental.engine.trace_converter import trace_record_to_step

        gw = GatewayClient(tinker_gateway.url)
        sid = gw.create_session(session_id="tinker-tools")
        url = gw.get_session_url(sid)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a math expression and return the result.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The math expression to evaluate",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            }
        ]

        oai = openai.OpenAI(base_url=url, api_key="dummy")

        # Turn 1: request with tools
        resp = oai.chat.completions.create(
            model=TINKER_MODEL_NAME,
            messages=[{"role": "user", "content": "What is 17 * 23? Use the calculator tool."}],
            tools=tools,
            max_tokens=256,
        )

        msg = resp.choices[0].message
        traces = gw.get_session_traces(sid)
        assert len(traces) >= 1

        t0 = traces[0]
        assert len(t0.prompt_token_ids) > 0
        assert len(t0.completion_token_ids) > 0
        assert t0.logprobs is not None and len(t0.logprobs) > 0

        # Model chose to use tools — verify the full round-trip
        assert resp.choices[0].finish_reason == "tool_calls"
        tc = msg.tool_calls[0]
        assert tc.function.name == "calculator"
        args = json.loads(tc.function.arguments)
        assert "expression" in args

        # Verify tool_calls in trace
        assert "tool_calls" in t0.response_message
        assert t0.response_message["tool_calls"][0]["function"]["name"] == "calculator"

        # Verify trace_record_to_step reconstructs tool_calls
        step = trace_record_to_step(t0)
        assert step.model_output.tool_calls is not None
        assert step.model_output.tool_calls[0].name == "calculator"

        # Turn 2: send tool result back
        result = str(eval(args["expression"]))
        resp2 = oai.chat.completions.create(
            model=TINKER_MODEL_NAME,
            messages=[
                {"role": "user", "content": "What is 17 * 23? Use the calculator tool."},
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls],
                },
                {"role": "tool", "tool_call_id": tc.id, "content": result},
            ],
            tools=tools,
            max_tokens=128,
        )
        assert resp2.choices[0].message.content is not None

        traces_after = gw.get_session_traces(sid)
        assert len(traces_after) == 2
        t1 = traces_after[1]
        assert len(t1.prompt_token_ids) > 0
        assert len(t1.completion_token_ids) > 0

        gw.close()
