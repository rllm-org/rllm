"""Integration test: real TinkerEngine → engine adapter → gateway → trace round-trip.

Requires TINKER_API_KEY env var (auto-skipped otherwise).
Uses a small model with short max_tokens to keep costs low.
"""

from __future__ import annotations

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
        chat_template_kwargs={"disable_thinking": True},
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
    """Start a gateway with a real TinkerEngine wrapped via the engine adapter."""
    from rllm_model_gateway import GatewayConfig, create_app
    from rllm_model_gateway.store.memory_store import MemoryTraceStore

    from rllm.experimental.engine.engine_adapter import create_engine_adapter

    engine = create_tinker_engine()
    adapter = create_engine_adapter(engine)
    config = GatewayConfig(
        admin_api_key="test-admin",
        agent_api_key="test-agent",
    )
    app = create_app(config, adapter=adapter, store=MemoryTraceStore())
    server = GatewayServer(app)
    server.start()
    yield server
    server.stop()


def _make_client(server):
    from rllm_model_gateway import GatewayClient

    return GatewayClient(server.url, api_key="test-admin")


def _make_openai(server, sid: str):
    import openai

    return openai.OpenAI(base_url=f"{server.url}/sessions/{sid}/v1", api_key="test-agent")


@requires_tinker
class TestTinkerAdapterE2E:
    def test_basic_completion(self, tinker_gateway):
        gw = _make_client(tinker_gateway)
        sid = gw.create_session(session_id="tinker-basic")
        oai = _make_openai(tinker_gateway, sid)

        resp = oai.chat.completions.create(
            model=TINKER_MODEL_NAME,
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            max_tokens=32,
        )

        assert resp.choices[0].message.content is not None
        assert len(resp.choices[0].message.content) > 0
        assert resp.choices[0].finish_reason in ("stop", "length", "tool_calls")
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0
        gw.close()

    def test_trace_has_token_data(self, tinker_gateway):
        from rllm_model_gateway import deserialize_extras

        gw = _make_client(tinker_gateway)
        sid = gw.create_session(session_id="tinker-trace")
        oai = _make_openai(tinker_gateway, sid)

        oai.chat.completions.create(
            model=TINKER_MODEL_NAME,
            messages=[{"role": "user", "content": "What is 1+1?"}],
            max_tokens=32,
        )

        gw.flush()
        traces = gw.get_session_traces(sid)
        assert len(traces) == 1
        t = traces[0]

        # Top-level structured response fields.
        assert t.content is not None and len(t.content) > 0
        assert t.finish_reason in ("stop", "length", "tool_calls")
        assert "gateway_latency_ms" in t.metrics

        # Token-level data lives in the extras blob.
        extras_blob = gw.get_trace_extras(t.trace_id)
        assert extras_blob is not None
        fmt, data = extras_blob
        extras = deserialize_extras(fmt, data)
        assert len(extras["prompt_ids"]) > 0
        assert len(extras["completion_ids"]) > 0
        assert len(extras["logprobs"]) == len(extras["completion_ids"])
        gw.close()

    def test_trace_round_trips_to_step(self, tinker_gateway):
        from rllm_model_gateway import deserialize_extras

        from rllm.experimental.engine.trace_converter import trace_record_to_step

        gw = _make_client(tinker_gateway)
        sid = gw.create_session(session_id="tinker-step")
        oai = _make_openai(tinker_gateway, sid)

        oai.chat.completions.create(
            model=TINKER_MODEL_NAME,
            messages=[{"role": "user", "content": "What is 3+4?"}],
            max_tokens=32,
        )

        gw.flush()
        traces = gw.get_session_traces(sid)
        trace = traces[0]
        extras_blob = gw.get_trace_extras(trace.trace_id)
        extras = deserialize_extras(*extras_blob) if extras_blob else {}
        step = trace_record_to_step(trace, extras)

        assert len(step.model_output.prompt_ids) > 0
        assert len(step.model_output.completion_ids) > 0
        assert len(step.model_output.logprobs) > 0
        assert len(step.model_output.content) > 0
        assert step.model_output.finish_reason in ("stop", "length", "tool_calls")
        gw.close()

    def test_streaming_delivers_content_and_trace(self, tinker_gateway):
        from rllm_model_gateway import deserialize_extras

        gw = _make_client(tinker_gateway)
        sid = gw.create_session(session_id="tinker-stream")
        oai = _make_openai(tinker_gateway, sid)

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

        gw.flush()
        traces = gw.get_session_traces(sid)
        assert len(traces) == 1
        extras = deserialize_extras(*gw.get_trace_extras(traces[0].trace_id))
        assert len(extras["prompt_ids"]) > 0
        assert len(extras["completion_ids"]) > 0
        gw.close()

    def test_multi_turn(self, tinker_gateway):
        from rllm_model_gateway import deserialize_extras

        gw = _make_client(tinker_gateway)
        sid = gw.create_session(session_id="tinker-multi")
        oai = _make_openai(tinker_gateway, sid)

        for q in ["What is 1+1?", "What is 2+2?"]:
            oai.chat.completions.create(
                model=TINKER_MODEL_NAME,
                messages=[{"role": "user", "content": q}],
                max_tokens=16,
            )

        gw.flush()
        traces = gw.get_session_traces(sid)
        assert len(traces) == 2
        for t in traces:
            extras = deserialize_extras(*gw.get_trace_extras(t.trace_id))
            assert len(extras["prompt_ids"]) > 0
            assert len(extras["completion_ids"]) > 0
        gw.close()

    def test_tool_calling_multi_turn(self, tinker_gateway):
        """Send tools, get tool_calls back, send tool result, get final answer."""
        from rllm_model_gateway import deserialize_extras

        from rllm.experimental.engine.trace_converter import trace_record_to_step

        gw = _make_client(tinker_gateway)
        sid = gw.create_session(session_id="tinker-tools")
        oai = _make_openai(tinker_gateway, sid)

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

        # Turn 1: request with tools
        resp = oai.chat.completions.create(
            model=TINKER_MODEL_NAME,
            messages=[{"role": "user", "content": "What is 17 * 23? Use the calculator tool."}],
            tools=tools,
            max_tokens=256,
        )

        msg = resp.choices[0].message
        gw.flush()
        traces = gw.get_session_traces(sid)
        assert len(traces) >= 1

        t0 = traces[0]
        extras0 = deserialize_extras(*gw.get_trace_extras(t0.trace_id))
        assert len(extras0["prompt_ids"]) > 0
        assert len(extras0["completion_ids"]) > 0
        assert len(extras0["logprobs"]) > 0

        # Model chose to use tools — verify the round-trip.
        assert resp.choices[0].finish_reason == "tool_calls"
        tc = msg.tool_calls[0]
        assert tc.function.name == "calculator"
        args = json.loads(tc.function.arguments)
        assert "expression" in args

        # tool_calls are top-level on the new TraceRecord.
        assert len(t0.tool_calls) >= 1
        assert t0.tool_calls[0].name == "calculator"

        # trace_record_to_step should reconstruct tool_calls.
        step = trace_record_to_step(t0, extras0)
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
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                },
                {"role": "tool", "tool_call_id": tc.id, "content": result},
            ],
            tools=tools,
            max_tokens=128,
        )
        assert resp2.choices[0].message.content is not None

        gw.flush()
        traces_after = gw.get_session_traces(sid)
        assert len(traces_after) == 2
        t1 = traces_after[1]
        extras1 = deserialize_extras(*gw.get_trace_extras(t1.trace_id))
        assert len(extras1["prompt_ids"]) > 0
        assert len(extras1["completion_ids"]) > 0

        gw.close()
