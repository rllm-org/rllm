"""Context-limited Fireworks completions retain their cause through the gateway."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from rllm_model_gateway.data_process import build_trace_record, strip_vllm_fields
from rllm_model_gateway.models import TraceGraph

from rllm.engine.agentflow_engine import enrich_episode_with_traces
from rllm.engine.rollout.fireworks_engine import FireworksEngine
from rllm.gateway.tinker_adapter import create_tinker_handler
from rllm.types import Episode, TerminationReason, Trajectory


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("token_path", [False, True])
@pytest.mark.parametrize(
    ("prompt_length", "finish_reason", "expected"),
    [
        (45, "length", TerminationReason.MAX_PROMPT_LENGTH_EXCEEDED),
        (5, "length", TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED),
        (45, "stop", TerminationReason.ENV_DONE),
    ],
)
def test_fireworks_context_limit_survives_adapter_and_trace(compact, token_path, prompt_length, finish_reason, expected):
    # Exercise the real Fireworks budget calculation without network/model loading.
    engine = FireworksEngine.__new__(FireworksEngine)
    engine.sampling_client = object()
    engine.weight_version = 0
    engine.max_prompt_length = 50
    engine.max_model_length = 49
    engine.max_response_length = 8
    engine.is_validation = False
    engine.train_sampling_params = {}
    engine.val_sampling_params = {}
    engine.reasoning_effort = None
    engine.router_replay = False
    engine.unified_renderer = None
    engine.bypass_render_with_parser = True
    engine.chat_parser = SimpleNamespace(parse_completion=lambda tokens: {"content": "generated"})
    engine.tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: "generated")
    effective = min(8, 49 - prompt_length)
    engine._completions_with_retry = AsyncMock(
        return_value=(
            {"choices": [{"raw_output": {"completion_token_ids": [9] * effective}, "finish_reason": finish_reason}]},
            {"ttft": 0.1},
        )
    )
    prompt = [1] * prompt_length

    async def chat_response(messages, **kwargs):
        return await engine.get_model_response_from_tokens(prompt, **kwargs)

    engine.get_model_response = chat_response
    request = {"max_tokens": 8, "model": "test"}
    request.update({"prompt": prompt} if token_path else {"messages": [{"role": "user", "content": "task"}]})
    response = asyncio.run(create_tinker_handler(engine)(request))
    engine._completions_with_retry.assert_awaited_once()
    assert engine._completions_with_retry.call_args.args[1] == effective
    # The cumulative proxy translates the completion choice back to chat format.
    if token_path:
        choice = response["choices"][0]
        choice["message"] = {"role": "assistant", "content": choice.pop("text")}
    trace = build_trace_record("task:0", request, response, 1.0, metadata={"other": "preserved"})
    assert trace.metadata["other"] == "preserved"
    if prompt_length == 45:
        assert trace.metadata["max_tokens_clamped"] is True
        assert trace.metadata["requested_max_tokens"] == 8
        assert trace.metadata["effective_max_tokens"] == 4
    else:
        assert "max_tokens_clamped" not in trace.metadata
    assert "_rllm_context_limit" not in strip_vllm_fields(response)
    traces = [trace]
    if compact:
        traces = TraceGraph(format="compact", version=1, deltas=[])
        traces.add(trace)
    initial_reason = TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED if finish_reason == "length" else TerminationReason.ENV_DONE
    episode = Episode(id="task:0", termination_reason=initial_reason, trajectories=[Trajectory(name="solver")])
    enriched = enrich_episode_with_traces(episode, traces, "task:0", {})
    assert enriched.termination_reason == expected
