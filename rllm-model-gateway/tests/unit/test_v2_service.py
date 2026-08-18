import asyncio
from collections import deque
from typing import Any

import pytest
from rllm_model_gateway.v2.service import GatewayService
from rllm_model_gateway.v2.types import GatewayError, GatewayRequest, TokenInput, TokenOutput

USER_0 = {"role": "user", "content": "question"}
ASSISTANT_0 = {"role": "assistant", "content": "answer-1"}
USER_1 = {"role": "user", "content": "follow-up"}
ASSISTANT_1 = {"role": "assistant", "content": "answer-2"}
USER_2 = {"role": "user", "content": "another follow-up"}


def _output(token_id: int) -> TokenOutput:
    return TokenOutput(
        completion_token_ids=[token_id],
        logprobs=[-0.1],
        finish_reason="stop",
        weight_version=3,
    )


def _request(
    request_id: str,
    messages: list[dict[str, Any]],
    *,
    session_id: str = "session-1",
    tools: list[dict[str, Any]] | None = None,
    sampling_params: dict[str, Any] | None = None,
) -> GatewayRequest:
    return GatewayRequest(
        request_id=request_id,
        session_id=session_id,
        messages=messages,
        tools=tools or [],
        sampling_params=sampling_params or {},
    )


class FakeTokenization:
    def __init__(self) -> None:
        self.render_calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        self.bridge_calls: list[tuple[list[int], list[int], list[dict[str, Any]], list[dict[str, Any]]]] = []
        self.bridge_result: list[int] | None | object = _DEFAULT_BRIDGE

    def stop_token_ids(self) -> list[int]:
        return [98, 99]

    def encode(self, prompt: str) -> list[int]:
        return [ord(character) for character in prompt]

    def decode(self, token_ids: list[int]) -> str:
        return f"decoded-{token_ids[0]}"

    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[int]:
        self.render_calls.append((messages, tools))
        return [100 + len(self.render_calls)]

    def bridge(
        self,
        prompt_token_ids: list[int],
        completion_token_ids: list[int],
        new_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[int] | None:
        self.bridge_calls.append((prompt_token_ids, completion_token_ids, new_messages, tools))
        if self.bridge_result is not _DEFAULT_BRIDGE:
            return self.bridge_result  # type: ignore[return-value]
        return [*prompt_token_ids, *completion_token_ids, 200 + len(self.bridge_calls)]

    def parse_completion(self, token_ids: list[int], tools: list[dict[str, Any]]) -> dict[str, Any]:
        token_id = token_ids[0]
        return {
            "content": f"answer-{token_id}",
            "reasoning_content": f"reasoning-{token_id}",
            "tool_calls": [],
        }


_DEFAULT_BRIDGE = object()


class FakeInferenceClient:
    def __init__(self, *results: TokenOutput | Exception) -> None:
        self.results = deque(results)
        self.requests: list[TokenInput] = []

    async def generate(self, request: TokenInput) -> TokenOutput:
        self.requests.append(request)
        result = self.results.popleft()
        if isinstance(result, Exception):
            raise result
        return result

    async def close(self) -> None:
        pass


class ConcurrentInferenceClient:
    def __init__(self) -> None:
        self.requests: list[TokenInput] = []
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request: TokenInput) -> TokenOutput:
        self.requests.append(request)
        call_number = len(self.requests)
        if call_number == 2:
            self.all_started.set()
        await self.release.wait()
        return _output(call_number + 1)

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_generation_merges_sampling_params_and_records_effective_input() -> None:
    tokenization = FakeTokenization()
    inference = FakeInferenceClient(_output(1))
    service = GatewayService(tokenization, inference)
    service.create_session("session-1", {"temperature": 0.4, "max_tokens": 12})

    request = _request(
        "root",
        [USER_0],
        sampling_params={"temperature": 0.9, "top_p": 0.8, "n": 1},
    )
    response = await service.generate(request)

    assert response.request_id == "root"
    assert response.content == "answer-1"
    assert inference.requests == [
        TokenInput(
            routing_key="session-1",
            prompt_token_ids=[101],
            sampling_params={
                "temperature": 0.4,
                "top_p": 0.8,
                "max_tokens": 12,
                "stop_token_ids": [98, 99],
            },
        )
    ]

    traces = service.get_session_traces("session-1")["traces"]
    assert len(traces) == 1
    assert traces[0]["request"]["sampling_params"] == {"temperature": 0.9, "top_p": 0.8, "n": 1}
    assert traces[0]["input"]["sampling_params"] == inference.requests[0].sampling_params
    assert traces[0]["lineage"] == {"parent_request_id": None, "root_request_id": "root"}


@pytest.mark.asyncio
async def test_generation_rejects_n_greater_than_one_before_inference() -> None:
    inference = FakeInferenceClient(_output(1))
    service = GatewayService(FakeTokenization(), inference)
    service.create_session("session-1")

    with pytest.raises(GatewayError, match="n=1"):
        await service.generate(_request("request", [USER_0], sampling_params={"n": 2}))

    assert inference.requests == []
    assert service.get_session_traces("session-1")["traces"] == []


@pytest.mark.asyncio
async def test_failed_generation_does_not_create_a_trace() -> None:
    service = GatewayService(FakeTokenization(), FakeInferenceClient(RuntimeError("backend failed")))
    service.create_session("session-1")

    with pytest.raises(RuntimeError, match="backend failed"):
        await service.generate(_request("request", [USER_0]))

    assert service.get_session_traces("session-1")["traces"] == []


@pytest.mark.asyncio
async def test_sequential_turns_bridge_from_the_unique_longest_parent() -> None:
    tokenization = FakeTokenization()
    service = GatewayService(tokenization, FakeInferenceClient(_output(1), _output(2), _output(3)))
    service.create_session("session-1")

    await service.generate(_request("root", [USER_0]))
    await service.generate(_request("child", [USER_0, ASSISTANT_0, USER_1]))
    await service.generate(_request("grandchild", [USER_0, ASSISTANT_0, USER_1, ASSISTANT_1, USER_2]))

    traces = service.get_session_traces("session-1")["traces"]
    assert [trace["lineage"] for trace in traces] == [
        {"parent_request_id": None, "root_request_id": "root"},
        {"parent_request_id": "root", "root_request_id": "root"},
        {"parent_request_id": "child", "root_request_id": "root"},
    ]
    assert len(tokenization.render_calls) == 1
    assert tokenization.bridge_calls == [
        ([101], [1], [USER_1], []),
        ([101, 1, 201], [2], [USER_2], []),
    ]


@pytest.mark.asyncio
async def test_matching_only_requires_assistant_fields_supplied_by_caller() -> None:
    tokenization = FakeTokenization()
    service = GatewayService(tokenization, FakeInferenceClient(_output(1), _output(2)))
    service.create_session("session-1")

    await service.generate(_request("root", [USER_0]))
    await service.generate(
        _request(
            "child",
            [USER_0, {"role": "assistant", "content": "answer-1"}, USER_1],
        )
    )

    child = service.get_session_traces("session-1")["traces"][1]
    assert child["lineage"] == {"parent_request_id": "root", "root_request_id": "root"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "assistant_message",
    [
        {"role": "assistant", "content": "wrong answer"},
        {"role": "assistant", "reasoning_content": "wrong reasoning"},
        {"role": "assistant", "tool_calls": [{"id": "wrong"}]},
        {"role": "assistant"},
        {"role": "user", "content": "answer-1"},
    ],
)
async def test_response_mismatch_does_not_create_lineage(
    assistant_message: dict[str, Any],
) -> None:
    tokenization = FakeTokenization()
    service = GatewayService(tokenization, FakeInferenceClient(_output(1), _output(2)))
    service.create_session("session-1")

    await service.generate(_request("root", [USER_0]))
    await service.generate(_request("mismatch", [USER_0, assistant_message, USER_1]))

    trace = service.get_session_traces("session-1")["traces"][1]
    assert trace["lineage"] == {"parent_request_id": None, "root_request_id": "mismatch"}
    assert len(tokenization.render_calls) == 2


@pytest.mark.asyncio
async def test_matching_can_use_reasoning_without_content() -> None:
    service = GatewayService(FakeTokenization(), FakeInferenceClient(_output(1), _output(2)))
    service.create_session("session-1")

    await service.generate(_request("root", [USER_0]))
    await service.generate(
        _request(
            "child",
            [USER_0, {"role": "assistant", "reasoning_content": "reasoning-1"}, USER_1],
        )
    )

    trace = service.get_session_traces("session-1")["traces"][1]
    assert trace["lineage"] == {"parent_request_id": "root", "root_request_id": "root"}


@pytest.mark.asyncio
async def test_tools_must_match_for_a_trace_to_be_a_parent() -> None:
    tool = {"type": "function", "function": {"name": "lookup"}}
    tokenization = FakeTokenization()
    service = GatewayService(tokenization, FakeInferenceClient(_output(1), _output(2)))
    service.create_session("session-1")

    await service.generate(_request("root", [USER_0], tools=[tool]))
    await service.generate(_request("other-tools", [USER_0, ASSISTANT_0, USER_1]))

    traces = service.get_session_traces("session-1")["traces"]
    assert traces[1]["lineage"] == {"parent_request_id": None, "root_request_id": "other-tools"}
    assert len(tokenization.render_calls) == 2
    assert tokenization.bridge_calls == []


@pytest.mark.asyncio
async def test_failed_bridge_falls_back_to_a_new_root() -> None:
    tokenization = FakeTokenization()
    tokenization.bridge_result = None
    service = GatewayService(tokenization, FakeInferenceClient(_output(1), _output(2)))
    service.create_session("session-1")

    await service.generate(_request("root", [USER_0]))
    await service.generate(_request("fallback", [USER_0, ASSISTANT_0, USER_1]))

    traces = service.get_session_traces("session-1")["traces"]
    assert traces[1]["lineage"] == {"parent_request_id": None, "root_request_id": "fallback"}
    assert len(tokenization.bridge_calls) == 1
    assert len(tokenization.render_calls) == 2


@pytest.mark.asyncio
async def test_equal_longest_matches_are_ambiguous_and_create_a_root(caplog: pytest.LogCaptureFixture) -> None:
    tokenization = FakeTokenization()
    service = GatewayService(
        tokenization,
        FakeInferenceClient(_output(1), _output(2), _output(2), _output(3)),
    )
    service.create_session("session-1")

    child_messages = [USER_0, ASSISTANT_0, USER_1]
    await service.generate(_request("root", [USER_0]))
    await service.generate(_request("child-1", child_messages))
    await service.generate(_request("child-2", child_messages))

    with caplog.at_level("WARNING"):
        await service.generate(_request("ambiguous", [*child_messages, ASSISTANT_1, USER_2]))

    traces = service.get_session_traces("session-1")["traces"]
    assert traces[3]["lineage"] == {"parent_request_id": None, "root_request_id": "ambiguous"}
    assert "2 lineage matches with the same message length" in caplog.text


@pytest.mark.asyncio
async def test_concurrent_duplicate_continuations_become_siblings() -> None:
    tokenization = FakeTokenization()
    initial_inference = FakeInferenceClient(_output(1))
    service = GatewayService(tokenization, initial_inference)
    service.create_session("session-1")
    await service.generate(_request("root", [USER_0]))

    concurrent_inference = ConcurrentInferenceClient()
    service._inference_client = concurrent_inference
    messages = [USER_0, ASSISTANT_0, USER_1]
    first = asyncio.create_task(service.generate(_request("sibling-1", messages)))
    second = asyncio.create_task(service.generate(_request("sibling-2", messages)))
    await asyncio.wait_for(concurrent_inference.all_started.wait(), timeout=1)
    concurrent_inference.release.set()
    await asyncio.gather(first, second)

    traces = service.get_session_traces("session-1")["traces"]
    siblings = {trace["request"]["request_id"]: trace for trace in traces[1:]}
    assert set(siblings) == {"sibling-1", "sibling-2"}
    assert {trace["lineage"]["parent_request_id"] for trace in siblings.values()} == {"root"}
    assert {trace["lineage"]["root_request_id"] for trace in siblings.values()} == {"root"}
    assert len(concurrent_inference.requests) == 2


@pytest.mark.asyncio
async def test_sessions_keep_independent_trace_graphs() -> None:
    service = GatewayService(FakeTokenization(), FakeInferenceClient(_output(1), _output(2)))
    service.create_session("session-1")
    service.create_session("session-2")

    await service.generate(_request("root-1", [USER_0], session_id="session-1"))
    await service.generate(_request("root-2", [USER_0], session_id="session-2"))

    first = service.get_session_traces("session-1")["traces"]
    second = service.get_session_traces("session-2")["traces"]
    assert [trace["request"]["request_id"] for trace in first] == ["root-1"]
    assert [trace["request"]["request_id"] for trace in second] == ["root-2"]


@pytest.mark.asyncio
async def test_text_and_token_id_completions_are_always_roots() -> None:
    tokenization = FakeTokenization()
    inference = FakeInferenceClient(_output(1), _output(2))
    service = GatewayService(tokenization, inference)
    service.create_session("session-1")

    await service.generate(GatewayRequest(request_id="text", session_id="session-1", prompt="hello"))
    await service.generate(GatewayRequest(request_id="tokens", session_id="session-1", prompt_token_ids=[7, 8]))

    traces = service.get_session_traces("session-1")["traces"]
    assert [trace["lineage"] for trace in traces] == [
        {"parent_request_id": None, "root_request_id": "text"},
        {"parent_request_id": None, "root_request_id": "tokens"},
    ]
    assert inference.requests[0].prompt_token_ids == [ord(character) for character in "hello"]
    assert inference.requests[1].prompt_token_ids == [7, 8]
