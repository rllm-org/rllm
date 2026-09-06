"""Regression tests for OpenAIEngine.completion()'s token_ids retokenization fallback."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from rllm.engine.rollout.openai_engine import OpenAIEngine


class _FakeTokenizer:
    """Always returns a fixed, deliberately non-native-length token stream.

    Stands in for a real tokenizer where encode(decode(ids)) != ids, which is common
    (not a rare edge case) with BPE-style tokenizers.
    """

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [101, 102, 103]


class _FakeChatParser:
    def parse_completion(self, completion_ids):  # noqa: ARG002
        return {"content": "", "reasoning": None, "tool_calls": []}


def _make_engine():
    engine = OpenAIEngine(
        model="test-model",
        tokenizer=_FakeTokenizer(),
        chat_parser=_FakeChatParser(),
        max_prompt_length=64,
        max_response_length=64,
        base_url="http://localhost:0/v1",
        api_key="test",
    )
    engine.sampling_params = {}
    return engine


def _completions_response(*, token_ids, logprobs):
    choice = SimpleNamespace(
        text="some decoded completion text",
        finish_reason="stop",
        logprobs=SimpleNamespace(token_logprobs=logprobs),
    )
    if token_ids is not None:
        choice.token_ids = token_ids
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=len(logprobs)),
    )


def test_logprobs_dropped_when_token_ids_fallback_fires():
    """No native token_ids -> completion_ids is retokenized text, so the server's
    logprobs (aligned to what it actually sampled) can no longer be trusted to line
    up with completion_ids and must be treated as absent, not silently paired."""
    engine = _make_engine()
    response = _completions_response(token_ids=None, logprobs=[-0.1, -0.2])
    engine.client = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response)))

    output = asyncio.run(engine.completion([1, 2, 3, 4]))

    assert output.completion_ids == [101, 102, 103]
    assert output.logprobs == []


def test_logprobs_kept_when_token_ids_are_native():
    """When the server returns native token_ids, completion_ids and logprobs come
    from the same sampled stream and remain aligned, so logprobs must be preserved."""
    engine = _make_engine()
    response = _completions_response(token_ids=[10, 20, 30], logprobs=[-0.5, -0.6, -0.7])
    engine.client = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=response)))

    output = asyncio.run(engine.completion([1, 2, 3, 4]))

    assert output.completion_ids == [10, 20, 30]
    assert output.logprobs == [-0.5, -0.6, -0.7]
