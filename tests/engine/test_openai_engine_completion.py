"""Regression tests for ``OpenAIEngine.completion`` token/logprob alignment."""

import asyncio
import logging
from types import SimpleNamespace

from rllm.engine.rollout.openai_engine import OpenAIEngine


class _FakeTokenizer:
    """Tokenizer whose ``encode(decode(ids))`` does not reproduce ``ids`` (like BPE merges)."""

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [ord(c) for c in text.replace("!!", "?")]


class _FakeChatParser:
    def parse_completion(self, completion_ids):  # noqa: ARG002
        return {"content": "", "reasoning": "", "tool_calls": []}


class _FakeCompletions:
    def __init__(self, choice):
        self._choice = choice

    async def create(self, **kwargs):  # noqa: ARG002
        return SimpleNamespace(choices=[self._choice], usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))


def _make_engine(choice) -> OpenAIEngine:
    engine = OpenAIEngine(model="m", tokenizer=_FakeTokenizer(), chat_parser=_FakeChatParser(), api_key="test", sampling_params={})
    engine.client = SimpleNamespace(completions=_FakeCompletions(choice))
    return engine


def _choice(text, *, token_ids, token_logprobs):
    return SimpleNamespace(
        text=text,
        token_ids=token_ids,
        finish_reason="stop",
        logprobs=SimpleNamespace(token_logprobs=token_logprobs),
    )


def test_retokenized_completion_drops_misaligned_logprobs(caplog):
    # Server sampled two "!" tokens (ids 0, 0) with one logprob each, but did not return token_ids.
    # Re-encoding "!!" yields a single different token, so the logprobs must not be paired with it.
    engine = _make_engine(_choice("!!", token_ids=None, token_logprobs=[-0.1, -0.2]))

    with caplog.at_level(logging.WARNING):
        out = asyncio.run(engine.completion([1, 2, 3]))

    assert out.completion_ids == [ord("?")]
    assert out.logprobs == []
    assert "retokenized" in caplog.text


def test_server_token_ids_keep_logprobs():
    engine = _make_engine(_choice("!!", token_ids=[0, 0], token_logprobs=[-0.1, -0.2]))

    out = asyncio.run(engine.completion([1, 2, 3]))

    assert out.completion_ids == [0, 0]
    assert out.logprobs == [-0.1, -0.2]
