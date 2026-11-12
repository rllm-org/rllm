"""Tests for OpenAIEngine.completion prompt-logprob extraction."""

import asyncio
import types

from rllm.engine.rollout.openai_engine import OpenAIEngine


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [1, 2, 3]


class _FakeParser:
    def parse_completion(self, completion_ids):  # noqa: ARG002
        return {"content": "ok", "reasoning": "", "tool_calls": []}


def _fake_response(prompt_ids, logprob_values):
    """vLLM-shaped completion response carrying ``prompt_logprobs``."""
    prompt_logprobs = [None]
    for tid, lp in zip(prompt_ids[1:], logprob_values, strict=True):
        prompt_logprobs.append({str(tid): {"logprob": lp}})
    choice = types.SimpleNamespace(
        text="ok",
        token_ids=[7],
        finish_reason="length",
        logprobs=None,
        prompt_logprobs=prompt_logprobs,
    )
    return types.SimpleNamespace(
        choices=[choice],
        usage=types.SimpleNamespace(prompt_tokens=len(prompt_ids), completion_tokens=1),
    )


def _make_engine(response):
    engine = OpenAIEngine(
        model="teacher",
        tokenizer=_FakeTokenizer(),
        chat_parser=_FakeParser(),
        max_prompt_length=128,
        max_response_length=128,
        max_model_length=1024,
        api_key="EMPTY",
    )

    async def _create(**kwargs):
        return response

    engine.client = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))
    return engine


class TestPromptLogprobs:
    def test_prompt_logprobs_requested_via_extra_body(self):
        """vLLM takes ``prompt_logprobs`` through ``extra_body`` (how the
        distillation teacher query asks for it)."""
        prompt_ids = [10, 11, 12]
        engine = _make_engine(_fake_response(prompt_ids, [-0.5, -0.25]))

        output = asyncio.run(
            engine.completion(
                prompt_ids,
                max_tokens=1,
                extra_body={"prompt_logprobs": 1},
            )
        )

        assert output.prompt_logprobs == [None, -0.5, -0.25]

    def test_prompt_logprobs_requested_as_top_level_param(self):
        prompt_ids = [10, 11, 12]
        engine = _make_engine(_fake_response(prompt_ids, [-0.5, -0.25]))

        output = asyncio.run(engine.completion(prompt_ids, max_tokens=1, prompt_logprobs=1))

        assert output.prompt_logprobs == [None, -0.5, -0.25]

    def test_prompt_logprobs_absent_when_not_requested(self):
        prompt_ids = [10, 11, 12]
        engine = _make_engine(_fake_response(prompt_ids, [-0.5, -0.25]))

        output = asyncio.run(engine.completion(prompt_ids, max_tokens=1))

        assert output.prompt_logprobs == []
