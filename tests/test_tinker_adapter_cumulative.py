"""Tinker adapter: pre-tokenized prompt path for cumulative token mode.

When the gateway runs in ``cumulative_token_mode``, turn 2+ arrives at the
in-process handler as a completions-style request whose ``prompt`` is raw token
IDs (built by ``renderers.bridge_to_next_turn``). The handler must sample
straight from those tokens — bypassing message rendering — and return a
completions-style body the gateway can extract token IDs from.

Uses a fake engine so no Tinker service / model is required.
"""

import asyncio

from rllm.gateway.tinker_adapter import create_tinker_handler


class _ModelOutput:
    text = "ls -la"
    content = "ls -la"
    reasoning = ""
    tool_calls = []
    prompt_ids = [1, 2, 3, 4]
    completion_ids = [10, 11]
    logprobs = [-0.1, -0.2]
    prompt_length = 4
    completion_length = 2
    finish_reason = "stop"


class _FakeEngine:
    model_name = "qwen"

    def __init__(self):
        self.token_input = None
        self.messages = None

    async def get_token_output_from_token_input(self, token_input, **kwargs):
        self.token_input = token_input
        self.sampling_kwargs = kwargs
        return "sampled"

    def assemble_model_output(self, token_input, token_output):
        return _ModelOutput()

    async def get_model_response(self, messages, **kwargs):
        self.messages = messages
        return _ModelOutput()


def test_token_prompt_path_samples_from_tokens():
    engine = _FakeEngine()
    handler = create_tinker_handler(engine)

    resp = asyncio.run(handler({"prompt": [1, 2, 3, 4], "temperature": 0.7, "model": "qwen"}))

    # Sampled directly from the token IDs — message rendering bypassed.
    assert engine.token_input == [1, 2, 3, 4]
    assert engine.messages is None
    assert engine.sampling_kwargs.get("temperature") == 0.7

    # Completions-style body the gateway's cumulative handler understands.
    assert resp["object"] == "text_completion"
    assert resp["prompt_token_ids"] == [1, 2, 3, 4]
    assert resp["choices"][0]["text"] == "ls -la"
    assert resp["choices"][0]["token_ids"] == [10, 11]
    assert resp["choices"][0]["logprobs"]["token_logprobs"] == [-0.1, -0.2]


def test_chat_path_unchanged_by_token_branch():
    engine = _FakeEngine()
    handler = create_tinker_handler(engine)

    resp = asyncio.run(handler({"messages": [{"role": "user", "content": "hi"}], "model": "qwen"}))

    assert engine.messages == [{"role": "user", "content": "hi"}]
    assert engine.token_input is None  # token path not taken
    assert resp["object"] == "chat.completion"
    assert resp["choices"][0]["message"]["content"] == "ls -la"
    assert resp["prompt_token_ids"] == [1, 2, 3, 4]


def test_non_int_prompt_falls_through_to_chat():
    """A string ``prompt`` (or none) must not trigger the token path."""
    engine = _FakeEngine()
    handler = create_tinker_handler(engine)

    # No messages and a string prompt: should still go down the chat path
    # (messages defaulting to []), not the token path.
    asyncio.run(handler({"prompt": "hello", "messages": [{"role": "user", "content": "hi"}]}))
    assert engine.token_input is None
    assert engine.messages == [{"role": "user", "content": "hi"}]
