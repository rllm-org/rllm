from __future__ import annotations

import pytest

from rllm.eval._litellm_callbacks import strip_thinking_blocks


@pytest.mark.asyncio
async def test_strips_anthropic_thinking_blocks_from_assistant_content():
    data = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "text", "text": "answer"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                ],
            },
        ]
    }
    out = await strip_thinking_blocks.async_pre_call_hook(None, None, data, "anthropic_messages")
    content = out["messages"][1]["content"]
    assert [b["type"] for b in content] == ["text", "tool_use"]
    # user message untouched
    assert out["messages"][0]["content"] == "hi"


@pytest.mark.asyncio
async def test_strips_openai_shaped_thinking_blocks_field():
    data = {
        "messages": [
            {
                "role": "assistant",
                "content": "answer",
                "thinking_blocks": [{"type": "thinking", "thinking": "hmm", "signature": ""}],
                "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}],
            }
        ]
    }
    out = await strip_thinking_blocks.async_pre_call_hook(None, None, data, "acompletion")
    msg = out["messages"][0]
    assert "thinking_blocks" not in msg
    assert msg["tool_calls"]


@pytest.mark.asyncio
async def test_thinking_only_assistant_content_becomes_empty_string():
    data = {"messages": [{"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}]}]}
    out = await strip_thinking_blocks.async_pre_call_hook(None, None, data, "anthropic_messages")
    assert out["messages"][0]["content"] == ""


@pytest.mark.asyncio
async def test_no_messages_is_passthrough():
    data = {"prompt": "hello"}
    out = await strip_thinking_blocks.async_pre_call_hook(None, None, data, "acompletion")
    assert out == data
