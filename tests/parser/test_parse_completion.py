"""Round-trip identity for chat-template parsers (property iii′).

For every combination of (content, reasoning, tool_calls) we exercise:

    msg = AssistantMessage(...)
    rendered = parse_assistant(msg, accumulate_reasoning=True)
    completion_body = strip_header(rendered)   # remove <|im_start|>assistant\n
    completion_ids = tokenizer.encode(completion_body, add_special_tokens=False)
    parsed = parse_completion(completion_ids)

    assert parsed["content"] == msg["content"].strip()
    assert parsed["reasoning"] == (msg.get("reasoning") or "").strip()
    assert tool_call_args_equal(parsed["tool_calls"], msg.get("tool_calls", []))

The test exercises ``QwenChatTemplateParser`` against ``Qwen/Qwen3-0.6B``
locally (skips if unavailable). Other parsers' round-trip behavior is
audited in ``tmp/chat_parser_refactor_loop/remarks.md`` (G2.6 audit).
"""

from __future__ import annotations

import json

import pytest


def _load_qwen_parser():
    try:
        from transformers import AutoTokenizer  # noqa: WPS433
    except ImportError:
        pytest.skip("transformers not available")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B",
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen/Qwen3-0.6B tokenizer not available: {exc}")

    from rllm.parser import QwenChatTemplateParser

    return QwenChatTemplateParser(tokenizer)


def _load_deepseek_qwen_parser():
    try:
        from transformers import AutoTokenizer  # noqa: WPS433
    except ImportError:
        pytest.skip("transformers not available")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DeepSeek-R1-Distill-Qwen-1.5B tokenizer not available: {exc}")

    from rllm.parser import DeepseekQwenChatTemplateParser

    return DeepseekQwenChatTemplateParser(tokenizer)


def _strip_generation_prompt(rendered: str, gen_prompt: str) -> str:
    """Drop the leading generation-prompt prefix so what remains is the
    model's completion body (which is what ``parse_completion`` operates on).

    For Qwen, ``gen_prompt == "<|im_start|>assistant\\n"`` — the model emits
    everything after, including its own ``<think>`` opener.

    For DeepseekQwen, ``gen_prompt == "<｜Assistant｜><think>\\n"`` — the
    ``<think>\\n`` is already in the prompt, so the model emits the reasoning
    content directly without re-opening the think tag.
    """
    assert rendered.startswith(gen_prompt), f"rendered string did not start with the expected generation prompt.\n  expected: {gen_prompt!r}\n  got:      {rendered[: len(gen_prompt) + 20]!r}"
    return rendered[len(gen_prompt) :]


def _tool_call_args_equal(parsed_tool_calls, expected_tool_calls) -> tuple[bool, str]:
    """Compare parser output (list[ToolCall]) against the original input
    (list[dict]). The arguments may live as a dict in one and a JSON string
    in the other — normalize before comparing.
    """
    if len(parsed_tool_calls) != len(expected_tool_calls):
        return False, f"length mismatch: got {len(parsed_tool_calls)}, want {len(expected_tool_calls)}"
    for i, (got, want) in enumerate(zip(parsed_tool_calls, expected_tool_calls, strict=True)):
        got_name = getattr(got, "name", None) or got.get("name")
        want_name = want.get("name") or want.get("function", {}).get("name")
        if got_name != want_name:
            return False, f"tool_call[{i}].name: got {got_name!r}, want {want_name!r}"

        got_args = getattr(got, "arguments", None)
        if got_args is None:
            got_args = got.get("arguments")
        if isinstance(got_args, str):
            try:
                got_args = json.loads(got_args)
            except json.JSONDecodeError:
                pass

        want_args = want.get("arguments")
        if want_args is None:
            want_args = want.get("function", {}).get("arguments")
        if isinstance(want_args, str):
            try:
                want_args = json.loads(want_args)
            except json.JSONDecodeError:
                pass

        if got_args != want_args:
            return False, f"tool_call[{i}].arguments: got {got_args!r}, want {want_args!r}"
    return True, ""


def _roundtrip(parser, msg):
    rendered = parser.parse_assistant(msg, accumulate_reasoning=True)
    body = _strip_generation_prompt(rendered, parser.generation_prompt)
    completion_ids = parser.tokenizer.encode(body, add_special_tokens=False)
    return parser.parse_completion(completion_ids)


CONTENT_ONLY = {
    "role": "assistant",
    "content": "The answer is 87.",
}

REASONING_ONLY = {
    "role": "assistant",
    "content": "",
    "reasoning": "Let me think: 12 * 7 = 84, plus 3 is 87.",
}

CONTENT_PLUS_REASONING = {
    "role": "assistant",
    "content": "The answer is 87.",
    "reasoning": "Let me think: 12 * 7 = 84, plus 3 is 87.",
}

CONTENT_REASONING_TOOL_CALL = {
    "role": "assistant",
    "content": "I'll use the compute tool to evaluate this safely.",
    "reasoning": "The user asks for 12*7+3. Safer to call the tool.",
    "tool_calls": [
        {
            "id": "call_compute_1",
            "type": "function",
            "function": {
                "name": "compute",
                "arguments": '{"expression": "12 * 7 + 3"}',
            },
        }
    ],
}

TOOL_CALL_NO_CONTENT_NO_REASONING = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call_compute_2",
            "type": "function",
            "function": {
                "name": "compute",
                "arguments": '{"expression": "12 * 7 - 3"}',
            },
        }
    ],
}


@pytest.mark.parametrize(
    ("label", "msg"),
    [
        ("content_only", CONTENT_ONLY),
        ("reasoning_only", REASONING_ONLY),
        ("content_plus_reasoning", CONTENT_PLUS_REASONING),
        ("content_reasoning_tool_call", CONTENT_REASONING_TOOL_CALL),
        ("tool_call_no_content_no_reasoning", TOOL_CALL_NO_CONTENT_NO_REASONING),
    ],
)
def test_qwen_parse_completion_roundtrip(label, msg):
    parser = _load_qwen_parser()
    parsed = _roundtrip(parser, msg)

    expected_content = (msg.get("content") or "").strip()
    expected_reasoning = (msg.get("reasoning") or "").strip()
    expected_tool_calls = msg.get("tool_calls", [])

    # tool_call blocks live in `content` until parse_completion strips them.
    # Compare AFTER strip.
    assert parsed["content"] == expected_content, f"{label}: content mismatch\n  got:  {parsed['content']!r}\n  want: {expected_content!r}"
    assert parsed["reasoning"] == expected_reasoning, f"{label}: reasoning mismatch\n  got:  {parsed['reasoning']!r}\n  want: {expected_reasoning!r}"
    ok, reason = _tool_call_args_equal(parsed["tool_calls"], expected_tool_calls)
    assert ok, f"{label}: tool_calls mismatch — {reason}"


def test_qwen_parse_completion_handles_no_think_block():
    """When the model is in non-thinking mode (no `<think>...</think>`),
    the whole completion is treated as content. We simulate this by feeding
    a hand-crafted body without `<think>`.
    """
    parser = _load_qwen_parser()
    body = "Hello, the answer is 87.<|im_end|>"
    completion_ids = parser.tokenizer.encode(body, add_special_tokens=False)
    parsed = parser.parse_completion(completion_ids)
    assert parsed["content"] == "Hello, the answer is 87."
    assert parsed["reasoning"] == ""
    assert parsed["tool_calls"] == []


def test_qwen_parse_completion_handles_unfinished_think_block():
    """When generation is cut off inside the think block (no `</think>`),
    everything up to that point should land in reasoning, with empty content.
    """
    parser = _load_qwen_parser()
    body = "<think>\nLet me think about this very carefully and"
    completion_ids = parser.tokenizer.encode(body, add_special_tokens=False)
    parsed = parser.parse_completion(completion_ids)
    assert parsed["content"] == ""
    assert "Let me think" in parsed["reasoning"]
    assert parsed["tool_calls"] == []


# ── DeepseekQwen round-trip cases ────────────────────────────────────────
#
# R1-style models always emit reasoning (the gen prompt forces ``<think>\n``).
# So the realistic round-trip set is: reasoning+content, reasoning-only
# (truncated), and reasoning+content+tool_call. The "content_only" case
# isn't representative of what the model actually does, so we skip it here.


DEEPSEEK_REASONING_ONLY = {
    "role": "assistant",
    "content": "",
    "reasoning": "Let me think: 12 * 7 = 84, plus 3 is 87.",
}

DEEPSEEK_CONTENT_PLUS_REASONING = {
    "role": "assistant",
    "content": "The answer is 87.",
    "reasoning": "Let me think: 12 * 7 = 84, plus 3 is 87.",
}

DEEPSEEK_CONTENT_REASONING_TOOL_CALL = {
    "role": "assistant",
    "content": "I'll use the compute tool.",
    "reasoning": "The user asks for 12*7+3. Safer to call the tool.",
    "tool_calls": [
        {
            "id": "call_compute_1",
            "type": "function",
            "function": {
                "name": "compute",
                "arguments": '{"expression": "12 * 7 + 3"}',
            },
        }
    ],
}


@pytest.mark.parametrize(
    ("label", "msg"),
    [
        ("reasoning_only", DEEPSEEK_REASONING_ONLY),
        ("content_plus_reasoning", DEEPSEEK_CONTENT_PLUS_REASONING),
        ("content_reasoning_tool_call", DEEPSEEK_CONTENT_REASONING_TOOL_CALL),
    ],
)
def test_deepseek_qwen_parse_completion_roundtrip(label, msg):
    parser = _load_deepseek_qwen_parser()
    parsed = _roundtrip(parser, msg)

    expected_content = (msg.get("content") or "").strip()
    expected_reasoning = (msg.get("reasoning") or "").strip()
    expected_tool_calls = msg.get("tool_calls", [])

    assert parsed["content"] == expected_content, f"{label}: content mismatch\n  got:  {parsed['content']!r}\n  want: {expected_content!r}"
    assert parsed["reasoning"] == expected_reasoning, f"{label}: reasoning mismatch\n  got:  {parsed['reasoning']!r}\n  want: {expected_reasoning!r}"
    ok, reason = _tool_call_args_equal(parsed["tool_calls"], expected_tool_calls)
    assert ok, f"{label}: tool_calls mismatch — {reason}"
