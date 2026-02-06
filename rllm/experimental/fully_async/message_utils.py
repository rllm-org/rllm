"""
Utilities for converting response token IDs to OpenAI message format.
"""

import json
import re


# Regex for thinking content: <think>...</think>
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def parse_response(tokenizer, parser, response_ids: list[int], skip_special_tokens: bool = True) -> dict:
    """
    Convert response token IDs to OpenAI message format.

    Args:
        tokenizer: HuggingFace tokenizer
        response_ids: List of token IDs from model generation
        skip_special_tokens: Whether to skip special tokens when decoding

    Returns:
        OpenAI message dict: {"role": "assistant", "content": str, "tool_calls": list (optional)}
    """
    text = tokenizer.decode(response_ids, skip_special_tokens=skip_special_tokens)

    # Use the existing tool parser
    tool_call_objs = parser.parse(text)

    # Convert ToolCall objects to OpenAI format
    tool_calls = []
    for idx, tc in enumerate(tool_call_objs):
        tool_calls.append(
            {
                "id": f"call_{idx}",
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False) if isinstance(tc.arguments, dict) else tc.arguments,
                },
            }
        )

    # Remove tool call tags from content
    content = text
    if hasattr(parser, "tool_call_begin") and hasattr(parser, "tool_call_end"):
        # Remove tool call tags
        pattern = re.compile(re.escape(parser.tool_call_begin) + r".*?" + re.escape(parser.tool_call_end), re.DOTALL)
        content = pattern.sub("", content).strip()

    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls

    return message


def extract_thinking(text: str) -> tuple[str, str]:
    """
    Extract <think>...</think> reasoning content from response.

    Returns:
        (thinking_content, regular_content)
    """
    matches = THINK_PATTERN.findall(text)
    thinking = "\n".join(m.strip() for m in matches)
    content = THINK_PATTERN.sub("", text).strip()
    return thinking, content


def build_tool_message(tool_name: str, tool_output: str, tool_call_id: str | None = None) -> dict:
    """Build a tool response message in OpenAI format."""
    message = {"role": "tool", "name": tool_name, "content": tool_output}
    if tool_call_id:
        message["tool_call_id"] = tool_call_id
    return message