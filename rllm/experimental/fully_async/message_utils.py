"""
Utilities for converting response token IDs to OpenAI message format.
Based on verl's tool_parser.py patterns.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ToolCall:
    """OpenAI-compatible tool call structure."""
    id: str
    type: str = "function"
    function: Dict[str, Any] = None  # {"name": str, "arguments": str (JSON)}


class MessageConverter:
    """
    Convert response token IDs to OpenAI message format.
    Handles tool call parsing for Qwen models.
    """

    def __init__(self, tokenizer, model_type: str = "qwen"):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            model_type: "qwen", "hermes", or "gpt-oss"
        """
        self.tokenizer = tokenizer
        self.model_type = model_type

        # Regex patterns for different model types
        if model_type in ["qwen", "hermes"]:
            # Qwen/Hermes: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
            self.tool_call_regex = re.compile(
                r'<tool_call>\s*(.*?)\s*</tool_call>',
                re.DOTALL
            )
        elif model_type == "gpt-oss":
            # GPT-OSS: <|start|>assistant<|channel|>commentary to=functions.{name}...
            self.tool_call_regex = re.compile(
                r'<\|start\|>assistant<\|channel\>commentary to=functions\.(\w+).*?'
                r'<\|constrain\|>json<\|message\|>(.*?)<\|call\|>',
                re.DOTALL
            )

    def token_ids_to_message(
        self,
        response_ids: List[int],
        skip_special_tokens: bool = True
    ) -> Dict[str, Any]:
        """
        Convert response token IDs to OpenAI message format.

        Args:
            response_ids: List of token IDs from model generation
            skip_special_tokens: Whether to skip special tokens when decoding

        Returns:
            OpenAI message dict with role, content, and optional tool_calls

        Example:
            >>> message = converter.token_ids_to_message([151644, 872, ...])
            >>> print(message)
            {
                "role": "assistant",
                "content": "Let me search for that.",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"query": "Python tutorial"}'
                        }
                    }
                ]
            }
        """
        # Step 1: Decode token IDs to text
        text = self.tokenizer.decode(response_ids, skip_special_tokens=skip_special_tokens)

        # Step 2: Extract tool calls and clean content
        content, tool_calls = self._parse_tool_calls(text)

        # Step 3: Build OpenAI message
        message = {
            "role": "assistant",
            "content": content.strip()
        }

        # Add tool_calls only if present
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": tc.function
                }
                for tc in tool_calls
            ]

        return message

    def _parse_tool_calls(self, text: str) -> Tuple[str, List[ToolCall]]:
        """
        Extract tool calls from text and return cleaned content.

        Args:
            text: Decoded text from model

        Returns:
            (cleaned_content, tool_calls)
        """
        tool_calls = []

        if self.model_type in ["qwen", "hermes"]:
            tool_calls = self._parse_qwen_tool_calls(text)
        elif self.model_type == "gpt-oss":
            tool_calls = self._parse_gpt_oss_tool_calls(text)

        # Remove tool call sections from content
        content = self.tool_call_regex.sub("", text).strip()

        return content, tool_calls

    def _parse_qwen_tool_calls(self, text: str) -> List[ToolCall]:
        """
        Parse Qwen/Hermes format tool calls.

        Format:
            <tool_call>
            {"name": "function_name", "arguments": {"param": "value"}}
            </tool_call>
        """
        tool_calls = []
        matches = self.tool_call_regex.findall(text)

        for idx, match in enumerate(matches):
            try:
                # Parse JSON from tool call
                tool_data = json.loads(match.strip())

                # Extract name and arguments
                name = tool_data.get("name", "")
                arguments = tool_data.get("arguments", {})

                # Create ToolCall object
                tool_calls.append(ToolCall(
                    id=f"call_{idx}",
                    type="function",
                    function={
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False)
                    }
                ))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse tool call: {e}")
                continue

        return tool_calls

    def _parse_gpt_oss_tool_calls(self, text: str) -> List[ToolCall]:
        """
        Parse GPT-OSS format tool calls.

        Format:
            <|start|>assistant<|channel|>commentary to=functions.function_name
            <|constrain|>json<|message|>{"param": "value"}<|call|>
        """
        tool_calls = []
        matches = self.tool_call_regex.findall(text)

        for idx, (name, arguments_str) in enumerate(matches):
            try:
                # Parse arguments JSON
                arguments = json.loads(arguments_str.strip())

                tool_calls.append(ToolCall(
                    id=f"call_{idx}",
                    type="function",
                    function={
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False)
                    }
                ))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse tool call arguments: {e}")
                continue

        return tool_calls

    def batch_token_ids_to_messages(
        self,
        batch_response_ids: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Convert a batch of response token IDs to OpenAI messages.

        Args:
            batch_response_ids: List of token ID lists
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of OpenAI message dicts
        """
        return [
            self.token_ids_to_message(response_ids, skip_special_tokens)
            for response_ids in batch_response_ids
        ]


def build_tool_response_message(
    tool_name: str,
    tool_output: str,
    tool_call_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a tool response message in OpenAI format.

    Args:
        tool_name: Name of the tool that was called
        tool_output: Output/result from the tool execution
        tool_call_id: ID of the tool call this is responding to

    Returns:
        OpenAI tool message dict

    Example:
        >>> msg = build_tool_response_message(
        ...     tool_name="search",
        ...     tool_output="Found 3 results...",
        ...     tool_call_id="call_0"
        ... )
        >>> print(msg)
        {
            "role": "tool",
            "name": "search",
            "content": "Found 3 results...",
            "tool_call_id": "call_0"
        }
    """
    message = {
        "role": "tool",
        "name": tool_name,
        "content": tool_output
    }

    if tool_call_id:
        message["tool_call_id"] = tool_call_id

    return message


def extract_thinking_content(text: str) -> Tuple[str, str]:
    """
    Extract <think>...</think> reasoning content from response.

    Args:
        text: Response text that may contain thinking tags

    Returns:
        (thinking_content, regular_content)

    Example:
        >>> thinking, content = extract_thinking_content(
        ...     "<think>Let me analyze this...</think>The answer is 42."
        ... )
        >>> print(thinking)
        "Let me analyze this..."
        >>> print(content)
        "The answer is 42."
    """
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    matches = think_pattern.findall(text)

    # Extract all thinking content
    thinking = "\n".join(match.strip() for match in matches)

    # Remove thinking tags from regular content
    content = think_pattern.sub("", text).strip()

    return thinking, content


# ========== Example Usage ==========

if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Example 1: Basic conversion
    print("="*60)
    print("Example 1: Basic Message Conversion")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    converter = MessageConverter(tokenizer, model_type="qwen")

    # Simulate response with tool call
    response_text = """Let me search for that information.
<tool_call>
{"name": "search", "arguments": {"query": "Python tutorial", "limit": 5}}
</tool_call>"""

    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    message = converter.token_ids_to_message(response_ids)

    print(json.dumps(message, indent=2))

    # Example 2: Multiple tool calls
    print("\n" + "="*60)
    print("Example 2: Multiple Tool Calls")
    print("="*60)

    response_text_multi = """I'll help with both tasks.
<tool_call>
{"name": "calculate", "arguments": {"expression": "15 * 23"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"city": "Tokyo", "units": "celsius"}}
</tool_call>"""

    response_ids_multi = tokenizer.encode(response_text_multi, add_special_tokens=False)
    message_multi = converter.token_ids_to_message(response_ids_multi)

    print(json.dumps(message_multi, indent=2))

    # Example 3: Tool response message
    print("\n" + "="*60)
    print("Example 3: Building Tool Response")
    print("="*60)

    tool_response = build_tool_response_message(
        tool_name="search",
        tool_output="Found 3 relevant Python tutorials...",
        tool_call_id="call_0"
    )

    print(json.dumps(tool_response, indent=2))

    # Example 4: Extracting thinking content
    print("\n" + "="*60)
    print("Example 4: Extract Thinking Content")
    print("="*60)

    text_with_thinking = """<think>
The user is asking about the capital of France. This is a straightforward geography question.
I should provide a direct answer.
</think>

The capital of France is Paris."""

    thinking, content = extract_thinking_content(text_with_thinking)
    print(f"Thinking:\n{thinking}\n")
    print(f"Content:\n{content}")
