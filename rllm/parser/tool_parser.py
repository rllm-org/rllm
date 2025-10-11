import json
from abc import ABC, abstractmethod
from typing import Any

from rllm.tools.tool_base import ToolCall


class ToolParser(ABC):
    @abstractmethod
    def parse(self, model_response: str) -> list[ToolCall]:
        """Extract tool calls from the model response."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_tool_prompt(self, tools_schema: str) -> str:
        """Get the tool prompt for the model."""
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def get_parser(cls, tokenizer) -> "ToolParser":
        """Factory method to get the appropriate tool parser based on a string identifier.

        Args:
            tokenizer: The tokenizer to use with the parser

        Returns:
            ToolParser: An instance of the requested parser

        Raises:
            ValueError: If the parser_type is not recognized
        """
        # Determine parser type based on tokenizer name or path
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            print(f"model_name: {model_name}, tokenizer_cls: {tokenizer_cls}")
            if any(x in model_name for x in ("deepseek", "deepscaler", "deepcoder")) and "llama" in tokenizer_cls:
                print(f"Using R1ToolParser for {tokenizer.name_or_path}")
                return R1ToolParser()
            elif "qwen" in model_name or "r2e" in model_name or "deepswe" in model_name or "qwen" in tokenizer_cls:
                print(f"Using QwenToolParser for {tokenizer.name_or_path}")
                return QwenToolParser()
        # TODO: add verfication to check equivalence of the parser with that from HuggingFace
        raise ValueError(f"No tool parser found for {tokenizer.name_or_path}")


class R1ToolParser(ToolParser):
    """Parser for R1 tool call format."""

    def __init__(self):
        """Initialize the R1 tool parser.

        Args:
            model (str): Model name for tokenizer (optional)
            tokenizer: Pre-initialized tokenizer (optional)
        """
        self.tool_calls_begin = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end = "<｜tool▁calls▁end｜>"
        self.tool_call_begin = "<｜tool▁call▁begin｜>"
        self.tool_call_end = "<｜tool▁call▁end｜>"
        self.tool_sep = "<｜tool▁sep｜>"
        self.tool_output_begin = "<｜tool▁response▁begin｜>"
        self.tool_output_end = "<｜tool_response_end｜>"

    def parse(self, model_response: str) -> list[ToolCall]:
        """Parse tool calls from model output.

        Args:
            model_output (str): Text containing tool calls

        Returns:
            ToolInputs: Parsed tool calls
        """
        tool_calls_dicts = self.parse_r1_tool_calls(model_response)

        # Convert dictionaries to ToolCall objects
        tool_calls = [ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in tool_calls_dicts]
        return tool_calls

    def parse_r1_tool_calls(self, text: str) -> list[dict]:
        """Parse tool calls from text using the R1 special token format.

        Format:
        <｜tool▁calls▁begin｜>
        <｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name
        ```json
        {"param1": "value1", "param2": "value2"}
        ```
        <｜tool▁call▁end｜>
        // Additional tool calls follow the same format
        <｜tool▁calls▁end｜>

        Returns:
            list[dict]: List of parsed tool calls, each containing 'name' and 'parameters'
        """
        tool_calls = []

        # Look for individual tool calls
        call_idx = 0
        while True:
            # Find the next tool call beginning
            call_idx = text.find(self.tool_call_begin, call_idx)
            if call_idx == -1:
                break

            # Find the end of this tool call
            call_start = call_idx + len(self.tool_call_begin)
            call_end = text.find(self.tool_call_end, call_start)
            if call_end == -1:
                break

            # Extract the content of this tool call
            call_content = text[call_start:call_end].strip()

            # Parse function name
            func_prefix = "function" + self.tool_sep
            func_start = call_content.find(func_prefix)

            if func_start != -1:
                # Extract function name after the prefix up to the next newline
                func_name_start = func_start + len(func_prefix)
                func_name_end = call_content.find("\n", func_name_start)

                if func_name_end == -1:
                    function_name = call_content[func_name_start:].strip()
                else:
                    function_name = call_content[func_name_start:func_name_end].strip()
            else:
                # If function prefix not found, skip this call
                call_idx = call_end + len(self.tool_call_end)
                continue

            # Extract JSON arguments
            json_start = call_content.find("```json\n")
            if json_start == -1:
                json_start = call_content.find("```json")
                if json_start == -1:
                    call_idx = call_end + len(self.tool_call_end)
                    continue
                json_start += len("```json")
            else:
                json_start += len("```json\n")

            json_end = call_content.find("```", json_start)
            if json_end == -1:
                call_idx = call_end + len(self.tool_call_end)
                continue

            args_str = call_content[json_start:json_end].strip()

            try:
                args_json = json.loads(args_str)
            except json.JSONDecodeError:
                call_idx = call_end + len(self.tool_call_end)
                continue

            # Add this tool call to our list
            tool_calls.append({"name": function_name, "arguments": args_json})

            # Move past this call for the next iteration
            call_idx = call_end + len(self.tool_call_end)

        return tool_calls

    def get_tool_prompt(self, tools_schema: str) -> str:
        return f"""
# Tools

You may call one or more functions to assist with the user query.
<tools>
{tools_schema}
</tools>

Output format for tool calls:

<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name
```json
{{"param1": "value1", "param2": "value2"}}
```
<｜tool▁call▁end｜>
// Additional tool calls follow the same format
<｜tool▁calls▁end｜>
"""


class QwenToolParser(ToolParser):
    def __init__(self):
        """Initialize the parser with specified type and model.

        Args:
            model (str): Model name for tokenizer (optional)
            parser_type (str): Type of parser to use ('qwen' or other parsers you might add)
        """
        self.tool_call_begin = "<tool_call>"
        self.tool_call_end = "</tool_call>"
        self.tool_output_begin = "<tool_response>"
        self.tool_output_end = "</tool_response>"

    def parse(self, model_response: str) -> list[ToolCall]:
        """Parse tool calls from model output.

        Args:
            model_output (str): Text containing tool calls

        Returns:
            ToolInputs: Parsed tool calls
        """
        tool_calls_dicts = self.parse_qwen_tool_calls(model_response)
        tool_calls = [ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in tool_calls_dicts]
        return tool_calls

    def parse_qwen_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """Parse tool calls from text using a simple token format.

        Format:
        <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>

        Returns:
            list[dict]: List of parsed tool calls, each containing 'name' and 'parameters'
        """

        tool_calls: list[dict[str, Any]] = []

        # Return empty list if no tool calls found
        if self.tool_call_begin not in text:
            return tool_calls

        # Process all tool calls in the text
        while self.tool_call_begin in text:
            start = text.find(self.tool_call_begin) + len(self.tool_call_begin)
            end = text.find(self.tool_call_end)
            if end == -1:
                break

            # Extract and parse the JSON content
            json_content = text[start:end].strip()
            try:
                call_data = json.loads(json_content)
                # Convert to common format matching parse_tool_calls output
                tool_calls.append({"name": call_data["name"], "arguments": call_data["arguments"]})
            except json.JSONDecodeError:
                print(f"Error parsing tool call: {json_content}")
                text = text[end + len(self.tool_call_end) :]
                continue

            # Move to next potential tool call
            text = text[end + len(self.tool_call_end) :]

        return tool_calls

    def get_tool_prompt(self, tools_schema: str) -> str:
        return f"""

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_schema}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
""".rstrip()
