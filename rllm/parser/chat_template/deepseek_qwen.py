"""DeepseekQwen chat-template parser (Distill-Qwen / DeepScaler / DeepCoder)."""

from __future__ import annotations

import json
import logging
import re

from rllm.parser.chat_template.base import ChatTemplateParser
from rllm.parser.messages import AssistantMessage, Messages
from rllm.tools.tool_base import Tool, ToolCall, ToolOutput

logger = logging.getLogger(__name__)


class DeepseekQwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, disable_thinking=False):
        super().__init__(tokenizer)

        self.disable_thinking = disable_thinking
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.system_token = ""
        self.user_token = "<｜User｜>"
        self.assistant_token = "<｜Assistant｜>"
        if disable_thinking:
            self.generation_prompt = self.assistant_token + "</think>\n"
        else:
            self.generation_prompt = self.assistant_token + "<think>\n"

        from rllm.parser.tool_parser import R1ToolParser

        self.tool_parser = R1ToolParser()

    def parse(self, messages: Messages, add_generation_prompt: bool = False, is_first_msg: bool = False, tools: list[Tool | dict] = None, accumulate_reasoning: bool = False, **kwargs) -> str:
        tools = tools or []
        tools_prompt_str = ""
        if tools:
            try:
                tool_schema_strs = []
                for tool in tools:
                    if isinstance(tool, Tool):
                        tool_schema_str = json.dumps(tool.json)
                    elif isinstance(tool, dict):
                        tool_schema_str = json.dumps(tool)
                    else:
                        tool_schema_str = tool
                    tool_schema_strs.append(tool_schema_str)
                tools_schema_str = "\n".join(tool_schema_strs)
                tools_prompt_str = self.tool_parser.get_tool_prompt(tools_schema_str)
            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"Failed to format tools: {e}")

        result = ""

        if is_first_msg:
            result += self.bos_token

        if is_first_msg and messages[0]["role"] != "system" and tools_prompt_str:
            result += self.system_token + tools_prompt_str

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message, tools_prompt_str)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message, accumulate_reasoning=accumulate_reasoning)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message, tools_prompt_str=""):
        content = message["content"]

        if "# Tools" not in content and tools_prompt_str:
            content += tools_prompt_str

        return self.system_token + content

    def parse_user(self, message):
        return self.user_token + message["content"]

    def parse_assistant(self, message: AssistantMessage, accumulate_reasoning: bool = False) -> str:
        content = (message.get("content", None) or "").strip()
        reasoning = (message.get("reasoning", None) or "").strip()
        tool_calls = message.get("tool_calls", None) or []

        if not accumulate_reasoning:
            return self.assistant_token + content + self.eos_token
        elif not reasoning:
            return self.assistant_token + "<think>\n" + content + self.eos_token
        else:
            result = self.assistant_token

            if reasoning and accumulate_reasoning:
                result += "<think>\n" + reasoning
                if content:
                    result += "\n</think>\n\n"

            if content:
                result += content
                if tool_calls:
                    result += "\n"

            if tool_calls:
                try:
                    tool_calls_strs = []
                    for tool_call in tool_calls:
                        if isinstance(tool_call, ToolCall):
                            tool_call_dict = tool_call.to_dict()
                        elif isinstance(tool_call, dict) and "function" in tool_call:
                            tool_call_dict = tool_call["function"]
                        else:
                            tool_call_dict = tool_call
                        arguments_obj = tool_call_dict.get("arguments")
                        if isinstance(arguments_obj, str):
                            try:
                                arguments_obj = json.loads(arguments_obj)
                            except json.JSONDecodeError:
                                pass
                        tool_call_json = f"```json\n{json.dumps(arguments_obj)}\n```"
                        tool_call_str = f"{self.tool_parser.tool_call_begin}function{self.tool_parser.tool_sep}{tool_call_dict['name']}\n{tool_call_json}\n{self.tool_parser.tool_call_end}"
                        tool_calls_strs.append(tool_call_str)
                    joined_calls_str = "\n".join(tool_calls_strs)
                    tool_calls_str = f"{self.tool_parser.tool_calls_begin}\n{joined_calls_str}\n{self.tool_parser.tool_calls_end}"
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    logger.error(f"Failed to format tool calls: {e}")
                    tool_calls_str = ""

                result += tool_calls_str

            result += self.eos_token
            return result

    def parse_tool(self, message):
        tool_outputs: list[ToolOutput | dict] = message.get("tool_outputs", [])

        if not tool_outputs:
            return self.user_token + self.tool_parser.tool_output_begin + "\n" + message["content"] + "\n" + self.tool_parser.tool_output_end

        else:
            try:
                tool_outputs_strs = []
                for tool_output in tool_outputs:
                    if not isinstance(tool_output, ToolOutput):
                        tool_output = ToolOutput(**tool_output)
                    tool_output_str = f"{self.tool_parser.tool_output_begin}\n{str(tool_output)}\n{self.tool_parser.tool_output_end}"
                    tool_outputs_strs.append(tool_output_str)
                tool_outputs_str = "\n".join(tool_outputs_strs)
            except Exception as e:
                logger.error(f"Failed to format tool outputs: {e}")
                tool_outputs_str = ""

            return self.user_token + tool_outputs_str

    def parse_completion(self, completion_ids):
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        if completion_text.count("</think>") == 1:
            reasoning, _, content = completion_text.partition("</think>")
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            reasoning = reasoning.strip()
            content = content.strip()
        else:
            # DeepSeekQwen should always have reasoning. The completion was
            # truncated mid-thinking; strip the trailing eos before stashing
            # everything into ``reasoning``.
            text = completion_text
            if text.endswith(self.eos_token):
                text = text[: -len(self.eos_token)]
            reasoning = text.strip()
            content = ""

        if content:
            # parse tool calls from content
            tool_calls = self.tool_parser.parse(content)
            begin_pattern = re.escape(self.tool_parser.tool_call_begin)
            end_pattern = re.escape(self.tool_parser.tool_call_end)
            wrapper_begin_pattern = re.escape(self.tool_parser.tool_calls_begin)
            wrapper_end_pattern = re.escape(self.tool_parser.tool_calls_end)
            content = re.sub(f"{begin_pattern}.*?{end_pattern}", "", content, flags=re.DOTALL)
            content = re.sub(f"{wrapper_begin_pattern}.*?{wrapper_end_pattern}", "", content, flags=re.DOTALL)
            content = content.strip()
        else:
            tool_calls = []

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
        }
