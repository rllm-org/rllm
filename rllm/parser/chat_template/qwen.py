"""Qwen3 / Qwen-VL chat-template parser."""

from __future__ import annotations

import json
import logging
import re
from copy import deepcopy

from rllm.parser.chat_template.base import ChatTemplateParser
from rllm.parser.messages import AssistantMessage, Messages
from rllm.tools.tool_base import Tool, ToolCall, ToolOutput

logger = logging.getLogger(__name__)


class QwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, processor=None, disable_thinking=False):
        super().__init__(tokenizer, processor=processor)
        self.disable_thinking = disable_thinking
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.eot_token = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        if disable_thinking:
            self.assistant_token += "<think>\n\n</think>\n\n"
        self.generation_prompt = self.assistant_token
        self.image_token = "<|image_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"
        self.stop_sequences = [151645]

        from rllm.parser.tool_parser import QwenToolParser

        self.tool_parser = QwenToolParser()

    def parse(self, messages: Messages, add_generation_prompt: bool = False, is_first_msg: bool = False, tools: list[Tool] = None, accumulate_reasoning: bool = False, **kwargs) -> str:
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
                logger.error(f"Failed to format tools: {e}")

        result = ""

        # if the first message is not a system message, add the system message
        if is_first_msg and messages[0]["role"] != "system":
            result += self.system_token + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." + tools_prompt_str + self.eot_token

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

        return self.system_token + content + self.eot_token

    def parse_user(self, message):
        if "images" in message and message["images"] is not None:
            assert isinstance(message["images"], list), "images must be a list"
            n_imgs = len(message["images"])
            content = message["content"]
            if message["content"].startswith("<image>"):
                content = content[len("<image>") :]
            vision_tokens = (self.vision_start_token + self.image_token + self.vision_end_token) * n_imgs
            return self.user_token + vision_tokens + content + self.eot_token

        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message: AssistantMessage, accumulate_reasoning: bool = False) -> str:
        content = (message.get("content", None) or "").strip()
        reasoning = (message.get("reasoning", None) or "").strip()
        tool_calls = message.get("tool_calls", None) or []

        if not reasoning and not tool_calls:
            return self.assistant_token + content + self.eot_token

        else:
            result = self.assistant_token
            if reasoning and accumulate_reasoning:
                result += "<think>\n" + reasoning
                if content or tool_calls:
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
                        tool_call_for_dump = dict(tool_call_dict)
                        if arguments_obj is not None:
                            tool_call_for_dump["arguments"] = arguments_obj
                        tool_call_str = f"{self.tool_parser.tool_call_begin}\n{json.dumps(tool_call_for_dump)}\n{self.tool_parser.tool_call_end}"
                        tool_calls_strs.append(tool_call_str)
                    tool_calls_str = "\n".join(tool_calls_strs)
                except Exception as e:
                    logger.error(f"Failed to format tool calls: {e}")
                    tool_calls_str = ""

                result += tool_calls_str

            result += self.eot_token
            return result

    def parse_tool(self, message):
        tool_outputs: list[ToolOutput | dict] = message.get("tool_outputs", [])

        if not tool_outputs:
            return self.user_token + self.tool_parser.tool_output_begin + "\n" + message["content"] + "\n" + self.tool_parser.tool_output_end + self.eot_token

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

            return self.user_token + tool_outputs_str + self.eot_token

    def _strip_special_tokens(self, text):
        if text.endswith(self.eos_token):
            text = text[: -len(self.eos_token)]
        if text.endswith(self.eot_token):
            text = text[: -len(self.eot_token)]
        return text.strip()

    def parse_completion(self, completion_ids):
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
        if completion_text.count("</think>") == 1:
            reasoning, _, content = completion_text.partition("</think>")
            if reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>") :]
            reasoning = reasoning.strip()
            content = self._strip_special_tokens(content)
        elif not self.disable_thinking:
            # Two cases where the model didn't output </think>:
            # 1. Started <think> but no </think> -> thinking model, treat rest as reasoning, content=""
            # 2. No <think> at all -> non-thinking model (e.g. instruct), treat full text as content
            if "<think>" in completion_text:
                reasoning = completion_text
                if reasoning.startswith("<think>"):
                    reasoning = reasoning[len("<think>") :]
                reasoning = self._strip_special_tokens(reasoning)
                content = ""
            else:
                reasoning = ""
                content = self._strip_special_tokens(completion_text)
        else:
            # thinking is disabled, so everything is content
            reasoning = ""
            content = self._strip_special_tokens(completion_text)

        if content:
            tool_calls = self.tool_parser.parse(content)
            begin_pattern = re.escape(self.tool_parser.tool_call_begin)
            end_pattern = re.escape(self.tool_parser.tool_call_end)
            content = re.sub(f"{begin_pattern}.*?{end_pattern}", "", content, flags=re.DOTALL)
            content = content.strip()
        else:
            tool_calls = []

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
        }

    def process_image_data(self, messages):
        from qwen_vl_utils import fetch_image

        messages = deepcopy(messages)
        image_data = []
        for message in messages:
            if "images" in message and message["images"] is not None:
                assert isinstance(message["images"], list), "images must be a list"
                images = message["images"]
                if not images or images[0] is None:
                    continue
                for image in images:
                    image_dict = image if isinstance(image, dict) else {"image": image}
                    processed_image = fetch_image(image_dict, image_patch_size=self.processor.image_processor.patch_size)  # PIL.Image.Image
                    image_data.append(processed_image)
        return image_data
