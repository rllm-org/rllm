"""DeepSeek-V3.2-Exp / DeepSeek-Math-V2 chat-template parser."""

from __future__ import annotations

from rllm.parser.chat_template.base import ChatTemplateParser
from rllm.parser.messages import AssistantMessage, Messages
from rllm.tools.tool_base import Tool


class DeepSeekV32ExpChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, disable_thinking=False):
        self.tokenizer = tokenizer
        self.disable_thinking = disable_thinking
        self.bos_token = "<｜begin▁of▁sentence｜>"
        self.eos_token = "<｜end▁of▁sentence｜>"
        self.system_token = ""
        self.user_token = "<｜User｜>"
        self.assistant_token = "<｜Assistant｜>"
        if disable_thinking:
            self.generation_prompt = self.assistant_token + "</think>"
        else:
            self.generation_prompt = self.assistant_token + "<think>"

    def parse(self, messages: Messages, add_generation_prompt: bool = False, is_first_msg: bool = False, tools: list[Tool | dict] = None, accumulate_reasoning: bool = False, **kwargs) -> str:
        if tools:
            raise NotImplementedError("Tools are not supported yet")

        result = ""

        if is_first_msg:
            result += self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
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

    def parse_system(self, message):
        return self.system_token + message["content"]

    def parse_user(self, message):
        return self.user_token + message["content"]

    def parse_assistant(self, message: AssistantMessage, accumulate_reasoning: bool = False) -> str:
        reasoning = message.get("reasoning", None)
        content = message.get("content", None)

        result = self.assistant_token
        if reasoning and accumulate_reasoning:
            result += "<think>" + reasoning
        if content:
            result += "</think>" + content
        result += self.eos_token

        return result

    def parse_tool(self, message):
        raise NotImplementedError("Tools are not supported yet")

    def parse_completion(self, completion_ids):
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        if completion_text.count("</think>") == 1:
            reasoning, _, content = completion_text.partition("</think>")
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            reasoning = reasoning.strip()
            content = content.strip()
        elif not self.disable_thinking:
            # generation was cut short during reasoning
            reasoning = completion_text
            reasoning = reasoning.strip()
            content = ""
        else:
            # thinking is disabled, so everything is content
            reasoning = ""
            content = completion_text
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            content = content.strip()

        # TODO: handle tool calls

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": [],
        }
