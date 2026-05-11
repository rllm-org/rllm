"""Kimi-K2 Thinking chat-template parser."""

from __future__ import annotations

from rllm.parser.chat_template.base import ChatTemplateParser
from rllm.parser.messages import AssistantMessage, Messages


class KimiK2ThinkingChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.tokenizer = tokenizer
        self.eos_token = "<|im_end|>"
        self.user_token = "<|im_user|>"
        self.assistant_token = "<|im_assistant|>"
        self.system_token = "<|im_system|>"
        self.middle_token = "<|im_middle|>"
        self.generation_prompt = f"{self.assistant_token}assistant{self.middle_token}"

    def parse(self, messages: Messages, add_generation_prompt: bool = False, is_first_msg: bool = False, tools: list = None, accumulate_reasoning: bool = False, **kwargs) -> str:
        if tools:
            raise NotImplementedError("Tools are not supported yet")

        result = ""

        # Add default system message if first message is not system
        if is_first_msg and (len(messages) == 0 or messages[0]["role"] != "system"):
            result += f"{self.system_token}system{self.middle_token}You are Kimi, an AI assistant created by Moonshot AI.{self.eos_token}"

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
        content = message.get("content", "")
        return f"{self.system_token}system{self.middle_token}{content}{self.eos_token}"

    def parse_user(self, message):
        content = message.get("content", "")
        return f"{self.user_token}user{self.middle_token}{content}{self.eos_token}"

    def parse_assistant(self, message: AssistantMessage, accumulate_reasoning: bool = False) -> str:
        content = message.get("content", "")
        reasoning = message.get("reasoning", "")

        result = f"{self.assistant_token}assistant{self.middle_token}"

        if reasoning and accumulate_reasoning:
            result += f"<think>{reasoning}</think>"
        else:
            result += "<think></think>"

        if content:
            result += content

        result += self.eos_token
        return result

    def parse_tool(self, message):
        raise NotImplementedError("Tools are not supported yet")

    def parse_completion(self, completion_ids: list[int]) -> dict[str, str | list]:
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        # Remove end token if present
        if completion_text.endswith(self.eos_token):
            completion_text = completion_text[: -len(self.eos_token)]

        # Parse thinking tags
        if completion_text.count("</think>") == 1:
            reasoning, _, content = completion_text.partition("</think>")
            if reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>") :]
            reasoning = reasoning.strip()
            content = content.strip()
        else:
            # generation was cut short during reasoning or no thinking tags
            if "<think>" in completion_text:
                reasoning = completion_text
                if reasoning.startswith("<think>"):
                    reasoning = reasoning[len("<think>") :]
                reasoning = reasoning.strip()
                content = ""
            else:
                reasoning = ""
                content = completion_text.strip()

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": [],
        }
