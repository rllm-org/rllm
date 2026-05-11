"""Harmony (gpt-oss / IMO) chat-template parser."""

from __future__ import annotations

from copy import deepcopy

from rllm.parser.chat_template.base import ChatTemplateParser
from rllm.parser.messages import Messages


class HarmonyChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer=None):
        from openai_harmony import (
            HarmonyEncodingName,
            load_harmony_encoding,
        )

        self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.generation_prompt = "<|start|>assistant"
        self.stop_sequences = [200002, 199999, 200012]  # <|endoftext|>, <|return|>, <|call|>

    def parse(self, messages: Messages, add_generation_prompt: bool = False, is_first_msg: bool = False, **kwargs) -> str:
        return self.parse_prompt_from_messages(messages, add_generation_prompt=add_generation_prompt, is_first_msg=is_first_msg, **kwargs)

    def parse_prompt_from_messages(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs):
        from openai_harmony import Conversation, DeveloperContent, Message, ReasoningEffort, RenderConversationConfig, Role, SystemContent

        messages = deepcopy(messages)
        harmony_messages: list[Message] = []

        if is_first_msg:
            # 1. system prompt
            reasoning_effort = ReasoningEffort(kwargs.get("reasoning_effort", "medium").capitalize())
            system_message = SystemContent.new().with_reasoning_effort(reasoning_effort)
            harmony_messages.append(Message.from_role_and_content(Role.SYSTEM, system_message))

            # 2. developer prompt
            if messages[0]["role"] == "system":
                instructions = messages.pop(0).get("content")
                developer_message = DeveloperContent.new().with_instructions(instructions)
                harmony_messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_message))

        # 3. the rest of the messages
        for message in messages:
            if message["role"] == "user":
                harmony_messages.append(Message.from_role_and_content(Role.USER, message["content"]))
            elif message["role"] == "assistant":
                reasoning = message.get("reasoning", None)
                content = message.get("content", None)
                if reasoning:
                    harmony_messages.append(Message.from_role_and_content(Role.ASSISTANT, reasoning).with_channel("analysis"))
                if content:
                    harmony_messages.append(Message.from_role_and_content(Role.ASSISTANT, content).with_channel("final"))
            elif message["role"] == "tool":
                raise NotImplementedError("Tool messages are not supported yet")
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        conv = Conversation.from_messages(harmony_messages)
        accumulate_thinking = kwargs.get("accumulate_thinking", False)
        config = RenderConversationConfig(auto_drop_analysis=not accumulate_thinking)
        prompt_ids: list[int] = self.enc.render_conversation(conv, config)

        try:
            prompt: str = self.enc.decode_utf8(prompt_ids)
        except UnicodeDecodeError:
            prompt: str = self.enc.decode(prompt_ids)
            print(f"Warning: UnicodeDecodeError when decoding prompt: {prompt[:1000]}...")

        if add_generation_prompt:
            prompt += self.generation_prompt

        return prompt

    def parse_completion(self, completion_ids: list[int], **kwargs) -> dict[str, str | list]:
        from openai_harmony import Role

        # NOTE: harmony will throw an error if the sequence ends during the header (e.g., due to length)
        harmony_messages = self.enc.parse_messages_from_completion_tokens(completion_ids, role=Role.ASSISTANT)

        analysis = ""
        final = ""
        for message in harmony_messages:
            content = message.content[0].text
            channel = message.channel

            if channel == "analysis":
                analysis += content
            elif channel == "final":
                final += content

        # TODO: handle tool calls

        return {
            "content": final,
            "reasoning": analysis,
            "tool_calls": [],
        }
