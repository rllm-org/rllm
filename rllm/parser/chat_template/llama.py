"""Llama-3 chat-template parser.

Targets Llama 3.1 / 3.2 / 3.3 Instruct chat templates. The format is:

    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>\\n\\n{content}<|eot_id|>
    <|start_header_id|>user<|end_header_id|>\\n\\n{content}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>\\n\\n{content_or_tool_call_json}<|eot_id|>
    <|start_header_id|>ipython<|end_header_id|>\\n\\n{tool_result}<|eot_id|>
    ...

Note: the tool-response role in Llama is ``ipython`` (not ``tool``);
parse_tool emits the ipython header. Tool calls are rendered as bare JSON
(Llama 3.2) — see ``LlamaToolParser`` for the wire format.

The official HF chat_template also injects ``Cutting Knowledge Date`` /
``Today Date`` system-prefix lines and (when tools are provided) an
``Environment: ipython`` prefix and a tools-in-first-user-message block.
Reproducing those byte-for-byte is the Phase B task of G2.7; the current
implementation focuses on parse_assistant / parse_completion / parse_tool
correctness (Phase A) so the multi-turn RL completer can round-trip
Llama tool-calling rollouts.
"""

from __future__ import annotations

import json
import logging

from rllm.parser.chat_template.base import ChatTemplateParser
from rllm.parser.messages import AssistantMessage, Messages
from rllm.parser.tool_parser import LlamaToolParser
from rllm.tools.tool_base import ToolCall

logger = logging.getLogger(__name__)


class LlamaChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token = "<|begin_of_text|>"
        self.system_token = "<|start_header_id|>system<|end_header_id|>\n\n"
        self.user_token = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.assistant_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.ipython_token = "<|start_header_id|>ipython<|end_header_id|>\n\n"
        self.eot_token = "<|eot_id|>"
        self.eos_token = getattr(tokenizer, "eos_token", "<|end_of_text|>") or "<|end_of_text|>"
        self.generation_prompt = self.assistant_token

        self.tool_parser = LlamaToolParser()

    def parse(self, messages: Messages, add_generation_prompt: bool = False, is_first_msg: bool = False, accumulate_reasoning: bool = False, tools: list | None = None, **kwargs) -> str:
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
            elif message["role"] in ("tool", "ipython"):
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message: AssistantMessage, accumulate_reasoning: bool = False) -> str:
        content = (message.get("content", None) or "").strip()
        tool_calls = message.get("tool_calls", None) or []

        if not tool_calls:
            return self.assistant_token + content + self.eot_token

        # Llama 3.2 official template raises if len(tool_calls) > 1; we
        # tolerate multiple by emitting them on consecutive lines, which
        # matches what some Llama-fine-tunes produce. The first call is
        # the only one that round-trips cleanly through the official
        # parser; subsequent calls won't be reachable via apply_chat_template.
        body_parts = []
        if content:
            body_parts.append(content)

        for tool_call in tool_calls:
            if isinstance(tool_call, ToolCall):
                tc_dict = {"name": tool_call.name, "parameters": tool_call.arguments}
            elif isinstance(tool_call, dict) and "function" in tool_call:
                fn = tool_call["function"]
                args = fn.get("arguments", fn.get("parameters", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tc_dict = {"name": fn.get("name"), "parameters": args}
            else:
                # raw {"name": ..., "parameters"|"arguments": ...} dict
                args = tool_call.get("parameters", tool_call.get("arguments", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tc_dict = {"name": tool_call.get("name"), "parameters": args}

            body_parts.append(json.dumps(tc_dict))

        body = "\n".join(body_parts)
        return self.assistant_token + body + self.eot_token

    def parse_tool(self, message):
        """Render a tool-result turn using Llama's ``ipython`` header.

        Accepts both ``{"role": "tool", "content": ...}`` and
        ``{"role": "ipython", "content": ...}`` shapes. If ``content`` is
        a dict/list, it is JSON-serialized to mirror what the official
        chat_template does.
        """
        content = message.get("content", "")
        if isinstance(content, dict | list):
            content = json.dumps(content)
        return self.ipython_token + str(content) + self.eot_token

    def _strip_special_tokens(self, text: str) -> str:
        for tok in (self.eos_token, self.eot_token, "<|eom_id|>"):
            if tok and text.endswith(tok):
                text = text[: -len(tok)]
        return text.strip()

    def parse_completion(self, completion_ids):
        """Decode a model completion and split it into
        ``{"content", "reasoning", "tool_calls"}``.

        Llama 3.x has no `<think>`/`</think>` reasoning convention in the
        Instruct chat template, so ``reasoning`` is always ``""``. If the
        completion contains a valid JSON tool-call body (optionally
        prefixed with ``<|python_tag|>``), it's extracted into
        ``tool_calls`` and stripped from ``content``.
        """
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
        body = self._strip_special_tokens(completion_text)

        tool_calls = self.tool_parser.parse(body)

        if tool_calls:
            # Strip the JSON tool-call body from content so we don't
            # double-emit it on the way back through parse_assistant.
            stripped = body
            if stripped.startswith(LlamaToolParser.PYTHON_TAG):
                stripped = stripped[len(LlamaToolParser.PYTHON_TAG) :].lstrip()
            json_span = LlamaToolParser._extract_first_json_object(stripped)
            if json_span is not None:
                idx = stripped.find(json_span)
                stripped = (stripped[:idx] + stripped[idx + len(json_span) :]).strip()
            content = stripped
        else:
            content = body

        return {
            "content": content,
            "reasoning": "",
            "tool_calls": tool_calls,
        }
