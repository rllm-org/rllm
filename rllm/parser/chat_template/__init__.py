"""Per-family chat-template parsers.

This package replaces the monolithic ``rllm/parser/chat_template_parser.py``.
Imports continue to work through ``rllm.parser`` (canonical) or
``rllm.parser.chat_template_parser`` (legacy shim).
"""

from rllm.parser.chat_template.base import ChatTemplateParser
from rllm.parser.chat_template.deepseek_qwen import DeepseekQwenChatTemplateParser
from rllm.parser.chat_template.deepseek_v32_exp import DeepSeekV32ExpChatTemplateParser
from rllm.parser.chat_template.harmony import HarmonyChatTemplateParser
from rllm.parser.chat_template.kimi_k2_thinking import KimiK2ThinkingChatTemplateParser
from rllm.parser.chat_template.llama import LlamaChatTemplateParser
from rllm.parser.chat_template.qwen import QwenChatTemplateParser

__all__ = [
    "ChatTemplateParser",
    "DeepSeekV32ExpChatTemplateParser",
    "DeepseekQwenChatTemplateParser",
    "HarmonyChatTemplateParser",
    "KimiK2ThinkingChatTemplateParser",
    "LlamaChatTemplateParser",
    "QwenChatTemplateParser",
]
