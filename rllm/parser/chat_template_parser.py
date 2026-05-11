"""Legacy re-export shim.

The chat-template parser classes used to live in this single file. They've
been split into ``rllm/parser/chat_template/`` (one module per family).
External callers that import from ``rllm.parser.chat_template_parser`` keep
working — every symbol is re-exported here.

Prefer ``from rllm.parser import QwenChatTemplateParser`` (etc.) in new code.
"""

from rllm.parser.chat_template import (
    ChatTemplateParser,
    DeepseekQwenChatTemplateParser,
    DeepSeekV32ExpChatTemplateParser,
    HarmonyChatTemplateParser,
    KimiK2ThinkingChatTemplateParser,
    LlamaChatTemplateParser,
    QwenChatTemplateParser,
)

__all__ = [
    "ChatTemplateParser",
    "DeepSeekV32ExpChatTemplateParser",
    "DeepseekQwenChatTemplateParser",
    "HarmonyChatTemplateParser",
    "KimiK2ThinkingChatTemplateParser",
    "LlamaChatTemplateParser",
    "QwenChatTemplateParser",
]
