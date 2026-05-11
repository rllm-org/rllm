from rllm.parser.chat_template import (
    ChatTemplateParser,
    DeepseekQwenChatTemplateParser,
    DeepSeekV32ExpChatTemplateParser,
    HarmonyChatTemplateParser,
    KimiK2ThinkingChatTemplateParser,
    LlamaChatTemplateParser,
    QwenChatTemplateParser,
)
from rllm.parser.messages import (
    AssistantMessage,
    ContentBlock,
    FunctionCall,
    ImageUrlBlock,
    Message,
    MessageList,
    Messages,
    MessageSnapshot,
    SystemMessage,
    TextBlock,
    ToolCallDict,
    ToolMessage,
    UserMessage,
    from_openai,
    to_openai,
)
from rllm.parser.tool_parser import QwenToolParser, R1ToolParser, ToolParser

__all__ = [
    "AssistantMessage",
    "ChatTemplateParser",
    "ContentBlock",
    "DeepSeekV32ExpChatTemplateParser",
    "DeepseekQwenChatTemplateParser",
    "FunctionCall",
    "HarmonyChatTemplateParser",
    "ImageUrlBlock",
    "KimiK2ThinkingChatTemplateParser",
    "LlamaChatTemplateParser",
    "Message",
    "MessageList",
    "MessageSnapshot",
    "Messages",
    "QwenChatTemplateParser",
    "QwenToolParser",
    "R1ToolParser",
    "SystemMessage",
    "TextBlock",
    "ToolCallDict",
    "ToolMessage",
    "ToolParser",
    "UserMessage",
    "from_openai",
    "to_openai",
]


PARSER_REGISTRY = {
    "r1": R1ToolParser,
    "qwen": QwenToolParser,
}


def get_tool_parser(parser_name: str) -> type[ToolParser]:
    assert parser_name in PARSER_REGISTRY, f"Tool parser {parser_name} not found in {PARSER_REGISTRY}"
    return PARSER_REGISTRY[parser_name]
