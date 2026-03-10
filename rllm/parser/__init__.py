from rllm.parser.chat_template_parser import ChatTemplateParser, DeepseekQwenChatTemplateParser, LlamaChatTemplateParser, QwenChatTemplateParser
from rllm.parser.tool_parser import QwenToolParser, R1ToolParser, ToolParser

__all__ = [
    "ChatTemplateParser",
    "DeepseekQwenChatTemplateParser",
    "QwenChatTemplateParser",
    "LlamaChatTemplateParser",
    "TinkerChatTemplateParser",
    "ToolParser",
    "R1ToolParser",
    "QwenToolParser",
]

PARSER_REGISTRY = {
    "r1": R1ToolParser,
    "qwen": QwenToolParser,
}


def get_tool_parser(parser_name: str) -> type[ToolParser]:
    assert parser_name in PARSER_REGISTRY, f"Tool parser {parser_name} not found in {PARSER_REGISTRY}"
    return PARSER_REGISTRY[parser_name]


def __getattr__(name):
    if name == "TinkerChatTemplateParser":
        from rllm.parser.tinker_parser import TinkerChatTemplateParser

        return TinkerChatTemplateParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
