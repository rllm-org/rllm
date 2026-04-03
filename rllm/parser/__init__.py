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


_CHAT_TEMPLATE_CLASSES = {
    "ChatTemplateParser",
    "DeepseekQwenChatTemplateParser",
    "LlamaChatTemplateParser",
    "QwenChatTemplateParser",
}


def __getattr__(name):
    if name in _CHAT_TEMPLATE_CLASSES:
        import importlib
        mod = importlib.import_module("rllm.parser.chat_template_parser")
        return getattr(mod, name)
    if name == "TinkerChatTemplateParser":
        from rllm.parser.tinker_parser import TinkerChatTemplateParser
        return TinkerChatTemplateParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
