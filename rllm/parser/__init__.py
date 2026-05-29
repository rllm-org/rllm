from rllm.parser.base import BaseChatParser, ChatMessage, ParsedCompletion, RenderedPrompt

__all__ = [
    "BaseChatParser",
    "ChatMessage",
    "ChatTemplateParser",
    "DeepseekQwenChatTemplateParser",
    "QwenChatTemplateParser",
    "LlamaChatTemplateParser",
    "ParsedCompletion",
    "RenderedPrompt",
    "SGLangParser",
    "TinkerParser",
    "VLLMParser",
    "get_chat_parser",
    "ToolParser",
    "R1ToolParser",
    "QwenToolParser",
]


def __getattr__(name):
    _lazy_classes = {
        "ChatTemplateParser": "rllm.parser.chat_template_parser",
        "DeepseekQwenChatTemplateParser": "rllm.parser.chat_template_parser",
        "LlamaChatTemplateParser": "rllm.parser.chat_template_parser",
        "QwenChatTemplateParser": "rllm.parser.chat_template_parser",
        "SGLangParser": "rllm.parser.sglang_parser",
        "TinkerParser": "rllm.parser.tinker_parser",
        "VLLMParser": "rllm.parser.vllm_parser",
        "ToolParser": "rllm.parser.tool_parser",
        "R1ToolParser": "rllm.parser.tool_parser",
        "QwenToolParser": "rllm.parser.tool_parser",
    }
    if name in _lazy_classes:
        import importlib

        mod = importlib.import_module(_lazy_classes[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_chat_parser(tokenizer, processor=None, image_processor=None, chat_template_config=None) -> BaseChatParser:
    cfg = chat_template_config or {}

    backend = cfg.get("parser_backend", "rllm")
    backend_init_kwargs = cfg.get("backend_init_kwargs", {}) or {}

    if backend in ("rllm", "vllm", "sglang"):
        chat_template_path = cfg.get("chat_template_path")
        if chat_template_path:
            from pathlib import Path

            tokenizer.chat_template = Path(chat_template_path).read_text()

    if backend == "rllm":
        from rllm.parser.chat_template_parser import ChatTemplateParser

        parser_kwargs = dict(backend_init_kwargs)
        parser_kwargs["accumulate_reasoning"] = cfg.get("accumulate_reasoning", False)
        parser_kwargs["reasoning_effort"] = cfg.get("reasoning_effort", "high")

        return ChatTemplateParser.get_parser(
            tokenizer,
            processor=processor,
            disable_thinking=cfg.get("disable_thinking", False),
            **parser_kwargs,
        )

    elif backend == "tinker":
        from rllm.parser.tinker_parser import TinkerParser

        return TinkerParser(
            tokenizer,
            renderer_name=cfg.get("renderer_name"),
            image_processor=image_processor,
            **backend_init_kwargs,
        )

    elif backend == "vllm":
        from rllm.parser.vllm_parser import VLLMParser

        return VLLMParser(
            tokenizer,
            reasoning_parser_name=cfg.get("reasoning_parser_name"),
            tool_parser_name=cfg.get("tool_parser_name"),
            processor=processor,
            **backend_init_kwargs,
        )

    elif backend == "sglang":
        from rllm.parser.sglang_parser import SGLangParser

        return SGLangParser(
            tokenizer,
            reasoning_parser_name=cfg.get("reasoning_parser_name"),
            tool_parser_name=cfg.get("tool_parser_name"),
            processor=processor,
            **backend_init_kwargs,
        )

    raise ValueError(f"Unknown chat parser backend {backend!r}. Expected one of: rllm, vllm, sglang, tinker")
