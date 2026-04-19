from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_chat_parser(
    tokenizer,
    processor=None,
    *,
    parser_backend: str = "rllm",
    reasoning_parser_name: str | None = None,
    tool_parser_name: str | None = None,
    renderer_name: str | None = None,
    **kwargs,
):
    """Create a chat parser for the given backend.

    Args:
        tokenizer: HuggingFace tokenizer.
        processor: Optional multimodal processor (used by rllm backend).
        parser_backend: Which backend's native parsers to use.
            "rllm" (default) uses ChatTemplateParser with auto-detection.
            "vllm" uses vLLM's ReasoningParserManager + ToolParserManager.
            "sglang" uses SGLang's ReasoningParser + FunctionCallParser.
            "tinker" uses Tinker's Renderer for rendering and parsing.
        reasoning_parser_name: Name of the reasoning parser to use (vllm/sglang).
            For vLLM: e.g. "deepseek_r1", "qwen3", "kimi_k2".
            For SGLang: e.g. "deepseek-r1", "qwen3", "kimi".
        tool_parser_name: Name of the tool parser to use (vllm/sglang).
            For vLLM: e.g. "deepseek_v3", "qwen3_coder", "kimi_k2".
            For SGLang: e.g. "deepseekv3", "qwen", "kimi_k2".
        renderer_name: Name of the Tinker renderer (tinker backend only).
            If None, auto-detects from the model name.
        **kwargs: Additional keyword arguments passed to the backend.

    Returns:
        A parser object with parse() and parse_completion().
            TinkerParser also exposes stop_sequences (needed by Tinker's sampling client).
    """
    if parser_backend == "rllm":
        from rllm.parser.chat_template_parser import ChatTemplateParser

        return ChatTemplateParser.get_parser(
            tokenizer,
            processor=processor,
            disable_thinking=kwargs.get("disable_thinking", False),
        )

    if parser_backend == "vllm":
        from rllm.experimental.parser.vllm_parser import VLLMParser

        return VLLMParser(
            tokenizer,
            reasoning_parser_name=reasoning_parser_name,
            tool_parser_name=tool_parser_name,
            processor=processor,
            **kwargs,
        )

    if parser_backend == "sglang":
        from rllm.experimental.parser.sglang_parser import SGLangParser

        return SGLangParser(
            tokenizer,
            reasoning_parser_name=reasoning_parser_name,
            tool_parser_name=tool_parser_name,
            processor=processor,
            **kwargs,
        )

    if parser_backend == "tinker":
        from rllm.experimental.parser.tinker_parser import TinkerParser

        return TinkerParser(
            tokenizer,
            renderer_name=renderer_name,
            **kwargs,
        )

    raise ValueError(f"Unknown parser_backend: {parser_backend!r}. Expected one of: rllm, vllm, sglang, tinker")
