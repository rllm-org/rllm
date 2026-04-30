import json

from rllm.parser import DeepseekQwenChatTemplateParser, QwenChatTemplateParser


class DummyQwenTokenizer:
    name_or_path = "Qwen/Qwen3-4B-Instruct-2507"
    bos_token = "<|endoftext|>"
    eos_token = "<|endoftext|>"

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        rendered = ""
        for message in messages:
            rendered += f"<|im_start|>{message['role']}\n{message.get('content', '')}<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


class DummyDeepseekQwenTokenizer:
    name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    bos_token = "<｜begin▁of▁sentence｜>"
    eos_token = "<｜end▁of▁sentence｜>"

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        rendered = ""
        for message in messages:
            rendered += f"{message.get('content', '')}"
        if add_generation_prompt:
            rendered += "<｜Assistant｜>"
        return rendered


def _completion_text(content: str = "Use the calculator."):
    tool_call = {"name": "calculate", "arguments": {"expression": "1+1"}}
    return f"{content}<tool_call>\n{json.dumps(tool_call)}\n</tool_call><|im_end|>\n"


def test_qwen_multi_turn_extension_preserves_tool_call_prefix():
    parser = QwenChatTemplateParser(DummyQwenTokenizer(), multi_turn_extension=True)
    completion = _completion_text()
    parsed = parser.parse_completion_text(completion)

    rerendered = parser.parse_assistant(parsed, accumulate_reasoning=True)

    assert rerendered == parser.assistant_token + completion


def test_qwen_default_render_canonicalizes_tool_call_separator():
    parser = QwenChatTemplateParser(DummyQwenTokenizer(), multi_turn_extension=False)
    completion = _completion_text()
    parsed = parser.parse_completion_text(completion)

    rerendered = parser.parse_assistant(parsed, accumulate_reasoning=True)

    assert rerendered != parser.assistant_token + completion
    assert "Use the calculator.\n<tool_call>" in rerendered


def _deepseek_completion_text(content: str = "Use the calculator."):
    return (
        "<think>\nreasoning\n</think>"
        f"{content}"
        "<｜tool▁calls▁begin｜>"
        "\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>calculate\n"
        "```json\n"
        '{"expression": "1+1"}\n'
        "```\n"
        "<｜tool▁call▁end｜>\n"
        "<｜tool▁calls▁end｜>"
        "<｜end▁of▁sentence｜>"
    )


def test_deepseek_qwen_multi_turn_extension_preserves_tool_call_separator():
    parser = DeepseekQwenChatTemplateParser(DummyDeepseekQwenTokenizer(), multi_turn_extension=True)
    parsed = parser.parse_completion_text(_deepseek_completion_text())

    rerendered = parser.parse_assistant(parsed, accumulate_reasoning=True)

    assert "Use the calculator.<｜tool▁calls▁begin｜>" in rerendered


def test_deepseek_qwen_default_render_canonicalizes_tool_call_separator():
    parser = DeepseekQwenChatTemplateParser(DummyDeepseekQwenTokenizer(), multi_turn_extension=False)
    parsed = parser.parse_completion_text(_deepseek_completion_text())

    rerendered = parser.parse_assistant(parsed, accumulate_reasoning=True)

    assert "Use the calculator.\n<｜tool▁calls▁begin｜>" in rerendered
