import json
import logging

import torch

from rllm.parser.chat_template_parser import ChatTemplateParser
from rllm.tools.tool_base import Tool, ToolCall

logger = logging.getLogger(__name__)


def _check_tinker_cookbook():
    try:
        import tinker_cookbook.renderers  # noqa: F401
    except ImportError:
        raise ImportError("tinker-cookbook is required for TinkerChatTemplateParser. Install it with: pip install tinker-cookbook") from None


def _make_render_context(idx, is_last, prev_message=None, last_user_index=-1):
    """Create a RenderContext, handling version differences in tinker-cookbook."""
    from tinker_cookbook.renderers.base import RenderContext

    try:
        return RenderContext(
            idx=idx,
            is_last=is_last,
            prev_message=prev_message,
            last_user_index=last_user_index,
        )
    except TypeError:
        # Older tinker-cookbook without last_user_index field
        return RenderContext(idx=idx, is_last=is_last, prev_message=prev_message)


class TinkerChatTemplateParser(ChatTemplateParser):
    """ChatTemplateParser that delegates to a tinker-cookbook Renderer.

    This allows users who have tinker-cookbook installed to use any tinker
    renderer through rllm's ChatTemplateParser interface, avoiding the need
    to write a manual parser for each model family.

    Example::

        from tinker_cookbook import renderers, tokenizer_utils
        from rllm.parser import TinkerChatTemplateParser

        tokenizer = tokenizer_utils.get_tokenizer("Qwen/Qwen3-8B")
        renderer = renderers.get_renderer("qwen3", tokenizer)
        parser = TinkerChatTemplateParser(renderer)

        prompt = parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
    """

    def __init__(self, renderer):
        _check_tinker_cookbook()

        from tinker_cookbook.renderers.base import Renderer

        if not isinstance(renderer, Renderer):
            raise TypeError(f"Expected a tinker_cookbook Renderer, got {type(renderer)}")

        self.renderer = renderer
        self.tokenizer = renderer.tokenizer
        self.processor = None

        # Compute generation_prompt by decoding the generation suffix tokens
        ctx = _make_render_context(idx=0, is_last=True)
        suffix_tokens = self.renderer._get_generation_suffix("assistant", ctx)
        self.generation_prompt = self.tokenizer.decode(suffix_tokens) if suffix_tokens else ""

        self.stop_sequences = self.renderer.get_stop_sequences()

    def _convert_message(self, msg, accumulate_reasoning=False):
        """Convert an rllm message dict to a tinker Message dict."""
        tinker_msg = {"role": msg["role"]}

        content = msg.get("content", "") or ""
        reasoning = (msg.get("reasoning", "") or "").strip()

        # Build structured content when reasoning or images are present
        if reasoning and accumulate_reasoning:
            parts = []
            parts.append({"type": "thinking", "thinking": reasoning})
            if content:
                parts.append({"type": "text", "text": content})
            tinker_msg["content"] = parts
        elif isinstance(msg.get("images"), list) and msg["images"]:
            parts = []
            for img in msg["images"]:
                parts.append({"type": "image", "image": img})
            if content:
                # Strip leading <image> tag if present (rllm convention)
                if content.startswith("<image>"):
                    content = content[len("<image>") :]
                parts.append({"type": "text", "text": content})
            tinker_msg["content"] = parts
        else:
            tinker_msg["content"] = content

        # Convert tool_calls to tinker ToolCall format
        if msg.get("tool_calls"):
            from tinker_cookbook.renderers.base import ToolCall as TinkerToolCall

            tool_calls = []
            for tc in msg["tool_calls"]:
                if isinstance(tc, ToolCall):
                    # rllm ToolCall dataclass
                    args = tc.arguments if isinstance(tc.arguments, str) else json.dumps(tc.arguments)
                    tool_calls.append(
                        TinkerToolCall(
                            function=TinkerToolCall.FunctionBody(name=tc.name, arguments=args),
                        )
                    )
                elif isinstance(tc, dict) and "function" in tc:
                    func = tc["function"]
                    args = func.get("arguments", "{}")
                    if not isinstance(args, str):
                        args = json.dumps(args)
                    tool_calls.append(
                        TinkerToolCall(
                            function=TinkerToolCall.FunctionBody(name=func["name"], arguments=args),
                            id=tc.get("id"),
                        )
                    )
                elif isinstance(tc, dict) and "name" in tc:
                    args = tc.get("arguments", "{}")
                    if not isinstance(args, str):
                        args = json.dumps(args)
                    tool_calls.append(
                        TinkerToolCall(
                            function=TinkerToolCall.FunctionBody(name=tc["name"], arguments=args),
                            id=tc.get("id"),
                        )
                    )
            if tool_calls:
                tinker_msg["tool_calls"] = tool_calls

        # Handle tool response fields
        if msg["role"] == "tool":
            if "tool_call_id" in msg:
                tinker_msg["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                tinker_msg["name"] = msg["name"]

        return tinker_msg

    def _convert_messages(self, messages, accumulate_reasoning=False):
        """Convert a list of rllm message dicts to tinker Message format."""
        return [self._convert_message(m, accumulate_reasoning) for m in messages]

    def _convert_tools(self, tools):
        """Convert rllm tools to tinker ToolSpec format."""
        tool_specs = []
        for tool in tools:
            if isinstance(tool, Tool):
                # rllm Tool object - extract from json property
                tool_json = tool.json
                if "function" in tool_json:
                    func = tool_json["function"]
                    tool_specs.append(
                        {
                            "name": func["name"],
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {}),
                        }
                    )
            elif isinstance(tool, dict):
                if "function" in tool:
                    func = tool["function"]
                    tool_specs.append(
                        {
                            "name": func["name"],
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {}),
                        }
                    )
                elif "name" in tool:
                    tool_specs.append(
                        {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        }
                    )
        return tool_specs

    def _render_to_tokens(self, tinker_messages, add_bos=False, add_generation_prompt=False):
        """Render tinker messages to a flat list of token IDs."""
        import tinker

        chunks = []

        if add_bos and self.renderer._bos_tokens:
            chunks.append(tinker.EncodedTextChunk(tokens=self.renderer._bos_tokens))

        last_user_idx = max(
            (i for i, m in enumerate(tinker_messages) if m["role"] == "user"),
            default=-1,
        )

        for idx, msg in enumerate(tinker_messages):
            ctx = _make_render_context(
                idx=idx,
                is_last=(idx == len(tinker_messages) - 1) and not add_generation_prompt,
                prev_message=tinker_messages[idx - 1] if idx > 0 else None,
                last_user_index=last_user_idx,
            )
            rendered = self.renderer.render_message(msg, ctx)
            if rendered.header:
                chunks.append(rendered.header)
            chunks.extend(x for x in rendered.output if not isinstance(x, tinker.EncodedTextChunk) or x.tokens)

        if add_generation_prompt:
            suffix_ctx = _make_render_context(
                idx=len(tinker_messages),
                is_last=True,
                prev_message=tinker_messages[-1] if tinker_messages else None,
                last_user_index=last_user_idx,
            )
            suffix_tokens = self.renderer._get_generation_suffix("assistant", suffix_ctx)
            if suffix_tokens:
                chunks.append(tinker.EncodedTextChunk(tokens=suffix_tokens))

        # Flatten chunks to token list
        tokens = []
        for chunk in chunks:
            if isinstance(chunk, tinker.EncodedTextChunk):
                tokens.extend(chunk.tokens)
            else:
                # ImageChunk or other non-token chunk - use length as placeholder
                # This path is for VL models; decode will produce placeholder tokens
                tokens.extend([0] * chunk.length)

        return tokens

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, tools=None, accumulate_reasoning=False, **kwargs):
        """Parse messages into a prompt string.

        Args:
            messages: List of rllm message dicts.
            add_generation_prompt: Whether to append the generation prompt.
            is_first_msg: Whether this is the first message (adds BOS token).
            tools: Optional list of tools to include in the prompt.
            accumulate_reasoning: Whether to include reasoning/thinking content.

        Returns:
            The rendered prompt string.
        """
        if not messages:
            return ""

        tinker_messages = self._convert_messages(messages, accumulate_reasoning=accumulate_reasoning)

        # Handle tools by prepending tool context messages
        if tools:
            tool_specs = self._convert_tools(tools)
            if tool_specs:
                try:
                    # Extract system prompt if first message is system
                    system_prompt = ""
                    if tinker_messages and tinker_messages[0]["role"] == "system":
                        content = tinker_messages[0]["content"]
                        if isinstance(content, str):
                            system_prompt = content
                        tinker_messages = tinker_messages[1:]
                    prefix = self.renderer.create_conversation_prefix_with_tools(tool_specs, system_prompt)
                    tinker_messages = prefix + tinker_messages
                except NotImplementedError:
                    logger.warning(f"Renderer {type(self.renderer).__name__} does not support tool calling. Tools will be ignored.")

        tokens = self._render_to_tokens(tinker_messages, add_bos=is_first_msg, add_generation_prompt=add_generation_prompt)
        result = self.tokenizer.decode(tokens, skip_special_tokens=False)

        # Tinker puts the \n separator in the next message's header, so the last
        # message lacks a trailing \n. HF templates always include it. Add it to
        # match HF's apply_chat_template output.
        if result and not result.endswith("\n"):
            result += "\n"

        return result

    def parse_completion(self, completion_ids: list[int]) -> dict[str, str | list]:
        """Parse completion token IDs into structured output.

        Args:
            completion_ids: List of token IDs from model generation.

        Returns:
            Dict with 'content', 'reasoning', and 'tool_calls' keys.
        """
        parsed_msg, _success = self.renderer.parse_response(completion_ids)

        content = ""
        reasoning = ""
        tool_calls = []

        msg_content = parsed_msg.get("content", "")
        if isinstance(msg_content, str):
            content = msg_content
        elif isinstance(msg_content, list):
            text_parts = []
            thinking_parts = []
            for part in msg_content:
                if part["type"] == "text":
                    text_parts.append(part["text"])
                elif part["type"] == "thinking":
                    thinking_parts.append(part["thinking"])
            content = "".join(text_parts)
            reasoning = "".join(thinking_parts)

        # Convert tinker ToolCall objects to rllm ToolCall dataclass
        if parsed_msg.get("tool_calls"):
            for tc in parsed_msg["tool_calls"]:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = tc.function.arguments
                tool_calls.append(ToolCall(name=tc.function.name, arguments=args))

        return {
            "content": content.strip(),
            "reasoning": reasoning.strip(),
            "tool_calls": tool_calls,
        }

    def tokenize_and_mask(self, messages):
        """Convert messages to token IDs with loss masks using tinker's supervised example builder.

        Returns:
            Tuple of (prompt_ids, response_ids, response_mask) as torch tensors.
        """
        from tinker_cookbook.renderers.base import TrainOnWhat

        tinker_messages = self._convert_messages(messages, accumulate_reasoning=True)
        model_input, weights = self.renderer.build_supervised_example(tinker_messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)

        all_tokens = model_input.to_ints()
        weights_list = weights.tolist()

        # Split at first non-zero weight
        boundary = next((i for i, w in enumerate(weights_list) if w > 0), len(weights_list))

        prompt_ids = torch.tensor(all_tokens[:boundary], dtype=torch.long)
        response_ids = torch.tensor(all_tokens[boundary:], dtype=torch.long)
        response_mask = weights[boundary:].long()

        return prompt_ids, response_ids, response_mask

    def tokenize_and_mask_cumulative(self, messages):
        """Convert multi-turn messages to token IDs with cumulative loss masks.

        Returns:
            Tuple of (prompt_ids, response_ids, response_mask) as torch tensors.
        """
        from tinker_cookbook.renderers.base import TrainOnWhat

        tinker_messages = self._convert_messages(messages, accumulate_reasoning=True)
        model_input, weights = self.renderer.build_supervised_example(tinker_messages, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES)

        all_tokens = model_input.to_ints()
        weights_list = weights.tolist()

        # Split at first non-zero weight
        boundary = next((i for i, w in enumerate(weights_list) if w > 0), len(weights_list))

        prompt_ids = torch.tensor(all_tokens[:boundary], dtype=torch.long)
        response_ids = torch.tensor(all_tokens[boundary:], dtype=torch.long)
        response_mask = weights[boundary:].long()

        return prompt_ids, response_ids, response_mask
