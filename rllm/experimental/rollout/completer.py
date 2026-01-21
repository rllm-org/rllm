"""
Implementation of an abstract `Completer` class that works with rLLM workflows to simplify the construction
of a single step from prompt-response interactions with the rollout engine.

We further implements a `TITOCompleter` that ensures the "token-in-token-out" property.

The name `completer` is inspired by `tinker_cookbook`.
"""

from collections.abc import Callable
from dataclasses import field
from typing import Any

from transformers import PreTrainedTokenizer

from rllm.agents.agent import Step
from rllm.experimental.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.experimental.rollout.types import TokenInput, TokenOutput
from rllm.parser import ChatTemplateParser


class Completer:
    """
    Basic completer that takes in messages and returns a single step.

    Args:
        rollout_engine: The rollout engine to use.
        action_hook: A hook to transform the model output into an action.
        kwargs: Additional kwargs to pass to the rollout engine.
    Returns:
        A single step with most information filled in.

    Examples:
    - Usage in solver-judge workflow:
        >>> completer = Completer(rollout_engine)
        >>> action_hook = lambda model_output: self._parse_solver_response(model_output.content)
        >>> step = await completer.complete(messages, action_hook=action_hook)
    """

    rollout_engine: RolloutEngine

    def __init__(self, rollout_engine: RolloutEngine):
        self.rollout_engine = rollout_engine

    async def complete(self, messages: list[dict], action_hook: Callable[[ModelOutput], Any] | None = None, **kwargs) -> Step:
        """Complete the messages and return a single step."""
        model_output: ModelOutput = await self.rollout_engine.get_model_response(messages, **kwargs)

        # construct the step
        chat_completions = messages + [{"role": "assistant", "content": model_output.content or "", "reasoning": model_output.reasoning or ""}]
        action = action_hook(model_output) if action_hook is not None else None
        return Step(
            prompt_ids=model_output.prompt_ids or [],  # type: ignore
            response_ids=model_output.completion_ids or [],
            logprobs=model_output.logprobs or [],
            chat_completions=chat_completions,
            thought=model_output.reasoning or "",
            action=action,
            model_output=model_output,  # type: ignore
        )


class TITOCompleter(Completer):
    """
    Completer that ensures the "token-in-token-out" property.
    """

    chat_parser: ChatTemplateParser
    tokenizer: PreTrainedTokenizer
    # stateful data taht this completer tracks over `complete` calls
    _prev_messages: list[dict] = field(default_factory=list)
    _prev_messages_str: str = ""  # the messages after applying chat template
    _prev_token_input: TokenInput = field(default_factory=list)

    def __init__(self, rollout_engine: RolloutEngine):
        super().__init__(rollout_engine)
        # we need to ensure that the rollout engine supports token-in-token-out
        if not self.rollout_engine.supports_token_in_token_out:
            cls_name = self.rollout_engine.__class__.__name__
            raise ValueError(f"The rollout engine {cls_name} does not support token-in-token-out")
        # we also require the rollout engine has a chat parser
        chat_parser = getattr(self.rollout_engine, "chat_parser", None)
        if not chat_parser or not isinstance(chat_parser, ChatTemplateParser):
            raise ValueError("The rollout engine must have a chat parser")

        tokenizer = getattr(self.rollout_engine, "tokenizer", None)
        if not tokenizer or not isinstance(tokenizer, PreTrainedTokenizer):
            raise ValueError("The rollout engine must have a tokenizer")
        self.tokenizer = tokenizer
        self.chat_parser = chat_parser

    def _parse_message_delta(self, messages: list[dict]) -> tuple[bool, TokenInput]:
        cur_messages_str = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
        # check if the previous message string is a prefix of the current message string
        if len(self._prev_messages_str) > 0 and self._prev_messages_str.startswith(cur_messages_str):
            message_str_delta = cur_messages_str[len(self._prev_messages_str) :]
            is_prefix = True
        else:
            message_str_delta = cur_messages_str
            is_prefix = False

        token_input_delta: list[int] = self.tokenizer.encode(message_str_delta, add_special_tokens=False)
        return is_prefix, token_input_delta

    async def complete(self, messages: list[dict], action_hook: Callable[[ModelOutput], Any] | None = None, **kwargs) -> Step:
        is_prefix, token_input_delta = self._parse_message_delta(messages)

        curr_token_input = self._prev_token_input + token_input_delta
        curr_token_output: TokenOutput = await self.rollout_engine.get_token_output_from_token_input(curr_token_input, **kwargs)

        """
        Part below can be refactored cleaner by changing the rollout engine API
        """
        response_tokens, logprobs = curr_token_output.tokens, curr_token_output.logprobs

        response_dict = self.chat_parser.parse_completion(response_tokens)
        content = response_dict.get("content", "")
        reasoning = response_dict.get("reasoning", "")
        tool_calls = response_dict.get("tool_calls", [])

        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        model_output = ModelOutput(
            prompt_ids=curr_token_input,
            completion_ids=response_tokens,
            logprobs=logprobs or [],
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
        )
        """
        Part above can be refactored cleaner by changing the rollout engine API
        """

        action = action_hook(model_output) if action_hook is not None else None
        return Step(
            prompt_ids=curr_token_input,
            response_ids=response_tokens,
            logprobs=logprobs or [],
            chat_completions=messages + [{"role": "assistant", "content": content, "reasoning": reasoning}],
            thought=reasoning,
            action=action,
            model_response=response_text,
            model_output=model_output,
        )
