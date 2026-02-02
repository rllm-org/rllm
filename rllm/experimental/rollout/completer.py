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
    Completer that ensures the "token-in-token-out" property. This is achieved by caching the previous messages and token input, and when
    a new message contains the previous messages as a prefix, we only compute the token ids for the "delta" (difference) part of the new message.
    And the new token id is the concatenation of the previous token id and the "delta" token id.

    Args:
        rollout_engine: The rollout engine to use.
        kwargs: Additional kwargs to pass to the rollout engine.
    Returns:
        A single step with most information filled in.
    """

    chat_parser: ChatTemplateParser
    tokenizer: PreTrainedTokenizer
    # stateful data taht this completer tracks over `complete` calls
    _prev_messages_str: str = ""  # the messages after applying chat template
    _prev_token_input: TokenInput = field(default_factory=list)

    def __init__(self, rollout_engine: RolloutEngine):
        super().__init__(rollout_engine)
        # we need to ensure that the rollout engine supports token-in-token-out
        if not self.rollout_engine.supports_token_in_token_out:
            cls_name = self.rollout_engine.__class__.__name__
            raise ValueError(f"The rollout engine {cls_name} does not support token-in-token-out")
        # we also require the rollout engine has a chat parser and a tokenizer
        if rollout_engine.chat_parser is None or rollout_engine.tokenizer is None:
            raise ValueError("The rollout engine must have a chat parser and a tokenizer")
        self.tokenizer = rollout_engine.tokenizer
        self.chat_parser = rollout_engine.chat_parser

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

        # current token input should be the previous token input plus the token input delta
        curr_token_input = self._prev_token_input + token_input_delta
        curr_token_output: TokenOutput = await self.rollout_engine.get_token_output_from_token_input(curr_token_input, **kwargs)

        model_output = self.rollout_engine.assemble_model_output(curr_token_input, curr_token_output)

        action = action_hook(model_output) if action_hook is not None else None

        # update the previous messages and token input
        self._prev_messages_str = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True)
        self._prev_token_input = curr_token_input + curr_token_output.completion_ids

        return Step(
            prompt_ids=model_output.prompt_ids or [],  # type: ignore
            response_ids=model_output.completion_ids or [],
            logprobs=model_output.logprobs or [],
            chat_completions=messages + [{"role": "assistant", "content": model_output.content, "reasoning": model_output.reasoning}],
            thought=model_output.reasoning or "",
            action=action,
            model_response=model_output.content or "",
            model_output=model_output,  # type: ignore
        )
