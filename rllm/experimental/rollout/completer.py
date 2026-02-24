"""
Implementation of an abstract `Completer` class that works with rLLM workflows to simplify the construction
of a single step from prompt-response interactions with the rollout engine.

We further implements a `TITOCompleter` that ensures the "token-in-token-out" property.

The name `completer` is inspired by `tinker_cookbook`.
"""

from collections.abc import Callable
from dataclasses import field
from typing import Any

from rllm.agents.agent import Step
from rllm.experimental.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.experimental.rollout.types import TokenInput, Tokenizer, TokenOutput
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
        action = action_hook(model_output) if action_hook is not None else None
        return Step.from_model_output(model_output, messages, action)  # type: ignore

    def reset(self):
        """Reset the completer to its initial state."""
        return  # nothing to do for the basic completer


class TITOCompleter(Completer):
    """
    Completer that ensures the "token-in-token-out" property. This is achieved by caching the previous messages and token input, and when
    a new message contains the previous messages as a prefix, we only compute the token ids for the "delta" (difference) part of the new message.
    And the new token id is the concatenation of the previous token id and the "delta" token id.

    Note that using this completer will automatically accumulate the reasoning of the assistant messages.

    Args:
        rollout_engine: The rollout engine to use.
        kwargs: Additional kwargs to pass to the rollout engine.
    Returns:
        A single step with most information filled in.
    """

    chat_parser: ChatTemplateParser
    tokenizer: Tokenizer
    # stateful data taht this completer tracks over `complete` calls
    _prev_messages_str: str = ""  # the messages after applying chat template
    _prev_token_input: TokenInput = field(default_factory=list)
    _n_completions: int = 0
    _n_prefixes: int = 0

    def __init__(self, rollout_engine: RolloutEngine):
        super().__init__(rollout_engine)
        # we need to ensure that the rollout engine supports token-in-token-out
        if not self.rollout_engine.supports_token_in_token_out:
            cls_name = self.rollout_engine.__class__.__name__
            raise ValueError(f"The rollout engine {cls_name} does not support token-in-token-out")
        # we also require the rollout engine has a chat parser and a tokenizer
        if rollout_engine.chat_parser is None or rollout_engine.tokenizer is None:
            raise ValueError("The rollout engine must have a chat parser and a tokenizer. For Tinker engine, make sure you have set bypass_render_with_parser=True.")
        self.tokenizer = rollout_engine.tokenizer
        self.chat_parser = rollout_engine.chat_parser

    def _parse_message_delta(self, messages: list[dict]) -> tuple[bool, TokenInput]:
        cur_messages_str = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, accumulate_reasoning=True)
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
        self._prev_messages_str = self.chat_parser.parse(messages, add_generation_prompt=True, is_first_msg=True, accumulate_reasoning=True)
        # backend-specific handling for retrieving the completion ids
        if hasattr(curr_token_output, "token_ids"):  # Verl
            curr_completion_ids: list[int] = curr_token_output.token_ids  # type: ignore[assignment]
        elif hasattr(curr_token_output, "tokens"):  # Tinker
            curr_completion_ids: list[int] = curr_token_output.tokens
        else:
            raise ValueError(f"Unsupported token output type: {type(curr_token_output)}")
        # update the number of completions and prefixes
        self._n_completions += 1
        self._n_prefixes += int(is_prefix)
        self._prev_token_input = curr_token_input + curr_completion_ids
        return Step.from_model_output(model_output, messages, action)  # type: ignore

    def reset(self):
        """Reset the completer to its initial state."""
        self._prev_messages_str = ""
        self._prev_token_input = []
        self._n_completions = 0
        self._n_prefixes = 0

    @property
    def n_completions(self) -> int:
        return self._n_completions

    @property
    def n_prefixes(self) -> int:
        return self._n_prefixes

    @property
    def prefix_ratio(self) -> float:
        return self._n_prefixes / self._n_completions if self._n_completions > 0 else 0.0
