from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from terminal_bench.agents.terminus_1 import (
    Command,
    CommandBatchResponse,
    Terminus,
)
from terminal_bench.llms.lite_llm import LiteLLM
from terminal_bench.llms.chat import Chat
from terminal_bench.terminal.tmux_session import TmuxSession

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine

class RLLMTerminus(Terminus):
    """
    rLLM integration subclass for Terminal Bench's `Terminus` agent.

    This class exposes public wrappers around selected private methods/fields of
    the upstream `Terminus` class, enabling rLLM workflows to leverage the
    agent's internal building blocks (prompt construction, command execution,
    and LLM interaction) without modifying the third-party source.
    """

    def __init__(
        self,
        model_name: str,
        max_episodes: int = 50,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_episodes=max_episodes,
            api_base=api_base,
            **kwargs,
        )

    # Public methods exposing private Terminus methods

    def handle_llm_interaction(
        self,
        chat: Chat,
        prompt: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
    ) -> CommandBatchResponse:
        """Public wrapper for the internal LLM interaction handler."""
        # Uses the upstream protected method directly
        return self._handle_llm_interaction( 
            chat=chat,
            prompt=prompt,
            logging_paths=logging_paths,
        )

    def build_initial_prompt(self, instruction: str, terminal_state: str) -> str:
        """
        Build the initial prompt without executing the agent loop.

        Mirrors the original formatting logic used by `perform_task`.
        """
        # Access upstream private fields for formatting consistency
        return self._prompt_template.format(  
            response_schema=self._response_schema,  
            instruction=instruction,
            history="",
            terminal_state=terminal_state,
        )

    def execute_commands(
        self,
        commands: list[Command],
        session: TmuxSession,
    ) -> Tuple[bool, str]:
        """
        Execute commands and return formatted output.

        Returns a tuple of (timeout_occurred, formatted_output).
        """
        return self._execute_commands(commands, session)  

    def get_response_schema(self) -> str:
        """Get the JSON schema for the response format."""
        return self._response_schema  

    def get_prompt_template(self) -> str:
        """Get the prompt template string."""
        return self._prompt_template  

    def get_timeout_template(self) -> str:
        """Get the timeout prompt template."""
        return self._timeout_template  

    def format_timeout_prompt(
        self,
        instruction: str,
        history: str,
        terminal_state: str,
    ) -> str:
        """Format the prompt after a timeout occurrence."""
        return self._timeout_template.format(  
            response_schema=self._response_schema,  
            instruction=instruction,
            history=history,
            terminal_state=terminal_state,
        )


class TerminalLiteLLMEngine(RolloutEngine):
    """Minimal rollout engine delegating to Terminal-Bench's LiteLLM + Chat.

    Args:
        model: LLM model identifier.
        tokenizer: Optional tokenizer (unused; Terminal-Bench handles counting).
        api_base: Optional base URL for the LLM API.
        sampling_params: Optional dict of generation parameters.
        max_episodes: Max steps used to configure the Terminus helper.
        logging_dir: Optional path to write logs (unused at per-call level).
        **kwargs: Reserved for future configuration.
    """

    def __init__(self, model: str, tokenizer=None, api_base: str | None = None, sampling_params: dict | None = None, max_episodes: int = 50, logging_dir: str | None = None, **kwargs: Any):
        self.model = model
        self.tokenizer = tokenizer  # Unused; Terminal-Bench handles token counting
        self.api_base = api_base
        self.sampling_params = sampling_params or {}

        self._llm = LiteLLM(model_name=model, api_base=api_base)
        self._terminus = RLLMTerminus(model_name=model, max_episodes=max_episodes, api_base=api_base)
        
    async def get_model_response(self, messages: list[dict], **kwargs: Any) -> ModelOutput:
        """Get a chat completion via Terminal-Bench's Chat abstraction.

        Expects ``messages`` in OpenAI-style format and returns a ``ModelOutput``
        containing the assistant text and token accounting computed by LiteLLM.

        Args:
            messages: List of role-content message dictionaries.
            **kwargs: Unused; present for API compatibility.

        Returns:
            ModelOutput: Assistant text and token counts.
        """
        # Expect the last message to be the user prompt
        assert messages and messages[-1]["role"] == "user", "Last message must be a user turn"
        prompt = messages[-1]["content"]
        message_history = messages[:-1]

        # Create per-call Chat to avoid cross-thread state contamination
        chat = Chat(self._llm)
        chat._messages = list(message_history)

        # Token counts
        prompt_messages_for_count = message_history + [{"role": "user", "content": prompt}]
        prompt_tokens = self._llm.count_tokens(prompt_messages_for_count)

        # Disable per-call file logging
        logging_paths = (None, None, None)
        # Delegate to Terminus' handle_llm_interaction with a per-call agent
        self._terminus.handle_llm_interaction(
            chat=chat,
            prompt=prompt,
            logging_paths=logging_paths,
        )

        # Assistant raw text is last assistant message in Chat history
        assistant_text = next((m["content"] for m in reversed(chat._messages) if m["role"] == "assistant"), "")

        # Completion tokens
        completion_tokens = self._llm.count_tokens([{"role": "assistant", "content": assistant_text}])

        return ModelOutput(
            text=assistant_text,
            tool_calls=[],
            finish_reason="stop",
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
        )