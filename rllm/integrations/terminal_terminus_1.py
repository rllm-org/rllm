from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

from terminal_bench.agents.terminus_1 import (
    Command,
    CommandBatchResponse,
    Terminus,
)
from terminal_bench.llms.chat import Chat
from terminal_bench.terminal.tmux_session import TmuxSession


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