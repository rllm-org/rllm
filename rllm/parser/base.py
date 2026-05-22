from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypedDict

from rllm.tools.tool_base import ToolCall

ChatMessage = dict[str, Any]


@dataclass(slots=True)
class RenderedPrompt:
    """Rendered prompt representation produced from chat messages."""

    token_ids: list[int]
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.token_ids)


class ParsedCompletion(TypedDict):
    content: str
    reasoning: str
    tool_calls: list[ToolCall]


class BaseChatParser(ABC):
    """Interface for rendering prompts and parsing completion tokens."""

    @abstractmethod
    def render_messages(
        self,
        messages: list[ChatMessage],
        *,
        add_generation_prompt: bool = False,
        is_first_msg: bool = False,
        **kwargs,
    ) -> RenderedPrompt:
        """Render chat messages to token IDs and optional text."""

    @abstractmethod
    def parse_completion(self, completion_ids: list[int], **kwargs) -> ParsedCompletion:
        """Parse completion token IDs into an output assistant message."""
