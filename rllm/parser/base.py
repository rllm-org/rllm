"""Common parser contract shared by the chat-template and renderer backends.

rLLM converts conversation messages into model input tokens in two ways:

* :class:`~rllm.parser.chat_template_parser.ChatTemplateParser` — messages ->
  string (via the model's chat template) -> token ids (via the tokenizer).
* :class:`~rllm.parser.renderer_parser.RendererParser` — messages -> token
  ids directly, via the external ``renderers`` package.

``BaseParser`` is the contract both honor so a rollout engine can hold a
parser without caring which backend produced it. ``ParserSession`` builds on
that contract to drive an efficient multi-turn rollout: it caches the previous
turn's prompt/completion token ids and extends them with
``bridge_to_next_turn`` instead of re-rendering the whole conversation each
step.

This module is intentionally dependency-free (no torch / transformers /
renderers) so ``import rllm.parser`` stays cheap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rllm.tools.tool_base import ToolCall

# A conversation message. Mirrors the OpenAI chat shape: a required ``role``
# and ``content``, plus optional ``reasoning`` / ``reasoning_content``,
# ``tool_calls`` and tool-result keys. Kept as a bare ``dict`` alias so the
# contract does not force a particular message class on callers.
Message = dict[str, Any]


@dataclass
class ParsedCompletion:
    """Structured form of a model completion, backend-agnostic.

    ``tool_calls`` holds :class:`rllm.tools.tool_base.ToolCall` instances so
    consumers get the same type regardless of which parser produced them.

    Supports mapping-style access (``pc["content"]`` / ``pc.get("reasoning")``)
    in addition to attribute access. Legacy callers that consumed the old
    ``ChatTemplateParser.parse_completion`` ``dict`` keep working unchanged.
    """

    content: str = ""
    reasoning: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    # Keys exposed through the mapping-style accessors below.
    _FIELDS = ("content", "reasoning", "tool_calls")

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
        }

    def __getitem__(self, key: str) -> Any:
        if key not in self._FIELDS:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key) if key in self._FIELDS else default

    def __contains__(self, key: object) -> bool:
        return key in self._FIELDS


class BaseParser(ABC):
    """Message <-> token conversion contract for a single model family.

    A parser is stateless with respect to any one rollout — the same
    instance may serve many concurrent rollouts. Per-rollout token state
    (needed for multi-turn extension) lives in :class:`ParserSession`,
    obtained via :meth:`new_session`.
    """

    @abstractmethod
    def render(
        self,
        messages: list[Message],
        *,
        tools: list[Any] | None = None,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> list[int]:
        """Render ``messages`` to model input token ids.

        With ``add_generation_prompt=True`` the result ends at the point
        where the next assistant turn begins generating. ``**kwargs`` carries
        backend-specific render options (e.g. ``accumulate_reasoning`` /
        ``reasoning_effort`` for the chat-template backend); a backend
        ignores options it does not recognise.
        """

    @abstractmethod
    def parse_completion(self, completion_ids: list[int]) -> ParsedCompletion:
        """Parse model-sampled ``completion_ids`` into a structured message."""

    @abstractmethod
    def get_stop_token_ids(self) -> list[int]:
        """Token ids that signal the assistant turn is complete."""

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[Any] | None = None,
    ) -> list[int] | None:
        """Extend a rollout by one turn without re-rendering sampled tokens.

        Returns the next turn's full prompt token ids — which start with
        ``previous_prompt_ids + previous_completion_ids`` verbatim and end
        at the next assistant generation point — or ``None`` if the backend
        cannot prove that extension is byte-safe, in which case the caller
        falls back to a full :meth:`render`.

        Re-rendering model-sampled history through a chat template silently
        breaks token identity (boolean round-trips, BPE retokenization
        drift, dropped reasoning blocks); a true bridge sidesteps all of
        that by reusing the sampled tokens.

        The base implementation returns ``None`` (no bridge support).
        Backends capable of token-level extension override this.
        """
        return None

    def new_session(self, *, tools: list[Any] | None = None) -> ParserSession:
        """Create a per-rollout :class:`ParserSession` bound to this parser."""
        return ParserSession(self, tools=tools)


class ParserSession:
    """Per-rollout multi-turn token-state cache.

    One session corresponds to one rollout/trajectory. It is **not** thread
    safe and must not be shared across concurrent rollouts — create one per
    rollout via :meth:`BaseParser.new_session`. The parser itself is
    stateless and may be shared freely.

    Lifecycle::

        session = parser.new_session(tools=tools)
        prompt_ids = session.start(messages)        # turn 0 prompt
        # ... model samples completion_ids ...
        session.observe_completion(completion_ids)
        prompt_ids = session.advance(env_messages)  # turn 1 prompt
        # ... repeat observe_completion / advance per turn ...

    :meth:`advance` uses :meth:`BaseParser.bridge_to_next_turn` when the
    backend supports it, so the model-sampled tokens of every prior turn are
    reused verbatim rather than re-rendered. When the bridge declines (the
    ``DefaultRenderer`` fallback, or any chat-template parser), it falls back
    to a full re-render and records that in :attr:`stats`.
    """

    def __init__(self, parser: BaseParser, *, tools: list[Any] | None = None):
        self.parser = parser
        self.tools = tools
        self.messages: list[Message] = []
        self.prompt_ids: list[int] = []
        self._completion_ids: list[int] | None = None
        self._started = False
        # Observability: lets callers confirm the bridge is actually being
        # taken rather than silently falling back to full re-renders.
        self.stats: dict[str, int] = {"turns": 0, "bridged": 0, "rerendered": 0}

    def start(self, messages: list[Message]) -> list[int]:
        """Render the turn-0 prompt and seed the session cache."""
        if self._started:
            raise RuntimeError("ParserSession.start() already called; create one session per rollout")
        self.messages = list(messages)
        self.prompt_ids = self.parser.render(self.messages, tools=self.tools, add_generation_prompt=True)
        self._started = True
        self._completion_ids = None
        return self.prompt_ids

    def observe_completion(
        self,
        completion_ids: list[int],
        assistant_message: Message | None = None,
    ) -> None:
        """Record the tokens the model sampled for the current turn.

        ``assistant_message`` is the structured assistant turn to append to
        the conversation history (used only if a later turn has to fall back
        to a full re-render). When omitted it is reconstructed via
        :meth:`BaseParser.parse_completion`.
        """
        if not self._started:
            raise RuntimeError("call start() before observe_completion()")
        if self._completion_ids is not None:
            raise RuntimeError("observe_completion() already called this turn; call advance() next")
        self._completion_ids = list(completion_ids)
        if assistant_message is None:
            parsed = self.parser.parse_completion(self._completion_ids)
            assistant_message = {"role": "assistant", "content": parsed.content}
            if parsed.reasoning:
                assistant_message["reasoning_content"] = parsed.reasoning
            if parsed.tool_calls:
                assistant_message["tool_calls"] = parsed.tool_calls
        self.messages.append(assistant_message)

    def advance(self, new_messages: list[Message]) -> list[int]:
        """Append ``new_messages`` (tool results, next user turn, ...) and
        return the next turn's prompt token ids.

        Uses the parser's bridge when available; otherwise re-renders the
        full conversation. ``new_messages`` must not contain an assistant
        turn — bridges refuse those, which forces a full re-render.
        """
        if self._completion_ids is None:
            raise RuntimeError("call observe_completion() before advance()")
        new_messages = list(new_messages)
        bridged = self.parser.bridge_to_next_turn(
            self.prompt_ids,
            self._completion_ids,
            new_messages,
            tools=self.tools,
        )
        self.messages.extend(new_messages)
        if bridged is not None:
            self.prompt_ids = bridged
            self.stats["bridged"] += 1
        else:
            self.prompt_ids = self.parser.render(self.messages, tools=self.tools, add_generation_prompt=True)
            self.stats["rerendered"] += 1
        self.stats["turns"] += 1
        self._completion_ids = None
        return self.prompt_ids

    @property
    def token_ids(self) -> list[int]:
        """Full token sequence so far: current prompt plus, if a completion
        has been observed but not yet advanced past, that completion."""
        if self._completion_ids is None:
            return list(self.prompt_ids)
        return self.prompt_ids + self._completion_ids
