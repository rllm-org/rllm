from __future__ import annotations
from typing import Any

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.integrations.terminal_terminus_1 import RLLMTerminus

from terminal_bench.llms.lite_llm import LiteLLM
from terminal_bench.llms.chat import Chat


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
