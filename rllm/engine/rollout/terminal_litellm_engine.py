from __future__ import annotations
from typing import Any

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.integrations.terminal_terminus_1 import RLLMTerminus

from terminal_bench.llms.base_llm import ContextLengthExceededError, OutputLengthExceededError
from terminal_bench.llms.lite_llm import LiteLLM
from terminal_bench.llms.chat import Chat


class TerminalLiteLLMEngine(RolloutEngine):
    """
    Rollout engine that routes chat completions through Terminal-Bench's LiteLLM.

    - Accepts OpenAI-style chat message lists (list of {role, content}).
    - Expects the last message to be a user turn; uses it as the 'prompt' and
      the preceding messages as 'message_history' to match Terminal-Bench Chat semantics.
    - Supports structured outputs via `response_format` (either a Pydantic model class or schema dict).
    - Returns ModelOutput with token counts computed via Terminal-Bench's token counter.
    """

    def __init__(self, model: str, tokenizer=None, api_base: str | None = None, sampling_params: dict | None = None, max_episodes: int = 50, logging_dir: str | None = None, **kwargs: Any):
        # Align with RolloutEngine interface
        self.model = model
        self.tokenizer = tokenizer  # not used (Terminal-Bench handles counting)
        self.api_base = api_base
        self.sampling_params = sampling_params or {}

        # Underlying Terminal-Bench LLM wrapper (stateless across calls)
        self._llm = LiteLLM(model_name=model, api_base=api_base)
        self._terminus = RLLMTerminus(model_name=model, max_episodes=max_episodes, api_base=api_base)
        self._max_episodes = max_episodes
        
    async def get_model_response(self, messages: list[dict], **kwargs: Any) -> ModelOutput:
        # Remove kwargs used only by other engines/workflows
        kwargs.pop("application_id", None)
        kwargs.pop("validate", None)

        sampling_params = self.sampling_params.copy()
        sampling_params.update(kwargs)

        # Extract structured output spec if provided
        response_format = sampling_params.pop("response_format", None)

        # Expect the last message to be the user prompt
        assert messages and messages[-1]["role"] == "user", "Last message must be a user turn"
        prompt = messages[-1]["content"]
        message_history = messages[:-1]

        # Create per-call Chat to avoid cross-thread state contamination
        chat = Chat(self._llm)
        chat._messages = list(message_history)

        # Compute prompt token count on the exact call payload
        prompt_messages_for_count = message_history + [{"role": "user", "content": prompt}]
        prompt_tokens = self._llm.count_tokens(prompt_messages_for_count)

        # No logging paths (disable per-call file logging here)
        logging_paths = (None, None, None)
        # Delegate to Terminus' handle_llm_interaction with a per-call agent
        try:
           
            parsed = self._terminus.handle_llm_interaction(
                chat=chat,
                prompt=prompt,
                logging_paths=logging_paths,
            )
            finish_reason = "stop"
        except OutputLengthExceededError as e:
            # TB signals truncation via exception; assistant text may be logged in chat
            finish_reason = "length"
        # Other exceptions (e.g., ContextLengthExceededError) propagate to workflow

        # Assistant raw text is last assistant message in Chat history
        assistant_text = next((m["content"] for m in reversed(chat._messages) if m["role"] == "assistant"), "")

        # Compute completion token count from returned text
        completion_tokens = self._llm.count_tokens([{"role": "assistant", "content": assistant_text}])

        return ModelOutput(
            text=assistant_text,
            tool_calls=[],
            finish_reason=finish_reason,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
        )
