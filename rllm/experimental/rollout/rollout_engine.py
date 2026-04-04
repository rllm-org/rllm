import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rllm.experimental.rollout.types import TokenInput, Tokenizer, TokenOutput
    from rllm.parser import ChatTemplateParser
    from rllm.tools.tool_base import ToolCall

logger = logging.getLogger(__name__)


@dataclass
class ModelOutput:
    text: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] | None = None
    prompt_ids: TokenInput | None = None
    completion_ids: list[int] | None = None
    multi_modal_inputs: dict[str, list] | None = None
    logprobs: list[float] | None = None  # completion logprobs
    prompt_logprobs: list[float] | None = None  # prompt logprobs aligned to prompt_ids
    routing_matrices: list[str] | None = None  # per-token routing matrices (R3, transient)
    prompt_length: int = 0
    completion_length: int = 0
    finish_reason: str | None = None
    weight_version: int | None = None  # policy version at time of generation
    metrics: dict | None = None  # per-turn server metrics (e.g. ttft, queue durations)

    def to_dict(self):
        return {
            "text": self.text,
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls] if self.tool_calls else [],
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
            "multi_modal_inputs": self.multi_modal_inputs,
            "logprobs": self.logprobs,
            "prompt_logprobs": self.prompt_logprobs,
            "prompt_length": self.prompt_length,
            "completion_length": self.completion_length,
            "finish_reason": self.finish_reason,
            "weight_version": self.weight_version,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict):
        from rllm.tools.tool_base import ToolCall

        return cls(
            text=data.get("text"),
            content=data.get("content"),
            reasoning=data.get("reasoning"),
            tool_calls=[ToolCall(**tool_call) for tool_call in data.get("tool_calls", [])] if data.get("tool_calls") else None,
            prompt_ids=data.get("prompt_ids"),
            completion_ids=data.get("completion_ids"),
            multi_modal_inputs=data.get("multi_modal_inputs"),
            logprobs=data.get("logprobs"),
            prompt_logprobs=data.get("prompt_logprobs"),
            prompt_length=data.get("prompt_length", 0),
            completion_length=data.get("completion_length", 0),
            finish_reason=data.get("finish_reason"),
            weight_version=data.get("weight_version"),
            metrics=data.get("metrics"),
        )


class RolloutEngine:
    chat_parser: ChatTemplateParser | None = None
    tokenizer: Tokenizer | None = None
    is_validation: bool = False  # flag enabled/disabled by AgentWorkflowEngine.execute_tasks

    def __init__(self, *args, **kwargs):
        # Gate mechanism for pausing model calls during weight sync
        self._gate: asyncio.Event = asyncio.Event()
        self._gate.set()  # open by default
        self._active_calls: int = 0
        self._drained_event: asyncio.Event = asyncio.Event()
        self._drained_event.set()  # initially drained (no active calls)
        self.weight_version: int = 0

    # --- Gate mechanism ---

    def close_gate(self) -> None:
        """Close the gate. New model calls will block at wait_for_gate()."""
        logger.info(f"[RolloutEngine] Closing gate. Active calls: {self._active_calls}")
        self._gate.clear()

    def open_gate(self) -> None:
        """Open the gate, releasing any blocked model calls."""
        logger.info(f"[RolloutEngine] Opening gate. Active calls: {self._active_calls}")
        self._gate.set()

    def on_model_call_complete(self) -> None:
        """Unregister active call. Engines will call this at the END of get_model_response()."""
        self._active_calls -= 1
        if self._active_calls <= 0:
            self._active_calls = 0
            self._drained_event.set()
            logger.debug("[RolloutEngine] All active calls drained.")
        else:
            logger.debug(f"[RolloutEngine] Model call complete. Active calls: {self._active_calls}")

    async def wait_for_gate(self) -> None:
        """Wait until gate is open, then register as active call. Engines will call this at the START of get_model_response()."""
        if not self._gate.is_set():
            logger.info(f"[RolloutEngine] Waiting for gate to open. Active calls: {self._active_calls}")
        await self._gate.wait()
        self._active_calls += 1
        self._drained_event.clear()
        logger.debug(f"[RolloutEngine] Gate passed. Active calls: {self._active_calls}")

    async def wait_for_drain(self) -> None:
        """Wait until all active model calls complete. Used during weight sync."""
        if not self._drained_event.is_set():
            logger.info(f"[RolloutEngine] Waiting for drain. Active calls: {self._active_calls}")
        await self._drained_event.wait()

    # --- Model response ---
    async def _get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        raise NotImplementedError(f"_get_model_response is not implemented for {self.__class__.__name__}")

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        await self.wait_for_gate()
        try:
            weight_version = self.weight_version
            result = await self._get_model_response(messages, **kwargs)
            result.weight_version = weight_version
            return result
        finally:
            self.on_model_call_complete()

    def assemble_model_output(self, token_input: TokenInput, token_output: TokenOutput) -> ModelOutput:
        """
        Assemble model output from a token output.
        """
        raise NotImplementedError("assemble_model_output is not implemented")

    async def get_token_output_from_token_input(self, token_input: TokenInput, **kwargs) -> TokenOutput:
        """Obtain the token output from the given token input."""
        raise NotImplementedError("get_token_output_from_token_input is not implemented")

    @property
    def supports_token_in_token_out(self) -> bool:
        """Whether the engine supports token-in-token-out (TITO) generation. Defaults to false."""
        return False

    async def wake_up(self):
        pass

    async def sleep(self):
        pass
