from dataclasses import dataclass
from typing import Any, TypeAlias

from rllm.tools.tool_base import ToolCall

"""
Type alias for TokenOutput and TokenInput -- need to take different backends into account.
"""
# Tinker types. See https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/data_processing.py
# for the rationale behind "FlatObElem" and "FlatOb" types.
try:
    from tinker.types import ModelInput, ModelInputChunk, SampledSequence

    TinkerFlatObElem: TypeAlias = ModelInputChunk | int
    TinkerTokenOutput: TypeAlias = SampledSequence
except ImportError:  # avoid cases when the tinker backend is not used
    TinkerFlatObElem: TypeAlias = Any
    TinkerTokenOutput: TypeAlias = Any

TinkerFlatOb: TypeAlias = list[TinkerFlatObElem]
TinkerTokenInput: TypeAlias = ModelInput | TinkerFlatOb

# Verl types
VerlTokenInput: TypeAlias = list[int]
try:
    from verl.workers.rollout.replica import TokenOutput

    VerlTokenOutput: TypeAlias = TokenOutput
except ImportError:  # avoid cases when the verl backend is not used
    VerlTokenOutput: TypeAlias = Any

# Union everything together
TokenInput: TypeAlias = TinkerTokenInput | VerlTokenInput
TokenOutput: TypeAlias = TinkerTokenOutput | VerlTokenOutput


@dataclass
class ModelOutput:
    text: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] | None = None
    prompt_ids: TokenInput | None = None
    completion_ids: TokenOutput | None = None
    multi_modal_inputs: dict[str, list] | None = None
    logprobs: list[float] | None = None  # completion logprobs
    prompt_logprobs: list[float] | None = None  # prompt logprobs aligned to prompt_ids
    prompt_length: int = 0
    completion_length: int = 0
    finish_reason: str | None = None

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
        }

    @classmethod
    def from_dict(cls, data: dict):
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
        )


class RolloutEngine:
    def __init__(self, *args, **kwargs):
        pass

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        raise NotImplementedError("get_model_response is not implemented")

    async def get_token_output_from_token_input(self, token_input: TokenInput, **kwargs) -> TokenOutput:
        """Obtain the token output from the given token input."""
        raise NotImplementedError("get_token_output_from_token_input is not implemented")

    async def wake_up(self):
        pass

    async def sleep(self):
        pass

    @property
    def supports_token_in_token_out(self) -> bool:
        """Whether the engine supports token-in-token-out (TITO) generation. Defaults to false."""
        return False
