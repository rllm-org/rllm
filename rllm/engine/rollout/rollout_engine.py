from dataclasses import dataclass

from rllm.tools.tool_base import ToolCall


@dataclass
class ModelOutput:
    text: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] | None = None
    prompt_ids: list[int] | None = None
    completion_ids: list[int] | None = None
    multi_modal_inputs: dict[str, list] | None = None
    logprobs: list[float] | None = None
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
            prompt_length=data.get("prompt_length", 0),
            completion_length=data.get("completion_length", 0),
            finish_reason=data.get("finish_reason"),
        )


class RolloutEngine:
    def __init__(self, *args, **kwargs):
        pass

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        raise NotImplementedError("get_model_response is not implemented")

    async def wake_up(self):
        pass

    async def sleep(self):
        pass
