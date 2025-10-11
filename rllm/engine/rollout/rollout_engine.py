from dataclasses import dataclass

from rllm.tools.tool_base import ToolCall


@dataclass
class ModelOutput:
    text: str
    content: str
    reasoning: str
    tool_calls: list[ToolCall]
    prompt_ids: list[int]
    completion_ids: list[int]
    prompt_length: int
    completion_length: int
    finish_reason: str

    def to_dict(self):
        return {
            "text": self.text,
            "content": self.content,
            "reasoning": self.reasoning,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
            "prompt_ids": self.prompt_ids,
            "completion_ids": self.completion_ids,
            "prompt_length": self.prompt_length,
            "completion_length": self.completion_length,
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            text=data["text"],
            content=data["content"],
            reasoning=data["reasoning"],
            tool_calls=[ToolCall(**tool_call) for tool_call in data["tool_calls"]],
            prompt_ids=data["prompt_ids"],
            completion_ids=data["completion_ids"],
            prompt_length=data["prompt_length"],
            completion_length=data["completion_length"],
            finish_reason=data["finish_reason"],
        )


class RolloutEngine:
    def __init__(self, *args, **kwargs):
        pass

    async def get_model_response(self, messages: list[dict], **kwargs) -> ModelOutput:
        raise NotImplementedError("get_model_response is not implemented")

    def wake_up(self):
        pass

    def sleep(self):
        pass
