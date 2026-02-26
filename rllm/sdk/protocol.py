from typing import Any

from pydantic import BaseModel, Field

from rllm.types import Step as StepView


class LLMInput(BaseModel):
    messages: list[dict]
    prompt_token_ids: list[int]


class LLMOutput(BaseModel):
    message: dict
    finish_reason: str
    output_token_ids: list[int]
    rollout_logprobs: None | list[float] = None


class Trace(BaseModel):
    """
    A trace is a dictionary with the following structure:

    {
        # Core LLM call information
        "name": str,              # e.g., "proxy/gpt-4"
        "model": str,             # e.g., "gpt-4", "claude-3-opus"
        "trace_id": str,          # e.g., "tr_abc123def456"
        "timestamp": float,       # Unix timestamp

        # Input/Output
        "input": {
            "messages": list[dict]  # OpenAI-style messages array
        },
        "output": {
            "choices": [
                {
                    "message": {
                        "content": str,      # Response text
                        "reasoning": str,    # Optional reasoning (for o1 models)
                        "role": str,         # Usually "assistant"
                    },
                    "finish_reason": str,    # e.g., "stop", "length"
                    "provider_specific_fields": {
                        "token_ids": list[int]  # Completion token IDs (vLLM only)
                    }
                }
            ],
            "prompt_token_ids": list[int],  # Prompt token IDs (vLLM only)
            # ... other OpenAI response fields
        },

        # Metadata
        "metadata": {
            "session_name": str,  # Format: "task_id:rollout_idx:retry_attempt"
            "job": str,           # Optional job identifier
            # ... other custom metadata from middleware
        },

        # Performance metrics
        "latency_ms": float,
        "tokens": {
            "prompt": int,
            "completion": int,
            "total": int
        },

        # Optional fields
        "session_name": str,    # Same as metadata.session_name
        "contexts": list,       # Context elements used
        "tools": list[dict],    # Available tools
        "cost": float,          # USD cost
        "environment": str,     # e.g., "production"
    }
    """

    trace_id: str
    session_name: str
    name: str
    input: LLMInput
    output: LLMOutput
    model: str
    latency_ms: float
    tokens: dict[str, int]
    metadata: dict = Field(default_factory=dict)
    timestamp: float
    parent_trace_id: str | None = None
    cost: float | None = None
    environment: str | None = None
    tools: list[dict] | None = None
    contexts: list[str | dict] | None = None
    tags: list[str] | None = None


def trace_to_step_view(trace: Trace) -> StepView:
    """Convert a trace to a StepView (trace wrapper with reward field)."""
    if hasattr(trace.input, "model_dump"):
        input_payload: Any = trace.input.model_dump()
    else:
        input_payload = trace.input

    if hasattr(trace.output, "model_dump"):
        output_payload: Any = trace.output.model_dump()
    else:
        output_payload = trace.output

    return StepView(
        id=trace.trace_id,
        input=input_payload,
        output=output_payload,
        reward=0.0,
        metadata=trace.metadata,
    )
