from typing import Any

from pydantic import BaseModel, Field


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


class StepView(BaseModel):
    """
    A concise view of a single LLM call (trace) with reward.

    StepView is essentially a trace wrapper that adds a reward field.

    Fields:
        - id: Trace ID, unique per trace, can be used to retrieve the full trace from the store
        - input: LLM input (from trace)
        - output: LLM response (from trace)
        - action: Parsed action (set manually by user)
        - reward: Step reward
        - metadata: Additional tracking data (can include model, tokens, latency, etc.)
    """

    id: str
    input: Any | None = None  # Serialized LLM input
    output: Any | None = None  # Serialized LLM output
    action: Any | None = None
    reward: float = 0.0
    metadata: dict | None = None


class TrajectoryView(BaseModel):
    """
    A view of a trajectory.

    Represents a collection of steps (each step = 1 trace)
    Each trace in the trajectory is automatically converted to a StepView.

    Hierarchy:
        TrajectoryView → StepView (1 trace each)

    Fields:
        - name: Trajectory name
        - steps: List of StepViews (auto-generated from traces)
        - reward: Trajectory reward (set manually)
        - input: Function arguments (dict)
        - output: Function return value (Any)
        - metadata: Additional tracking data
    """

    name: str = "agent"
    steps: list[StepView] = Field(default_factory=list)
    reward: float = 0.0
    input: dict | None = None  # Function arguments
    output: Any = None  # Function return value
    metadata: dict | None = None  # Additional tracking data

    @property
    def result(self):
        """Get the output from the trajectory (backward compatibility)."""
        return self.output


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
