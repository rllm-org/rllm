"""Convert gateway TraceRecord to training-compatible Step."""

from __future__ import annotations

from rllm_model_gateway.models import TraceRecord

from rllm.agents.agent import Step
from rllm.experimental.rollout import ModelOutput


def trace_record_to_step(trace: TraceRecord) -> Step:
    """Convert a gateway TraceRecord to a training Step.

    TraceRecord has clean top-level fields from vLLM:
    - prompt_token_ids
    - completion_token_ids
    - logprobs (per-token)
    """
    content = trace.response_message.get("content", "") or ""
    reasoning = trace.response_message.get("reasoning", "") or ""

    model_output = ModelOutput(
        content=content,
        reasoning=reasoning,
        prompt_ids=trace.prompt_token_ids,
        completion_ids=trace.completion_token_ids,
        logprobs=trace.logprobs or [],
        prompt_length=len(trace.prompt_token_ids),
        completion_length=len(trace.completion_token_ids),
        finish_reason=trace.finish_reason,
    )

    # Build chat_completions: input messages + assistant response
    chat_completions = list(trace.messages)
    chat_completions.append(trace.response_message)

    return Step(
        id=trace.trace_id,
        chat_completions=chat_completions,
        model_output=model_output,
        model_response=content,
        thought=reasoning,
        metadata=trace.metadata,
    )
