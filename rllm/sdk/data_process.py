import json
import logging
import uuid
from collections import defaultdict

from rllm.agents.agent import Step, Trajectory
from rllm.engine.rollout import ModelOutput
from rllm.sdk.protocol import LLMInput, LLMOutput, Trace

logger = logging.getLogger(__name__)


def _extract_prompt_token_ids(output_payload: dict) -> list[int]:
    """
    Extract prompt token IDs from the output payload root level.
    Per protocol.py: output["prompt_token_ids"] is the canonical location.
    """
    prompt_ids = output_payload.get("prompt_token_ids")
    if prompt_ids is None:
        return []
    return list(prompt_ids)


def _extract_completion_token_ids(output_payload: dict) -> list[int]:
    completion_ids = output_payload.get("choices")[0].get("provider_specific_fields", {}).get("token_ids")
    if completion_ids is None:
        return []
    return list(completion_ids)


def _extract_logprobs(output_payload: dict) -> list[float]:
    """
    Extract logprobs from the output payload.

    Prioritizes compact response_logprobs (from vLLM instrumentation) over
    verbose OpenAI format logprobs for efficiency.
    """
    choices = output_payload.get("choices")
    if not choices:
        return []

    choice = choices[0]
    provider_fields = choice.get("provider_specific_fields", {})
    response_logprobs = provider_fields.get("response_logprobs")

    if response_logprobs is not None:
        logprobs_list = list(response_logprobs)
        return logprobs_list

    # Fallback to parsing OpenAI format logprobs
    logprobs_obj = choice.get("logprobs")
    if logprobs_obj is None:
        logger.debug("⚠️ [DATA_PROCESS] No logprobs found in response (neither response_logprobs nor logprobs)")
        return []
    logprobs = logprobs_obj.get("content")
    if logprobs is None:
        logger.debug("⚠️ [DATA_PROCESS] logprobs object found but no 'content' field")
        return []
    return [float(entry.get("logprob")) for entry in logprobs if entry and entry.get("logprob") is not None]


def build_llm_output(payload: dict) -> LLMOutput:
    """Normalize raw OpenAI-style output payloads into LLMOutput."""
    if not isinstance(payload, dict):
        raise TypeError(f"LLM output must be dict or LLMOutput, got {type(payload)}")

    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("LLM output payload missing 'choices'")
    choice = choices[0]

    token_ids = _extract_completion_token_ids(payload)
    rollout_logprobs = _extract_logprobs(payload)

    return LLMOutput(
        message=choice.get("message") or {},
        finish_reason=choice.get("finish_reason"),
        output_token_ids=token_ids,
        rollout_logprobs=rollout_logprobs,
    )


def build_llm_io(input_payload: dict, output_payload: dict) -> tuple[LLMInput, LLMOutput]:
    """Normalize raw OpenAI input/output payloads into structured LLMInput/LLMOutput."""
    llm_output = build_llm_output(output_payload)
    prompt_token_ids = _extract_prompt_token_ids(output_payload)
    llm_input = LLMInput(messages=input_payload.get("messages") or [], prompt_token_ids=prompt_token_ids)
    return llm_input, llm_output


def trace_to_model_output(trace: Trace) -> ModelOutput:
    """Convert stored Trace protocol to ModelOutput."""
    input_block = trace.input
    output_block = trace.output

    prompt_ids = input_block.prompt_token_ids
    completion_ids = output_block.output_token_ids

    content = output_block.message.get("content", "")
    reasoning = output_block.message.get("reasoning", "")
    tool_calls = output_block.message.get("tool_calls", [])
    finish_reason = output_block.finish_reason or "stop"

    assert prompt_ids, "Prompt IDs are required"
    assert completion_ids, "Completion IDs are required"

    return ModelOutput(
        text="",
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=output_block.rollout_logprobs or [],
        prompt_length=len(prompt_ids),
        completion_length=len(completion_ids),
        finish_reason=finish_reason,
    )


def trace_to_step(trace: Trace) -> Step:
    """Convert stored Trace protocol to Step."""
    messages = trace.input.messages
    response_message = trace.output.message
    assert response_message, "Response message is required in trace output"

    return Step(
        chat_completions=messages + [response_message],
        model_output=trace_to_model_output(trace),
        info=trace.metadata,
    )


def get_trajectory_name(steps: list[Step], name_key: str | None = None) -> str:
    if name_key is None:
        return "agent"
    else:
        return steps[0].info.get(name_key, "agent")


def group_steps(steps: list[Step], by: str | None = None, name_key: str | None = None) -> list[Trajectory]:
    # if some step doesnt have the group key, we assign a random key to avoid grouping them together
    # in this case, the grpo reduce to reinforce
    if by is None:
        return [Trajectory(name="agent", steps=steps)]
    else:
        step_groups = defaultdict(list)
        for step in steps:
            step_groups[step.info.get(by, str(uuid.uuid4()))].append(step)
        return [Trajectory(name=get_trajectory_name(group_steps, name_key), steps=group_steps) for group_key, group_steps in step_groups.items()]


class SequenceAccumulator:
    def __init__(self):
        self.full_sequence = []
        self.logprobs = []
        self.advantages = []
        self.mask = []

    def is_empty(self):
        return len(self.full_sequence) == 0

    def clear(self):
        self.full_sequence = []
        self.logprobs = []
        self.advantages = []
        self.mask = []

    def add_step(self, step: Step, advantage: float, is_extension: bool = False):
        """Add a step to the accumulated sequence."""
        if is_extension:
            # Only add the new tokens (delta)
            prev_len = len(self.full_sequence)
            delta_prompt = step.prompt_ids[prev_len:]
            delta_prompt_len = len(delta_prompt)
        else:
            # Add entire prompt
            delta_prompt = step.prompt_ids
            delta_prompt_len = len(delta_prompt)

        # Add prompt tokens (observation)
        self.full_sequence.extend(delta_prompt)
        self.logprobs.extend([0.0] * delta_prompt_len)
        self.advantages.extend([0.0] * delta_prompt_len)
        self.mask.extend([0.0] * delta_prompt_len)

        # Add response tokens (action)
        self.full_sequence.extend(step.response_ids)
        self.logprobs.extend(step.logprobs)
        self.advantages.extend([advantage] * len(step.response_ids))
        self.mask.extend([1.0] * len(step.response_ids))


# def build_trajectories_from_steps(steps: List[Step]) -> list[Trajectory]:
#     """
#     Build one or more Datums from a trajectory, merging steps when possible.

#     Steps are merged when the next step's prompt is an extension of the
#     previous step's full sequence (prompt + response).

#     Args:
#         trajectory: Trajectory with steps
#         advantage: Advantage value for this trajectory

#     Returns:
#         List of Datum objects (may contain 1+ datums depending on merging)
#     """
#     if not steps:
#         return []

#     assert all(step.model_output is not None for step in steps), "model_output is None for some steps"
#     model_outputs = [step.model_output for step in steps]

#     # Build datums by iterating through steps
#     datums = []
#     accumulator = SequenceAccumulator()

#     for step_idx, step in enumerate(trajectory.steps):
#         if accumulator.is_empty():
#             # First step - start accumulating
#             accumulator.add_step(step, advantage, is_extension=False)
#         else:
#             # Check if current step extends previous sequence
#             prev_full_sequence = accumulator.full_sequence
#             current_prompt = step.prompt_ids

#             if TinkerDatumBuilder._is_prefix(prev_full_sequence, current_prompt):
#                 # Step extends previous - merge
#                 accumulator.add_step(step, advantage, is_extension=True)
#             else:
#                 # Step doesn't extend - create datum and start fresh
#                 datums.append(accumulator.to_datum())
#                 accumulator.clear()
#                 accumulator.add_step(step, advantage, is_extension=False)

#     # Create final datum from accumulated sequence
#     if not accumulator.is_empty():
#         datums.append(accumulator.to_datum())

#     return datums


def try_serialize(data):
    if isinstance(data, dict):
        serialized_data = {}
        for key, value in data.items():
            serialized_data[key] = try_serialize(value)
        return serialized_data
    elif getattr(data, "model_dump", None) is not None:
        return data.model_dump()
    elif isinstance(data, list):
        serialized_data = [try_serialize(item) for item in data]
        return serialized_data
    else:
        try:
            return json.dumps(data)
        except Exception:
            return "NOT_SERIALIZABLE"
