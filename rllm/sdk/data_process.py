import logging
import uuid
from collections import defaultdict

from rllm.agents.agent import Step, Trajectory
from rllm.engine.rollout import ModelOutput

logger = logging.getLogger(__name__)


def trace_to_model_output(trace: dict) -> ModelOutput:
    output = trace.get("output", {})

    # Extract prompt token IDs (vLLM-specific field)
    prompt_ids = output.get("prompt_token_ids", [])

    # Extract response choices (OpenAI-compatible format)
    choices = output.get("choices", [])

    # Extract message content and reasoning from first choice
    content = choices[0].get("message", {}).get("content", "")
    reasoning = choices[0].get("message", {}).get("reasoning", "")

    # Extract completion token IDs from provider-specific fields (vLLM only)
    provider_specific_fields = choices[0].get("provider_specific_fields", {})
    completion_ids = provider_specific_fields.get("token_ids", [])

    # Validate required fields
    assert output, trace
    assert len(choices) == 1, "Only one choice is supported for now"
    assert prompt_ids, "Prompt IDs are required"
    assert completion_ids, "Completion IDs are required"

    return ModelOutput(
        text="",  # Not used in current implementation
        content=content,
        reasoning=reasoning,
        tool_calls=[],  # TODO: Extract tool calls from message if present
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        logprobs=[],  # TODO: Extract logprobs if available
        prompt_length=len(prompt_ids),
        completion_length=len(completion_ids),
        finish_reason=choices[0].get("finish_reason", "stop"),
    )


def trace_to_step(trace: dict) -> Step:
    # Extract input messages (conversation history before this LLM call)
    messages = trace.get("input", {}).get("messages", [])

    # Extract response message (the LLM's response)
    response_message = trace.get("output", {}).get("choices", [])[0].get("message", {})

    assert response_message, "Response message is required in trace output"

    return Step(
        chat_completions=messages + [response_message],
        model_output=trace_to_model_output(trace),
        info=trace.get("metadata", {}),
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
