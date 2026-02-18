"""Helper for computing per-token distillation advantages on a Step."""

import logging
from collections.abc import Callable

from rllm.agents.agent import Step

logger = logging.getLogger(__name__)


async def compute_step_distill_advantage(
    step: Step,
    teacher_engine,
    student_tokenizer=None,
    teacher_tokenizer=None,
    shared_tokenizer: bool = False,
    teacher_chat_parser=None,
    teacher_prompt_fn: Callable[[list[dict]], list[dict]] | None = None,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> list[float]:
    """
    Compute per-token distillation advantages for a single Step.

    Queries the teacher for logprobs on the same completion, aligns them to student
    tokens, and computes advantages = teacher_logprob - student_logprob.

    Args:
        step: Step with populated prompt_ids, response_ids, logprobs, and chat_completions.
        teacher_engine: Engine with compute_logprobs(ids: list[int]) -> list[float].
        student_tokenizer: Student tokenizer (required when shared_tokenizer=False).
        teacher_tokenizer: Teacher tokenizer (required when shared_tokenizer=False).
        shared_tokenizer: If True, student and teacher share the same tokenizer.
        teacher_chat_parser: Chat parser for the teacher (required when shared_tokenizer=False or teacher_prompt_fn is set).
        teacher_prompt_fn: Optional function (prompt_messages) -> transformed_messages.
            Transforms prompt messages for the teacher, e.g., to inject ground-truth
            for privileged-context distillation (OPSD).
        clip_min: Optional lower bound for clipping advantages.
        clip_max: Optional upper bound for clipping advantages.

    Returns:
        Per-token advantages.
    """
    student_prompt_ids = step.prompt_ids
    student_completion_ids = step.response_ids
    student_logprobs = step.logprobs

    if not student_prompt_ids:
        raise ValueError("Missing prompt_ids on step for distillation.")
    if not student_completion_ids or not student_logprobs:
        raise ValueError("Missing response_ids or logprobs on step for distillation.")

    if shared_tokenizer and teacher_prompt_fn is None:
        # Fast path: directly use student token IDs for teacher query
        teacher_ids = student_prompt_ids + student_completion_ids
        teacher_prompt_length = len(student_prompt_ids)
        teacher_full_logprobs = await teacher_engine.compute_logprobs(teacher_ids)
        aligned_teacher_logprobs = teacher_full_logprobs[teacher_prompt_length:]

    elif shared_tokenizer and teacher_prompt_fn is not None:
        # Same tokenizer but different teacher prompt — re-encode prompt, no alignment needed
        if teacher_chat_parser is None or teacher_tokenizer is None:
            raise ValueError("teacher_chat_parser and teacher_tokenizer are required when teacher_prompt_fn is set.")
        if not step.chat_completions:
            raise ValueError("Missing chat_completions on step for distillation.")

        teacher_prompt_messages = teacher_prompt_fn(step.chat_completions[:-1])

        teacher_prompt = teacher_chat_parser.parse(
            teacher_prompt_messages,
            is_first_msg=True,
            add_generation_prompt=True,
            tools=[],
            accumulate_reasoning=False,
        )
        teacher_prompt_ids = teacher_tokenizer.encode(teacher_prompt, add_special_tokens=False)
        teacher_ids = teacher_prompt_ids + student_completion_ids
        teacher_full_logprobs = await teacher_engine.compute_logprobs(teacher_ids)
        aligned_teacher_logprobs = teacher_full_logprobs[len(teacher_prompt_ids):]

    else:
        # Different tokenizers: re-encode through teacher chat parser and align
        from rllm.trainer.distill import align_teacher_logprobs

        if teacher_chat_parser is None:
            raise ValueError("teacher_chat_parser is required when shared_tokenizer=False.")
        if teacher_tokenizer is None or student_tokenizer is None:
            raise ValueError("Both student_tokenizer and teacher_tokenizer are required when shared_tokenizer=False.")
        if not step.chat_completions:
            raise ValueError("Missing chat_completions on step for cross-tokenizer distillation.")

        teacher_prompt_messages = step.chat_completions[:-1]
        if teacher_prompt_fn is not None:
            teacher_prompt_messages = teacher_prompt_fn(teacher_prompt_messages)
        teacher_completion_messages = step.chat_completions[-1:]

        reasoning_str = teacher_completion_messages[0].get("reasoning", "")
        content_str = teacher_completion_messages[0].get("content", "")
        if not reasoning_str and not content_str:
            # Nothing to align — zero advantage
            return [0.0] * len(student_logprobs)

        # Build teacher prompt and completion token IDs
        teacher_prompt = teacher_chat_parser.parse(
            teacher_prompt_messages,
            is_first_msg=True,
            add_generation_prompt=True,
            tools=[],
            accumulate_reasoning=False,
        )
        teacher_prompt_ids = teacher_tokenizer.encode(teacher_prompt, add_special_tokens=False)

        teacher_completion = teacher_chat_parser.parse(
            teacher_completion_messages,
            is_first_msg=False,
            add_generation_prompt=False,
            tools=[],
            accumulate_reasoning=True,
        )
        if teacher_completion.startswith(teacher_chat_parser.generation_prompt):
            teacher_completion = teacher_completion[len(teacher_chat_parser.generation_prompt):]
        teacher_completion_ids = teacher_tokenizer.encode(teacher_completion, add_special_tokens=False)

        # Query teacher for logprobs
        teacher_ids = teacher_prompt_ids + teacher_completion_ids
        teacher_full_logprobs = await teacher_engine.compute_logprobs(teacher_ids)
        teacher_logprobs = teacher_full_logprobs[len(teacher_prompt_ids):]

        # Align teacher logprobs to student tokens
        aligned_teacher_logprobs = align_teacher_logprobs(
            student_ids=student_completion_ids,
            student_tokenizer=student_tokenizer,
            teacher_ids=teacher_completion_ids,
            teacher_tokenizer=teacher_tokenizer,
            teacher_logprobs=teacher_logprobs,
            student_logprobs=student_logprobs,
            reasoning_str=reasoning_str,
            content_str=content_str,
        )

    # Compute per-token advantages: teacher_logprob - student_logprob
    advantages = [t_lp - s_lp for t_lp, s_lp in zip(aligned_teacher_logprobs, student_logprobs, strict=False)]

    # Optionally clip
    if clip_min is not None or clip_max is not None:
        lo = clip_min if clip_min is not None else float("-inf")
        hi = clip_max if clip_max is not None else float("inf")
        advantages = [max(lo, min(hi, adv)) for adv in advantages]

    return advantages
