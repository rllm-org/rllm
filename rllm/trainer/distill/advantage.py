"""Helper for computing per-token distillation advantages on a Step."""

import logging
from collections.abc import Callable

from rllm.agents.agent import Step

logger = logging.getLogger(__name__)


def discounted_future_sum(values: list[float], discount_factor: float) -> list[float]:
    """Compute discounted future sum for a sequence of values.

    For each position i, computes: sum_{j=i}^{n-1} discount_factor^{j-i} * values[j]

    Args:
        values: Sequence of values.
        discount_factor: Discount factor (gamma), typically in [0, 1].

    Returns:
        Discounted future sums for each position.
    """
    if not values:
        return []
    n = len(values)
    result = [0.0] * n
    result[-1] = values[-1]
    for i in range(n - 2, -1, -1):
        result[i] = values[i] + discount_factor * result[i + 1]
    return result


def compute_distill_reverse_kl(
    teacher_logprobs: list[float],
    student_logprobs: list[float],
    clip_min: float,
    clip_max: float,
    kl_penalty_coef: float = 1.0,
    kl_discount_factor: float = 0.0,
) -> list[float]:
    """Compute per-token distillation advantages from teacher and student logprobs.

    Computes: advantage = kl_penalty_coef * (teacher_logprob - student_logprob)

    Applies clipping and optionally discounted future sum.

    Args:
        teacher_logprobs: Per-token log probabilities from teacher.
        student_logprobs: Per-token log probabilities from student.
        clip_min: Lower bound for clipping advantages (e.g., -5.0).
        clip_max: Upper bound for clipping advantages (e.g., 5.0).
        kl_penalty_coef: Optional coefficient for the KL penalty (default 1.0).
        kl_discount_factor: Optional discount factor for future sum (default 0.0, no discounting).

    Returns:
        Per-token advantages.
    """
    if len(teacher_logprobs) != len(student_logprobs):
        min_len = min(len(teacher_logprobs), len(student_logprobs))
        teacher_logprobs = teacher_logprobs[:min_len]
        student_logprobs = student_logprobs[:min_len]

    # Compute reverse KL: student_lp - teacher_lp, then negate with coefficient
    # This gives: advantage = kl_penalty_coef * (teacher_lp - student_lp)
    advantages = [kl_penalty_coef * (t_lp - s_lp) for t_lp, s_lp in zip(teacher_logprobs, student_logprobs, strict=True)]

    # Apply discounted future sum if requested
    if kl_discount_factor > 0:
        advantages = discounted_future_sum(advantages, kl_discount_factor)

    # Clip advantages
    advantages = [max(clip_min, min(clip_max, adv)) for adv in advantages]

    return advantages


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
    kl_penalty_coef: float = 1.0,
    kl_discount_factor: float = 0.0,
    visualize: bool = False,
) -> list[float]:
    """
    Compute per-token distillation advantages for a single Step.

    Queries the teacher for logprobs on the same completion, aligns them to student
    tokens, and computes advantages using compute_distill_reverse_kl.

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
        kl_penalty_coef: Coefficient for the KL penalty (default 1.0).
        kl_discount_factor: Discount factor for future sum (default 0.0, no discounting).
        visualize: Whether to visualize cross-tokenizer alignment (for debugging).

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
        aligned_teacher_logprobs = teacher_full_logprobs[len(teacher_prompt_ids) :]

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
            teacher_completion = teacher_completion[len(teacher_chat_parser.generation_prompt) :]
        teacher_ids = teacher_tokenizer.encode(teacher_prompt + teacher_completion, add_special_tokens=False)
        teacher_completion_ids = teacher_ids[len(teacher_prompt_ids) :]

        # Query teacher for logprobs
        teacher_full_logprobs = await teacher_engine.compute_logprobs(teacher_ids)
        teacher_logprobs = teacher_full_logprobs[len(teacher_prompt_ids) :]

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

        if visualize:
            from rllm.trainer.distill import visualize_alignment

            visualize_alignment(
                student_ids=student_completion_ids,
                student_tokenizer=student_tokenizer,
                teacher_ids=teacher_completion_ids,
                teacher_tokenizer=teacher_tokenizer,
                teacher_logprobs=teacher_logprobs,
                student_logprobs=student_logprobs,
                reasoning_str=reasoning_str,
                content_str=content_str,
            )

    # Pad aligned_teacher_logprobs if needed
    if len(aligned_teacher_logprobs) < len(student_logprobs):
        aligned_teacher_logprobs = list(aligned_teacher_logprobs) + [0.0] * (len(student_logprobs) - len(aligned_teacher_logprobs))

    # Compute advantages using the pure algo function
    # Use -inf/inf as defaults if clip bounds not specified
    return compute_distill_reverse_kl(
        teacher_logprobs=aligned_teacher_logprobs,
        student_logprobs=student_logprobs,
        clip_min=clip_min if clip_min is not None else float("-inf"),
        clip_max=clip_max if clip_max is not None else float("inf"),
        kl_penalty_coef=kl_penalty_coef,
        kl_discount_factor=kl_discount_factor,
    )
