import asyncio

import numpy as np
from tinker import ModelInput, SamplingClient
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized

from rllm.agents.agent import Step, TrajectoryGroup
from rllm.experimental.opsd.utils import OPSDConfig
from rllm.parser.chat_template_parser import ChatTemplateParser


async def calculate_reverse_kl_advantage(
    step: Step,
    sampling_client: SamplingClient,
    parser: ChatTemplateParser,
    kl_penalty_coef: float,
    kl_discount_factor: float,
    teacher_messages: list[dict] | None = None,
    teacher_prompt_ids: list[int] | None = None,
):
    """Calculate the reverse KL advantage for a step.

    Args:
        step: The step to calculate the reverse KL advantage for.
        teacher_messages (optional): The teacher messages to use for the reverse KL advantage.
        teacher_prompt_ids (optional): The teacher prompt ids to use for the reverse KL advantage.
        sampling_client: The sampling client to use for the reverse KL advantage.
        tokenizer: The tokenizer to use for the reverse KL advantage.
        kl_penalty_coef: The penalty coefficient for the reverse KL advantage.
        kl_discount_factor: The discount factor for the reverse KL advantage.
    """
    # teacher messages and teacher prompt ids cannot both be None
    assert teacher_messages is not None or teacher_prompt_ids is not None, "Either teacher messages or teacher prompt ids must be provided"
    student_response_ids = step.response_ids

    if teacher_prompt_ids is None:
        teacher_prompt_str = parser.parse(teacher_messages, add_generation_prompt=True)
        teacher_prompt_ids = parser.tokenizer.encode(teacher_prompt_str, add_special_tokens=False)

    teacher_full_model_input = ModelInput.from_ints(tokens=teacher_prompt_ids + student_response_ids)
    teacher_full_logprobs = await sampling_client.compute_logprobs_async(teacher_full_model_input)
    teacher_response_logprobs = teacher_full_logprobs[len(teacher_prompt_ids) :]
    assert len(teacher_response_logprobs) == len(step.logprobs), f"Length mismatch in response logprobs: teacher={len(teacher_response_logprobs)}, student={len(step.logprobs)}"

    teacher_response_logprobs = np.array(teacher_response_logprobs)
    student_response_logprobs = np.array(step.logprobs)

    reverse_kl = student_response_logprobs - teacher_response_logprobs
    advantage = -kl_penalty_coef * reverse_kl
    if kl_discount_factor > 0:
        advantage = discounted_future_sum_vectorized(advantage, kl_discount_factor)

    step.advantage = advantage.tolist()


async def calculate_advantage_opsd(
    trajectory_groups: list[TrajectoryGroup],
    sampling_client: SamplingClient,
    parser: ChatTemplateParser,
    opsd_config: OPSDConfig,
):
    """Calculate the advantage for a trajectory group using OPSD."""
    all_tasks = []
    teacher_messages_key = opsd_config.teacher_messages_key
    for trajectory_group in trajectory_groups:
        for trajectory in trajectory_group.trajectories:
            # use trajectory-level teacher prompt ids if available
            teacher_messages = trajectory.info.get(teacher_messages_key, None)
            if teacher_messages is not None:
                # pre-compute the prompt ids for every step to reuse
                teacher_prompt_ids = parser.tokenizer.encode(parser.parse(teacher_messages, add_generation_prompt=True), add_special_tokens=False)
                for step in trajectory.steps:
                    all_tasks.append(calculate_reverse_kl_advantage(step, sampling_client, parser, opsd_config.kl_penalty_coef, opsd_config.kl_discount_factor, teacher_prompt_ids=teacher_prompt_ids))
            else:
                for step in trajectory.steps:
                    teacher_messages = trajectory.info.get(teacher_messages_key)
                    all_tasks.append(calculate_reverse_kl_advantage(step, sampling_client, parser, opsd_config.kl_penalty_coef, opsd_config.kl_discount_factor, teacher_messages=teacher_messages))

    await asyncio.gather(*all_tasks)
