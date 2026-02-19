"""On-Policy Self-Distillation (OPSD) workflow.

Based on: https://arxiv.org/abs/2601.18734

The student and teacher share the same model, but the teacher sees the
ground-truth solution as privileged context in its prompt.
"""

from functools import partial

from rllm.agents.agent import Episode, Step
from rllm.engine import ModelOutput, RolloutEngine
from rllm.trainer.distill import compute_step_distill_advantage
from rllm.workflows.distillation_workflow import DistillationWorkflow
from rllm.workflows.workflow import TerminationEvent, TerminationReason
from rllm.rewards.reward_fn import RewardFunction


REFERENCE_SOLUTION_TEMPLATE = (
    "Here is a reference solution:\n{solution}\n\n"
    "After understanding the reference solution, "
    "please try to solve this problem using your own approach below."
)


def inject_reference_solution(messages: list[dict], ground_truth: str) -> list[dict]:
    """Append reference solution context to the last user message."""
    messages = [msg.copy() for msg in messages]
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            messages[i]["content"] += "\n\n" + REFERENCE_SOLUTION_TEMPLATE.format(solution=ground_truth)
            break
    return messages


class OPSDWorkflow(DistillationWorkflow):
    def __init__(self, rollout_engine: RolloutEngine, reward_function: RewardFunction | None = None, clip_min: float | None = None, clip_max: float | None = None, **kwargs):
        super().__init__(rollout_engine, reward_function=reward_function, teacher_engine=None, shared_tokenizer=True, clip_min=clip_min, clip_max=clip_max, **kwargs)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        question = task.get("question", "")
        ground_truth = task.get("ground_truth", "")
        messages = [{"role": "user", "content": question}]

        output: ModelOutput = await self.rollout_engine.get_model_response(messages, application_id=uid, **kwargs)
        step = Step(
            chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning, "tool_calls": output.tool_calls}],
            model_output=output,
            reward = self.reward_function(task, output.text).reward,
        )

        teacher_prompt_fn = partial(inject_reference_solution, ground_truth=ground_truth) if ground_truth else None

        step.advantage = await compute_step_distill_advantage(
            step=step,
            teacher_engine=self.rollout_engine,
            student_tokenizer=self.rollout_engine.tokenizer,
            teacher_tokenizer=self.rollout_engine.tokenizer,
            shared_tokenizer=True,
            teacher_chat_parser=self.rollout_engine.chat_parser,
            teacher_prompt_fn=teacher_prompt_fn,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
        )
        self.trajectory.steps.append(step)

        if output.finish_reason == "length":
            raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

        raise TerminationEvent(TerminationReason.ENV_DONE)