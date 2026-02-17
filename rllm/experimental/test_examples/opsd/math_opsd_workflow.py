from rllm.agents.agent import Episode, Trajectory
from rllm.experimental.opsd.workflow_utils import OPSDConfig, opsd_postprocess
from rllm.experimental.rollout.completer import Completer
from rllm.experimental.rollout.rollout_engine import RolloutEngine
from rllm.rewards.reward_fn import math_reward_fn
from rllm.workflows.workflow import Workflow


class MathOPSDWorkflow(Workflow):
    def __init__(self, rollout_engine: RolloutEngine, **kwargs):
        # Extract opsd_config before passing kwargs to super
        opsd_config = kwargs.pop("opsd_config", None)
        super().__init__(rollout_engine, **kwargs)  # type: ignore
        self.reward_function = math_reward_fn
        # completer is used to construct step based on the messages input and model output
        self.completer = Completer(rollout_engine)

        self.opsd_config = opsd_config or OPSDConfig(
            kl_penalty_coef=1.0,
            kl_discount_factor=0.0,
        )

    @opsd_postprocess
    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute the math OPSD workflow."""
        self.reset(task, uid)

        student_prompt = f"Problem: {task['question']}"
        teacher_prompt = student_prompt + "\n\n" + f"Here is a reference solution:\n\n{task['ground_truth']}" + "\n\n" + "After understanding the reference solution, please try to solve this problem using your own approach below:"

        student_messages = [{"role": "user", "content": student_prompt}]
        teacher_messages = [{"role": "user", "content": teacher_prompt}]

        student_step = await self.completer.complete(student_messages, action_hook=lambda model_output: model_output.content)
        # store the teacher messages in the student step info for later use
        student_step.info["teacher_messages"] = teacher_messages

        # we still evaluate whether the student's solution is correct
        reward_output = self.reward_function(task, student_step.action)
        is_correct = reward_output.is_correct or False

        traj = Trajectory(name="solver", steps=[student_step])
        return Episode(id=uid, task=task, trajectories=[traj], is_correct=is_correct)
