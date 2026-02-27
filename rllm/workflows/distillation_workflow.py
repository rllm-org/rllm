from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow
from rllm.trainer.distill import compute_step_distill_advantage
from rllm.rewards.reward_fn import RewardFunction

class DistillationWorkflow(Workflow):
    def __init__(self, rollout_engine: RolloutEngine, reward_function: RewardFunction | None = None, teacher_engine: RolloutEngine | None = None, shared_tokenizer: bool = False, clip_min: float | None = None, clip_max: float | None = None, **kwargs):
        super().__init__(rollout_engine, **kwargs)
        self.reward_function = reward_function
        self.teacher_engine = teacher_engine
        self.shared_tokenizer = shared_tokenizer
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.trajectory = Trajectory(name="student")

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task, uid)

        if task.get("messages", None) is not None:
            messages = task["messages"]
        elif task.get("question", None) is not None:
            messages = [{"role": "user", "content": task.get("question")}]
        elif task.get("prompt", None) is not None:
            messages = [{"role": "user", "content": task.get("prompt")}]
        elif task.get("problem", None) is not None:
            messages = [{"role": "user", "content": task.get("problem")}]
        else:
            raise ValueError("No question, problem, messages, or prompt key found in task")

        output: ModelOutput = await self.rollout_engine.get_model_response(messages, application_id=uid, **kwargs)
        step = Step(
            chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning, "tool_calls": output.tool_calls}],
            model_output=output,
            reward = self.reward_function(task, output.content).reward,
        )
        if not self.rollout_engine.is_validation:
            step.advantage = await compute_step_distill_advantage(
                step=step,
                teacher_engine=self.teacher_engine,
                student_tokenizer=self.rollout_engine.tokenizer,
                teacher_tokenizer=self.teacher_engine.tokenizer,
                shared_tokenizer=self.shared_tokenizer,
                teacher_chat_parser=self.teacher_engine.chat_parser,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
                visualize=not self.shared_tokenizer and hash(uid) % 100 == 0,
            )
        self.trajectory.steps.append(step)

        if output.finish_reason == "length":
            raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

        raise TerminationEvent(TerminationReason.ENV_DONE)

    def collect_trajectories(self) -> Episode:
        return Episode(trajectories=[self.trajectory])

    def reset(self, task: dict, uid: str | None = None) -> None:
        super().reset(task, uid)
        self.trajectory = Trajectory(name="student")