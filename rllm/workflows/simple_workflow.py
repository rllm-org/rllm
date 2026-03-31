from rllm.agents.agent import Action, BaseAgent, Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


class SimpleAgent(BaseAgent):
    def __init__(self, **kwargs):
        self._trajectory = Trajectory()

    def reset(self):
        self._trajectory = Trajectory()

    def update_from_model(*args, **kwargs):
        pass

    def update_from_env(*args, **kwargs):
        pass

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory


class SimpleWorkflow(Workflow):
    def __init__(self, rollout_engine: RolloutEngine, reward_function: RewardFunction, **kwargs):
        super().__init__(rollout_engine, **kwargs)
        self.agent = SimpleAgent()
        self.reward_function = reward_function

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute the single agent workflow."""
        # Reset components for new task
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
        action = Action(action=output.content)
        reward_result = self.reward_function(task, action)

        trajectory = self.agent.trajectory
        trajectory.steps.append(
            Step(
                chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
                thought=output.reasoning,
                action=action,
                reward=reward_result.reward,
                model_output=output,
            )
        )

        self.commit(agent=self.agent, reset=True)

        if output.finish_reason == "length":
            raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)

        raise TerminationEvent(TerminationReason.ENV_DONE)
