from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import Workflow


class SingleTurnWorkflow(Workflow):
    def __init__(self, rollout_engine: RolloutEngine, reward_function: RewardFunction = None, **kwargs):
        super().__init__(rollout_engine, **kwargs)

        self.reward_function = reward_function

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute the single agent workflow."""
        # Reset components for new task
        self.reset(task, uid)

        messages = task["messages"]
        response = await self.rollout_engine.get_model_response(messages)
        reward_result = self.reward_function(task, response)


        trajectory = Trajectory()
        trajectory.steps.append(
            Step(
                model_response=response.text,
                action=Action(response.text),
                chat_completions=messages + [{"role": "assistant", "content": response.text}],
                reward=reward_result.reward,
            )
        )

        is_correct = reward_result.is_correct
        # Create episode with trajectories as list of tuples
        episode = Episode(
            id=uid,
            task=task,
            is_correct=is_correct,
            trajectories=[("single_agent", trajectory)],
            metrics={},
        )

        return episode

    def reset(self, task: dict, uid: str):
        self.messages = []
        self.task = task
        self.uid = uid
