from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.math_reward import rllm_reward_fn_math
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow


def _get_full_response(output: ModelOutput) -> str:
    """Get the full response from ModelOutput, handling thinking models."""
    content = output.content or ""
    reasoning = output.reasoning or ""

    if not content and reasoning:
        return reasoning
    if content:
        return content
    return ""


def math_reward_fn(task: dict, action: str):
    """Wrapper for math reward function compatible with workflow interface."""
    ground_truth = task.get("ground_truth", "")
    data_source = task.get("data_source", "math")
    return rllm_reward_fn_math(data_source, action, ground_truth)


class SimpleMathWorkflow(Workflow):
    """
    Workflow:
    1. Generates one solution per problem
    2. Evaluates it with the reward function
    3. Returns an episode with a single trajectory
    """

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        reward_function: RewardFunction = None,
        **kwargs,
    ):
        super().__init__(rollout_engine, **kwargs)
        self.reward_function = reward_function or math_reward_fn

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        """Execute the simple math workflow."""
        self.reset(task, uid)
        
        # Get problem text
        problem = task.get("question") or task.get("problem", "")
        
        # Create math prompt (matching the solver prompt from SolverJudgeMathWorkflow)
        messages = [
            {
                "role": "user",
                "content": f"{problem}\n\nThink step by step and output your final answer within \\boxed{{}}.",
            }
        ]
        
        # Get model response
        output: ModelOutput = await self.rollout_engine.get_model_response(
            messages, application_id=uid, **kwargs
        )
        
        # Get full response for reward evaluation
        full_response = _get_full_response(output)
        
        # Evaluate with reward function
        reward_result = self.reward_function(task, full_response)
        
        # Create single trajectory
        trajectory = Trajectory(
            name="solver",
            steps=[
                Step(
                    chat_completions=messages + [
                        {"role": "assistant", "content": output.content, "reasoning": output.reasoning}
                    ],
                    thought=output.reasoning,
                    action=Action(full_response),
                    reward=reward_result.reward,
                    model_output=output,
                )
            ],
            reward=reward_result.reward,
        )
        
        # Check for length exceeded
        if output.finish_reason == "length":
            raise TerminationEvent(TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED)
        
        # Return episode with single trajectory
        return Episode(
            id=uid,
            task=task,
            trajectories=[trajectory],
            is_correct=reward_result.is_correct,
            metrics={"accuracy": float(reward_result.is_correct)},
        )

