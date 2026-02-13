import logging

from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine import ModelOutput, RolloutEngine
from rllm.rewards.math_reward import rllm_reward_fn_math
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import TerminationReason, Workflow

logger = logging.getLogger(__name__)


def _get_full_response(output: ModelOutput) -> str:
    """Get the full response from ModelOutput, handling thinking models.
    
    The Qwen3Renderer parses <think>...</think> and separates:
    - reasoning: content BETWEEN <think> and </think> (tags stripped)
    - content: text AFTER </think>
    
    For reward evaluation, we reconstruct the full response with the </think>
    delimiter so RewardMathFn can find the answer after the thinking block.
    
    Edge case: If the model hits max_response_length before closing </think>,
    the entire response ends up in content (starting with <think>). In this case,
    we use the raw text which may contain \\boxed{} somewhere.
    """
    content = output.content or ""
    reasoning = output.reasoning or ""

    if reasoning and content:
        # Re-add the </think> delimiter so reward function can find the answer
        return f"{reasoning}</think>{content}"
    if content:
        # Check if this is an unparsed thinking response (starts with <think> but no reasoning)
        # This happens when model hits length limit before </think>
        if content.startswith("<think>") and not reasoning:
            # Use the raw text which may have \boxed{} somewhere
            raw_text = output.text or ""
            if "\\boxed" in raw_text:
                return raw_text
        return content
    if reasoning:
        # If only reasoning, still add delimiter in case answer is at end of reasoning
        return f"{reasoning}</think>"
    return ""


def math_reward_fn(task: dict, action: str):
    """Wrapper for math reward function compatible with workflow interface."""
    ground_truth = task.get("ground_truth", "")
    data_source = task.get("data_source", "math")
    return rllm_reward_fn_math(data_source, action, ground_truth)


class SimpleMathWorkflow(Workflow):
    """
    Simple single-turn math workflow for distillation training.
    
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
        
        # Create math prompt
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
        
        # Check for length exceeded - if exceeded, count as wrong (reward=0)
        is_length_exceeded = output.finish_reason == "length"
        
        # If length exceeded, override the reward to 0 (wrong answer)
        final_reward = 0.0 if is_length_exceeded else reward_result.reward
        final_is_correct = False if is_length_exceeded else reward_result.is_correct
        
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
                    reward=final_reward,
                    model_output=output,
                )
            ],
            reward=final_reward,
        )
        
        # Return episode with single trajectory
        episode = Episode(
            id=uid,
            task=task,
            trajectories=[trajectory],
            is_correct=final_is_correct,
            metrics={"accuracy": float(final_is_correct)},
        )
        
        # Set termination reason if length exceeded
        if is_length_exceeded:
            episode.termination_reason = TerminationReason.MAX_RESPONSE_LENGTH_EXCEEDED
        
        return episode
