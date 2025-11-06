"""
MathAgent with few-shot prompting support to match tinker-cookbook math_rl.

This agent variant includes:
1. Few-shot prefix with a standard example (strawberry)
2. Instruction text matching math_rl: " Write your answer in \\boxed{} format."
"""

import copy
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class MathAgentWithFewshot(BaseAgent):
    """
    A math agent with few-shot prompting that matches tinker-cookbook math_rl behavior.
    """

    # Standard few-shot example from tinker-cookbook math_rl
    STANDARD_FEWSHOT_PREFIX = [
        {
            "role": "user",
            "content": "How many r's are in strawberry? Provide a numerical answer without units, written inside \\boxed{}.",
        },
        {
            "role": "assistant",
            "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
        },
    ]

    def __init__(self, accumulate_thinking=True, use_fewshot=True):
        """
        Initialize the MathAgent with few-shot support.

        Args:
            accumulate_thinking: Whether to accumulate thinking in conversation history
            use_fewshot: Whether to use few-shot prompting
        """
        self._trajectory = Trajectory()
        self.messages = []
        self.accumulate_thinking = accumulate_thinking
        self.use_fewshot = use_fewshot

        # Add few-shot prefix if enabled
        if self.use_fewshot:
            self.messages.extend(copy.deepcopy(self.STANDARD_FEWSHOT_PREFIX))

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Process environment feedback and update internal state."""

        # Reward update for existing step (None OR empty dict)
        if observation is None or (isinstance(observation, dict) and observation == {}):
            if self.trajectory.steps:
                cur_step = self.get_current_state()
                cur_step.reward = reward
                cur_step.done = done
                cur_step.info = info
            return

        # Update reward/done/info on existing step if we have steps already
        if self.trajectory.steps:
            cur_step = self.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            if done:
                return

        # This is a new observation, create a new step
        if isinstance(observation, dict):
            if "question" not in observation:
                raise ValueError(f"Observation dict missing required 'question' field: {observation}")
            # Match math_rl instruction text exactly
            formatted_observation = observation["question"] + " Provide a numerical answer without units, written inside \\boxed{}."
        elif isinstance(observation, str):
            formatted_observation = observation + " Provide a numerical answer without units, written inside \\boxed{}."
        else:
            raise ValueError(f"Invalid observation type: {type(observation)}")

        self.messages.append({"role": "user", "content": formatted_observation})

        new_step = Step(observation=formatted_observation)
        self._trajectory.steps.append(new_step)

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's internal state based on the model's response.
        """

        # Update the latest step
        self.messages.append({"role": "assistant", "content": response})

        cur_step = self.get_current_state()
        cur_step.chat_completions = self.chat_completions
        cur_step.model_response = response

        if response.count("</think>") == 1:
            thought, sep, action = response.partition("</think>")
            thought = thought + sep
            action = Action(action.strip())
        else:
            thought = None
            action = Action(response.strip())

        cur_step.thought = thought
        cur_step.action = action

        # TODO: remove this temporary fix
        return Action(response.strip())

    def reset(self) -> None:
        """Reset agent state for new episode (wipes trajectory but keeps few-shot prefix)."""
        self._trajectory = Trajectory()
        self.messages = []

        # Re-add few-shot prefix after reset
        if self.use_fewshot:
            self.messages.extend(copy.deepcopy(self.STANDARD_FEWSHOT_PREFIX))

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Return conversation history for model interaction."""
        # remove thinking from assistant messages if not accumulate_thinking except the last one
        messages = copy.deepcopy(self.messages)
        if not self.accumulate_thinking:
            for msg in messages[:-1]:
                if msg["role"] == "assistant":
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after
        return messages

    @property
    def trajectory(self) -> Trajectory:
        """Return complete interaction trajectory."""
        return self._trajectory

    def get_current_state(self) -> Step:
        """Returns the current step/state of the agent."""
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]
