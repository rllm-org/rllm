import copy
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class MathAgent(BaseAgent):
    """
    A math agent that solves mathematical problems step by step, following the BaseAgent interface.
    """

    def __init__(self, accumulate_thinking=True):
        """
        Initialize the MathAgent.
        """
        self._trajectory = Trajectory()
        self.messages = []
        self.accumulate_thinking = accumulate_thinking

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Process environment feedback and update internal state."""

        # If observation is None, this is a reward update for the existing step
        if observation is None:
            if self.trajectory.steps:
                cur_step = self.get_current_state()
                cur_step.reward = reward
                cur_step.done = done
                cur_step.info = info
            return

        # This is a new observation, create a new step
        if isinstance(observation, dict):
            print("--------------------------------")
            print("--------------------------------")
            print("OBSERVATION:")
            print(observation)
            print("++++++++++++++++++++++++++++++++")
            print("QUESTION:" + observation.get("question"))
            print("++++++++++++++++++++++++++++++++")
            print("--------------------------------")
            print("--------------------------------")

            formatted_observation = observation["question"]
        elif isinstance(observation, str):
            formatted_observation = observation
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

        return action

    def reset(self) -> None:
        """Reset agent state for new episode (wipes trajectory and messages)."""
        self._trajectory = Trajectory()
        self.messages = []

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
