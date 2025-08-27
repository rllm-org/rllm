from typing import Any, Dict, List, Optional
import copy

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class TerminalTerminusAgent(BaseAgent):
    """Thin agent wrapper; environment handles Terminal-Bench specifics.

    Maintains a simple alternating chat message history and mirrors raw
    model responses to ``Action`` objects consumed by the environment.
    """
    
    def __init__(self, **kwargs):
        """Initialize internal state."""
        self.reset()
        
    def update_from_env(
        self,
        observation: Any,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        **kwargs,
    ) -> None:
        """Update agent state from an environment transition.

        Args:
            observation: Latest observation dict from the environment.
            reward: Scalar reward from the previous action.
            done: Whether the episode has terminated.
            info: Auxiliary environment info.
            **kwargs: Unused; reserved for extensions.
        """
        if self._trajectory.steps:
            prior_step = self._trajectory.steps[-1]
            prior_step.observation = observation
            prior_step.reward = reward
            prior_step.done = done
            prior_step.info = info
            
        self.messages.append({"role": "user", "content": observation["prompt"]})
        self.cur_step = Step(observation=observation)
      
    def update_from_model(self, response: str, **kwargs) -> Action:
        """Record model response and produce an action.

        Args:
            response: Raw assistant text.
            **kwargs: Unused; reserved for extensions.

        Returns:
            Action: Action object whose ``action`` is the raw response.
        """
        self._trajectory.steps.append(self.cur_step)
        
        cur_step = self._trajectory.steps[-1]
        cur_step.model_response = response
        cur_step.action = response
        
        self.messages.append({"role": "assistant", "content": response})
        cur_step.chat_completions = copy.deepcopy(self.messages)
        self.step += 1
        return Action(action=response)
    
    def get_current_state(self) -> Optional[Step]:
        """Return the most recent step in the trajectory.

        Returns:
            Optional[Step]: Last step if available.
        """
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]
    
    def reset(self) -> None:
        """Reset message history and trajectory."""
        self._trajectory = Trajectory()
        self.messages = []
        self.step = 0

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """OpenAI-style message history consumed by the rollout engine."""
        return self.messages
    
    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory
    
    