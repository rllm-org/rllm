"""
Terminal-Bench agent integration for rLLM.

This module provides the TerminalTerminusAgent class which serves as a thin
interface between rLLM's AgentExecutionEngine and Terminal-Bench environments.
"""

from typing import Any, Dict, List, Optional
import copy

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory


class TerminalTerminusAgent(BaseAgent):
    """
    Agent for Terminal-Bench integration.
    
    Provides minimal interface between rLLM's AgentExecutionEngine
    and Terminal-Bench environment. All Terminal-Bench specific logic
    is handled by TerminalTerminusEnv.
    """
    
    def __init__(self, **kwargs):
        self.reset()
        
    def update_from_env(
        self, 
        observation: Any, 
        reward: float, 
        done: bool, 
        info: Dict[str, Any], 
        **kwargs
    ):
        """
        Update agent state from environment observation.
        
        Args:
            observation: Dict containing prompt from environment
            reward: Reward signal from environment
            done: Episode termination flag
            info: Additional metadata
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
        """
        Update agent state from model response.
        
        Args:
            response: Raw model response (JSON command batch)
            
        Returns:
            Action containing the raw response for environment processing
        """
        self._trajectory.steps.append(self.cur_step)
        
        # Update Trajectory
        cur_step = self._trajectory.steps[-1]
        cur_step.model_response = response
        cur_step.action = response
        
        # Update Chat Completions
        self.messages.append({"role": "assistant", "content": response})
        cur_step.chat_completions = copy.deepcopy(self.messages)
        self.step += 1
        return Action(action=response)
    
    def get_current_state(self) -> Optional[Step]:
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]
    
    def reset(self):
        self._trajectory = Trajectory()
        # Terminus has no system prompt
        self.messages = []
        self.step = 0

    
    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        return self.messages
    
    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory
    
    