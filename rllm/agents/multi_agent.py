from typing import Any, Dict, List, Optional

from rllm.agents.agent import BaseAgent, Action, Step, Trajectory
from rllm.engine.multi_agent_execution_engine import AgentRole


class MultiAgentBase(BaseAgent):
    """Base class for Multi-Agent workflows"""
    
    def __init__(
        self, 
        agent_id: str, 
        role: AgentRole = AgentRole.SPECIALIST,
        system_prompt: str = "",
        **kwargs
    ):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self._trajectory = Trajectory()
        self.multi_agent_context: Dict[str, Any] = {}
        
    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Convert internal state to chat completions format"""
        messages = []
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if self.multi_agent_context:
            context_content = self._format_multi_agent_context()
            if context_content:
                messages.append({"role": "system", "content": context_content})

        for step in self._trajectory.steps:
            if step.observation:
                messages.append({"role": "user", "content": str(step.observation)})
            if step.model_response:
                messages.append({"role": "assistant", "content": step.model_response})
                
        return messages
    
    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory
    
    def reset(self):
        """Reset agent state"""
        self._trajectory = Trajectory()
        self.multi_agent_context = {}
    
    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        
        actual_observation = observation
        chain_context = None
        if isinstance(observation, dict):
            if "base_observation" in observation:
                actual_observation = observation["base_observation"]
            if "chain_context" in observation:
                chain_context = observation["chain_context"]
        
        if chain_context:
            self.multi_agent_context["chain_context"] = chain_context
        
        if not self._trajectory.steps or self._trajectory.steps[-1].done:
            step = Step(observation=actual_observation, reward=reward, done=done, info=info)
            self._trajectory.steps.append(step)
        else:
            current_step = self._trajectory.steps[-1]
            current_step.observation = actual_observation
            current_step.reward = reward
            current_step.done = done
            current_step.info.update(info)
    
    def update_from_model(self, response: str, **kwargs) -> Action:
        if self._trajectory.steps:
            current_step = self._trajectory.steps[-1]
            current_step.model_response = response
            current_step.action = self._parse_action(response)
        
        return Action(action=response)
    
    def get_current_state(self) -> Optional[Step]:
        return self._trajectory.steps[-1] if self._trajectory.steps else None
    
    def _format_multi_agent_context(self) -> str:
        if "chain_context" in self.multi_agent_context:
            return f"CHAIN CONTEXT:\n{self.multi_agent_context['chain_context']}"
        return ""
    
    def _parse_action(self, response: str) -> Any:
        return response
