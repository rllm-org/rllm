from typing import Any, Dict, List, Optional, Tuple
from abc import abstractmethod

from rllm.environments.base.base_env import BaseEnv


class MultiAgentEnv(BaseEnv):
    """Base environment class for multi-agent workflows"""
    
    def __init__(self, task_data: Dict[str, Any] = None, **kwargs):
        super().__init__()
        self.task_data = task_data or {}
        self.multi_agent_context: Dict[str, Any] = {}
        self.agent_history: List[Dict[str, Any]] = []
        self.current_agent_id: Optional[str] = None
        
    def reset(self) -> Tuple[Dict, Dict]:
        self.multi_agent_context = {}
        self.agent_history = []
        self.current_agent_id = None
        return self._create_observation(), self._create_info()
    
    def reset_with_input(self, agent_input: Dict[str, Any]) -> Tuple[Dict, Dict]:
        self.multi_agent_context = agent_input
        
        self._process_multi_agent_input(agent_input)
        
        observation = self._create_observation_with_context()
        info = self._create_info_with_context()
        
        return observation, info
    
    def _process_multi_agent_input(self, agent_input: Dict[str, Any]):
        for source_agent, agent_output in agent_input.items():
            if isinstance(agent_output, dict) and "response" in agent_output:
                self.agent_history.append({
                    "agent_id": source_agent,
                    "response": agent_output["response"],
                    "timestamp": agent_output.get("timestamp", None)
                })
    
    @abstractmethod
    def _create_observation(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod  
    def _create_info(self) -> Dict[str, Any]:
        pass
    
    def _create_observation_with_context(self) -> Dict[str, Any]:
        base_obs = self._create_observation()
        
        if self.agent_history:
            base_obs["previous_agents"] = self.agent_history
            base_obs["collaboration_prompt"] = self._format_collaboration_prompt()
        
        return base_obs
    
    def _create_info_with_context(self) -> Dict[str, Any]:
        base_info = self._create_info()
        base_info["multi_agent_mode"] = True
        base_info["current_agent"] = self.current_agent_id
        base_info["agent_count"] = len(self.agent_history) + 1
        return base_info
    
    def _format_collaboration_prompt(self) -> str:
        if not self.agent_history:
            return ""
        
        prompt = "Previous agent responses in the Chain of Experts:\n\n"
        for i, agent_data in enumerate(self.agent_history):
            prompt += f"Agent {agent_data['agent_id']}:\n{agent_data['response']}\n\n"
        
        prompt += "Please consider the above responses and provide your analysis or solution:"
        return prompt
