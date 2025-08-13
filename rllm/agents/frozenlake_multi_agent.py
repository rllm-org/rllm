import copy
import logging
import re
from typing import Any, Dict, List

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.multi_agent import MultiAgentBase
from rllm.agents.frozenlake_agent import FrozenLakeAgent

logger = logging.getLogger(__name__)


class FrozenLakeProposerAgent(FrozenLakeAgent, MultiAgentBase):
    """
    Proposer agent for FrozenLake Chain of Experts.
    
    Role: Analyzes the initial state and proposes a high-level strategy.
    Focus: Safe path planning and obstacle identification.
    """
    
    SYSTEM_PROMPT = """You are the PROPOSER in a Chain of Experts for FrozenLake navigation.
Your role: Analyze the frozen lake and propose a high-level strategic approach.

FrozenLake Quick Guide:
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Actions: Up | Down | Left | Right

As the PROPOSER, you should:
1. Analyze the current board layout
2. Identify the safest general path direction
3. Note any immediate dangers (holes near player)
4. Suggest a strategic approach (e.g., "move right first to avoid holes", "take indirect path for safety")

Your analysis will help the next expert make the specific move decision.

You should show your strategic thinking and then propose the NEXT ACTION in ``` ```.
The final action MUST be one of: Up, Down, Left, Right.
Focus on SAFETY and STRATEGY rather than just the shortest path.
"""

    def __init__(self, agent_id: str, max_steps: int = None, **kwargs):
        FrozenLakeAgent.__init__(self, max_steps=max_steps, use_accumulate_history=True, **kwargs)
        MultiAgentBase.__init__(self, agent_id=agent_id, system_prompt=self.SYSTEM_PROMPT, **kwargs)
        self.max_steps = max_steps
        self.step = 0
        self.reset()

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Combine FrozenLake conversation history with simple Chain context"""
        base_messages = FrozenLakeAgent.chat_completions.fget(self)
        
        if self.multi_agent_context and "chain_context" in self.multi_agent_context:
            context_content = self.multi_agent_context["chain_context"]
            context_msg = {"role": "user", "content": f"CHAIN CONTEXT:\n{context_content}"}
            if len(base_messages) > 1:
                return [base_messages[0], context_msg] + base_messages[1:]
            else:
                return base_messages + [context_msg]
        
        return base_messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Update proposer agent with environment observation and chain context"""
        MultiAgentBase.update_from_env(self, observation, reward, done, info, **kwargs)
        
        actual_observation = observation
        if isinstance(observation, dict) and "base_observation" in observation:
            actual_observation = observation["base_observation"]
        
        FrozenLakeAgent.update_from_env(self, actual_observation, reward, done, info, **kwargs)
        
        # NOTE: Step-wise history is automatically maintained by FrozenLakeAgent's use_accumulate_history=True
        # This means each agent keeps track of all previous environment steps and model responses
        # which provides context the user requested
        
    def update_from_model(self, response: str, **kwargs) -> Action:
        return FrozenLakeAgent.update_from_model(self, response, **kwargs)


class FrozenLakeExpertAgent(FrozenLakeAgent, MultiAgentBase):
    """
    Expert agent for FrozenLake Chain of Experts.
    
    Role: Takes the Proposer's strategy and makes the tactical move decision.
    Focus: Precise movement execution with risk assessment.
    """
    
    SYSTEM_PROMPT = """You are the EXPERT in a Chain of Experts for FrozenLake navigation.
Your role: Take the Proposer's strategic advice and make the specific tactical move.

FrozenLake Quick Guide:
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Actions: Up | Down | Left | Right

As the EXPERT, you should:
1. Review the Proposer's strategic analysis
2. Assess immediate movement options and risks
3. Consider slip probability and backup plans
4. Make the precise tactical decision

You will receive context from the PROPOSER about their strategic recommendations.
Use this guidance to make the optimal specific move.

You should show your tactical analysis and then output the NEXT ACTION in ``` ```.
The final action MUST be one of: Up, Down, Left, Right.
Focus on PRECISE EXECUTION of the proposed strategy.
"""

    def __init__(self, agent_id: str, max_steps: int = None, **kwargs):
        FrozenLakeAgent.__init__(self, max_steps=max_steps, use_accumulate_history=True, **kwargs)
        MultiAgentBase.__init__(self, agent_id=agent_id, system_prompt=self.SYSTEM_PROMPT, **kwargs)
        self.max_steps = max_steps
        self.step = 0
        self.reset()

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        base_messages = FrozenLakeAgent.chat_completions.fget(self)
        
        if self.multi_agent_context and "chain_context" in self.multi_agent_context:
            context_content = self.multi_agent_context["chain_context"]
            context_msg = {"role": "user", "content": f"CHAIN CONTEXT:\n{context_content}"}
            if len(base_messages) > 1:
                return [base_messages[0], context_msg] + base_messages[1:]
            else:
                return base_messages + [context_msg]
        
        return base_messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        MultiAgentBase.update_from_env(self, observation, reward, done, info, **kwargs)
        
        actual_observation = observation
        if isinstance(observation, dict) and "base_observation" in observation:
            actual_observation = observation["base_observation"]
        
        FrozenLakeAgent.update_from_env(self, actual_observation, reward, done, info, **kwargs)
        
    def update_from_model(self, response: str, **kwargs) -> Action:
        """Use FrozenLakeAgent's action parsing logic"""
        return FrozenLakeAgent.update_from_model(self, response, **kwargs)


class FrozenLakeJudgeAgent(FrozenLakeAgent, MultiAgentBase):
    """
    Judge agent for FrozenLake Chain of Experts.
    
    Role: Reviews both Proposer and Expert recommendations to make the final decision.
    Focus: Safety validation and optimal choice selection.
    """
    
    SYSTEM_PROMPT = """You are the JUDGE in a Chain of Experts for FrozenLake navigation.
Your role: Review both the Proposer's strategy and Expert's tactical decision to make the final move.

FrozenLake Quick Guide:
Goal: Reach the goal (G). Player (P) and Goal (G) must overlap.

Symbols:
_ Frozen | O Hole | G Goal | P Player

Rules:
1. Avoid falling into holes (O).
2. Frozen tiles are slippery, you may move perpendicular to your intended direction.

Valid Actions: Up | Down | Left | Right

As the JUDGE, you should:
1. Review the Proposer's strategic analysis
2. Evaluate the Expert's tactical recommendation
3. Validate safety and optimality of the proposed move
4. Make the final authoritative decision

You will receive context from both the PROPOSER and EXPERT.
Your job is to synthesize their input and make the best final decision.

You should show your judgment process and then output the FINAL ACTION in ``` ```.
The final action MUST be one of: Up, Down, Left, Right.
Focus on SAFETY VALIDATION and OPTIMAL CHOICE SELECTION.
"""

    def __init__(self, agent_id: str, max_steps: int = None, **kwargs):
        FrozenLakeAgent.__init__(self, max_steps=max_steps, use_accumulate_history=True, **kwargs)
        MultiAgentBase.__init__(self, agent_id=agent_id, system_prompt=self.SYSTEM_PROMPT, **kwargs)
        self.max_steps = max_steps
        self.step = 0
        self.reset()

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        base_messages = FrozenLakeAgent.chat_completions.fget(self)
        
        if self.multi_agent_context and "chain_context" in self.multi_agent_context:
            context_content = self.multi_agent_context["chain_context"]
            context_msg = {"role": "user", "content": f"CHAIN CONTEXT:\n{context_content}"}
            if len(base_messages) > 1:
                return [base_messages[0], context_msg] + base_messages[1:]
            else:
                return base_messages + [context_msg]
        
        return base_messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        MultiAgentBase.update_from_env(self, observation, reward, done, info, **kwargs)
        
        actual_observation = observation
        if isinstance(observation, dict) and "base_observation" in observation:
            actual_observation = observation["base_observation"]
        
        FrozenLakeAgent.update_from_env(self, actual_observation, reward, done, info, **kwargs)
        
    def update_from_model(self, response: str, **kwargs) -> Action:
        return FrozenLakeAgent.update_from_model(self, response, **kwargs)
