import copy
import json
import logging
import uuid
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.agents.system_prompts import TOOL_SYSTEM_PROMPT
from rllm.parser import ToolParser, get_tool_parser
from rllm.tools.mcp_tool import MCPTool
from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import Tool

logger = logging.getLogger(__name__)


class ToolAgent(BaseAgent):
    """
    An tool agent that can use tools to interact with the environment,
    refactored to follow the BaseAgent abstraction.
    """

    def __init__(
        self,
        system_prompt=TOOL_SYSTEM_PROMPT,
        parser_name="qwen",
        tools: list[str] | None = None,
        tool_map: dict[str, type[Tool]] | None = None,
    ):
        """
        Initialize the ToolAgent.

        Args:
            system_prompt: System prompt for the agent.
            parser_name: Name of the parser to use for tool calls.
            tools: List of tool names available to the agent (legacy behavior).
            tool_map: Dictionary mapping tool names to Tool classes (new behavior).
        """
        if tool_map is not None and tools is not None:
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")

        self.system_prompt = system_prompt

        # Initialize MultiTool with either tools or tool_map
        if tool_map is not None:
            self.tools = MultiTool(tool_map=tool_map)
        elif tools is not None:
            self.tools = MultiTool(tools=tools)
        else:
            self.tools = MultiTool(tools=[])

        parser_class: type[ToolParser] = get_tool_parser(parser_name=parser_name)
        self.tool_parser = parser_class()

        self.tools_prompt = self.tool_parser.get_tool_prompt(json.dumps(self.tools.json, indent=2))

        # Initialize state according to BaseAgent
        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.current_observation = None
        self.reset()  # Call reset to set initial state

    def _format_observation_as_messages(self, obs: Any) -> list[dict]:
        """Helper to format observation into messages."""
        messages = []
        if isinstance(obs, dict):
            if "question" in obs:
                messages.append({"role": "user", "content": obs["question"]})
            elif "tool_outputs" in obs:
                # Format tool outputs from environment observation
                for tool_call_id, tool_output_str in obs["tool_outputs"].items():
                    messages.append(
                        {
                            "role": "tool",
                            "content": tool_output_str,
                            "tool_call_id": tool_call_id,
                        }
                    )
        elif isinstance(obs, str):
            messages.append({"role": "user", "content": obs})
        elif obs:
            messages.append({"role": "user", "content": str(obs)})

        return messages

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's state based on environment feedback.
        Formats observation and updates the trajectory.
        """

        # Format the observation for the next model call
        obs_messages = self._format_observation_as_messages(observation)
        self.messages.extend(obs_messages)
        self.current_observation = observation

        if self._trajectory.steps:
            self._trajectory.steps[-1].reward = reward
            self._trajectory.steps[-1].done = done
            self._trajectory.steps[-1].info = info

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's state based on the model's response.
        Parses the response, updates messages, and the current step in the trajectory.
        """
        tool_calls_dict = []
        assistant_content = response
        # Attempt to parse tool calls from string response
        try:
            tool_calls = self.tool_parser.parse(response)
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": tool_call.to_dict(),
                }
                for tool_call in tool_calls
            ]

        except Exception as e:
            logger.error(f"Failed to parse tool calls from string response: {e}")
            tool_calls_dict = []  # Indicate no valid tool calls parsed

        # Append assistant message to chat history
        assistant_message = {"role": "assistant", "content": assistant_content}
        if len(tool_calls_dict) > 0:
            # Ensure arguments within tool_calls_dict are strings if needed by downstream processing
            for call in tool_calls_dict:
                if isinstance(call.get("function", {}).get("arguments"), dict):
                    call["function"]["arguments"] = json.dumps(call["function"]["arguments"])
        else:
            tool_calls_dict = [
                {
                    "id": str(uuid.uuid4()),
                    "type": "function",
                    "function": {
                        "name": "finish",
                        "arguments": {
                            "response": assistant_content,
                        },
                    },
                }
            ]

        self.messages.append(assistant_message)

        new_step = Step(chat_completions=copy.deepcopy(self.chat_completions), action=tool_calls_dict, model_response=response, observation=self.current_observation)
        self._trajectory.steps.append(new_step)

        return Action(action=tool_calls_dict)

    def reset(self):
        """Resets the agent's state for a new episode."""
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.system_prompt + self.tools_prompt}]

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Returns the current message history for the model."""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """Returns the trajectory recorded so far."""
        return self._trajectory


class MCPToolAgent(ToolAgent):
    def __init__(self, system_prompt=TOOL_SYSTEM_PROMPT, parser_name="qwen", tool_map=list[MCPTool]):
        self.system_prompt = system_prompt
        self.tool_map = tool_map

        parser_class: type[ToolParser] = get_tool_parser(parser_name=parser_name)
        self.tool_parser = parser_class()

        tools_json = [tool.json for tool in self.tool_map.values()]
        self.tools_prompt = self.tool_parser.get_tool_prompt(json.dumps(tools_json, indent=2))

        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.reset()
