import json
import queue
import warnings
from typing import Any

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import RewardFunction, zero_reward
from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import Tool


class ToolEnvironment(BaseEnv):
    """
    A simple environment for tool-based agents that provides questions and evaluates responses.
    """

    def __init__(self, task: dict | None = None, tools: list[str] | None = None, tool_map: dict[str, type[Tool]] | None = None, reward_fn: RewardFunction | None = None, max_steps=10):
        """
        Initialize the ToolEnvironment.

        Args:
            task: Task information for the environment.
            tools: List of tool names to look up in the registry (legacy behavior).
            tool_map: Dictionary mapping tool names to Tool classes (new behavior).
            reward_fn: Reward function to use for evaluation.
            max_steps: Maximum number of steps allowed in the environment.
        """
        if tool_map is not None and tools is not None:
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")

        self.step_count = 0
        self.max_steps = max_steps

        # Initialize MultiTool with either tools or tool_map
        if tool_map is not None:
            self.tools = MultiTool(tool_map=tool_map)
        elif tools is not None:
            self.tools = MultiTool(tools=tools)
        else:
            self.tools = MultiTool(tools=[])

        self.task = task
        if reward_fn is None:
            warnings.warn("No reward function specified, will get 0 reward.", stacklevel=2)
            self.reward_fn = zero_reward
        else:
            self.reward_fn = reward_fn

    def reset(self):
        """Reset the environment and return initial observations."""
        self.step_count = 0

        return self.task, {}

    def step(self, action: list[dict] | str | dict):
        """
        Take a step in the environment based on the action.

        Args:
            actions: List containing a single action string from the agent

        Returns:
            next_observations, rewards, terminateds, infos
        """
        if action is None:
            action = []

        if isinstance(action, dict):
            action = [action]
        self.step_count += 1

        reward = 0
        # Check if we should terminate
        done = self.step_count >= self.max_steps or isinstance(action, str)
        # Check if action contains a "finish" tool call
        if isinstance(action, list) and action:
            for tool_call in action:
                if tool_call.get("function", {}).get("name") == "finish":
                    done = True
                    break
        if done:
            # Cannot find tool calls which means the agent is not using the tool and is done.
            if isinstance(action, str):
                llm_response = action
            elif isinstance(action, list):
                # Find the finish tool call
                finish_action = None
                for tool_call in action:
                    if tool_call.get("function", {}).get("name") == "finish":
                        finish_action = tool_call
                        break
                if finish_action:
                    arguments = finish_action.get("function", {}).get("arguments", {})
                    llm_response = arguments.get("response", "")
                else:
                    # No finish tool call found, use the action itself
                    llm_response = str(action)

            task_info = self.task if self.task is not None else {}
            reward_output = self.reward_fn(task_info=task_info, action=llm_response)
            return {}, reward_output.reward, done, {"response": action, "metadata": reward_output.metadata, "is_correct": reward_output.is_correct}

        tool_calls = action
        assert isinstance(tool_calls, list)
        tool_outputs = self._execute_tool_calls(tool_calls)
        next_obs = {"tool_outputs": tool_outputs}

        # Return results as lists with single items to maintain batch structure
        return next_obs, reward, done, {"response": action, "metadata": {}}

    def _execute_tool_calls(self, tool_calls: list[dict[Any, Any]]) -> dict[str, str]:
        import threading

        # Create a dictionary to store results in order
        tool_outputs: dict[str, str] = {}
        output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        threads = []

        def execute_tool(tool_call):
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_output = self.tools(tool_name=tool_name, **tool_args)
            tool_output_str = tool_output.to_string()

            output_queue.put((tool_call["id"], tool_output_str))

        # Create and start a thread for each tool call
        for idx, tool_call in enumerate(tool_calls):
            thread = threading.Thread(target=execute_tool, args=(tool_call,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Collect results and store in order
        while not output_queue.empty():
            tool_call_id, output_str = output_queue.get()
            tool_outputs[tool_call_id] = output_str

        return tool_outputs

    @staticmethod
    def from_dict(env_args: dict) -> "ToolEnvironment":
        tools = env_args.pop("tools", None)
        tool_map = env_args.pop("tool_map", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)
        return ToolEnvironment(task=env_args, tools=tools, tool_map=tool_map, max_steps=max_steps, reward_fn=reward_fn)
