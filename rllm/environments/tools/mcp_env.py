import asyncio
import json
import queue
import threading
import warnings
from contextlib import AsyncExitStack
from typing import Any

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError as err:
    raise ImportError("MCP environment requires extra dependencies. Install with: pip install rllm[tools]") from err

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import RewardFunction, zero_reward
from rllm.tools.mcp_tool import MCPTool


class MCPConnectionManager:
    """Manages MCP connections in a dedicated thread to avoid asyncio context issues."""

    def __init__(self, mcp_server_command: str, mcp_server_args: list[str] | None = None, mcp_server_env: dict[str, str] | None = None):
        self.mcp_server_command = mcp_server_command
        self.mcp_server_args = mcp_server_args or []
        self.mcp_server_env = mcp_server_env

        self.request_queue: queue.Queue[tuple[str, Any, queue.Queue[tuple[str, Any]] | None]] = queue.Queue()
        self.response_queues: dict[str, queue.Queue[Any]] = {}
        self.worker_thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.session: ClientSession | None = None
        self.stdio_transport: Any = None
        self.tool_map: dict[str, MCPTool] = {}
        self.running = False

    def start(self):
        """Start the connection manager thread."""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self.worker_thread.start()

        # Wait for initialization
        response_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.request_queue.put(("init", None, response_queue))
        result = response_queue.get(timeout=30)
        if result[0] == "error":
            raise Exception(f"Failed to initialize MCP connection: {result[1]}")

    def stop(self):
        """Stop the connection manager thread."""
        if not self.running:
            return

        self.running = False
        self.request_queue.put(("stop", None, None))
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> dict[str, str]:
        """Execute tool calls and return results."""
        if not self.running:
            raise Exception("Connection manager not running")

        response_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.request_queue.put(("execute", tool_calls, response_queue))
        result = response_queue.get(timeout=30)
        if result[0] == "error":
            raise Exception(f"Tool execution failed: {result[1]}")
        return result[1]  # type: ignore

    def _run_worker(self):
        """Worker thread that runs the asyncio event loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._worker_loop())
        finally:
            if self.session:
                try:
                    self.loop.run_until_complete(self._cleanup())
                except Exception:
                    pass
            if self.loop:
                self.loop.close()

    async def _worker_loop(self):
        """Main worker loop that processes requests."""
        while self.running:
            try:
                # Check for requests with timeout
                try:
                    request = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                command, data, response_queue = request

                if command == "init":
                    try:
                        await self._initialize_connection()
                        if response_queue:
                            response_queue.put(("success", self.tool_map))
                    except Exception as e:
                        if response_queue:
                            response_queue.put(("error", str(e)))

                elif command == "execute":
                    try:
                        result = await self._execute_tools(data)
                        if response_queue:
                            response_queue.put(("success", result))
                    except Exception as e:
                        if response_queue:
                            response_queue.put(("error", str(e)))

                elif command == "stop":
                    break

            except Exception as e:
                print(f"Worker loop error: {e}")

    async def _initialize_connection(self):
        """Initialize the MCP connection."""
        server_params = StdioServerParameters(command=self.mcp_server_command, args=self.mcp_server_args, env=self.mcp_server_env)

        # Use AsyncExitStack properly within this event loop
        self.exit_stack = AsyncExitStack()
        self.stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = self.stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

        if self.session:
            await self.session.initialize()

            response = await self.session.list_tools()
            tools = response.tools
            print(f"\nConnected to MCP server with tools: {[tool.name for tool in tools]}")

            self.tool_map = {}
            for tool in tools:
                mcp_tool = MCPTool(session=self.session, tool_name=tool.name, tool_description=tool.description, tool_schema=tool.inputSchema)
                self.tool_map[tool.name] = mcp_tool
                mapped_name = tool.name.replace("-", "_")
                if mapped_name != tool.name:
                    mapped_tool = MCPTool(session=self.session, tool_name=tool.name, tool_description=tool.description, tool_schema=tool.inputSchema)
                    self.tool_map[mapped_name] = mapped_tool

    async def _execute_tools(self, tool_calls: list[dict[str, Any]]) -> dict[str, str]:
        """Execute tool calls."""
        tool_outputs: dict[str, str] = {}

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            if tool_name in self.tool_map:
                tool_instance = self.tool_map[tool_name]
                result = await tool_instance.async_forward(**tool_args)
                tool_outputs[tool_call["id"]] = result.to_string()
            else:
                tool_outputs[tool_call["id"]] = f"Error: Tool {tool_name} not found"

        return tool_outputs

    async def _cleanup(self) -> None:
        """Clean up the connection."""
        if hasattr(self, "exit_stack") and self.exit_stack:
            await self.exit_stack.aclose()


class MCPEnvironment(BaseEnv):
    """
    An environment for MCP-based tools that provides questions and evaluates responses.
    Uses a dedicated connection manager to avoid asyncio context issues.
    """

    # Class-level connection manager to share across instances
    _connection_manager: MCPConnectionManager | None = None
    _manager_lock = threading.Lock()

    def __init__(self, task: dict[str, Any] | None = None, mcp_server_command: str | None = None, mcp_server_args: list[str] | None = None, mcp_server_env: dict[str, str] | None = None, reward_fn: RewardFunction | None = None, max_steps: int = 10):
        """
        Initialize the MCPEnvironment.

        Args:
            task: Task information for the environment.
            mcp_server_command: Command to run the MCP server.
            mcp_server_args: Arguments for the MCP server.
            mcp_server_env: Environment variables for the MCP server.
            reward_fn: Reward function to use for evaluation.
            max_steps: Maximum number of steps allowed in the environment.
        """
        self.step_count = 0
        self.max_steps = max_steps
        self.task = task
        self.reward_fn = reward_fn
        if reward_fn is None:
            warnings.warn("No reward function specified, will get 0 reward.", stacklevel=2)
            self.reward_fn = zero_reward

        self.mcp_server_command = mcp_server_command
        self.mcp_server_args = mcp_server_args or []
        self.mcp_server_env = mcp_server_env

        # Initialize shared connection manager
        with MCPEnvironment._manager_lock:
            if MCPEnvironment._connection_manager is None and mcp_server_command is not None:
                MCPEnvironment._connection_manager = MCPConnectionManager(mcp_server_command, mcp_server_args, mcp_server_env)
                MCPEnvironment._connection_manager.start()

    def reset(self):
        """Reset the environment and return initial observations."""
        self.step_count = 0
        obs = self.task if self.task is not None else {}
        return obs, {}

    def step(self, action: Any):
        """
        Take a step in the environment based on the action.

        Args:
            action: Action from the agent (tool calls or final response)

        Returns:
            next_observations, rewards, terminateds, infos
        """
        if isinstance(action, dict):
            action = [action]
        self.step_count += 1

        reward = 0.0
        # Check if we should terminate
        done = self.step_count >= self.max_steps or isinstance(action, str)
        # Check if action contains a "finish" tool call
        if isinstance(action, list) and action:
            for tool_call in action:
                if tool_call.get("function", {}).get("name") == "finish":
                    done = True
                    break

        if done:
            # Agent is done - evaluate the response
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
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)

                    if isinstance(arguments, dict):
                        llm_response = arguments.get("response", "")
                    else:
                        llm_response = str(arguments)
                else:
                    llm_response = str(action)

            if self.reward_fn and self.task is not None:
                reward_output = self.reward_fn(task_info=self.task, action=llm_response)
                return {}, reward_output.reward, done, {"response": action, "metadata": reward_output.metadata}
            else:
                return {}, 0.0, done, {"response": action, "metadata": {}}

        # Execute tool calls using the connection manager
        tool_calls = action
        try:
            if MCPEnvironment._connection_manager is not None:
                tool_outputs = MCPEnvironment._connection_manager.execute_tool_calls(tool_calls)
                next_obs = {"tool_outputs": tool_outputs}
            else:
                next_obs = {"tool_outputs": {}}
        except Exception as e:
            print(f"Tool execution error: {e}")
            next_obs = {"tool_outputs": {}}

        return next_obs, reward, done, {"response": action, "metadata": {}}

    def close(self):
        """Clean up resources."""
        # Connection manager is shared and cleaned up globally
        pass

    @staticmethod
    def cleanup_global_resources():
        """Clean up global connection manager."""
        with MCPEnvironment._manager_lock:
            if MCPEnvironment._connection_manager:
                MCPEnvironment._connection_manager.stop()
                MCPEnvironment._connection_manager = None

    @staticmethod
    def from_dict(env_args: dict[str, Any]) -> "MCPEnvironment":
        mcp_server_command = env_args.pop("mcp_server_command", None)
        mcp_server_args = env_args.pop("mcp_server_args", None)
        mcp_server_env = env_args.pop("mcp_server_env", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)
        return MCPEnvironment(task=env_args, mcp_server_command=mcp_server_command, mcp_server_args=mcp_server_args, mcp_server_env=mcp_server_env, max_steps=max_steps, reward_fn=reward_fn)
