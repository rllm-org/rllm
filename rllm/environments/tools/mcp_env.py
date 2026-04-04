import asyncio
import json
import queue
import threading
import warnings
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError as err:
    raise ImportError("MCP environment requires extra dependencies. Install with: pip install rllm[tools]") from err

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import RewardFunction, zero_reward
from rllm.tools.mcp_tool import MCPTool


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env_items: tuple[tuple[str, str], ...] | None = None

    @property
    def args_list(self) -> list[str]:
        return list(self.args)

    @property
    def env_dict(self) -> dict[str, str] | None:
        if self.env_items is None:
            return None
        return dict(self.env_items)


def _normalize_server_spec(name: str, config: dict[str, Any]) -> MCPServerSpec:
    if not isinstance(config, dict):
        raise ValueError(f"Config for MCP server '{name}' must be a dictionary")

    command = config.get("command", config.get("mcp_server_command"))
    if not command:
        raise ValueError(f"Config for MCP server '{name}' must include 'command'")

    raw_args = config.get("args", config.get("mcp_server_args")) or []
    if not isinstance(raw_args, list | tuple):
        raise ValueError(f"Config for MCP server '{name}' must include list-like 'args'")
    args = tuple(str(arg) for arg in raw_args)

    raw_env = config.get("env", config.get("mcp_server_env"))
    if raw_env is not None and not isinstance(raw_env, dict):
        raise ValueError(f"Config for MCP server '{name}' must include dict-like 'env'")
    env_items = tuple(sorted((str(key), str(value)) for key, value in raw_env.items())) if raw_env is not None else None

    return MCPServerSpec(name=name, command=str(command), args=args, env_items=env_items)


def _normalize_mcp_servers(
    mcp_server_command: str | None,
    mcp_server_args: list[str] | None,
    mcp_server_env: dict[str, str] | None,
    mcp_servers: dict[str, dict[str, Any]] | None,
) -> dict[str, MCPServerSpec]:
    has_legacy_config = mcp_server_command is not None or mcp_server_args is not None or mcp_server_env is not None

    if mcp_servers is not None and has_legacy_config:
        raise ValueError("Cannot specify both legacy single-server MCP args and 'mcp_servers'")

    if mcp_servers is not None:
        if not isinstance(mcp_servers, dict):
            raise ValueError("'mcp_servers' must be a dictionary mapping server names to configs")
        return {server_name: _normalize_server_spec(server_name, server_config) for server_name, server_config in mcp_servers.items()}

    if mcp_server_command is None:
        return {}

    return {
        "default": MCPServerSpec(
            name="default",
            command=str(mcp_server_command),
            args=tuple(str(arg) for arg in (mcp_server_args or [])),
            env_items=tuple(sorted((str(key), str(value)) for key, value in mcp_server_env.items())) if mcp_server_env is not None else None,
        )
    }


def _tool_call_id(tool_call: Any, fallback_idx: int) -> str:
    if isinstance(tool_call, dict):
        tool_call_id = tool_call.get("id")
        if isinstance(tool_call_id, str) and tool_call_id:
            return tool_call_id
    return f"tool_call_{fallback_idx}"


def _tool_call_name(tool_call: Any) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None
    tool_name = function.get("name")
    if isinstance(tool_name, str) and tool_name:
        return tool_name
    return None


def _parse_tool_arguments(tool_call: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(tool_call, dict):
        return None, "Tool call must be a dictionary"

    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None, "Tool call missing function payload"

    raw_arguments = function.get("arguments", {})
    if isinstance(raw_arguments, dict):
        return raw_arguments, None
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return None, f"Invalid tool arguments JSON: {exc}"
        if not isinstance(parsed, dict):
            return None, "Tool arguments JSON must decode to an object"
        return parsed, None
    return None, "Tool arguments must be a dict or JSON string"


def _assign_missing_tool_call_ids(tool_calls: list[Any]) -> list[Any]:
    normalized_tool_calls: list[Any] = []
    for idx, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            normalized_tool_calls.append(tool_call)
            continue

        tool_call_id = tool_call.get("id")
        if isinstance(tool_call_id, str) and tool_call_id:
            normalized_tool_calls.append(tool_call)
            continue

        normalized_tool_call = dict(tool_call)
        normalized_tool_call["id"] = _tool_call_id(tool_call, idx)
        normalized_tool_calls.append(normalized_tool_call)

    return normalized_tool_calls


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

        for idx, tool_call in enumerate(tool_calls):
            tool_call_id = _tool_call_id(tool_call, idx)
            tool_name = _tool_call_name(tool_call)
            if tool_name is None:
                tool_outputs[tool_call_id] = "Error: Tool call missing function.name"
                continue

            tool_args, parse_error = _parse_tool_arguments(tool_call)
            if parse_error is not None or tool_args is None:
                tool_outputs[tool_call_id] = f"Error: {parse_error}"
                continue

            if tool_name in self.tool_map:
                tool_instance = self.tool_map[tool_name]
                result = await tool_instance.async_forward(**tool_args)
                tool_outputs[tool_call_id] = result.to_string()
            else:
                tool_outputs[tool_call_id] = f"Error: Tool {tool_name} not found"

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

    # Class-level connection managers shared across instances
    _connection_manager: MCPConnectionManager | None = None  # backward-compatible alias for single-server usage
    _connection_managers: dict[str, MCPConnectionManager] = {}
    _server_specs: dict[str, MCPServerSpec] = {}
    _manager_lock = threading.Lock()

    def __init__(
        self,
        task: dict[str, Any] | None = None,
        mcp_server_command: str | None = None,
        mcp_server_args: list[str] | None = None,
        mcp_server_env: dict[str, str] | None = None,
        mcp_servers: dict[str, dict[str, Any]] | None = None,
        tool_name_to_server_name: dict[str, str] | None = None,
        reward_fn: RewardFunction | None = None,
        max_steps: int = 10,
    ):
        """
        Initialize the MCPEnvironment.

        Args:
            task: Task information for the environment.
            mcp_server_command: Command to run the MCP server.
            mcp_server_args: Arguments for the MCP server.
            mcp_server_env: Environment variables for the MCP server.
            mcp_servers: Named MCP server configurations for multi-server routing.
            tool_name_to_server_name: Optional explicit mapping from public tool names
                to MCP server names.
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
        self.mcp_servers = _normalize_mcp_servers(mcp_server_command, mcp_server_args, mcp_server_env, mcp_servers)
        self.tool_name_to_server_name = dict(tool_name_to_server_name or {})
        self._resolved_tool_name_to_server_name: dict[str, str] = {}

        newly_created_server_names: list[str] = []
        try:
            newly_created_server_names = self._ensure_connection_managers()
            self._resolved_tool_name_to_server_name = self._build_tool_routing()
        except Exception:
            if newly_created_server_names:
                self._rollback_connection_managers(newly_created_server_names)
            raise

    @classmethod
    def _sync_connection_manager_alias_locked(cls) -> None:
        cls._connection_manager = next(iter(cls._connection_managers.values())) if len(cls._connection_managers) == 1 else None

    @classmethod
    def _rollback_connection_managers(cls, server_names: list[str]) -> None:
        managers_to_stop: list[MCPConnectionManager] = []
        with cls._manager_lock:
            for server_name in server_names:
                manager = cls._connection_managers.pop(server_name, None)
                cls._server_specs.pop(server_name, None)
                if manager is not None:
                    managers_to_stop.append(manager)
            cls._sync_connection_manager_alias_locked()

        for manager in managers_to_stop:
            try:
                manager.stop()
            except Exception:
                pass

    def _ensure_connection_managers(self) -> list[str]:
        newly_created_server_names: list[str] = []
        managers_to_stop: list[MCPConnectionManager] = []

        try:
            with MCPEnvironment._manager_lock:
                for server_name, server_spec in self.mcp_servers.items():
                    existing_spec = MCPEnvironment._server_specs.get(server_name)
                    if existing_spec is not None:
                        if existing_spec != server_spec:
                            raise ValueError(
                                f"MCP server '{server_name}' is already initialized with a different configuration"
                            )
                        continue

                    manager = MCPConnectionManager(
                        mcp_server_command=server_spec.command,
                        mcp_server_args=server_spec.args_list,
                        mcp_server_env=server_spec.env_dict,
                    )
                    try:
                        manager.start()
                    except Exception:
                        managers_to_stop.append(manager)
                        raise

                    MCPEnvironment._connection_managers[server_name] = manager
                    MCPEnvironment._server_specs[server_name] = server_spec
                    newly_created_server_names.append(server_name)

                MCPEnvironment._sync_connection_manager_alias_locked()
        except Exception:
            with MCPEnvironment._manager_lock:
                for server_name in newly_created_server_names:
                    manager = MCPEnvironment._connection_managers.pop(server_name, None)
                    MCPEnvironment._server_specs.pop(server_name, None)
                    if manager is not None:
                        managers_to_stop.append(manager)
                MCPEnvironment._sync_connection_manager_alias_locked()

            for manager in managers_to_stop:
                try:
                    manager.stop()
                except Exception:
                    pass
            raise

        return newly_created_server_names

    def _build_tool_routing(self) -> dict[str, str]:
        if not self.mcp_servers:
            return {}

        discovered_tool_servers: dict[str, set[str]] = {}
        for server_name in self.mcp_servers:
            manager = MCPEnvironment._connection_managers.get(server_name)
            if manager is None:
                continue
            for public_tool_name in getattr(manager, "tool_map", {}):
                discovered_tool_servers.setdefault(public_tool_name, set()).add(server_name)

        resolved: dict[str, str] = {}

        for public_tool_name, candidate_servers in discovered_tool_servers.items():
            explicit_server_name = self.tool_name_to_server_name.get(public_tool_name)
            if explicit_server_name is not None:
                if explicit_server_name not in candidate_servers:
                    raise ValueError(
                        f"Tool '{public_tool_name}' is not provided by mapped MCP server '{explicit_server_name}'"
                    )
                resolved[public_tool_name] = explicit_server_name
            elif len(candidate_servers) == 1:
                resolved[public_tool_name] = next(iter(candidate_servers))
            else:
                raise ValueError(
                    f"Tool '{public_tool_name}' is provided by multiple MCP servers {sorted(candidate_servers)}. "
                    "Supply 'tool_name_to_server_name' to disambiguate."
                )

        for public_tool_name, mapped_server_name in self.tool_name_to_server_name.items():
            if mapped_server_name not in self.mcp_servers:
                raise ValueError(
                    f"Tool mapping for '{public_tool_name}' references unknown MCP server '{mapped_server_name}'"
                )
            if public_tool_name not in discovered_tool_servers:
                raise ValueError(
                    f"Tool mapping for '{public_tool_name}' does not match any discovered tool on the configured MCP servers"
                )

        return resolved

    @staticmethod
    def _is_finish_tool_call(tool_call: Any) -> bool:
        return _tool_call_name(tool_call) == "finish"

    def _extract_final_response(self, action: list[dict[str, Any]] | str) -> str:
        if isinstance(action, str):
            return action

        finish_action = None
        for tool_call in action:
            if self._is_finish_tool_call(tool_call):
                finish_action = tool_call
                break

        if finish_action is None:
            return str(action)

        arguments, parse_error = _parse_tool_arguments(finish_action)
        if parse_error is not None or arguments is None:
            return str(action)

        response = arguments.get("response", "")
        return response if isinstance(response, str) else str(response)

    def _execute_tool_calls_by_server(self, tool_calls: list[dict[str, Any]]) -> dict[str, str]:
        tool_calls = _assign_missing_tool_call_ids(tool_calls)
        tool_outputs: dict[str, str] = {}
        grouped_calls: dict[str, list[dict[str, Any]]] = {}

        for idx, tool_call in enumerate(tool_calls):
            tool_call_id = _tool_call_id(tool_call, idx)
            tool_name = _tool_call_name(tool_call)
            if tool_name is None:
                tool_outputs[tool_call_id] = "Error: Tool call missing function.name"
                continue

            server_name = self._resolved_tool_name_to_server_name.get(tool_name)
            if server_name is None and len(self.mcp_servers) == 1:
                # Preserve legacy single-server behavior where every tool call is
                # forwarded to the sole configured MCP server.
                server_name = next(iter(self.mcp_servers))
            if server_name is None:
                tool_outputs[tool_call_id] = f"Error: Tool {tool_name} not found"
                continue

            grouped_calls.setdefault(server_name, []).append(tool_call)

        for server_name, grouped_tool_calls in grouped_calls.items():
            manager = MCPEnvironment._connection_managers.get(server_name)
            if manager is None:
                for idx, tool_call in enumerate(grouped_tool_calls):
                    tool_outputs[_tool_call_id(tool_call, idx)] = f"Error: MCP server {server_name} is not available"
                continue

            try:
                tool_outputs.update(manager.execute_tool_calls(grouped_tool_calls))
            except Exception as exc:
                for idx, tool_call in enumerate(grouped_tool_calls):
                    tool_outputs[_tool_call_id(tool_call, idx)] = f"Error: MCP server {server_name} failed: {exc}"

        ordered_tool_outputs: dict[str, str] = {}
        for idx, tool_call in enumerate(tool_calls):
            tool_call_id = _tool_call_id(tool_call, idx)
            if tool_call_id in tool_outputs:
                ordered_tool_outputs[tool_call_id] = tool_outputs[tool_call_id]

        for tool_call_id, tool_output in tool_outputs.items():
            if tool_call_id not in ordered_tool_outputs:
                ordered_tool_outputs[tool_call_id] = tool_output

        return ordered_tool_outputs

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
                if self._is_finish_tool_call(tool_call):
                    done = True
                    break

        if done:
            # Agent is done - evaluate the response
            llm_response = self._extract_final_response(action)

            if self.reward_fn and self.task is not None:
                reward_output = self.reward_fn(task_info=self.task, action=llm_response)
                return {}, reward_output.reward, done, {"response": action, "metadata": reward_output.metadata}
            else:
                return {}, 0.0, done, {"response": action, "metadata": {}}

        # Execute tool calls using the connection manager
        tool_calls = action
        try:
            tool_outputs = self._execute_tool_calls_by_server(tool_calls) if isinstance(tool_calls, list) else {}
            next_obs = {"tool_outputs": tool_outputs}
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
        """Clean up global connection managers."""
        managers_to_stop: list[MCPConnectionManager] = []
        with MCPEnvironment._manager_lock:
            managers_to_stop = list(MCPEnvironment._connection_managers.values())
            MCPEnvironment._connection_managers = {}
            MCPEnvironment._server_specs = {}
            MCPEnvironment._sync_connection_manager_alias_locked()

        for manager in managers_to_stop:
            try:
                manager.stop()
            except Exception:
                pass

    @staticmethod
    def from_dict(env_args: dict[str, Any]) -> "MCPEnvironment":
        env_args = dict(env_args)
        mcp_server_command = env_args.pop("mcp_server_command", None)
        mcp_server_args = env_args.pop("mcp_server_args", None)
        mcp_server_env = env_args.pop("mcp_server_env", None)
        mcp_servers = env_args.pop("mcp_servers", None)
        tool_name_to_server_name = env_args.pop("tool_name_to_server_name", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)
        return MCPEnvironment(
            task=env_args,
            mcp_server_command=mcp_server_command,
            mcp_server_args=mcp_server_args,
            mcp_server_env=mcp_server_env,
            mcp_servers=mcp_servers,
            tool_name_to_server_name=tool_name_to_server_name,
            max_steps=max_steps,
            reward_fn=reward_fn,
        )
