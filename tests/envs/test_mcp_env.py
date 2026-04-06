from unittest.mock import Mock, patch

import pytest

from rllm.environments.tools.mcp_env import MCPConnectionManager, MCPEnvironment
from rllm.rewards.reward_fn import RewardFunction, RewardOutput


class MockRewardFunction(RewardFunction):
    """Mock reward function for testing."""

    def __call__(self, task_info, action, **kwargs):
        # Simple mock reward based on action content
        reward = 1.0 if "correct" in str(action).lower() else 0.0
        metadata = {"evaluated": True, "action": action}
        return RewardOutput(reward=reward, metadata=metadata)


@pytest.fixture(autouse=True)
def reset_mcp_environment_state():
    MCPEnvironment._connection_manager = None
    MCPEnvironment._connection_managers = {}
    MCPEnvironment._server_specs = {}
    yield
    MCPEnvironment._connection_manager = None
    MCPEnvironment._connection_managers = {}
    MCPEnvironment._server_specs = {}


def make_start_side_effect(command_to_tools):
    def _start(manager):
        manager.running = True
        manager.tool_map = command_to_tools.get(manager.mcp_server_command, {})

    return _start


class TestMCPConnectionManager:
    """Test suite for MCPConnectionManager class."""

    def test_init(self):
        """Test MCPConnectionManager initialization."""
        manager = MCPConnectionManager(mcp_server_command="test_command", mcp_server_args=["--arg1", "--arg2"], mcp_server_env={"VAR": "value"})

        assert manager.mcp_server_command == "test_command"
        assert manager.mcp_server_args == ["--arg1", "--arg2"]
        assert manager.mcp_server_env == {"VAR": "value"}
        assert manager.running is False
        assert manager.tool_map == {}

    def test_init_default_args(self):
        """Test MCPConnectionManager initialization with default arguments."""
        manager = MCPConnectionManager("test_command")

        assert manager.mcp_server_command == "test_command"
        assert manager.mcp_server_args == []
        assert manager.mcp_server_env is None

    @patch("threading.Thread")
    @patch("queue.Queue")
    def test_start(self, mock_queue, mock_thread):
        """Test starting the connection manager."""
        # Mock the response queue to simulate successful initialization
        mock_response_queue = Mock()
        mock_response_queue.get.return_value = ("success", {"tool1": "mock_tool"})
        mock_queue.return_value = mock_response_queue

        manager = MCPConnectionManager("test_command")

        # Mock the worker thread
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        manager.start()

        assert manager.running is True
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()

    @patch("threading.Thread")
    @patch("queue.Queue")
    def test_start_initialization_error(self, mock_queue, mock_thread):
        """Test starting the connection manager with initialization error."""
        # Mock the response queue to simulate initialization error
        mock_response_queue = Mock()
        mock_response_queue.get.return_value = ("error", "Connection failed")
        mock_queue.return_value = mock_response_queue

        manager = MCPConnectionManager("test_command")

        # Mock the worker thread
        mock_thread_instance = Mock()
        mock_thread.return_value = mock_thread_instance

        with pytest.raises(Exception, match="Failed to initialize MCP connection"):
            manager.start()

    def test_stop_not_running(self):
        """Test stopping the connection manager when not running."""
        manager = MCPConnectionManager("test_command")

        # Should not raise any errors
        manager.stop()
        assert manager.running is False

    @patch("threading.Thread")
    def test_stop_running(self, mock_thread):
        """Test stopping a running connection manager."""
        manager = MCPConnectionManager("test_command")
        manager.running = True
        manager.worker_thread = Mock()

        manager.stop()

        assert manager.running is False
        manager.worker_thread.join.assert_called_once_with(timeout=5)

    @patch("queue.Queue")
    def test_execute_tool_calls_not_running(self, mock_queue):
        """Test executing tool calls when manager is not running."""
        manager = MCPConnectionManager("test_command")

        with pytest.raises(Exception, match="Connection manager not running"):
            manager.execute_tool_calls([])

    @patch("queue.Queue")
    def test_execute_tool_calls_success(self, mock_queue):
        """Test successful tool call execution."""
        # Mock the response queue
        mock_response_queue = Mock()
        mock_response_queue.get.return_value = ("success", {"call_1": "result"})
        mock_queue.return_value = mock_response_queue

        manager = MCPConnectionManager("test_command")
        manager.running = True

        tool_calls = [{"id": "call_1", "function": {"name": "test_tool"}}]
        result = manager.execute_tool_calls(tool_calls)

        assert result == {"call_1": "result"}

    @patch("queue.Queue")
    def test_execute_tool_calls_error(self, mock_queue):
        """Test tool call execution with error."""
        # Mock the response queue
        mock_response_queue = Mock()
        mock_response_queue.get.return_value = ("error", "Tool execution failed")
        mock_queue.return_value = mock_response_queue

        manager = MCPConnectionManager("test_command")
        manager.running = True

        tool_calls = [{"id": "call_1", "function": {"name": "test_tool"}}]

        with pytest.raises(Exception, match="Tool execution failed"):
            manager.execute_tool_calls(tool_calls)


class TestMCPEnvironment:
    """Test suite for MCPEnvironment class."""

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_init_default(self, mock_init, mock_start):
        """Test MCPEnvironment initialization with default parameters."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        env = MCPEnvironment(mcp_server_command="test_command")
        assert env.step_count == 0
        assert env.max_steps == 10
        assert env.task is None
        assert env.reward_fn is not None  # Should use zero_reward

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_init_with_custom_parameters(self, mock_init, mock_start):
        """Test MCPEnvironment initialization with custom parameters."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        max_steps = 5

        env = MCPEnvironment(task=task, mcp_server_command="test_command", mcp_server_args=["--arg1"], mcp_server_env={"VAR": "value"}, reward_fn=reward_fn, max_steps=max_steps)

        assert env.task == task
        assert env.reward_fn == reward_fn
        assert env.max_steps == max_steps
        assert env.mcp_server_command == "test_command"
        assert env.mcp_server_args == ["--arg1"]
        assert env.mcp_server_env == {"VAR": "value"}

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_init_no_reward_function_warning(self, mock_init, mock_start):
        """Test that warning is issued when no reward function is provided."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        with pytest.warns(UserWarning, match="No reward function specified"):
            env = MCPEnvironment(mcp_server_command="test_command", reward_fn=None)
            assert env.reward_fn is not None  # Should use zero_reward

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_reset(self, mock_init, mock_start):
        """Test the reset method."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test question"}
        env = MCPEnvironment(task=task, max_steps=5, mcp_server_command="test_command")

        # Set some non-initial state
        env.step_count = 3

        obs, info = env.reset()

        assert env.step_count == 0
        assert obs == task
        assert isinstance(info, dict)

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_with_string_action(self, mock_init, mock_start):
        """Test stepping with string action (should terminate)."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = MCPEnvironment(task=task, reward_fn=reward_fn, mcp_server_command="test_command")
        env.reset()

        action = "This is my final answer"
        obs, reward, done, info = env.step(action)

        assert obs == {}
        assert reward == 0.0  # MockRewardFunction returns 0 for non-"correct" answers
        assert done is True
        assert info["response"] == action
        assert "metadata" in info

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_with_string_action_correct(self, mock_init, mock_start):
        """Test stepping with string action containing 'correct'."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = MCPEnvironment(task=task, reward_fn=reward_fn, mcp_server_command="test_command")
        env.reset()

        action = "The correct answer is 42"
        obs, reward, done, info = env.step(action)

        assert reward == 1.0  # MockRewardFunction returns 1 for "correct" answers
        assert done is True

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_with_finish_tool_call(self, mock_init, mock_start):
        """Test stepping with finish tool call."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = MCPEnvironment(task=task, reward_fn=reward_fn, mcp_server_command="test_command")
        env.reset()

        action = [{"id": "call_1", "function": {"name": "finish", "arguments": {"response": "Final answer"}}}]

        obs, reward, done, info = env.step(action)

        assert obs == {}
        assert done is True
        assert info["response"] == action

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_with_regular_tool_calls(self, mock_init, mock_start):
        """Test stepping with regular tool calls."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test question"}
        env = MCPEnvironment(task=task, mcp_server_command="test_command")
        env.reset()

        # Mock the connection manager
        mock_manager = Mock()
        mock_manager.execute_tool_calls.return_value = {"call_1": "Tool output"}
        MCPEnvironment._connection_manager = mock_manager
        MCPEnvironment._connection_managers = {"default": mock_manager}

        action = [{"id": "call_1", "function": {"name": "search", "arguments": {"query": "test"}}}]

        obs, reward, done, info = env.step(action)

        assert obs == {"tool_outputs": {"call_1": "Tool output"}}
        assert reward == 0
        assert done is False
        assert info["response"] == action
        mock_manager.execute_tool_calls.assert_called_once_with(action)

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_max_steps_termination(self, mock_init, mock_start):
        """Test that environment terminates when max_steps is reached."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        env = MCPEnvironment(max_steps=2, mcp_server_command="test_command")
        env.reset()

        # Mock the connection manager
        mock_manager = Mock()
        mock_manager.execute_tool_calls.return_value = {"call_1": "Tool output"}
        MCPEnvironment._connection_manager = mock_manager
        MCPEnvironment._connection_managers = {"default": mock_manager}

        # Take steps until max_steps
        for i in range(2):
            action = [{"id": "call_1", "function": {"name": "search", "arguments": {"query": "test"}}}]

            obs, reward, done, info = env.step(action)

            if i == 1:  # Last step
                assert done is True
            else:
                assert done is False

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_with_dict_action(self, mock_init, mock_start):
        """Test stepping with dict action (converted to list)."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        env = MCPEnvironment(mcp_server_command="test_command")
        env.reset()

        # Mock the connection manager
        mock_manager = Mock()
        mock_manager.execute_tool_calls.return_value = {"call_1": "Tool output"}
        MCPEnvironment._connection_manager = mock_manager
        MCPEnvironment._connection_managers = {"default": mock_manager}

        action = {"id": "call_1", "function": {"name": "search", "arguments": {"query": "test"}}}

        obs, reward, done, info = env.step(action)

        # Should convert dict to list
        mock_manager.execute_tool_calls.assert_called_once_with([action])

    def test_close_with_connection(self):
        """Test close method with active connection."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        with patch.object(MCPConnectionManager, "start"), patch.object(MCPConnectionManager, "__init__", return_value=None):
            env = MCPEnvironment(mcp_server_command="test_command")

            # Should not raise any errors - close method does nothing
            env.close()

    def test_close_no_connection(self):
        """Test close method without connection."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        with patch.object(MCPConnectionManager, "start"), patch.object(MCPConnectionManager, "__init__", return_value=None):
            env = MCPEnvironment(mcp_server_command="test_command")

            # Should not raise any errors
            env.close()

    def test_cleanup_global_resources_no_manager(self):
        """Test cleanup_global_resources when no manager exists."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        # Should not raise any errors
        MCPEnvironment.cleanup_global_resources()

    def test_cleanup_global_resources_with_manager(self):
        """Test cleanup_global_resources with existing manager."""
        mock_manager = Mock()
        MCPEnvironment._connection_manager = mock_manager
        MCPEnvironment._connection_managers = {"default": mock_manager}

        MCPEnvironment.cleanup_global_resources()

        mock_manager.stop.assert_called_once()
        assert MCPEnvironment._connection_manager is None
        assert MCPEnvironment._connection_managers == {}
        assert MCPEnvironment._server_specs == {}

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_idx_property(self, mock_init, mock_start):
        """Test the idx property from BaseEnv."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        env = MCPEnvironment(mcp_server_command="test_command")

        # Initially should be None
        assert env.idx is None

        # Should be able to set and get
        env.idx = 5
        assert env.idx == 5

    def test_is_multithread_safe(self):
        """Test the is_multithread_safe static method."""
        assert MCPEnvironment.is_multithread_safe() is True

    def test_from_dict(self):
        """Test creating environment from dictionary."""
        env_args = {
            "question": "Test question",
            "mcp_server_command": "test_command",
            "mcp_server_args": ["--arg1"],
            "mcp_server_env": {"VAR": "value"},
            "max_steps": 15,
            "reward_fn": MockRewardFunction(),
        }

        with patch.object(MCPConnectionManager, "start"), patch.object(MCPConnectionManager, "__init__", return_value=None):
            # Clear any existing manager
            MCPEnvironment._connection_manager = None

            env = MCPEnvironment.from_dict(env_args)

            assert isinstance(env, MCPEnvironment)
            assert env.max_steps == 15
            assert env.task == {"question": "Test question"}
            assert env.mcp_server_command == "test_command"
            assert env.mcp_server_args == ["--arg1"]
            assert env.mcp_server_env == {"VAR": "value"}

    def test_from_dict_minimal(self):
        """Test creating environment from minimal dictionary."""
        env_args = {"question": "Test question"}

        with patch.object(MCPConnectionManager, "start"), patch.object(MCPConnectionManager, "__init__", return_value=None):
            # Clear any existing manager
            MCPEnvironment._connection_manager = None

            env = MCPEnvironment.from_dict(env_args)

            assert isinstance(env, MCPEnvironment)
            assert env.task == {"question": "Test question"}
            assert env.max_steps == 10  # Default value

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_full_interaction_flow(self, mock_init, mock_start):
        """Test a complete interaction flow."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "What is the capital of France?"}
        reward_fn = MockRewardFunction()
        env = MCPEnvironment(task=task, reward_fn=reward_fn, max_steps=3, mcp_server_command="test_command")

        # Reset environment
        obs, info = env.reset()
        assert obs == task
        assert env.step_count == 0

        # Step 1: Use a tool
        mock_manager = Mock()
        mock_manager.execute_tool_calls.return_value = {"call_1": "Paris is the capital of France"}
        MCPEnvironment._connection_manager = mock_manager
        MCPEnvironment._connection_managers = {"default": mock_manager}

        action1 = [{"id": "call_1", "function": {"name": "search", "arguments": {"query": "capital of France"}}}]

        obs1, reward1, done1, info1 = env.step(action1)

        assert obs1 == {"tool_outputs": {"call_1": "Paris is the capital of France"}}
        assert reward1 == 0
        assert done1 is False
        assert env.step_count == 1

        # Step 2: Finish with answer
        action2 = "The correct answer is Paris"
        obs2, reward2, done2, info2 = env.step(action2)

        assert obs2 == {}
        assert reward2 == 1.0  # MockRewardFunction gives 1.0 for "correct"
        assert done2 is True
        assert info2["response"] == action2

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_with_finish_tool_call_with_arguments(self, mock_init, mock_start):
        """Test stepping with finish tool call containing arguments."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = MCPEnvironment(task=task, reward_fn=reward_fn, mcp_server_command="test_command")
        env.reset()

        action = [{"id": "call_1", "function": {"name": "finish", "arguments": {"response": "The correct final answer"}}}]

        obs, reward, done, info = env.step(action)

        assert reward == 1.0  # Should extract response from arguments
        assert done is True

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_step_with_tool_execution_error(self, mock_init, mock_start):
        """Test stepping when tool execution fails."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        env = MCPEnvironment(mcp_server_command="test_command")
        env.reset()

        # Mock the connection manager to raise an error
        mock_manager = Mock()
        mock_manager.execute_tool_calls.side_effect = Exception("Tool execution failed")
        MCPEnvironment._connection_manager = mock_manager
        MCPEnvironment._connection_managers = {"default": mock_manager}

        action = [{"id": "call_1", "function": {"name": "search", "arguments": {"query": "test"}}}]

        obs, reward, done, info = env.step(action)
        assert obs == {"tool_outputs": {"call_1": "Error: MCP server default failed: Tool execution failed"}}
        assert reward == 0
        assert done is False
        assert info["response"] == action

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_edge_cases(self, mock_init, mock_start):
        """Test various edge cases."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        env = MCPEnvironment(mcp_server_command="test_command")
        env.reset()

        # Mock the connection manager
        mock_manager = Mock()
        mock_manager.execute_tool_calls.return_value = {}
        MCPEnvironment._connection_manager = mock_manager
        MCPEnvironment._connection_managers = {"default": mock_manager}

        # Empty action list
        obs, reward, done, info = env.step([])
        assert obs == {"tool_outputs": {}}
        assert done is False

        # None action - check how it's handled
        try:
            obs, reward, done, info = env.step(None)
            # If it doesn't raise an error, that's also fine
            assert isinstance(obs, dict)
        except (TypeError, AttributeError):
            # This is expected behavior
            pass

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_reward_function_integration(self, mock_init, mock_start):
        """Test integration with different reward functions."""

        class CustomReward(RewardFunction):
            def __call__(self, task_info, action, **kwargs):
                score = len(str(action)) / 10.0  # Simple length-based reward
                return RewardOutput(reward=score, metadata={"length": len(str(action))})

        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        task = {"question": "Test"}
        reward_fn = CustomReward()
        env = MCPEnvironment(task=task, reward_fn=reward_fn, mcp_server_command="test_command")

        env.reset()

        # Test with different length responses
        short_action = "Short"
        long_action = "This is a much longer response that should get a higher reward"

        _, reward1, _, info1 = env.step(short_action)
        env.reset()
        _, reward2, _, info2 = env.step(long_action)

        assert reward2 > reward1  # Longer response should get higher reward
        assert info1["metadata"]["length"] < info2["metadata"]["length"]

    def test_connection_manager_thread_safety(self):
        """Test that connection manager handles thread safety correctly."""
        # Both environments should be able to access the class-level manager
        assert hasattr(MCPEnvironment, "_connection_manager")
        assert hasattr(MCPEnvironment, "_connection_managers")
        assert hasattr(MCPEnvironment, "_server_specs")
        assert hasattr(MCPEnvironment, "_manager_lock")

    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    @patch.object(MCPConnectionManager, "start")
    def test_connection_manager_singleton_behavior(self, mock_start, mock_init):
        """Test that connection manager behaves as singleton."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        MCPEnvironment(mcp_server_command="test_command")
        MCPEnvironment(mcp_server_command="test_command")

        # Both environments should use the same manager
        assert MCPEnvironment._connection_manager is not None
        assert len(MCPEnvironment._connection_managers) == 1
        assert mock_init.call_count == 1
        assert mock_start.call_count == 1

    @patch.object(MCPConnectionManager, "start")
    @patch.object(MCPConnectionManager, "__init__", return_value=None)
    def test_malformed_tool_call_handling(self, mock_init, mock_start):
        """Test handling of malformed tool calls."""
        # Clear any existing manager
        MCPEnvironment._connection_manager = None

        env = MCPEnvironment(mcp_server_command="test_command")
        env.reset()

        # Mock the connection manager
        mock_manager = Mock()
        mock_manager.execute_tool_calls.return_value = {"call_1": "Tool output"}
        MCPEnvironment._connection_manager = mock_manager
        MCPEnvironment._connection_managers = {"default": mock_manager}

        # Malformed action (missing required fields)
        action = [{"id": "call_1"}]  # Missing function field

        obs, reward, done, info = env.step(action)

        assert obs == {"tool_outputs": {"call_1": "Error: Tool call missing function.name"}}
        assert done is False
        mock_manager.execute_tool_calls.assert_not_called()

    @patch.object(MCPConnectionManager, "start", autospec=True)
    def test_init_with_multiple_servers(self, mock_start):
        """Test initializing MCPEnvironment with multiple named servers."""
        mock_start.side_effect = make_start_side_effect(
            {
                "search-command": {"search": Mock()},
                "wiki-command": {"lookup": Mock()},
            }
        )

        env = MCPEnvironment(
            mcp_servers={
                "search_server": {"command": "search-command"},
                "wiki_server": {"command": "wiki-command"},
            }
        )

        assert set(env.mcp_servers) == {"search_server", "wiki_server"}
        assert set(MCPEnvironment._connection_managers) == {"search_server", "wiki_server"}
        assert MCPEnvironment._connection_manager is None
        assert env._resolved_tool_name_to_server_name == {
            "search": "search_server",
            "lookup": "wiki_server",
        }
        assert mock_start.call_count == 2

    @patch.object(MCPConnectionManager, "start", autospec=True)
    def test_same_server_name_with_different_config_raises(self, mock_start):
        """Test that reusing a server name with a different config fails fast."""
        mock_start.side_effect = make_start_side_effect({"search-command": {"search": Mock()}})

        MCPEnvironment(mcp_servers={"shared": {"command": "search-command"}})

        with pytest.raises(ValueError, match="already initialized with a different configuration"):
            MCPEnvironment(mcp_servers={"shared": {"command": "different-command"}})

    @patch.object(MCPConnectionManager, "start", autospec=True)
    def test_explicit_tool_name_to_server_name_resolves_ambiguity(self, mock_start):
        """Test that explicit tool routing resolves duplicate tool names."""
        mock_start.side_effect = make_start_side_effect(
            {
                "command-a": {"shared_tool": Mock()},
                "command-b": {"shared_tool": Mock()},
            }
        )

        env = MCPEnvironment(
            mcp_servers={
                "server_a": {"command": "command-a"},
                "server_b": {"command": "command-b"},
            },
            tool_name_to_server_name={"shared_tool": "server_b"},
        )

        assert env._resolved_tool_name_to_server_name == {"shared_tool": "server_b"}

    @patch.object(MCPConnectionManager, "start", autospec=True)
    def test_duplicate_public_tool_name_without_mapping_raises(self, mock_start):
        """Test that duplicate tool names across servers require explicit routing."""
        mock_start.side_effect = make_start_side_effect(
            {
                "command-a": {"shared_tool": Mock()},
                "command-b": {"shared_tool": Mock()},
            }
        )

        with pytest.raises(ValueError, match="Tool 'shared_tool' is provided by multiple MCP servers"):
            MCPEnvironment(
                mcp_servers={
                    "server_a": {"command": "command-a"},
                    "server_b": {"command": "command-b"},
                }
            )

    @patch.object(MCPConnectionManager, "start", autospec=True)
    def test_step_routes_tool_calls_to_correct_server(self, mock_start):
        """Test that tool calls are routed to the correct MCP server."""
        mock_start.side_effect = make_start_side_effect(
            {
                "search-command": {"search": Mock()},
                "wiki-command": {"lookup": Mock()},
            }
        )
        env = MCPEnvironment(
            mcp_servers={
                "search_server": {"command": "search-command"},
                "wiki_server": {"command": "wiki-command"},
            }
        )
        env.reset()

        search_manager = MCPEnvironment._connection_managers["search_server"]
        wiki_manager = MCPEnvironment._connection_managers["wiki_server"]
        search_manager.execute_tool_calls = Mock(return_value={"call_1": "Search output"})
        wiki_manager.execute_tool_calls = Mock(return_value={"call_2": "Lookup output"})

        action = [
            {"id": "call_1", "function": {"name": "search", "arguments": {"query": "France"}}},
            {"id": "call_2", "function": {"name": "lookup", "arguments": {"topic": "Paris"}}},
        ]

        obs, reward, done, info = env.step(action)

        assert obs == {"tool_outputs": {"call_1": "Search output", "call_2": "Lookup output"}}
        assert reward == 0
        assert done is False
        assert info["response"] == action
        search_manager.execute_tool_calls.assert_called_once_with([action[0]])
        wiki_manager.execute_tool_calls.assert_called_once_with([action[1]])

    @patch.object(MCPConnectionManager, "start", autospec=True)
    def test_partial_server_failure_does_not_erase_other_outputs(self, mock_start):
        """Test that one server failure does not discard successful tool outputs."""
        mock_start.side_effect = make_start_side_effect(
            {
                "search-command": {"search": Mock()},
                "wiki-command": {"lookup": Mock()},
            }
        )
        env = MCPEnvironment(
            mcp_servers={
                "search_server": {"command": "search-command"},
                "wiki_server": {"command": "wiki-command"},
            }
        )
        env.reset()

        search_manager = MCPEnvironment._connection_managers["search_server"]
        wiki_manager = MCPEnvironment._connection_managers["wiki_server"]
        search_manager.execute_tool_calls = Mock(return_value={"call_1": "Search output"})
        wiki_manager.execute_tool_calls = Mock(side_effect=Exception("wiki unavailable"))

        action = [
            {"id": "call_1", "function": {"name": "search", "arguments": {"query": "France"}}},
            {"id": "call_2", "function": {"name": "lookup", "arguments": {"topic": "Paris"}}},
        ]

        obs, reward, done, info = env.step(action)

        assert obs == {
            "tool_outputs": {
                "call_1": "Search output",
                "call_2": "Error: MCP server wiki_server failed: wiki unavailable",
            }
        }
        assert reward == 0
        assert done is False
        assert info["response"] == action

    @patch.object(MCPConnectionManager, "start", autospec=True)
    def test_step_assigns_missing_tool_call_ids_across_servers(self, mock_start):
        """Test that synthetic tool call ids stay unique across routed server groups."""
        mock_start.side_effect = make_start_side_effect(
            {
                "search-command": {"search": Mock()},
                "wiki-command": {"lookup": Mock()},
            }
        )
        env = MCPEnvironment(
            mcp_servers={
                "search_server": {"command": "search-command"},
                "wiki_server": {"command": "wiki-command"},
            }
        )
        env.reset()

        search_manager = MCPEnvironment._connection_managers["search_server"]
        wiki_manager = MCPEnvironment._connection_managers["wiki_server"]
        search_manager.execute_tool_calls = Mock(side_effect=lambda tool_calls: {tool_calls[0]["id"]: "Search output"})
        wiki_manager.execute_tool_calls = Mock(side_effect=lambda tool_calls: {tool_calls[0]["id"]: "Lookup output"})

        action = [
            {"function": {"name": "search", "arguments": {"query": "France"}}},
            {"function": {"name": "lookup", "arguments": {"topic": "Paris"}}},
        ]

        obs, reward, done, info = env.step(action)

        assert obs == {
            "tool_outputs": {
                "tool_call_0": "Search output",
                "tool_call_1": "Lookup output",
            }
        }
        assert reward == 0
        assert done is False
        assert info["response"] == action

    @patch.object(MCPConnectionManager, "start", autospec=True)
    def test_step_preserves_interleaved_tool_output_order_across_servers(self, mock_start):
        """Test that output ordering follows the original tool-call order."""
        mock_start.side_effect = make_start_side_effect(
            {
                "search-command": {"search": Mock()},
                "wiki-command": {"lookup": Mock()},
            }
        )
        env = MCPEnvironment(
            mcp_servers={
                "search_server": {"command": "search-command"},
                "wiki_server": {"command": "wiki-command"},
            }
        )
        env.reset()

        search_manager = MCPEnvironment._connection_managers["search_server"]
        wiki_manager = MCPEnvironment._connection_managers["wiki_server"]
        search_manager.execute_tool_calls = Mock(
            return_value={
                "call_1": "Search output 1",
                "call_3": "Search output 2",
            }
        )
        wiki_manager.execute_tool_calls = Mock(return_value={"call_2": "Lookup output"})

        action = [
            {"id": "call_1", "function": {"name": "search", "arguments": {"query": "France"}}},
            {"id": "call_2", "function": {"name": "lookup", "arguments": {"topic": "Paris"}}},
            {"id": "call_3", "function": {"name": "search", "arguments": {"query": "Europe"}}},
        ]

        obs, reward, done, info = env.step(action)

        assert list(obs["tool_outputs"]) == ["call_1", "call_2", "call_3"]
        assert reward == 0
        assert done is False
        assert info["response"] == action

    @patch.object(MCPConnectionManager, "stop", autospec=True)
    @patch.object(MCPConnectionManager, "start", autospec=True)
    def test_start_failure_rolls_back_previously_started_managers(self, mock_start, mock_stop):
        """Test that manager startup failures do not leave partial global state behind."""

        def _start(manager):
            if manager.mcp_server_command == "search-command":
                manager.running = True
                manager.tool_map = {"search": Mock()}
                return
            raise RuntimeError("startup failed")

        mock_start.side_effect = _start

        with pytest.raises(RuntimeError, match="startup failed"):
            MCPEnvironment(
                mcp_servers={
                    "search_server": {"command": "search-command"},
                    "wiki_server": {"command": "wiki-command"},
                }
            )

        assert MCPEnvironment._connection_manager is None
        assert MCPEnvironment._connection_managers == {}
        assert MCPEnvironment._server_specs == {}
        assert mock_stop.call_count == 2

    @patch.object(MCPConnectionManager, "stop", autospec=True)
    @patch.object(MCPConnectionManager, "start", autospec=True)
    def test_invalid_tool_mapping_rolls_back_new_managers(self, mock_start, mock_stop):
        """Test that routing validation failures clean up newly started managers."""
        mock_start.side_effect = make_start_side_effect(
            {
                "search-command": {"search": Mock()},
                "wiki-command": {"lookup": Mock()},
            }
        )

        with pytest.raises(ValueError, match="does not match any discovered tool"):
            MCPEnvironment(
                mcp_servers={
                    "search_server": {"command": "search-command"},
                    "wiki_server": {"command": "wiki-command"},
                },
                tool_name_to_server_name={"missing_tool": "search_server"},
            )

        assert MCPEnvironment._connection_manager is None
        assert MCPEnvironment._connection_managers == {}
        assert MCPEnvironment._server_specs == {}
        assert mock_stop.call_count == 2

    @patch.object(MCPConnectionManager, "start", autospec=True)
    def test_from_dict_with_mcp_servers(self, mock_start):
        """Test creating an environment from dictionary with multi-server config."""
        mock_start.side_effect = make_start_side_effect(
            {
                "search-command": {"search": Mock()},
                "wiki-command": {"lookup": Mock()},
            }
        )
        env_args = {
            "question": "Test question",
            "mcp_servers": {
                "search_server": {"command": "search-command"},
                "wiki_server": {"command": "wiki-command"},
            },
            "tool_name_to_server_name": {"lookup": "wiki_server"},
            "max_steps": 15,
            "reward_fn": MockRewardFunction(),
        }

        env = MCPEnvironment.from_dict(env_args)

        assert isinstance(env, MCPEnvironment)
        assert env.task == {"question": "Test question"}
        assert env.max_steps == 15
        assert set(env.mcp_servers) == {"search_server", "wiki_server"}
        assert env.tool_name_to_server_name == {"lookup": "wiki_server"}

    @patch.object(MCPConnectionManager, "start", autospec=True)
    def test_from_dict_does_not_mutate_input(self, mock_start):
        """Test that from_dict does not mutate the provided env_args dictionary."""
        mock_start.side_effect = make_start_side_effect({"search-command": {"search": Mock()}})
        env_args = {
            "question": "Test question",
            "mcp_servers": {"search_server": {"command": "search-command"}},
            "tool_name_to_server_name": {"search": "search_server"},
            "max_steps": 7,
        }
        expected_env_args = {
            "question": "Test question",
            "mcp_servers": {"search_server": {"command": "search-command"}},
            "tool_name_to_server_name": {"search": "search_server"},
            "max_steps": 7,
        }

        MCPEnvironment.from_dict(env_args)

        assert env_args == expected_env_args

    @patch.object(MCPConnectionManager, "start", autospec=True)
    def test_hyphenated_tool_aliases_are_checked_for_duplicates(self, mock_start):
        """Test that hyphenated tool aliases participate in duplicate detection."""
        mock_start.side_effect = make_start_side_effect(
            {
                "command-a": {"search-tool": Mock(), "search_tool": Mock()},
                "command-b": {"search-tool": Mock(), "search_tool": Mock()},
            }
        )

        with pytest.raises(ValueError, match="Tool 'search-tool' is provided by multiple MCP servers"):
            MCPEnvironment(
                mcp_servers={
                    "server_a": {"command": "command-a"},
                    "server_b": {"command": "command-b"},
                }
            )
