import json
from unittest.mock import Mock, patch

import pytest

from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import RewardFunction, RewardOutput
from rllm.tools.tool_base import Tool, ToolOutput


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(self, name="mock_tool"):
        self.name = name
        self.description = "A mock tool for testing"
        self.parameters = {"type": "object", "properties": {"query": {"type": "string", "description": "Test query"}}, "required": ["query"]}

    @property
    def json(self):
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def call(self, **kwargs):
        return ToolOutput(name=self.name, output=f"Mock result for {kwargs}")

    def forward(self, **kwargs):
        return self.call(**kwargs)


class MockRewardFunction(RewardFunction):
    """Mock reward function for testing."""

    def __call__(self, task_info, action, **kwargs):
        # Simple mock reward based on action content
        reward = 1.0 if "correct" in str(action).lower() else 0.0
        metadata = {"evaluated": True, "action": action}
        return RewardOutput(reward=reward, metadata=metadata)


class TestToolEnvironment:
    """Test suite for ToolEnvironment class."""

    def test_init_default(self):
        """Test ToolEnvironment initialization with default parameters."""
        env = ToolEnvironment()
        assert env.step_count == 0
        assert env.max_steps == 10
        assert env.tools is not None
        assert env.task is None
        assert env.reward_fn is not None  # Should use zero_reward

    def test_init_with_tools_list(self):
        """Test ToolEnvironment initialization with tools list."""
        with patch("rllm.environments.tools.tool_env.MultiTool") as mock_multi_tool:
            mock_instance = Mock()
            mock_multi_tool.return_value = mock_instance

            env = ToolEnvironment(tools=["calculator", "search"])

            mock_multi_tool.assert_called_once_with(tools=["calculator", "search"])
            assert env.tools == mock_instance

    def test_init_with_tool_map(self):
        """Test ToolEnvironment initialization with tool_map."""
        tool_map = {"mock_tool": MockTool}

        with patch("rllm.environments.tools.tool_env.MultiTool") as mock_multi_tool:
            mock_instance = Mock()
            mock_multi_tool.return_value = mock_instance

            env = ToolEnvironment(tool_map=tool_map)

            mock_multi_tool.assert_called_once_with(tool_map=tool_map)
            assert env.tools == mock_instance

    def test_init_with_both_tools_and_tool_map_raises_error(self):
        """Test that providing both tools and tool_map raises ValueError."""
        with pytest.raises(ValueError, match="Cannot specify both 'tools' and 'tool_map' parameters"):
            ToolEnvironment(tools=["calculator"], tool_map={"mock": MockTool})

    def test_init_with_custom_parameters(self):
        """Test ToolEnvironment initialization with custom parameters."""
        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        max_steps = 5

        env = ToolEnvironment(task=task, reward_fn=reward_fn, max_steps=max_steps)

        assert env.task == task
        assert env.reward_fn == reward_fn
        assert env.max_steps == max_steps

    def test_init_no_reward_function_warning(self):
        """Test that warning is issued when no reward function is provided."""
        with pytest.warns(UserWarning, match="No reward function specified"):
            env = ToolEnvironment(reward_fn=None)
            assert env.reward_fn is not None  # Should use zero_reward

    def test_reset(self):
        """Test the reset method."""
        task = {"question": "Test question"}
        env = ToolEnvironment(task=task, max_steps=5)

        # Set some non-initial state
        env.step_count = 3

        obs, info = env.reset()

        assert env.step_count == 0
        assert obs == task
        assert isinstance(info, dict)

    def test_step_with_string_action(self):
        """Test stepping with string action (should terminate)."""
        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = ToolEnvironment(task=task, reward_fn=reward_fn)
        env.reset()

        action = "This is my final answer"
        obs, reward, done, info = env.step(action)

        assert obs == {}
        assert reward == 0.0  # MockRewardFunction returns 0 for non-"correct" answers
        assert done is True
        assert info["response"] == action
        assert "metadata" in info

    def test_step_with_string_action_correct(self):
        """Test stepping with string action containing 'correct'."""
        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = ToolEnvironment(task=task, reward_fn=reward_fn)
        env.reset()

        action = "The correct answer is 42"
        obs, reward, done, info = env.step(action)

        assert reward == 1.0  # MockRewardFunction returns 1 for "correct" answers
        assert done is True

    def test_step_with_finish_tool_call(self):
        """Test stepping with finish tool call."""
        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = ToolEnvironment(task=task, reward_fn=reward_fn)
        env.reset()

        action = [{"id": "call_1", "function": {"name": "finish", "arguments": {"response": "Final answer"}}}]

        obs, reward, done, info = env.step(action)

        assert obs == {}
        assert done is True
        assert info["response"] == action

    def test_step_with_regular_tool_calls(self):
        """Test stepping with regular tool calls."""
        task = {"question": "Test question"}
        env = ToolEnvironment(task=task)
        env.reset()

        # Mock the tool execution
        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {"call_1": "Tool output"}

            action = [{"id": "call_1", "function": {"name": "search", "arguments": {"query": "test"}}}]

            obs, reward, done, info = env.step(action)

            assert obs == {"tool_outputs": {"call_1": "Tool output"}}
            assert reward == 0
            assert done is False
            assert info["response"] == action
            mock_execute.assert_called_once_with(action)

    def test_step_max_steps_termination(self):
        """Test that environment terminates when max_steps is reached."""
        env = ToolEnvironment(max_steps=2)
        env.reset()

        # Take steps until max_steps
        for i in range(2):
            with patch.object(env, "_execute_tool_calls") as mock_execute:
                mock_execute.return_value = {"call_1": "Tool output"}

                action = [{"id": "call_1", "function": {"name": "search", "arguments": {"query": "test"}}}]

                obs, reward, done, info = env.step(action)

                if i == 1:  # Last step - should be done due to max_steps
                    assert done is True
                else:
                    assert done is False

    def test_step_with_dict_action(self):
        """Test stepping with dict action (converted to list)."""
        env = ToolEnvironment()
        env.reset()

        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {"call_1": "Tool output"}

            action = {"id": "call_1", "function": {"name": "search", "arguments": {"query": "test"}}}

            obs, reward, done, info = env.step(action)

            # Should convert dict to list
            mock_execute.assert_called_once_with([action])

    def test_execute_tool_calls(self):
        """Test _execute_tool_calls method."""
        tool_map = {"mock_tool": MockTool}
        env = ToolEnvironment(tool_map=tool_map)
        env.reset()

        # Mock the MultiTool forward method
        mock_tool_output = ToolOutput(name="mock_tool", output="Mock result")
        env.tools.forward = Mock(return_value=mock_tool_output)

        tool_calls = [{"id": "call_1", "function": {"name": "mock_tool", "arguments": json.dumps({"query": "test"})}}]

        result = env._execute_tool_calls(tool_calls)

        assert "call_1" in result
        assert "Mock result" in result["call_1"]

    def test_execute_tool_calls_multiple_tools(self):
        """Test _execute_tool_calls with multiple tool calls."""
        tool_map = {"mock_tool": MockTool}
        env = ToolEnvironment(tool_map=tool_map)
        env.reset()

        # Mock the MultiTool forward method
        mock_tool_output = ToolOutput(name="mock_tool", output="Mock result")
        env.tools.forward = Mock(return_value=mock_tool_output)

        tool_calls = [
            {"id": "call_1", "function": {"name": "mock_tool", "arguments": json.dumps({"query": "test1"})}},
            {"id": "call_2", "function": {"name": "mock_tool", "arguments": json.dumps({"query": "test2"})}},
        ]

        result = env._execute_tool_calls(tool_calls)

        assert len(result) == 2
        assert "call_1" in result
        assert "call_2" in result

    def test_execute_tool_calls_threading(self):
        """Test that _execute_tool_calls uses threading correctly."""
        tool_map = {"mock_tool": MockTool}
        env = ToolEnvironment(tool_map=tool_map)
        env.reset()

        # Mock the MultiTool to track calls
        call_count = 0

        def mock_forward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return ToolOutput(name="mock_tool", output=f"Result {call_count}")

        env.tools.forward = mock_forward

        tool_calls = [
            {"id": "call_1", "function": {"name": "mock_tool", "arguments": json.dumps({"query": "test1"})}},
            {"id": "call_2", "function": {"name": "mock_tool", "arguments": json.dumps({"query": "test2"})}},
        ]

        result = env._execute_tool_calls(tool_calls)

        # Both tools should have been called
        assert call_count == 2
        assert len(result) == 2

    def test_idx_property(self):
        """Test the idx property from BaseEnv."""
        env = ToolEnvironment()

        # Initially should be None
        assert env.idx is None

        # Should be able to set and get
        env.idx = 5
        assert env.idx == 5

    def test_is_multithread_safe(self):
        """Test the is_multithread_safe static method."""
        assert ToolEnvironment.is_multithread_safe() is True

    def test_from_dict(self):
        """Test creating environment from dictionary."""
        env_args = {"question": "Test question", "tools": ["calculator"], "max_steps": 15, "reward_fn": MockRewardFunction()}

        with patch("rllm.environments.tools.tool_env.MultiTool"):
            env = ToolEnvironment.from_dict(env_args)

            assert isinstance(env, ToolEnvironment)
            assert env.max_steps == 15
            assert env.task == {"question": "Test question"}

    def test_from_dict_with_tool_map(self):
        """Test creating environment from dictionary with tool_map."""
        tool_map = {"mock_tool": MockTool}
        env_args = {"question": "Test question", "tool_map": tool_map, "max_steps": 20}

        with patch("rllm.environments.tools.tool_env.MultiTool"):
            env = ToolEnvironment.from_dict(env_args)

            assert isinstance(env, ToolEnvironment)
            assert env.max_steps == 20

    def test_close_method(self):
        """Test the close method (inherited from BaseEnv)."""
        env = ToolEnvironment()
        # Should not raise any errors
        env.close()

    def test_full_interaction_flow(self):
        """Test a complete interaction flow."""
        task = {"question": "What is 2 + 2?"}
        reward_fn = MockRewardFunction()
        env = ToolEnvironment(task=task, reward_fn=reward_fn, max_steps=3)

        # Reset environment
        obs, info = env.reset()
        assert obs == task
        assert env.step_count == 0

        # Step 1: Use a tool
        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {"call_1": "Calculator result: 4"}

            action1 = [{"id": "call_1", "function": {"name": "calculator", "arguments": json.dumps({"expression": "2 + 2"})}}]

            obs1, reward1, done1, info1 = env.step(action1)

            assert obs1 == {"tool_outputs": {"call_1": "Calculator result: 4"}}
            assert reward1 == 0
            assert done1 is False
            assert env.step_count == 1

        # Step 2: Finish with answer
        action2 = "The correct answer is 4"
        obs2, reward2, done2, info2 = env.step(action2)

        assert obs2 == {}
        assert reward2 == 1.0  # MockRewardFunction gives 1.0 for "correct"
        assert done2 is True
        assert info2["response"] == action2

    def test_step_with_finish_tool_call_with_arguments(self):
        """Test stepping with finish tool call containing arguments."""
        task = {"question": "Test question"}
        reward_fn = MockRewardFunction()
        env = ToolEnvironment(task=task, reward_fn=reward_fn)
        env.reset()

        action = [{"id": "call_1", "function": {"name": "finish", "arguments": {"response": "The correct final answer"}}}]

        obs, reward, done, info = env.step(action)

        assert reward == 1.0  # Should extract response from arguments
        assert done is True

    def test_step_with_malformed_tool_call(self):
        """Test stepping with malformed tool call structure."""
        env = ToolEnvironment()
        env.reset()

        with patch.object(env, "_execute_tool_calls") as mock_execute:
            mock_execute.return_value = {"call_1": "Tool output"}

            # Malformed action (missing required fields)
            action = [{"id": "call_1"}]

            obs, reward, done, info = env.step(action)

            # Should still process the action
            assert obs == {"tool_outputs": {"call_1": "Tool output"}}
            assert done is False

    def test_concurrent_tool_execution(self):
        """Test that multiple tools can be executed concurrently."""
        import time

        class SlowMockTool(MockTool):
            def call(self, **kwargs):
                time.sleep(0.1)  # Simulate slow operation
                return ToolOutput(name=self.name, output=f"Slow result for {kwargs}")

        tool_map = {"slow_tool": SlowMockTool}
        env = ToolEnvironment(tool_map=tool_map)
        env.reset()

        # Mock the MultiTool forward method
        def slow_forward(*args, **kwargs):
            time.sleep(0.1)
            return ToolOutput(name="slow_tool", output="Slow result")

        env.tools.forward = slow_forward

        tool_calls = [
            {"id": "call_1", "function": {"name": "slow_tool", "arguments": json.dumps({"query": "test1"})}},
            {"id": "call_2", "function": {"name": "slow_tool", "arguments": json.dumps({"query": "test2"})}},
        ]

        start_time = time.time()
        result = env._execute_tool_calls(tool_calls)
        end_time = time.time()

        # Should take roughly 0.1 seconds (concurrent) rather than 0.2 (sequential)
        assert end_time - start_time < 0.15  # Some tolerance for threading overhead
        assert len(result) == 2

    def test_edge_cases(self):
        """Test various edge cases."""
        env = ToolEnvironment()
        env.reset()

        # Empty action list
        obs, reward, done, info = env.step([])
        assert obs == {"tool_outputs": {}}
        assert done is False

        # None action (should be handled gracefully)
        obs, reward, done, info = env.step(None)
        assert obs == {"tool_outputs": {}}
        assert done is False

    def test_reward_function_integration(self):
        """Test integration with different reward functions."""

        class CustomReward(RewardFunction):
            def __call__(self, task_info, action, **kwargs):
                score = len(str(action)) / 10.0  # Simple length-based reward
                return RewardOutput(reward=score, metadata={"length": len(str(action))})

        task = {"question": "Test"}
        reward_fn = CustomReward()
        env = ToolEnvironment(task=task, reward_fn=reward_fn)
        env.reset()

        # Test with different length responses
        short_action = "Short"
        long_action = "This is a much longer response that should get a higher reward"

        _, reward1, _, info1 = env.step(short_action)
        env.reset()
        _, reward2, _, info2 = env.step(long_action)

        assert reward2 > reward1  # Longer response should get higher reward
        assert info1["metadata"]["length"] < info2["metadata"]["length"]
