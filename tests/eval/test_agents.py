"""Tests for built-in agent flows."""

from unittest.mock import MagicMock, patch

import pytest

from rllm.experimental.eval.types import AgentConfig, Task
from rllm.types import Episode


def _mock_openai_response(content: str, tool_calls=None):
    """Create a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_choice.message.tool_calls = tool_calls
    mock_response.choices = [mock_choice]
    return mock_response


def _mock_tool_call(call_id: str, name: str, arguments: str):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


@pytest.fixture
def base_config():
    return AgentConfig(
        base_url="http://localhost:8000/v1",
        model="test-model",
        session_uid="test-001",
    )


class TestReactAgentFlow:
    def test_returns_episode_with_spec(self, base_config):
        from rllm.experimental.agents.react_agent import react_agent

        task = Task(data={"question": "What is 2+2?", "ground_truth": "4"})
        task.spec = MagicMock()
        task.spec.instruction = "Solve the math problem."
        task.spec.render_input.return_value = "What is 2+2?"

        with patch("rllm.experimental.agents.react_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("The answer is \\boxed{4}")
            MockOpenAI.return_value = mock_client

            result = react_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert len(result.trajectories) == 1
        assert result.trajectories[0].steps[0].done is True
        assert result.artifacts["answer"] == "The answer is \\boxed{4}"

    def test_returns_episode_without_spec(self, base_config):
        from rllm.experimental.agents.react_agent import react_agent

        task = Task(data={"question": "What is 2+2?"})

        with patch("rllm.experimental.agents.react_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("4")
            MockOpenAI.return_value = mock_client

            result = react_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert result.artifacts["answer"] == "4"

    def test_llm_failure(self, base_config):
        from rllm.experimental.agents.react_agent import react_agent

        task = Task(data={"question": "What is 2+2?"})

        with patch("rllm.experimental.agents.react_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = Exception("Connection error")
            MockOpenAI.return_value = mock_client

            result = react_agent.run(task, base_config)

        assert isinstance(result, Episode)
        assert result.artifacts["answer"] == ""

    def test_no_reward_computed(self, base_config):
        """AgentFlow should NOT compute reward — that's the evaluator's job."""
        from rllm.experimental.agents.react_agent import react_agent

        task = Task(data={"question": "What is 2+2?", "ground_truth": "4"})

        with patch("rllm.experimental.agents.react_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("The answer is \\boxed{4}")
            MockOpenAI.return_value = mock_client

            result = react_agent.run(task, base_config)

        assert result.trajectories[0].reward is None

    def test_task_data_stored_in_episode(self, base_config):
        from rllm.experimental.agents.react_agent import react_agent

        data = {"question": "Hello?", "ground_truth": "Hi"}
        task = Task(data=data)

        with patch("rllm.experimental.agents.react_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("Hi")
            MockOpenAI.return_value = mock_client

            result = react_agent.run(task, base_config)

        assert result.task == data

    def test_multimodal_content_passed_through(self, base_config):
        """When spec.render_input returns a list (multimodal), it should be passed as user content."""
        from rllm.experimental.agents.react_agent import react_agent

        multimodal_content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "What is this?"},
        ]

        task = Task(data={"question": "What is this?", "images": ["img.png"]})
        task.spec = MagicMock()
        task.spec.instruction = "Describe the image."
        task.spec.render_input.return_value = multimodal_content

        with patch("rllm.experimental.agents.react_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("A cat")
            MockOpenAI.return_value = mock_client

            result = react_agent.run(task, base_config)

            call_args = mock_client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            assert messages[1]["content"] == multimodal_content

        assert result.artifacts["answer"] == "A cat"

    def test_multi_turn_tool_calling(self, base_config):
        """Agent loops when model returns tool calls, stops on plain response."""
        from rllm.experimental.agents.react_agent import ReactAgentFlow

        call_log = []

        def mock_lookup(city: str) -> str:
            call_log.append(city)
            return "72°F and sunny"

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
                "_execute": mock_lookup,
            }
        ]
        config = AgentConfig(
            base_url="http://localhost:8000/v1",
            model="test-model",
            session_uid="test-001",
            metadata={"tools": tools},
        )

        task = Task(data={"question": "What's the weather in Paris?"})

        # Turn 1: model calls the tool
        tc = _mock_tool_call("tc_1", "get_weather", '{"city": "Paris"}')
        resp_turn1 = _mock_openai_response("Let me look that up.", tool_calls=[tc])
        # Turn 2: model gives final answer
        resp_turn2 = _mock_openai_response("It's 72°F and sunny in Paris.")

        agent = ReactAgentFlow()

        with patch("rllm.experimental.agents.react_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [resp_turn1, resp_turn2]
            MockOpenAI.return_value = mock_client

            result = agent.run(task, config)

        assert call_log == ["Paris"]
        assert result.artifacts["answer"] == "It's 72°F and sunny in Paris."
        # Should have 2 steps: one intermediate (tool turn), one final
        assert len(result.trajectories[0].steps) == 2
        assert result.trajectories[0].steps[0].done is False
        assert result.trajectories[0].steps[1].done is True

    def test_multi_turn_max_turns_reached(self):
        """Agent stops after max_turns even if model keeps calling tools."""
        from rllm.experimental.agents.react_agent import ReactAgentFlow

        tools = [
            {
                "type": "function",
                "function": {"name": "search", "description": "Search", "parameters": {}},
                "_execute": lambda **kw: "result",
            }
        ]
        config = AgentConfig(
            base_url="http://localhost:8000/v1",
            model="test-model",
            session_uid="test-001",
            metadata={"tools": tools},
        )

        task = Task(data={"question": "Keep searching."})

        # Every response has a tool call — never gives a final answer
        def always_tool_call(*a, **kw):
            tc = _mock_tool_call("tc", "search", "{}")
            return _mock_openai_response("Searching...", tool_calls=[tc])

        agent = ReactAgentFlow(max_turns=3)

        with patch("rllm.experimental.agents.react_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = always_tool_call
            MockOpenAI.return_value = mock_client

            result = agent.run(task, config)

        # Should have exactly max_turns steps, last one marked done
        assert len(result.trajectories[0].steps) == 3
        assert result.trajectories[0].steps[-1].done is True

    def test_no_tools_single_turn(self, base_config):
        """Without tools, agent completes in exactly one LLM call."""
        from rllm.experimental.agents.react_agent import react_agent

        task = Task(data={"question": "What is 2+2?"})

        with patch("rllm.experimental.agents.react_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = _mock_openai_response("4")
            MockOpenAI.return_value = mock_client

            result = react_agent.run(task, base_config)
            assert mock_client.chat.completions.create.call_count == 1

        assert len(result.trajectories[0].steps) == 1
        assert result.trajectories[0].steps[0].done is True

    def test_tool_execution_error_handled(self, base_config):
        """Tool errors are caught and returned as error messages."""
        from rllm.experimental.agents.react_agent import ReactAgentFlow

        def bad_tool(**kw):
            raise ValueError("something broke")

        tools = [
            {
                "type": "function",
                "function": {"name": "broken", "description": "A broken tool", "parameters": {}},
                "_execute": bad_tool,
            }
        ]
        config = AgentConfig(
            base_url="http://localhost:8000/v1",
            model="test-model",
            session_uid="test-001",
            metadata={"tools": tools},
        )

        task = Task(data={"question": "Use the tool."})
        tc = _mock_tool_call("tc_1", "broken", "{}")
        resp1 = _mock_openai_response("Calling tool.", tool_calls=[tc])
        resp2 = _mock_openai_response("Tool failed, here's my answer.")

        agent = ReactAgentFlow()

        with patch("rllm.experimental.agents.react_agent.OpenAI") as MockOpenAI:
            mock_client = MagicMock()
            mock_client.chat.completions.create.side_effect = [resp1, resp2]
            MockOpenAI.return_value = mock_client

            result = agent.run(task, config)

            # Check the tool error was passed back to the model
            all_calls = mock_client.chat.completions.create.call_args_list
            second_call_messages = all_calls[1][1]["messages"]
            tool_msg = [m for m in second_call_messages if m.get("role") == "tool"]
            assert len(tool_msg) == 1
            assert "Error:" in tool_msg[0]["content"]

        assert result.artifacts["answer"] == "Tool failed, here's my answer."
