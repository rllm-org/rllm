from rllm.agents.agent import Step, Trajectory
from rllm.agents.appworld_react_agents import AppWorldReactAgent


class TestAppWorldReactAgent:
    """AppWorldReactAgent test suite"""

    def test_init_default(self):
        """Test AppWorldReactAgent initialization with default parameters"""
        agent = AppWorldReactAgent()
        assert isinstance(agent._trajectory, Trajectory)
        assert agent.messages == []
        assert agent.current_observation is None
        assert agent.task_instruction is None
        assert agent.user_info is None
        assert agent.initialized is False

    def test_reset(self):
        """Test reset method"""
        agent = AppWorldReactAgent()

        # Add some states to reset
        agent.messages = [{"role": "user", "content": "test"}]
        agent._trajectory.steps.append(Step())
        agent.current_observation = "test observation"
        agent.task_instruction = "test task"
        agent.user_info = {"email": "test@example.com"}
        agent.initialized = True

        agent.reset()

        # Verify all states are reset
        assert isinstance(agent._trajectory, Trajectory)
        assert agent._trajectory.steps == []
        assert agent.messages == []
        assert agent.current_observation is None
        assert agent.task_instruction is None
        assert agent.user_info is None
        assert agent.initialized is False

    def test_properties(self):
        """Test key properties"""
        agent = AppWorldReactAgent()
        test_messages = [{"role": "user", "content": "test question"}, {"role": "assistant", "content": "test response"}]
        agent.messages = test_messages

        assert agent.chat_completions == test_messages
        assert agent.trajectory == agent._trajectory
        assert isinstance(agent.chat_completions, list)
        assert isinstance(agent.trajectory, Trajectory)

    def test_update_from_env_initial_task(self):
        """Test updating environment with initial task observation"""
        agent = AppWorldReactAgent()

        observation = {"instruction": "How many playlists do I have in Spotify?", "user_info": {"first_name": "Test", "last_name": "User", "email": "test@example.com", "phone_number": "+1234567890"}, "app_descriptions": "spotify: Music streaming app\nsupervisor: User management"}

        agent.update_from_env(observation, 0.0, False, {})

        # Verify agent is initialized
        assert agent.initialized is True
        assert agent.task_instruction == "How many playlists do I have in Spotify?"
        assert agent.user_info["email"] == "test@example.com"
        assert len(agent.messages) > 0

        # Verify system prompt contains task instruction
        full_prompt = "\n".join([msg["content"] for msg in agent.messages])
        assert "How many playlists do I have in Spotify?" in full_prompt
        assert "test@example.com" in full_prompt
        assert "Test User" in full_prompt
        assert "1234567890" in full_prompt

    def test_update_from_env_execution_result(self):
        """Test updating environment with code execution result"""
        agent = AppWorldReactAgent()

        # First initialize agent
        agent._initialize_from_task({"instruction": "Test task", "user_info": {"first_name": "Test", "last_name": "User", "email": "test@example.com", "phone_number": "+1234567890"}})

        initial_message_count = len(agent.messages)

        # Add a step to update its reward
        agent._trajectory.steps.append(Step())

        # Test successful execution result
        observation = {"success": True, "output": "['login', 'logout', 'show_playlist_library']", "stdout": ""}

        agent.update_from_env(observation, 0.5, False, {"step": 1})

        # Verify messages are correctly added
        assert len(agent.messages) == initial_message_count + 1
        assert agent.messages[-1]["role"] == "user"
        assert "Output:" in agent.messages[-1]["content"]
        assert "['login', 'logout', 'show_playlist_library']" in agent.messages[-1]["content"]

        # Verify last step is updated
        last_step = agent._trajectory.steps[-1]
        assert last_step.reward == 0.5
        assert last_step.done is False
        assert last_step.info["step"] == 1

    def test_update_from_env_error_result(self):
        """Test updating environment with error result"""
        agent = AppWorldReactAgent()

        # Initialize agent
        agent._initialize_from_task({"instruction": "Test task", "user_info": {"first_name": "Test", "last_name": "User", "email": "test@example.com", "phone_number": "+1234567890"}})

        initial_message_count = len(agent.messages)

        # Test error result
        observation = {"success": False, "error": "NameError: name 'undefined_var' is not defined", "stderr": "Traceback..."}

        agent.update_from_env(observation, 0.0, False, {})

        # Verify error message is correctly formatted
        assert len(agent.messages) == initial_message_count + 1
        assert agent.messages[-1]["role"] == "user"
        assert "Error:" in agent.messages[-1]["content"]
        assert "NameError" in agent.messages[-1]["content"]

    def test_update_from_model_basic(self):
        """Test basic update_from_model functionality"""
        agent = AppWorldReactAgent()

        # First provide task
        agent._initialize_from_task({"instruction": "Test task", "user_info": {"first_name": "Test", "last_name": "User", "email": "test@example.com", "phone_number": "+1234567890"}})
        agent.current_observation = "test observation"

        response = """I'll check the available APIs in Spotify.
Code:
```python
print(apis.api_docs.show_api_descriptions(app_name='spotify'))
```
"""
        action = agent.update_from_model(response)

        # Verify step is created
        assert len(agent._trajectory.steps) == 1
        current_step = agent._trajectory.steps[0]
        assert current_step.model_response == response
        assert "print(apis.api_docs.show_api_descriptions(app_name='spotify'))" in current_step.action

        # Verify messages are added
        assert agent.messages[-1]["role"] == "assistant"
        assert agent.messages[-1]["content"] == response

        # Verify return value
        assert "apis.api_docs.show_api_descriptions" in action.action

    def test_extract_code_from_response_markdown(self):
        """Test extracting code from markdown code block"""
        agent = AppWorldReactAgent()

        # Test code block with python marker
        response = """Let me call the API.
```python
result = apis.spotify.login(username='test@example.com', password='pass123')
print(result)
```
"""
        code = agent._extract_code_from_response(response)
        assert "apis.spotify.login" in code
        assert "print(result)" in code

        # Test code block without language marker
        response2 = """
```
apis.supervisor.complete_task(answer=5)
```
"""
        code2 = agent._extract_code_from_response(response2)
        assert "apis.supervisor.complete_task" in code2

    def test_extract_code_from_response_code_marker(self):
        """Test extracting code from Code: marker"""
        agent = AppWorldReactAgent()

        response = """I need to login first.
Code:
apis.spotify.login(username='test@example.com', password='password')
print('Logged in successfully')
"""
        code = agent._extract_code_from_response(response)
        assert "apis.spotify.login" in code
        assert "print('Logged in successfully')" in code

    def test_extract_code_from_response_whole_response(self):
        """Test extracting whole response as code (when no obvious marker)"""
        agent = AppWorldReactAgent()

        response = "apis.supervisor.complete_task()"
        code = agent._extract_code_from_response(response)
        assert code == "apis.supervisor.complete_task()"

    def test_multiple_code_blocks_extraction(self):
        """Test extracting multiple code blocks (should only extract first)"""
        agent = AppWorldReactAgent()

        response = """First, let me do this:
```python
print("First block")
```

And then this:
```python
print("Second block")
```
"""
        code = agent._extract_code_from_response(response)

        # Should only extract first code block
        assert 'print("First block")' in code
        assert 'print("Second block")' not in code

    def test_text_to_messages(self):
        """Test text to messages format"""
        agent = AppWorldReactAgent()

        text = """USER:
Hello, what's the task?

ASSISTANT:
I'll help you with that.
Code:
```python
print("test")
```

USER:
Output:
test
"""
        messages = agent.text_to_messages(text)

        # Verify message format
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert "Hello, what's the task?" in messages[0]["content"]
        assert messages[1]["role"] == "assistant"
        assert "I'll help you with that" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert "Output:" in messages[2]["content"]

    def test_basic_interaction_flow(self):
        """Test complete basic interaction flow"""
        agent = AppWorldReactAgent()

        # Step 1: Receive initial task
        task_observation = {"instruction": "How many playlists do I have in Spotify?", "user_info": {"first_name": "Test", "last_name": "User", "email": "test@example.com", "phone_number": "+1234567890"}, "app_descriptions": "spotify: Music streaming app"}
        agent.update_from_env(task_observation, 0.0, False, {})

        assert agent.initialized is True
        assert len(agent._trajectory.steps) == 0  # No steps before model response

        # Step 2: Model generates code
        response1 = """Let me check the available APIs.
Code:
```python
print(apis.api_docs.show_api_descriptions(app_name='spotify'))
```
"""
        action1 = agent.update_from_model(response1)

        assert len(agent._trajectory.steps) == 1
        assert "apis.api_docs.show_api_descriptions" in action1.action

        # Step 3: Environment returns execution result
        execution_result = {"success": True, "output": "['login', 'logout', 'show_playlist_library']", "stdout": ""}
        agent.update_from_env(execution_result, 0.0, False, {})
        assert len(agent._trajectory.steps) == 1

        # Should have updated trajectory
        assert len(agent.messages) >= 2  # At least assistant and user messages

        # Step 4: Model generates another action
        response2 = """Now I'll login to Spotify.
Code:
```python
result = apis.spotify.login(username='test@example.com', password='password')
print(result)
```
"""
        action2 = agent.update_from_model(response2)

        assert len(agent._trajectory.steps) == 2
        assert "apis.spotify.login" in action2.action

        # Step 5: Task completed
        completion_result = {"success": True, "output": "Task completed", "stdout": ""}
        agent.update_from_env(completion_result, 1.0, True, {"success": True})

        # Verify last step is marked as done
        last_step = agent._trajectory.steps[-1]
        assert last_step.reward == 1.0
        assert last_step.done is True
        assert last_step.info["success"] is True

        # Verify trajectory can be converted to dict
        trajectory_dict = agent.trajectory.to_dict()
        assert isinstance(trajectory_dict, dict)
        assert "steps" in trajectory_dict
        assert isinstance(trajectory_dict["steps"], list)
        assert len(trajectory_dict["steps"]) == 2

    def test_message_history_accumulation(self):
        """Test message history accumulation"""
        agent = AppWorldReactAgent()

        # Initialize
        agent._initialize_from_task({"instruction": "Test task", "user_info": {"first_name": "Test", "last_name": "User", "email": "test@example.com", "phone_number": "+1234567890"}})
        initial_message_count = len(agent.messages)

        # Multiple rounds of conversation
        for i in range(3):
            agent.current_observation = f"observation {i}"
            agent.update_from_model(f"```python\nprint('{i}')\n```")
            agent.update_from_env({"success": True, "output": f"Result {i}"}, 0.0, False, {})

        # Verify message history growth
        # Each round: 1 assistant + 1 user = 2 messages per iteration
        expected_count = initial_message_count + (3 * 2)
        assert len(agent.messages) == expected_count

        # Verify message role alternation (ignore initial system message)
        roles = [msg["role"] for msg in agent.messages[initial_message_count:]]
        expected_roles = ["assistant", "user"] * 3
        assert roles == expected_roles
