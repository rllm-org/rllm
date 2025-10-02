from typing import Any

from rllm.agents.agent import Action, BaseAgent, Trajectory


class AppWorldReactAgent(BaseAgent):
    """
    React agent adapted for AppWorld integration with rLLM.

    This agent implements the ReAct (Reasoning, Action) pattern specifically designed
    for AppWorld's multi-app environment with API interactions.

    Interaction process:
    1. Agent receives observation from the environment (task instruction or previous execution result)
    2. Agent formats the observation into messages (for LLM inference)
    3. LLM generates Python code (including thought process and actual code)
    4. Agent parses the response, extracts Python code
    5. Environment executes the code and returns the result
    """

    SYSTEM_PROMPT: str = """You are an intelligent AI assistant capable of interacting with multiple applications through their APIs to complete tasks.

You have access to various apps (like spotify, gmail, calendar, etc.) and can call their APIs using Python code. You can also get information about available APIs using the api_docs APIs.

Key Guidelines:
1. Always look at API specifications using apis.api_docs.show_api_doc() before calling any API
2. Write small chunks of code, only one main action per step
3. Use variables to store and reuse information across steps
4. Loop through pages when APIs return paginated results
5. Complete the task by calling apis.supervisor.complete_task() with the answer if needed

Available Helper APIs:
- apis.api_docs.show_app_descriptions() - Get list of available apps
- apis.api_docs.show_api_descriptions(app_name='app') - Get APIs for a specific app
- apis.api_docs.show_api_doc(app_name='app', api_name='api') - Get detailed API specification

Response Format:
You should respond with your thought process followed by the Python code to execute.
For example:

Thought: I need to first check what APIs are available for spotify.

Code:
```python
apis.api_docs.show_api_descriptions(app_name='spotify')
```

Remember to always check API documentation before making calls."""

    def __init__(self):
        """Initialize the AppWorld ReAct Agent."""
        self._trajectory = Trajectory()
        self.messages: list[dict[str, Any]] = []
        self.current_observation = None
        self.reset()

    def reset(self):
        """Reset the agent's state for a new task."""
        self._trajectory = Trajectory()
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        self.current_observation = None

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Update the agent's state based on environment feedback.

        Args:
            observation: Environment observation (task instruction or code execution result)
            reward: Reward value
            done: Whether the task is completed
            info: Additional information
        """
        # Update current observation
        self.current_observation = observation

        # Build user message based on observation type
        if isinstance(observation, dict):
            if "instruction" in observation:
                # Initial task instruction
                user_message = self._format_initial_instruction(observation)
            else:
                # Code execution result
                user_message = self._format_execution_result(observation)
        elif isinstance(observation, str):
            user_message = observation
        else:
            user_message = str(observation)

        # Add to message history
        self.messages.append({"role": "user", "content": user_message})

        # Update the reward and done of the last step in the trajectory
        if self._trajectory.steps:
            last_step = self._trajectory.steps[-1]
            last_step.reward = reward
            last_step.done = done
            last_step.info.update(info)

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Update the agent's state based on the model's response.

        Args:
            response: model's response (including thought and code)

        Returns:
            Action: Action (string) containing the Python code to execute
        """
        pass

    def _format_initial_instruction(self, observation: dict) -> str:
        """Format initial task instruction as user message."""
        parts = [f"Task: {observation.get('instruction', 'No instruction provided')}"]

        if "available_apps" in observation:
            apps = ", ".join(observation["available_apps"])
            parts.append(f"\nAvailable Apps: {apps}")

        if "helper_apis" in observation:
            parts.append("\nHelper APIs:")
            for name, usage in observation["helper_apis"].items():
                parts.append(f"- {name}: {usage}")

        return "\n".join(parts)

    def _format_execution_result(self, observation: dict) -> str:
        """Format code execution result as user message."""
        if not observation.get("success", True):
            return f"Error: {observation.get('error', 'Unknown error')}\n{observation.get('stderr', '')}"

        parts = []
        if observation.get("output"):
            parts.append(f"Output: {observation['output']}")
        if observation.get("stdout"):
            parts.append(f"Stdout: {observation['stdout']}")
        if observation.get("stderr"):
            parts.append(f"Stderr: {observation['stderr']}")

        return "\n".join(parts) if parts else "Code executed successfully (no output)"

    def _extract_code_from_response(self, response: str) -> str:
        """
        Extract Python code from the model's response.
        """
        # TODO: Implement this method
        pass

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Returns the history of messages for chat completion."""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """Returns the trajectory object."""
        return self._trajectory
