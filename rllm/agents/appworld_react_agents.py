from typing import Any

from rllm.agents.agent import BaseAgent
from rllm.environments.appworld.appworld_env import AppWorldEnv


class AppWorldReactAgent(BaseAgent):
    """
    React agent adapted for AppWorld integration with rLLM.

    This agent implements the ReAct (Reasoning, Action) pattern specifically designed for AppWorld's multi-app environment with API interations.
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

Remember to always check API documentation before making calls."""

    def __init__(self, appworld_env: AppWorldEnv):
        self.appworld_env = appworld_env

    def reset(self):
        pass

    def step(self, action: str):
        pass

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        pass

    def update_from_model(self, response: str, **kwargs):
        pass
