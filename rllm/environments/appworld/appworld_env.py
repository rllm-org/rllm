from typing import Any

from rllm.environments.base.base_env import BaseEnv


class AppWorldEnv(BaseEnv):
    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def step(self, action: str):
        pass

    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        pass

    @staticmethod
    def from_dict(env_args: dict) -> "AppWorldEnv":
        return AppWorldEnv()
