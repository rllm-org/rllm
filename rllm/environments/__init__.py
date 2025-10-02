from rllm.environments.base.base_env import BaseEnv
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.environments.tools.tool_env import ToolEnvironment

__all__ = ["BaseEnv", "SingleTurnEnvironment", "ToolEnvironment"]


def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        return None


ENVIRONMENT_IMPORTS = [
    ("rllm.environments.browsergym.browsergym", "BrowserGymEnv"),
    ("rllm.environments.frozenlake.frozenlake", "FrozenLakeEnv"),
    # ("rllm.environments.swe.swe", "SWEEnv"),
    ("rllm.environments.code.competition_coding", "CompetitionCodingEnv"),
    ("rllm.environments.appworld.appworld_env", "AppWorldEnv"),
]

for module_path, class_name in ENVIRONMENT_IMPORTS:
    imported_class = safe_import(module_path, class_name)
    if imported_class is not None:
        globals()[class_name] = imported_class
        __all__.append(class_name)
