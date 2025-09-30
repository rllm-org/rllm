from rllm.agents.agent import Action, BaseAgent, Episode, Step, Trajectory
from rllm.agents.math_agent import MathAgent
from rllm.agents.tool_agent import ToolAgent

__all__ = ["BaseAgent", "Action", "Step", "Trajectory", "Episode", "MathAgent", "ToolAgent"]


def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        return None


# Define all agent imports
AGENT_IMPORTS = [
    ("rllm.agents.geo3k_agent", "Geo3kAgent"),
    ("rllm.agents.miniwob_agent", "MiniWobAgent"),
    ("rllm.agents.frozenlake_agent", "FrozenLakeAgent"),
    ("rllm.agents.swe_agent", "SWEAgent"),
    ("rllm.agents.code_agent", "CompetitionCodingAgent"),
    ("rllm.agents.webarena_agent", "WebArenaAgent"),
]

for module_path, class_name in AGENT_IMPORTS:
    imported_class = safe_import(module_path, class_name)
    if imported_class is not None:
        globals()[class_name] = imported_class
        __all__.append(class_name)
