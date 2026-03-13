from rllm.agents.agent import Action, BaseAgent, Episode, Step, Trajectory
from rllm.agents.math_agent import MathAgent

__all__ = ["BaseAgent", "Action", "Step", "Trajectory", "Episode", "MathAgent", "ToolAgent"]


def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        return None


# Define all agent imports (lazy, since they may have heavy/optional dependencies)
AGENT_IMPORTS = [
    ("rllm.agents.tool_agent", "ToolAgent"),
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
        if class_name not in __all__:
            __all__.append(class_name)
