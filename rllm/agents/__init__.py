from rllm.agents.agent import BaseAgent
from rllm.types import Action, Episode, Step, Trajectory

# Concrete agent implementations load on first access (see __getattr__) so optional /
# heavy dependencies are not pulled in by `import rllm.agents`.

__all__ = [
    "BaseAgent",
    "Action",
    "Step",
    "Trajectory",
    "Episode",
    "MathAgent",
    "ToolAgent",
    "MiniWobAgent",
    "FrozenLakeAgent",
    "SWEAgent",
    "CompetitionCodingAgent",
    "WebArenaAgent",
]

# name -> (module_path, attribute, optional)
# If optional is True, ImportError is turned into AttributeError (missing optional deps).
_LAZY_IMPORTS: dict[str, tuple[str, str, bool]] = {
    "MathAgent": ("rllm.agents.math_agent", "MathAgent", False),
    "ToolAgent": ("rllm.agents.tool_agent", "ToolAgent", True),
    "MiniWobAgent": ("rllm.agents.miniwob_agent", "MiniWobAgent", True),
    "FrozenLakeAgent": ("rllm.agents.frozenlake_agent", "FrozenLakeAgent", True),
    "SWEAgent": ("rllm.agents.swe_agent", "SWEAgent", True),
    "CompetitionCodingAgent": ("rllm.agents.code_agent", "CompetitionCodingAgent", True),
    "WebArenaAgent": ("rllm.agents.webarena_agent", "WebArenaAgent", True),
}


def __getattr__(name: str):
    try:
        module_path, attr_name, optional = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    try:
        module = __import__(module_path, fromlist=[attr_name])
        value = getattr(module, attr_name)
    except ImportError:
        if optional:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
        raise
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(__all__) | set(_LAZY_IMPORTS))
