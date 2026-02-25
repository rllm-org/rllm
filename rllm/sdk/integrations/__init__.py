"""rLLM SDK integrations with third-party agent frameworks."""

try:
    from rllm.sdk.integrations.adk import RLLMTrajectoryPlugin
except ImportError:
    RLLMTrajectoryPlugin = None  # type: ignore[assignment,misc]

try:
    from rllm.sdk.integrations.openai_agents import RLLMTrajectoryHooks
except ImportError:
    RLLMTrajectoryHooks = None  # type: ignore[assignment,misc]

__all__ = [
    "RLLMTrajectoryPlugin",
    "RLLMTrajectoryHooks",
]
