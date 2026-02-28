"""rLLM SDK integrations with third-party agent frameworks."""

try:
    from rllm.sdk.integrations.adk import RLLMTrajectoryPlugin
except Exception:
    RLLMTrajectoryPlugin = None  # type: ignore[assignment,misc]

try:
    from rllm.sdk.integrations.openai_agents import RLLMTrajectoryHooks
except Exception:
    RLLMTrajectoryHooks = None  # type: ignore[assignment,misc]

try:
    from rllm.sdk.integrations.strands import RLLMTrajectoryHookProvider
except Exception:
    RLLMTrajectoryHookProvider = None  # type: ignore[assignment,misc]

__all__ = [
    "RLLMTrajectoryPlugin",
    "RLLMTrajectoryHooks",
    "RLLMTrajectoryHookProvider",
]
