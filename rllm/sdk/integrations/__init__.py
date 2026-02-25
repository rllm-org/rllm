"""rLLM SDK integrations with third-party agent frameworks."""

try:
    from rllm.sdk.integrations.adk import RLLMTrajectoryPlugin
except ImportError:
    RLLMTrajectoryPlugin = None  # type: ignore[assignment,misc]

__all__ = [
    "RLLMTrajectoryPlugin",
]
