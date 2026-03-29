"""Remote agent runtime support for “agent-in-sandbox" runtimes"""

from rllm.experimental.engine.remote_runtime.protocol import (
    RemoteAgentRuntime,
    RemoteRuntimeConfig,
    RemoteTaskResult,
    TaskSubmission,
)


def create_remote_runtime(
    config: RemoteRuntimeConfig,
    exp_id: str = "",
    model_id: str = "",
) -> RemoteAgentRuntime:
    """Factory: create a RemoteAgentRuntime from config.

    Args:
        config: RemoteRuntimeConfig with backend type and backend-specific config.
        exp_id: Experiment ID (typically from trainer.experiment_name).
        model_id: Model name (typically from config.model.name).

    Returns:
        A RemoteAgentRuntime instance (not yet initialized — call initialize()).
    """
    if config.backend == "agentcore":
        from rllm.experimental.engine.remote_runtime.agentcore_runtime import (
            AgentCoreRuntime,
        )

        return AgentCoreRuntime(config, exp_id=exp_id, model_id=model_id)

    raise ValueError(f"Unknown remote runtime backend: {config.backend!r}")


__all__ = [
    "RemoteAgentRuntime",
    "RemoteRuntimeConfig",
    "RemoteTaskResult",
    "TaskSubmission",
    "create_remote_runtime",
]
